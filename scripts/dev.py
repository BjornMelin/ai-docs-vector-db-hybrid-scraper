#!/usr/bin/env python3
"""Unified developer CLI for local workflows."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import error as url_error, request as url_request


if TYPE_CHECKING:
    from collections.abc import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = PROJECT_ROOT / "docs"


def _normalize_command(command: Sequence[str | os.PathLike[str]]) -> list[str]:
    """Convert supported command tokens into plain strings."""
    normalized: list[str] = []
    for token in command:
        token_str = os.fspath(token)
        if "\x00" in token_str:
            msg = "Command tokens must not contain NUL characters."
            raise ValueError(msg)
        normalized.append(token_str)
    return normalized


@dataclass(frozen=True)
class PytestProfile:
    """Configuration describing how to invoke pytest for a profile."""

    args: tuple[str, ...]
    uses_workers: bool = True
    forces_coverage: bool = False


PYTEST_PROFILES: dict[str, PytestProfile] = {
    "quick": PytestProfile(("tests/unit", "-m", "not slow", "-q")),
    "unit": PytestProfile(("tests/unit", "-m", "unit or fast")),
    "full": PytestProfile(("tests",)),
    "ci": PytestProfile(
        ("tests", "-m", "not local_only", "--maxfail=3"), forces_coverage=True
    ),
}


def run_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command and stream its output."""
    normalized = _normalize_command(command)
    print(f"$ {shlex.join(normalized)}")
    result = subprocess.run(normalized, cwd=cwd, env=env, check=False, shell=False)  # noqa: S603
    if result.returncode != 0:
        print(f"Command exited with status {result.returncode}", file=sys.stderr)
    return result.returncode


@lru_cache(maxsize=1)
def _ensure_uv_available() -> bool:
    """Validate that the uv CLI is available before running uv-backed commands."""
    if shutil.which("uv"):
        return True

    message = (
        "The 'uv' command is required for this workflow but was not found on your "
        "PATH.\nInstall uv from https://github.com/astral-sh/uv or adjust your "
        "PATH before rerunning."
    )
    print(message, file=sys.stderr)
    return False


def _auto_worker_count(explicit: int | None) -> str:
    """Return an appropriate worker count for pytest-xdist."""
    if explicit and explicit > 0:
        return str(explicit)
    cpu_count = os.cpu_count() or 1
    return str(max(1, cpu_count - 1))


def _coverage_arguments(enable: bool) -> list[str]:
    """Return coverage arguments when requested."""
    if not enable:
        return []
    return [
        "--cov=src",
        "--cov-report=term-missing:skip-covered",
        "--cov-report=xml",
        "--cov-fail-under=80",
    ]


def _build_pytest_command(
    profile: str,
    *,
    workers: str,
    coverage: bool,
    verbose: bool,
    extra: Sequence[str],
) -> list[str]:
    """Construct the pytest command for the selected profile."""
    try:
        profile_cfg = PYTEST_PROFILES[profile]
    except KeyError as exc:  # pragma: no cover - safeguarded by argparse choices
        message = f"Unknown pytest profile: {profile}"
        raise ValueError(message) from exc

    command: list[str] = ["uv", "run", "pytest", *profile_cfg.args]

    if "--benchmark-disable" not in command:
        command.append("--benchmark-disable")

    if profile_cfg.uses_workers:
        command.extend(["-n", workers, "--dist", "worksteal"])

    effective_coverage = coverage or profile_cfg.forces_coverage
    command.extend(_coverage_arguments(effective_coverage))

    has_quiet_flag = any(arg in {"-q", "--quiet"} for arg in profile_cfg.args)
    if verbose and not has_quiet_flag:
        command.append("-vv")
    elif not verbose and not has_quiet_flag and profile not in {"performance", "ci"}:
        command.append("-v")

    command.extend(extra)
    return command


def cmd_test(args: argparse.Namespace) -> int:
    """Execute one of the supported pytest profiles."""
    if not _ensure_uv_available():
        return 1

    workers = _auto_worker_count(args.workers)
    coverage_enabled = args.coverage or args.profile == "ci"
    extra_args = list(args.extra) if args.extra else []
    command = _build_pytest_command(
        args.profile,
        workers=workers,
        coverage=coverage_enabled,
        verbose=args.verbose,
        extra=extra_args,
    )
    coverage_state = "on" if coverage_enabled else "off"
    print(
        f"Running tests with profile='{args.profile}', "
        f"workers={workers}, coverage={coverage_state}"
    )
    return run_command(command)


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the RAG golden evaluation harness."""
    if not _ensure_uv_available():
        return 1

    command = [
        "uv",
        "run",
        "python",
        "scripts/eval/rag_golden_eval.py",
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.output:
        command.extend(["--output", args.output])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.namespace:
        command.extend(["--namespace", args.namespace])
    if args.ragas_llm_model:
        command.extend(["--ragas-llm-model", args.ragas_llm_model])
    if args.ragas_embedding_model:
        command.extend(["--ragas-embedding-model", args.ragas_embedding_model])
    if args.max_semantic_samples is not None:
        command.extend(["--max-semantic-samples", str(args.max_semantic_samples)])
    if args.metrics_allowlist:
        command.append("--metrics-allowlist")
        command.extend(args.metrics_allowlist)
    return run_command(command)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run pytest-powered benchmark suites."""
    if not _ensure_uv_available():
        return 1

    suite_arguments: dict[str, list[str]] = {
        "performance": ["tests/performance"],
        "integration": ["tests/integration"],
        "all": ["tests"],
    }

    try:
        base_args = suite_arguments[args.suite]
    except KeyError as exc:  # pragma: no cover - safeguarded by argparse choices
        message = f"Unknown benchmark suite: {args.suite}"
        raise ValueError(message) from exc

    command: list[str] = ["uv", "run", "pytest", *base_args, "--benchmark-only"]

    if args.workers and args.workers > 0:
        command.extend(["-n", str(args.workers), "--dist", "worksteal"])

    if args.verbose:
        command.append("-vv")
    else:
        command.append("-v")

    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_path = Path(args.output_dir) / f"{args.suite}_benchmark.json"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--benchmark-json", str(output_path)])

    if args.compare_baseline:
        command.extend(["--benchmark-compare", args.baseline])

    return run_command(command)


def _import_check(module: str) -> bool:
    """Safely test whether a module can be imported."""
    try:
        __import__(module)
    except ModuleNotFoundError:
        return False
    return True


def _is_external_link(link: str) -> bool:
    """Return ``True`` when the link points outside of the repository."""
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", link)) or link.startswith("#")


def _resolve_link_target(source: Path, link: str) -> Path | None:
    """Resolve a documentation link to an absolute path."""
    parts = link.split("#", 1)
    if (clean_link := parts[0]) == "":
        return source

    if clean_link.startswith(("./", "../")):
        return (source.parent / clean_link).resolve()

    clean_link = clean_link.removeprefix("/")
    return (DOCS_ROOT / clean_link).resolve()


def _validate_docs_links() -> list[tuple[Path, int, str]]:
    """Validate documentation links and report any missing targets."""
    issues: list[tuple[Path, int, str]] = []
    if not DOCS_ROOT.exists():
        return issues

    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    markdown_files = DOCS_ROOT.rglob("*.md")

    for file_path in markdown_files:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(lines, start=1):
            for match in pattern.finditer(line):
                link_target = match.group(2).strip()
                if _is_external_link(link_target):
                    continue
                target_path = _resolve_link_target(file_path, link_target)
                if target_path is None or not target_path.exists():
                    issues.append(
                        (file_path.relative_to(DOCS_ROOT), line_number, link_target)
                    )
    return issues


def _check_service_health(url: str) -> bool:
    """Return ``True`` when the given HTTP endpoint responds successfully."""
    try:
        response = url_request.urlopen(url, timeout=5)  # noqa: S310
    except (url_error.URLError, ValueError):
        return False

    with response:
        return 200 <= response.status < 300


def cmd_validate(args: argparse.Namespace) -> int:  # pylint: disable=too-many-branches
    """Validate configuration, dependencies, and optionally documentation."""
    errors: list[str] = []
    warnings: list[str] = []

    env_files = [".env", ".env.local", ".env.example"]
    if any((PROJECT_ROOT / env).exists() for env in env_files):
        print("✅ Environment templates detected")
    else:
        warnings.append("No environment file found (.env, .env.local, .env.example)")

    required_paths = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "tests",
    ]
    for path in required_paths:
        if path.exists():
            continue
        errors.append(
            f"Missing required file or directory: {path.relative_to(PROJECT_ROOT)}"
        )

    recommended_paths = [
        PROJECT_ROOT / "src/api/main.py",
        PROJECT_ROOT / "src/cli/unified.py",
    ]
    for path in recommended_paths:
        if path.exists():
            continue
        warnings.append(
            "Recommended file missing (optional surface): "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

    for module in ("fastapi", "qdrant_client", "pytest"):
        if _import_check(module):
            print(f"✅ Dependency available: {module}")
        else:
            warnings.append(f"Optional dependency missing: {module}")

    if args.check_services:
        if _check_service_health("http://localhost:6333/health"):
            print("✅ Qdrant service reachable at http://localhost:6333")
        else:
            warnings.append("Qdrant service is not reachable on http://localhost:6333")

    if args.check_docs:
        if issues := _validate_docs_links():
            errors.extend(
                [f"Broken link {link} in {file}:{line}" for file, line, link in issues]
            )
        else:
            print("✅ Documentation links validated")

    if errors:
        print("\n❌ Validation errors detected:")
        for message in errors:
            print(f"  • {message}")

    if warnings:
        print("\n⚠️  Validation warnings:")
        for message in warnings:
            print(f"  • {message}")

    if errors or (args.strict and warnings):
        return 1
    return 0


def _compose_base_command() -> list[str]:
    """Determine the docker compose executable to use."""
    if docker := shutil.which("docker"):
        compose_probe = subprocess.run(  # noqa: S603
            _normalize_command([docker, "compose", "version"]),
            check=False,
            capture_output=True,
        )
        if compose_probe.returncode == 0:
            return [docker, "compose"]
    if compose := shutil.which("docker-compose"):
        return [compose]
    message = (
        "Docker Compose is required but was not found on your PATH.\n"
        "Please install Docker Compose:\n"
        "  - For Docker Compose v2 (recommended): "
        "https://docs.docker.com/compose/install/\n"
        "  - Or install legacy docker-compose: "
        "https://docs.docker.com/compose/compose-v1/\n"
        "After installation, ensure 'docker compose' or 'docker-compose' "
        "is available in your terminal."
    )
    raise RuntimeError(message)


def _compose_command(
    base: Sequence[str], *, file: str, action: str, services: Sequence[str]
) -> list[str]:
    """Build a docker compose command for the requested action."""
    command = [*base, "-f", file]
    if action == "start":
        command.extend(["up", "-d"])
    elif action == "stop":
        command.append("stop" if services else "down")
    else:
        command.append("ps")

    if services:
        command.extend(services)
    return command


def cmd_services(args: argparse.Namespace) -> int:
    """Manage supporting docker-compose services."""
    compose_cmd = _compose_base_command()

    if args.stack == "vector":
        command = _compose_command(
            compose_cmd,
            file="docker-compose.yml",
            action=args.action,
            services=("qdrant", "dragonfly"),
        )
    else:
        command = [*compose_cmd, "--profile", "monitoring", "-f", "docker-compose.yml"]
        if args.action == "start":
            command.extend(["up", "-d"])
        elif args.action == "stop":
            command.append("down")
        else:
            command.append("ps")

    exit_code = run_command(command)
    if (
        exit_code == 0
        and args.action == "start"
        and args.stack == "vector"
        and not args.skip_health_check
    ):
        if _check_service_health("http://localhost:6333/health"):
            print("✅ Qdrant is healthy")
        else:
            print("⚠️  Qdrant health endpoint not reachable; check docker logs")
    return exit_code


def cmd_deploy(args: argparse.Namespace) -> int:
    """Render and optionally execute the active deployment plan."""

    from src.config.loader import get_settings
    from src.services.deployment import DeploymentExecutionError, DeploymentManager

    settings = get_settings()
    manager = DeploymentManager(settings)

    try:
        plan = manager.build_plan(args.strategy)
    except ValueError as exc:  # pragma: no cover - guarded by argparse choices
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        manager.validate_plan(plan)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 2

    print(f"Strategy: {plan.strategy.value}")
    print(f"Description: {plan.description}")
    print(f"Entrypoint: {plan.entrypoint}")
    print(f"Available: {'yes' if plan.available else 'no'}")
    print("Commands:")
    for command in plan.formatted_commands():
        print(f"  - {command}")
    if plan.notes:
        print("Notes:")
        for note in plan.notes:
            print(f"  - {note}")
    if not args.apply:
        return 0

    try:
        manager.execute_plan(plan)
    except DeploymentExecutionError as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 3

    print("Deployment commands executed successfully.")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    """Run Ruff linting, optionally applying fixes."""
    if not _ensure_uv_available():
        return 1

    command = ["uv", "run", "ruff", "check", "."]
    if args.fix:
        command.append("--fix")
    return run_command(command)


def cmd_format(_: argparse.Namespace) -> int:
    """Format the codebase using Ruff."""
    if not _ensure_uv_available():
        return 1

    return run_command(["uv", "run", "ruff", "format", "."])


def cmd_typecheck(_: argparse.Namespace) -> int:
    """Run Pyright static type checking."""
    if not _ensure_uv_available():
        return 1

    return run_command(["uv", "run", "pyright"])


def cmd_quality(args: argparse.Namespace) -> int:
    """Execute the standard quality gate (format, lint, typecheck)."""
    if not _ensure_uv_available():
        return 1

    commands: list[list[str]] = []
    if not args.skip_format:
        commands.append(["uv", "run", "ruff", "format", "."])

    lint_cmd = ["uv", "run", "ruff", "check", "."]
    if args.fix_lint:
        lint_cmd.append("--fix")
    commands.extend(
        (
            lint_cmd,
            ["uv", "run", "pylint", "src", "tests"],
            ["uv", "run", "pyright"],
        )
    )

    exit_code = 0
    for command in commands:
        exit_code = max(exit_code, run_command(command))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Developer workflow helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run pytest with helpful defaults")
    test_parser.add_argument(
        "--profile",
        choices=sorted(PYTEST_PROFILES),
        default="quick",
        help="Select the preset test profile.",
    )
    test_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of pytest-xdist workers (0 selects an automatic value).",
    )
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Collect coverage data.",
    )
    test_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase verbosity.",
    )
    test_parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Forward any additional arguments to pytest.",
    )
    test_parser.set_defaults(func=cmd_test)

    eval_parser = subparsers.add_parser(
        "eval", help="Run the RAG golden evaluation harness"
    )
    eval_parser.add_argument(
        "--dataset",
        default="tests/data/rag/golden_set.jsonl",
        help="Path to the golden dataset (JSONL)",
    )
    eval_parser.add_argument(
        "--output",
        help="Optional JSON report destination",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum documents to retrieve per query",
    )
    eval_parser.add_argument(
        "--namespace",
        default="ml_app",
        help="Metrics namespace for the Prometheus snapshot",
    )
    eval_parser.add_argument(
        "--ragas-llm-model",
        default="gpt-4o-mini",
        help="Override the LLM model used for semantic evaluation",
    )
    eval_parser.add_argument(
        "--ragas-embedding-model",
        default="text-embedding-3-small",
        help="Override the embedding model used for semantic evaluation",
    )
    eval_parser.add_argument(
        "--max-semantic-samples",
        type=int,
        help="Maximum number of examples evaluated with RAGAS",
    )
    eval_parser.add_argument(
        "--metrics-allowlist",
        nargs="*",
        help="Prometheus metric names to include in the snapshot",
    )
    eval_parser.set_defaults(func=cmd_eval)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run performance benchmarks"
    )
    benchmark_parser.add_argument(
        "--suite",
        choices=["performance", "integration", "all"],
        default="performance",
        help="Benchmark suite to execute.",
    )
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for performance benchmarks.",
    )
    benchmark_parser.add_argument(
        "--output",
        help="Optional path to a JSON file where benchmark results should be stored.",
    )
    benchmark_parser.add_argument(
        "--verbose", action="store_true", help="Verbose pytest output."
    )
    benchmark_parser.add_argument(
        "--output-dir",
        help="Directory where per-suite benchmark JSON reports should be written.",
    )
    benchmark_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare benchmark results against a stored baseline JSON file.",
    )
    benchmark_parser.add_argument(
        "--baseline",
        default="benchmark_baseline.json",
        help="Baseline JSON file used when --compare-baseline is supplied.",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate project configuration"
    )
    validate_parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures."
    )
    validate_parser.add_argument(
        "--check-docs",
        action="store_true",
        help="Validate internal documentation links.",
    )
    validate_parser.add_argument(
        "--check-services",
        action="store_true",
        help="Verify local service health (Qdrant).",
    )
    validate_parser.set_defaults(func=cmd_validate)

    services_parser = subparsers.add_parser(
        "services", help="Manage docker-compose services"
    )
    services_parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="Action to perform.",
    )
    services_parser.add_argument(
        "--stack",
        choices=["vector", "monitoring"],
        default="vector",
        help="Service stack to manage.",
    )
    services_parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the post-start Qdrant health verification.",
    )
    services_parser.set_defaults(func=cmd_services)

    from src.config.models import (
        DeploymentStrategy,
    )  # Local import to avoid module import during startup

    deploy_parser = subparsers.add_parser(
        "deploy", help="Show the active deployment plan or execute it"
    )
    deploy_parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in DeploymentStrategy],
        help="Override the configured deployment strategy.",
    )
    deploy_parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the resolved deployment commands after validation.",
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    lint_parser = subparsers.add_parser("lint", help="Run Ruff lint checks")
    lint_parser.add_argument(
        "--fix", action="store_true", help="Automatically apply fixes."
    )
    lint_parser.set_defaults(func=cmd_lint)

    format_parser = subparsers.add_parser(
        "format", help="Format the codebase with Ruff"
    )
    format_parser.set_defaults(func=cmd_format)

    typecheck_parser = subparsers.add_parser(
        "typecheck", help="Run Pyright type checking"
    )
    typecheck_parser.set_defaults(func=cmd_typecheck)

    quality_parser = subparsers.add_parser("quality", help="Run the full quality gate")
    quality_parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Skip running Ruff format before linting.",
    )
    quality_parser.add_argument(
        "--fix-lint",
        action="store_true",
        help="Allow Ruff to apply lint fixes.",
    )
    quality_parser.set_defaults(func=cmd_quality)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
