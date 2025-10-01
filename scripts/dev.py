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
    "integration": PytestProfile(("tests/integration", "-m", "integration")),
    "performance": PytestProfile(
        (
            "tests/performance",
            "-m",
            "performance or benchmark",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-group-by=group",
        ),
        uses_workers=False,
    ),
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


def _build_integration_benchmark_command(
    verbose: bool, output: str | None
) -> list[str]:
    """Return the command used for integration benchmarks."""

    command = [
        "uv",
        "run",
        "pytest",
        "tests/integration",
        "-m",
        "performance",
        "--benchmark-only",
        "--benchmark-sort=mean",
    ]
    if output:
        command.append(f"--benchmark-json={output}")
    if verbose:
        command.append("-v")
    return command


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run performance benchmark suites."""

    workers = _auto_worker_count(args.workers)
    suites: tuple[str, ...] = (
        ("performance", "integration") if args.suite == "all" else (args.suite,)
    )

    if len(suites) > 1 and args.output and not args.output_dir:
        print(
            "⚠️ Ignoring --output because multiple suites are selected; "
            "use --output-dir for per-suite reports.",
            file=sys.stderr,
        )
        output_override: str | None = None
    else:
        output_override = args.output

    exit_code = 0
    for suite in suites:
        output_path = output_override
        if args.output_dir:
            target = Path(args.output_dir) / f"{suite}_benchmark_results.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(target)

        if suite == "performance":
            command = _build_pytest_command(
                "performance",
                workers=workers,
                coverage=False,
                verbose=args.verbose,
                extra=[],
            )
            if output_path:
                command.append(f"--benchmark-json={output_path}")
        else:
            command = _build_integration_benchmark_command(args.verbose, output_path)

        if args.compare_baseline:
            baseline_path = Path(args.baseline).resolve()
            if baseline_path.exists():
                command.append(f"--benchmark-compare={baseline_path}")
            else:
                print(
                    f"⚠️ Baseline file '{baseline_path}' not found; "
                    f"skipping comparison.",
                    file=sys.stderr,
                )

        print(f"\n⚡ Running {suite} benchmarks")
        exit_code = max(exit_code, run_command(command))

    return exit_code


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


def cmd_validate(args: argparse.Namespace) -> int:
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
        PROJECT_ROOT / "src/api/main.py",
        PROJECT_ROOT / "src/cli/unified.py",
        PROJECT_ROOT / "tests",
    ]
    for path in required_paths:
        if path.exists():
            continue
        errors.append(
            f"Missing required file or directory: {path.relative_to(PROJECT_ROOT)}"
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


def cmd_lint(args: argparse.Namespace) -> int:
    """Run Ruff linting, optionally applying fixes."""

    command = ["uv", "run", "ruff", "check", "."]
    if args.fix:
        command.append("--fix")
    return run_command(command)


def cmd_format(_: argparse.Namespace) -> int:
    """Format the codebase using Ruff."""

    return run_command(["uv", "run", "ruff", "format", "."])


def cmd_typecheck(_: argparse.Namespace) -> int:
    """Run Pyright static type checking."""

    return run_command(["uv", "run", "pyright"])


def cmd_quality(args: argparse.Namespace) -> int:
    """Execute the standard quality gate (format, lint, typecheck)."""

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
