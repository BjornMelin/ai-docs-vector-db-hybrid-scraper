#!/usr/bin/env python3
"""Unified CLI interface for AI Docs Vector DB development workflow."""

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_SCRIPT = REPO_ROOT / "scripts" / "dev.py"

BENCHMARK_SUITES: dict[str, str] = {
    "standard": "performance",
    "performance": "performance",
    "integration": "integration",
    "all": "all",
}


def _normalize_command(command: Sequence[str | os.PathLike[str]]) -> list[str]:
    """Convert command arguments into safe string tokens."""
    normalized: list[str] = []
    for token in command:
        token_str = os.fspath(token)
        if "\x00" in token_str:
            msg = "Command tokens must not contain NUL characters."
            raise ValueError(msg)
        normalized.append(token_str)
    return normalized


def _run_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    check: bool = False,
    capture_output: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Execute a subprocess with defensively normalised arguments."""
    if "shell" in kwargs:
        msg = "shell execution is not permitted for CLI commands"
        raise ValueError(msg)
    normalized = _normalize_command(command)
    return subprocess.run(
        normalized,
        check=check,
        capture_output=capture_output,
        shell=False,
        **kwargs,
    )


def _dev_script_command(*args: str) -> list[str | os.PathLike[str]]:
    """Build an invocation of the shared dev helper script."""
    return [sys.executable, DEV_SCRIPT, *args]


@click.group()
def cli():
    """AI Docs Vector DB - Unified Development CLI."""


@cli.command()
@click.option("--reload/--no-reload", default=True)
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
def dev(reload: bool, host: str, port: int):
    """Start development server using the unified configuration."""
    click.echo("Starting development server with unified configuration")
    click.echo(f"Server: http://{host}:{port}")

    uvicorn.run("src.api.main:app", reload=reload, host=host, port=port)


@cli.command()
@click.option(
    "--profile",
    type=click.Choice(["quick", "unit", "integration", "performance", "full", "ci"]),
    default="quick",
)
@click.option("--coverage/--no-coverage", default=False)
@click.option("--verbose/--quiet", default=False)
@click.option(
    "--workers", type=int, default=0, help="Number of parallel workers (0 = auto)"
)
@click.argument("extra_args", nargs=-1)
def test(
    profile: str,
    coverage: bool,
    verbose: bool,
    workers: int,
    extra_args: tuple[str, ...],
):
    """Run test suite with optimized feedback loops."""
    cmd: list[str | os.PathLike[str]] = _dev_script_command(
        "test", "--profile", profile
    )

    if coverage:
        cmd.append("--coverage")
    if verbose:
        cmd.append("--verbose")
    if workers > 0:
        cmd.extend(["--workers", str(workers)])
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    click.echo(f"Running {profile} test profile")
    _run_command(cmd)


@cli.command()
def setup():
    """Complete development environment setup."""
    click.echo("Setting up development environment...")

    # Create .env.local if it doesn't exist
    env_local = Path(".env.local")
    if not env_local.exists():
        click.echo("Creating .env.local from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            env_local.write_text(
                env_example.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            env_local.write_text(
                (
                    "AI_DOCS__ENABLE_ADVANCED_MONITORING=true\n"
                    "AI_DOCS__ENABLE_DEPLOYMENT_FEATURES=true\n"
                    "AI_DOCS__ENABLE_AB_TESTING=false\n"
                    "AI_DOCS__DEBUG=true\n"
                ),
                encoding="utf-8",
            )

    # Install pre-commit hooks
    click.echo("Installing pre-commit hooks...")
    _run_command(["uv", "run", "pre-commit", "install"], capture_output=True)

    # Validate configuration
    click.echo("Validating configuration...")
    _run_command(_dev_script_command("validate"), capture_output=True)

    click.echo("Setup complete. Run 'task dev' to start development.")


@cli.command()
@click.option("--skip-format/--no-skip-format", default=False)
@click.option("--fix-lint/--no-fix-lint", default=True)
def quality(skip_format: bool, fix_lint: bool):
    """Run code quality checks (format, lint, typecheck)."""
    cmd: list[str | os.PathLike[str]] = _dev_script_command("quality")
    if skip_format:
        cmd.append("--skip-format")
    if fix_lint:
        cmd.append("--fix-lint")

    click.echo("Running code quality checks...")
    result = _run_command(cmd)

    if result.returncode == 0:
        click.echo("All quality checks passed.")
    else:
        click.echo("Quality checks failed.")
        sys.exit(result.returncode)


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8001)
def docs(host: str, port: int):
    """Serve documentation locally."""
    click.echo(f"Starting documentation server at http://{host}:{port}")
    _run_command(["mkdocs", "serve", "--host", host, "--port", str(port)])


@cli.command()
@click.option(
    "--action",
    type=click.Choice(["start", "stop", "status"]),
    default="start",
)
@click.option("--stack", type=click.Choice(["vector", "monitoring"]), default="vector")
@click.option("--skip-health-check/--no-skip-health-check", default=False)
def services(action: str, stack: str, skip_health_check: bool):
    """Manage local services (Qdrant, monitoring stack)."""
    cmd: list[str | os.PathLike[str]] = _dev_script_command(
        "services",
        action,
        "--stack",
        stack,
    )
    if skip_health_check:
        cmd.append("--skip-health-check")

    click.echo(f"Running services command: {action} ({stack})")
    _run_command(cmd)


@cli.command()
@click.option(
    "--profile", type=click.Choice(tuple(BENCHMARK_SUITES)), default="standard"
)
def benchmark(profile: str):
    """Run performance benchmarks."""
    click.echo(f"Running {profile} benchmark profile...")
    suite = BENCHMARK_SUITES.get(profile, "performance")
    _run_command(_dev_script_command("benchmark", "--suite", suite))


@cli.command(name="eval")
@click.option(
    "--dataset",
    default="tests/data/rag/golden_set.jsonl",
    show_default=True,
    help="Golden dataset path (JSONL)",
)
@click.option("--output", help="Optional JSON report destination")
@click.option("--limit", type=int, default=5, show_default=True)
@click.option("--namespace", default="ml_app", show_default=True)
@click.option(
    "--ragas-llm-model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model for semantic evaluation",
)
@click.option(
    "--ragas-embedding-model",
    default="text-embedding-3-small",
    show_default=True,
    help="Embedding model for semantic evaluation",
)
@click.option("--max-semantic-samples", type=int)
@click.option(
    "--metrics-allowlist",
    multiple=True,
    help="Restrict the Prometheus metrics snapshot to specific names",
)
def run_eval(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dataset: str,
    output: str | None,
    limit: int,
    namespace: str,
    ragas_llm_model: str,
    ragas_embedding_model: str,
    max_semantic_samples: int | None,
    metrics_allowlist: tuple[str, ...],
):
    """Run the RAG golden evaluation harness."""
    click.echo("Running RAG golden evaluation harness")
    cmd: list[str | os.PathLike[str]] = _dev_script_command(
        "eval",
        "--dataset",
        dataset,
        "--limit",
        str(limit),
        "--namespace",
        namespace,
    )

    if output:
        cmd.extend(["--output", output])
    cmd.extend(["--ragas-llm-model", ragas_llm_model])
    cmd.extend(["--ragas-embedding-model", ragas_embedding_model])
    if max_semantic_samples is not None:
        cmd.extend(["--max-semantic-samples", str(max_semantic_samples)])
    if metrics_allowlist:
        cmd.append("--metrics-allowlist")
        cmd.extend(metrics_allowlist)

    _run_command(cmd)


@cli.command()
def validate():
    """Validate project configuration and health."""
    click.echo("Validating project configuration...")

    result = _run_command(
        _dev_script_command("validate", "--check-docs"),
    )

    if result.returncode == 0:
        click.echo("All validations passed.")
    else:
        click.echo("Validation failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    cli()
