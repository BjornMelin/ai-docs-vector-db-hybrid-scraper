"""Developer task runner commands exposed through the main CLI.

These commands bridge the curated developer workflows implemented in
``scripts/dev.py`` with the Rich-powered Click interface. They replace the
legacy ``unified`` CLI so that all entry points share the same user
experience while still reusing the battle-tested automation helpers.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import click
import uvicorn

from scripts import dev as dev_script


APP_IMPORT_PATH = "src.api.main:app"


@dataclass(frozen=True)
class RagasOptions:
    """Ragas-specific configuration for evaluation commands."""

    enabled: bool
    model: str | None
    embedding: str | None
    max_samples: int | None


@dataclass(frozen=True)
class EvalOptions:
    """Structured container for evaluation command parameters."""

    dataset: str
    output: str | None
    limit: int
    namespace: str
    ragas: RagasOptions
    metrics_allowlist: tuple[str, ...]


def _invoke_dev_script(*args: str) -> None:
    """Execute a ``scripts.dev`` command and surface failures as CLI errors.

    Args:
        *args: Positional arguments that form the ``scripts.dev`` CLI call.

    Raises:
        click.ClickException: Raised when the underlying command exits with a
            non-zero status code.
    """

    exit_code = dev_script.main(list(args))
    if exit_code != 0:
        raise click.ClickException(
            f"scripts.dev {' '.join(args)} failed with exit code {exit_code}"
        )


def _run_subprocess(
    command: Sequence[str], *, env: dict[str, str] | None = None
) -> None:
    """Run a subprocess command and surface failures as Click exceptions.

    Args:
        command: Command arguments to execute.
        env: Optional environment variables for the subprocess.

    Raises:
        click.ClickException: Raised when the command exits with a non-zero
            status code.
    """

    result = subprocess.run(  # noqa: S603
        list(command),
        check=False,
        env=env,
        shell=False,
    )
    if result.returncode != 0:
        joined = " ".join(command)
        raise click.ClickException(
            f"Command '{joined}' failed with exit code {result.returncode}"
        )


def _build_eval_arguments(options: EvalOptions) -> list[str]:
    """Construct the ``scripts.dev`` argument list from evaluation options."""

    args = [
        "eval",
        "--dataset",
        options.dataset,
        "--limit",
        str(options.limit),
        "--namespace",
        options.namespace,
    ]
    if options.output:
        args.extend(["--output", options.output])
    if options.ragas.enabled:
        args.append("--enable-ragas")
    if options.ragas.model:
        args.extend(["--ragas-model", options.ragas.model])
    if options.ragas.embedding:
        args.extend(["--ragas-embedding", options.ragas.embedding])
    if options.ragas.max_samples is not None:
        args.extend(["--ragas-max-samples", str(options.ragas.max_samples)])
    if options.metrics_allowlist:
        args.append("--metrics-allowlist")
        args.extend(options.metrics_allowlist)
    return args


@click.command()
@click.option("--mode", type=click.Choice(["simple", "enterprise"]), default="simple")
@click.option("--reload/--no-reload", default=True)
@click.option("--host", default="0.0.0.0")  # noqa: S104
@click.option("--port", default=8000)
def dev(mode: str, reload: bool, host: str, port: int) -> None:
    """Start the FastAPI development server with the selected configuration.

    Args:
        mode: Application mode to expose through the ``AI_DOCS__MODE`` setting.
        reload: Whether to enable auto-reload for local development.
        host: Interface to bind the server to.
        port: Port to expose the FastAPI service on.
    """

    os.environ["AI_DOCS__MODE"] = mode
    click.echo(f"🚀 Starting development server in {mode} mode")
    click.echo(f"📍 Server: http://{host}:{port}")
    uvicorn.run(APP_IMPORT_PATH, reload=reload, host=host, port=port)


@click.command()
@click.option(
    "--profile",
    type=click.Choice(["quick", "unit", "integration", "performance", "full", "ci"]),
    default="quick",
)
@click.option("--coverage/--no-coverage", default=False)
@click.option("--verbose/--quiet", default=False)
@click.option(
    "--workers", type=int, default=0, help="Number of xdist workers (0 = auto)"
)
@click.argument("extra_args", nargs=-1)
def test(
    profile: str,
    coverage: bool,
    verbose: bool,
    workers: int,
    extra_args: tuple[str, ...],
) -> None:
    """Run the pytest task profiles through ``scripts.dev``.

    Args:
        profile: Named pytest profile to execute.
        coverage: Whether to enable coverage instrumentation.
        verbose: Toggle verbose pytest output.
        workers: Explicit worker count when using pytest-xdist.
        extra_args: Additional arguments forwarded to pytest.
    """

    args = ["test", "--profile", profile]
    if coverage:
        args.append("--coverage")
    if verbose:
        args.append("--verbose")
    if workers > 0:
        args.extend(["--workers", str(workers)])
    if extra_args:
        args.append("--")
        args.extend(extra_args)
    click.echo(f"🧪 Running {profile} test profile")
    _invoke_dev_script(*args)


@click.command()
@click.option("--skip-format/--no-skip-format", default=False)
@click.option("--fix-lint/--no-fix-lint", default=True)
def quality(skip_format: bool, fix_lint: bool) -> None:
    """Run the aggregated lint, type-check, and test quality gate.

    Args:
        skip_format: Skip running the formatter before linting.
        fix_lint: Allow Ruff to automatically fix lint issues.
    """

    args = ["quality"]
    if skip_format:
        args.append("--skip-format")
    if fix_lint:
        args.append("--fix-lint")
    click.echo("🔍 Running code quality checks...")
    _invoke_dev_script(*args)


@click.command()
@click.option("--host", default="0.0.0.0")  # noqa: S104
@click.option("--port", default=8001)
def docs(host: str, port: int) -> None:
    """Serve the MkDocs documentation locally.

    Args:
        host: Interface for the MkDocs server.
        port: Port for the MkDocs server.
    """

    click.echo(f"📚 Starting documentation server at http://{host}:{port}")
    _run_subprocess(["mkdocs", "serve", "--host", host, "--port", str(port)])


@click.command()
@click.option(
    "--action", type=click.Choice(["start", "stop", "status"]), default="start"
)
@click.option("--stack", type=click.Choice(["vector", "monitoring"]), default="vector")
@click.option("--skip-health-check/--no-skip-health-check", default=False)
def services(action: str, stack: str, skip_health_check: bool) -> None:
    """Manage the local docker-compose service stacks.

    Args:
        action: Service lifecycle operation to perform.
        stack: Target docker-compose profile.
        skip_health_check: Skip post-start health verification.
    """

    args = ["services", action, "--stack", stack]
    if skip_health_check:
        args.append("--skip-health-check")
    click.echo(f"🚀 Services command: {action} ({stack})")
    _invoke_dev_script(*args)


@click.command()
@click.option(
    "--profile",
    type=click.Choice(["standard", "performance", "integration", "all"]),
    default="standard",
)
def benchmark(profile: str) -> None:
    """Run the performance benchmarking suites.

    Args:
        profile: Named benchmark profile to execute.
    """

    click.echo(f"⚡ Running {profile} benchmark profile...")
    _invoke_dev_script("benchmark", "--suite", profile)


@click.command(name="eval")
@click.option(
    "--dataset",
    default="tests/data/rag/golden_set.jsonl",
    show_default=True,
    help="Golden dataset path (JSONL)",
)
@click.option("--output", help="Optional JSON report destination")
@click.option("--limit", type=int, default=5, show_default=True)
@click.option("--namespace", default="ml_app", show_default=True)
@click.option("--enable-ragas", is_flag=True, default=False)
@click.option("--ragas-model")
@click.option("--ragas-embedding")
@click.option("--ragas-max-samples", type=int)
@click.option(
    "--metrics-allowlist",
    multiple=True,
    help="Restrict the Prometheus metrics snapshot to specific names",
)
@click.pass_context
def run_eval(ctx: click.Context, **params: Any) -> None:
    """Run the RAG golden evaluation harness via ``scripts.dev``.

    Args:
        ctx: Click context providing configuration and parameters.
        **params: Keyword arguments collected from Click options.
    """

    ragas = RagasOptions(
        enabled=params["enable_ragas"],
        model=params["ragas_model"],
        embedding=params["ragas_embedding"],
        max_samples=params["ragas_max_samples"],
    )
    options = EvalOptions(
        dataset=params["dataset"],
        output=params["output"],
        limit=params["limit"],
        namespace=params["namespace"],
        ragas=ragas,
        metrics_allowlist=params["metrics_allowlist"],
    )
    args = _build_eval_arguments(options)
    click.echo("🎯 Running RAG golden evaluation harness")
    _invoke_dev_script(*args)


@click.command()
def validate() -> None:
    """Validate project configuration and service health.

    Raises:
        click.ClickException: Raised when validation fails.
    """

    click.echo("🔍 Validating project configuration...")
    _invoke_dev_script("validate", "--check-docs")


__all__ = [
    "APP_IMPORT_PATH",
    "benchmark",
    "dev",
    "docs",
    "EvalOptions",
    "RagasOptions",
    "quality",
    "run_eval",
    "services",
    "test",
    "validate",
]
