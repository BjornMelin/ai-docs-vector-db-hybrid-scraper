#!/usr/bin/env python3
"""Unified CLI interface for AI Docs Vector DB development workflow."""

import os
import subprocess
import sys
from pathlib import Path

import click
import uvicorn


@click.group()
def cli():
    """AI Docs Vector DB - Unified Development CLI"""


@cli.command()
@click.option("--mode", type=click.Choice(["simple", "enterprise"]), default="simple")
@click.option("--reload/--no-reload", default=True)
@click.option("--host", default="0.0.0.0")  # noqa: S104
@click.option("--port", default=8000)
def dev(mode: str, reload: bool, host: str, port: int):
    """Start development server"""

    os.environ["AI_DOCS__MODE"] = mode
    click.echo(f"🚀 Starting development server in {mode} mode")
    click.echo(f"📍 Server: http://{host}:{port}")

    uvicorn.run("src.api.main:app", reload=reload, host=host, port=port)


@cli.command()
@click.option(
    "--profile",
    type=click.Choice(["unit", "fast", "integration", "full"]),
    default="fast",
)
@click.option("--coverage/--no-coverage", default=False)
@click.option("--verbose/--quiet", default=False)
@click.option(
    "--parallel", type=int, default=0, help="Number of parallel workers (0 = auto)"
)
def test(profile: str, coverage: bool, verbose: bool, parallel: int):
    """Run test suite with optimized feedback loops"""
    cmd = ["python", "scripts/run_fast_tests.py", "--profile", profile]

    if coverage:
        cmd.append("--coverage")
    if verbose:
        cmd.append("--verbose")
    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])

    click.echo(f"🧪 Running {profile} test profile")
    subprocess.run(cmd, check=False)


@cli.command()
def setup():
    """Complete development environment setup"""
    click.echo("🔧 Setting up development environment...")

    # Create .env.local if it doesn't exist
    env_local = Path(".env.local")
    if not env_local.exists():
        click.echo("📝 Creating .env.local from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            env_local.write_text(env_example.read_text())
        else:
            env_local.write_text("AI_DOCS__MODE=simple\nAI_DOCS__DEBUG=true\n")

    # Install pre-commit hooks
    click.echo("🪝 Installing pre-commit hooks...")
    subprocess.run(
        ["uv", "run", "pre-commit", "install"],  # noqa: S607
        check=False,
        capture_output=True,
    )

    # Validate configuration
    click.echo("✅ Validating configuration...")
    subprocess.run(
        ["python", "scripts/validate_config.py"],  # noqa: S607
        check=False,
        capture_output=True,
    )

    click.echo("✅ Setup complete! Run 'task dev' to start development.")


@cli.command()
@click.option("--fix/--no-fix", default=True)
def quality():
    """Run code quality checks (format, lint, typecheck)"""
    click.echo("🔍 Running code quality checks...")

    # Format code
    click.echo("📝 Formatting code...")
    subprocess.run(["ruff", "format", "."], check=False)  # noqa: S607

    # Lint code
    click.echo("🧹 Linting code...")
    subprocess.run(["ruff", "check", ".", "--fix"], check=False)  # noqa: S607

    # Type check
    click.echo("🔍 Type checking...")
    result = subprocess.run(
        ["mypy", "src/", "--config-file", "pyproject.toml"],  # noqa: S607
        check=False,
    )

    if result.returncode == 0:
        click.echo("✅ All quality checks passed!")
    else:
        click.echo("❌ Quality checks failed!")
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0")  # noqa: S104
@click.option("--port", default=8001)
def docs(host: str, port: int):
    """Serve documentation locally"""
    click.echo(f"📚 Starting documentation server at http://{host}:{port}")
    subprocess.run(
        ["mkdocs", "serve", "--host", host, "--port", str(port)],  # noqa: S607
        check=False,
    )


@cli.command()
def services():
    """Start local services (Qdrant, Redis)"""
    click.echo("🚀 Starting local services...")
    subprocess.run(["./scripts/start-services.sh"], check=False)


@cli.command()
@click.option("--profile", default="standard")
def benchmark(profile: str):
    """Run performance benchmarks"""
    click.echo(f"⚡ Running {profile} benchmark profile...")
    subprocess.run(
        ["python", "scripts/run_benchmarks.py", "--profile", profile],  # noqa: S607
        check=False,
    )


@cli.command()
def validate():
    """Validate project configuration and health"""
    click.echo("🔍 Validating project configuration...")

    # Validate configuration
    result1 = subprocess.run(["python", "scripts/validate_config.py"], check=False)  # noqa: S607

    # Validate documentation links
    result2 = subprocess.run(["python", "scripts/validate_docs_links.py"], check=False)  # noqa: S607

    if result1.returncode == 0 and result2.returncode == 0:
        click.echo("✅ All validations passed!")
    else:
        click.echo("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    cli()
