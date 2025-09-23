"""Main unified CLI entry point for AI Documentation Scraper.

This module provides a comprehensive CLI interface that integrates all
existing functionality under a unified command structure with enhanced
user experience features.
"""

import sys
from pathlib import Path

import click
from click.shell_completion import get_completion_class
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import Config, get_config
from src.utils.health_checks import ServiceHealthChecker

# Import command groups
from .commands import (
    batch as batch_commands,
    config as config_commands,
    database as db_commands,
    setup as setup_commands,
)


console = Console()


class RichCLI:
    """Enhanced CLI interface with Rich console integration."""

    def __init__(self):
        """Initialize the Rich CLI."""
        self.console = Console()

    def show_welcome(self):
        """Display welcome message with project info."""
        welcome_text = Text()
        welcome_text.append("🚀 AI Documentation Scraper\n", style="bold cyan")
        welcome_text.append("Advanced CLI Interface v1.0.0\n", style="dim")
        welcome_text.append(
            "\nHybrid AI documentation scraping system with vector database integration",
            style="",
        )

        panel = Panel(
            welcome_text,
            title="Welcome",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def show_error(self, message: str, details: str | None = None):
        """Display error message with Rich formatting."""
        error_text = Text()
        error_text.append("❌ Error: ", style="bold red")
        error_text.append(message, style="red")

        if details:
            error_text.append(f"\n\nDetails: {details}", style="dim red")

        panel = Panel(
            error_text,
            title="Error",
            title_align="left",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print(panel)


# Initialize Rich CLI
rich_cli = RichCLI()


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress welcome message")
@click.version_option(version="1.0.0", prog_name="AI Documentation Scraper CLI")
@click.pass_context
def main(ctx: click.Context, config: Path | None, quiet: bool):
    """🚀 AI Documentation Scraper - Advanced CLI Interface.

    A comprehensive command-line interface for managing your AI documentation
    scraping workflow with vector database integration.

    Use --help with any command to get detailed information.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    try:
        if config:
            # Load from specific file using Pydantic BaseSettings parse_file method
            ctx.obj["config"] = Config.parse_file(config)
        else:
            ctx.obj["config"] = get_config()
    except (OSError, ValueError, RuntimeError) as e:
        rich_cli.show_error("Failed to load configuration", details=str(e))
        sys.exit(1)

    # Store Rich CLI instance
    ctx.obj["rich_cli"] = rich_cli

    # Show welcome message if no command specified and not quiet
    if ctx.invoked_subcommand is None:
        if not quiet:
            rich_cli.show_welcome()

        # Show available commands
        click.echo("\nAvailable commands:")
        click.echo("  setup    🧙 Interactive configuration wizard")
        click.echo("  config   ⚙️  Configuration management")
        click.echo("  database 🗄️  Vector database operations")
        click.echo("  batch    📦 Batch operations")
        click.echo("  --help   ❓ Show this help message")
        click.echo("\nUse 'ai-docs COMMAND --help' for command-specific help.")


# Add command groups
main.add_command(setup_commands.setup, "setup")
main.add_command(config_commands.config, "config")
main.add_command(db_commands.database, "database")
main.add_command(batch_commands.batch, "batch")


@main.command()
@click.pass_context
def version(ctx: click.Context):
    """Show version information."""
    rich_cli = ctx.obj["rich_cli"]

    version_text = Text()
    version_text.append("AI Documentation Scraper CLI\n", style="bold cyan")
    version_text.append("Version: 1.0.0\n", style="green")
    version_text.append("Python: ", style="dim")
    version_text.append(f"{sys.version.split()[0]}\n", style="yellow")

    panel = Panel(
        version_text,
        title="Version Information",
        title_align="left",
        border_style="green",
    )
    rich_cli.console.print(panel)


@main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish", "powershell"]))
def completion(shell: str):
    r"""Generate shell completion script.

    To enable auto-completion, run:

    \b
    Bash:
        eval "$(ai-docs completion bash)"

    Zsh:
        eval "$(ai-docs completion zsh)"

    Fish:
        ai-docs completion fish | source

    PowerShell:
        ai-docs completion powershell | Out-String | Invoke-Expression

    Add the above line to your shell's configuration file (.bashrc, .zshrc, etc.)
    to make completion persistent across sessions.
    """
    # Generate completion script using Click's built-in support
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        rich_cli.show_error(f"Shell '{shell}' is not supported for completion")
        sys.exit(1)

    # Get the command name (ai-docs)
    prog_name = "ai-docs"

    # Generate completion script
    comp = comp_cls(
        cli=main,
        ctx_args={},
        prog_name=prog_name,
        complete_var=f"_{prog_name.upper().replace('-', '_')}_COMPLETE",
    )

    try:
        completion_script = comp.source()
        click.echo(completion_script)
    except (ValueError, RuntimeError, OSError) as e:
        rich_cli.show_error("Failed to generate completion script", str(e))
        sys.exit(1)


@main.command()
@click.pass_context
def status(ctx: click.Context):
    """Show system status and health check."""
    rich_cli = ctx.obj["rich_cli"]
    config = ctx.obj["config"]

    # Run health checks
    with rich_cli.console.status("[bold green]Checking system status..."):
        results = ServiceHealthChecker.perform_all_health_checks(config)

    # Create status table
    table = Table(title="System Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim", width=20)
    table.add_column("Status", width=10)
    table.add_column("Details", width=40)

    for component, result in results.items():
        status_style = "green" if result["connected"] else "red"
        status_icon = "✅" if result["connected"] else "❌"

        table.add_row(
            component.title(),
            f"{status_icon} {'Healthy' if result['connected'] else 'Error'}",
            result.get("error", "") or "Connected",
            style=status_style,
        )

    rich_cli.console.print(table)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
