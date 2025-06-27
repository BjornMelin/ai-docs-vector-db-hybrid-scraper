"""Simplified configuration management commands with Rich styling.

Core configuration management functionality for V1 release.
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.config import Config

try:
    import yaml
except ImportError:
    yaml = None


console = Console()


@click.group()
def config():
    """⚙️ Configuration management commands."""
    pass


@config.command()
@click.option(
    "--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table"
)
@click.pass_context
def show(ctx: click.Context, format: str):
    """Show current configuration."""
    config_obj = ctx.obj["config"]

    if format == "table":
        _show_config_table(config_obj)
    elif format == "json":
        _show_config_json(config_obj)
    elif format == "yaml":
        _show_config_yaml(config_obj)


@config.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "yaml"]), default="json")
@click.pass_context
def export(ctx: click.Context, output: str, format: str):
    """Export configuration to file."""
    config_obj = ctx.obj["config"]

    output_path = Path(output) if output else Path(f"config.{format}")

    try:
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(config_obj.model_dump(), f, indent=2)
        elif format == "yaml":
            if yaml is None:
                console.print("❌ YAML support not available. Please install PyYAML", style="red")
                return
            with open(output_path, "w") as f:
                yaml.dump(config_obj.model_dump(), f, default_flow_style=False)

        console.print(f"✅ Configuration exported to {output_path}", style="green")
    except Exception as e:
        console.print(f"❌ Export failed: {e}", style="red")


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--validate-only", is_flag=True, help="Only validate, don't load")
def load(config_file: str, validate_only: bool):
    """Load configuration from file."""
    try:
        config_path = Path(config_file)
        config_obj = Config.load_from_file(config_path)

        if validate_only:
            console.print("✅ Configuration file is valid", style="green")
        else:
            console.print(f"✅ Configuration loaded from {config_path}", style="green")
            _show_config_table(config_obj)

    except Exception as e:
        console.print(f"❌ Failed to load configuration: {e}", style="red")


@config.command()
@click.pass_context
def validate(ctx: click.Context):
    """Validate current configuration."""
    config_obj = ctx.obj["config"]

    try:
        # Basic validation - config is already validated on creation
        console.print("✅ Configuration is valid", style="green")

        # Show environment and provider info
        table = Table(title="Configuration Summary", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Environment", str(config_obj.environment))
        table.add_row("Debug Mode", str(config_obj.debug))
        table.add_row("Log Level", str(config_obj.log_level))
        table.add_row("Embedding Provider", str(config_obj.embedding_provider))
        table.add_row("Crawl Provider", str(config_obj.crawl_provider))

        console.print(table)

    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")


def _show_config_table(config_obj: Config):
    """Display configuration as a formatted table."""
    table = Table(
        title="Configuration Settings", show_header=True, header_style="bold cyan"
    )
    table.add_column("Component", style="dim", width=20)
    table.add_column("Settings", width=60)

    # Core settings
    core_settings = [
        f"Environment: {config_obj.environment}",
        f"Debug: {config_obj.debug}",
        f"Log Level: {config_obj.log_level}",
        f"App Name: {config_obj.app_name}",
        f"Version: {config_obj.version}",
    ]
    table.add_row("Core", "\n".join(core_settings))

    # Provider settings
    provider_settings = [
        f"Embedding: {config_obj.embedding_provider}",
        f"Crawl: {config_obj.crawl_provider}",
    ]
    table.add_row("Providers", "\n".join(provider_settings))

    # Database settings
    db_settings = [
        f"Qdrant URL: {config_obj.qdrant.url}",
        f"Collection: {config_obj.qdrant.collection_name}",
        f"Timeout: {config_obj.qdrant.timeout}s",
    ]
    table.add_row("Database", "\n".join(db_settings))

    # Cache settings
    cache_settings = [
        f"Enabled: {config_obj.cache.enable_caching}",
        f"URL: {config_obj.cache.dragonfly_url}",
        f"Max Size: {config_obj.cache.local_max_size}",
    ]
    table.add_row("Cache", "\n".join(cache_settings))

    console.print(table)


def _show_config_json(config_obj: Config):
    """Display configuration as JSON."""
    config_json = json.dumps(config_obj.model_dump(), indent=2)
    syntax = Syntax(config_json, "json", theme="monokai", line_numbers=True)

    panel = Panel(
        syntax,
        title="Configuration (JSON)",
        border_style="cyan",
    )
    console.print(panel)


def _show_config_yaml(config_obj: Config):
    """Display configuration as YAML."""
    if yaml is None:
        console.print("❌ YAML support not available. Please install PyYAML", style="red")
        return
    
    config_yaml = yaml.dump(config_obj.model_dump(), default_flow_style=False)
    syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)

    panel = Panel(
        syntax,
        title="Configuration (YAML)",
        border_style="cyan",
    )
    console.print(panel)
