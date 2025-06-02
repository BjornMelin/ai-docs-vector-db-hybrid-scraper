"""CLI tool for managing unified configuration.

This tool provides commands for creating, validating, and converting
configuration files.
"""

from pathlib import Path
from typing import Any

import click
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .loader import ConfigLoader
from .schema import ConfigSchemaGenerator

console = Console()


def _get_service_icon(service_name: str) -> str:
    """Get icon for service display."""
    icons = {
        "qdrant": "🗃️",
        "redis": "🔄",
        "dragonfly": "🔄",
        "openai": "🤖",
        "firecrawl": "🕷️",
    }
    return icons.get(service_name, "🔍")


def _display_health_check_results(results: dict[str, dict[str, Any]]) -> None:
    """Display health check results in table format."""
    rprint("[bold]Checking Service Connections...[/bold]\n")

    for service_name, result in results.items():
        service_icon = _get_service_icon(service_name)
        rprint(f"{service_icon} Checking {service_name.capitalize()}...")

        if result["connected"]:
            status_msg = f"[green]✓[/green] {service_name.capitalize()} connected"

            # Add details if available
            details = result.get("details", {})
            if details:
                detail_parts = []
                if "collections_count" in details:
                    detail_parts.append(f"{details['collections_count']} collections")
                if "model" in details:
                    detail_parts.append(f"model: {details['model']}")
                if "available_models_count" in details:
                    detail_parts.append(
                        f"{details['available_models_count']} models available"
                    )
                if detail_parts:
                    status_msg += f" ({', '.join(detail_parts)})"

            rprint(status_msg)
        else:
            error_msg = result.get("error", "Unknown error")
            rprint(
                f"[red]✗[/red] {service_name.capitalize()} connection failed: {error_msg}"
            )

        rprint()  # Add spacing between services


@click.group()
def cli():
    """AI Documentation Vector DB configuration management tool."""
    pass


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["json", "yaml", "toml"]),
    default="json",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.example.json",
    help="Output file path",
)
def create_example(format: str, output: str):
    """Create an example configuration file."""
    output_path = Path(output)

    # Update extension based on format
    if format == "yaml":
        output_path = output_path.with_suffix(".yaml")
    elif format == "toml":
        output_path = output_path.with_suffix(".toml")

    ConfigLoader.create_example_config(output_path, format=format)
    rprint(f"[green]✓[/green] Created example config at: {output_path}")


@cli.command()
@click.option(
    "--output", "-o", type=click.Path(), default=".env.example", help="Output file path"
)
def create_env_template(output: str):
    """Create a .env template file."""
    ConfigLoader.create_env_template(output)
    rprint(f"[green]✓[/green] Created .env template at: {output}")


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file to validate",
)
@click.option("--env-file", "-e", type=click.Path(exists=True), help=".env file path")
@click.option("--show-config", is_flag=True, help="Show the loaded configuration")
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format for results (when --show-config is used)",
)
def validate(
    config_file: str | None, env_file: str | None, show_config: bool, output_format: str
):
    """Validate configuration from various sources."""
    try:
        # Load configuration
        config = ConfigLoader.load_config(
            config_file=config_file,
            env_file=env_file,
            include_env=True,
        )

        # Validate
        is_valid, issues = ConfigLoader.validate_config(config)

        if is_valid:
            rprint("[green]✓[/green] Configuration is valid!")
        else:
            rprint("[red]✗[/red] Configuration has issues:")
            for issue in issues:
                rprint(f"  [yellow]•[/yellow] {issue}")

        if show_config:
            if output_format == "json":
                console.print_json(data=config.model_dump())
            else:
                rprint("\n[bold]Loaded Configuration:[/bold]")
                rprint(config.model_dump_json(indent=2))

    except Exception as e:
        rprint(f"[red]Error loading configuration:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--from-format",
    type=click.Choice(["json", "yaml", "toml", "env"]),
    help="Input format (auto-detected if not specified)",
)
@click.option(
    "--to-format",
    type=click.Choice(["json", "yaml", "toml"]),
    required=True,
    help="Output format",
)
def convert(input_file: str, output_file: str, from_format: str | None, to_format: str):
    """Convert configuration between formats."""
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Auto-detect input format if not specified
    if not from_format:
        if input_path.suffix == ".json":
            from_format = "json"
        elif input_path.suffix in [".yaml", ".yml"]:
            from_format = "yaml"
        elif input_path.suffix == ".toml":
            from_format = "toml"
        elif input_path.name.startswith(".env"):
            from_format = "env"
        else:
            rprint(
                "[red]Could not auto-detect input format. Please specify --from-format[/red]"
            )
            raise click.Exit(1)

    try:
        # Load configuration
        if from_format == "env":
            # Special handling for .env files - load with env file support
            config = ConfigLoader.load_config(env_file=input_path, include_env=True)
        else:
            config = ConfigLoader.load_config(config_file=input_path, include_env=True)

        # Save in new format
        config.save_to_file(output_path, format=to_format)
        rprint(
            f"[green]✓[/green] Converted {input_file} ({from_format}) → {output_file} ({to_format})"
        )

    except Exception as e:
        rprint(f"[red]Error converting configuration:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file to load",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format for results",
)
def show_providers(config_file: str | None, output_format: str):
    """Show active provider configuration."""
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_file=config_file, include_env=True)

        # Get provider display data
        provider_data = ConfigLoader.get_provider_display_data(config)

        if output_format == "json":
            # Output JSON format
            console.print_json(data=provider_data)
        else:
            # Output table format
            table = Table(title="Active Provider Configuration")
            table.add_column("Provider Type", style="cyan")
            table.add_column("Selected Provider", style="green")
            table.add_column("Configuration", style="yellow")

            # Add rows for each provider
            for provider_type, provider_info in provider_data.items():
                config_lines = []
                for key, value in provider_info["configuration"].items():
                    config_lines.append(f"{key}: {value}")

                table.add_row(
                    provider_type.capitalize(),
                    provider_info["provider_name"],
                    "\n".join(config_lines),
                )

            console.print(table)

    except Exception as e:
        rprint(f"[red]Error loading configuration:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file to check",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format for results",
)
def check_connections(config_file: str | None, output_format: str):
    """Check connections to all configured services."""
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_file=config_file, include_env=True)

        # Import the centralized health checker
        from ..utils.health_checks import ServiceHealthChecker

        # Perform health checks
        results = ServiceHealthChecker.perform_all_health_checks(config)

        if output_format == "json":
            # Output JSON format
            console.print_json(data=results)
        else:
            # Output table format
            _display_health_check_results(results)

    except Exception as e:
        rprint(f"[red]Error checking connections:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="schema",
    help="Output directory for schema files",
)
@click.option(
    "--format",
    "-f",
    multiple=True,
    type=click.Choice(["json", "typescript", "markdown"]),
    help="Schema formats to generate",
)
def generate_schema(output_dir: str, format: tuple[str]):
    """Generate configuration schema in various formats."""
    formats = list(format) if format else None

    try:
        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats)

        rprint("[green]✓[/green] Generated configuration schema files:")
        for fmt, path in saved_files.items():
            rprint(f"  - {fmt}: {path}")

    except Exception as e:
        rprint(f"[red]Error generating schema:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format for schema",
)
def show_schema(output_format: str):
    """Display configuration schema in the terminal."""
    try:
        schema = ConfigSchemaGenerator.generate_json_schema()

        if output_format == "yaml":
            import yaml

            yaml_output = yaml.dump(schema, default_flow_style=False, indent=2)
            rprint(yaml_output)
        else:
            # Pretty print the schema as JSON
            console.print_json(data=schema)

    except Exception as e:
        rprint(f"[red]Error displaying schema:[/red] {e}")
        raise click.Exit(1) from e


if __name__ == "__main__":
    cli()
