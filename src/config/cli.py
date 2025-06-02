"""CLI tool for managing unified configuration.

This tool provides commands for creating, validating, and converting
configuration files.
"""

import json
from pathlib import Path
from typing import Any

import click
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .loader import ConfigLoader
from .migrator import ConfigMigrator
from .models import UnifiedConfig
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
    "--source",
    type=click.Path(exists=True),
    default="config/documentation-sites.json",
    help="Source documentation sites file",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Target configuration file",
)
def migrate_sites(source: str, config_file: str | None):
    """Migrate documentation sites from old format to unified config."""
    try:
        # Load documentation sites
        sites = ConfigLoader.load_documentation_sites(source)

        # Load or create configuration using standard loader
        if config_file:
            config = ConfigLoader.load_config(config_file=config_file, include_env=True)
        else:
            config = UnifiedConfig()

        # Update sites
        config.documentation_sites = sites

        # Save
        output_path = Path(config_file) if config_file else Path("config.json")
        config.save_to_file(output_path, format="json")

        rprint(
            f"[green]✓[/green] Migrated {len(sites)} documentation sites to {output_path}"
        )

    except Exception as e:
        rprint(f"[red]Error migrating sites:[/red] {e}")
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


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--target-version", "-v", default="0.3.0", help="Target version to migrate to"
)
@click.option("--no-backup", is_flag=True, help="Skip creating backup file")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be migrated without making changes"
)
def migrate(config_file: str, target_version: str, no_backup: bool, dry_run: bool):
    """Migrate a configuration file to the latest version."""
    config_path = Path(config_file)

    try:
        # Load configuration
        with open(config_path) as f:
            if config_path.suffix == ".json":
                config_data = json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            else:
                rprint(f"[red]Unsupported file format: {config_path.suffix}[/red]")
                raise click.Exit(1)

        # Detect current version
        from_version = ConfigMigrator.detect_config_version(config_data)
        if from_version is None:
            rprint("[red]Could not detect configuration version[/red]")
            raise click.Exit(1)

        rprint(f"Current version: [cyan]{from_version}[/cyan]")
        rprint(f"Target version: [cyan]{target_version}[/cyan]")

        if from_version == target_version:
            rprint("[green]Configuration already at target version[/green]")
            return

        # Perform migration (in memory for dry run)
        if from_version == "legacy":
            migrated = ConfigMigrator.migrate_legacy_to_unified(config_data)
        else:
            migrated = ConfigMigrator.migrate_between_versions(
                config_data, from_version, target_version
            )

        # Generate report
        report = ConfigMigrator.create_migration_report(
            config_data, migrated, from_version, target_version
        )

        rprint("\n[bold]Migration Report:[/bold]")
        rprint(report)

        if dry_run:
            rprint("\n[yellow]Dry run complete - no changes made[/yellow]")
        else:
            # Apply migration
            success, message = ConfigMigrator.auto_migrate(
                config_path, target_version, backup=not no_backup
            )
            if success:
                rprint(f"\n[green]✓[/green] {message}")
            else:
                rprint(f"\n[red]✗[/red] {message}")
                raise click.Exit(1)

    except Exception as e:
        rprint(f"[red]Error during migration:[/red] {e}")
        raise click.Exit(1) from e


@cli.command()
@click.option("--from-version", "-f", help="Source version")
@click.option("--to-version", "-t", help="Target version")
def show_migration_path(from_version: str | None, to_version: str | None):
    """Show available migration paths between versions."""
    table = Table(title="Configuration Version History")
    table.add_column("Version", style="cyan")
    table.add_column("Description", style="yellow")

    for version, description in ConfigMigrator.VERSIONS.items():
        table.add_row(version, description)

    console.print(table)

    if from_version and to_version:
        rprint(f"\nMigration path from {from_version} to {to_version}:")
        if from_version == "legacy":
            rprint("  1. Convert legacy format to unified configuration")
            rprint(f"  2. Apply migrations to reach version {to_version}")
        else:
            rprint(
                f"  1. Apply incremental migrations from {from_version} to {to_version}"
            )


if __name__ == "__main__":
    cli()
