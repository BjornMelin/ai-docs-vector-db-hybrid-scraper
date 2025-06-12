"""Enhanced configuration management commands with Rich styling.

This module provides advanced configuration management with Rich progress
indicators, beautiful table displays, and interactive features.
"""

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from src.config.backup_restore import ConfigBackupManager
from src.config.loader import ConfigLoader
from src.config.migrations import ConfigMigrationManager
from src.config.models import UnifiedConfig
from src.config.templates import ConfigurationTemplates
from src.config.wizard import ConfigurationWizard
from src.utils.health_checks import ServiceHealthChecker

console = Console()


@click.group()
def config():
    """⚙️ Configuration management with enhanced features.

    Manage your AI Documentation Scraper configuration with Rich visual
    feedback, validation, and health checking capabilities.
    """
    pass


@config.command("create-example")
@click.option(
    "--format",
    "config_format",
    type=click.Choice(["json", "yaml", "toml"]),
    default="json",
    help="Configuration file format",
)
@click.option(
    "--template",
    type=click.Choice(["minimal", "development", "production", "personal-use"]),
    default="development",
    help="Configuration template to use",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file path"
)
@click.pass_context
def create_example(
    ctx: click.Context, config_format: str, template: str, output: Path | None
):
    """Generate example configuration files with templates.

    \b
    Available templates:
    • minimal: Basic configuration with defaults
    • development: Full development setup
    • production: Production-ready configuration
    • personal-use: Optimized for personal projects
    """
    from src.config.cli import create_example as create_example_config

    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Generating {template} configuration...", total=None)

        try:
            # Use existing implementation from config CLI
            output_path = output or Path(f"config.{config_format}")
            create_example_config(config_format, str(output_path))

            success_text = Text()
            success_text.append(
                "✅ Configuration created successfully!\n", style="bold green"
            )
            success_text.append(f"Format: {config_format.upper()}\n", style="cyan")
            success_text.append(f"Template: {template}\n", style="cyan")
            success_text.append(f"Location: {output_path}", style="yellow")

            panel = Panel(
                success_text,
                title="Configuration Created",
                title_align="left",
                border_style="green",
            )
            rich_cli.console.print(panel)

        except Exception as e:
            rich_cli.show_error("Failed to create configuration", str(e))
            raise click.Abort() from e


@config.command("validate")
@click.argument(
    "config_file", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--health-check", is_flag=True, help="Run health checks on configuration")
@click.pass_context
def validate_config(ctx: click.Context, config_file: Path | None, health_check: bool):
    """Validate configuration with enhanced visual feedback.

    Validates configuration syntax, required fields, and optionally
    runs health checks to verify service connectivity.
    """
    rich_cli = ctx.obj["rich_cli"]

    # Load configuration
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading configuration...", total=None)

        try:
            if config_file:
                config = ConfigLoader.from_file(config_file)
                config_source = str(config_file)
            else:
                config = ConfigLoader.load_config()
                config_source = "Environment variables and defaults"

            progress.update(task, description="Validating configuration...")

            # Basic validation is done during loading

            if health_check:
                progress.update(task, description="Running health checks...")
                health_results = ServiceHealthChecker.perform_all_health_checks(config)

        except Exception as e:
            rich_cli.show_error("Configuration validation failed", str(e))
            raise click.Abort() from e

    # Display validation results
    validation_text = Text()
    validation_text.append("✅ Configuration is valid!\n", style="bold green")
    validation_text.append(f"Source: {config_source}\n", style="dim")

    # Configuration summary table
    summary_table = Table(title="Configuration Summary", show_header=True)
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Details", style="dim")

    # Add configuration details
    summary_table.add_row(
        "Qdrant", "Configured", f"{config.qdrant.host}:{config.qdrant.port}"
    )

    if config.openai.api_key:
        summary_table.add_row("OpenAI", "Configured", "API key provided")

    if config.fastembed.enabled:
        summary_table.add_row("FastEmbed", "Enabled", config.fastembed.model)

    if config.cache.redis.enabled:
        summary_table.add_row(
            "Redis Cache",
            "Enabled",
            f"{config.cache.redis.host}:{config.cache.redis.port}",
        )

    rich_cli.console.print(validation_text)
    rich_cli.console.print(summary_table)

    # Health check results
    if health_check:
        console.print("\n[bold cyan]Health Check Results:[/bold cyan]")

        health_table = Table(show_header=True)
        health_table.add_column("Service", style="cyan")
        health_table.add_column("Status", style="")
        health_table.add_column("Details", style="dim")

        for service, result in health_results.items():
            status_style = "green" if result["connected"] else "red"
            status_text = "✅ Healthy" if result["connected"] else "❌ Error"

            health_table.add_row(
                service.title(),
                Text(status_text, style=status_style),
                result.get("error", "") or "Connected",
            )

        rich_cli.console.print(health_table)


@config.command("show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.option(
    "--section",
    type=click.Choice(["qdrant", "openai", "fastembed", "cache", "browser"]),
    help="Show specific configuration section",
)
@click.pass_context
def show_config(ctx: click.Context, output_format: str, section: str | None):
    """Display current configuration with Rich formatting."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    if output_format == "table":
        _show_config_table(config, section, rich_cli)
    elif output_format == "json":
        _show_config_json(config, section, rich_cli)
    elif output_format == "yaml":
        _show_config_yaml(config, section, rich_cli)


def _show_config_table(config: UnifiedConfig, section: str | None, rich_cli):
    """Display configuration as Rich table."""
    if section:
        # Show specific section
        table = Table(title=f"{section.title()} Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="")

        section_config = getattr(config, section)
        for field_name, field_value in section_config.model_dump().items():
            # Mask sensitive values
            if "key" in field_name.lower() or "password" in field_name.lower():
                display_value = "***" if field_value else "Not set"
            else:
                display_value = str(field_value)

            table.add_row(field_name, display_value)

        rich_cli.console.print(table)
    else:
        # Show overview of all sections
        overview_table = Table(title="Configuration Overview")
        overview_table.add_column("Section", style="cyan")
        overview_table.add_column("Status", style="")
        overview_table.add_column("Key Settings", style="dim")

        # Qdrant section
        overview_table.add_row(
            "Qdrant",
            "✅ Configured",
            f"Host: {config.qdrant.host}:{config.qdrant.port}",
        )

        # OpenAI section
        openai_status = "✅ Configured" if config.openai.api_key else "❌ No API key"
        overview_table.add_row("OpenAI", openai_status, f"Model: {config.openai.model}")

        # FastEmbed section
        fastembed_status = "✅ Enabled" if config.fastembed.enabled else "❌ Disabled"
        overview_table.add_row(
            "FastEmbed", fastembed_status, f"Model: {config.fastembed.model}"
        )

        # Cache section
        cache_status = "✅ Enabled" if config.cache.redis.enabled else "❌ Disabled"
        overview_table.add_row(
            "Cache",
            cache_status,
            f"Redis: {config.cache.redis.host}:{config.cache.redis.port}",
        )

        rich_cli.console.print(overview_table)


def _show_config_json(config: UnifiedConfig, section: str | None, rich_cli):
    """Display configuration as JSON with syntax highlighting."""
    import json

    if section:
        config_data = getattr(config, section).model_dump()
    else:
        config_data = config.model_dump()

    # Mask sensitive data
    config_data = _mask_sensitive_data(config_data)

    json_str = json.dumps(config_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)

    panel = Panel(
        syntax, title="Configuration (JSON)", title_align="left", border_style="blue"
    )
    rich_cli.console.print(panel)


def _show_config_yaml(config: UnifiedConfig, section: str | None, rich_cli):
    """Display configuration as YAML with syntax highlighting."""
    import yaml

    if section:
        config_data = getattr(config, section).model_dump()
    else:
        config_data = config.model_dump()

    # Mask sensitive data
    config_data = _mask_sensitive_data(config_data)

    yaml_str = yaml.dump(config_data, default_flow_style=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)

    panel = Panel(
        syntax, title="Configuration (YAML)", title_align="left", border_style="blue"
    )
    rich_cli.console.print(panel)


def _mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively mask sensitive configuration data."""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if (
                "key" in key.lower()
                or "password" in key.lower()
                or "secret" in key.lower()
            ):
                masked[key] = "***" if value else None
            elif isinstance(value, dict):
                masked[key] = _mask_sensitive_data(value)
            else:
                masked[key] = value
        return masked
    return data


@config.command("convert")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "toml"]),
    help="Output format (auto-detected from extension if not provided)",
)
@click.pass_context
def convert_config(
    ctx: click.Context, input_file: Path, output_file: Path, output_format: str | None
):
    """Convert configuration between formats (JSON ↔ YAML ↔ TOML)."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Converting configuration...", total=None)

        try:
            from src.config.cli import convert as convert_config_format

            # Auto-detect output format if not provided
            if not output_format:
                if output_file.suffix == ".json":
                    output_format = "json"
                elif output_file.suffix in [".yaml", ".yml"]:
                    output_format = "yaml"
                elif output_file.suffix == ".toml":
                    output_format = "toml"
                else:
                    output_format = "json"

            convert_config_format(
                str(input_file), str(output_file), None, output_format
            )

            success_text = Text()
            success_text.append(
                "✅ Configuration converted successfully!\n", style="bold green"
            )
            success_text.append(f"Input: {input_file}\n", style="dim")
            success_text.append(f"Output: {output_file}\n", style="dim")
            success_text.append(f"Format: {output_format}", style="cyan")

            panel = Panel(
                success_text,
                title="Conversion Complete",
                title_align="left",
                border_style="green",
            )
            rich_cli.console.print(panel)

        except Exception as e:
            rich_cli.show_error("Configuration conversion failed", str(e))
            raise click.Abort() from e


# Template Management Commands
@config.group("template")
def template():
    """🎨 Configuration template management commands.

    Manage configuration templates for different deployment scenarios
    including development, production, high-performance, and distributed setups.
    """
    pass


@template.command("list")
@click.pass_context
def list_templates(ctx: click.Context):
    """List available configuration templates."""
    rich_cli = ctx.obj["rich_cli"]

    try:
        templates = ConfigurationTemplates()
        available_templates = templates.list_available_templates()

        if not available_templates:
            rich_cli.console.print("[yellow]No templates available[/yellow]")
            return

        # Create templates table
        table = Table(title="Available Configuration Templates", show_header=True)
        table.add_column("Template", style="cyan")
        table.add_column("Description", style="")
        table.add_column("Use Case", style="dim")

        # Get template descriptions
        template_descriptions = {
            "development": (
                "Development with debugging",
                "Local development and testing",
            ),
            "production": (
                "Production with security hardening",
                "Production deployment",
            ),
            "high_performance": (
                "Maximum throughput optimization",
                "High-traffic applications",
            ),
            "memory_optimized": (
                "Resource-constrained environments",
                "Memory-limited deployments",
            ),
            "distributed": (
                "Multi-node cluster deployment",
                "Large-scale distributed systems",
            ),
        }

        for template_name in available_templates:
            desc, use_case = template_descriptions.get(
                template_name, ("Custom template", "Custom use case")
            )
            table.add_row(template_name, desc, use_case)

        rich_cli.console.print(table)

    except Exception as e:
        rich_cli.show_error("Failed to list templates", str(e))
        raise click.Abort() from e


@template.command("apply")
@click.argument("template_name")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output configuration file path",
)
@click.option("--environment-override", help="Environment-specific overrides to apply")
@click.pass_context
def apply_template(
    ctx: click.Context,
    template_name: str,
    output: Path | None,
    environment_override: str | None,
):
    """Apply a template to create a new configuration."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Applying template '{template_name}'...", total=None)

        try:
            templates = ConfigurationTemplates()
            config_data = templates.apply_template_to_config(
                template_name, environment_overrides=environment_override
            )

            if not config_data:
                rich_cli.show_error(
                    "Template not found", f"Template '{template_name}' does not exist"
                )
                raise click.Abort()

            # Determine output path
            if not output:
                output = Path(f"{template_name}_config.json")

            # Create configuration and save
            config = UnifiedConfig(**config_data)
            config.save_to_file(output)

            success_text = Text()
            success_text.append(
                "✅ Template applied successfully!\n", style="bold green"
            )
            success_text.append(f"Template: {template_name}\n", style="cyan")
            success_text.append(f"Output: {output}\n", style="yellow")
            if environment_override:
                success_text.append(
                    f"Environment Override: {environment_override}", style="dim"
                )

            panel = Panel(
                success_text,
                title="Template Applied",
                title_align="left",
                border_style="green",
            )
            rich_cli.console.print(panel)

        except Exception as e:
            rich_cli.show_error("Failed to apply template", str(e))
            raise click.Abort() from e


# Configuration Wizard Commands
@config.command("wizard")
@click.option(
    "--config-path",
    type=click.Path(path_type=Path),
    help="Target path for configuration file",
)
@click.pass_context
def wizard(ctx: click.Context, config_path: Path | None):
    """🧙‍♂️ Interactive configuration setup wizard.

    Launch an interactive wizard to guide you through configuration setup,
    template selection, and environment-specific customization.
    """
    rich_cli = ctx.obj["rich_cli"]

    try:
        wizard = ConfigurationWizard()
        created_config_path = wizard.run_setup_wizard(config_path)

        success_text = Text()
        success_text.append("✅ Configuration wizard completed!\n", style="bold green")
        success_text.append(
            f"Configuration created: {created_config_path}", style="yellow"
        )

        panel = Panel(
            success_text,
            title="Wizard Complete",
            title_align="left",
            border_style="green",
        )
        rich_cli.console.print(panel)

    except Exception as e:
        rich_cli.show_error("Configuration wizard failed", str(e))
        raise click.Abort() from e


# Backup Management Commands
@config.group("backup")
def backup():
    """💾 Configuration backup and restore commands.

    Create, manage, and restore configuration backups with versioning
    and metadata tracking for safe configuration management.
    """
    pass


@backup.command("create")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--description", help="Backup description")
@click.option("--tags", help="Comma-separated tags for the backup")
@click.option("--compress/--no-compress", default=True, help="Compress the backup")
@click.pass_context
def create_backup(
    ctx: click.Context,
    config_file: Path,
    description: str | None,
    tags: str | None,
    compress: bool,
):
    """Create a backup of a configuration file."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Creating backup...", total=None)

        try:
            backup_manager = ConfigBackupManager()

            # Parse tags
            tag_list = (
                [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
            )

            backup_id = backup_manager.create_backup(
                config_file, description=description, tags=tag_list, compress=compress
            )

            success_text = Text()
            success_text.append("✅ Backup created successfully!\n", style="bold green")
            success_text.append(f"Backup ID: {backup_id}\n", style="cyan")
            success_text.append(f"Source: {config_file}\n", style="dim")
            success_text.append(
                f"Compressed: {'Yes' if compress else 'No'}", style="dim"
            )

            panel = Panel(
                success_text,
                title="Backup Created",
                title_align="left",
                border_style="green",
            )
            rich_cli.console.print(panel)

        except Exception as e:
            rich_cli.show_error("Failed to create backup", str(e))
            raise click.Abort() from e


@backup.command("list")
@click.option("--config-name", help="Filter by configuration name")
@click.option("--environment", help="Filter by environment")
@click.option("--tags", help="Filter by tags (comma-separated)")
@click.option("--limit", type=int, default=20, help="Limit number of results")
@click.pass_context
def list_backups(
    ctx: click.Context,
    config_name: str | None,
    environment: str | None,
    tags: str | None,
    limit: int,
):
    """List available configuration backups."""
    rich_cli = ctx.obj["rich_cli"]

    try:
        backup_manager = ConfigBackupManager()

        # Parse tags filter
        tag_filter = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
        )

        backups = backup_manager.list_backups(
            config_name=config_name,
            environment=environment,
            tags=tag_filter,
            limit=limit,
        )

        if not backups:
            rich_cli.console.print("[yellow]No backups found[/yellow]")
            return

        # Create backups table
        table = Table(title="Configuration Backups", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Config", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Environment", style="blue")
        table.add_column("Size", style="magenta")
        table.add_column("Description", style="dim")

        for backup in backups:
            # Format file size
            size_mb = backup.file_size / (1024 * 1024)
            size_str = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{backup.file_size}B"

            table.add_row(
                backup.backup_id[:12] + "...",
                backup.config_name,
                backup.created_at[:19],  # Remove milliseconds
                backup.environment or "unknown",
                size_str,
                backup.description or "No description",
            )

        rich_cli.console.print(table)

    except Exception as e:
        rich_cli.show_error("Failed to list backups", str(e))
        raise click.Abort() from e


@backup.command("restore")
@click.argument("backup_id")
@click.option(
    "--target",
    type=click.Path(path_type=Path),
    help="Target path for restored configuration",
)
@click.option("--force", is_flag=True, help="Force restore despite conflicts")
@click.option(
    "--no-pre-backup", is_flag=True, help="Skip creating backup before restore"
)
@click.pass_context
def restore_backup(
    ctx: click.Context,
    backup_id: str,
    target: Path | None,
    force: bool,
    no_pre_backup: bool,
):
    """Restore a configuration from backup."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Restoring backup {backup_id[:12]}...", total=None)

        try:
            backup_manager = ConfigBackupManager()

            result = backup_manager.restore_backup(
                backup_id,
                target_path=target,
                create_pre_restore_backup=not no_pre_backup,
                force=force,
            )

            if result.success:
                success_text = Text()
                success_text.append(
                    "✅ Configuration restored successfully!\n", style="bold green"
                )
                success_text.append(f"Backup ID: {backup_id}\n", style="cyan")
                success_text.append(
                    f"Restored to: {result.config_path}\n", style="yellow"
                )
                if result.pre_restore_backup:
                    success_text.append(
                        f"Pre-restore backup: {result.pre_restore_backup}", style="dim"
                    )

                panel = Panel(
                    success_text,
                    title="Restore Complete",
                    title_align="left",
                    border_style="green",
                )
                rich_cli.console.print(panel)

                if result.conflicts:
                    rich_cli.console.print(
                        "\n[yellow]Conflicts resolved during restore:[/yellow]"
                    )
                    for conflict in result.conflicts:
                        rich_cli.console.print(f"  • {conflict}")

            else:
                error_text = Text()
                error_text.append("❌ Restore failed\n", style="bold red")
                if result.conflicts:
                    error_text.append("Conflicts detected:\n", style="yellow")
                    for conflict in result.conflicts:
                        error_text.append(f"  • {conflict}\n", style="")
                    error_text.append(
                        "\nUse --force to override conflicts", style="dim"
                    )

                for warning in result.warnings:
                    error_text.append(f"⚠️  {warning}\n", style="yellow")

                panel = Panel(
                    error_text,
                    title="Restore Failed",
                    title_align="left",
                    border_style="red",
                )
                rich_cli.console.print(panel)
                raise click.Abort()

        except Exception as e:
            rich_cli.show_error("Failed to restore backup", str(e))
            raise click.Abort() from e


# Migration Management Commands
@config.group("migrate")
def migrate():
    """🔄 Configuration migration and versioning commands.

    Manage configuration schema migrations, version upgrades,
    and rollback operations for safe configuration evolution.
    """
    pass


@migrate.command("plan")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.argument("target_version")
@click.pass_context
def migration_plan(ctx: click.Context, config_file: Path, target_version: str):
    """Create a migration plan to upgrade configuration to target version."""
    rich_cli = ctx.obj["rich_cli"]

    try:
        migration_manager = ConfigMigrationManager()
        current_version = migration_manager.get_current_version(config_file)

        plan = migration_manager.create_migration_plan(current_version, target_version)

        if not plan:
            rich_cli.show_error(
                "No migration path found",
                f"Cannot migrate from {current_version} to {target_version}",
            )
            raise click.Abort()

        # Display migration plan
        plan_text = Text()
        plan_text.append("📋 Migration Plan\n\n", style="bold cyan")
        plan_text.append(
            f"From: {plan.source_version} → To: {plan.target_version}\n", style="yellow"
        )
        plan_text.append(
            f"Estimated Duration: {plan.estimated_duration}\n", style="dim"
        )

        if plan.requires_downtime:
            plan_text.append("⚠️  Requires Downtime: Yes\n", style="bold red")
        else:
            plan_text.append("✅ No Downtime Required\n", style="green")

        plan_text.append("\nMigration Steps:\n", style="bold")
        for i, migration_id in enumerate(plan.migrations, 1):
            plan_text.append(f"  {i}. {migration_id}\n", style="")

        if plan.rollback_plan:
            plan_text.append("\nRollback Plan Available:\n", style="bold yellow")
            for i, rollback_id in enumerate(plan.rollback_plan, 1):
                plan_text.append(f"  {i}. {rollback_id}\n", style="dim")

        panel = Panel(
            plan_text,
            title="Migration Plan",
            title_align="left",
            border_style="cyan",
        )
        rich_cli.console.print(panel)

    except Exception as e:
        rich_cli.show_error("Failed to create migration plan", str(e))
        raise click.Abort() from e


@migrate.command("apply")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.argument("target_version")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying"
)
@click.option("--force", is_flag=True, help="Force migration despite warnings")
@click.pass_context
def apply_migration(
    ctx: click.Context,
    config_file: Path,
    target_version: str,
    dry_run: bool,
    force: bool,
):
    """Apply migrations to upgrade configuration to target version."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Planning migration...", total=None)

        try:
            migration_manager = ConfigMigrationManager()
            current_version = migration_manager.get_current_version(config_file)

            # Create migration plan
            plan = migration_manager.create_migration_plan(
                current_version, target_version
            )

            if not plan:
                rich_cli.show_error(
                    "No migration path found",
                    f"Cannot migrate from {current_version} to {target_version}",
                )
                raise click.Abort()

            progress.update(task, description="Applying migrations...")

            # Apply migration plan
            results = migration_manager.apply_migration_plan(
                plan, config_file, dry_run=dry_run, force=force
            )

            # Display results
            all_successful = all(r.success for r in results)

            if all_successful:
                success_text = Text()
                success_text.append(
                    "✅ Migration completed successfully!\n", style="bold green"
                )
                success_text.append(
                    f"Version: {current_version} → {target_version}\n", style="cyan"
                )
                success_text.append(f"Applied {len(results)} migrations\n", style="dim")

                if dry_run:
                    success_text.append("(Dry run - no changes made)", style="yellow")

                panel = Panel(
                    success_text,
                    title="Migration Complete",
                    title_align="left",
                    border_style="green",
                )
                rich_cli.console.print(panel)

                # Show changes made
                if results:
                    rich_cli.console.print("\n[bold]Changes Made:[/bold]")
                    for result in results:
                        if result.success and result.changes_made:
                            rich_cli.console.print(
                                f"[green]✅ {result.migration_id}[/green]"
                            )
                            for change in result.changes_made:
                                rich_cli.console.print(f"    • {change}")
            else:
                error_text = Text()
                error_text.append("❌ Migration failed\n", style="bold red")

                for result in results:
                    if not result.success:
                        error_text.append(
                            f"Failed: {result.migration_id}\n", style="red"
                        )
                        for error in result.errors:
                            error_text.append(f"  • {error}\n", style="")

                panel = Panel(
                    error_text,
                    title="Migration Failed",
                    title_align="left",
                    border_style="red",
                )
                rich_cli.console.print(panel)
                raise click.Abort()

        except Exception as e:
            rich_cli.show_error("Failed to apply migration", str(e))
            raise click.Abort() from e


@migrate.command("rollback")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.argument("migration_id")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying"
)
@click.pass_context
def rollback_migration(
    ctx: click.Context, config_file: Path, migration_id: str, dry_run: bool
):
    """Rollback a specific migration."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Rolling back {migration_id}...", total=None)

        try:
            migration_manager = ConfigMigrationManager()

            result = migration_manager.rollback_migration(
                migration_id, config_file, dry_run=dry_run
            )

            if result.success:
                success_text = Text()
                success_text.append(
                    "✅ Rollback completed successfully!\n", style="bold green"
                )
                success_text.append(f"Migration: {migration_id}\n", style="cyan")
                success_text.append(
                    f"Version: {result.from_version} → {result.to_version}\n",
                    style="yellow",
                )

                if dry_run:
                    success_text.append("(Dry run - no changes made)", style="yellow")

                panel = Panel(
                    success_text,
                    title="Rollback Complete",
                    title_align="left",
                    border_style="green",
                )
                rich_cli.console.print(panel)

                # Show changes made
                if result.changes_made:
                    rich_cli.console.print("\n[bold]Changes Made:[/bold]")
                    for change in result.changes_made:
                        rich_cli.console.print(f"  • {change}")
            else:
                error_text = Text()
                error_text.append("❌ Rollback failed\n", style="bold red")

                for error in result.errors:
                    error_text.append(f"  • {error}\n", style="")

                panel = Panel(
                    error_text,
                    title="Rollback Failed",
                    title_align="left",
                    border_style="red",
                )
                rich_cli.console.print(panel)
                raise click.Abort()

        except Exception as e:
            rich_cli.show_error("Failed to rollback migration", str(e))
            raise click.Abort() from e


@migrate.command("status")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def migration_status(ctx: click.Context, config_file: Path):
    """Show current migration status and available migrations."""
    rich_cli = ctx.obj["rich_cli"]

    try:
        migration_manager = ConfigMigrationManager()
        current_version = migration_manager.get_current_version(config_file)
        available_migrations = migration_manager.list_available_migrations()
        applied_migrations = migration_manager.list_applied_migrations()

        # Status info
        status_text = Text()
        status_text.append("📊 Migration Status\n\n", style="bold cyan")
        status_text.append(f"Current Version: {current_version}\n", style="yellow")
        status_text.append(
            f"Available Migrations: {len(available_migrations)}\n", style="green"
        )
        status_text.append(
            f"Applied Migrations: {len(applied_migrations)}", style="blue"
        )

        panel = Panel(
            status_text,
            title="Migration Status",
            title_align="left",
            border_style="cyan",
        )
        rich_cli.console.print(panel)

        # Available migrations table
        if available_migrations:
            table = Table(title="Available Migrations", show_header=True)
            table.add_column("Migration ID", style="cyan")
            table.add_column("From", style="yellow")
            table.add_column("To", style="green")
            table.add_column("Description", style="")
            table.add_column("Applied", style="blue")

            for migration in available_migrations:
                applied = "✅" if migration.migration_id in applied_migrations else "❌"
                table.add_row(
                    migration.migration_id,
                    migration.from_version,
                    migration.to_version,
                    migration.description,
                    applied,
                )

            rich_cli.console.print(table)

    except Exception as e:
        rich_cli.show_error("Failed to get migration status", str(e))
        raise click.Abort() from e
