"""Enhanced configuration management commands with Rich styling.

This module provides advanced configuration management with Rich progress
indicators, beautiful table displays, and interactive features.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from src.config.loader import ConfigLoader
from src.config.models import UnifiedConfig
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
    help="Configuration file format"
)
@click.option(
    "--template",
    type=click.Choice(["minimal", "development", "production", "personal-use"]),
    default="development",
    help="Configuration template to use"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path"
)
@click.pass_context
def create_example(
    ctx: click.Context,
    config_format: str,
    template: str,
    output: Optional[Path]
):
    """Generate example configuration files with templates.
    
    \b
    Available templates:
    • minimal: Basic configuration with defaults
    • development: Full development setup
    • production: Production-ready configuration
    • personal-use: Optimized for personal projects
    """
    from src.config.cli import create_example_config
    
    rich_cli = ctx.obj["rich_cli"]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Generating {template} configuration...", total=None)
        
        try:
            # Use existing implementation from config CLI
            result = create_example_config(config_format, template, output)
            
            success_text = Text()
            success_text.append("✅ Configuration created successfully!\n", style="bold green")
            success_text.append(f"Format: {config_format.upper()}\n", style="cyan")
            success_text.append(f"Template: {template}\n", style="cyan") 
            success_text.append(f"Location: {result['path']}", style="yellow")
            
            panel = Panel(
                success_text,
                title="Configuration Created",
                title_align="left",
                border_style="green"
            )
            rich_cli.console.print(panel)
            
        except Exception as e:
            rich_cli.show_error("Failed to create configuration", str(e))
            raise click.Abort()


@config.command("validate")
@click.argument(
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=False
)
@click.option(
    "--health-check",
    is_flag=True,
    help="Run health checks on configuration"
)
@click.pass_context
def validate_config(
    ctx: click.Context,
    config_file: Optional[Path],
    health_check: bool
):
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
            raise click.Abort()
    
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
        "Qdrant",
        "Configured", 
        f"{config.qdrant.host}:{config.qdrant.port}"
    )
    
    if config.openai.api_key:
        summary_table.add_row("OpenAI", "Configured", "API key provided")
    
    if config.fastembed.enabled:
        summary_table.add_row("FastEmbed", "Enabled", config.fastembed.model)
    
    if config.cache.redis.enabled:
        summary_table.add_row(
            "Redis Cache",
            "Enabled",
            f"{config.cache.redis.host}:{config.cache.redis.port}"
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
                result.get("error", "") or "Connected"
            )
        
        rich_cli.console.print(health_table)


@config.command("show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
@click.option(
    "--section",
    type=click.Choice(["qdrant", "openai", "fastembed", "cache", "browser"]),
    help="Show specific configuration section"
)
@click.pass_context
def show_config(
    ctx: click.Context,
    output_format: str,
    section: Optional[str]
):
    """Display current configuration with Rich formatting."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]
    
    if output_format == "table":
        _show_config_table(config, section, rich_cli)
    elif output_format == "json":
        _show_config_json(config, section, rich_cli)
    elif output_format == "yaml":
        _show_config_yaml(config, section, rich_cli)


def _show_config_table(config: UnifiedConfig, section: Optional[str], rich_cli):
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
            f"Host: {config.qdrant.host}:{config.qdrant.port}"
        )
        
        # OpenAI section
        openai_status = "✅ Configured" if config.openai.api_key else "❌ No API key"
        overview_table.add_row(
            "OpenAI",
            openai_status,
            f"Model: {config.openai.model}"
        )
        
        # FastEmbed section
        fastembed_status = "✅ Enabled" if config.fastembed.enabled else "❌ Disabled"
        overview_table.add_row(
            "FastEmbed",
            fastembed_status,
            f"Model: {config.fastembed.model}"
        )
        
        # Cache section
        cache_status = "✅ Enabled" if config.cache.redis.enabled else "❌ Disabled"
        overview_table.add_row(
            "Cache",
            cache_status,
            f"Redis: {config.cache.redis.host}:{config.cache.redis.port}"
        )
        
        rich_cli.console.print(overview_table)


def _show_config_json(config: UnifiedConfig, section: Optional[str], rich_cli):
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
        syntax,
        title="Configuration (JSON)",
        title_align="left",
        border_style="blue"
    )
    rich_cli.console.print(panel)


def _show_config_yaml(config: UnifiedConfig, section: Optional[str], rich_cli):
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
        syntax,
        title="Configuration (YAML)",
        title_align="left",
        border_style="blue"
    )
    rich_cli.console.print(panel)


def _mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively mask sensitive configuration data."""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if "key" in key.lower() or "password" in key.lower() or "secret" in key.lower():
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
    help="Output format (auto-detected from extension if not provided)"
)
@click.pass_context
def convert_config(
    ctx: click.Context,
    input_file: Path,
    output_file: Path,
    output_format: Optional[str]
):
    """Convert configuration between formats (JSON ↔ YAML ↔ TOML)."""
    rich_cli = ctx.obj["rich_cli"]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Converting configuration...", total=None)
        
        try:
            from src.config.cli import convert_config_format
            
            result = convert_config_format(input_file, output_file, output_format)
            
            success_text = Text()
            success_text.append("✅ Configuration converted successfully!\n", style="bold green")
            success_text.append(f"Input: {input_file}\n", style="dim")
            success_text.append(f"Output: {output_file}\n", style="dim")
            success_text.append(f"Format: {result['format']}", style="cyan")
            
            panel = Panel(
                success_text,
                title="Conversion Complete",
                title_align="left",
                border_style="green"
            )
            rich_cli.console.print(panel)
            
        except Exception as e:
            rich_cli.show_error("Configuration conversion failed", str(e))
            raise click.Abort()