"""CLI tool for managing unified configuration.

This tool provides commands for creating, validating, and converting
configuration files.
"""

import json
from pathlib import Path

import click
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .models import UnifiedConfig, get_config, reset_config, set_config
from .loader import ConfigLoader
from .migrator import ConfigMigrator
from .schema import ConfigSchemaGenerator

console = Console()


@click.group()
def cli():
    """AI Documentation Vector DB configuration management tool."""
    pass


@cli.command()
@click.option("--format", type=click.Choice(["json", "yaml", "toml"]), default="json", help="Output format")
@click.option("--output", "-o", type=click.Path(), default="config.example.json", help="Output file path")
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
@click.option("--output", "-o", type=click.Path(), default=".env.example", help="Output file path")
def create_env_template(output: str):
    """Create a .env template file."""
    ConfigLoader.create_env_template(output)
    rprint(f"[green]✓[/green] Created .env template at: {output}")


@cli.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file to validate")
@click.option("--env-file", "-e", type=click.Path(exists=True), help=".env file path")
@click.option("--show-config", is_flag=True, help="Show the loaded configuration")
def validate(config_file: str | None, env_file: str | None, show_config: bool):
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
            rprint("\n[bold]Loaded Configuration:[/bold]")
            rprint(config.model_dump_json(indent=2))
            
    except Exception as e:
        rprint(f"[red]Error loading configuration:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--from-format", type=click.Choice(["json", "yaml", "toml", "env"]), help="Input format (auto-detected if not specified)")
@click.option("--to-format", type=click.Choice(["json", "yaml", "toml"]), required=True, help="Output format")
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
            rprint("[red]Could not auto-detect input format. Please specify --from-format[/red]")
            raise click.Exit(1)
    
    try:
        # Load configuration
        if from_format == "env":
            # Special handling for .env files
            config = UnifiedConfig()
        else:
            config = UnifiedConfig.load_from_file(input_path)
        
        # Save in new format
        config.save_to_file(output_path, format=to_format)
        rprint(f"[green]✓[/green] Converted {input_file} ({from_format}) → {output_file} ({to_format})")
        
    except Exception as e:
        rprint(f"[red]Error converting configuration:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file to load")
def show_providers(config_file: str | None):
    """Show active provider configuration."""
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_file=config_file, include_env=True)
        
        # Create table
        table = Table(title="Active Provider Configuration")
        table.add_column("Provider Type", style="cyan")
        table.add_column("Selected Provider", style="green")
        table.add_column("Configuration", style="yellow")
        
        # Add rows
        providers = config.get_active_providers()
        
        # Embedding provider
        embedding_config = providers["embedding"]
        if config.embedding_provider == "openai":
            table.add_row(
                "Embedding",
                "OpenAI",
                f"Model: {embedding_config.model}\n"
                f"Dimensions: {embedding_config.dimensions}\n"
                f"API Key: {'Set' if embedding_config.api_key else 'Not Set'}"
            )
        else:
            table.add_row(
                "Embedding",
                "FastEmbed",
                f"Model: {embedding_config.model}\n"
                f"Max Length: {embedding_config.max_length}"
            )
        
        # Crawl provider
        crawl_config = providers["crawl"]
        if config.crawl_provider == "firecrawl":
            table.add_row(
                "Crawl",
                "Firecrawl",
                f"API URL: {crawl_config.api_url}\n"
                f"API Key: {'Set' if crawl_config.api_key else 'Not Set'}"
            )
        else:
            table.add_row(
                "Crawl",
                "Crawl4AI",
                f"Browser: {crawl_config.browser_type}\n"
                f"Headless: {crawl_config.headless}\n"
                f"Max Concurrent: {crawl_config.max_concurrent_crawls}"
            )
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]Error loading configuration:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.option("--source", type=click.Path(exists=True), default="config/documentation-sites.json", help="Source documentation sites file")
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Target configuration file")
def migrate_sites(source: str, config_file: str | None):
    """Migrate documentation sites from old format to unified config."""
    try:
        # Load documentation sites
        sites = ConfigLoader.load_documentation_sites(source)
        
        # Load or create configuration
        if config_file:
            config = UnifiedConfig.load_from_file(config_file)
        else:
            config = UnifiedConfig()
        
        # Update sites
        config.documentation_sites = sites
        
        # Save
        output_path = Path(config_file) if config_file else Path("config.json")
        config.save_to_file(output_path, format="json")
        
        rprint(f"[green]✓[/green] Migrated {len(sites)} documentation sites to {output_path}")
        
    except Exception as e:
        rprint(f"[red]Error migrating sites:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file to check")
def check_connections(config_file: str | None):
    """Check connections to all configured services."""
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_file=config_file, include_env=True)
        
        rprint("[bold]Checking Service Connections...[/bold]\n")
        
        # Check Qdrant
        rprint("🔍 Checking Qdrant...")
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
            collections = client.get_collections()
            rprint(f"[green]✓[/green] Qdrant connected ({len(collections.collections)} collections)")
        except Exception as e:
            rprint(f"[red]✗[/red] Qdrant connection failed: {e}")
        
        # Check Redis if enabled
        if config.cache.enable_redis_cache:
            rprint("\n🔍 Checking Redis...")
            try:
                import redis
                r = redis.from_url(config.cache.redis_url)
                r.ping()
                rprint(f"[green]✓[/green] Redis connected")
            except Exception as e:
                rprint(f"[red]✗[/red] Redis connection failed: {e}")
        
        # Check OpenAI if configured
        if config.embedding_provider == "openai" and config.openai.api_key:
            rprint("\n🔍 Checking OpenAI...")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.openai.api_key)
                models = client.models.list()
                rprint(f"[green]✓[/green] OpenAI connected")
            except Exception as e:
                rprint(f"[red]✗[/red] OpenAI connection failed: {e}")
        
        # Check Firecrawl if configured
        if config.crawl_provider == "firecrawl" and config.firecrawl.api_key:
            rprint("\n🔍 Checking Firecrawl...")
            try:
                import httpx
                headers = {"Authorization": f"Bearer {config.firecrawl.api_key}"}
                response = httpx.get(f"{config.firecrawl.api_url}/health", headers=headers)
                if response.status_code == 200:
                    rprint(f"[green]✓[/green] Firecrawl connected")
                else:
                    rprint(f"[red]✗[/red] Firecrawl returned status {response.status_code}")
            except Exception as e:
                rprint(f"[red]✗[/red] Firecrawl connection failed: {e}")
        
    except Exception as e:
        rprint(f"[red]Error checking connections:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default="schema", help="Output directory for schema files")
@click.option("--format", "-f", multiple=True, type=click.Choice(["json", "typescript", "markdown"]), help="Schema formats to generate")
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
        raise click.Exit(1)


@cli.command()
def show_schema():
    """Display configuration schema in the terminal."""
    try:
        schema = ConfigSchemaGenerator.generate_json_schema()
        
        # Pretty print the schema
        console.print_json(data=schema)
        
    except Exception as e:
        rprint(f"[red]Error displaying schema:[/red] {e}")
        raise click.Exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--target-version", "-v", default="0.3.0", help="Target version to migrate to")
@click.option("--no-backup", is_flag=True, help="Skip creating backup file")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without making changes")
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
        raise click.Exit(1)


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
            rprint(f"  1. Apply incremental migrations from {from_version} to {to_version}")


if __name__ == "__main__":
    cli()