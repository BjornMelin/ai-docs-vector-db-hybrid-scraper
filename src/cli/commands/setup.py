"""Interactive configuration wizard for initial setup.

This module provides a step-by-step configuration wizard that guides
users through setting up their AI Documentation Scraper instance.
"""

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

console = Console()


class ConfigurationWizard:
    """Interactive configuration wizard for step-by-step setup."""

    def __init__(self):
        """Initialize the configuration wizard."""
        self.console = Console()
        self.config_data: dict[str, Any] = {}

    def welcome(self):
        """Display welcome message for the wizard."""
        welcome_text = Text()
        welcome_text.append("🧙 Configuration Wizard\n", style="bold magenta")
        welcome_text.append(
            "Let's set up your AI Documentation Scraper!\n\n", style="dim"
        )
        welcome_text.append(
            "This wizard will guide you through configuring:\n", style=""
        )
        welcome_text.append("• Vector database connection\n", style="cyan")
        welcome_text.append("• API keys for embedding providers\n", style="cyan")
        welcome_text.append("• Caching and performance settings\n", style="cyan")
        welcome_text.append("• Browser automation preferences", style="cyan")

        panel = Panel(
            welcome_text,
            title="Welcome to Setup",
            title_align="left",
            border_style="magenta",
            padding=(1, 2),
        )
        self.console.print(panel)

    def configure_database(self) -> dict[str, Any]:
        """Configure vector database settings."""
        self.console.print("\n[bold cyan]📊 Vector Database Configuration[/bold cyan]")

        # Qdrant configuration
        db_config = {}

        use_local = Confirm.ask("Use local Qdrant instance?", default=True)

        if use_local:
            host = Prompt.ask("Qdrant host", default="localhost")
            port = Prompt.ask("Qdrant port", default="6333")
            db_config["qdrant"] = {"host": host, "port": int(port), "use_memory": False}
        else:
            url = Prompt.ask("Qdrant Cloud URL", default="")
            api_key = Prompt.ask("Qdrant API key (optional)", default="", password=True)
            db_config["qdrant"] = {"url": url, "api_key": api_key if api_key else None}

        return db_config

    def configure_embeddings(self) -> dict[str, Any]:
        """Configure embedding providers and API keys."""
        self.console.print(
            "\n[bold cyan]🔑 Embedding Provider Configuration[/bold cyan]"
        )

        embedding_config = {}

        # Provider selection
        provider_table = Table(title="Available Embedding Providers")
        provider_table.add_column("Option", style="cyan")
        provider_table.add_column("Provider", style="")
        provider_table.add_column("Description", style="dim")

        provider_table.add_row(
            "1", "OpenAI", "High-quality embeddings, requires API key"
        )
        provider_table.add_row("2", "FastEmbed", "Local embeddings, no API key needed")
        provider_table.add_row("3", "Both", "Use both providers (recommended)")

        self.console.print(provider_table)

        choice = Prompt.ask(
            "Select embedding provider", choices=["1", "2", "3"], default="3"
        )

        if choice in ["1", "3"]:  # OpenAI
            openai_key = Prompt.ask("OpenAI API key", password=True)

            if openai_key:
                embedding_config["openai"] = {
                    "api_key": openai_key,
                    "model": "text-embedding-3-small",
                }

        if choice in ["2", "3"]:  # FastEmbed
            embedding_config["fastembed"] = {
                "model": "BAAI/bge-small-en-v1.5",
                "cache_dir": "~/.cache/fastembed",
            }

        return embedding_config

    def configure_browser(self) -> dict[str, Any]:
        """Configure browser automation settings."""
        self.console.print(
            "\n[bold cyan]🌐 Browser Automation Configuration[/bold cyan]"
        )

        browser_config = {}

        # Browser selection
        headless = Confirm.ask(
            "Run browsers in headless mode? (recommended for servers)", default=True
        )

        # Anti-detection settings
        use_stealth = Confirm.ask("Enable anti-detection features?", default=True)

        browser_config["browser"] = {
            "headless": headless,
            "anti_detection": use_stealth,
            "timeout": 30000,  # 30 seconds
            "max_concurrent": 3,
        }

        return browser_config

    def configure_performance(self) -> dict[str, Any]:
        """Configure performance and caching settings."""
        self.console.print("\n[bold cyan]⚡ Performance Configuration[/bold cyan]")

        perf_config = {}

        # Cache settings
        enable_cache = Confirm.ask(
            "Enable Redis caching? (improves performance)", default=True
        )

        if enable_cache:
            redis_host = Prompt.ask("Redis host", default="localhost")
            redis_port = Prompt.ask("Redis port", default="6379")

            perf_config["cache"] = {
                "redis": {"host": redis_host, "port": int(redis_port), "db": 0}
            }

        # Task queue
        enable_queue = Confirm.ask("Enable background task queue?", default=True)

        if enable_queue:
            perf_config["task_queue"] = {
                "redis_url": f"redis://{redis_host}:{redis_port}/1"
            }

        return perf_config

    def save_configuration(self, config_data: dict[str, Any]) -> Path:
        """Save the configuration to file."""
        self.console.print("\n[bold cyan]💾 Saving Configuration[/bold cyan]")

        # Choose config format
        format_choice = Prompt.ask(
            "Configuration format", choices=["json", "yaml", "toml"], default="json"
        )

        # Choose config location
        default_path = Path.cwd() / f"ai_docs_config.{format_choice}"
        config_path = Prompt.ask("Configuration file path", default=str(default_path))

        config_path = Path(config_path)

        try:
            # Save configuration using ConfigLoader
            if format_choice == "json":
                import json

                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
            elif format_choice == "yaml":
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            elif format_choice == "toml":
                import tomli_w

                with open(config_path, "wb") as f:
                    tomli_w.dump(config_data, f)

            self.console.print(
                f"✅ Configuration saved to: [green]{config_path}[/green]"
            )
            return config_path

        except Exception as e:
            self.console.print(f"❌ Error saving configuration: [red]{e}[/red]")
            raise

    def run_setup(self) -> Path:
        """Run the complete setup wizard."""
        self.welcome()

        if not Confirm.ask("\nReady to start configuration?", default=True):
            self.console.print("Setup cancelled.")
            raise click.Abort()

        # Collect configuration
        config_data = {}

        # Database configuration
        config_data.update(self.configure_database())

        # Embedding configuration
        config_data.update(self.configure_embeddings())

        # Browser configuration
        config_data.update(self.configure_browser())

        # Performance configuration
        config_data.update(self.configure_performance())

        # Save configuration
        config_path = self.save_configuration(config_data)

        # Success message
        success_text = Text()
        success_text.append("🎉 Setup Complete!\n\n", style="bold green")
        success_text.append(
            "Your AI Documentation Scraper is now configured.\n\n", style=""
        )
        success_text.append("Next steps:\n", style="bold")
        success_text.append(
            f"1. Test your configuration: ai-docs config validate {config_path}\n",
            style="cyan",
        )
        success_text.append("2. Check system status: ai-docs status\n", style="cyan")
        success_text.append(
            "3. Create your first collection: ai-docs database create my-docs",
            style="cyan",
        )

        panel = Panel(
            success_text,
            title="Setup Complete",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

        return config_path


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output configuration file path",
)
@click.option(
    "--format",
    "config_format",
    type=click.Choice(["json", "yaml", "toml"]),
    default="json",
    help="Configuration file format",
)
@click.pass_context
def setup(ctx: click.Context, output: Path | None, config_format: str):
    """🧙 Interactive configuration wizard.

    This wizard will guide you through setting up your AI Documentation Scraper
    with step-by-step configuration for all components.

    \b
    The wizard configures:
    • Vector database connection (Qdrant)
    • Embedding providers (OpenAI, FastEmbed)
    • Browser automation settings
    • Caching and performance options
    """
    wizard = ConfigurationWizard()

    try:
        config_path = wizard.run_setup()

        # Validate the created configuration
        if Confirm.ask("\nValidate the new configuration?", default=True):
            from .config import validate_config

            ctx.invoke(validate_config, config_file=config_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user.[/yellow]")
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        raise click.Abort() from e
