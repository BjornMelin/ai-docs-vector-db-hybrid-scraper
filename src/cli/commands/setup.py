"""Interactive configuration wizard for initial setup.

This module provides a step-by-step configuration wizard that guides
users through setting up their AI Documentation Scraper instance.
Modern template-driven approach with real-time validation.
"""

import json
from pathlib import Path
from typing import Any

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.cli.wizard import ProfileManager, TemplateManager, WizardValidator


# Optional import for config validation
try:
    from .config import validate_config
except ImportError:
    validate_config = None


console = Console()


def _abort_no_template(profile: str) -> None:
    """Helper function to abort when no template is found for profile."""
    msg = f"No template found for profile '{profile}'"
    raise ValueError(msg)


def _abort_validation_errors() -> None:
    """Helper function to abort due to validation errors."""
    msg = "Setup cancelled due to validation errors"
    raise click.Abort(msg)


def _abort_profile_not_found(profile: str, available_profiles: list[str]) -> None:
    """Helper function to abort when profile is not found."""
    console.print(
        f"[red]Profile '{profile}' not found. Available: {', '.join(available_profiles)}[/red]"
    )
    raise click.Abort()


class ConfigurationWizard:
    """Modern template-driven configuration wizard with real-time validation."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the configuration wizard.

        Args:
            config_dir: Directory for configuration files

        """
        self.console = Console()
        self.template_manager = TemplateManager()
        self.profile_manager = ProfileManager(config_dir)
        self.validator = WizardValidator()

        self.selected_template: str | None = None
        self.selected_profile: str | None = None
        self.config_data: dict[str, Any] = {}
        self.customizations: dict[str, Any] = {}

    def welcome(self):
        """Display welcome message for the wizard."""
        welcome_text = Text()
        welcome_text.append("🧙 Modern Configuration Wizard\n", style="bold magenta")
        welcome_text.append(
            "Welcome to the AI Documentation Scraper setup wizard!\n\n", style="dim"
        )
        welcome_text.append("This modern wizard offers:\n", style="")
        welcome_text.append(
            "• 🎯 Profile-based configuration templates\n", style="cyan"
        )
        welcome_text.append(
            "• ⚡ Real-time validation with helpful feedback\n", style="cyan"
        )
        welcome_text.append("• 🛠️ Smart customization options\n", style="cyan")
        welcome_text.append("• 📋 Template preview and comparison\n", style="cyan")
        welcome_text.append("• 🔧 Environment-specific optimizations", style="cyan")

        panel = Panel(
            welcome_text,
            title="🚀 Template-Driven Setup",
            title_align="left",
            border_style="magenta",
            padding=(1, 2),
        )
        self.console.print(panel)

    def select_profile(self) -> str:
        """Let user select a configuration profile."""
        self.console.print("\n[bold cyan]🎯 Profile Selection[/bold cyan]")

        # Show available profiles
        self.profile_manager.show_profiles_table()

        # Get user choice using questionary
        profile_choices = [
            questionary.Choice(
                title=f"🏆 {name} (Recommended for most users)", value=name
            )
            if name == "personal"
            else questionary.Choice(
                title=f"🛠️ {name} (Development and debugging)", value=name
            )
            if name == "development"
            else questionary.Choice(
                title=f"🚀 {name} (Production deployment)", value=name
            )
            if name == "production"
            else questionary.Choice(title=name, value=name)
            for name in self.profile_manager.list_profiles()
        ]

        selected_profile = questionary.select(
            "Choose a configuration profile:",
            choices=profile_choices,
            default="personal",
        ).ask()

        if not selected_profile:
            msg = "Profile selection cancelled"
            raise click.Abort(msg)

        # Show profile details
        self.profile_manager.show_profile_setup_instructions(selected_profile)

        # Confirm selection
        confirm = questionary.confirm(
            f"Use the '{selected_profile}' profile as your starting point?",
            default=True,
        ).ask()

        if not confirm:
            return self.select_profile()  # Recursively ask again

        return selected_profile

    def customize_template(self, template_name: str) -> dict[str, Any]:
        """Allow user to customize the selected template."""
        self.console.print(
            f"\n[bold cyan]🛠️ Customizing '{template_name}' Template[/bold cyan]"
        )

        template_data = self.template_manager.get_template(template_name)
        if not template_data:
            msg = f"Template '{template_name}' not found"
            raise ValueError(msg)

        customizations = {}

        # Show template preview
        preview_choice = questionary.confirm(
            "Would you like to preview the template configuration?", default=True
        ).ask()

        if preview_choice:
            self.template_manager.preview_template(template_name)

        # Ask if user wants to customize
        customize_choice = questionary.confirm(
            "Do you want to customize any settings? (or use template as-is)",
            default=False,
        ).ask()

        if not customize_choice:
            return customizations

        # Guide through key customization options
        customization_sections = [
            ("API Keys", self._customize_api_keys),
            ("Database Connection", self._customize_database),
            ("Performance Settings", self._customize_performance),
            ("Advanced Options", self._customize_template),
        ]

        for section_name, customize_func in customization_sections:
            if questionary.confirm(f"Customize {section_name}?", default=False).ask():
                section_customizations = customize_func(template_data)
                customizations.update(section_customizations)

        return customizations

    def _customize_api_keys(self, _template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize API key settings."""
        customizations = {}

        # OpenAI API Key
        if questionary.confirm("Set OpenAI API key?", default=True).ask():
            while True:
                api_key = questionary.password("Enter OpenAI API key:").ask()
                if api_key:
                    is_valid, error = self.validator.validate_api_key("openai", api_key)
                    if is_valid:
                        customizations.setdefault("openai", {})["api_key"] = api_key
                        break
                    self.console.print(f"[red]Invalid API key: {error}[/red]")
                    if not questionary.confirm("Try again?", default=True).ask():
                        break

        # Firecrawl API Key (optional)
        if questionary.confirm(
            "Set Firecrawl API key? (optional)", default=False
        ).ask():
            while True:
                api_key = questionary.password("Enter Firecrawl API key:").ask()
                if api_key:
                    is_valid, error = self.validator.validate_api_key(
                        "firecrawl", api_key
                    )
                    if is_valid:
                        customizations.setdefault("firecrawl", {})["api_key"] = api_key
                        break
                    self.console.print(f"[red]Invalid API key: {error}[/red]")
                    if not questionary.confirm("Try again?", default=True).ask():
                        break

        return customizations

    def customize_database(self, template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize database connection settings (public method for testing)."""
        return self._customize_database(template_data)

    def customize_api_keys(self, template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize API keys (public method for testing)."""
        return self._customize_api_keys(template_data)

    def customize_performance(self, template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize performance settings (public method for testing)."""
        return self._customize_performance(template_data)

    def customize_template(self, template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize template settings (public method for testing)."""
        return self._customize_template(template_data)

    def show_success_message(self, config_file: Path) -> None:
        """Show success message (public method for testing)."""
        return self._show_success_message(config_file)

    def _customize_database(self, _template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize database connection settings."""
        customizations = {}

        connection_type = questionary.select(
            "Qdrant connection type:",
            choices=[
                "Local (localhost:6333)",
                "Custom host/port",
                "Qdrant Cloud URL",
                "Keep template default",
            ],
            default="Keep template default",
        ).ask()

        if connection_type == "Custom host/port":
            host = questionary.text("Qdrant host:", default="localhost").ask()
            port = questionary.text("Qdrant port:", default="6333").ask()

            is_valid, error = self.validator.validate_url(f"http://{host}:{port}")
            if is_valid:
                customizations["qdrant"] = {"host": host, "port": int(port)}
            else:
                self.console.print(f"[red]Invalid connection: {error}[/red]")

        elif connection_type == "Qdrant Cloud URL":
            url = questionary.text("Qdrant Cloud URL:").ask()
            api_key = questionary.password("Qdrant API key:").ask()

            is_valid, error = self.validator.validate_url(url, allow_localhost=False)
            if is_valid:
                customizations["qdrant"] = {"url": url, "api_key": api_key}
            else:
                self.console.print(f"[red]Invalid URL: {error}[/red]")

        return customizations

    def _customize_performance(self, _template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize performance settings."""
        customizations = {}

        # Chunk size customization
        if questionary.confirm(
            "Customize text processing settings?", default=False
        ).ask():
            chunk_size = questionary.text(
                "Chunk size (characters per text chunk):", default="1600"
            ).ask()

            try:
                chunk_size_int = int(chunk_size)
                if chunk_size_int > 0:
                    customizations.setdefault("text_processing", {})["chunk_size"] = (
                        chunk_size_int
                    )
            except ValueError:
                self.console.print("[red]Invalid chunk size[/red]")

        return customizations

    def _customize_template(self, _template_data: dict[str, Any]) -> dict[str, Any]:
        """Customize template settings."""
        customizations = {}

        # Debug mode
        if questionary.confirm("Enable debug mode?", default=False).ask():
            customizations["debug"] = True
            customizations["log_level"] = "DEBUG"

        return customizations

    def save_configuration(
        self, profile_name: str, config_data: dict[str, Any]
    ) -> Path:
        """Save the configuration using profile manager."""
        self.console.print("\n[bold cyan]💾 Saving Configuration[/bold cyan]")

        try:
            # Create profile configuration
            profile_config_path = self.profile_manager.create_profile_config(
                profile_name, customizations=config_data
            )

            # Ask if user wants to activate this profile
            activate_choice = questionary.confirm(
                f"Activate '{profile_name}' profile as your default configuration?",
                default=True,
            ).ask()

            if activate_choice:
                config_path = self.profile_manager.activate_profile(profile_name)
            else:
                config_path = profile_config_path

            # Generate .env file
            env_choice = questionary.confirm(
                "Generate .env file with recommended environment variables?",
                default=True,
            ).ask()

            if env_choice:
                env_path = self.profile_manager.generate_env_file(profile_name)
                self.console.print(
                    f"📄 Environment file created: [green]{env_path}[/green]"
                )

        except Exception as e:
            self.console.print(f"❌ Error saving configuration: [red]{e}[/red]")
            raise
        else:
            return config_path

    def run_setup(self) -> Path:
        """Run the complete modern template-driven setup wizard."""
        self.welcome()

        # Check if user wants to proceed
        ready = questionary.confirm(
            "Ready to start the modern configuration wizard?", default=True
        ).ask()

        if not ready:
            self.console.print("Setup cancelled.")
            raise click.Abort()

        try:
            # Step 1: Profile Selection (unless pre-selected)
            if not self.selected_profile:
                self.selected_profile = self.select_profile()

            # Get template name for the selected profile
            template_name = self.profile_manager.profile_templates.get(
                self.selected_profile
            )
            if not template_name:
                _abort_no_template(self.selected_profile)

            # Step 2: Template Customization
            self.customizations = self.customize_template(template_name)

            # Step 3: Validation
            self.console.print("\n[bold cyan]✅ Validating Configuration[/bold cyan]")

            # Create config from template + customizations
            config = self.template_manager.create_config_from_template(
                template_name, self.customizations
            )

            # Validate configuration
            is_valid = self.validator.validate_and_show_errors(config.model_dump())

            if is_valid:
                self.validator.show_validation_summary(config)
            else:
                fix_choice = questionary.confirm(
                    "Configuration has validation errors. Continue anyway?",
                    default=False,
                ).ask()
                if not fix_choice:
                    _abort_validation_errors()

            # Step 4: Save Configuration
            config_path = self.save_configuration(
                self.selected_profile, self.customizations
            )

            # Step 5: Success Message
            self._show_success_message(config_path)

            return config_path

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user.[/yellow]")
            raise click.Abort() from None
        except Exception as e:
            self.console.print(f"\n[red]Setup failed: {e}[/red]")
            raise

    def _show_success_message(self, config_path: Path) -> None:
        """Show final success message with next steps."""
        success_text = Text()
        success_text.append("🎉 Modern Setup Complete!\n\n", style="bold green")
        success_text.append(
            f"Your '{self.selected_profile}' profile is now configured and ready to use.\n\n",
            style="",
        )

        success_text.append("Configuration details:\n", style="bold")
        success_text.append(f"• Profile: {self.selected_profile}\n", style="cyan")
        success_text.append(f"• Config file: {config_path}\n", style="cyan")
        success_text.append(
            f"• Template: {self.profile_manager.profile_templates.get(self.selected_profile)}\n",
            style="cyan",
        )

        success_text.append("\nNext steps:\n", style="bold")
        success_text.append(
            "1. Test configuration: uv run python -m src.cli.main config validate\n",
            style="cyan",
        )
        success_text.append(
            "2. Start services: ./scripts/start-services.sh\n", style="cyan"
        )
        success_text.append(
            "3. Check system status: uv run python -m src.cli.main status\n",
            style="cyan",
        )
        success_text.append(
            "4. Create your first collection: uv run python -m src.cli.main database create my-docs\n",
            style="cyan",
        )

        success_text.append(
            f"\n💡 To use this profile again: ./setup.sh --profile {self.selected_profile}",
            style="dim",
        )

        panel = Panel(
            success_text,
            title="🚀 Template-Driven Setup Complete",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)


@click.command()
@click.option(
    "--profile",
    "-p",
    type=str,
    help="Pre-select a configuration profile (personal, development, production, etc.)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output configuration file path (optional, uses profile default)",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path, exists=True),
    default=Path("config"),
    help="Configuration directory (default: config/)",
)
@click.pass_context
def setup(
    ctx: click.Context, profile: str | None, _output: Path | None, config_dir: Path
):
    """🧙 Modern template-driven configuration wizard.

    This wizard uses configuration profiles and templates to guide you through
    setting up your AI Documentation Scraper with best practices and validation.

    \b
    Features:
    • 🎯 Profile-based templates (personal, development, production)
    • ⚡ Real-time validation with helpful error messages
    • 🛠️ Smart customization with questionary interactions
    • 📋 Template preview and comparison
    • 🔧 Environment-specific optimizations

    \b
    Available profiles:
    • personal     - Recommended for individual developers
    • development  - Local development with debugging
    • production   - High-performance deployment
    • testing      - CI/CD and automated testing
    • local-only   - Privacy-focused, no cloud services
    • minimal      - Quick start with essential settings
    """
    wizard = ConfigurationWizard(config_dir)

    try:
        # If profile is pre-selected, use it
        if profile:
            available_profiles = wizard.profile_manager.list_profiles()
            if profile not in available_profiles:
                _abort_profile_not_found(profile, available_profiles)
            wizard.selected_profile = profile
            console.print(f"[cyan]Using pre-selected profile: {profile}[/cyan]")

        config_path = wizard.run_setup()

        # Validate the created configuration
        validate_choice = questionary.confirm(
            "Validate the new configuration?", default=True
        ).ask()

        if validate_choice:
            if validate_config is not None:
                ctx.invoke(validate_config, config_file=config_path)
            else:
                # Fallback validation using our wizard validator
                console.print(
                    "[yellow]Using wizard validation (config command not available)[/yellow]"
                )
                config_data = json.loads(config_path.read_text())
                wizard.validator.validate_and_show_errors(config_data)

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user.[/yellow]")
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        raise click.Abort() from e
