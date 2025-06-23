import typing

"""Profile management for configuration wizard.

Handles profile-specific configuration operations, including
profile switching, environment-specific settings, and template mapping.
"""

import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .template_manager import TemplateManager

console = Console()


class ProfileManager:
    """Manages configuration profiles for different environments."""

    def __init__(self, config_dir: typing.Optional[Path] = None):
        """Initialize profile manager.

        Args:
            config_dir: Directory for storing profile configurations
        """
        self.config_dir = config_dir or Path("config")
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.template_manager = TemplateManager()

        # Profile to template mapping
        self.profile_templates = {
            "development": "development",
            "production": "production",
            "testing": "testing",
            "local-dev": "development",
            "personal": "personal-use",
            "privacy": "local-only",
            "minimal": "minimal",
        }

    def list_profiles(self) -> list[str]:
        """Get list of available profiles."""
        return list(self.profile_templates.keys())

    def get_profile_info(self, profile_name: str) -> dict[str, str] | None:
        """Get information about a specific profile."""
        template_name = self.profile_templates.get(profile_name)
        if not template_name:
            return None

        metadata = self.template_manager.get_template_metadata(template_name)
        if not metadata:
            return None

        # Add profile-specific information
        profile_info = {
            "template": template_name,
            "description": metadata["description"],
            "use_case": metadata["use_case"],
            "features": metadata["features"],
        }

        # Add profile-specific setup instructions
        setup_instructions = {
            "development": "Local development with debugging enabled",
            "production": "Production deployment with security and monitoring",
            "testing": "CI/CD pipeline and automated testing",
            "local-dev": "Local development (alias for development)",
            "personal": "Personal projects and learning (recommended for individuals)",
            "privacy": "Privacy-focused deployment without cloud services",
            "minimal": "Quick start with minimal configuration",
        }

        profile_info["setup"] = setup_instructions.get(
            profile_name, "Custom profile configuration"
        )

        return profile_info

    def show_profiles_table(self) -> None:
        """Display a table of available profiles."""
        table = Table(
            title="🎯 Available Configuration Profiles",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )

        table.add_column("Profile", style="bold", width=12)
        table.add_column("Template", style="cyan", width=15)
        table.add_column("Use Case", width=25)
        table.add_column("Key Features", width=35)

        # Group profiles by recommendation
        recommended_profiles = ["personal", "development", "production"]
        other_profiles = [
            p for p in self.list_profiles() if p not in recommended_profiles
        ]

        # Add recommended profiles first
        for profile_name in recommended_profiles:
            info = self.get_profile_info(profile_name)
            if info:
                # Highlight recommended profiles
                if profile_name == "personal":
                    name_text = Text(f"🏆 {profile_name}", style="bold green")
                elif profile_name == "development":
                    name_text = Text(f"🛠️ {profile_name}", style="bold blue")
                elif profile_name == "production":
                    name_text = Text(f"🚀 {profile_name}", style="bold magenta")
                else:
                    name_text = Text(profile_name, style="bold")

                table.add_row(
                    name_text,
                    info["template"],
                    info["use_case"],
                    info["features"],
                )

        # Add a separator
        if other_profiles:
            table.add_section()

        # Add other profiles
        for profile_name in other_profiles:
            info = self.get_profile_info(profile_name)
            if info:
                table.add_row(
                    profile_name,
                    info["template"],
                    info["use_case"],
                    info["features"],
                )

        console.print(table)
        console.print("\n[dim]💡 Most users should start with 'personal' profile[/dim]")

    def create_profile_config(
        self,
        profile_name: str,
        output_path: typing.Optional[Path] = None,
        customizations: dict | None = None,
    ) -> Path:
        """Create configuration file for a specific profile.

        Args:
            profile_name: Name of the profile to create
            output_path: Optional custom output path
            customizations: Optional customizations to apply

        Returns:
            Path to created configuration file

        Raises:
            ValueError: If profile not found
        """
        template_name = self.profile_templates.get(profile_name)
        if not template_name:
            raise ValueError(f"Profile '{profile_name}' not found")

        # Create config from template
        config = self.template_manager.create_config_from_template(
            template_name, customizations
        )

        # Determine output path
        if output_path is None:
            output_path = self.profiles_dir / f"{profile_name}.json"

        # Save configuration
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            import json

            json.dump(config.model_dump(), f, indent=2)

        return output_path

    def activate_profile(
        self, profile_name: str, target_path: Path | None = None
    ) -> Path:
        """Activate a profile by copying it to the main config location.

        Args:
            profile_name: Name of profile to activate
            target_path: Target path for active config (default: config.json)

        Returns:
            Path to activated configuration file
        """
        if target_path is None:
            target_path = Path("config.json")

        # Create profile config if it doesn't exist
        profile_config_path = self.profiles_dir / f"{profile_name}.json"
        if not profile_config_path.exists():
            profile_config_path = self.create_profile_config(profile_name)

        # Copy to target location
        shutil.copy2(profile_config_path, target_path)

        console.print(f"✅ Activated profile '{profile_name}' -> {target_path}")
        return target_path

    def get_environment_overrides(self, profile_name: str) -> dict[str, str]:
        """Get recommended environment variable overrides for a profile.

        Args:
            profile_name: Name of the profile

        Returns:
            Dictionary of environment variable overrides
        """
        # Profile-specific environment recommendations
        env_overrides = {
            "development": {
                "AI_DOCS__ENVIRONMENT": "development",
                "AI_DOCS__DEBUG": "true",
                "AI_DOCS__LOG_LEVEL": "DEBUG",
            },
            "production": {
                "AI_DOCS__ENVIRONMENT": "production",
                "AI_DOCS__DEBUG": "false",
                "AI_DOCS__LOG_LEVEL": "INFO",
                "AI_DOCS__SECURITY__REQUIRE_API_KEYS": "true",
            },
            "testing": {
                "AI_DOCS__ENVIRONMENT": "testing",
                "AI_DOCS__DEBUG": "false",
                "AI_DOCS__LOG_LEVEL": "WARNING",
                "AI_DOCS__CACHE__ENABLE_CACHING": "false",
            },
            "personal": {
                "AI_DOCS__ENVIRONMENT": "development",
                "AI_DOCS__DEBUG": "false",
                "AI_DOCS__LOG_LEVEL": "INFO",
            },
        }

        # Map aliases to their primary profiles
        profile_aliases = {
            "local-dev": "development",
            "privacy": "personal",
            "minimal": "development",
        }

        # Resolve aliases
        resolved_profile = profile_aliases.get(profile_name, profile_name)

        return env_overrides.get(resolved_profile, {})

    def generate_env_file(
        self, profile_name: str, output_path: Path | None = None
    ) -> Path:
        """Generate .env file for a profile.

        Args:
            profile_name: Name of the profile
            output_path: Output path for .env file (default: .env.{profile})

        Returns:
            Path to generated .env file
        """
        if output_path is None:
            output_path = Path(f".env.{profile_name}")

        env_overrides = self.get_environment_overrides(profile_name)

        # Create .env file content
        env_content = [
            f"# Environment configuration for {profile_name} profile",
            "# Generated by AI Documentation Vector DB Hybrid Scraper",
            "",
        ]

        # Add environment overrides
        for key, value in env_overrides.items():
            env_content.append(f"{key}={value}")

        # Add common API key placeholders
        env_content.extend(
            [
                "",
                "# API Keys (replace with your actual keys)",
                "# AI_DOCS__OPENAI__API_KEY=sk-your_openai_key_here",
                "# AI_DOCS__FIRECRAWL__API_KEY=fc-your_firecrawl_key_here",
                "# AI_DOCS__ANTHROPIC__API_KEY=sk-ant-your_anthropic_key_here",
            ]
        )

        # Write .env file
        with open(output_path, "w") as f:
            f.write("\n".join(env_content))

        return output_path

    def show_profile_setup_instructions(self, profile_name: str) -> None:
        """Show setup instructions for a specific profile."""
        info = self.get_profile_info(profile_name)
        if not info:
            console.print(f"[red]Profile '{profile_name}' not found[/red]")
            return

        instructions_text = Text()
        instructions_text.append(
            f"🎯 Setup Instructions for '{profile_name}' Profile\n\n", style="bold cyan"
        )

        instructions_text.append(f"Description: {info['description']}\n", style="")
        instructions_text.append(f"Use Case: {info['use_case']}\n\n", style="dim")

        # Profile-specific instructions
        instructions_text.append("Setup Steps:\n", style="bold")

        instructions_text.append("1. Activate this profile:\n", style="")
        instructions_text.append(
            f"   ./setup.sh --profile {profile_name}\n\n", style="cyan"
        )

        instructions_text.append("2. Set required API keys in environment:\n", style="")
        env_vars = self.get_environment_overrides(profile_name)
        if env_vars:
            for key, value in env_vars.items():
                instructions_text.append(f"   export {key}={value}\n", style="green")
        instructions_text.append(
            "   export AI_DOCS__OPENAI__API_KEY=sk-your_key_here\n", style="green"
        )
        instructions_text.append("\n")

        instructions_text.append("3. Start services:\n", style="")
        if profile_name == "production":
            instructions_text.append("   docker-compose up -d\n\n", style="cyan")
        else:
            instructions_text.append(
                "   docker-compose -f docker-compose.yml up -d\n\n", style="cyan"
            )

        instructions_text.append("4. Verify setup:\n", style="")
        instructions_text.append(
            "   uv run python -m src.cli.main config validate\n", style="cyan"
        )

        panel = Panel(
            instructions_text,
            title=f"Setup Guide: {profile_name}",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)
