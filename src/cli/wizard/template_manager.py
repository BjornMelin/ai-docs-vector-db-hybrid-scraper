"""Template management for configuration wizard.

Handles loading, validating, and managing configuration templates
with Pydantic integration for real-time validation.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import Config


console = Console()


class TemplateManager:
    """Manages configuration templates for the wizard."""

    def __init__(self, templates_dir: Path | None = None):
        """Initialize template manager.

        Args:
            templates_dir: Directory containing templates. Defaults to config/templates

        """
        self.templates_dir = templates_dir or Path("config/templates")
        self._templates: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all templates from the templates directory."""
        if not self.templates_dir.exists():
            console.print(
                f"[yellow]Warning: Templates directory not found: {self.templates_dir}[/yellow]"
            )
            return

        template_files = list(self.templates_dir.glob("*.json"))
        if not template_files:
            console.print(
                f"[yellow]Warning: No template files found in {self.templates_dir}[/yellow]"
            )
            return

        for template_file in template_files:
            try:
                with template_file.open() as f:
                    template_data = json.load(f)

                template_name = template_file.stem
                self._templates[template_name] = template_data
                self._metadata[template_name] = self._extract_metadata(
                    template_data, template_name
                )

            except Exception as e:
                console.print(f"[red]Error loading template {template_file}: {e}[/red]")

    def _extract_metadata(
        self, _template_data: dict[str, Any], name: str
    ) -> dict[str, str]:
        """Extract metadata from template data."""
        # Template metadata mapping
        metadata_map = {
            "development": {
                "description": "Local development with debug features",
                "use_case": "Development and testing",
                "features": "Debug mode, local FastEmbed, visual browser",
            },
            "production": {
                "description": "High-performance production deployment",
                "use_case": "Production deployment",
                "features": "OpenAI embeddings, caching, security enabled",
            },
            "personal-use": {
                "description": "Resource-optimized for individual developers",
                "use_case": "Personal projects and learning",
                "features": "Cost-effective, moderate resources",
            },
            "local-only": {
                "description": "Privacy-focused without cloud dependencies",
                "use_case": "Privacy-conscious deployment",
                "features": "Local FastEmbed only, no external APIs",
            },
            "testing": {
                "description": "Optimized for automated testing and CI/CD",
                "use_case": "Automated testing pipelines",
                "features": "Minimal resources, fast execution",
            },
            "minimal": {
                "description": "Quick start with essential settings only",
                "use_case": "Getting started quickly",
                "features": "Simple configuration, easy to understand",
            },
        }

        return metadata_map.get(
            name,
            {
                "description": f"Configuration template for {name}",
                "use_case": f"{name.title()} deployment",
                "features": "Custom configuration template",
            },
        )

    def list_templates(self) -> list[str]:
        """Get list of available template names."""
        return list(self._templates.keys())

    def get_template(self, name: str) -> dict[str, Any] | None:
        """Get template data by name."""
        return self._templates.get(name)

    def get_template_metadata(self, name: str) -> dict[str, str] | None:
        """Get template metadata by name."""
        return self._metadata.get(name)

    def validate_template(
        self, template_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate template data against Config model.

        Returns:
            Tuple of (is_valid, error_message)

        """
        try:
            Config(**template_data)
        except Exception as e:
            return False, str(e)
        else:
            return True, None

    def show_template_comparison(self) -> None:
        """Display a comparison table of all templates."""
        if not self._templates:
            console.print("[yellow]No templates available[/yellow]")
            return

        table = Table(
            title="📋 Available Configuration Templates",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )

        table.add_column("Template", style="bold", width=15)
        table.add_column("Use Case", width=25)
        table.add_column("Key Features", width=40)
        table.add_column("Embedding", style="cyan", width=12)
        table.add_column("Caching", style="green", width=10)

        # Sort templates by recommended order
        template_order = [
            "personal-use",
            "development",
            "production",
            "local-only",
            "testing",
            "minimal",
        ]

        ordered_templates = [
            template_name
            for template_name in template_order
            if template_name in self._templates
        ]

        # Add any templates not in the order
        ordered_templates.extend(
            [
                template_name
                for template_name in self._templates
                if template_name not in ordered_templates
            ]
        )

        for template_name in ordered_templates:
            template_data = self._templates[template_name]
            metadata = self._metadata[template_name]

            # Extract key info from template
            embedding_provider = template_data.get("embedding_provider", "unknown")
            cache_enabled = template_data.get("cache", {}).get("enable_caching", False)

            # Style the template name based on recommendation
            if template_name == "personal-use":
                name_text = Text(f"🏆 {template_name}", style="bold green")
            elif template_name == "development":
                name_text = Text(f"🛠️ {template_name}", style="bold blue")
            elif template_name == "production":
                name_text = Text(f"🚀 {template_name}", style="bold magenta")
            else:
                name_text = Text(f"📄 {template_name}", style="bold")

            table.add_row(
                name_text,
                metadata["use_case"],
                metadata["features"],
                embedding_provider,
                "✅" if cache_enabled else "❌",
            )

        console.print(table)
        console.print(
            "\n[dim]💡 Recommended: 'personal-use' for individual developers, 'production' for deployment[/dim]"
        )

    def preview_template(self, name: str) -> None:
        """Show a detailed preview of a specific template."""
        template_data = self.get_template(name)
        if not template_data:
            console.print(f"[red]Template '{name}' not found[/red]")
            return

        metadata = self.get_template_metadata(name)

        # Create preview panel
        console.print(f"\n[bold cyan]📋 Template Preview: {name}[/bold cyan]")
        console.print(f"[dim]{metadata['description']}[/dim]\n")

        # Show key configuration sections
        sections = [
            ("Environment", ["environment", "debug", "log_level"]),
            ("Providers", ["embedding_provider", "crawl_provider"]),
            ("Cache", ["cache"]),
            ("Database", ["qdrant"]),
            ("Performance", ["performance"]),
            ("Security", ["security"]),
        ]

        for section_name, keys in sections:
            section_data = {}
            for key in keys:
                if key in template_data:
                    section_data[key] = template_data[key]

            if section_data:
                console.print(f"[bold]{section_name}:[/bold]")
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        console.print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            console.print(f"    {sub_key}: {sub_value}")
                    else:
                        console.print(f"  {key}: {value}")
                console.print()

    def create_config_from_template(
        self, template_name: str, overrides: dict[str, Any] | None = None
    ) -> Config:
        """Create a Config object from template with optional overrides.

        Args:
            template_name: Name of the template to use
            overrides: Optional dictionary of values to override in template

        Returns:
            Config object created from template

        Raises:
            ValueError: If template not found or validation fails

        """
        template_data = self.get_template(template_name)
        if not template_data:
            msg = f"Template '{template_name}' not found"
            raise ValueError(msg)

        # Apply overrides if provided
        if overrides:
            template_data = {**template_data, **overrides}

        # Validate and create config
        try:
            return Config(**template_data)
        except Exception as e:
            msg = f"Failed to create config from template: {e}"
            raise ValueError(msg) from e

    def save_template(self, name: str, config: Config, description: str = "") -> Path:
        """Save a Config object as a new template.

        Args:
            name: Name for the new template
            config: Config object to save
            description: Optional description for the template

        Returns:
            Path to saved template file

        """
        template_file = self.templates_dir / f"{name}.json"

        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Save template data
        with template_file.open("w") as f:
            json.dump(config.model_dump(), f, indent=2)

        # Update internal cache
        self._templates[name] = config.model_dump()
        self._metadata[name] = {
            "description": description or f"Custom template: {name}",
            "use_case": "Custom configuration",
            "features": "User-defined settings",
        }

        return template_file
