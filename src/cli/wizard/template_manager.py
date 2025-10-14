"""Template management utilities for the configuration wizard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import validate_settings_payload
from src.config.template_utils import calculate_diff, merge_overrides


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from src.config import Settings


console = Console()

_BASE_TEMPLATE_FILENAME = "base.json"
_PROFILE_INDEX_FILENAME = "profiles.json"


class TemplateManager:
    """Manages configuration templates for the wizard."""

    def __init__(self, templates_dir: Path | None = None):
        """Initialize the template manager.

        Args:
            templates_dir: Directory containing template assets. Defaults to
                ``config/templates``.
        """

        self.templates_dir = templates_dir or Path("config/templates")
        self._templates: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self._base_template: dict[str, Any] = {}
        self._profile_index: dict[str, Any] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load base template and profile overrides from disk."""

        if not self.templates_dir.exists():
            console.print(
                "[yellow]Warning: Templates directory not found: "
                f"{self.templates_dir}[/yellow]"
            )
            return

        base_path = self.templates_dir / _BASE_TEMPLATE_FILENAME
        profiles_path = self.templates_dir / _PROFILE_INDEX_FILENAME

        try:
            self._base_template = self._load_json(base_path)
        except FileNotFoundError:
            console.print(
                f"[yellow]Warning: Base template missing at {base_path}[/yellow]"
            )
            self._base_template = {}
        except json.JSONDecodeError as exc:
            console.print(
                f"[red]Invalid JSON in base template {base_path}: {exc}[/red]"
            )
            self._base_template = {}

        try:
            self._profile_index = self._load_json(profiles_path)
        except FileNotFoundError:
            console.print(
                f"[yellow]Warning: Profile index missing at {profiles_path}[/yellow]"
            )
            self._profile_index = {}
        except json.JSONDecodeError as exc:
            console.print(
                f"[red]Invalid JSON in profile index {profiles_path}: {exc}[/red]"
            )
            self._profile_index = {}

        self._templates.clear()
        self._metadata.clear()

        for name, record in self._profile_index.items():
            overrides = record.get("overrides", {})
            if not isinstance(overrides, dict):
                console.print(
                    f"[yellow]Profile '{name}' overrides must be a mapping; "
                    "ignoring invalid overrides[/yellow]"
                )
                overrides = {}
            template = self._merge_template(overrides)
            self._templates[name] = template
            self._metadata[name] = self._extract_metadata(name, record)

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Return JSON content from ``path`` as a dictionary."""

        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            msg = f"Template asset {path} must contain a JSON object."
            raise ValueError(msg)
        return data

    def _merge_template(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge overrides onto the base template and return a new mapping."""

        return merge_overrides(self._base_template, overrides)

    def _extract_metadata(self, name: str, record: dict[str, Any]) -> dict[str, str]:
        """Return human-readable metadata for ``name``."""

        description = record.get("description", f"Configuration template for {name}")
        use_case = record.get("use_case", f"{name.title()} deployment")
        features = record.get("features", "Custom configuration template")
        return {
            "description": description,
            "use_case": use_case,
            "features": features,
        }

    def list_templates(self) -> list[str]:
        """Return the available template identifiers."""

        return list(self._templates.keys())

    def get_template(self, name: str) -> dict[str, Any] | None:
        """Return the template data for ``name`` if present."""

        return self._templates.get(name)

    def get_template_metadata(self, name: str) -> dict[str, str] | None:
        """Return metadata describing the template named ``name``."""

        return self._metadata.get(name)

    def validate_template(
        self, template_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate template data against the Settings model."""

        is_valid, errors, _ = validate_settings_payload(template_data)
        if not is_valid:
            return False, errors[0] if errors else "Validation failed"
        return True, None

    def show_template_comparison(self) -> None:
        """Display a comparison table of all templates."""

        if not self._templates:
            console.print("[yellow]No templates available[/yellow]")
            return

        table = Table(
            title="Available Configuration Templates",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )

        table.add_column("Template", style="bold", width=15)
        table.add_column("Use Case", width=25)
        table.add_column("Key Features", width=40)
        table.add_column("Embedding", style="cyan", width=12)
        table.add_column("Caching", style="green", width=10)

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

        ordered_templates.extend(
            [
                template_name
                for template_name in self._templates
                if template_name not in ordered_templates
            ]
        )

        for template_name in ordered_templates:
            template_data = self._templates[template_name]
            metadata = self._metadata.get(
                template_name,
                {"use_case": "Unknown", "features": "No description available"},
            )

            embedding_provider = template_data.get("embedding_provider", "unknown")
            cache_enabled = template_data.get("cache", {}).get("enable_caching", False)

            if template_name == "personal-use":
                name_text = Text(template_name, style="bold green")
            elif template_name == "development":
                name_text = Text(template_name, style="bold blue")
            elif template_name == "production":
                name_text = Text(template_name, style="bold magenta")
            else:
                name_text = Text(template_name, style="bold")

            table.add_row(
                name_text,
                metadata["use_case"],
                metadata["features"],
                embedding_provider,
                "Enabled" if cache_enabled else "Disabled",
            )

        console.print(table)
        console.print(
            "\n[dim]Recommendation: 'personal-use' for individual developers, "
            "'production' for deployment.[/dim]"
        )

    def preview_template(self, name: str) -> None:
        """Show a detailed preview of a specific template."""

        template_data = self.get_template(name)
        if not template_data:
            console.print(f"[red]Template '{name}' not found[/red]")
            return

        metadata = self.get_template_metadata(name)

        console.print(f"\n[bold cyan]Template Preview: {name}[/bold cyan]")
        if metadata and "description" in metadata:
            console.print(f"[dim]{metadata['description']}[/dim]\n")
        else:
            console.print("[dim]No description available[/dim]\n")

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
    ) -> Settings:
        """Create a Settings object from template with optional overrides."""

        template_data = self.get_template(template_name)
        if not template_data:
            msg = f"Template '{template_name}' not found"
            raise ValueError(msg)

        if overrides:
            template_data = merge_overrides(template_data, overrides)

        is_valid, errors, settings = validate_settings_payload(template_data)
        if not is_valid or settings is None:
            details = "; ".join(errors) if errors else "unknown error"
            msg = f"Failed to create config from template: {details}"
            raise ValueError(msg)

        return settings

    def save_template(self, name: str, config: Settings, description: str = "") -> Path:
        """Persist a Settings object as a profile override.

        Args:
            name: Name for the new template.
            config: Settings object to persist.
            description: Optional human readable description.

        Returns:
            Path to the updated profile index file.
        """

        config_data = config.model_dump()
        overrides = calculate_diff(self._base_template, config_data)
        metadata = {
            "description": description or f"Custom template: {name}",
            "use_case": "Custom profile configuration",
            "features": "User-defined settings",
            "overrides": overrides,
        }

        self._profile_index[name] = metadata
        self._metadata[name] = {
            "description": metadata["description"],
            "use_case": metadata["use_case"],
            "features": metadata["features"],
        }
        self._templates[name] = self._merge_template(overrides)

        profiles_path = self.templates_dir / _PROFILE_INDEX_FILENAME
        profiles_path.write_text(
            json.dumps(self._profile_index, indent=2), encoding="utf-8"
        )

        return profiles_path


__all__ = [
    "TemplateManager",
]
