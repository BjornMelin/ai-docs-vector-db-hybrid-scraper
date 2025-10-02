"""Validation for configuration wizard.

Provides validation feedback during wizard interaction
using Pydantic models and user-friendly error messages.
"""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.config import Config


console = Console()


class WizardValidator:
    """Provides validation for wizard inputs."""

    def __init__(self):
        """Initialize validator."""
        self.validation_cache: dict[str, bool] = {}

    def validate_api_key(self, provider: str, api_key: str) -> tuple[bool, str | None]:
        """Validate API key format for specific providers.

        Args:
            provider: Provider name (openai, firecrawl, etc.)
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, error_message)

        """
        if not api_key:
            return False, "API key cannot be empty"

        validation_rules = {
            "openai": {
                "pattern": r"^sk-[a-zA-Z0-9_-]{20,}$",
                "message": "OpenAI API key must start with 'sk-' and be at least "
                "20 characters",
            },
            "firecrawl": {
                "pattern": r"^fc-[a-zA-Z0-9_-]{20,}$",
                "message": "Firecrawl API key must start with 'fc-' and be at least "
                "20 characters",
            },
            "anthropic": {
                "pattern": r"^sk-ant-[a-zA-Z0-9_-]{20,}$",
                "message": "Anthropic API key must start with 'sk-ant-' "
                "and be at least 20 characters",
            },
        }

        rule = validation_rules.get(provider.lower())
        if not rule:
            # Generic validation for unknown providers
            if len(api_key) < 10:
                return (
                    False,
                    f"API key for {provider} seems too short (minimum 10 characters)",
                )
            return True, None

        if not re.match(rule["pattern"], api_key):
            return False, rule["message"]

        return True, None

    def validate_url(
        self, url: str, allow_localhost: bool = True
    ) -> tuple[bool, str | None]:
        """Validate URL format.

        Args:
            url: URL to validate
            allow_localhost: Whether to allow localhost URLs

        Returns:
            Tuple of (is_valid, error_message)

        """
        if not url:
            return False, "URL cannot be empty"

        # Basic URL pattern
        url_pattern = r"^https?://[a-zA-Z0-9.-]+(?::[0-9]+)?(?:/.*)?$"

        if not re.match(url_pattern, url):
            return False, "Invalid URL format. Must start with http:// or https://"

        if not allow_localhost and ("localhost" in url or "127.0.0.1" in url):
            return False, "Localhost URLs not allowed"

        return True, None

    def validate_port(self, port: str | int) -> tuple[bool, str | None]:
        """Validate port number.

        Args:
            port: Port number to validate

        Returns:
            Tuple of (is_valid, error_message)

        """
        try:
            port_int = int(port)
        except (ValueError, TypeError):
            return False, "Port must be a number"

        if port_int < 1 or port_int > 65535:
            return False, "Port must be between 1 and 65535"

        if port_int < 1024:
            return False, "Port numbers below 1024 require root privileges"

        return True, None

    def validate_path(
        self, path: str, must_exist: bool = False, must_be_dir: bool = False
    ) -> tuple[bool, str | None]:
        """Validate file/directory path.

        Args:
            path: Path to validate
            must_exist: Whether path must already exist
            must_be_dir: Whether path must be a directory

        Returns:
            Tuple of (is_valid, error_message)

        """
        if not path:
            return False, "Path cannot be empty"

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False, f"Path does not exist: {path}"

        if must_be_dir and path_obj.exists() and not path_obj.is_dir():
            return False, f"Path is not a directory: {path}"

        # Check if path is writable (for parent directory if path doesn't exist)
        try:
            check_path = path_obj if path_obj.exists() else path_obj.parent
            if not check_path.exists():
                check_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return False, f"No write permission for path: {path}"
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}"

        return True, None

    def validate_config_partial(
        self, config_data: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate partial configuration data.

        Args:
            config_data: Partial configuration dictionary

        Returns:
            Tuple of (is_valid, error_messages)

        """
        errors = []

        try:
            # Try to create Config object with minimal required fields
            # Add defaults for missing required fields for validation
            validation_data = {
                "environment": "development",
                "debug": False,
                "log_level": "INFO",
                **config_data,
            }

            Config(**validation_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error["loc"])
                error_msg = error["msg"]
                errors.append(f"{field_path}: {error_msg}")

        except (TypeError, ValueError) as e:
            errors.append(f"Validation error: {e!s}")
        else:
            return True, []

        return False, errors

    def validate_and_show_errors(self, config_data: dict[str, Any]) -> bool:
        """Validate configuration and show user-friendly error messages.

        Args:
            config_data: Configuration data to validate

        Returns:
            True if valid, False otherwise

        """
        is_valid, errors = self.validate_config_partial(config_data)

        if not is_valid:
            self._show_validation_errors(errors)

        return is_valid

    def _show_validation_errors(self, errors: list[str]) -> None:
        """Display validation errors in a user-friendly format."""
        if not errors:
            return

        error_text = Text()
        error_text.append("Configuration validation failed:\n\n", style="bold red")

        for i, error in enumerate(errors, 1):
            error_text.append(f"{i}. ", style="red")
            error_text.append(f"{error}\n", style="")

        error_text.append(
            "\n💡 Tip: Check your input values and try again.", style="dim"
        )

        panel = Panel(
            error_text,
            title="❌ Validation Errors",
            border_style="red",
            padding=(1, 2),
        )
        console.print(panel)

    def suggest_fixes(self, errors: list[str]) -> dict[str, str]:
        """Suggest automatic fixes for common validation errors.

        Args:
            errors: List of validation error messages

        Returns:
            Dictionary of field -> suggested_fix

        """
        suggestions = {}

        for error in errors:
            if "api_key" in error.lower():
                if "openai" in error.lower():
                    suggestions["openai_api_key"] = "Format: sk-your_actual_key_here"
                elif "firecrawl" in error.lower():
                    suggestions["firecrawl_api_key"] = "Format: fc-your_actual_key_here"

            elif "url" in error.lower():
                if "qdrant" in error.lower():
                    suggestions["qdrant_url"] = "http://localhost:6333"
                elif "redis" in error.lower():
                    suggestions["redis_url"] = "redis://localhost:6379"

            elif "port" in error.lower():
                suggestions["port"] = "Use a port number between 1024-65535"

            elif "chunk_size" in error.lower():
                suggestions["chunk_size"] = (
                    "Recommended: 1600 (must be positive integer)"
                )

            elif "batch_size" in error.lower():
                suggestions["batch_size"] = (
                    "Recommended: 100 (must be positive integer)"
                )

        return suggestions

    def validate_file_path(
        self, path: str, must_exist: bool = False
    ) -> tuple[bool, str | None]:
        """Validate file path.

        Args:
            path: File path to validate
            must_exist: Whether file must already exist

        Returns:
            Tuple of (is_valid, error_message)

        """
        return self.validate_path(path, must_exist=must_exist, must_be_dir=False)

    def validate_directory_path(
        self, path: str, must_exist: bool = False
    ) -> tuple[bool, str | None]:
        """Validate directory path.

        Args:
            path: Directory path to validate
            must_exist: Whether directory must already exist

        Returns:
            Tuple of (is_valid, error_message)

        """
        return self.validate_path(path, must_exist=must_exist, must_be_dir=True)

    def validate_json_string(
        self, json_str: str
    ) -> tuple[bool, str | None, dict | None]:
        """Validate JSON string.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_data)

        """
        if not json_str:
            return False, "JSON string cannot be empty", None

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None
        else:
            return True, None, parsed

    def show_validation_summary(self, config) -> None:
        """Show validation summary for successful configuration.

        Args:
            config: Validated configuration object

        """
        summary_text = Text()
        summary_text.append("✅ Configuration is valid!\n\n", style="bold green")

        summary_text.append("Configuration Summary:\n", style="bold")

        # Show key configuration points
        if hasattr(config, "qdrant"):
            if hasattr(config.qdrant, "host"):
                summary_text.append(
                    f"• Database: Qdrant at {config.qdrant.host}:"
                    f"{config.qdrant.port}\n",
                    style="cyan",
                )
            elif hasattr(config.qdrant, "url"):
                summary_text.append(
                    f"• Database: Qdrant Cloud at {config.qdrant.url}\n", style="cyan"
                )

        if hasattr(config, "openai") and hasattr(config.openai, "model"):
            summary_text.append(
                f"• Embeddings: OpenAI {config.openai.model}\n", style="cyan"
            )

        if hasattr(config, "debug") and config.debug:
            summary_text.append("• Debug mode: Enabled\n", style="yellow")

        summary_text.append(
            "\n🎉 Ready to start processing documents!", style="bold green"
        )

        panel = Panel(
            summary_text,
            title="✅ Configuration Valid",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)

    def validate_template_customization(
        self, template_data: dict[str, Any], customizations: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate template customizations.

        Args:
            template_data: Base template data
            customizations: User customizations to apply

        Returns:
            Tuple of (is_valid, error_messages)

        """
        # Merge template with customizations
        merged_data = {**template_data, **customizations}

        # Validate the merged configuration
        return self.validate_config_partial(merged_data)
