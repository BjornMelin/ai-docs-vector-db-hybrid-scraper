"""Shared utilities for advanced configuration management.

This module provides common utilities used across the configuration management system
including versioning, hashing, and path management.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ConfigVersioning:
    """Utilities for configuration versioning and change tracking."""

    @staticmethod
    def generate_config_hash(config_data: dict[str, Any]) -> str:
        """Generate a SHA-256 hash for configuration data.

        Args:
            config_data: Configuration dictionary to hash

        Returns:
            str: SHA-256 hash of the configuration
        """

        # Convert non-JSON-serializable objects for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list | tuple):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, set):
                return sorted(obj)  # Convert sets to sorted lists
            elif hasattr(obj, "__fspath__"):  # Path-like objects
                return str(obj)
            elif hasattr(obj, "value"):  # Enum objects
                return obj.value
            else:
                return obj

        serializable_data = convert_paths(config_data)
        # Sort keys for consistent hashing
        sorted_json = json.dumps(
            serializable_data, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(sorted_json.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def create_version_metadata(
        config_hash: str,
        template_source: str | None = None,
        migration_version: str | None = None,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Create version metadata for configuration tracking.

        Args:
            config_hash: Configuration hash identifier
            template_source: Source template name if applicable
            migration_version: Schema migration version
            environment: Target environment

        Returns:
            Dict containing version metadata
        """
        return {
            "config_version": "1.0.0",
            "config_hash": config_hash,
            "created_at": datetime.utcnow().isoformat(),
            "template_source": template_source,
            "migration_version": migration_version or "1.0.0",
            "environment": environment,
            "schema_version": "2025.1.0",  # Schema version for compatibility tracking
        }


class ConfigPathManager:
    """Utilities for managing configuration file paths and directories."""

    def __init__(self, base_dir: Path = Path("config")):
        """Initialize path manager with base configuration directory.

        Args:
            base_dir: Base directory for configuration files
        """
        self.base_dir = Path(base_dir)
        self.templates_dir = self.base_dir / "templates"
        self.backups_dir = self.base_dir / "backups"
        self.migrations_dir = self.base_dir / "migrations"

    def ensure_directories(self) -> None:
        """Create all necessary configuration directories."""
        for directory in [
            self.base_dir,
            self.templates_dir,
            self.backups_dir,
            self.migrations_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_config_path(self, name: str, format: str = "json") -> Path:
        """Get path for a configuration file.

        Args:
            name: Configuration name
            format: File format (json, yaml, toml)

        Returns:
            Path to configuration file
        """
        return self.base_dir / f"{name}.{format}"

    def get_template_path(self, template_name: str) -> Path:
        """Get path for a template file.

        Args:
            template_name: Template name

        Returns:
            Path to template file
        """
        return self.templates_dir / f"{template_name}.json"

    def get_backup_path(self, config_name: str, timestamp: str) -> Path:
        """Get path for a backup file.

        Args:
            config_name: Configuration name
            timestamp: Backup timestamp

        Returns:
            Path to backup file
        """
        return self.backups_dir / f"{config_name}_{timestamp}.json"

    def list_backups(self, config_name: str | None = None) -> list[Path]:
        """List available backup files.

        Args:
            config_name: Filter by configuration name (optional)

        Returns:
            List of backup file paths
        """
        pattern = f"{config_name}_*.json" if config_name else "*.json"

        return sorted(self.backups_dir.glob(pattern), reverse=True)


class ConfigMerger:
    """Utilities for merging configuration dictionaries with conflict resolution."""

    @staticmethod
    def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigMerger.deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def apply_environment_overrides(
        config: dict[str, Any], environment: str, overrides: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply environment-specific overrides to configuration.

        Args:
            config: Base configuration
            environment: Environment name
            overrides: Environment-specific overrides

        Returns:
            Configuration with environment overrides applied
        """
        if environment in overrides:
            return ConfigMerger.deep_merge(config, overrides[environment])
        return config


class ValidationHelper:
    """Helper utilities for configuration validation."""

    @staticmethod
    def validate_url_format(url: str) -> bool:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if URL format is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False

        # Basic URL validation - accept common schemes
        valid_schemes = (
            "http://",
            "https://",
            "redis://",
            "postgresql://",
            "sqlite://",
        )
        return url.startswith(valid_schemes)

    @staticmethod
    def suggest_fix_for_error(error_message: str, field_name: str) -> str | None:
        """Suggest potential fixes for common validation errors.

        Args:
            error_message: Validation error message
            field_name: Field that failed validation

        Returns:
            Suggested fix or None if no suggestion available
        """
        error_lower = error_message.lower()

        # Common error patterns and suggestions
        if "api_key" in field_name.lower():
            if "none" in error_lower or "required" in error_lower:
                return f"Set {field_name} environment variable or add to config file"
            elif "invalid" in error_lower or "format" in error_lower:
                return f"Check {field_name} format - should start with expected prefix"

        if "url" in field_name.lower() and "invalid" in error_lower:
            return f"Ensure {field_name} includes protocol (http:// or https://)"

        if "port" in field_name.lower() and "range" in error_lower:
            return "Port must be between 1 and 65535"

        if (
            "directory" in field_name.lower() or "path" in field_name.lower()
        ) and "exist" in error_lower:
            return f"Create directory: mkdir -p {field_name}"

        return None

    @staticmethod
    def categorize_validation_error(error_message: str) -> str:
        """Categorize validation errors by type.

        Args:
            error_message: Validation error message

        Returns:
            Error category (syntax, business_rule, environment, etc.)
        """
        error_lower = error_message.lower()

        if any(word in error_lower for word in ["required", "missing", "none"]):
            return "missing_required"
        elif any(word in error_lower for word in ["format", "invalid", "pattern"]):
            return "format_error"
        elif any(
            word in error_lower for word in ["range", "greater", "less", "between"]
        ):
            return "value_range"
        elif any(word in error_lower for word in ["production", "environment"]):
            return "environment_constraint"
        else:
            return "business_rule"


def generate_timestamp() -> str:
    """Generate a timestamp string for file naming.

    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in file paths.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized name safe for file paths
    """
    # Replace spaces and special characters with underscores
    import re

    return re.sub(r"[^\w\-.]", "_", name).lower()
