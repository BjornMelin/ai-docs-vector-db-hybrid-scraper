"""Configuration migration utilities for upgrading between versions.

This module provides tools to migrate configurations between different
versions of the unified configuration system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import ClassVar

from .models import UnifiedConfig


class ConfigMigrator:
    """Handles migration of configurations between versions."""

    # Configuration version history
    VERSIONS: ClassVar[dict[str, str]] = {
        "0.1.0": "Initial unified configuration",
        "0.2.0": "Added cache configuration enhancements",
        "0.3.0": "Added security and performance settings",
    }

    @staticmethod
    def detect_config_version(config_data: dict[str, Any]) -> str | None:
        """Detect the version of a configuration.

        Args:
            config_data: Configuration dictionary

        Returns:
            Version string or None if unknown
        """
        # Check for version field
        if "version" in config_data:
            return config_data["version"]

        # Check for version indicators
        if "cache" in config_data and "redis_pool_size" in config_data.get("cache", {}):
            return "0.2.0"
        elif "security" in config_data:
            return "0.3.0"
        elif "unified_config" in config_data:
            return "0.1.0"

        # Legacy configuration (pre-unified)
        if "sites" in config_data or "settings" in config_data:
            return "legacy"

        return None

    @staticmethod
    def migrate_legacy_to_unified(legacy_data: dict[str, Any]) -> dict[str, Any]:
        """Migrate legacy configuration to unified format.

        Args:
            legacy_data: Legacy configuration data

        Returns:
            Unified configuration dictionary
        """
        unified = {
            "version": "0.3.0",
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
        }

        # Migrate documentation sites
        if "sites" in legacy_data:
            unified["documentation_sites"] = []
            for site in legacy_data["sites"]:
                unified["documentation_sites"].append(
                    {
                        "name": site.get("name", ""),
                        "url": site.get("url", ""),
                        "max_pages": site.get("max_pages", 50),
                        "priority": site.get("priority", "medium"),
                        "description": site.get("description", ""),
                    }
                )

        # Migrate settings
        if "settings" in legacy_data:
            settings = legacy_data["settings"]

            # Embedding settings
            if "embedding_model" in settings:
                model = settings["embedding_model"]
                if "openai" in model or "ada" in model:
                    unified["embedding_provider"] = "openai"
                    unified["openai"] = {"model": model}
                else:
                    unified["embedding_provider"] = "fastembed"

            # Collection name
            if "collection_name" in settings:
                unified["qdrant"] = {"collection_name": settings["collection_name"]}

            # Chunk size
            if "chunk_size" in settings:
                unified["chunking"] = {"chunk_size": settings["chunk_size"]}

            # Concurrent crawls
            if "max_concurrent_crawls" in settings:
                if "crawl4ai" not in unified:
                    unified["crawl4ai"] = {}
                unified["crawl4ai"]["max_concurrent_crawls"] = settings[
                    "max_concurrent_crawls"
                ]

        return unified

    @staticmethod
    def migrate_between_versions(
        config_data: dict[str, Any], from_version: str, to_version: str
    ) -> dict[str, Any]:
        """Migrate configuration between specific versions.

        Args:
            config_data: Configuration data
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated configuration dictionary
        """
        # Create a copy to avoid modifying original
        migrated = config_data.copy()

        # Apply migrations in sequence
        if from_version == "0.1.0" and to_version >= "0.2.0":
            # Add cache enhancements
            if "cache" not in migrated:
                migrated["cache"] = {}
            cache = migrated["cache"]
            if "redis_pool_size" not in cache:
                cache["redis_pool_size"] = 10
            if "redis_password" not in cache:
                cache["redis_password"] = None
            if "redis_ssl" not in cache:
                cache["redis_ssl"] = False

        if from_version <= "0.2.0" and to_version >= "0.3.0":
            # Add security and performance settings
            if "security" not in migrated:
                migrated["security"] = {
                    "require_api_keys": True,
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 100,
                    "allowed_domains": [],
                    "blocked_domains": [],
                }

            if "performance" not in migrated:
                migrated["performance"] = {
                    "max_concurrent_requests": 10,
                    "request_timeout": 30.0,
                    "max_retries": 3,
                    "retry_base_delay": 1.0,
                    "retry_max_delay": 60.0,
                    "max_memory_usage_mb": 1000.0,
                    "gc_threshold": 0.8,
                }

        # Update version
        migrated["version"] = to_version

        return migrated

    @staticmethod
    def create_migration_report(  # noqa: PLR0912
        original: dict[str, Any],
        migrated: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> str:
        """Create a detailed migration report.

        Args:
            original: Original configuration
            migrated: Migrated configuration
            from_version: Source version
            to_version: Target version

        Returns:
            Formatted migration report
        """
        report = []
        report.append("=" * 60)
        report.append("Configuration Migration Report")
        report.append("=" * 60)
        report.append(f"Migration Date: {datetime.now().isoformat()}")
        report.append(f"From Version: {from_version}")
        report.append(f"To Version: {to_version}")
        report.append("")

        # Find added fields
        report.append("Added Fields:")
        added = ConfigMigrator._find_added_fields(original, migrated)
        if added:
            for field in added:
                report.append(f"  + {field}")
        else:
            report.append("  (none)")
        report.append("")

        # Find removed fields
        report.append("Removed Fields:")
        removed = ConfigMigrator._find_removed_fields(original, migrated)
        if removed:
            for field in removed:
                report.append(f"  - {field}")
        else:
            report.append("  (none)")
        report.append("")

        # Find modified fields
        report.append("Modified Fields:")
        modified = ConfigMigrator._find_modified_fields(original, migrated)
        if modified:
            for field, (old_val, new_val) in modified.items():
                report.append(f"  * {field}: {old_val} → {new_val}")
        else:
            report.append("  (none)")
        report.append("")

        # Validation status
        report.append("Validation Status:")
        try:
            config = UnifiedConfig(**migrated)
            issues = config.validate_completeness()
            if issues:
                report.append("  ⚠️  Validation issues found:")
                for issue in issues:
                    report.append(f"     - {issue}")
            else:
                report.append("  ✅ Configuration is valid")
        except Exception as e:
            report.append(f"  ❌ Validation failed: {e}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    @staticmethod
    def _find_added_fields(
        original: dict, migrated: dict, prefix: str = ""
    ) -> list[str]:
        """Find fields added in migration."""
        added = []
        for key, value in migrated.items():
            full_key = f"{prefix}{key}" if prefix else key
            if key not in original:
                added.append(full_key)
            elif isinstance(value, dict) and isinstance(original.get(key), dict):
                added.extend(
                    ConfigMigrator._find_added_fields(
                        original[key], value, f"{full_key}."
                    )
                )
        return added

    @staticmethod
    def _find_removed_fields(
        original: dict, migrated: dict, prefix: str = ""
    ) -> list[str]:
        """Find fields removed in migration."""
        removed = []
        for key, value in original.items():
            full_key = f"{prefix}{key}" if prefix else key
            if key not in migrated:
                removed.append(full_key)
            elif isinstance(value, dict) and isinstance(migrated.get(key), dict):
                removed.extend(
                    ConfigMigrator._find_removed_fields(
                        value, migrated[key], f"{full_key}."
                    )
                )
        return removed

    @staticmethod
    def _find_modified_fields(
        original: dict, migrated: dict, prefix: str = ""
    ) -> dict[str, tuple[Any, Any]]:
        """Find fields with changed values."""
        modified = {}
        for key in set(original.keys()) & set(migrated.keys()):
            full_key = f"{prefix}{key}" if prefix else key
            orig_val = original[key]
            new_val = migrated[key]

            if isinstance(orig_val, dict) and isinstance(new_val, dict):
                # Recurse into nested dicts
                modified.update(
                    ConfigMigrator._find_modified_fields(
                        orig_val, new_val, f"{full_key}."
                    )
                )
            elif orig_val != new_val:
                modified[full_key] = (orig_val, new_val)

        return modified

    @staticmethod
    def auto_migrate(
        config_path: Path | str, target_version: str = "0.3.0", backup: bool = True
    ) -> tuple[bool, str]:
        """Automatically migrate a configuration file.

        Args:
            config_path: Path to configuration file
            target_version: Target version to migrate to
            backup: Whether to create a backup

        Returns:
            Tuple of (success, message)
        """
        config_path = Path(config_path)

        if not config_path.exists():
            return False, f"Configuration file not found: {config_path}"

        try:
            # Load configuration
            with open(config_path) as f:
                if config_path.suffix == ".json":
                    config_data = json.load(f)
                else:
                    return False, f"Unsupported file format: {config_path.suffix}"

            # Detect version
            from_version = ConfigMigrator.detect_config_version(config_data)
            if from_version is None:
                return False, "Could not detect configuration version"

            if from_version == target_version:
                return True, f"Configuration already at version {target_version}"

            # Create backup if requested
            if backup:
                backup_path = config_path.with_suffix(
                    f".backup-{datetime.now():%Y%m%d-%H%M%S}"
                )
                with open(backup_path, "w") as f:
                    json.dump(config_data, f, indent=2)

            # Perform migration
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

            # Save migrated configuration
            with open(config_path, "w") as f:
                json.dump(migrated, f, indent=2)

            # Save report
            report_path = config_path.with_suffix(".migration-report.txt")
            report_path.write_text(report)

            return (
                True,
                f"Successfully migrated from {from_version} to {target_version}\nReport saved to: {report_path}",
            )

        except Exception as e:
            return False, f"Migration failed: {e}"
