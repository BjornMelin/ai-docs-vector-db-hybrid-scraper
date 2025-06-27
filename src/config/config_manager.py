"""Configuration manager using pydantic-settings built-in features.

This module provides a configuration manager with:
- Custom settings sources for dynamic configuration updates
- Built-in SecretStr for sensitive data
- Native validation and type conversion
- Watchdog integration for file monitoring
- Simple drift detection using configuration hashing
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Type

from pydantic import Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .core import Config, FirecrawlConfig, OpenAIConfig


logger = logging.getLogger(__name__)


class ConfigFileSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads from JSON/YAML/TOML files with hot-reload support."""

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        config_file: Path | None = None,
        config_data: dict[str, Any] | None = None,
    ):
        super().__init__(settings_cls)
        self.config_file = config_file
        self._config_data = config_data or {}

        # Load config file if provided
        if self.config_file and self.config_file.exists():
            self._load_config_file()

    def _load_config_file(self) -> None:
        """Load configuration from file based on extension."""
        if not self.config_file or not self.config_file.exists():
            return

        try:
            suffix = self.config_file.suffix.lower()

            if suffix == ".json":
                with open(self.config_file) as f:
                    self._config_data = json.load(f)

            elif suffix in [".yaml", ".yml"]:
                import yaml

                with open(self.config_file) as f:
                    self._config_data = yaml.safe_load(f) or {}

            elif suffix == ".toml":
                import tomli

                with open(self.config_file, "rb") as f:
                    self._config_data = tomli.load(f)

            else:
                logger.warning(f"Unsupported config file format: {suffix}")

        except Exception as e:
            logger.exception(f"Failed to load config file {self.config_file}: {e}")

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from config data."""
        field_value = self._config_data.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """Prepare field value - no special preparation needed."""
        return value

    def __call__(self) -> dict[str, Any]:
        """Return all config data."""
        return self._config_data.copy()


class OpenAIConfigSecure(OpenAIConfig):
    """OpenAI config with SecretStr for API key."""

    api_key: SecretStr | None = Field(default=None)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr | None) -> SecretStr | None:
        if v and not v.get_secret_value().startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class FirecrawlConfigSecure(FirecrawlConfig):
    """Firecrawl config with SecretStr for API key."""

    api_key: SecretStr | None = Field(default=None)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr | None) -> SecretStr | None:
        if v and not v.get_secret_value().startswith("fc-"):
            raise ValueError("Firecrawl API key must start with 'fc-'")
        return v


class SecureConfig(Config):
    """Config with built-in SecretStr for sensitive fields."""

    # Override with secure versions
    openai: OpenAIConfigSecure = Field(default_factory=OpenAIConfigSecure)
    firecrawl: FirecrawlConfigSecure = Field(default_factory=FirecrawlConfigSecure)


class ConfigFileWatcher(FileSystemEventHandler):
    """Watchdog handler for configuration file changes."""

    def __init__(self, config_file: Path, reload_callback: Callable[[], None]):
        self.config_file = config_file
        self.reload_callback = reload_callback
        self._last_modified = None

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if our config file was modified
        if Path(event.src_path).resolve() == self.config_file.resolve():
            # Debounce rapid changes
            current_time = datetime.now(UTC)
            if (
                self._last_modified
                and (current_time - self._last_modified).total_seconds() < 1
            ):
                return

            self._last_modified = current_time
            logger.info(f"Configuration file changed: {self.config_file}")

            try:
                self.reload_callback()
            except Exception as e:
                logger.exception(f"Failed to reload configuration: {e}")


class ConfigManager:
    """Configuration manager using pydantic-settings features.

    This provides:
    - Uses pydantic-settings custom sources for loading
    - Integrates watchdog for file monitoring
    - Provides simple drift detection via hashing
    - Supports hot-reload via __init__ re-invocation
    """

    def __init__(
        self,
        config_class: Type[BaseSettings] = SecureConfig,
        config_file: Path | None = None,
        enable_file_watching: bool = True,
        enable_drift_detection: bool = True,
    ):
        """Initialize configuration manager.

        Args:
            config_class: Configuration class (defaults to SecureConfig)
            config_file: Optional configuration file path
            enable_file_watching: Enable automatic file watching
            enable_drift_detection: Enable configuration drift detection
        """
        self.config_class = config_class
        self.config_file = config_file or Path(".env")
        self.enable_file_watching = enable_file_watching
        self.enable_drift_detection = enable_drift_detection

        # Current configuration instance
        self._config: BaseSettings | None = None
        self._config_hash: str | None = None

        # File watcher
        self._observer: Observer | None = None

        # Change listeners
        self._change_listeners: list[Callable[[BaseSettings, BaseSettings], None]] = []

        # Drift detection
        self._baseline_hash: str | None = None

        # Load initial configuration
        self.reload_config()

        # Start file watching if enabled
        if self.enable_file_watching and self.config_file.exists():
            self._start_file_watching()

    def _calculate_config_hash(self, config: BaseSettings) -> str:
        """Calculate hash of configuration for change/drift detection."""
        # Convert to dict, excluding SecretStr values for security
        config_dict = config.model_dump()

        # Mask sensitive values
        self._mask_secrets(config_dict)

        # Create stable JSON representation
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        # Calculate SHA256 hash
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def _mask_secrets(self, data: dict[str, Any]) -> None:
        """Recursively mask SecretStr values in dictionary."""
        import hashlib

        for key, value in data.items():
            if isinstance(value, dict):
                self._mask_secrets(value)
            elif key.lower() in ["api_key", "token", "password", "secret"]:
                # Include a hash of the secret value for change detection
                if value:
                    value_hash = hashlib.sha256(str(value).encode()).hexdigest()[:8]
                    data[key] = f"***MASKED_{value_hash}***"
                else:
                    data[key] = "***MASKED_EMPTY***"

    def reload_config(self) -> bool:
        """Reload configuration using pydantic-settings __init__ pattern.

        Returns:
            True if configuration was successfully reloaded and changed
        """
        try:
            old_config = self._config
            old_hash = self._config_hash

            # Create new configuration instance with custom source
            if self.config_file.exists() and self.config_file.suffix in [
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            ]:
                # Use custom file source for structured config files
                self._config = self._create_config_with_file_source()
            else:
                # Use standard pydantic-settings loading for .env files
                self._config = self.config_class()

            # Calculate new hash
            self._config_hash = self._calculate_config_hash(self._config)

            # Set baseline hash if not set
            if self._baseline_hash is None:
                self._baseline_hash = self._config_hash

            # Check if configuration actually changed
            if old_hash and old_hash == self._config_hash:
                logger.debug("Configuration unchanged after reload")
                return False

            # Notify change listeners
            if old_config:
                self._notify_change_listeners(old_config, self._config)

            logger.info(
                f"Configuration reloaded successfully (hash: {self._config_hash})"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to reload configuration: {e}")
            return False

    def _create_config_with_file_source(self) -> BaseSettings:
        """Create configuration with custom file source."""
        # This is a simplified version - in practice, you'd properly integrate
        # the custom source with pydantic-settings
        file_source = ConfigFileSettingsSource(self.config_class, self.config_file)

        # For now, just merge file data with environment
        config_data = file_source()

        # Create config instance with merged data
        return self.config_class(**config_data)

    def _start_file_watching(self) -> None:
        """Start watching configuration file for changes."""
        try:
            self._observer = Observer()

            # Watch the directory containing the config file
            watch_dir = self.config_file.parent
            event_handler = ConfigFileWatcher(self.config_file, self.reload_config)

            self._observer.schedule(event_handler, str(watch_dir), recursive=False)
            self._observer.start()

            logger.info(f"Started watching configuration file: {self.config_file}")

        except Exception as e:
            logger.exception(f"Failed to start file watching: {e}")

    def stop_file_watching(self) -> None:
        """Stop watching configuration file."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Stopped configuration file watching")

    def add_change_listener(
        self, callback: Callable[[BaseSettings, BaseSettings], None]
    ) -> None:
        """Add a configuration change listener.

        Args:
            callback: Function called with (old_config, new_config) on changes
        """
        self._change_listeners.append(callback)

    def remove_change_listener(
        self, callback: Callable[[BaseSettings, BaseSettings], None]
    ) -> bool:
        """Remove a configuration change listener.

        Returns:
            True if listener was found and removed
        """
        try:
            self._change_listeners.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_change_listeners(
        self, old_config: BaseSettings, new_config: BaseSettings
    ) -> None:
        """Notify all change listeners about configuration update."""
        for listener in self._change_listeners:
            try:
                listener(old_config, new_config)
            except Exception as e:
                logger.exception(f"Change listener failed: {e}")

    def get_config(self) -> BaseSettings:
        """Get current configuration instance."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def check_drift(self) -> dict[str, Any] | None:
        """Check for configuration drift from baseline.

        Returns:
            Drift information if drift detected, None otherwise
        """
        if not self.enable_drift_detection:
            return None

        if self._baseline_hash is None:
            return None

        current_hash = self._calculate_config_hash(self._config)

        if current_hash != self._baseline_hash:
            return {
                "drift_detected": True,
                "baseline_hash": self._baseline_hash,
                "current_hash": current_hash,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        return None

    def update_baseline(self) -> None:
        """Update drift detection baseline to current configuration."""
        if self._config:
            self._baseline_hash = self._calculate_config_hash(self._config)
            logger.info(f"Updated configuration baseline (hash: {self._baseline_hash})")

    def get_config_info(self) -> dict[str, Any]:
        """Get configuration information and statistics."""
        return {
            "config_file": str(self.config_file),
            "config_exists": self.config_file.exists(),
            "current_hash": self._config_hash,
            "baseline_hash": self._baseline_hash,
            "file_watching_enabled": self.enable_file_watching,
            "drift_detection_enabled": self.enable_drift_detection,
            "change_listeners_count": len(self._change_listeners),
            "drift_status": self.check_drift(),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop file watching."""
        self.stop_file_watching()


# Async helper for creating and loading configuration
async def create_and_load_config_async(
    config_file: Path | None = None,
    config_class: Type[BaseSettings] = SecureConfig,
) -> tuple[ConfigManager, BaseSettings]:
    """Create and load configuration asynchronously.

    Args:
        config_file: Optional configuration file path
        config_class: Configuration class to use

    Returns:
        Tuple of (manager, config) instances
    """
    manager = ConfigManager(
        config_class=config_class,
        config_file=config_file,
    )

    # In a real async implementation, we might load from async sources
    await asyncio.sleep(0)  # Yield control

    return manager, manager.get_config()


# Global config manager instance
_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager


def set_config_manager(manager: ConfigManager) -> None:
    """Set the global configuration manager instance."""
    global _manager
    _manager = manager
