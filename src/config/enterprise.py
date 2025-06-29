"""Enterprise Configuration Management System.

This module provides unified configuration management for all enterprise features,
implementing centralized configuration orchestration with change detection,
validation, and propagation across all services.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.config.modern import Config as BaseConfig
from src.services.enterprise.cache import EnterpriseCacheService
from src.services.enterprise.search import EnterpriseSearchService


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ConfigSourceType(str, Enum):
    """Configuration source types."""

    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    VAULT = "vault"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"


class ConfigChangeType(str, Enum):
    """Configuration change types."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RELOADED = "reloaded"


@dataclass
class ConfigChange:
    """Represents a configuration change event."""

    change_type: ConfigChangeType
    source: str
    key: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""

    def __init__(self, source_type: ConfigSourceType, priority: int = 0):
        self.source_type = source_type
        self.priority = priority
        self.is_watching = False

    @abstractmethod
    async def load_config(self) -> dict[str, Any]:
        """Load configuration from source."""

    @abstractmethod
    async def watch_changes(self, callback: Callable[[ConfigChange], None]) -> None:
        """Watch for configuration changes."""

    @abstractmethod
    async def stop_watching(self) -> None:
        """Stop watching for changes."""


class EnvironmentConfigSource(ConfigSource):
    """Environment variable configuration source."""

    def __init__(self, prefix: str = "ENTERPRISE_", priority: int = 100):
        super().__init__(ConfigSourceType.ENVIRONMENT, priority)
        self.prefix = prefix
        self.last_env_snapshot: dict[str, str] = {}

    async def load_config(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix) :].lower().replace("_", ".")

                # Try to parse as JSON first, then as string
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value

        self.last_env_snapshot = dict(os.environ)
        return config

    async def watch_changes(self, callback: Callable[[ConfigChange], None]) -> None:
        """Watch for environment variable changes."""
        # Environment variables don't change during runtime typically
        # This is a placeholder for systems that support env var watching
        self.is_watching = True

    async def stop_watching(self) -> None:
        """Stop watching for changes."""
        self.is_watching = False


class FileConfigSource(ConfigSource):
    """File-based configuration source (YAML/JSON)."""

    def __init__(self, file_path: str, priority: int = 50):
        super().__init__(ConfigSourceType.FILE, priority)
        self.file_path = Path(file_path)
        self.observer: Observer | None = None
        self.callback: Callable[[ConfigChange], None] | None = None

    async def load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            logger.warning(f"Configuration file {self.file_path} does not exist")
            return {}

        try:
            with open(self.file_path, encoding="utf-8") as f:
                if self.file_path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                if self.file_path.suffix.lower() == ".json":
                    return json.load(f)
                logger.warning(f"Unsupported file format: {self.file_path.suffix}")
                return {}

        except Exception as e:
            logger.exception(f"Failed to load config from {self.file_path}: {e}")
            return {}

    async def watch_changes(self, callback: Callable[[ConfigChange], None]) -> None:
        """Watch for file changes."""
        self.callback = callback

        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, source: FileConfigSource):
                self.source = source

            def on_modified(self, event):
                if (
                    not event.is_directory
                    and Path(event.src_path) == self.source.file_path
                ):
                    if self.source.callback:
                        change = ConfigChange(
                            change_type=ConfigChangeType.MODIFIED,
                            source=str(self.source.file_path),
                            key="file_changed",
                            metadata={"file_path": str(self.source.file_path)},
                        )
                        # Run callback in async context
                        asyncio.create_task(self._async_callback(change))

            async def _async_callback(self, change: ConfigChange):
                try:
                    self.source.callback(change)
                except Exception as e:
                    logger.exception(f"Error in config change callback: {e}")

        self.observer = Observer()
        self.observer.schedule(
            ConfigFileHandler(self), str(self.file_path.parent), recursive=False
        )
        self.observer.start()
        self.is_watching = True

    async def stop_watching(self) -> None:
        """Stop watching for changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        self.is_watching = False


class DatabaseConfigSource(ConfigSource):
    """Database-based configuration source."""

    def __init__(
        self, connection_string: str, table_name: str = "config", priority: int = 75
    ):
        super().__init__(ConfigSourceType.DATABASE, priority)
        self.connection_string = connection_string
        self.table_name = table_name

    async def load_config(self) -> dict[str, Any]:
        """Load configuration from database."""
        # Placeholder implementation
        logger.info("Database config source not yet implemented")
        return {}

    async def watch_changes(self, callback: Callable[[ConfigChange], None]) -> None:
        """Watch for database changes."""
        # Placeholder implementation
        self.is_watching = True

    async def stop_watching(self) -> None:
        """Stop watching for changes."""
        self.is_watching = False


class VaultConfigSource(ConfigSource):
    """HashiCorp Vault configuration source for secrets."""

    def __init__(self, vault_url: str, token: str, path: str, priority: int = 90):
        super().__init__(ConfigSourceType.VAULT, priority)
        self.vault_url = vault_url
        self.token = token
        self.path = path

    async def load_config(self) -> dict[str, Any]:
        """Load configuration from Vault."""
        # Placeholder implementation
        logger.info("Vault config source not yet implemented")
        return {}

    async def watch_changes(self, callback: Callable[[ConfigChange], None]) -> None:
        """Watch for Vault changes."""
        # Placeholder implementation
        self.is_watching = True

    async def stop_watching(self) -> None:
        """Stop watching for changes."""
        self.is_watching = False


# Enterprise Configuration Models


class EnterpriseCacheConfig(BaseModel):
    """Enterprise cache configuration."""

    enabled: bool = True
    max_size: int = 10000
    max_memory_mb: int = 1000
    default_ttl: int = 3600
    enable_compression: bool = True
    enable_analytics: bool = True
    enable_distributed: bool = True
    redis_url: str | None = None

    @validator("max_size")
    def validate_max_size(cls, v):
        if v <= 0:
            raise ValueError("max_size must be positive")
        return v


class EnterpriseSearchConfig(BaseModel):
    """Enterprise search configuration."""

    enabled: bool = True
    max_results: int = 1000
    enable_reranking: bool = True
    enable_hybrid_search: bool = True
    enable_query_expansion: bool = True
    enable_personalization: bool = True
    enable_analytics: bool = True
    enable_ab_testing: bool = True

    @validator("max_results")
    def validate_max_results(cls, v):
        if v <= 0 or v > 10000:
            raise ValueError("max_results must be between 1 and 10000")
        return v


class SecurityFrameworkConfig(BaseModel):
    """Security framework configuration."""

    enabled: bool = True
    api_key_required: bool = False
    api_keys: list[str] = Field(default_factory=list)
    allowed_origins: list[str] = Field(default_factory=list)
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    enable_threat_detection: bool = True
    enable_audit_logging: bool = True

    @validator("api_keys")
    def validate_api_keys(cls, v):
        if any(len(key) < 32 for key in v):
            raise ValueError("API keys must be at least 32 characters long")
        return v


class DeploymentFrameworkConfig(BaseModel):
    """Deployment framework configuration."""

    strategy: str = "blue_green"  # blue_green, canary, rolling
    health_check_endpoint: str = "/health"
    health_check_timeout: int = 30
    health_check_retries: int = 3
    enable_automatic_rollback: bool = True
    max_deployment_time_minutes: int = 30

    @validator("strategy")
    def validate_strategy(cls, v):
        if v not in ["blue_green", "canary", "rolling"]:
            raise ValueError("strategy must be one of: blue_green, canary, rolling")
        return v


class ObservabilityFrameworkConfig(BaseModel):
    """Observability framework configuration."""

    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_enabled: bool = True
    alerting_enabled: bool = True
    retention_days: int = 30
    sample_rate: float = 1.0

    @validator("sample_rate")
    def validate_sample_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        return v


class EnterpriseConfig(BaseModel):
    """Complete enterprise configuration schema."""

    # Service Configuration
    cache: EnterpriseCacheConfig = Field(default_factory=EnterpriseCacheConfig)
    search: EnterpriseSearchConfig = Field(default_factory=EnterpriseSearchConfig)
    security: SecurityFrameworkConfig = Field(default_factory=SecurityFrameworkConfig)
    deployment: DeploymentFrameworkConfig = Field(
        default_factory=DeploymentFrameworkConfig
    )
    observability: ObservabilityFrameworkConfig = Field(
        default_factory=ObservabilityFrameworkConfig
    )

    # Global Configuration
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"

    # Metadata
    config_version: str = "1.0.0"
    last_updated: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    class Config:
        extra = "allow"  # Allow additional configuration fields

    def validate_enterprise_requirements(self) -> dict[str, list[str]]:
        """Validate configuration meets enterprise standards."""
        issues = {}

        # Validate security requirements
        if self.environment == "production":
            security_issues = []

            if not self.security.enabled:
                security_issues.append(
                    "Security framework must be enabled in production"
                )

            if not self.security.api_key_required and not self.security.allowed_origins:
                security_issues.append(
                    "Either API keys or CORS origins must be configured"
                )

            if not self.security.rate_limiting_enabled:
                security_issues.append("Rate limiting should be enabled in production")

            if security_issues:
                issues["security"] = security_issues

        # Validate observability requirements
        if self.environment == "production":
            observability_issues = []

            if not self.observability.enabled:
                observability_issues.append(
                    "Observability must be enabled in production"
                )

            if not self.observability.alerting_enabled:
                observability_issues.append("Alerting should be enabled in production")

            if observability_issues:
                issues["observability"] = observability_issues

        return issues

    def generate_service_configs(self) -> dict[str, dict[str, Any]]:
        """Generate service-specific configurations."""
        return {
            "enterprise_cache": self.cache.dict(),
            "enterprise_search": self.search.dict(),
            "security_framework": self.security.dict(),
            "deployment_framework": self.deployment.dict(),
            "observability_framework": self.observability.dict(),
        }


class ConfigWatcher:
    """Watches for configuration changes and notifies handlers."""

    def __init__(self, config_key: str):
        self.config_key = config_key
        self.handlers: list[Callable[[ConfigChange], None]] = []

    def add_handler(self, handler: Callable[[ConfigChange], None]) -> None:
        """Add change handler."""
        self.handlers.append(handler)

    def remove_handler(self, handler: Callable[[ConfigChange], None]) -> None:
        """Remove change handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    async def notify_change(self, change: ConfigChange) -> None:
        """Notify all handlers of configuration change."""
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(change)
                else:
                    handler(change)
            except Exception as e:
                logger.exception(f"Error in config change handler: {e}")


class EnterpriseConfigurationOrchestrator:
    """Unified configuration management for all enterprise features."""

    def __init__(self):
        self.config_sources: list[ConfigSource] = []
        self.watchers: dict[str, ConfigWatcher] = {}
        self.current_config: EnterpriseConfig | None = None
        self.config_history: list[dict[str, Any]] = []

        # State management
        self.is_watching = False
        self.watch_tasks: list[asyncio.Task] = []

        logger.info("Enterprise configuration orchestrator initialized")

    def add_config_source(self, source: ConfigSource) -> None:
        """Add configuration source."""
        self.config_sources.append(source)
        # Sort by priority (higher priority first)
        self.config_sources.sort(key=lambda s: s.priority, reverse=True)
        logger.info(
            f"Added config source: {source.source_type.value} (priority: {source.priority})"
        )

    async def load_enterprise_config(self) -> EnterpriseConfig:
        """Load complete enterprise configuration from all sources."""
        merged_config = {}

        # Load from all sources (low priority to high priority)
        for source in reversed(self.config_sources):
            try:
                source_config = await source.load_config()
                merged_config = self._deep_merge(merged_config, source_config)
                logger.debug(f"Loaded config from {source.source_type.value}")
            except Exception as e:
                logger.exception(
                    f"Failed to load config from {source.source_type.value}: {e}"
                )

        # Create enterprise config with validation
        try:
            self.current_config = EnterpriseConfig(**merged_config)

            # Validate enterprise requirements
            validation_issues = self.current_config.validate_enterprise_requirements()
            if validation_issues:
                logger.warning(f"Configuration validation issues: {validation_issues}")

            # Store in history
            self.config_history.append(merged_config)
            if len(self.config_history) > 10:  # Keep last 10 configurations
                self.config_history.pop(0)

            logger.info("Enterprise configuration loaded successfully")
            return self.current_config

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    async def watch_configuration_changes(self) -> None:
        """Monitor configuration changes and notify services."""
        if self.is_watching:
            return

        self.is_watching = True

        # Start watching all sources
        for source in self.config_sources:
            try:
                task = asyncio.create_task(
                    source.watch_changes(self._handle_config_change)
                )
                self.watch_tasks.append(task)
                logger.info(f"Started watching {source.source_type.value}")
            except Exception as e:
                logger.exception(
                    f"Failed to start watching {source.source_type.value}: {e}"
                )

    async def stop_watching(self) -> None:
        """Stop watching configuration changes."""
        if not self.is_watching:
            return

        self.is_watching = False

        # Stop all watch tasks
        for task in self.watch_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.watch_tasks.clear()

        # Stop all sources
        for source in self.config_sources:
            try:
                await source.stop_watching()
            except Exception as e:
                logger.exception(
                    f"Error stopping watch for {source.source_type.value}: {e}"
                )

    def add_change_handler(
        self, config_key: str, handler: Callable[[ConfigChange], None]
    ) -> None:
        """Add handler for specific configuration changes."""
        if config_key not in self.watchers:
            self.watchers[config_key] = ConfigWatcher(config_key)

        self.watchers[config_key].add_handler(handler)
        logger.info(f"Added change handler for {config_key}")

    def remove_change_handler(
        self, config_key: str, handler: Callable[[ConfigChange], None]
    ) -> None:
        """Remove change handler."""
        if config_key in self.watchers:
            self.watchers[config_key].remove_handler(handler)

    async def reload_configuration(self) -> EnterpriseConfig:
        """Force reload configuration from all sources."""
        logger.info("Reloading enterprise configuration")

        old_config = self.current_config
        new_config = await self.load_enterprise_config()

        # Notify about reload
        change = ConfigChange(
            change_type=ConfigChangeType.RELOADED,
            source="orchestrator",
            key="enterprise_config",
            old_value=old_config.dict() if old_config else None,
            new_value=new_config.dict(),
        )

        await self._notify_watchers("enterprise_config", change)

        return new_config

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary and status."""
        return {
            "config_loaded": self.current_config is not None,
            "config_version": self.current_config.config_version
            if self.current_config
            else None,
            "last_updated": self.current_config.last_updated.isoformat()
            if self.current_config
            else None,
            "sources_count": len(self.config_sources),
            "watching_enabled": self.is_watching,
            "watchers_count": len(self.watchers),
            "config_history_count": len(self.config_history),
        }

    async def _handle_config_change(self, change: ConfigChange) -> None:
        """Handle configuration change from any source."""
        logger.info(
            f"Configuration change detected: {change.change_type} in {change.source}"
        )

        try:
            # Reload configuration
            await self.reload_configuration()

        except Exception as e:
            logger.exception(f"Failed to handle config change: {e}")

    async def _notify_watchers(self, config_key: str, change: ConfigChange) -> None:
        """Notify watchers about configuration changes."""
        if config_key in self.watchers:
            await self.watchers[config_key].notify_change(change)

    def _deep_merge(
        self, base: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# Global configuration orchestrator instance
_config_orchestrator: EnterpriseConfigurationOrchestrator | None = None


async def get_config_orchestrator() -> EnterpriseConfigurationOrchestrator:
    """Get or create the global configuration orchestrator."""
    global _config_orchestrator

    if _config_orchestrator is None:
        _config_orchestrator = EnterpriseConfigurationOrchestrator()

        # Add default configuration sources
        _config_orchestrator.add_config_source(EnvironmentConfigSource(priority=100))

        # Add file-based configuration if it exists
        config_file = Path("config/enterprise.yaml")
        if config_file.exists():
            _config_orchestrator.add_config_source(
                FileConfigSource(str(config_file), priority=50)
            )

    return _config_orchestrator


async def load_enterprise_configuration() -> EnterpriseConfig:
    """Load enterprise configuration using the global orchestrator."""
    orchestrator = await get_config_orchestrator()
    return await orchestrator.load_enterprise_config()


async def start_configuration_watching() -> None:
    """Start watching for configuration changes."""
    orchestrator = await get_config_orchestrator()
    await orchestrator.watch_configuration_changes()


async def stop_configuration_watching() -> None:
    """Stop watching for configuration changes."""
    global _config_orchestrator

    if _config_orchestrator:
        await _config_orchestrator.stop_watching()
