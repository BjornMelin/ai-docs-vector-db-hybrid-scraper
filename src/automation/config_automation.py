"""Configuration automation for zero-maintenance operations.

This module provides:
- Auto-detecting configuration with sensible defaults
- Configuration drift detection and auto-correction
- Environment-specific optimization
- Validation and error correction
"""

import asyncio
import hashlib
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict

import psutil
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


def detect_environment() -> str:
    """Auto-detect the deployment environment."""
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if os.getenv("AWS_EXECUTION_ENV"):
        return "aws"
    if os.getenv("DOCKER_CONTAINER") or (
        platform.system() == "Linux" and Path("/.dockerenv").exists()
    ):
        return "docker"
    if os.getenv("CI"):
        return "ci"
    if os.getenv("PYTEST_CURRENT_TEST"):
        return "test"
    return "development"


def auto_detect_database() -> str:
    """Auto-detect database connection based on environment."""
    env = detect_environment()

    if env == "kubernetes":
        return "postgresql://postgres:password@postgres-service:5432/vectordb"
    if env == "docker":
        return "postgresql://postgres:password@localhost:5432/vectordb"
    if env == "test":
        return "sqlite+aiosqlite:///./test.db"
    return "sqlite+aiosqlite:///./local.db"


def adaptive_threshold() -> float:
    """Calculate adaptive threshold based on system capacity."""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count() or 4

    # Higher thresholds for more powerful systems
    base_threshold = 0.7
    memory_factor = min(memory_gb / 16, 2.0)  # Cap at 2x for 16GB+
    cpu_factor = min(cpu_count / 8, 2.0)  # Cap at 2x for 8+ cores

    return min(base_threshold * (memory_factor + cpu_factor) / 2, 0.9)


class ConfigAutoCorrector:
    """Auto-corrects common configuration mistakes."""

    @staticmethod
    def fix_value(value: Any, field_name: str) -> Any:
        """Apply auto-corrections to configuration values."""
        if field_name.endswith("_url") and isinstance(value, str):
            # Fix common URL mistakes
            if value.startswith("http//"):
                return value.replace("http//", "http://")
            if value.startswith("https//"):
                return value.replace("https//", "https://")

        elif field_name.endswith("_timeout") and isinstance(value, int | float):
            # Ensure reasonable timeout values
            return max(1.0, min(value, 300.0))

        elif field_name.endswith("_pool_size") and isinstance(value, int):
            # Ensure reasonable pool sizes
            return max(1, min(value, 100))

        elif field_name == "log_level" and isinstance(value, str):
            # Normalize log levels
            return (
                value.upper()
                if value.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                else "INFO"
            )

        return value


class ZeroMaintenanceConfig(BaseSettings):
    """Auto-detecting configuration with sensible defaults and auto-correction."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # Environment detection
    environment: str = Field(default_factory=detect_environment)

    # Database configuration
    database_url: str = Field(default_factory=auto_detect_database)
    database_pool_size: int = Field(
        default_factory=lambda: min(os.cpu_count() or 4, 20)
    )
    database_max_overflow: int = Field(
        default_factory=lambda: (os.cpu_count() or 4) * 2
    )
    database_pool_timeout: float = Field(default=30.0)

    # Qdrant configuration
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="documents")
    qdrant_timeout: float = Field(default=60.0)

    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379")
    redis_database: int = Field(default=0)
    redis_timeout: float = Field(default=10.0)

    # Worker configuration
    worker_count: int = Field(default_factory=lambda: os.cpu_count() or 4)
    max_workers: int = Field(default_factory=lambda: (os.cpu_count() or 4) * 2)

    # Memory configuration (in MB)
    memory_limit: int = Field(
        default_factory=lambda: int(psutil.virtual_memory().total / (1024**2) * 0.8)
    )
    cache_size: int = Field(
        default_factory=lambda: int(psutil.virtual_memory().total / (1024**2) * 0.1)
    )

    # Performance thresholds
    scale_up_threshold: float = Field(default_factory=adaptive_threshold)
    scale_down_threshold: float = Field(
        default_factory=lambda: adaptive_threshold() * 0.3
    )

    # Monitoring configuration
    health_check_interval: float = Field(default=30.0)
    metrics_retention_days: int = Field(default=30)

    # Security configuration
    api_key_rotation_days: int = Field(default=90)
    max_request_size: int = Field(default=100 * 1024 * 1024)  # 100MB

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    @field_validator("database_pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        """Ensure pool size is reasonable for the environment."""
        return min(v, 20)  # Simplified validation for Pydantic v2

    @field_validator("memory_limit")
    @classmethod
    def validate_memory_limit(cls, v):
        """Ensure memory limit doesn't exceed system memory."""
        total_memory = psutil.virtual_memory().total / (1024**2)
        return min(v, int(total_memory * 0.9))


class AutoConfigManager:
    """Automatically manages configuration with smart defaults and validation."""

    def __init__(self):
        self.config: ZeroMaintenanceConfig | None = None
        self._config_hash: str | None = None

    async def initialize(self) -> ZeroMaintenanceConfig:
        """Initialize configuration with auto-detection and validation."""
        try:
            self.config = ZeroMaintenanceConfig()
            self._config_hash = self._calculate_hash()

            logger.info(
                f"Configuration initialized for environment: {self.config.environment}"
            )
            logger.info(
                f"Auto-detected settings: "
                f"workers={self.config.worker_count}, "
                f"memory_limit={self.config.memory_limit}MB, "
                f"scale_threshold={self.config.scale_up_threshold:.2f}"
            )

            return self.config

        except Exception as e:
            logger.exception(f"Failed to initialize configuration: {e}")
            # Fallback to minimal configuration
            return await self._create_fallback_config()

    async def _create_fallback_config(self) -> ZeroMaintenanceConfig:
        """Create minimal fallback configuration."""
        # Use environment variables or absolute minimums
        fallback_data = {
            "environment": "development",
            "database_url": "sqlite+aiosqlite:///./fallback.db",
            "worker_count": 2,
            "memory_limit": 1024,  # 1GB
            "log_level": "WARNING",
        }

        return ZeroMaintenanceConfig(**fallback_data)

    def _calculate_hash(self) -> str:
        """Calculate configuration hash for drift detection."""
        if not self.config:
            return ""

        config_dict = self.config.model_dump()
        config_str = str(sorted(config_dict.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def check_drift(self) -> bool:
        """Check if configuration has drifted from expected values."""
        if not self.config or not self._config_hash:
            return False

        current_hash = self._calculate_hash()
        return current_hash != self._config_hash

    async def refresh_config(self) -> bool:
        """Refresh configuration with current environment state."""
        try:
            old_config = self.config
            new_config = ZeroMaintenanceConfig()

            # Check if refresh is needed
            if old_config and self._configs_equivalent(old_config, new_config):
                return False

            self.config = new_config
            self._config_hash = self._calculate_hash()

            logger.info("Configuration refreshed successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to refresh configuration: {e}")
            return False

    def _configs_equivalent(
        self, config1: ZeroMaintenanceConfig, config2: ZeroMaintenanceConfig
    ) -> bool:
        """Check if two configurations are functionally equivalent."""
        # Compare essential fields that affect system behavior
        essential_fields = [
            "database_url",
            "qdrant_url",
            "redis_url",
            "worker_count",
            "memory_limit",
            "scale_up_threshold",
        ]

        for field in essential_fields:
            if getattr(config1, field) != getattr(config2, field):
                return False

        return True


class ConfigDriftHealer:
    """Automatically detects and heals configuration drift."""

    def __init__(self, config_manager: AutoConfigManager):
        self.config_manager = config_manager
        self.drift_patterns: dict[str, Any] = {}
        self.healing_enabled = True

    async def start_monitoring(self, check_interval: float = 300):
        """Start continuous configuration drift monitoring."""
        logger.info(
            f"Starting configuration drift monitoring (interval: {check_interval}s)"
        )

        while self.healing_enabled:
            try:
                await self._check_and_heal_drift()
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.exception(f"Configuration drift monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter interval on error

    async def _check_and_heal_drift(self):
        """Check for configuration drift and automatically heal if detected."""
        if await self.config_manager.check_drift():
            logger.warning("Configuration drift detected")

            # Attempt to heal drift
            if await self._heal_drift():
                logger.info("Configuration drift healed successfully")
            else:
                logger.error("Failed to heal configuration drift")
                await self._notify_manual_intervention_needed()

    async def _heal_drift(self) -> bool:
        """Attempt to automatically heal configuration drift."""
        try:
            # Refresh configuration with current environment
            if await self.config_manager.refresh_config():
                return True

            # If refresh fails, try to restore from backup
            return await self._restore_from_backup()

        except Exception as e:
            logger.exception(f"Configuration healing failed: {e}")
            return False

    async def _restore_from_backup(self) -> bool:
        """Restore configuration from backup."""
        # Implementation would restore from a known good configuration
        # For now, just log the attempt
        logger.info("Attempting to restore configuration from backup")
        return False

    async def _notify_manual_intervention_needed(self):
        """Notify that manual intervention is needed for configuration issues."""
        logger.critical(
            "Configuration drift could not be automatically healed - manual intervention required"
        )

        # In a real implementation, this would send alerts through configured channels
        # (email, Slack, PagerDuty, etc.)

    def stop_monitoring(self):
        """Stop configuration drift monitoring."""
        self.healing_enabled = False
        logger.info("Configuration drift monitoring stopped")


# Global instance for easy access
_config_manager: AutoConfigManager | None = None


async def get_auto_config() -> ZeroMaintenanceConfig:
    """Get the global auto-managed configuration instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = AutoConfigManager()

    if _config_manager.config is None:
        await _config_manager.initialize()

    return _config_manager.config


async def start_config_automation():
    """Start the configuration automation system."""
    global _config_manager

    if _config_manager is None:
        _config_manager = AutoConfigManager()
        await _config_manager.initialize()

    # Start drift healing
    drift_healer = ConfigDriftHealer(_config_manager)
    asyncio.create_task(drift_healer.start_monitoring())

    logger.info("Configuration automation system started")
