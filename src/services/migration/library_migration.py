"""Library migration manager for transitioning to modern implementations.

This module provides a migration manager that handles the transition from
custom implementations to modern battle-tested libraries, ensuring backward
compatibility and gradual migration.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any


try:
    import redis
except ImportError:
    redis = None

from src.config import Config
from src.services.cache.manager import CacheManager
from src.services.cache.modern import ModernCacheManager
from src.services.circuit_breaker.modern import ModernCircuitBreakerManager
from src.services.functional.circuit_breaker import CircuitBreaker
from src.services.middleware.rate_limiting import ModernRateLimiter


logger = logging.getLogger(__name__)


class MigrationMode(Enum):
    """Migration modes for library transition."""

    LEGACY_ONLY = "legacy_only"  # Use only legacy implementations
    MODERN_ONLY = "modern_only"  # Use only modern implementations
    PARALLEL = "parallel"  # Run both for comparison
    GRADUAL = "gradual"  # Gradual migration with feature flags


@dataclass
class MigrationConfig:
    """Configuration for library migration."""

    mode: MigrationMode = MigrationMode.GRADUAL
    circuit_breaker_enabled: bool = True
    cache_enabled: bool = True
    rate_limiting_enabled: bool = True
    performance_monitoring: bool = True
    rollback_threshold: float = 0.1  # Error rate threshold for rollback


class LibraryMigrationManager:
    """Manager for migrating from custom to modern library implementations.

    Handles gradual migration with feature flags, performance monitoring,
    and automatic rollback on issues.
    """

    def __init__(
        self,
        config: Config,
        migration_config: MigrationConfig | None = None,
        redis_url: str = "redis://localhost:6379",
    ):
        """Initialize library migration manager.

        Args:
            config: Application configuration
            migration_config: Migration-specific configuration
            redis_url: Redis URL for modern implementations

        """
        self.config = config
        self.migration_config = migration_config or MigrationConfig()
        self.redis_url = redis_url

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {
            "circuit_breaker": {"legacy": {}, "modern": {}},
            "cache": {"legacy": {}, "modern": {}},
            "rate_limiting": {"legacy": {}, "modern": {}},
        }

        # Migration state
        self.migration_state: dict[str, bool] = {
            "circuit_breaker_migrated": False,
            "cache_migrated": False,
            "rate_limiting_migrated": False,
        }

        # Service instances
        self._modern_circuit_breaker: ModernCircuitBreakerManager | None = None
        self._modern_cache: ModernCacheManager | None = None
        self._modern_rate_limiter: ModernRateLimiter | None = None
        self._legacy_services: dict[str, Any] = {}

        logger.info(
            "LibraryMigrationManager initialized with mode: %s",
            self.migration_config.mode.value,
        )

    async def initialize(self) -> None:
        """Initialize migration manager and create service instances."""
        await self._initialize_modern_services()
        await self._initialize_legacy_services()
        await self._setup_monitoring()

    async def _initialize_modern_services(self) -> None:
        """Initialize modern service implementations."""
        try:
            if self.migration_config.circuit_breaker_enabled:
                self._modern_circuit_breaker = ModernCircuitBreakerManager(
                    redis_url=self.redis_url,
                    config=self.config,
                )
                logger.info("Initialized modern circuit breaker")

            if self.migration_config.cache_enabled:
                self._modern_cache = ModernCacheManager(
                    redis_url=self.redis_url,
                    config=self.config,
                )
                logger.info("Initialized modern cache")

            # Rate limiter initialization will be handled in FastAPI app setup
            logger.info("Modern services initialized successfully")

        except Exception:
            logger.exception("Failed to initialize modern services")
            raise

    async def _initialize_legacy_services(self) -> None:
        """Initialize legacy service implementations for fallback."""
        try:
            # Import legacy services

            # Create legacy instances
            if self.migration_config.circuit_breaker_enabled:
                legacy_config = {
                    "failure_threshold": 5,
                    "recovery_timeout": 60.0,
                    "half_open_requests": 1,
                }
                self._legacy_services["circuit_breaker"] = CircuitBreaker(
                    **legacy_config,
                )
                logger.info("Initialized legacy circuit breaker")

            if self.migration_config.cache_enabled:
                self._legacy_services["cache"] = CacheManager(
                    dragonfly_url=self.redis_url,
                    enable_local_cache=True,
                    enable_distributed_cache=True,
                )
                logger.info("Initialized legacy cache")

            logger.info("Legacy services initialized successfully")

        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Failed to initialize legacy services: %s", e)
            # Continue without legacy services if they fail

    async def _setup_monitoring(self) -> None:
        """Set up performance monitoring for migration."""
        if self.migration_config.performance_monitoring:
            # Start background monitoring task
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            # Store reference to prevent task garbage collection
            monitoring_task.add_done_callback(
                lambda _: logger.debug("Migration monitoring loop completed"),
            )
            logger.info("Performance monitoring enabled")

    async def get_circuit_breaker(self, service_name: str = "default") -> Any:
        """Get circuit breaker implementation based on migration mode.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker instance (modern or legacy)

        """
        mode = self.migration_config.mode

        if (
            mode == MigrationMode.MODERN_ONLY
            or self.migration_state["circuit_breaker_migrated"]
        ):
            if self._modern_circuit_breaker:
                return await self._modern_circuit_breaker.get_breaker(service_name)

        elif mode == MigrationMode.LEGACY_ONLY:
            return self._legacy_services.get("circuit_breaker")

        elif mode == MigrationMode.PARALLEL:
            # Return modern but track both
            await self._track_parallel_performance("circuit_breaker", service_name)
            if self._modern_circuit_breaker:
                return await self._modern_circuit_breaker.get_breaker(service_name)

        elif mode == MigrationMode.GRADUAL:
            # Gradual migration with feature flags
            use_modern = await self._should_use_modern("circuit_breaker")
            if use_modern and self._modern_circuit_breaker:
                self.migration_state["circuit_breaker_migrated"] = True
                return await self._modern_circuit_breaker.get_breaker(service_name)

        # Fallback to legacy
        return self._legacy_services.get("circuit_breaker")

    async def get_cache_manager(self) -> Any:
        """Get cache manager implementation based on migration mode.

        Returns:
            Cache manager instance (modern or legacy)

        """
        mode = self.migration_config.mode

        if mode == MigrationMode.MODERN_ONLY or self.migration_state["cache_migrated"]:
            return self._modern_cache

        if mode == MigrationMode.LEGACY_ONLY:
            return self._legacy_services.get("cache")

        if mode == MigrationMode.PARALLEL:
            # Return modern but track both
            await self._track_parallel_performance("cache", "default")
            return self._modern_cache

        if mode == MigrationMode.GRADUAL:
            # Gradual migration with feature flags
            use_modern = await self._should_use_modern("cache")
            if use_modern and self._modern_cache:
                self.migration_state["cache_migrated"] = True
                return self._modern_cache

        # Fallback to legacy
        return self._legacy_services.get("cache")

    def get_rate_limiter(self) -> ModernRateLimiter | None:
        """Get rate limiter implementation.

        Note: Rate limiting is only available in modern implementation.

        Returns:
            ModernRateLimiter instance or None

        """
        if self.migration_config.rate_limiting_enabled:
            return self._modern_rate_limiter
        return None

    async def _should_use_modern(self, service: str) -> bool:
        """Determine if modern implementation should be used for a service.

        Args:
            service: Service name

        Returns:
            True if modern implementation should be used

        """
        # Check error rates and performance metrics
        modern_metrics = self.performance_metrics[service]["modern"]
        legacy_metrics = self.performance_metrics[service]["legacy"]

        # Use modern if error rate is acceptable
        modern_error_rate = modern_metrics.get("error_rate", 0)
        if modern_error_rate > self.migration_config.rollback_threshold:
            logger.warning(
                "High error rate for modern %s: %s",
                service,
                modern_error_rate,
            )
            return False

        # Use modern if performance is comparable or better
        modern_latency = modern_metrics.get("avg_latency", float("inf"))
        legacy_latency = legacy_metrics.get("avg_latency", float("inf"))

        # Return the condition directly instead of if-else
        return (
            modern_latency <= legacy_latency * 1.2
        )  # Allow 20% performance degradation

    async def _track_parallel_performance(self, service: str, operation: str) -> None:
        """Track performance for parallel mode operation.

        Args:
            service: Service name
            operation: Operation being performed

        """
        # This would run both modern and legacy implementations
        # and compare their performance

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for performance tracking."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._collect_metrics()
                await self._check_rollback_conditions()
            except Exception:
                logger.exception("Error in monitoring loop")

    async def _collect_metrics(self) -> None:
        """Collect performance metrics from services."""
        try:
            # Collect circuit breaker metrics
            if self._modern_circuit_breaker:
                cb_status = await self._modern_circuit_breaker.get_all_statuses()
                self.performance_metrics["circuit_breaker"]["modern"].update(cb_status)

            # Collect cache metrics
            if self._modern_cache:
                cache_stats = await self._modern_cache.get_stats()
                self.performance_metrics["cache"]["modern"].update(cache_stats)

            logger.debug("Performance metrics collected")

        except Exception:
            logger.exception("Error collecting metrics")

    async def _check_rollback_conditions(self) -> None:
        """Check if rollback conditions are met."""
        for service in ["circuit_breaker", "cache"]:
            modern_metrics = self.performance_metrics[service]["modern"]
            error_rate = modern_metrics.get("error_rate", 0)

            if error_rate > self.migration_config.rollback_threshold:
                logger.warning(
                    "Triggering rollback for %s due to high error rate",
                    service,
                )
                self.migration_state[f"{service}_migrated"] = False

    async def get_migration_status(self) -> dict[str, Any]:
        """Get current migration status and metrics.

        Returns:
            Dictionary with migration status and performance metrics

        """
        return {
            "mode": self.migration_config.mode.value,
            "migration_state": self.migration_state,
            "performance_metrics": self.performance_metrics,
            "services": {
                "modern_circuit_breaker": self._modern_circuit_breaker is not None,
                "modern_cache": self._modern_cache is not None,
                "modern_rate_limiter": self._modern_rate_limiter is not None,
                "legacy_services": list(self._legacy_services.keys()),
            },
        }

    async def force_migration(self, service: str, to_modern: bool = True) -> bool:
        """Force migration of a specific service.

        Args:
            service: Service to migrate
            to_modern: True to migrate to modern, False to rollback to legacy

        Returns:
            True if migration was successful

        """
        try:
            if service in self.migration_state:
                self.migration_state[f"{service}_migrated"] = to_modern
                logger.info(
                    "Forced migration of %s to %s",
                    service,
                    "modern" if to_modern else "legacy",
                )
                return True
            logger.error("Unknown service for migration: %s", service)
        except Exception:
            logger.exception("Error forcing migration of %s", service)
            return False

        else:
            return False

    async def cleanup(self) -> None:
        """Clean up migration manager resources."""
        try:
            if self._modern_circuit_breaker:
                await self._modern_circuit_breaker.close()

            if self._modern_cache:
                await self._modern_cache.close()

            logger.info("LibraryMigrationManager cleaned up successfully")

        except Exception:
            logger.exception("Error cleaning up LibraryMigrationManager")


# Convenience function for creating migration manager
def create_migration_manager(
    config: Config,
    mode: MigrationMode = MigrationMode.GRADUAL,
    redis_url: str = "redis://localhost:6379",
) -> LibraryMigrationManager:
    """Create a library migration manager instance.

    Args:
        config: Application configuration
        mode: Migration mode
        redis_url: Redis URL for modern implementations

    Returns:
        LibraryMigrationManager instance

    """
    migration_config = MigrationConfig(mode=mode)
    return LibraryMigrationManager(config, migration_config, redis_url)
