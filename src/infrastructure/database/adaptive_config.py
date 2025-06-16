"""Adaptive configuration management for database connections."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AdaptationStrategy(str, Enum):
    """Strategies for adapting database configuration."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PoolMetrics:
    """Database pool performance metrics."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    avg_checkout_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: float = 0.0


@dataclass
class AdaptiveConfiguration:
    """Adaptive configuration parameters."""

    min_pool_size: int = 5
    max_pool_size: int = 20
    current_pool_size: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # Adaptation thresholds
    high_utilization_threshold: float = 0.8
    low_utilization_threshold: float = 0.3
    adaptation_cooldown_seconds: int = 60

    # Performance targets
    target_checkout_time_ms: float = 50.0
    target_query_time_ms: float = 100.0
    max_error_rate: float = 0.05


class AdaptiveConfigManager:
    """Manages adaptive configuration for database connections."""

    def __init__(self, strategy: AdaptationStrategy = AdaptationStrategy.BALANCED):
        """Initialize adaptive config manager.

        Args:
            strategy: Adaptation strategy to use
        """
        self.strategy = strategy
        self.config = AdaptiveConfiguration()
        self.metrics_history: list[PoolMetrics] = []
        self.last_adaptation_time = 0.0

        # Strategy-specific parameters
        if strategy == AdaptationStrategy.CONSERVATIVE:
            self.config.adaptation_cooldown_seconds = 120
            self.config.high_utilization_threshold = 0.9
            self.config.low_utilization_threshold = 0.2
        elif strategy == AdaptationStrategy.AGGRESSIVE:
            self.config.adaptation_cooldown_seconds = 30
            self.config.high_utilization_threshold = 0.7
            self.config.low_utilization_threshold = 0.4

    def add_metrics(self, metrics: PoolMetrics) -> None:
        """Add new metrics for analysis.

        Args:
            metrics: Pool metrics to add
        """
        self.metrics_history.append(metrics)

        # Keep only recent metrics (last 100 entries)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    def should_adapt(self) -> bool:
        """Check if configuration should be adapted.

        Returns:
            True if adaptation is needed
        """
        if not self.metrics_history:
            return False

        # Check cooldown period
        current_time = time.time()
        if (
            current_time - self.last_adaptation_time
        ) < self.config.adaptation_cooldown_seconds:
            return False

        latest_metrics = self.metrics_history[-1]
        utilization = latest_metrics.active_connections / max(
            latest_metrics.total_connections, 1
        )

        # Check if utilization is outside target range
        return (
            utilization > self.config.high_utilization_threshold
            or utilization < self.config.low_utilization_threshold
        )

    def adapt_configuration(self) -> dict[str, Any]:
        """Adapt configuration based on current metrics.

        Returns:
            Dictionary of configuration changes
        """
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]
        utilization = latest_metrics.active_connections / max(
            latest_metrics.total_connections, 1
        )
        changes = {}

        if utilization > self.config.high_utilization_threshold:
            # Increase pool size
            new_size = min(
                self.config.current_pool_size + self._get_adaptation_step(),
                self.config.max_pool_size,
            )
            if new_size > self.config.current_pool_size:
                changes["pool_size"] = new_size
                self.config.current_pool_size = new_size
                logger.info(
                    f"Increased pool size to {new_size} (utilization: {utilization:.2f})"
                )

        elif utilization < self.config.low_utilization_threshold:
            # Decrease pool size
            new_size = max(
                self.config.current_pool_size - self._get_adaptation_step(),
                self.config.min_pool_size,
            )
            if new_size < self.config.current_pool_size:
                changes["pool_size"] = new_size
                self.config.current_pool_size = new_size
                logger.info(
                    f"Decreased pool size to {new_size} (utilization: {utilization:.2f})"
                )

        # Check for performance issues
        if (
            latest_metrics.avg_checkout_time_ms
            > self.config.target_checkout_time_ms * 2
        ) and "pool_timeout" not in changes:
            new_timeout = min(self.config.pool_timeout + 10, 60)
            changes["pool_timeout"] = new_timeout
            self.config.pool_timeout = new_timeout
            logger.info(f"Increased pool timeout to {new_timeout}s")

        if changes:
            self.last_adaptation_time = time.time()

        return changes

    def _get_adaptation_step(self) -> int:
        """Get the step size for pool size changes.

        Returns:
            Step size based on strategy
        """
        if self.strategy == AdaptationStrategy.CONSERVATIVE:
            return 1
        elif self.strategy == AdaptationStrategy.AGGRESSIVE:
            return 3
        else:  # BALANCED
            return 2

    def get_current_config(self) -> dict[str, Any]:
        """Get current configuration as dictionary.

        Returns:
            Current configuration parameters
        """
        return {
            "pool_size": self.config.current_pool_size,
            "pool_timeout": self.config.pool_timeout,
            "pool_recycle": self.config.pool_recycle,
            "min_size": self.config.min_pool_size,
            "max_size": self.config.max_pool_size,
        }

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = AdaptiveConfiguration()
        self.metrics_history.clear()
        self.last_adaptation_time = 0.0
        logger.info("Reset adaptive configuration to defaults")
