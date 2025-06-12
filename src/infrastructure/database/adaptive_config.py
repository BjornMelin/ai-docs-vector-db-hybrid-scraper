"""Adaptive configuration system for database connection management.

This module provides dynamic configuration adjustments based on system load,
performance metrics, and operational patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class SystemLoadLevel(Enum):
    """System load level classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdaptationStrategy(Enum):
    """Configuration adaptation strategies."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class SystemMetrics:
    """Current system resource metrics."""

    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    load_average: float
    timestamp: float


@dataclass
class PerformanceThresholds:
    """Performance thresholds for adaptation decisions."""

    # Response time thresholds (milliseconds)
    good_response_time_ms: float = 50.0
    acceptable_response_time_ms: float = 200.0
    poor_response_time_ms: float = 500.0

    # System resource thresholds
    low_cpu_threshold: float = 30.0
    medium_cpu_threshold: float = 60.0
    high_cpu_threshold: float = 80.0

    low_memory_threshold: float = 40.0
    medium_memory_threshold: float = 70.0
    high_memory_threshold: float = 85.0

    # Connection pool thresholds
    min_pool_utilization: float = 0.2
    optimal_pool_utilization: float = 0.6
    max_pool_utilization: float = 0.9


@dataclass
class AdaptiveSettings:
    """Adaptive configuration settings."""

    # Monitoring intervals (seconds)
    base_monitoring_interval: float = 5.0
    high_load_monitoring_interval: float = 1.0
    low_load_monitoring_interval: float = 10.0

    # Pool sizing parameters
    min_pool_size: int = 5
    max_pool_size: int = 50
    pool_scale_step: int = 2

    # Circuit breaker adaptation
    adaptive_failure_thresholds: bool = True
    base_failure_threshold: int = 5
    high_load_failure_threshold: int = 3
    low_load_failure_threshold: int = 8

    # Query timeout adaptation
    adaptive_timeouts: bool = True
    base_timeout_ms: float = 30000.0
    min_timeout_ms: float = 5000.0
    max_timeout_ms: float = 120000.0


class AdaptiveConfigManager:
    """Dynamic configuration manager that adapts to system conditions.

    This manager continuously monitors system performance and automatically
    adjusts configuration parameters to optimize database connection management.
    """

    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.MODERATE,
        thresholds: PerformanceThresholds | None = None,
        settings: AdaptiveSettings | None = None,
    ):
        """Initialize adaptive configuration manager.

        Args:
            strategy: Adaptation strategy (conservative, moderate, aggressive)
            thresholds: Performance thresholds for adaptation decisions
            settings: Adaptive configuration settings
        """
        self.strategy = strategy
        self.thresholds = thresholds or PerformanceThresholds()
        self.settings = settings or AdaptiveSettings()

        # Current configuration state
        self.current_pool_size = self.settings.min_pool_size
        self.current_monitoring_interval = self.settings.base_monitoring_interval
        self.current_failure_threshold = self.settings.base_failure_threshold
        self.current_timeout_ms = self.settings.base_timeout_ms

        # Adaptation history
        self.adaptation_history: list[dict[str, Any]] = []
        self.max_history_size = 100

        # Performance tracking
        self.recent_system_metrics: list[SystemMetrics] = []
        self.recent_db_metrics: list[dict[str, Any]] = []
        self.metrics_window_size = 20

        # Adaptation state
        self.last_adaptation_time = 0.0
        self.adaptation_cooldown = 30.0  # Minimum time between adaptations
        self.is_monitoring = False
        self._monitoring_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        """Start adaptive configuration monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            f"Started adaptive configuration monitoring with {self.strategy.value} strategy"
        )

    async def stop_monitoring(self) -> None:
        """Stop adaptive configuration monitoring."""
        self.is_monitoring = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                logger.debug("Adaptive monitoring task cancelled")
            except Exception as e:
                logger.debug(f"Expected cancellation during stop: {e}")
            finally:
                self._monitoring_task = None
        logger.info("Stopped adaptive configuration monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for adaptive configuration."""
        try:
            while self.is_monitoring:
                try:
                    # Collect system metrics
                    system_metrics = await self._collect_system_metrics()
                    self.recent_system_metrics.append(system_metrics)

                    # Maintain metrics window
                    if len(self.recent_system_metrics) > self.metrics_window_size:
                        self.recent_system_metrics.pop(0)

                    # Analyze and adapt if needed
                    await self._analyze_and_adapt()

                    # Wait for next monitoring cycle with cancellable sleep
                    sleep_time = int(self.current_monitoring_interval)
                    for _ in range(max(1, sleep_time)):
                        if not self.is_monitoring:
                            return
                        await asyncio.sleep(1.0)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in adaptive monitoring loop: {e}")
                    # Use cancellable sleep for error recovery
                    sleep_time = int(self.settings.base_monitoring_interval)
                    for _ in range(max(1, sleep_time)):
                        if not self.is_monitoring:
                            return
                        await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("Adaptive monitoring loop terminated")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            # Calculate disk I/O percentage (approximation)
            disk_io_percent = 0.0
            if disk_io:
                # Simple heuristic based on disk utilization
                disk_io_percent = min(
                    100.0,
                    (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100),
                )

            # Calculate network I/O percentage (approximation)
            network_io_percent = 0.0
            if network_io:
                # Simple heuristic based on network utilization
                network_io_percent = min(
                    100.0,
                    (network_io.bytes_sent + network_io.bytes_recv)
                    / (1024 * 1024 * 10),
                )

            # Get load average (Unix-like systems only)
            load_average = 0.0
            try:
                load_average = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                # Fallback for systems without load average
                load_average = cpu_percent / 100.0

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_percent=disk_io_percent,
                network_io_percent=network_io_percent,
                load_average=load_average,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_io_percent=0.0,
                network_io_percent=0.0,
                load_average=0.0,
                timestamp=time.time(),
            )

    async def _analyze_and_adapt(self) -> None:
        """Analyze current conditions and adapt configuration if needed."""
        current_time = time.time()

        # Check adaptation cooldown
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return

        if len(self.recent_system_metrics) < 3:
            return  # Need more data

        # Determine current system load level
        load_level = self._determine_load_level()

        # Calculate adaptation recommendations
        adaptations = await self._calculate_adaptations(load_level)

        if adaptations:
            await self._apply_adaptations(adaptations, load_level)
            self.last_adaptation_time = current_time

    def _determine_load_level(self) -> SystemLoadLevel:
        """Determine current system load level."""
        if not self.recent_system_metrics:
            return SystemLoadLevel.MEDIUM

        # Analyze recent metrics
        recent_metrics = self.recent_system_metrics[-5:]  # Last 5 measurements
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_load = sum(m.load_average for m in recent_metrics) / len(recent_metrics)

        # Determine load level based on thresholds
        high_indicators = 0

        if avg_cpu > self.thresholds.high_cpu_threshold:
            high_indicators += 2
        elif avg_cpu > self.thresholds.medium_cpu_threshold:
            high_indicators += 1

        if avg_memory > self.thresholds.high_memory_threshold:
            high_indicators += 2
        elif avg_memory > self.thresholds.medium_memory_threshold:
            high_indicators += 1

        if avg_load > 2.0:  # High load average
            high_indicators += 1

        # Classify load level
        if high_indicators >= 4:
            return SystemLoadLevel.CRITICAL
        elif high_indicators >= 2:
            return SystemLoadLevel.HIGH
        elif high_indicators >= 1:
            return SystemLoadLevel.MEDIUM
        else:
            return SystemLoadLevel.LOW

    async def _calculate_adaptations(
        self, load_level: SystemLoadLevel
    ) -> dict[str, Any]:
        """Calculate recommended configuration adaptations."""
        adaptations = {}

        # Adjust monitoring interval based on load
        if load_level in [SystemLoadLevel.HIGH, SystemLoadLevel.CRITICAL]:
            new_interval = self.settings.high_load_monitoring_interval
        elif load_level == SystemLoadLevel.LOW:
            new_interval = self.settings.low_load_monitoring_interval
        else:
            new_interval = self.settings.base_monitoring_interval

        if new_interval != self.current_monitoring_interval:
            adaptations["monitoring_interval"] = new_interval

        # Adjust pool size based on load and strategy
        pool_adaptation = self._calculate_pool_size_adaptation(load_level)
        if pool_adaptation != self.current_pool_size:
            adaptations["pool_size"] = pool_adaptation

        # Adjust circuit breaker thresholds
        if self.settings.adaptive_failure_thresholds:
            threshold_adaptation = self._calculate_failure_threshold_adaptation(
                load_level
            )
            if threshold_adaptation != self.current_failure_threshold:
                adaptations["failure_threshold"] = threshold_adaptation

        # Adjust query timeouts
        if self.settings.adaptive_timeouts:
            timeout_adaptation = self._calculate_timeout_adaptation(load_level)
            if (
                abs(timeout_adaptation - self.current_timeout_ms) > 1000
            ):  # 1 second threshold
                adaptations["timeout_ms"] = timeout_adaptation

        return adaptations

    def _calculate_pool_size_adaptation(self, load_level: SystemLoadLevel) -> int:
        """Calculate optimal pool size based on load level and strategy."""
        strategy_multipliers = {
            AdaptationStrategy.CONSERVATIVE: {
                SystemLoadLevel.LOW: 0.8,
                SystemLoadLevel.MEDIUM: 1.0,
                SystemLoadLevel.HIGH: 1.2,
                SystemLoadLevel.CRITICAL: 1.3,
            },
            AdaptationStrategy.MODERATE: {
                SystemLoadLevel.LOW: 0.7,
                SystemLoadLevel.MEDIUM: 1.0,
                SystemLoadLevel.HIGH: 1.5,
                SystemLoadLevel.CRITICAL: 1.8,
            },
            AdaptationStrategy.AGGRESSIVE: {
                SystemLoadLevel.LOW: 0.5,
                SystemLoadLevel.MEDIUM: 1.0,
                SystemLoadLevel.HIGH: 2.0,
                SystemLoadLevel.CRITICAL: 2.5,
            },
        }

        base_size = self.settings.min_pool_size + (
            (self.settings.max_pool_size - self.settings.min_pool_size) * 0.5
        )

        multiplier = strategy_multipliers[self.strategy][load_level]
        adapted_size = int(base_size * multiplier)

        # Apply constraints
        adapted_size = max(
            self.settings.min_pool_size, min(self.settings.max_pool_size, adapted_size)
        )

        return adapted_size

    def _calculate_failure_threshold_adaptation(
        self, load_level: SystemLoadLevel
    ) -> int:
        """Calculate optimal failure threshold based on load level."""
        if load_level in [SystemLoadLevel.HIGH, SystemLoadLevel.CRITICAL]:
            return self.settings.high_load_failure_threshold
        elif load_level == SystemLoadLevel.LOW:
            return self.settings.low_load_failure_threshold
        else:
            return self.settings.base_failure_threshold

    def _calculate_timeout_adaptation(self, load_level: SystemLoadLevel) -> float:
        """Calculate optimal timeout based on load level and recent performance."""
        base_timeout = self.settings.base_timeout_ms

        # Adjust based on load level
        if load_level == SystemLoadLevel.CRITICAL:
            adapted_timeout = base_timeout * 2.0
        elif load_level == SystemLoadLevel.HIGH:
            adapted_timeout = base_timeout * 1.5
        elif load_level == SystemLoadLevel.LOW:
            adapted_timeout = base_timeout * 0.8
        else:
            adapted_timeout = base_timeout

        # Apply constraints
        adapted_timeout = max(
            self.settings.min_timeout_ms,
            min(self.settings.max_timeout_ms, adapted_timeout),
        )

        return adapted_timeout

    async def _apply_adaptations(
        self, adaptations: dict[str, Any], load_level: SystemLoadLevel
    ) -> None:
        """Apply configuration adaptations."""
        changes_made = []

        if "monitoring_interval" in adaptations:
            old_value = self.current_monitoring_interval
            self.current_monitoring_interval = adaptations["monitoring_interval"]
            changes_made.append(
                f"monitoring_interval: {old_value} -> {self.current_monitoring_interval}"
            )

        if "pool_size" in adaptations:
            old_value = self.current_pool_size
            self.current_pool_size = adaptations["pool_size"]
            changes_made.append(f"pool_size: {old_value} -> {self.current_pool_size}")

        if "failure_threshold" in adaptations:
            old_value = self.current_failure_threshold
            self.current_failure_threshold = adaptations["failure_threshold"]
            changes_made.append(
                f"failure_threshold: {old_value} -> {self.current_failure_threshold}"
            )

        if "timeout_ms" in adaptations:
            old_value = self.current_timeout_ms
            self.current_timeout_ms = adaptations["timeout_ms"]
            changes_made.append(f"timeout_ms: {old_value} -> {self.current_timeout_ms}")

        if changes_made:
            # Record adaptation in history
            adaptation_record = {
                "timestamp": time.time(),
                "load_level": load_level.value,
                "strategy": self.strategy.value,
                "changes": adaptations,
                "reason": f"Adapted to {load_level.value} load conditions",
            }

            self.adaptation_history.append(adaptation_record)

            # Maintain history size
            if len(self.adaptation_history) > self.max_history_size:
                self.adaptation_history.pop(0)

            logger.info(
                f"Applied adaptive configuration changes for {load_level.value} load: {', '.join(changes_made)}"
            )

    async def get_current_configuration(self) -> dict[str, Any]:
        """Get current adaptive configuration state."""
        load_level = (
            self._determine_load_level()
            if self.recent_system_metrics
            else SystemLoadLevel.MEDIUM
        )

        return {
            "strategy": self.strategy.value,
            "current_load_level": load_level.value,
            "current_settings": {
                "pool_size": self.current_pool_size,
                "monitoring_interval": self.current_monitoring_interval,
                "failure_threshold": self.current_failure_threshold,
                "timeout_ms": self.current_timeout_ms,
            },
            "thresholds": {
                "cpu_thresholds": {
                    "low": self.thresholds.low_cpu_threshold,
                    "medium": self.thresholds.medium_cpu_threshold,
                    "high": self.thresholds.high_cpu_threshold,
                },
                "memory_thresholds": {
                    "low": self.thresholds.low_memory_threshold,
                    "medium": self.thresholds.medium_memory_threshold,
                    "high": self.thresholds.high_memory_threshold,
                },
            },
            "recent_adaptations": len(self.adaptation_history),
            "monitoring_active": self.is_monitoring,
        }

    async def get_adaptation_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent adaptation history."""
        return self.adaptation_history[-limit:] if self.adaptation_history else []

    async def force_adaptation(
        self, adaptations: dict[str, Any], reason: str = "Manual override"
    ) -> None:
        """Force specific configuration adaptations."""
        if "pool_size" in adaptations:
            self.current_pool_size = max(
                self.settings.min_pool_size,
                min(self.settings.max_pool_size, adaptations["pool_size"]),
            )

        if "monitoring_interval" in adaptations:
            self.current_monitoring_interval = max(
                0.1, adaptations["monitoring_interval"]
            )

        if "failure_threshold" in adaptations:
            self.current_failure_threshold = max(1, adaptations["failure_threshold"])

        if "timeout_ms" in adaptations:
            self.current_timeout_ms = max(
                self.settings.min_timeout_ms,
                min(self.settings.max_timeout_ms, adaptations["timeout_ms"]),
            )

        # Record forced adaptation
        adaptation_record = {
            "timestamp": time.time(),
            "load_level": "manual",
            "strategy": "manual",
            "changes": adaptations,
            "reason": reason,
        }

        self.adaptation_history.append(adaptation_record)

        logger.info(f"Applied forced configuration changes: {adaptations} - {reason}")

    async def get_performance_analysis(self) -> dict[str, Any]:
        """Get performance analysis based on recent metrics."""
        if not self.recent_system_metrics:
            return {"error": "No metrics available"}

        recent = self.recent_system_metrics[-10:]  # Last 10 measurements

        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        avg_load = sum(m.load_average for m in recent) / len(recent)

        # Calculate stability (coefficient of variation)
        cpu_values = [m.cpu_percent for m in recent]
        cpu_stability = (
            1.0 - ((max(cpu_values) - min(cpu_values)) / max(1.0, avg_cpu))
            if len(cpu_values) > 1
            else 1.0
        )

        return {
            "current_load_level": self._determine_load_level().value,
            "resource_utilization": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "avg_load_average": round(avg_load, 2),
                "cpu_stability": round(max(0.0, cpu_stability), 2),
            },
            "adaptation_effectiveness": {
                "total_adaptations": len(self.adaptation_history),
                "recent_adaptations": len(
                    [
                        a
                        for a in self.adaptation_history
                        if time.time() - a["timestamp"] < 3600  # Last hour
                    ]
                ),
                "strategy": self.strategy.value,
            },
            "recommendations": self._generate_recommendations(
                avg_cpu, avg_memory, avg_load
            ),
            "metrics_collected": len(self.recent_system_metrics),
        }

    def _generate_recommendations(
        self, avg_cpu: float, avg_memory: float, avg_load: float
    ) -> list[str]:
        """Generate optimization recommendations based on current metrics."""
        recommendations = []

        if avg_cpu > self.thresholds.high_cpu_threshold:
            recommendations.append(
                "High CPU usage detected - consider optimizing queries or scaling resources"
            )

        if avg_memory > self.thresholds.high_memory_threshold:
            recommendations.append(
                "High memory usage detected - monitor for memory leaks and optimize caching"
            )

        if avg_load > 2.0:
            recommendations.append(
                "High system load - consider distributing workload or increasing capacity"
            )

        if (
            self.strategy == AdaptationStrategy.CONSERVATIVE
            and avg_cpu < self.thresholds.low_cpu_threshold
        ):
            recommendations.append(
                "System underutilized - consider more aggressive adaptation strategy"
            )

        if len(self.adaptation_history) > 20:  # Many adaptations recently
            recent_adaptations = [
                a
                for a in self.adaptation_history
                if time.time() - a["timestamp"] < 1800  # Last 30 minutes
            ]
            if len(recent_adaptations) > 5:
                recommendations.append(
                    "Frequent adaptations detected - consider reviewing thresholds"
                )

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations
