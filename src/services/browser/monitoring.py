"""Monitoring and alerting for 5-tier browser automation system.

This module provides comprehensive monitoring capabilities including:
- Real-time performance metrics collection
- Health status monitoring for all tiers
- Alert generation for performance degradation
- System resource utilization tracking
"""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""

    TIER_FAILURE = "tier_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_RESPONSE_TIME = "slow_response_time"
    CACHE_MISS_RATE = "cache_miss_rate"


class Alert(BaseModel):
    """Alert model for monitoring system."""

    id: str = Field(description="Unique alert identifier")
    timestamp: float = Field(default_factory=time.time)
    severity: AlertSeverity = Field(description="Alert severity level")
    alert_type: AlertType = Field(description="Type of alert")
    tier: str | None = Field(default=None, description="Affected tier")
    message: str = Field(description="Human-readable alert message")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional alert data"
    )
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_at: float | None = Field(default=None, description="Resolution timestamp")


class HealthStatus(BaseModel):
    """Health status for a tier or system component."""

    component: str = Field(description="Component name")
    status: str = Field(description="healthy, degraded, or unhealthy")
    last_check: float = Field(default_factory=time.time)
    success_rate: float = Field(description="Success rate (0-1)")
    avg_response_time_ms: float = Field(description="Average response time")
    error_count: int = Field(default=0, description="Recent error count")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional health details"
    )


class PerformanceMetrics(BaseModel):
    """Performance metrics for monitoring."""

    timestamp: float = Field(default_factory=time.time)
    tier: str = Field(description="Tier name")
    requests_per_minute: float = Field(description="Current request rate")
    success_rate: float = Field(description="Success rate (0-1)")
    avg_response_time_ms: float = Field(description="Average response time")
    p95_response_time_ms: float = Field(description="95th percentile response time")
    active_requests: int = Field(description="Currently active requests")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    error_types: dict[str, int] = Field(
        default_factory=dict, description="Error type counts"
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""

    # Alert thresholds
    error_rate_threshold: float = Field(
        default=0.1, description="Error rate threshold for alerts"
    )
    response_time_threshold_ms: float = Field(
        default=10000, description="Response time threshold"
    )
    cache_miss_threshold: float = Field(
        default=0.8, description="Cache miss rate threshold"
    )

    # Monitoring intervals
    health_check_interval_seconds: int = Field(
        default=30, description="Health check frequency"
    )
    metrics_collection_interval_seconds: int = Field(
        default=10, description="Metrics collection frequency"
    )

    # Alert settings
    alert_cooldown_seconds: int = Field(
        default=300, description="Cooldown between similar alerts"
    )
    max_alerts_per_hour: int = Field(default=10, description="Maximum alerts per hour")

    # Data retention
    metrics_retention_hours: int = Field(
        default=24, description="How long to keep metrics"
    )
    alert_retention_hours: int = Field(
        default=168, description="How long to keep alerts (1 week)"
    )


class BrowserAutomationMonitor:
    """Comprehensive monitoring system for 5-tier browser automation."""

    def __init__(self, config: MonitoringConfig = None):
        """Initialize monitoring system.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()

        # Metrics storage
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=10000)
        self.health_status: dict[str, HealthStatus] = {}
        self.alerts: deque[Alert] = deque(maxlen=1000)

        # Alert tracking
        self.alert_cooldowns: dict[str, float] = {}
        self.alerts_per_hour: deque[float] = deque(maxlen=100)

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None

        # Alert handlers
        self.alert_handlers: list[Callable[[Alert], None]] = []

        # Data locks
        self.metrics_lock = asyncio.Lock()
        self.alerts_lock = asyncio.Lock()

        logger.info("BrowserAutomationMonitor initialized")

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
        logger.info("Monitoring stopped")

    async def record_request_metrics(
        self,
        tier: str,
        success: bool,
        response_time_ms: float,
        error_type: str | None = None,
        cache_hit: bool = False,
    ):
        """Record metrics for a single request.

        Args:
            tier: Tier name
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
            error_type: Type of error if failed
            cache_hit: Whether request was served from cache
        """
        async with self.metrics_lock:
            # Calculate current metrics for this tier
            current_time = time.time()
            recent_window = 60  # 1 minute window

            # Get recent metrics for this tier
            recent_metrics = [
                m
                for m in self.metrics_history
                if m.tier == tier and current_time - m.timestamp <= recent_window
            ]

            # Calculate aggregated metrics
            total_requests = len(recent_metrics) + 1  # Include current request
            successful_requests = sum(
                1 for m in recent_metrics if m.success_rate > 0.5
            ) + (1 if success else 0)

            success_rate = (
                successful_requests / total_requests if total_requests > 0 else 0.0
            )

            # Calculate response time metrics
            all_response_times = [m.avg_response_time_ms for m in recent_metrics] + [
                response_time_ms
            ]
            avg_response_time = sum(all_response_times) / len(all_response_times)

            sorted_times = sorted(all_response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]

            # Calculate cache hit rate
            cache_requests = [m for m in recent_metrics if m.cache_hit_rate > 0]
            total_cache_hits = sum(m.cache_hit_rate for m in cache_requests) + (
                1 if cache_hit else 0
            )
            cache_hit_rate = (
                total_cache_hits / total_requests if total_requests > 0 else 0.0
            )

            # Error type tracking
            error_types = defaultdict(int)
            for m in recent_metrics:
                for error, count in m.error_types.items():
                    error_types[error] += count
            if error_type:
                error_types[error_type] += 1

            # Create metrics entry
            metrics = PerformanceMetrics(
                tier=tier,
                requests_per_minute=total_requests,
                success_rate=success_rate,
                avg_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                active_requests=0,  # Would need external tracking
                cache_hit_rate=cache_hit_rate,
                error_types=dict(error_types),
            )

            self.metrics_history.append(metrics)

            # Update health status
            await self._update_health_status(tier, metrics)

            # Check for alerts
            await self._check_alert_conditions(tier, metrics)

    async def _update_health_status(self, tier: str, metrics: PerformanceMetrics):
        """Update health status for a tier based on metrics."""
        # Determine health status
        if (
            metrics.success_rate >= 0.95
            and metrics.avg_response_time_ms <= self.config.response_time_threshold_ms
        ):
            status = "healthy"
        elif (
            metrics.success_rate >= 0.8
            and metrics.avg_response_time_ms
            <= self.config.response_time_threshold_ms * 2
        ):
            status = "degraded"
        else:
            status = "unhealthy"

        # Create health status
        health = HealthStatus(
            component=tier,
            status=status,
            success_rate=metrics.success_rate,
            avg_response_time_ms=metrics.avg_response_time_ms,
            error_count=sum(metrics.error_types.values()),
            details={
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "cache_hit_rate": metrics.cache_hit_rate,
                "requests_per_minute": metrics.requests_per_minute,
                "error_types": metrics.error_types,
            },
        )

        self.health_status[tier] = health

    async def _check_alert_conditions(self, tier: str, metrics: PerformanceMetrics):
        """Check if any alert conditions are met."""
        alerts_to_raise = []

        # High error rate
        if metrics.success_rate < (1 - self.config.error_rate_threshold):
            alerts_to_raise.append(
                {
                    "alert_type": AlertType.HIGH_ERROR_RATE,
                    "severity": AlertSeverity.HIGH,
                    "message": f"High error rate in {tier}: {(1 - metrics.success_rate) * 100:.1f}%",
                    "metadata": {
                        "success_rate": metrics.success_rate,
                        "threshold": self.config.error_rate_threshold,
                    },
                }
            )

        # Slow response time
        if metrics.avg_response_time_ms > self.config.response_time_threshold_ms:
            alerts_to_raise.append(
                {
                    "alert_type": AlertType.SLOW_RESPONSE_TIME,
                    "severity": AlertSeverity.MEDIUM,
                    "message": f"Slow response time in {tier}: {metrics.avg_response_time_ms:.0f}ms",
                    "metadata": {
                        "response_time_ms": metrics.avg_response_time_ms,
                        "threshold": self.config.response_time_threshold_ms,
                    },
                }
            )

        # Low cache hit rate
        if metrics.cache_hit_rate < (1 - self.config.cache_miss_threshold):
            alerts_to_raise.append(
                {
                    "alert_type": AlertType.CACHE_MISS_RATE,
                    "severity": AlertSeverity.LOW,
                    "message": f"Low cache hit rate in {tier}: {metrics.cache_hit_rate * 100:.1f}%",
                    "metadata": {
                        "cache_hit_rate": metrics.cache_hit_rate,
                        "threshold": self.config.cache_miss_threshold,
                    },
                }
            )

        # Raise alerts
        for alert_data in alerts_to_raise:
            await self._raise_alert(tier=tier, **alert_data)

    async def _raise_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        tier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Raise an alert if conditions are met."""
        # Check cooldown
        cooldown_key = f"{tier}:{alert_type}"
        current_time = time.time()

        if cooldown_key in self.alert_cooldowns and (
            current_time - self.alert_cooldowns[cooldown_key]
            < self.config.alert_cooldown_seconds
        ):
            return  # Still in cooldown

        # Check rate limiting
        recent_alerts = [t for t in self.alerts_per_hour if current_time - t <= 3600]
        if len(recent_alerts) >= self.config.max_alerts_per_hour:
            logger.warning("Alert rate limit exceeded, skipping alert")
            return

        # Create alert
        alert = Alert(
            id=f"{tier}_{alert_type}_{int(current_time)}",
            severity=severity,
            alert_type=alert_type,
            tier=tier,
            message=message,
            metadata=metadata or {},
        )

        async with self.alerts_lock:
            self.alerts.append(alert)
            self.alerts_per_hour.append(current_time)
            self.alert_cooldowns[cooldown_key] = current_time

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"Alert raised: {alert.message}")

    async def _monitoring_loop(self):
        """Main monitoring loop that runs in background."""
        while self.monitoring_active:
            try:
                # Cleanup old data
                await self._cleanup_old_data()

                # Perform health checks
                await self._perform_health_checks()

                # Wait for next iteration
                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        current_time = time.time()

        async with self.metrics_lock:
            # Clean old metrics
            cutoff_time = current_time - (self.config.metrics_retention_hours * 3600)
            self.metrics_history = deque(
                [m for m in self.metrics_history if m.timestamp > cutoff_time],
                maxlen=self.metrics_history.maxlen,
            )

        async with self.alerts_lock:
            # Clean old alerts
            cutoff_time = current_time - (self.config.alert_retention_hours * 3600)
            self.alerts = deque(
                [a for a in self.alerts if a.timestamp > cutoff_time],
                maxlen=self.alerts.maxlen,
            )

    async def _perform_health_checks(self):
        """Perform periodic health checks."""
        # Update overall system health
        if self.health_status:
            unhealthy_tiers = [
                name
                for name, health in self.health_status.items()
                if health.status == "unhealthy"
            ]
            degraded_tiers = [
                name
                for name, health in self.health_status.items()
                if health.status == "degraded"
            ]

            if unhealthy_tiers:
                await self._raise_alert(
                    alert_type=AlertType.TIER_FAILURE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Unhealthy tiers detected: {', '.join(unhealthy_tiers)}",
                    metadata={"unhealthy_tiers": unhealthy_tiers},
                )
            elif degraded_tiers:
                await self._raise_alert(
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Degraded performance in tiers: {', '.join(degraded_tiers)}",
                    metadata={"degraded_tiers": degraded_tiers},
                )

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function.

        Args:
            handler: Function that takes an Alert object
        """
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status.

        Returns:
            Dictionary with system health information
        """
        total_tiers = len(self.health_status)
        healthy_tiers = sum(
            1 for h in self.health_status.values() if h.status == "healthy"
        )
        degraded_tiers = sum(
            1 for h in self.health_status.values() if h.status == "degraded"
        )
        unhealthy_tiers = sum(
            1 for h in self.health_status.values() if h.status == "unhealthy"
        )

        overall_status = "healthy"
        if unhealthy_tiers > 0:
            overall_status = "unhealthy"
        elif degraded_tiers > 0:
            overall_status = "degraded"

        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp <= 3600]

        return {
            "overall_status": overall_status,
            "tier_health": {
                "total": total_tiers,
                "healthy": healthy_tiers,
                "degraded": degraded_tiers,
                "unhealthy": unhealthy_tiers,
            },
            "recent_alerts": len(recent_alerts),
            "monitoring_active": self.monitoring_active,
            "tier_details": {
                name: health.dict() for name, health in self.health_status.items()
            },
        }

    def get_recent_metrics(
        self, tier: str | None = None, hours: int = 1
    ) -> list[PerformanceMetrics]:
        """Get recent performance metrics.

        Args:
            tier: Specific tier to filter by (None for all)
            hours: Number of hours of history to return

        Returns:
            List of performance metrics
        """
        cutoff_time = time.time() - (hours * 3600)

        metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        if tier:
            metrics = [m for m in metrics if m.tier == tier]

        return sorted(metrics, key=lambda m: m.timestamp)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active (unresolved) alerts.

        Args:
            severity: Filter by severity level

        Returns:
            List of active alerts
        """
        alerts = [a for a in self.alerts if not a.resolved]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    async def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved.

        Args:
            alert_id: ID of alert to resolve
        """
        async with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    logger.info(f"Alert resolved: {alert_id}")
                    break
