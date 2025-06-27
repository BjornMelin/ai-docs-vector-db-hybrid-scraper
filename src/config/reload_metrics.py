"""Prometheus metrics for configuration reload monitoring.

This module provides detailed metrics collection for configuration reload operations,
including timing, success rates, and performance characteristics.
"""

import time
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Summary


# Reload operation metrics
reload_total = Counter(
    "config_reload_total",
    "Total number of configuration reload attempts",
    ["trigger", "status"],
)

reload_duration = Histogram(
    "config_reload_duration_seconds",
    "Configuration reload duration in seconds",
    ["trigger", "phase"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

reload_success_rate = Gauge(
    "config_reload_success_rate",
    "Configuration reload success rate (rolling window)",
    ["time_window"],
)

# Validation metrics
validation_duration = Histogram(
    "config_validation_duration_seconds",
    "Configuration validation duration in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

validation_errors = Counter(
    "config_validation_errors_total",
    "Total number of configuration validation errors",
    ["error_type"],
)

# Change notification metrics
listener_notification_duration = Histogram(
    "config_listener_notification_duration_seconds",
    "Duration of configuration change listener notifications",
    ["listener_name", "status"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
)

listener_success_rate = Gauge(
    "config_listener_success_rate",
    "Configuration change listener success rate",
    ["listener_name"],
)

# Backup and rollback metrics
config_backups = Gauge(
    "config_backups_total", "Total number of configuration backups maintained"
)

rollback_operations = Counter(
    "config_rollback_total",
    "Total number of configuration rollback operations",
    ["status", "reason"],
)

# File watching metrics
file_watch_checks = Counter(
    "config_file_watch_checks_total",
    "Total number of file watch check operations",
    ["file_path", "changed"],
)

file_watch_latency = Summary(
    "config_file_watch_latency_seconds", "File watch check latency in seconds"
)

# Memory usage metrics
config_memory_usage = Gauge(
    "config_memory_usage_bytes", "Memory usage of configuration objects", ["component"]
)

snapshot_memory_usage = Gauge(
    "config_snapshot_memory_bytes",
    "Memory usage of configuration snapshots",
    ["source"],
)

# Performance target metrics
performance_target_violations = Counter(
    "config_performance_violations_total",
    "Number of times performance targets were violated",
    ["target_type", "threshold_ms"],
)

# Sub-100ms tracking
sub_100ms_reloads = Gauge(
    "config_reload_sub_100ms_percentage",
    "Percentage of configuration reloads completing under 100ms",
)


class ReloadMetricsCollector:
    """Collects and tracks configuration reload metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._reload_times = []  # Track last N reload times
        self._max_history = 100
        self._listener_stats: dict[str, dict[str, int]] = {}

    def record_reload_attempt(self, trigger: str, status: str) -> None:
        """Record a configuration reload attempt.

        Args:
            trigger: Reload trigger type
            status: Reload status
        """
        reload_total.labels(trigger=trigger, status=status).inc()

    def record_reload_duration(
        self, trigger: str, phase: str, duration_seconds: float
    ) -> None:
        """Record reload duration for a specific phase.

        Args:
            trigger: Reload trigger type
            phase: Reload phase (validation, apply, total)
            duration_seconds: Duration in seconds
        """
        reload_duration.labels(trigger=trigger, phase=phase).observe(duration_seconds)

        # Track for sub-100ms percentage
        if phase == "total":
            self._reload_times.append(duration_seconds)
            if len(self._reload_times) > self._max_history:
                self._reload_times.pop(0)

            # Update sub-100ms percentage
            sub_100ms_count = sum(1 for t in self._reload_times if t < 0.1)
            if self._reload_times:
                percentage = (sub_100ms_count / len(self._reload_times)) * 100
                sub_100ms_reloads.set(percentage)

            # Check performance target
            if duration_seconds > 0.1:  # 100ms
                performance_target_violations.labels(
                    target_type="reload_latency", threshold_ms="100"
                ).inc()

    def record_validation_duration(self, duration_seconds: float) -> None:
        """Record configuration validation duration.

        Args:
            duration_seconds: Validation duration in seconds
        """
        validation_duration.observe(duration_seconds)

        # Check performance target
        if duration_seconds > 0.05:  # 50ms
            performance_target_violations.labels(
                target_type="validation_latency", threshold_ms="50"
            ).inc()

    def record_validation_error(self, error_type: str) -> None:
        """Record a validation error.

        Args:
            error_type: Type of validation error
        """
        validation_errors.labels(error_type=error_type).inc()

    def record_listener_notification(
        self, listener_name: str, status: str, duration_seconds: float
    ) -> None:
        """Record listener notification metrics.

        Args:
            listener_name: Name of the listener
            status: Notification status (success, failed, timeout)
            duration_seconds: Notification duration
        """
        listener_notification_duration.labels(
            listener_name=listener_name, status=status
        ).observe(duration_seconds)

        # Track listener statistics
        if listener_name not in self._listener_stats:
            self._listener_stats[listener_name] = {"success": 0, "total": 0}

        self._listener_stats[listener_name]["total"] += 1
        if status == "success":
            self._listener_stats[listener_name]["success"] += 1

        # Update success rate
        stats = self._listener_stats[listener_name]
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            listener_success_rate.labels(listener_name=listener_name).set(success_rate)

    def update_backup_count(self, count: int) -> None:
        """Update configuration backup count.

        Args:
            count: Current number of backups
        """
        config_backups.set(count)

    def record_rollback(self, status: str, reason: str) -> None:
        """Record a configuration rollback operation.

        Args:
            status: Rollback status
            reason: Reason for rollback
        """
        rollback_operations.labels(status=status, reason=reason).inc()

    def record_file_watch_check(self, file_path: str, changed: bool) -> None:
        """Record a file watch check operation.

        Args:
            file_path: Path of file being watched
            changed: Whether file changed
        """
        file_watch_checks.labels(
            file_path=file_path, changed=str(changed).lower()
        ).inc()

    @contextmanager
    def measure_file_watch_latency(self):
        """Context manager to measure file watch latency."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            file_watch_latency.observe(duration)

    def update_memory_usage(self, component: str, bytes_used: int) -> None:
        """Update memory usage metrics.

        Args:
            component: Component name
            bytes_used: Memory usage in bytes
        """
        config_memory_usage.labels(component=component).set(bytes_used)

    def update_snapshot_memory(self, source: str, bytes_used: int) -> None:
        """Update snapshot memory usage.

        Args:
            source: Configuration source
            bytes_used: Memory usage in bytes
        """
        snapshot_memory_usage.labels(source=source).set(bytes_used)

    def get_reload_statistics(self) -> dict[str, Any]:
        """Get comprehensive reload statistics.

        Returns:
            Dictionary of reload statistics
        """
        if not self._reload_times:
            return {
                "total_reloads": 0,
                "sub_100ms_percentage": 0.0,
                "average_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "p99_duration_ms": 0.0,
            }

        sorted_times = sorted(self._reload_times)
        total = len(sorted_times)

        # Calculate percentiles
        p95_index = int(total * 0.95)
        p99_index = int(total * 0.99)

        return {
            "total_reloads": total,
            "sub_100ms_percentage": (
                sum(1 for t in self._reload_times if t < 0.1) / total
            )
            * 100,
            "average_duration_ms": (sum(self._reload_times) / total) * 1000,
            "p95_duration_ms": sorted_times[p95_index] * 1000
            if p95_index < total
            else 0,
            "p99_duration_ms": sorted_times[p99_index] * 1000
            if p99_index < total
            else 0,
            "min_duration_ms": min(self._reload_times) * 1000,
            "max_duration_ms": max(self._reload_times) * 1000,
        }


# Global metrics collector instance
_metrics_collector: ReloadMetricsCollector | None = None


def get_reload_metrics_collector() -> ReloadMetricsCollector:
    """Get the global reload metrics collector instance.

    Returns:
        Global metrics collector
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = ReloadMetricsCollector()
    return _metrics_collector


@contextmanager
def track_reload_operation(trigger: str):
    """Context manager to track a complete reload operation.

    Args:
        trigger: Reload trigger type

    Yields:
        Dictionary to store operation metadata
    """
    collector = get_reload_metrics_collector()
    start_time = time.time()
    metadata = {"phases": {}}

    try:
        yield metadata

        # Record successful reload
        duration = time.time() - start_time
        collector.record_reload_attempt(trigger, "success")
        collector.record_reload_duration(trigger, "total", duration)

    except Exception as e:
        # Record failed reload
        duration = time.time() - start_time
        collector.record_reload_attempt(trigger, "failed")
        collector.record_reload_duration(trigger, "total", duration)
        raise


@contextmanager
def track_reload_phase(phase_name: str, trigger: str, metadata: dict[str, Any]):
    """Context manager to track a specific reload phase.

    Args:
        phase_name: Name of the phase
        trigger: Reload trigger type
        metadata: Operation metadata dictionary

    Yields:
        None
    """
    collector = get_reload_metrics_collector()
    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        metadata["phases"][phase_name] = duration
        collector.record_reload_duration(trigger, phase_name, duration)
