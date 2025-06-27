"""Performance monitoring and analysis for AI/ML operations and system resources.

This module provides comprehensive performance monitoring including database queries,
external API calls, resource utilization, and AI/ML operation performance analysis
with automated performance degradation detection and alerting.
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

import psutil
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .instrumentation import get_tracer
from .metrics_bridge import get_metrics_bridge


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    operation_name: str
    duration_ms: float
    cpu_usage_percent: float | None = None
    memory_usage_mb: float | None = None
    disk_io_mb: float | None = None
    network_io_mb: float | None = None
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for monitoring."""

    max_duration_ms: float = 30000  # 30 seconds
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 1024.0
    max_error_rate: float = 0.05  # 5%
    min_throughput_ops_per_sec: float = 1.0


class PerformanceMonitor:
    """Advanced performance monitoring and analysis."""

    def __init__(self, thresholds: PerformanceThresholds | None = None):
        """Initialize performance monitor.

        Args:
            thresholds: Performance thresholds for alerting
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.tracer = get_tracer()

        # Performance history for trend analysis
        self._operation_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._error_counts: dict[str, int] = defaultdict(int)
        self._success_counts: dict[str, int] = defaultdict(int)

        # System resource monitoring
        self._last_cpu_check = time.time()
        self._last_memory_check = time.time()
        self._cpu_history = deque(maxlen=60)  # Last 60 readings
        self._memory_history = deque(maxlen=60)

    def _get_system_metrics(self) -> dict[str, float]:
        """Get current system performance metrics.

        Returns:
            Dictionary of system metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

            # Network I/O metrics
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0

            return {
                "cpu_percent": cpu_percent,
                "memory_used_mb": memory_used_mb,
                "memory_percent": memory_percent,
                "disk_read_mb": disk_read_mb,
                "disk_write_mb": disk_write_mb,
                "network_sent_mb": net_sent_mb,
                "network_recv_mb": net_recv_mb,
            }

        except Exception:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}

    @contextmanager
    def monitor_operation(
        self,
        operation_name: str,
        category: str = "general",
        track_resources: bool = True,
        alert_on_threshold: bool = True,
    ):
        """Context manager for monitoring operation performance.

        Args:
            operation_name: Name of the operation
            category: Operation category for grouping
            track_resources: Whether to track system resources
            alert_on_threshold: Whether to alert on threshold violations

        Yields:
            Performance metrics dictionary for updating
        """
        start_time = time.time()
        self._get_system_metrics() if track_resources else {}

        performance_data = {"custom_metrics": {}}

        with self.tracer.start_as_current_span(f"performance.{operation_name}") as span:
            span.set_attribute("performance.operation_name", operation_name)
            span.set_attribute("performance.category", category)
            span.set_attribute("performance.tracking_resources", track_resources)

            try:
                yield performance_data

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Get end metrics
                end_metrics = self._get_system_metrics() if track_resources else {}

                # Calculate resource usage
                cpu_usage = end_metrics.get("cpu_percent")
                memory_usage = end_metrics.get("memory_used_mb")

                # Create performance metrics
                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_mb=memory_usage,
                    success=True,
                    metadata=performance_data.get("custom_metrics", {}),
                )

                # Record metrics
                self._record_performance_metrics(metrics, category)

                # Set span attributes
                span.set_attribute("performance.duration_ms", duration_ms)
                if cpu_usage:
                    span.set_attribute("performance.cpu_percent", cpu_usage)
                if memory_usage:
                    span.set_attribute("performance.memory_mb", memory_usage)

                # Check thresholds and alert if necessary
                if alert_on_threshold:
                    self._check_thresholds(metrics, span)

                span.set_status(Status(StatusCode.OK))

            except Exception:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self._record_performance_metrics(error_metrics, category)

                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                raise

    @asynccontextmanager
    async def monitor_async_operation(
        self,
        operation_name: str,
        category: str = "general",
        track_resources: bool = True,
        alert_on_threshold: bool = True,
    ):
        """Async context manager for monitoring operation performance.

        Args:
            operation_name: Name of the operation
            category: Operation category
            track_resources: Whether to track system resources
            alert_on_threshold: Whether to alert on threshold violations

        Yields:
            Performance metrics dictionary for updating
        """
        start_time = time.time()
        self._get_system_metrics() if track_resources else {}

        performance_data = {"custom_metrics": {}}

        with self.tracer.start_as_current_span(f"performance.{operation_name}") as span:
            span.set_attribute("performance.operation_name", operation_name)
            span.set_attribute("performance.category", category)

            try:
                yield performance_data

                # Calculate performance metrics (same as sync version)
                duration_ms = (time.time() - start_time) * 1000
                end_metrics = self._get_system_metrics() if track_resources else {}

                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    cpu_usage_percent=end_metrics.get("cpu_percent"),
                    memory_usage_mb=end_metrics.get("memory_used_mb"),
                    success=True,
                    metadata=performance_data.get("custom_metrics", {}),
                )

                self._record_performance_metrics(metrics, category)

                # Set span attributes
                span.set_attribute("performance.duration_ms", duration_ms)

                if alert_on_threshold:
                    self._check_thresholds(metrics, span)

                span.set_status(Status(StatusCode.OK))

            except Exception:
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self._record_performance_metrics(error_metrics, category)

                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                raise

    def _record_performance_metrics(
        self, metrics: PerformanceMetrics, category: str
    ) -> None:
        """Record performance metrics for analysis.

        Args:
            metrics: Performance metrics to record
            category: Operation category
        """
        # Add to operation history
        self._operation_history[metrics.operation_name].append(metrics)

        # Update counts
        if metrics.success:
            self._success_counts[metrics.operation_name] += 1
        else:
            self._error_counts[metrics.operation_name] += 1

        # Record metrics via bridge
        try:
            bridge = get_metrics_bridge()
            bridge.record_batch_metrics(
                {
                    "operation_duration": {
                        "value": metrics.duration_ms,
                        "labels": {
                            "operation": metrics.operation_name,
                            "category": category,
                            "success": str(metrics.success),
                        },
                    }
                }
            )

            if metrics.cpu_usage_percent:
                bridge.record_batch_metrics(
                    {
                        "operation_cpu_usage": {
                            "value": metrics.cpu_usage_percent,
                            "labels": {
                                "operation": metrics.operation_name,
                                "category": category,
                            },
                        }
                    }
                )

        except Exception:
            logger.warning(f"Failed to record performance metrics: {e}")

    def _check_thresholds(self, metrics: PerformanceMetrics, span: trace.Span) -> None:
        """Check performance thresholds and create alerts.

        Args:
            metrics: Performance metrics to check
            span: Current span for recording alerts
        """
        alerts = []

        # Check duration threshold
        if metrics.duration_ms > self.thresholds.max_duration_ms:
            alerts.append(
                f"Duration {metrics.duration_ms:.1f}ms exceeds threshold {self.thresholds.max_duration_ms:.1f}ms"
            )

        # Check CPU threshold
        if (
            metrics.cpu_usage_percent
            and metrics.cpu_usage_percent > self.thresholds.max_cpu_percent
        ):
            alerts.append(
                f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds threshold {self.thresholds.max_cpu_percent:.1f}%"
            )

        # Check memory threshold
        if (
            metrics.memory_usage_mb
            and metrics.memory_usage_mb > self.thresholds.max_memory_mb
        ):
            alerts.append(
                f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds threshold {self.thresholds.max_memory_mb:.1f}MB"
            )

        # Check error rate
        total_ops = (
            self._success_counts[metrics.operation_name]
            + self._error_counts[metrics.operation_name]
        )
        if total_ops > 10:  # Only check error rate after sufficient operations
            error_rate = self._error_counts[metrics.operation_name] / total_ops
            if error_rate > self.thresholds.max_error_rate:
                alerts.append(
                    f"Error rate {error_rate:.3f} exceeds threshold {self.thresholds.max_error_rate:.3f}"
                )

        # Record alerts
        if alerts:
            for alert in alerts:
                span.add_event("performance_threshold_violation", {"alert": alert})
                logger.warning(
                    f"Performance alert for {metrics.operation_name}: {alert}"
                )

            span.set_attribute("performance.alerts_count", len(alerts))

    def get_operation_statistics(self, operation_name: str) -> dict[str, Any]:
        """Get performance statistics for an operation.

        Args:
            operation_name: Name of operation

        Returns:
            Dictionary of performance statistics
        """
        history = self._operation_history.get(operation_name, deque())
        if not history:
            return {"error": "No performance data available"}

        # Calculate statistics
        durations = [m.duration_ms for m in history]
        cpu_usages = [m.cpu_usage_percent for m in history if m.cpu_usage_percent]
        memory_usages = [m.memory_usage_mb for m in history if m.memory_usage_mb]

        stats = {
            "operation_name": operation_name,
            "total_operations": len(history),
            "success_count": self._success_counts[operation_name],
            "error_count": self._error_counts[operation_name],
            "error_rate": self._error_counts[operation_name] / max(len(history), 1),
            "duration_stats": {
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": sum(durations) / len(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)]
                if durations
                else 0,
                "p99_ms": sorted(durations)[int(len(durations) * 0.99)]
                if durations
                else 0,
            },
        }

        if cpu_usages:
            stats["cpu_stats"] = {
                "avg_percent": sum(cpu_usages) / len(cpu_usages),
                "max_percent": max(cpu_usages),
            }

        if memory_usages:
            stats["memory_stats"] = {
                "avg_mb": sum(memory_usages) / len(memory_usages),
                "max_mb": max(memory_usages),
            }

        return stats

    def get_system_performance_summary(self) -> dict[str, Any]:
        """Get overall system performance summary.

        Returns:
            System performance summary
        """
        current_metrics = self._get_system_metrics()

        summary = {
            "timestamp": time.time(),
            "current_metrics": current_metrics,
            "operation_count": len(self._operation_history),
            "total_operations": sum(
                len(history) for history in self._operation_history.values()
            ),
            "total_errors": sum(self._error_counts.values()),
            "overall_error_rate": sum(self._error_counts.values())
            / max(sum(len(history) for history in self._operation_history.values()), 1),
        }

        # Add trending information
        if len(self._cpu_history) > 1:
            summary["cpu_trend"] = {
                "current": current_metrics.get("cpu_percent", 0),
                "avg_last_minute": sum(self._cpu_history) / len(self._cpu_history),
            }

        if len(self._memory_history) > 1:
            summary["memory_trend"] = {
                "current_mb": current_metrics.get("memory_used_mb", 0),
                "avg_last_minute_mb": sum(self._memory_history)
                / len(self._memory_history),
            }

        return summary

    def monitor_database_query(self, query_type: str = "select"):
        """Context manager for monitoring database query performance.

        Args:
            query_type: Type of database query

        Returns:
            Context manager for query monitoring
        """
        return self.monitor_operation(
            f"database_{query_type}", category="database", track_resources=True
        )

    def monitor_external_api_call(self, api_name: str, endpoint: str):
        """Context manager for monitoring external API call performance.

        Args:
            api_name: Name of the API service
            endpoint: API endpoint being called

        Returns:
            Context manager for API monitoring
        """
        return self.monitor_operation(
            f"api_{api_name}_{endpoint}", category="external_api", track_resources=False
        )

    def monitor_ai_model_inference(self, model_name: str, provider: str):
        """Context manager for monitoring AI model inference performance.

        Args:
            model_name: Name of the AI model
            provider: AI service provider

        Returns:
            Context manager for AI inference monitoring
        """
        return self.monitor_operation(
            f"ai_inference_{provider}_{model_name}",
            category="ai_inference",
            track_resources=True,
        )

    def monitor_cache_operation(self, cache_type: str, operation: str):
        """Context manager for monitoring cache operation performance.

        Args:
            cache_type: Type of cache (redis, local, etc.)
            operation: Cache operation (get, set, delete, etc.)

        Returns:
            Context manager for cache monitoring
        """
        return self.monitor_operation(
            f"cache_{cache_type}_{operation}", category="cache", track_resources=False
        )


# Global performance monitor instance
_performance_monitor: PerformanceMonitor | None = None


def initialize_performance_monitor(
    thresholds: PerformanceThresholds | None = None,
) -> PerformanceMonitor:
    """Initialize global performance monitor.

    Args:
        thresholds: Performance thresholds

    Returns:
        Initialized performance monitor
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(thresholds)
    return _performance_monitor


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.

    Returns:
        Global performance monitor

    Raises:
        RuntimeError: If monitor not initialized
    """
    if _performance_monitor is None:
        raise RuntimeError(
            "Performance monitor not initialized. Call initialize_performance_monitor() first."
        )
    return _performance_monitor


# Convenience functions
def monitor_operation(operation_name: str, category: str = "general", **kwargs):
    """Monitor operation using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_operation(operation_name, category, **kwargs)


def monitor_async_operation(operation_name: str, category: str = "general", **kwargs):
    """Monitor async operation using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_operation(operation_name, category, **kwargs)


def monitor_database_query(query_type: str = "select"):
    """Monitor database query using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_database_query(query_type)


def monitor_external_api_call(api_name: str, endpoint: str):
    """Monitor external API call using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_external_api_call(api_name, endpoint)


def monitor_ai_model_inference(model_name: str, provider: str):
    """Monitor AI model inference using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_ai_model_inference(model_name, provider)


def get_operation_statistics(operation_name: str) -> dict[str, Any]:
    """Get operation statistics using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.get_operation_statistics(operation_name)


def get_system_performance_summary() -> dict[str, Any]:
    """Get system performance summary using global performance monitor."""
    monitor = get_performance_monitor()
    return monitor.get_system_performance_summary()
