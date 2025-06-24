"""Configuration performance monitoring and metrics collection.

This module provides specialized performance monitoring for configuration operations
including load times, validation performance, auto-detection latency, and
configuration change impact analysis.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter
from opentelemetry.trace import get_current_span


logger = logging.getLogger(__name__)


@dataclass
class ConfigOperationMetrics:
    """Metrics for a configuration operation."""

    operation_type: str
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Configuration-specific metrics
    config_size_bytes: int = 0
    sections_count: int = 0
    keys_count: int = 0
    validation_errors: int = 0
    validation_warnings: int = 0

    # Auto-detection metrics
    services_detected: int = 0
    detection_confidence: float = 0.0

    # Performance metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

    # Context
    environment: Optional[str] = None
    deployment_tier: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class ConfigPerformanceStats:
    """Aggregate performance statistics for configuration operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # Latency statistics (in milliseconds)
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    # Error statistics
    error_rate: float = 0.0
    common_errors: Dict[str, int] = field(default_factory=dict)

    # Configuration size statistics
    avg_config_size_bytes: float = 0.0
    avg_sections_count: float = 0.0
    avg_keys_count: float = 0.0

    # Validation statistics
    avg_validation_errors: float = 0.0
    avg_validation_warnings: float = 0.0

    # Auto-detection statistics
    avg_services_detected: float = 0.0
    avg_detection_confidence: float = 0.0

    # Time period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class ConfigPerformanceMonitor:
    """Performance monitor specialized for configuration operations."""

    def __init__(self, max_history: int = 1000):
        """Initialize configuration performance monitor.

        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.operation_history: deque[ConfigOperationMetrics] = deque(
            maxlen=max_history
        )
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_durations: Dict[str, List[float]] = defaultdict(list)

        # OpenTelemetry metrics
        self.meter = metrics.get_meter(__name__)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize OpenTelemetry metrics instruments."""
        # Counter metrics
        self.config_operations_total = self.meter.create_counter(
            "config_operations_total",
            description="Total number of configuration operations",
            unit="1",
        )

        self.config_operation_errors_total = self.meter.create_counter(
            "config_operation_errors_total",
            description="Total number of configuration operation errors",
            unit="1",
        )

        # Histogram metrics
        self.config_operation_duration = self.meter.create_histogram(
            "config_operation_duration_seconds",
            description="Configuration operation duration",
            unit="s",
        )

        self.config_size_bytes = self.meter.create_histogram(
            "config_size_bytes",
            description="Configuration size in bytes",
            unit="bytes",
        )

        self.config_validation_duration = self.meter.create_histogram(
            "config_validation_duration_seconds",
            description="Configuration validation duration",
            unit="s",
        )

        self.config_auto_detection_duration = self.meter.create_histogram(
            "config_auto_detection_duration_seconds",
            description="Configuration auto-detection duration",
            unit="s",
        )

        # Gauge metrics (using UpDownCounter)
        self.config_sections_count = self.meter.create_up_down_counter(
            "config_sections_count",
            description="Number of configuration sections",
            unit="1",
        )

        self.config_keys_count = self.meter.create_up_down_counter(
            "config_keys_count",
            description="Number of configuration keys",
            unit="1",
        )

        self.config_validation_errors = self.meter.create_up_down_counter(
            "config_validation_errors",
            description="Number of configuration validation errors",
            unit="1",
        )

        self.config_services_detected = self.meter.create_up_down_counter(
            "config_services_detected",
            description="Number of services detected during auto-detection",
            unit="1",
        )

    def record_operation(
        self,
        operation_type: str,
        operation_name: str,
        duration_ms: float,
        success: bool,
        **kwargs,
    ) -> None:
        """Record a configuration operation.

        Args:
            operation_type: Type of operation (load, validate, update, etc.)
            operation_name: Name of the operation
            duration_ms: Operation duration in milliseconds
            success: Whether the operation succeeded
            **kwargs: Additional metrics (config_size_bytes, sections_count, etc.)
        """
        # Create operation metrics
        metrics_data = ConfigOperationMetrics(
            operation_type=operation_type,
            operation_name=operation_name,
            start_time=time.time() - (duration_ms / 1000),
            end_time=time.time(),
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )

        # Add to history
        self.operation_history.append(metrics_data)
        self.operation_counts[operation_type] += 1
        self.operation_durations[operation_type].append(duration_ms)

        # Record OpenTelemetry metrics
        labels = {
            "operation_type": operation_type,
            "operation_name": operation_name,
            "environment": kwargs.get("environment", "unknown"),
            "deployment_tier": kwargs.get("deployment_tier", "unknown"),
        }

        # Count metrics
        self.config_operations_total.add(1, labels)

        if not success:
            error_labels = {**labels, "error_type": kwargs.get("error_type", "unknown")}
            self.config_operation_errors_total.add(1, error_labels)

        # Duration metrics
        self.config_operation_duration.record(duration_ms / 1000, labels)

        # Configuration content metrics
        if kwargs.get("config_size_bytes", 0) > 0:
            self.config_size_bytes.record(kwargs["config_size_bytes"], labels)

        if kwargs.get("sections_count", 0) > 0:
            self.config_sections_count.add(kwargs["sections_count"], labels)

        if kwargs.get("keys_count", 0) > 0:
            self.config_keys_count.add(kwargs["keys_count"], labels)

        # Validation metrics
        if operation_type == "validate":
            self.config_validation_duration.record(duration_ms / 1000, labels)

            if kwargs.get("validation_errors", 0) > 0:
                self.config_validation_errors.add(kwargs["validation_errors"], labels)

        # Auto-detection metrics
        if operation_type == "auto_detect":
            self.config_auto_detection_duration.record(duration_ms / 1000, labels)

            if kwargs.get("services_detected", 0) > 0:
                self.config_services_detected.add(kwargs["services_detected"], labels)

        # Add span attributes if in an active span
        current_span = get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("config.performance.recorded", True)
            current_span.set_attribute("config.performance.duration_ms", duration_ms)

    def get_operation_stats(
        self,
        operation_type: Optional[str] = None,
        time_window_hours: int = 24,
    ) -> ConfigPerformanceStats:
        """Get performance statistics for configuration operations.

        Args:
            operation_type: Filter by operation type
            time_window_hours: Time window for statistics

        Returns:
            Performance statistics
        """
        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter operations
        filtered_ops = [
            op
            for op in self.operation_history
            if op.end_time >= cutoff_time
            and (operation_type is None or op.operation_type == operation_type)
        ]

        if not filtered_ops:
            return ConfigPerformanceStats()

        # Calculate statistics
        durations = [op.duration_ms for op in filtered_ops]
        successful_ops = [op for op in filtered_ops if op.success]
        failed_ops = [op for op in filtered_ops if not op.success]

        stats = ConfigPerformanceStats(
            total_operations=len(filtered_ops),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            error_rate=len(failed_ops) / len(filtered_ops) if filtered_ops else 0.0,
        )

        # Duration statistics
        if durations:
            stats.avg_duration_ms = sum(durations) / len(durations)
            stats.min_duration_ms = min(durations)
            stats.max_duration_ms = max(durations)

            # Calculate percentiles
            sorted_durations = sorted(durations)
            stats.p95_duration_ms = sorted_durations[int(len(sorted_durations) * 0.95)]
            stats.p99_duration_ms = sorted_durations[int(len(sorted_durations) * 0.99)]

        # Error statistics
        error_counts = defaultdict(int)
        for op in failed_ops:
            error_type = op.error_type or "unknown"
            error_counts[error_type] += 1
        stats.common_errors = dict(error_counts)

        # Configuration content statistics
        config_sizes = [
            op.config_size_bytes for op in filtered_ops if op.config_size_bytes > 0
        ]
        if config_sizes:
            stats.avg_config_size_bytes = sum(config_sizes) / len(config_sizes)

        sections_counts = [
            op.sections_count for op in filtered_ops if op.sections_count > 0
        ]
        if sections_counts:
            stats.avg_sections_count = sum(sections_counts) / len(sections_counts)

        keys_counts = [op.keys_count for op in filtered_ops if op.keys_count > 0]
        if keys_counts:
            stats.avg_keys_count = sum(keys_counts) / len(keys_counts)

        # Validation statistics
        validation_errors = [op.validation_errors for op in filtered_ops]
        if validation_errors:
            stats.avg_validation_errors = sum(validation_errors) / len(
                validation_errors
            )

        validation_warnings = [op.validation_warnings for op in filtered_ops]
        if validation_warnings:
            stats.avg_validation_warnings = sum(validation_warnings) / len(
                validation_warnings
            )

        # Auto-detection statistics
        services_detected = [
            op.services_detected for op in filtered_ops if op.services_detected > 0
        ]
        if services_detected:
            stats.avg_services_detected = sum(services_detected) / len(
                services_detected
            )

        detection_confidences = [
            op.detection_confidence
            for op in filtered_ops
            if op.detection_confidence > 0
        ]
        if detection_confidences:
            stats.avg_detection_confidence = sum(detection_confidences) / len(
                detection_confidences
            )

        # Time period
        if filtered_ops:
            stats.period_start = datetime.fromtimestamp(
                min(op.start_time for op in filtered_ops)
            )
            stats.period_end = datetime.fromtimestamp(
                max(op.end_time for op in filtered_ops)
            )

        return stats

    def get_slow_operations(
        self,
        threshold_ms: float = 1000.0,
        limit: int = 10,
    ) -> List[ConfigOperationMetrics]:
        """Get slowest configuration operations above threshold.

        Args:
            threshold_ms: Duration threshold in milliseconds
            limit: Maximum number of operations to return

        Returns:
            List of slow operations
        """
        slow_ops = [
            op for op in self.operation_history if op.duration_ms >= threshold_ms
        ]

        # Sort by duration (descending)
        slow_ops.sort(key=lambda op: op.duration_ms, reverse=True)

        return slow_ops[:limit]

    def get_error_summary(
        self,
        time_window_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get error summary for configuration operations.

        Args:
            time_window_hours: Time window for error analysis

        Returns:
            Error summary with counts and patterns
        """
        cutoff_time = time.time() - (time_window_hours * 3600)

        failed_ops = [
            op
            for op in self.operation_history
            if not op.success and op.end_time >= cutoff_time
        ]

        if not failed_ops:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}

        # Count error types
        error_types = defaultdict(int)
        for op in failed_ops:
            error_type = op.error_type or "unknown"
            error_types[error_type] += 1

        # Get recent errors
        recent_errors = sorted(failed_ops, key=lambda op: op.end_time, reverse=True)[:5]

        return {
            "total_errors": len(failed_ops),
            "error_rate": len(failed_ops) / len(self.operation_history)
            if self.operation_history
            else 0.0,
            "error_types": dict(error_types),
            "recent_errors": [
                {
                    "operation_type": op.operation_type,
                    "operation_name": op.operation_name,
                    "error_type": op.error_type,
                    "error_message": op.error_message,
                    "timestamp": datetime.fromtimestamp(op.end_time).isoformat(),
                    "duration_ms": op.duration_ms,
                }
                for op in recent_errors
            ],
        }

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.operation_history.clear()
        self.operation_counts.clear()
        self.operation_durations.clear()

    def export_metrics_summary(self) -> Dict[str, Any]:
        """Export comprehensive metrics summary.

        Returns:
            Dictionary with all metrics and statistics
        """
        # Overall statistics
        overall_stats = self.get_operation_stats()

        # Per-operation-type statistics
        operation_types = set(op.operation_type for op in self.operation_history)
        per_type_stats = {
            op_type: self.get_operation_stats(operation_type=op_type)
            for op_type in operation_types
        }

        # Error summary
        error_summary = self.get_error_summary()

        # Slow operations
        slow_operations = self.get_slow_operations()

        return {
            "timestamp": datetime.now().isoformat(),
            "total_operations": len(self.operation_history),
            "overall_stats": overall_stats.__dict__,
            "per_type_stats": {
                op_type: stats.__dict__ for op_type, stats in per_type_stats.items()
            },
            "error_summary": error_summary,
            "slow_operations": [
                {
                    "operation_type": op.operation_type,
                    "operation_name": op.operation_name,
                    "duration_ms": op.duration_ms,
                    "timestamp": datetime.fromtimestamp(op.end_time).isoformat(),
                }
                for op in slow_operations
            ],
        }


# Global configuration performance monitor instance
_config_performance_monitor: Optional[ConfigPerformanceMonitor] = None


def get_config_performance_monitor() -> ConfigPerformanceMonitor:
    """Get the global configuration performance monitor instance.

    Returns:
        ConfigPerformanceMonitor instance
    """
    global _config_performance_monitor
    if _config_performance_monitor is None:
        _config_performance_monitor = ConfigPerformanceMonitor()
    return _config_performance_monitor


def record_config_operation(
    operation_type: str,
    operation_name: str,
    duration_ms: float,
    success: bool,
    **kwargs,
) -> None:
    """Record a configuration operation in the global performance monitor.

    Args:
        operation_type: Type of operation (load, validate, update, etc.)
        operation_name: Name of the operation
        duration_ms: Operation duration in milliseconds
        success: Whether the operation succeeded
        **kwargs: Additional metrics
    """
    monitor = get_config_performance_monitor()
    monitor.record_operation(
        operation_type=operation_type,
        operation_name=operation_name,
        duration_ms=duration_ms,
        success=success,
        **kwargs,
    )


def get_config_performance_stats(
    operation_type: Optional[str] = None,
    time_window_hours: int = 24,
) -> ConfigPerformanceStats:
    """Get configuration performance statistics.

    Args:
        operation_type: Filter by operation type
        time_window_hours: Time window for statistics

    Returns:
        Performance statistics
    """
    monitor = get_config_performance_monitor()
    return monitor.get_operation_stats(operation_type, time_window_hours)


def get_config_error_summary(time_window_hours: int = 24) -> Dict[str, Any]:
    """Get configuration error summary.

    Args:
        time_window_hours: Time window for error analysis

    Returns:
        Error summary
    """
    monitor = get_config_performance_monitor()
    return monitor.get_error_summary(time_window_hours)
