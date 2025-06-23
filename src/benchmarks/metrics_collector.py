"""Metrics collection and aggregation for benchmarking.

This module provides comprehensive metrics collection capabilities
for performance analysis and monitoring integration.
"""

import json
import logging
import statistics
import time
from collections import defaultdict
from collections import deque
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ..config import Config

logger = logging.getLogger(__name__)


class MetricPoint(BaseModel):
    """Single metric measurement point."""

    timestamp: float = Field(..., description="Unix timestamp")
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    labels: dict[str, str] = Field(default_factory=dict, description="Metric labels")
    source: str = Field(..., description="Source component")


class MetricSummary(BaseModel):
    """Summary statistics for a metric."""

    metric_name: str = Field(..., description="Name of the metric")
    sample_count: int = Field(..., description="Number of samples")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    avg_value: float = Field(..., description="Average value")
    median_value: float = Field(..., description="Median value")
    p95_value: float = Field(..., description="95th percentile value")
    p99_value: float = Field(..., description="99th percentile value")
    std_deviation: float = Field(..., description="Standard deviation")
    first_timestamp: float = Field(..., description="First measurement timestamp")
    last_timestamp: float = Field(..., description="Last measurement timestamp")


class MetricsCollector:
    """Advanced metrics collection and aggregation system."""

    def __init__(self, config: Config, max_points: int = 10000):
        """Initialize metrics collector.

        Args:
            config: Unified configuration
            max_points: Maximum metric points to keep in memory
        """
        self.config = config
        self.max_points = max_points

        # Storage for metric points
        self.metric_points: list[MetricPoint] = []
        self.metric_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Real-time aggregation windows
        self.real_time_windows = {
            "1m": deque(maxlen=60),  # 1-minute window
            "5m": deque(maxlen=300),  # 5-minute window
            "15m": deque(maxlen=900),  # 15-minute window
        }

        # Collection state
        self.collection_active = False
        self.collection_start_time = None

        # Prometheus-style metrics for integration
        self.prometheus_metrics = {}

    async def start_collection(self) -> None:
        """Start metrics collection."""
        self.collection_active = True
        self.collection_start_time = time.time()
        self.metric_points = []

        logger.info("Started metrics collection")

    async def stop_collection(self) -> dict[str, Any]:
        """Stop metrics collection and return summary.

        Returns:
            Collection summary with aggregated metrics
        """
        self.collection_active = False
        collection_duration = time.time() - (self.collection_start_time or time.time())

        # Generate summary
        summary = await self.generate_collection_summary()
        summary["collection_duration_seconds"] = collection_duration
        summary["total_metric_points"] = len(self.metric_points)

        logger.info(
            f"Stopped metrics collection. Duration: {collection_duration:.2f}s, Points: {len(self.metric_points)}"
        )

        return summary

    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None,
        source: str = "unknown",
        timestamp: float | None = None,
    ) -> None:
        """Record a single metric point.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
            source: Source component that generated the metric
            timestamp: Optional timestamp (uses current time if not provided)
        """
        if not self.collection_active:
            return

        point = MetricPoint(
            timestamp=timestamp or time.time(),
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            source=source,
        )

        # Add to main storage
        self.metric_points.append(point)

        # Add to metric-specific buffer
        self.metric_buffers[metric_name].append(point)

        # Update real-time windows
        self._update_real_time_windows(point)

        # Trim storage if needed
        if len(self.metric_points) > self.max_points:
            self.metric_points = self.metric_points[-self.max_points :]

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        labels: dict[str, str] | None = None,
        source: str = "search_service",
    ) -> None:
        """Record operation latency metric.

        Args:
            operation: Name of the operation
            latency_ms: Latency in milliseconds
            labels: Optional labels
            source: Source component
        """
        metric_name = f"{operation}_latency_ms"
        self.record_metric(metric_name, latency_ms, labels, source)

    def record_throughput(
        self,
        operation: str,
        requests_per_second: float,
        labels: dict[str, str] | None = None,
        source: str = "search_service",
    ) -> None:
        """Record throughput metric.

        Args:
            operation: Name of the operation
            requests_per_second: Throughput in requests per second
            labels: Optional labels
            source: Source component
        """
        metric_name = f"{operation}_throughput_qps"
        self.record_metric(metric_name, requests_per_second, labels, source)

    def record_error_rate(
        self,
        operation: str,
        error_rate: float,
        labels: dict[str, str] | None = None,
        source: str = "search_service",
    ) -> None:
        """Record error rate metric.

        Args:
            operation: Name of the operation
            error_rate: Error rate (0.0 - 1.0)
            labels: Optional labels
            source: Source component
        """
        metric_name = f"{operation}_error_rate"
        self.record_metric(metric_name, error_rate, labels, source)

    def record_resource_usage(
        self,
        resource_type: str,
        usage_value: float,
        unit: str = "count",
        labels: dict[str, str] | None = None,
        source: str = "system",
    ) -> None:
        """Record resource usage metric.

        Args:
            resource_type: Type of resource (cpu, memory, disk, etc.)
            usage_value: Usage value
            unit: Unit of measurement
            labels: Optional labels
            source: Source component
        """
        metric_name = f"{resource_type}_usage_{unit}"
        self.record_metric(metric_name, usage_value, labels, source)

    def record_custom_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        labels: dict[str, str] | None = None,
        source: str = "custom",
    ) -> None:
        """Record custom metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (gauge, counter, histogram)
            labels: Optional labels
            source: Source component
        """
        # Add metric type to labels
        if labels is None:
            labels = {}
        labels["metric_type"] = metric_type

        self.record_metric(metric_name, value, labels, source)

    async def get_metric_summary(self, metric_name: str) -> MetricSummary | None:
        """Get summary statistics for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric summary or None if metric not found
        """
        points = [p for p in self.metric_points if p.metric_name == metric_name]

        if not points:
            return None

        values = [p.value for p in points]

        return MetricSummary(
            metric_name=metric_name,
            sample_count=len(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=self._percentile(values, 95),
            p99_value=self._percentile(values, 99),
            std_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
            first_timestamp=points[0].timestamp,
            last_timestamp=points[-1].timestamp,
        )

    async def get_all_metric_summaries(self) -> dict[str, MetricSummary]:
        """Get summary statistics for all collected metrics.

        Returns:
            Dictionary mapping metric names to summaries
        """
        metric_names = {p.metric_name for p in self.metric_points}
        summaries = {}

        for metric_name in metric_names:
            summary = await self.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary

        return summaries

    async def generate_collection_summary(self) -> dict[str, Any]:
        """Generate comprehensive collection summary.

        Returns:
            Summary of collected metrics and analysis
        """
        summaries = await self.get_all_metric_summaries()

        # Categorize metrics
        latency_metrics = {k: v for k, v in summaries.items() if "latency" in k}
        throughput_metrics = {k: v for k, v in summaries.items() if "throughput" in k}
        error_metrics = {k: v for k, v in summaries.items() if "error" in k}
        resource_metrics = {k: v for k, v in summaries.items() if "usage" in k}

        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics()

        # Identify anomalies
        anomalies = self._detect_anomalies()

        return {
            "metric_summaries": {k: v.model_dump() for k, v in summaries.items()},
            "categorized_metrics": {
                "latency": {k: v.model_dump() for k, v in latency_metrics.items()},
                "throughput": {
                    k: v.model_dump() for k, v in throughput_metrics.items()
                },
                "errors": {k: v.model_dump() for k, v in error_metrics.items()},
                "resources": {k: v.model_dump() for k, v in resource_metrics.items()},
            },
            "overall_statistics": overall_stats,
            "anomalies": anomalies,
            "collection_metadata": {
                "start_time": self.collection_start_time,
                "unique_metrics": len(summaries),
                "unique_sources": len({p.source for p in self.metric_points}),
            },
        }

    def get_real_time_metrics(self, window: str = "1m") -> dict[str, Any]:
        """Get real-time metrics for a specific time window.

        Args:
            window: Time window ("1m", "5m", "15m")

        Returns:
            Real-time metrics for the window
        """
        if window not in self.real_time_windows:
            return {}

        window_points = list(self.real_time_windows[window])

        if not window_points:
            return {}

        # Group by metric name
        metrics_by_name = defaultdict(list)
        for point in window_points:
            metrics_by_name[point.metric_name].append(point.value)

        # Calculate window statistics
        window_stats = {}
        for metric_name, values in metrics_by_name.items():
            window_stats[metric_name] = {
                "current": values[-1] if values else 0,
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "samples": len(values),
            }

        return {"window": window, "timestamp": time.time(), "metrics": window_stats}

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        prometheus_lines = []

        # Get latest metrics grouped by name
        latest_metrics = {}
        for point in reversed(self.metric_points):
            if point.metric_name not in latest_metrics:
                latest_metrics[point.metric_name] = point

        # Convert to Prometheus format
        for metric_name, point in latest_metrics.items():
            # Clean metric name for Prometheus
            prom_metric_name = metric_name.replace(".", "_").replace("-", "_")

            # Add help text
            prometheus_lines.append(
                f"# HELP {prom_metric_name} {metric_name} from {point.source}"
            )
            prometheus_lines.append(f"# TYPE {prom_metric_name} gauge")

            # Add labels
            label_str = ""
            if point.labels:
                labels = [f'{k}="{v}"' for k, v in point.labels.items()]
                label_str = "{" + ",".join(labels) + "}"

            prometheus_lines.append(f"{prom_metric_name}{label_str} {point.value}")

        return "\n".join(prometheus_lines)

    def export_json_metrics(self) -> str:
        """Export metrics in JSON format.

        Returns:
            JSON-formatted metrics string
        """
        export_data = {
            "timestamp": time.time(),
            "collection_start": self.collection_start_time,
            "total_points": len(self.metric_points),
            "metrics": [point.model_dump() for point in self.metric_points],
        }

        return json.dumps(export_data, indent=2)

    def _update_real_time_windows(self, point: MetricPoint) -> None:
        """Update real-time metric windows."""
        current_time = time.time()

        # Add to all windows
        for window_deque in self.real_time_windows.values():
            window_deque.append(point)

        # Clean old points from windows
        window_seconds = {"1m": 60, "5m": 300, "15m": 900}

        for window_name, seconds in window_seconds.items():
            window_deque = self.real_time_windows[window_name]
            while window_deque and current_time - window_deque[0].timestamp > seconds:
                window_deque.popleft()

    def _calculate_overall_statistics(self) -> dict[str, Any]:
        """Calculate overall collection statistics."""
        if not self.metric_points:
            return {}

        # Time range
        timestamps = [p.timestamp for p in self.metric_points]
        time_range = max(timestamps) - min(timestamps)

        # Source distribution
        sources = [p.source for p in self.metric_points]
        source_counts = {source: sources.count(source) for source in set(sources)}

        # Metric distribution
        metric_names = [p.metric_name for p in self.metric_points]
        metric_counts = {name: metric_names.count(name) for name in set(metric_names)}

        return {
            "collection_time_range_seconds": time_range,
            "points_per_second": len(self.metric_points) / max(time_range, 1),
            "source_distribution": source_counts,
            "metric_distribution": metric_counts,
            "most_frequent_metric": max(metric_counts.items(), key=lambda x: x[1])[0]
            if metric_counts
            else None,
        }

    def _detect_anomalies(self) -> list[dict[str, Any]]:
        """Detect anomalies in collected metrics."""
        anomalies = []

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for point in self.metric_points:
            metrics_by_name[point.metric_name].append(point.value)

        # Check for anomalies in each metric
        for metric_name, values in metrics_by_name.items():
            if len(values) < 10:  # Need enough samples
                continue

            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0

            # Check for outliers (values > 3 standard deviations from mean)
            outliers = [v for v in values if abs(v - mean_val) > 3 * std_val]

            if outliers:
                anomalies.append(
                    {
                        "metric_name": metric_name,
                        "anomaly_type": "outliers",
                        "outlier_count": len(outliers),
                        "outlier_values": outliers[:5],  # Show first 5
                        "mean": mean_val,
                        "std_deviation": std_val,
                    }
                )

            # Check for sudden spikes (values > 10x median)
            median_val = statistics.median(values)
            spikes = [v for v in values if v > 10 * median_val]

            if spikes:
                anomalies.append(
                    {
                        "metric_name": metric_name,
                        "anomaly_type": "spikes",
                        "spike_count": len(spikes),
                        "spike_values": spikes[:5],
                        "median": median_val,
                    }
                )

        return anomalies

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metric_points = []
        self.metric_buffers.clear()
        for window in self.real_time_windows.values():
            window.clear()
        logger.info("Cleared all collected metrics")

    def get_collection_status(self) -> dict[str, Any]:
        """Get current collection status.

        Returns:
            Current status information
        """
        return {
            "collection_active": self.collection_active,
            "collection_start_time": self.collection_start_time,
            "total_points": len(self.metric_points),
            "unique_metrics": len({p.metric_name for p in self.metric_points}),
            "memory_usage_points": len(self.metric_points),
            "buffer_usage": {
                name: len(buffer) for name, buffer in self.metric_buffers.items()
            },
        }
