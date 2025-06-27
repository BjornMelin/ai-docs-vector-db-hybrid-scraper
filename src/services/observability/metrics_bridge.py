"""OpenTelemetry metrics bridge with existing Prometheus monitoring system.

This module provides a bridge between OpenTelemetry metrics and the existing
Prometheus-based monitoring, enabling dual export and correlation between
trace and metric data while maintaining backward compatibility.
"""

import logging  # noqa: PLC0415
from typing import Any

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

from ..monitoring.metrics import MetricsRegistry


logger = logging.getLogger(__name__)


class OpenTelemetryMetricsBridge:
    """Bridge between OpenTelemetry metrics and existing Prometheus metrics."""

    def __init__(self, prometheus_registry: MetricsRegistry | None = None):
        """Initialize metrics bridge.

        Args:
            prometheus_registry: Existing Prometheus metrics registry
        """
        self.prometheus_registry = prometheus_registry
        self.meter = metrics.get_meter(__name__)
        self._instruments: dict[str, Any] = {}
        self._setup_otel_metrics()

    def _setup_otel_metrics(self) -> None:
        """Set up OpenTelemetry metric instruments."""

        # AI/ML Operation Metrics
        self._instruments["ai_operation_duration"] = self.meter.create_histogram(
            "ai_operation_duration_ms",
            description="Duration of AI/ML operations in milliseconds",
            unit="ms",
        )

        self._instruments["ai_operation_requests"] = self.meter.create_counter(
            "ai_operation_requests_total",
            description="Total number of AI/ML operation requests",
        )

        self._instruments["ai_tokens_used"] = self.meter.create_counter(
            "ai_tokens_used_total", description="Total tokens used in AI operations"
        )

        self._instruments["ai_cost"] = self.meter.create_counter(
            "ai_cost_total_usd", description="Total cost of AI operations in USD"
        )

        # Vector Search Metrics
        self._instruments["vector_search_duration"] = self.meter.create_histogram(
            "vector_search_duration_ms",
            description="Vector search operation duration",
            unit="ms",
        )

        self._instruments["vector_search_results"] = self.meter.create_histogram(
            "vector_search_results_count",
            description="Number of results returned by vector search",
        )

        self._instruments["vector_search_quality"] = self.meter.create_histogram(
            "vector_search_quality_score",
            description="Quality score of vector search results",
        )

        # Cache Performance Metrics
        self._instruments["cache_operations"] = self.meter.create_counter(
            "cache_operations_total", description="Total cache operations"
        )

        self._instruments["cache_hit_ratio"] = self.meter.create_gauge(
            "cache_hit_ratio", description="Cache hit ratio (0.0 to 1.0)"
        )

        self._instruments["cache_latency"] = self.meter.create_histogram(
            "cache_operation_latency_ms",
            description="Cache operation latency",
            unit="ms",
        )

        # Request Processing Metrics
        self._instruments["request_duration"] = self.meter.create_histogram(
            "request_duration_ms", description="HTTP request duration", unit="ms"
        )

        self._instruments["request_size"] = self.meter.create_histogram(
            "request_size_bytes", description="HTTP request size in bytes", unit="bytes"
        )

        self._instruments["response_size"] = self.meter.create_histogram(
            "response_size_bytes",
            description="HTTP response size in bytes",
            unit="bytes",
        )

        # Error Tracking Metrics
        self._instruments["error_count"] = self.meter.create_counter(
            "errors_total", description="Total number of errors"
        )

        self._instruments["error_rate"] = self.meter.create_gauge(
            "error_rate", description="Current error rate (errors per second)"
        )

        # Performance Metrics
        self._instruments["concurrent_operations"] = self.meter.create_up_down_counter(
            "concurrent_operations", description="Number of concurrent operations"
        )

        self._instruments["queue_depth"] = self.meter.create_gauge(
            "queue_depth", description="Current queue depth"
        )

        # Health Metrics
        self._instruments["service_health"] = self.meter.create_gauge(
            "service_health_status",
            description="Service health status (1=healthy, 0=unhealthy)",
        )

        self._instruments["dependency_health"] = self.meter.create_gauge(
            "dependency_health_status",
            description="Dependency health status (1=healthy, 0=unhealthy)",
        )

    def record_ai_operation(
        self,
        operation_type: str,
        provider: str,
        model: str,
        duration_ms: float,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        success: bool = True,
    ) -> None:
        """Record AI/ML operation metrics.

        Args:
            operation_type: Type of AI operation
            provider: AI provider
            model: Model name
            duration_ms: Operation duration in milliseconds
            tokens_used: Number of tokens used
            cost_usd: Cost in USD
            success: Whether operation succeeded
        """
        labels = {
            "operation_type": operation_type,
            "provider": provider,
            "model": model,
            "success": str(success),
        }

        # Record OpenTelemetry metrics
        self._instruments["ai_operation_duration"].record(duration_ms, labels)
        self._instruments["ai_operation_requests"].add(1, labels)

        if tokens_used is not None:
            self._instruments["ai_tokens_used"].add(tokens_used, labels)

        if cost_usd is not None:
            self._instruments["ai_cost"].add(cost_usd, labels)

        # Bridge to Prometheus if available
        if self.prometheus_registry:
            if hasattr(self.prometheus_registry, "record_embedding_cost") and cost_usd:
                self.prometheus_registry.record_embedding_cost(
                    provider, model, cost_usd
                )

    def record_vector_search(
        self,
        collection: str,
        query_type: str,
        duration_ms: float,
        results_count: int,
        top_score: float | None = None,
        success: bool = True,
    ) -> None:
        """Record vector search operation metrics.

        Args:
            collection: Vector collection name
            query_type: Type of query
            duration_ms: Search duration in milliseconds
            results_count: Number of results returned
            top_score: Top similarity score
            success: Whether search succeeded
        """
        labels = {
            "collection": collection,
            "query_type": query_type,
            "success": str(success),
        }

        # Record OpenTelemetry metrics
        self._instruments["vector_search_duration"].record(duration_ms, labels)
        self._instruments["vector_search_results"].record(results_count, labels)

        if top_score is not None:
            self._instruments["vector_search_quality"].record(top_score, labels)

        # Bridge to Prometheus if available
        if self.prometheus_registry:
            # Update Prometheus metrics if methods exist
            if hasattr(self.prometheus_registry, "_metrics"):
                try:
                    status = "success" if success else "error"
                    self.prometheus_registry._metrics["search_requests"].labels(
                        collection=collection, status=status
                    ).inc()
                    self.prometheus_registry._metrics["search_duration"].labels(
                        collection=collection, query_type=query_type
                    ).observe(duration_ms / 1000)  # Convert to seconds for Prometheus
                except Exception as e:
                    logger.warning(f"Failed to update Prometheus metrics: {e}")

    def record_cache_operation(
        self,
        cache_type: str,
        operation: str,
        duration_ms: float,
        hit: bool,
        cache_name: str = "default",
    ) -> None:
        """Record cache operation metrics.

        Args:
            cache_type: Type of cache
            operation: Cache operation
            duration_ms: Operation duration
            hit: Whether it was a cache hit
            cache_name: Cache instance name
        """
        labels = {
            "cache_type": cache_type,
            "operation": operation,
            "cache_name": cache_name,
            "result": "hit" if hit else "miss",
        }

        # Record OpenTelemetry metrics
        self._instruments["cache_operations"].add(1, labels)
        self._instruments["cache_latency"].record(duration_ms, labels)

        # Bridge to Prometheus if available
        if self.prometheus_registry:
            if hit:
                self.prometheus_registry.record_cache_hit(cache_type, cache_name)
            else:
                self.prometheus_registry.record_cache_miss(cache_type)

    def record_request_metrics(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        request_size_bytes: int | None = None,
        response_size_bytes: int | None = None,
    ) -> None:
        """Record HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: HTTP status code
            duration_ms: Request duration
            request_size_bytes: Request body size
            response_size_bytes: Response body size
        """
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx",
        }

        # Record OpenTelemetry metrics
        self._instruments["request_duration"].record(duration_ms, labels)

        if request_size_bytes is not None:
            self._instruments["request_size"].record(request_size_bytes, labels)

        if response_size_bytes is not None:
            self._instruments["response_size"].record(response_size_bytes, labels)

    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "error",
        user_impact: str = "medium",
    ) -> None:
        """Record error metrics.

        Args:
            error_type: Type of error
            component: Component where error occurred
            severity: Error severity
            user_impact: Impact on user experience
        """
        labels = {
            "error_type": error_type,
            "component": component,
            "severity": severity,
            "user_impact": user_impact,
        }

        # Record OpenTelemetry metrics
        self._instruments["error_count"].add(1, labels)

    def update_concurrent_operations(self, operation_type: str, delta: int) -> None:
        """Update concurrent operations counter.

        Args:
            operation_type: Type of operation
            delta: Change in concurrent operations (+1 for start, -1 for end)
        """
        labels = {"operation_type": operation_type}
        self._instruments["concurrent_operations"].add(delta, labels)

    def update_queue_depth(self, queue_type: str, depth: int) -> None:
        """Update queue depth gauge.

        Args:
            queue_type: Type of queue
            depth: Current queue depth
        """
        labels = {"queue_type": queue_type}
        self._instruments["queue_depth"].set(depth, labels)

    def update_service_health(self, service: str, healthy: bool) -> None:
        """Update service health status.

        Args:
            service: Service name
            healthy: Whether service is healthy
        """
        labels = {"service": service}
        self._instruments["service_health"].set(1 if healthy else 0, labels)

        # Bridge to Prometheus if available
        if self.prometheus_registry:
            self.prometheus_registry.update_service_health(service, healthy)

    def update_dependency_health(self, dependency: str, healthy: bool) -> None:
        """Update dependency health status.

        Args:
            dependency: Dependency name
            healthy: Whether dependency is healthy
        """
        labels = {"dependency": dependency}
        self._instruments["dependency_health"].set(1 if healthy else 0, labels)

        # Bridge to Prometheus if available
        if self.prometheus_registry:
            self.prometheus_registry.update_dependency_health(dependency, healthy)

    def record_batch_metrics(self, metrics_batch: dict[str, Any]) -> None:
        """Record multiple metrics in a batch operation.

        Args:
            metrics_batch: Dictionary of metrics to record
        """
        for metric_name, metric_data in metrics_batch.items():
            try:
                if metric_name in self._instruments:
                    instrument = self._instruments[metric_name]
                    value = metric_data.get("value")
                    labels = metric_data.get("labels", {})

                    if isinstance(instrument, Counter | UpDownCounter):
                        instrument.add(value, labels)
                    elif isinstance(instrument, Histogram):
                        instrument.record(value, labels)
                    elif hasattr(instrument, "set"):
                        instrument.set(value, labels)

            except Exception as e:
                logger.warning(f"Failed to record metric {metric_name}: {e}")

    def create_custom_counter(
        self, name: str, description: str, unit: str = ""
    ) -> Counter:
        """Create a custom counter instrument.

        Args:
            name: Metric name
            description: Metric description
            unit: Metric unit

        Returns:
            OpenTelemetry Counter instrument
        """
        counter = self.meter.create_counter(name, description, unit)
        self._instruments[name] = counter
        return counter

    def create_custom_gauge(self, name: str, description: str, unit: str = "") -> Any:
        """Create a custom gauge instrument.

        Args:
            name: Metric name
            description: Metric description
            unit: Metric unit

        Returns:
            OpenTelemetry Gauge instrument
        """
        gauge = self.meter.create_gauge(name, description, unit)
        self._instruments[name] = gauge
        return gauge

    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str = "",
        boundaries: list | None = None,
    ) -> Histogram:
        """Create a custom histogram instrument.

        Args:
            name: Metric name
            description: Metric description
            unit: Metric unit
            boundaries: Histogram boundaries

        Returns:
            OpenTelemetry Histogram instrument
        """
        if boundaries:
            histogram = self.meter.create_histogram(
                name, description, unit, boundaries=boundaries
            )
        else:
            histogram = self.meter.create_histogram(name, description, unit)

        self._instruments[name] = histogram
        return histogram

    def get_instrument(self, name: str) -> Any | None:
        """Get metric instrument by name.

        Args:
            name: Instrument name

        Returns:
            Metric instrument or None
        """
        return self._instruments.get(name)


# Global metrics bridge instance
_metrics_bridge: OpenTelemetryMetricsBridge | None = None


def initialize_metrics_bridge(
    prometheus_registry: MetricsRegistry | None = None,
) -> OpenTelemetryMetricsBridge:
    """Initialize global metrics bridge.

    Args:
        prometheus_registry: Existing Prometheus metrics registry

    Returns:
        Initialized metrics bridge
    """
    global _metrics_bridge
    _metrics_bridge = OpenTelemetryMetricsBridge(prometheus_registry)
    return _metrics_bridge


def get_metrics_bridge() -> OpenTelemetryMetricsBridge:
    """Get global metrics bridge instance.

    Returns:
        Global metrics bridge instance

    Raises:
        RuntimeError: If bridge not initialized
    """
    if _metrics_bridge is None:
        raise RuntimeError(
            "Metrics bridge not initialized. Call initialize_metrics_bridge() first."
        )
    return _metrics_bridge


# Convenience functions for common operations
def record_ai_operation(
    operation_type: str, provider: str, model: str, duration_ms: float, **kwargs
) -> None:
    """Record AI operation using global metrics bridge."""
    bridge = get_metrics_bridge()
    bridge.record_ai_operation(operation_type, provider, model, duration_ms, **kwargs)


def record_vector_search(
    collection: str, query_type: str, duration_ms: float, results_count: int, **kwargs
) -> None:
    """Record vector search using global metrics bridge."""
    bridge = get_metrics_bridge()
    bridge.record_vector_search(
        collection, query_type, duration_ms, results_count, **kwargs
    )


def record_cache_operation(
    cache_type: str, operation: str, duration_ms: float, hit: bool, **kwargs
) -> None:
    """Record cache operation using global metrics bridge."""
    bridge = get_metrics_bridge()
    bridge.record_cache_operation(cache_type, operation, duration_ms, hit, **kwargs)


def update_service_health(service: str, healthy: bool) -> None:
    """Update service health using global metrics bridge."""
    bridge = get_metrics_bridge()
    bridge.update_service_health(service, healthy)
