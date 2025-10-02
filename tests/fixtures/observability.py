"""Observability-specific fixtures for testing distributed tracing, metrics,
and monitoring.

This module provides reusable fixtures and infrastructure for testing
observability components across the distributed system.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest_asyncio


@dataclass
class TraceSpan:
    """Represents a distributed trace span for testing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str
    operation_name: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Metric:
    """Represents a collected metric for testing."""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str]
    metric_type: str  # counter, gauge, histogram


@dataclass
class LogEntry:
    """Represents a structured log entry for testing."""

    timestamp: float
    level: str
    service: str
    message: str
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)


class MockDistributedTracer:
    """Mock distributed tracer for testing tracing scenarios."""

    def __init__(self):
        self.spans: list[TraceSpan] = []
        self.current_span: TraceSpan | None = None
        self.trace_storage: dict[str, list[TraceSpan]] = {}

    def start_span(
        self,
        service_name: str,
        operation_name: str,
        parent_span_id: str | None = None,
        trace_id: str | None = None,
    ) -> TraceSpan:
        """Start a new span in the trace."""
        span_id = str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
            or (self.current_span.span_id if self.current_span else None),
            service_name=service_name,
            operation_name=operation_name,
            start_time=time.time(),
        )

        self.spans.append(span)
        self.current_span = span
        return span

    def finish_span(self, span: TraceSpan, tags: dict[str, Any] | None = None) -> None:
        """Finish a span and store it."""
        span.end_time = time.time()
        if tags:
            span.tags.update(tags)

        # Store in trace storage
        if span.trace_id not in self.trace_storage:
            self.trace_storage[span.trace_id] = []
        self.trace_storage[span.trace_id].append(span)

    def get_spans_for_trace(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a specific trace."""
        return self.trace_storage.get(trace_id, [])

    def clear(self) -> None:
        """Clear all stored spans and traces."""
        self.spans.clear()
        self.current_span = None
        self.trace_storage.clear()


class MockMetricsCollector:
    """Mock metrics collector for testing metrics scenarios."""

    def __init__(self):
        self.metrics: list[Metric] = []

    def record_counter(
        self, name: str, value: float = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type="counter",
        )
        self.metrics.append(metric)

    def record_gauge(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type="gauge",
        )
        self.metrics.append(metric)

    def record_histogram(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type="histogram",
        )
        self.metrics.append(metric)

    def get_metrics_by_name(self, name: str) -> list[Metric]:
        """Get all metrics with a specific name."""
        return [m for m in self.metrics if m.name == name]

    def get_metrics_by_type(self, metric_type: str) -> list[Metric]:
        """Get all metrics of a specific type."""
        return [m for m in self.metrics if m.metric_type == metric_type]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()


class MockStructuredLogger:
    """Mock structured logger for testing logging scenarios."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logs: list[LogEntry] = []

    def log(
        self,
        level: str,
        message: str,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        **extra_context: Any,
    ) -> None:
        """Log a structured message."""
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level.upper(),
            service=self.service_name,
            message=message,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            extra_context=extra_context,
        )
        self.logs.append(log_entry)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.log("DEBUG", message, **kwargs)

    def get_logs_by_level(self, level: str) -> list[LogEntry]:
        """Get all logs with a specific level."""
        return [log for log in self.logs if log.level == level.upper()]

    def get_logs_by_correlation_id(self, correlation_id: str) -> list[LogEntry]:
        """Get all logs with a specific correlation ID."""
        return [log for log in self.logs if log.correlation_id == correlation_id]

    def clear(self) -> None:
        """Clear all stored logs."""
        self.logs.clear()


@pytest_asyncio.fixture
async def mock_distributed_tracer():
    """Mock distributed tracer for testing tracing scenarios."""
    tracer = MockDistributedTracer()
    yield tracer
    tracer.clear()


@pytest_asyncio.fixture
async def mock_metrics_collector():
    """Mock metrics collector for testing metrics scenarios."""
    collector = MockMetricsCollector()
    yield collector
    collector.clear()


@pytest_asyncio.fixture
async def mock_structured_logger():
    """Mock structured logger for testing logging scenarios."""
    logger = MockStructuredLogger("test_service")
    yield logger
    logger.clear()


@pytest_asyncio.fixture
async def mock_health_check_functions():
    """Mock health check functions for different services."""

    async def healthy_api_gateway():
        return {
            "status": "healthy",
            "details": {
                "active_connections": 15,
                "memory_usage": 0.45,
                "cpu_usage": 0.23,
                "response_time_p95": 0.125,
            },
        }

    async def degraded_embedding_service():
        return {
            "status": "degraded",
            "details": {
                "openai_api_status": "slow",
                "cache_hit_rate": 0.67,
                "queue_size": 5,
                "avg_response_time": 2.1,
            },
        }

    async def healthy_vector_db():
        return {
            "status": "healthy",
            "details": {
                "collections_count": 3,
                "memory_usage": 0.68,
                "query_performance_ms": 45,
                "index_status": "optimized",
            },
        }

    async def failing_cache_service():
        raise ConnectionError("Cache service unreachable")

    return {
        "api_gateway": healthy_api_gateway,
        "embedding_service": degraded_embedding_service,
        "vector_db_service": healthy_vector_db,
        "cache_service": failing_cache_service,
    }


@pytest_asyncio.fixture
async def sample_trace_context(mock_distributed_tracer):
    """Sample trace context with realistic service interactions."""
    tracer = mock_distributed_tracer

    # Create a realistic request trace
    trace_id = str(uuid.uuid4())

    # API Gateway span
    api_span = tracer.start_span(
        "api_gateway", "handle_search_request", trace_id=trace_id
    )

    # Embedding service span
    embedding_span = tracer.start_span(
        "embedding_service",
        "generate_embeddings",
        parent_span_id=api_span.span_id,
        trace_id=trace_id,
    )
    tracer.finish_span(
        embedding_span,
        {
            "embedding.model": "text-embedding-3-small",
            "embedding.tokens": 150,
            "embedding.cost_usd": 0.001,
        },
    )

    # Vector DB span
    vector_span = tracer.start_span(
        "vector_db_service",
        "hybrid_search",
        parent_span_id=api_span.span_id,
        trace_id=trace_id,
    )
    tracer.finish_span(
        vector_span,
        {
            "vector_db.collection": "documents",
            "vector_db.search_type": "hybrid",
            "vector_db.results_count": 15,
            "vector_db.search_time_ms": 45,
        },
    )

    # Finish API span
    tracer.finish_span(
        api_span,
        {
            "http.method": "POST",
            "http.url": "/api/v1/search",
            "http.status_code": 200,
            "user.id": "user_123",
        },
    )

    return {
        "trace_id": trace_id,
        "spans": tracer.get_spans_for_trace(trace_id),
        "tracer": tracer,
    }


@pytest_asyncio.fixture
async def sample_metrics_data(mock_metrics_collector):
    """Sample metrics data for testing aggregation and alerting."""
    collector = mock_metrics_collector

    # API Gateway metrics
    collector.record_counter(
        "http.requests_total",
        100,
        {
            "service": "api_gateway",
            "method": "POST",
            "endpoint": "/search",
            "status_code": "200",
        },
    )
    collector.record_histogram(
        "http.request.duration", 0.125, {"service": "api_gateway", "method": "POST"}
    )
    collector.record_gauge("http.active_connections", 15, {"service": "api_gateway"})

    # Embedding service metrics
    collector.record_counter(
        "embeddings.requests_total",
        1,
        {
            "service": "embedding_service",
            "provider": "openai",
            "model": "text-embedding-3-small",
        },
    )
    collector.record_histogram(
        "embeddings.generation.duration", 0.089, {"service": "embedding_service"}
    )
    collector.record_counter(
        "embeddings.tokens_total", 150, {"service": "embedding_service"}
    )
    collector.record_gauge(
        "embeddings.cost.usd", 0.001, {"service": "embedding_service"}
    )

    # Vector DB metrics
    collector.record_counter(
        "search.requests_total",
        1,
        {
            "service": "vector_db_service",
            "collection": "documents",
            "search_type": "hybrid",
        },
    )
    collector.record_histogram(
        "search.duration", 0.045, {"service": "vector_db_service"}
    )
    collector.record_gauge("search.results.count", 15, {"service": "vector_db_service"})
    collector.record_gauge(
        "vector_db.memory.usage", 0.67, {"service": "vector_db_service"}
    )

    return collector
