"""Test data factories for observability testing scenarios.

This module provides factories for generating consistent, realistic test data
for observability components including traces, metrics, logs, and health checks.
"""

import time
import uuid
from typing import Any

from tests.fixtures.observability import LogEntry, Metric, TraceSpan


class ObservabilityTestDataFactory:
    """Factory for generating test data for observability scenarios."""

    @staticmethod
    def create_sample_trace_span(
        trace_id: str | None = None,
        span_id: str | None = None,
        service_name: str = "test_service",
        operation_name: str = "test_operation",
        parent_span_id: str | None = None,
        duration_ms: float = 100.0,
        tags: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Create a sample trace span for testing."""
        trace_id = trace_id or str(uuid.uuid4())
        span_id = span_id or str(uuid.uuid4())
        start_time = time.time()

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=service_name,
            operation_name=operation_name,
            start_time=start_time,
            end_time=start_time + (duration_ms / 1000),
            tags=tags or {},
            logs=[],
        )

        return span

    @staticmethod
    def create_request_trace_spans(
        trace_id: str | None = None,
        include_errors: bool = False,
    ) -> list[TraceSpan]:
        """Create a complete request trace with API, embedding, and vector DB spans."""
        trace_id = trace_id or str(uuid.uuid4())

        spans = []

        # API Gateway span
        api_span = ObservabilityTestDataFactory.create_sample_trace_span(
            trace_id=trace_id,
            service_name="api_gateway",
            operation_name="handle_search_request",
            duration_ms=150.0,
            tags={
                "http.method": "POST",
                "http.url": "/api/v1/search",
                "http.status_code": 200,
                "user.id": "user_123",
            },
        )
        spans.append(api_span)

        # Embedding service span
        embedding_span = ObservabilityTestDataFactory.create_sample_trace_span(
            trace_id=trace_id,
            service_name="embedding_service",
            operation_name="generate_embeddings",
            parent_span_id=api_span.span_id,
            duration_ms=80.0,
            tags={
                "embedding.model": "text-embedding-3-small",
                "embedding.tokens": 150,
                "embedding.cost_usd": 0.001,
            },
        )
        spans.append(embedding_span)

        # Vector DB span
        vector_span = ObservabilityTestDataFactory.create_sample_trace_span(
            trace_id=trace_id,
            service_name="vector_db_service",
            operation_name="hybrid_search",
            parent_span_id=api_span.span_id,
            duration_ms=45.0,
            tags={
                "vector_db.collection": "documents",
                "vector_db.search_type": "hybrid",
                "vector_db.results_count": 15,
                "vector_db.search_time_ms": 45,
            },
        )
        spans.append(vector_span)

        # Content Intelligence span (optional)
        content_span = ObservabilityTestDataFactory.create_sample_trace_span(
            trace_id=trace_id,
            service_name="content_intelligence",
            operation_name="analyze_results",
            parent_span_id=api_span.span_id,
            duration_ms=25.0,
            tags={
                "content_intelligence.analysis_type": "quality_assessment",
                "content_intelligence.documents_analyzed": 15,
                "content_intelligence.avg_quality_score": 0.87,
            },
        )
        spans.append(content_span)

        if include_errors:
            # Add error to embedding span
            embedding_span.tags.update(
                {
                    "error": True,
                    "error.type": "ConnectionError",
                    "error.message": "OpenAI API connection timeout",
                }
            )

        return spans

    @staticmethod
    def create_sample_metric(
        name: str,
        value: float,
        metric_type: str = "counter",
        service: str = "test_service",
        additional_tags: dict[str, str] | None = None,
    ) -> Metric:
        """Create a sample metric for testing."""
        tags = {"service": service}
        if additional_tags:
            tags.update(additional_tags)

        return Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metric_type=metric_type,
        )

    @staticmethod
    def create_service_metrics_suite(service_name: str) -> list[Metric]:
        """Create a complete set of metrics for a service."""
        metrics = []

        if service_name == "api_gateway":
            metrics.extend(
                [
                    ObservabilityTestDataFactory.create_sample_metric(
                        "http.requests_total",
                        100,
                        "counter",
                        service_name,
                        {"method": "POST", "endpoint": "/search", "status_code": "200"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "http.request.duration",
                        0.125,
                        "histogram",
                        service_name,
                        {"method": "POST", "endpoint": "/search"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "http.active_connections", 15, "gauge", service_name
                    ),
                ]
            )
        elif service_name == "embedding_service":
            metrics.extend(
                [
                    ObservabilityTestDataFactory.create_sample_metric(
                        "embeddings.requests_total",
                        1,
                        "counter",
                        service_name,
                        {"provider": "openai", "model": "text-embedding-3-small"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "embeddings.generation.duration",
                        0.089,
                        "histogram",
                        service_name,
                        {"provider": "openai"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "embeddings.tokens_total",
                        150,
                        "counter",
                        service_name,
                        {"provider": "openai"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "embeddings.cost_usd",
                        0.001,
                        "gauge",
                        service_name,
                        {"provider": "openai"},
                    ),
                ]
            )
        elif service_name == "vector_db_service":
            metrics.extend(
                [
                    ObservabilityTestDataFactory.create_sample_metric(
                        "search.requests_total",
                        1,
                        "counter",
                        service_name,
                        {"collection": "documents", "search_type": "hybrid"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "search.duration",
                        0.045,
                        "histogram",
                        service_name,
                        {"collection": "documents"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "search.results.count",
                        15,
                        "gauge",
                        service_name,
                        {"collection": "documents"},
                    ),
                    ObservabilityTestDataFactory.create_sample_metric(
                        "vector_db.memory.usage", 0.67, "gauge", service_name
                    ),
                ]
            )

        return metrics

    @staticmethod
    def create_sample_log_entry(
        level: str = "INFO",
        service: str = "test_service",
        message: str = "Test log message",
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Create a sample log entry for testing."""
        return LogEntry(
            timestamp=time.time(),
            level=level,
            service=service,
            message=message,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            extra_context=extra_context or {},
        )

    @staticmethod
    def create_request_log_sequence(
        correlation_id: str | None = None,
        trace_id: str | None = None,
        include_errors: bool = False,
    ) -> list[LogEntry]:
        """Create a sequence of logs for a complete request flow."""
        correlation_id = correlation_id or str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())

        logs = []

        # API Gateway logs
        api_span_id = str(uuid.uuid4())
        logs.append(
            ObservabilityTestDataFactory.create_sample_log_entry(
                level="INFO",
                service="api_gateway",
                message="Received search request",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=api_span_id,
                extra_context={
                    "user_id": "user_123",
                    "endpoint": "/api/v1/search",
                    "request_size_bytes": 256,
                },
            )
        )

        logs.append(
            ObservabilityTestDataFactory.create_sample_log_entry(
                level="INFO",
                service="api_gateway",
                message="Request completed successfully",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=api_span_id,
                extra_context={
                    "total_duration_ms": 156,
                    "response_size_bytes": 2048,
                },
            )
        )

        # Embedding service logs
        embedding_span_id = str(uuid.uuid4())
        if include_errors:
            logs.append(
                ObservabilityTestDataFactory.create_sample_log_entry(
                    level="ERROR",
                    service="embedding_service",
                    message="Failed to connect to OpenAI API: Connection timeout",
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    span_id=embedding_span_id,
                    extra_context={
                        "api_endpoint": "https://api.openai.com/v1/embeddings",
                        "timeout_seconds": 30,
                        "retry_attempt": 3,
                    },
                )
            )
        else:
            logs.append(
                ObservabilityTestDataFactory.create_sample_log_entry(
                    level="INFO",
                    service="embedding_service",
                    message="Starting embedding generation",
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    span_id=embedding_span_id,
                    extra_context={
                        "text_count": 1,
                        "model": "text-embedding-3-small",
                    },
                )
            )

            logs.append(
                ObservabilityTestDataFactory.create_sample_log_entry(
                    level="INFO",
                    service="embedding_service",
                    message="Embedding generation completed",
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    span_id=embedding_span_id,
                    extra_context={
                        "duration_ms": 89,
                        "tokens_used": 150,
                        "cost_usd": 0.001,
                    },
                )
            )

        # Vector DB logs
        vector_span_id = str(uuid.uuid4())
        logs.append(
            ObservabilityTestDataFactory.create_sample_log_entry(
                level="INFO",
                service="vector_db_service",
                message="Performing hybrid search",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=vector_span_id,
                extra_context={
                    "collection": "documents",
                    "search_type": "hybrid",
                },
            )
        )

        logs.append(
            ObservabilityTestDataFactory.create_sample_log_entry(
                level="INFO",
                service="vector_db_service",
                message="Search completed",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=vector_span_id,
                extra_context={
                    "results_count": 15,
                    "search_time_ms": 45,
                },
            )
        )

        return logs

    @staticmethod
    def create_health_check_responses() -> dict[str, dict[str, Any]]:
        """Create sample health check responses for different services."""
        return {
            "api_gateway": {
                "status": "healthy",
                "details": {
                    "active_connections": 15,
                    "memory_usage": 0.45,
                    "cpu_usage": 0.23,
                    "response_time_p95": 0.125,
                },
            },
            "embedding_service": {
                "status": "degraded",
                "details": {
                    "openai_api_status": "slow",
                    "cache_hit_rate": 0.67,
                    "queue_size": 5,
                    "avg_response_time": 2.1,
                },
            },
            "vector_db_service": {
                "status": "healthy",
                "details": {
                    "collections_count": 3,
                    "memory_usage": 0.68,
                    "query_performance_ms": 45,
                    "index_status": "optimized",
                },
            },
            "cache_service": {
                "status": "error",
                "error": "Cache service unreachable",
            },
        }

