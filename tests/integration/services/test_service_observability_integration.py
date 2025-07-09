"""Service observability integration tests.

This module tests the integration of observability components across services,
validating metrics collection, distributed tracing, logging correlation, and
monitoring capabilities in the distributed system.

Tests include:
- Distributed tracing across service boundaries
- Metrics collection and aggregation
- Log correlation and structured logging
- Health monitoring and alerting
- Performance monitoring integration
- Error tracking and debugging
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest


logger = logging.getLogger(__name__)


# from src.services.observability.tracking import TrackingManager  # Not implemented yet


@dataclass
class TraceSpan:
    """Represents a distributed trace span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str
    operation_name: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, Any] = None
    logs: list[dict] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []


@dataclass
class Metric:
    """Represents a collected metric."""

    name: str
    value: float
    timestamp: float
    tags: dict[str, str]
    metric_type: str  # counter, gauge, histogram


class TestDistributedTracing:
    """Test distributed tracing across service boundaries."""

    @pytest.fixture
    async def tracing_infrastructure(self):
        """Setup distributed tracing infrastructure."""
        services = {
            "tracer": AsyncMock(),
            "span_collector": AsyncMock(),
            "trace_analyzer": AsyncMock(),
        }

        # Mock trace storage
        trace_storage = {}

        return {"trace_storage": trace_storage, **services}

    @pytest.mark.asyncio
    async def test_end_to_end_request_tracing(self, tracing_infrastructure):
        """Test end-to-end request tracing across multiple services."""
        setup = tracing_infrastructure
        trace_storage = setup["trace_storage"]

        # Simulate a complete request flow: API -> Embedding -> Vector DB
        trace_id = str(uuid.uuid4())

        # Mock distributed trace context
        class DistributedTraceContext:
            def __init__(self, trace_id: str):
                self.trace_id = trace_id
                self.spans = []
                self.current_span = None

            def start_span(
                self,
                service_name: str,
                operation_name: str,
                parent_span_id: str | None = None,
            ) -> TraceSpan:
                """Start a  span in the trace."""
                span_id = str(uuid.uuid4())
                span = TraceSpan(
                    trace_id=self.trace_id,
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

            def finish_span(self, span: TraceSpan, tags: dict[str, Any] | None = None):
                """Finish a span and add tags."""
                span.end_time = time.time()
                if tags:
                    span.tags.update(tags)

                # Store span
                if self.trace_id not in trace_storage:
                    trace_storage[self.trace_id] = []
                trace_storage[self.trace_id].append(span)

        trace_context = DistributedTraceContext(trace_id)

        # Simulate API Gateway request
        api_span = trace_context.start_span("api_gateway", "handle_search_request")
        await asyncio.sleep(0.01)  # Simulate processing time

        # Simulate Embedding Service call
        embedding_span = trace_context.start_span(
            "embedding_service", "generate_embeddings", api_span.span_id
        )
        await asyncio.sleep(0.02)  # Simulate embedding generation
        trace_context.finish_span(
            embedding_span,
            {
                "embedding.model": "text-embedding-3-small",
                "embedding.tokens": 150,
                "embedding.cost_usd": 0.001,
            },
        )

        # Simulate Vector DB search
        vector_db_span = trace_context.start_span(
            "vector_db_service", "hybrid_search", api_span.span_id
        )
        await asyncio.sleep(0.015)  # Simulate search
        trace_context.finish_span(
            vector_db_span,
            {
                "vector_db.collection": "documents",
                "vector_db.search_type": "hybrid",
                "vector_db.results_count": 15,
                "vector_db.search_time_ms": 45,
            },
        )

        # Simulate Content Intelligence analysis
        content_span = trace_context.start_span(
            "content_intelligence", "analyze_results", api_span.span_id
        )
        await asyncio.sleep(0.008)  # Simulate analysis
        trace_context.finish_span(
            content_span,
            {
                "content_intelligence.analysis_type": "quality_assessment",
                "content_intelligence.documents_analyzed": 15,
                "content_intelligence.avg_quality_score": 0.87,
            },
        )

        # Finish root span
        trace_context.finish_span(
            api_span,
            {
                "http.method": "POST",
                "http.url": "/api/v1/search",
                "http.status_code": 200,
                "user.id": "user_123",
            },
        )

        # Verify trace structure
        spans = trace_storage[trace_id]
        assert len(spans) == 4

        # Verify span hierarchy
        root_span = next(s for s in spans if s.service_name == "api_gateway")
        child_spans = [s for s in spans if s.parent_span_id == root_span.span_id]

        assert len(child_spans) == 3  # embedding, vector_db, content_intelligence
        assert all(s.parent_span_id == root_span.span_id for s in child_spans)

        # Verify timing
        _total_request_time = root_span.end_time - root_span.start_time
        assert _total_request_time > 0.04  # Should be sum of child operations

        # Verify tags
        embedding_span_found = next(
            s for s in spans if s.service_name == "embedding_service"
        )
        assert embedding_span_found.tags["embedding.model"] == "text-embedding-3-small"

        vector_span_found = next(
            s for s in spans if s.service_name == "vector_db_service"
        )
        assert vector_span_found.tags["vector_db.results_count"] == 15

    @pytest.mark.asyncio
    async def test_trace_sampling_and_propagation(self, _tracing_infrastructure):
        """Test trace sampling and context propagation."""

        class TraceSampler:
            def __init__(self, sampling_rate: float = 0.1):
                self.sampling_rate = sampling_rate
                self.sampled_traces = set()

            def should_sample(self, trace_id: str) -> bool:
                """Decide whether to sample a trace."""
                # Simple hash-based sampling
                hash_value = hash(trace_id) % 100
                should_sample = hash_value < (self.sampling_rate * 100)

                if should_sample:
                    self.sampled_traces.add(trace_id)

                return should_sample

            def is_sampled(self, trace_id: str) -> bool:
                """Check if trace is being sampled."""
                return trace_id in self.sampled_traces

        sampler = TraceSampler(sampling_rate=0.3)  # 30% sampling rate

        # Test multiple traces
        test_traces = [str(uuid.uuid4()) for _ in range(100)]
        sampling_decisions = []

        for trace_id in test_traces:
            decision = sampler.should_sample(trace_id)
            sampling_decisions.append(decision)

        # Verify sampling rate is approximately correct
        sampled_count = sum(sampling_decisions)
        actual_rate = sampled_count / len(test_traces)

        # Allow for some variance in sampling rate
        assert 0.2 <= actual_rate <= 0.4  # 30% ± 10%

        # Test context propagation headers
        class TraceContextPropagator:
            def inject_headers(self, trace_id: str, span_id: str) -> dict[str, str]:
                """Inject trace context into HTTP headers."""
                return {
                    "X-Trace-Id": trace_id,
                    "X-Span-Id": span_id,
                    "X-Sampled": "1" if sampler.is_sampled(trace_id) else "0",
                }

            def extract_context(self, headers: dict[str, str]) -> dict[str, str] | None:
                """Extract trace context from HTTP headers."""
                if "X-Trace-Id" not in headers:
                    return None

                return {
                    "trace_id": headers["X-Trace-Id"],
                    "span_id": headers.get("X-Span-Id"),
                    "sampled": headers.get("X-Sampled") == "1",
                }

        propagator = TraceContextPropagator()

        # Test header injection and extraction
        test_trace_id = test_traces[0]
        test_span_id = str(uuid.uuid4())

        headers = propagator.inject_headers(test_trace_id, test_span_id)
        extracted_context = propagator.extract_context(headers)

        assert extracted_context["trace_id"] == test_trace_id
        assert extracted_context["span_id"] == test_span_id
        assert extracted_context["sampled"] == sampler.is_sampled(test_trace_id)

    @pytest.mark.asyncio
    async def test_error_tracking_in_traces(self, tracing_infrastructure):
        """Test error tracking and debugging in distributed traces."""
        setup = tracing_infrastructure
        setup["trace_storage"]

        trace_id = str(uuid.uuid4())

        class ErrorTrackingTracer:
            def __init__(self):
                self.error_spans = []

            def record_error(
                self,
                span: TraceSpan,
                error: Exception,
                error_context: dict | None = None,
            ):
                """Record an error in a span."""
                span.tags.update(
                    {
                        "error": True,
                        "error.type": type(error).__name__,
                        "error.message": str(error),
                        "error.stack_trace": getattr(error, "__traceback__", None),
                    }
                )

                # Add error log
                error_log = {
                    "timestamp": time.time(),
                    "level": "ERROR",
                    "message": f"Error in {span.operation_name}: {error!s}",
                    "error_context": error_context or {},
                }
                span.logs.append(error_log)

                self.error_spans.append(span)

            def analyze_error_patterns(self) -> dict:
                """Analyze error patterns across spans."""
                error_types = {}
                error_services = {}

                for span in self.error_spans:
                    error_type = span.tags.get("error.type")
                    service = span.service_name

                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    error_services[service] = error_services.get(service, 0) + 1

                return {
                    "_total_errors": len(self.error_spans),
                    "error_types": error_types,
                    "affected_services": error_services,
                }

        error_tracer = ErrorTrackingTracer()

        # Simulate request with errors
        api_span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            service_name="api_gateway",
            operation_name="handle_request",
            start_time=time.time(),
        )

        # Simulate embedding service error
        embedding_span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=api_span.span_id,
            service_name="embedding_service",
            operation_name="generate_embeddings",
            start_time=time.time(),
        )

        # Record embedding service error
        embedding_error = ConnectionError("OpenAI API connection timeout")
        error_tracer.record_error(
            embedding_span,
            embedding_error,
            {
                "api_endpoint": "https://api.openai.com/v1/embeddings",
                "retry_count": 3,
                "timeout_seconds": 30,
            },
        )

        # Simulate vector db service error
        vector_span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=api_span.span_id,
            service_name="vector_db_service",
            operation_name="search",
            start_time=time.time(),
        )

        vector_error = ValueError("Invalid vector dimension: expected 1536, got 512")
        error_tracer.record_error(
            vector_span,
            vector_error,
            {"collection": "documents", "query_vector_dim": 512, "expected_dim": 1536},
        )

        # Analyze error patterns
        error_analysis = error_tracer.analyze_error_patterns()

        # Verify error tracking
        assert error_analysis["_total_errors"] == 2
        assert "ConnectionError" in error_analysis["error_types"]
        assert "ValueError" in error_analysis["error_types"]
        assert "embedding_service" in error_analysis["affected_services"]
        assert "vector_db_service" in error_analysis["affected_services"]

        # Verify error details in spans
        assert embedding_span.tags["error"] is True
        assert embedding_span.tags["error.type"] == "ConnectionError"
        assert "OpenAI API connection timeout" in embedding_span.tags["error.message"]

        assert vector_span.tags["error"] is True
        assert vector_span.tags["error.type"] == "ValueError"
        assert "Invalid vector dimension" in vector_span.tags["error.message"]

        # Verify error logs
        assert len(embedding_span.logs) == 1
        assert embedding_span.logs[0]["level"] == "ERROR"
        assert "retry_count" in embedding_span.logs[0]["error_context"]


class TestMetricsCollection:
    """Test metrics collection and aggregation across services."""

    @pytest.fixture
    async def metrics_infrastructure(self):
        """Setup metrics collection infrastructure."""
        services = {
            "metrics_collector": AsyncMock(),
            "metrics_aggregator": AsyncMock(),
            "alerting_manager": AsyncMock(),
        }

        # Mock metrics storage
        metrics_storage = []

        return {"metrics_storage": metrics_storage, **services}

    @pytest.mark.asyncio
    async def test_service_metrics_collection(self, metrics_infrastructure):
        """Test metrics collection from multiple services."""
        setup = metrics_infrastructure
        metrics_storage = setup["metrics_storage"]

        class ServiceMetricsCollector:
            def __init__(self):
                self.metrics_buffer = []

            def record_counter(
                self, name: str, value: float = 1, tags: dict[str, str] | None = None
            ):
                """Record a counter metric."""
                metric = Metric(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    tags=tags or {},
                    metric_type="counter",
                )
                self.metrics_buffer.append(metric)

            def record_gauge(
                self, name: str, value: float, tags: dict[str, str] | None = None
            ):
                """Record a gauge metric."""
                metric = Metric(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    tags=tags or {},
                    metric_type="gauge",
                )
                self.metrics_buffer.append(metric)

            def record_histogram(
                self, name: str, value: float, tags: dict[str, str] | None = None
            ):
                """Record a histogram metric."""
                metric = Metric(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    tags=tags or {},
                    metric_type="histogram",
                )
                self.metrics_buffer.append(metric)

            def flush_metrics(self):
                """Flush metrics to storage."""
                metrics_storage.extend(self.metrics_buffer)
                self.metrics_buffer.clear()

        # Create collectors for different services
        api_collector = ServiceMetricsCollector()
        embedding_collector = ServiceMetricsCollector()
        vector_db_collector = ServiceMetricsCollector()

        # Simulate API Gateway metrics
        api_collector.record_counter(
            "http.requests._total",
            1,
            {
                "service": "api_gateway",
                "method": "POST",
                "endpoint": "/search",
                "status_code": "200",
            },
        )
        api_collector.record_histogram(
            "http.request.duration",
            0.125,
            {"service": "api_gateway", "method": "POST", "endpoint": "/search"},
        )
        api_collector.record_gauge(
            "http.active_connections", 15, {"service": "api_gateway"}
        )

        # Simulate Embedding Service metrics
        embedding_collector.record_counter(
            "embeddings.requests._total",
            1,
            {
                "service": "embedding_service",
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
        )
        embedding_collector.record_histogram(
            "embeddings.generation.duration",
            0.089,
            {"service": "embedding_service", "provider": "openai"},
        )
        embedding_collector.record_counter(
            "embeddings.tokens._total",
            150,
            {"service": "embedding_service", "provider": "openai"},
        )
        embedding_collector.record_gauge(
            "embeddings.cost.usd",
            0.001,
            {"service": "embedding_service", "provider": "openai"},
        )

        # Simulate Vector DB metrics
        vector_db_collector.record_counter(
            "search.requests._total",
            1,
            {
                "service": "vector_db_service",
                "collection": "documents",
                "search_type": "hybrid",
            },
        )
        vector_db_collector.record_histogram(
            "search.duration",
            0.045,
            {"service": "vector_db_service", "collection": "documents"},
        )
        vector_db_collector.record_gauge(
            "search.results.count",
            15,
            {"service": "vector_db_service", "collection": "documents"},
        )
        vector_db_collector.record_gauge(
            "vector_db.memory.usage", 0.67, {"service": "vector_db_service"}
        )

        # Flush all metrics
        api_collector.flush_metrics()
        embedding_collector.flush_metrics()
        vector_db_collector.flush_metrics()

        # Verify metrics collection
        assert len(metrics_storage) == 10  # Total metrics collected

        # Verify specific metrics
        http_requests = [m for m in metrics_storage if m.name == "http.requests._total"]
        assert len(http_requests) == 1
        assert http_requests[0].tags["service"] == "api_gateway"
        assert http_requests[0].tags["status_code"] == "200"

        embedding_requests = [
            m for m in metrics_storage if m.name == "embeddings.requests._total"
        ]
        assert len(embedding_requests) == 1
        assert embedding_requests[0].tags["provider"] == "openai"

        search_requests = [
            m for m in metrics_storage if m.name == "search.requests._total"
        ]
        assert len(search_requests) == 1
        assert search_requests[0].tags["search_type"] == "hybrid"

        # Verify metric types
        counters = [m for m in metrics_storage if m.metric_type == "counter"]
        gauges = [m for m in metrics_storage if m.metric_type == "gauge"]
        histograms = [m for m in metrics_storage if m.metric_type == "histogram"]

        assert len(counters) == 4  # requests and tokens counters
        assert len(gauges) == 4  # active connections, cost, results count, memory usage
        assert len(histograms) == 2  # duration histograms

    @pytest.mark.asyncio
    async def test_metrics_aggregation_and_alerting(self, metrics_infrastructure):
        """Test metrics aggregation and alerting rules."""
        setup = metrics_infrastructure
        metrics_storage = setup["metrics_storage"]

        # Populate with sample metrics
        sample_metrics = [
            Metric(
                "http.requests._total",
                100,
                time.time() - 60,
                {"service": "api_gateway"},
                "counter",
            ),
            Metric(
                "http.requests._total",
                150,
                time.time() - 30,
                {"service": "api_gateway"},
                "counter",
            ),
            Metric(
                "http.requests._total",
                200,
                time.time(),
                {"service": "api_gateway"},
                "counter",
            ),
            Metric(
                "http.request.duration",
                0.1,
                time.time() - 60,
                {"service": "api_gateway"},
                "histogram",
            ),
            Metric(
                "http.request.duration",
                0.5,
                time.time() - 30,
                {"service": "api_gateway"},
                "histogram",
            ),
            Metric(
                "http.request.duration",
                1.2,
                time.time(),
                {"service": "api_gateway"},
                "histogram",
            ),
            Metric(
                "embeddings.errors._total",
                0,
                time.time() - 60,
                {"service": "embedding_service"},
                "counter",
            ),
            Metric(
                "embeddings.errors._total",
                2,
                time.time() - 30,
                {"service": "embedding_service"},
                "counter",
            ),
            Metric(
                "embeddings.errors._total",
                5,
                time.time(),
                {"service": "embedding_service"},
                "counter",
            ),
        ]
        metrics_storage.extend(sample_metrics)

        class MetricsAggregator:
            def __init__(self, metrics: list[Metric]):
                self.metrics = metrics

            def calculate_rate(
                self, metric_name: str, time_window: float = 60
            ) -> float:
                """Calculate rate of change for counter metrics."""
                current_time = time.time()
                recent_metrics = [
                    m
                    for m in self.metrics
                    if m.name == metric_name
                    and current_time - m.timestamp <= time_window
                ]

                if len(recent_metrics) < 2:
                    return 0.0

                # Sort by timestamp
                recent_metrics.sort(key=lambda x: x.timestamp)

                # Calculate rate (simple difference)
                latest = recent_metrics[-1]
                earliest = recent_metrics[0]

                time_diff = latest.timestamp - earliest.timestamp
                value_diff = latest.value - earliest.value

                return value_diff / time_diff if time_diff > 0 else 0.0

            def calculate_percentile(
                self, metric_name: str, percentile: float, time_window: float = 60
            ) -> float:
                """Calculate percentile for histogram metrics."""
                current_time = time.time()
                recent_metrics = [
                    m
                    for m in self.metrics
                    if m.name == metric_name
                    and current_time - m.timestamp <= time_window
                ]

                if not recent_metrics:
                    return 0.0

                values = [m.value for m in recent_metrics]
                values.sort()

                index = int(percentile * len(values))
                return values[min(index, len(values) - 1)]

        class AlertingManager:
            def __init__(self):
                self.alerts = []
                self.alert_rules = [
                    {
                        "name": "high_request_rate",
                        "condition": lambda aggregator: aggregator.calculate_rate(
                            "http.requests._total"
                        )
                        > 3.0,
                        "message": "High request rate detected",
                    },
                    {
                        "name": "high_latency",
                        "condition": lambda aggregator: aggregator.calculate_percentile(
                            "http.request.duration", 0.95
                        )
                        > 1.0,
                        "message": "High latency detected (P95 > 1s)",
                    },
                    {
                        "name": "embedding_errors",
                        "condition": lambda aggregator: aggregator.calculate_rate(
                            "embeddings.errors._total"
                        )
                        > 0.1,
                        "message": "Embedding service error rate increased",
                    },
                ]

            def evaluate_alerts(self, aggregator: MetricsAggregator):
                """Evaluate alert rules against current metrics."""
                triggered_alerts = []

                for rule in self.alert_rules:
                    try:
                        if rule["condition"](aggregator):
                            alert = {
                                "name": rule["name"],
                                "message": rule["message"],
                                "timestamp": time.time(),
                                "severity": "warning",
                            }
                            triggered_alerts.append(alert)
                            self.alerts.append(alert)
                    except (TimeoutError, ConnectionError, RuntimeError, ValueError):
                        # Handle errors in alert evaluation
                        logger.debug("Exception suppressed during cleanup/testing")

                return triggered_alerts

        aggregator = MetricsAggregator(metrics_storage)
        alerting_manager = AlertingManager()

        # Test aggregation
        request_rate = aggregator.calculate_rate("http.requests._total")
        latency_p95 = aggregator.calculate_percentile("http.request.duration", 0.95)
        error_rate = aggregator.calculate_rate("embeddings.errors._total")

        # Verify aggregation calculations
        assert request_rate > 0  # Should detect increasing request rate
        assert latency_p95 >= 1.0  # Should detect high latency
        assert error_rate > 0  # Should detect increasing error rate

        # Test alerting
        triggered_alerts = alerting_manager.evaluate_alerts(aggregator)

        # Verify alerts
        alert_names = [alert["name"] for alert in triggered_alerts]
        assert "high_latency" in alert_names  # Should trigger due to 1.2s latency
        assert (
            "embedding_errors" in alert_names
        )  # Should trigger due to error rate increase

        # Verify alert structure
        for alert in triggered_alerts:
            assert "name" in alert
            assert "message" in alert
            assert "timestamp" in alert
            assert "severity" in alert

    @pytest.mark.asyncio
    async def test_custom_business_metrics(self, metrics_infrastructure):
        """Test custom business metrics collection and analysis."""
        setup = metrics_infrastructure
        metrics_storage = setup["metrics_storage"]

        class BusinessMetricsCollector:
            def __init__(self):
                self.business_metrics = []

            def record_search_quality(
                self, query: str, results_count: int, user_satisfaction: float
            ):
                """Record search quality metrics."""
                self.business_metrics.extend(
                    [
                        Metric(
                            "search.quality.results_count",
                            results_count,
                            time.time(),
                            {"query_type": self._classify_query(query)},
                            "gauge",
                        ),
                        Metric(
                            "search.quality.user_satisfaction",
                            user_satisfaction,
                            time.time(),
                            {"query_type": self._classify_query(query)},
                            "gauge",
                        ),
                    ]
                )

            def record_content_processing(
                self, processing_time: float, quality_score: float, content_type: str
            ):
                """Record content processing metrics."""
                self.business_metrics.extend(
                    [
                        Metric(
                            "content.processing.duration",
                            processing_time,
                            time.time(),
                            {"content_type": content_type},
                            "histogram",
                        ),
                        Metric(
                            "content.quality.score",
                            quality_score,
                            time.time(),
                            {"content_type": content_type},
                            "gauge",
                        ),
                    ]
                )

            def record_user_engagement(
                self,
                session_duration: float,
                queries_per_session: int,
                bounce_rate: float,
            ):
                """Record user engagement metrics."""
                self.business_metrics.extend(
                    [
                        Metric(
                            "user.session.duration",
                            session_duration,
                            time.time(),
                            {},
                            "histogram",
                        ),
                        Metric(
                            "user.queries.per_session",
                            queries_per_session,
                            time.time(),
                            {},
                            "gauge",
                        ),
                        Metric(
                            "user.bounce_rate", bounce_rate, time.time(), {}, "gauge"
                        ),
                    ]
                )

            def _classify_query(self, query: str) -> str:
                """Simple query classification."""
                if "?" in query:
                    return "question"
                if len(query.split()) > 5:
                    return "complex"
                return "simple"

            def flush_metrics(self):
                """Flush business metrics to storage."""
                metrics_storage.extend(self.business_metrics)
                self.business_metrics.clear()

        business_collector = BusinessMetricsCollector()

        # Record various business metrics
        business_collector.record_search_quality(
            "machine learning algorithms", 15, 0.85
        )
        business_collector.record_search_quality("What is deep learning?", 8, 0.92)
        business_collector.record_search_quality("AI", 25, 0.73)

        business_collector.record_content_processing(1.2, 0.88, "article")
        business_collector.record_content_processing(0.8, 0.91, "tutorial")
        business_collector.record_content_processing(2.1, 0.76, "research_paper")

        business_collector.record_user_engagement(180.5, 5, 0.23)
        business_collector.record_user_engagement(420.8, 12, 0.15)
        business_collector.record_user_engagement(95.2, 2, 0.67)

        business_collector.flush_metrics()

        # Analyze business metrics
        class BusinessMetricsAnalyzer:
            def __init__(self, metrics: list[Metric]):
                self.metrics = metrics

            def analyze_search_quality_by_type(self) -> dict:
                """Analyze search quality metrics by query type."""
                quality_by_type = {}

                satisfaction_metrics = [
                    m
                    for m in self.metrics
                    if m.name == "search.quality.user_satisfaction"
                ]

                for metric in satisfaction_metrics:
                    query_type = metric.tags.get("query_type", "unknown")
                    if query_type not in quality_by_type:
                        quality_by_type[query_type] = []
                    quality_by_type[query_type].append(metric.value)

                # Calculate averages
                return {
                    query_type: {
                        "avg_satisfaction": sum(values) / len(values),
                        "sample_count": len(values),
                    }
                    for query_type, values in quality_by_type.items()
                }

            def analyze_content_quality_trends(self) -> dict:
                """Analyze content quality trends by type."""
                quality_by_content_type = {}

                quality_metrics = [
                    m for m in self.metrics if m.name == "content.quality.score"
                ]

                for metric in quality_metrics:
                    content_type = metric.tags.get("content_type", "unknown")
                    if content_type not in quality_by_content_type:
                        quality_by_content_type[content_type] = []
                    quality_by_content_type[content_type].append(metric.value)

                return {
                    content_type: {
                        "avg_quality": sum(scores) / len(scores),
                        "min_quality": min(scores),
                        "max_quality": max(scores),
                    }
                    for content_type, scores in quality_by_content_type.items()
                }

        analyzer = BusinessMetricsAnalyzer(metrics_storage)

        # Test business metrics analysis
        search_quality_analysis = analyzer.analyze_search_quality_by_type()
        content_quality_analysis = analyzer.analyze_content_quality_trends()

        # Verify search quality analysis
        assert "question" in search_quality_analysis
        assert "simple" in search_quality_analysis
        assert search_quality_analysis["question"]["avg_satisfaction"] == 0.92
        assert search_quality_analysis["simple"]["sample_count"] == 2

        # Verify content quality analysis
        assert "article" in content_quality_analysis
        assert "tutorial" in content_quality_analysis
        assert "research_paper" in content_quality_analysis
        assert content_quality_analysis["tutorial"]["avg_quality"] == 0.91
        assert content_quality_analysis["research_paper"]["min_quality"] == 0.76


class TestLogCorrelation:
    """Test log correlation and structured logging across services."""

    @pytest.mark.asyncio
    async def test_structured_logging_correlation(self):
        """Test structured logging with correlation IDs across services."""

        class StructuredLogger:
            def __init__(self, service_name: str):
                self.service_name = service_name
                self.logs = []

            def log(
                self,
                level: str,
                message: str,
                correlation_id: str | None = None,
                trace_id: str | None = None,
                span_id: str | None = None,
                **_kwargs,
            ):
                """Log with structured format."""
                log_entry = {
                    "timestamp": time.time(),
                    "level": level,
                    "service": self.service_name,
                    "message": message,
                    "correlation_id": correlation_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    **_kwargs,
                }
                self.logs.append(log_entry)

            def info(self, message: str, **_kwargs):
                self.log("INFO", message, **_kwargs)

            def error(self, message: str, **_kwargs):
                self.log("ERROR", message, **_kwargs)

            def warning(self, message: str, **_kwargs):
                self.log("WARNING", message, **_kwargs)

        # Create loggers for different services
        api_logger = StructuredLogger("api_gateway")
        embedding_logger = StructuredLogger("embedding_service")
        vector_db_logger = StructuredLogger("vector_db_service")

        # Simulate correlated request flow
        correlation_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        # API Gateway logs
        api_span_id = str(uuid.uuid4())
        api_logger.info(
            "Received search request",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=api_span_id,
            user_id="user_123",
            endpoint="/api/v1/search",
            request_size_bytes=256,
        )

        # Embedding Service logs
        embedding_span_id = str(uuid.uuid4())
        embedding_logger.info(
            "Starting embedding generation",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=embedding_span_id,
            parent_span_id=api_span_id,
            text_count=1,
            model="text-embedding-3-small",
        )

        embedding_logger.info(
            "Embedding generation completed",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=embedding_span_id,
            duration_ms=89,
            tokens_used=150,
            cost_usd=0.001,
        )

        # Vector DB Service logs
        vector_db_span_id = str(uuid.uuid4())
        vector_db_logger.info(
            "Performing hybrid search",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=vector_db_span_id,
            parent_span_id=api_span_id,
            collection="documents",
            search_type="hybrid",
        )

        vector_db_logger.info(
            "Search completed",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=vector_db_span_id,
            results_count=15,
            search_time_ms=45,
        )

        # API Gateway completion log
        api_logger.info(
            "Request completed successfully",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=api_span_id,
            _total_duration_ms=156,
            response_size_bytes=2048,
        )

        # Collect all logs
        all_logs = []
        all_logs.extend(api_logger.logs)
        all_logs.extend(embedding_logger.logs)
        all_logs.extend(vector_db_logger.logs)

        # Test log correlation
        correlated_logs = [
            log for log in all_logs if log["correlation_id"] == correlation_id
        ]
        trace_logs = [log for log in all_logs if log["trace_id"] == trace_id]

        assert len(correlated_logs) == 5  # All logs should have correlation ID
        assert len(trace_logs) == 5  # All logs should have trace ID

        # Verify log structure
        for log in correlated_logs:
            assert "timestamp" in log
            assert "level" in log
            assert "service" in log
            assert "message" in log
            assert log["correlation_id"] == correlation_id
            assert log["trace_id"] == trace_id

        # Test log filtering by service
        api_logs = [log for log in all_logs if log["service"] == "api_gateway"]
        embedding_logs = [
            log for log in all_logs if log["service"] == "embedding_service"
        ]
        vector_db_logs = [
            log for log in all_logs if log["service"] == "vector_db_service"
        ]

        assert len(api_logs) == 2
        assert len(embedding_logs) == 2
        assert len(vector_db_logs) == 1

    @pytest.mark.asyncio
    async def test_error_correlation_and_debugging(self):
        """Test error correlation and debugging across services."""

        class ErrorCorrelationManager:
            def __init__(self):
                self.error_chains = {}
                self.error_contexts = {}

            def record_error(
                self,
                error_id: str,
                service: str,
                error_type: str,
                message: str,
                correlation_id: str,
                context: dict | None = None,
            ):
                """Record an error with correlation context."""
                error_record = {
                    "error_id": error_id,
                    "service": service,
                    "error_type": error_type,
                    "message": message,
                    "correlation_id": correlation_id,
                    "timestamp": time.time(),
                    "context": context or {},
                }

                if correlation_id not in self.error_chains:
                    self.error_chains[correlation_id] = []
                self.error_chains[correlation_id].append(error_record)

                self.error_contexts[error_id] = error_record

            def get_error_chain(self, correlation_id: str) -> list[dict]:
                """Get all errors in a correlation chain."""
                return self.error_chains.get(correlation_id, [])

            def analyze_error_propagation(self, correlation_id: str) -> dict:
                """Analyze how errors propagated through services."""
                error_chain = self.get_error_chain(correlation_id)

                if not error_chain:
                    return {"has_errors": False}

                # Sort by timestamp
                error_chain.sort(key=lambda x: x["timestamp"])

                root_cause = error_chain[0]
                affected_services = list({error["service"] for error in error_chain})

                return {
                    "has_errors": True,
                    "error_count": len(error_chain),
                    "root_cause": root_cause,
                    "affected_services": affected_services,
                    "propagation_timeline": error_chain,
                }

        error_manager = ErrorCorrelationManager()

        # Simulate error propagation scenario
        correlation_id = str(uuid.uuid4())

        # Root cause: OpenAI API error in embedding service
        error_manager.record_error(
            error_id="err_001",
            service="embedding_service",
            error_type="APIConnectionError",
            message="Failed to connect to OpenAI API: Connection timeout",
            correlation_id=correlation_id,
            context={
                "api_endpoint": "https://api.openai.com/v1/embeddings",
                "timeout_seconds": 30,
                "retry_attempt": 3,
                "model": "text-embedding-3-small",
            },
        )

        # Propagated error: Vector DB can't perform search without embeddings
        await asyncio.sleep(0.01)  # Small delay to ensure timestamp order
        error_manager.record_error(
            error_id="err_002",
            service="vector_db_service",
            error_type="MissingEmbeddingError",
            message="Cannot perform vector search: embedding generation failed",
            correlation_id=correlation_id,
            context={
                "collection": "documents",
                "search_type": "hybrid",
                "fallback_attempted": True,
                "fallback_result": "keyword_search_only",
            },
        )

        # Final error: API Gateway returns degraded response
        await asyncio.sleep(0.01)
        error_manager.record_error(
            error_id="err_003",
            service="api_gateway",
            error_type="DegradedServiceError",
            message="Returning degraded search results due to embedding service failure",
            correlation_id=correlation_id,
            context={
                "user_id": "user_123",
                "degraded_mode": "keyword_only",
                "error_code": "EMBEDDING_SERVICE_UNAVAILABLE",
                "user_notified": True,
            },
        )

        # Analyze error propagation
        error_analysis = error_manager.analyze_error_propagation(correlation_id)

        # Verify error chain analysis
        assert error_analysis["has_errors"] is True
        assert error_analysis["error_count"] == 3
        assert error_analysis["root_cause"]["service"] == "embedding_service"
        assert error_analysis["root_cause"]["error_type"] == "APIConnectionError"

        affected_services = set(error_analysis["affected_services"])
        expected_services = {"embedding_service", "vector_db_service", "api_gateway"}
        assert affected_services == expected_services

        # Verify timeline order
        timeline = error_analysis["propagation_timeline"]
        assert timeline[0]["service"] == "embedding_service"  # Root cause first
        assert timeline[1]["service"] == "vector_db_service"  # Propagated second
        assert timeline[2]["service"] == "api_gateway"  # Final error last

        # Verify context preservation
        assert (
            timeline[0]["context"]["api_endpoint"]
            == "https://api.openai.com/v1/embeddings"
        )
        assert timeline[1]["context"]["fallback_attempted"] is True
        assert timeline[2]["context"]["degraded_mode"] == "keyword_only"


class TestHealthMonitoring:
    """Test health monitoring and alerting integration."""

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self):
        """Test comprehensive service health monitoring."""

        class ServiceHealthMonitor:
            def __init__(self):
                self.health_checks = {}
                self.health_history = {}

            def register_health_check(
                self, service: str, check_func, check_interval: float = 30
            ):
                """Register a health check for a service."""
                self.health_checks[service] = {
                    "check_func": check_func,
                    "interval": check_interval,
                    "last_check": 0,
                    "status": "unknown",
                }
                self.health_history[service] = []

            async def perform_health_checks(self):
                """Perform health checks for all registered services."""
                current_time = time.time()
                health_results = {}

                for service, check_config in self.health_checks.items():
                    if (
                        current_time - check_config["last_check"]
                        >= check_config["interval"]
                    ):
                        try:
                            health_result = await check_config["check_func"]()
                            check_config["status"] = health_result["status"]
                            check_config["last_check"] = current_time

                            # Record health history
                            health_record = {
                                "timestamp": current_time,
                                "status": health_result["status"],
                                "details": health_result.get("details", {}),
                            }
                            self.health_history[service].append(health_record)

                            health_results[service] = health_result

                        except (
                            TimeoutError,
                            ConnectionError,
                            RuntimeError,
                            ValueError,
                        ) as e:
                            check_config["status"] = "error"
                            health_results[service] = {
                                "status": "error",
                                "error": str(e),
                            }

                return health_results

            def get_overall_health(self) -> dict:
                """Get overall system health."""
                service_statuses = [
                    config["status"] for config in self.health_checks.values()
                ]

                if all(status == "healthy" for status in service_statuses):
                    overall_status = "healthy"
                elif any(status == "error" for status in service_statuses):
                    overall_status = "unhealthy"
                else:
                    overall_status = "degraded"

                return {
                    "overall_status": overall_status,
                    "service_count": len(self.health_checks),
                    "healthy_services": sum(
                        1 for s in service_statuses if s == "healthy"
                    ),
                    "degraded_services": sum(
                        1 for s in service_statuses if s == "degraded"
                    ),
                    "unhealthy_services": sum(
                        1 for s in service_statuses if s == "error"
                    ),
                }

        # Mock health check functions
        async def api_gateway_health():
            return {
                "status": "healthy",
                "details": {
                    "active_connections": 15,
                    "memory_usage": 0.45,
                    "cpu_usage": 0.23,
                    "response_time_p95": 0.125,
                },
            }

        async def embedding_service_health():
            return {
                "status": "degraded",
                "details": {
                    "openai_api_status": "slow",
                    "cache_hit_rate": 0.67,
                    "queue_size": 5,
                    "avg_response_time": 2.1,  # Slower than normal
                },
            }

        async def vector_db_health():
            return {
                "status": "healthy",
                "details": {
                    "collections_count": 3,
                    "memory_usage": 0.68,
                    "query_performance_ms": 45,
                    "index_status": "optimized",
                },
            }

        async def cache_service_health():
            msg = "Cache service unreachable"
            raise ConnectionError(msg)

        # Setup health monitoring
        health_monitor = ServiceHealthMonitor()
        health_monitor.register_health_check("api_gateway", api_gateway_health, 30)
        health_monitor.register_health_check(
            "embedding_service", embedding_service_health, 30
        )
        health_monitor.register_health_check("vector_db_service", vector_db_health, 30)
        health_monitor.register_health_check("cache_service", cache_service_health, 30)

        # Perform health checks
        health_results = await health_monitor.perform_health_checks()
        overall_health = health_monitor.get_overall_health()

        # Verify health check results
        assert "api_gateway" in health_results
        assert "embedding_service" in health_results
        assert "vector_db_service" in health_results
        assert "cache_service" in health_results

        assert health_results["api_gateway"]["status"] == "healthy"
        assert health_results["embedding_service"]["status"] == "degraded"
        assert health_results["vector_db_service"]["status"] == "healthy"
        assert health_results["cache_service"]["status"] == "error"

        # Verify overall health assessment
        assert (
            overall_health["overall_status"] == "unhealthy"
        )  # Due to cache service error
        assert overall_health["service_count"] == 4
        assert overall_health["healthy_services"] == 2
        assert overall_health["degraded_services"] == 1
        assert overall_health["unhealthy_services"] == 1

        # Verify health details
        assert health_results["api_gateway"]["details"]["active_connections"] == 15
        assert (
            health_results["embedding_service"]["details"]["openai_api_status"]
            == "slow"
        )
        assert (
            health_results["vector_db_service"]["details"]["index_status"]
            == "optimized"
        )
        assert "Cache service unreachable" in health_results["cache_service"]["error"]

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration with observability."""

        class PerformanceMonitor:
            def __init__(self):
                self.performance_data = []
                self.baselines = {}

            def record_performance(
                self,
                service: str,
                operation: str,
                duration: float,
                metadata: dict | None = None,
            ):
                """Record performance data."""
                record = {
                    "timestamp": time.time(),
                    "service": service,
                    "operation": operation,
                    "duration": duration,
                    "metadata": metadata or {},
                }
                self.performance_data.append(record)

            def establish_baseline(
                self,
                service: str,
                operation: str,
                baseline_duration: float,
                tolerance: float = 0.2,
            ):
                """Establish performance baseline."""
                self.baselines[f"{service}.{operation}"] = {
                    "baseline_duration": baseline_duration,
                    "tolerance": tolerance,
                    "warning_threshold": baseline_duration * (1 + tolerance),
                    "critical_threshold": baseline_duration * (1 + tolerance * 2),
                }

            def analyze_performance_degradation(self) -> list[dict]:
                """Analyze performance degradation against baselines."""
                degradations = []

                for record in self.performance_data:
                    key = f"{record['service']}.{record['operation']}"
                    if key in self.baselines:
                        baseline = self.baselines[key]

                        if record["duration"] > baseline["critical_threshold"]:
                            severity = "critical"
                        elif record["duration"] > baseline["warning_threshold"]:
                            severity = "warning"
                        else:
                            continue  # Performance is acceptable

                        degradation = {
                            "service": record["service"],
                            "operation": record["operation"],
                            "current_duration": record["duration"],
                            "baseline_duration": baseline["baseline_duration"],
                            "degradation_percentage": (
                                (record["duration"] - baseline["baseline_duration"])
                                / baseline["baseline_duration"]
                                * 100
                            ),
                            "severity": severity,
                            "timestamp": record["timestamp"],
                        }
                        degradations.append(degradation)

                return degradations

        performance_monitor = PerformanceMonitor()

        # Establish baselines
        performance_monitor.establish_baseline(
            "api_gateway", "search_request", 0.15, 0.3
        )
        performance_monitor.establish_baseline(
            "embedding_service", "generate_embeddings", 0.1, 0.5
        )
        performance_monitor.establish_baseline(
            "vector_db_service", "hybrid_search", 0.05, 0.4
        )

        # Record normal performance
        performance_monitor.record_performance("api_gateway", "search_request", 0.14)
        performance_monitor.record_performance(
            "embedding_service", "generate_embeddings", 0.09
        )
        performance_monitor.record_performance(
            "vector_db_service", "hybrid_search", 0.048
        )

        # Record degraded performance
        performance_monitor.record_performance(
            "api_gateway", "search_request", 0.22
        )  # Warning
        performance_monitor.record_performance(
            "embedding_service", "generate_embeddings", 0.18
        )  # Warning
        performance_monitor.record_performance(
            "vector_db_service", "hybrid_search", 0.12
        )  # Critical

        # Analyze performance degradation
        degradations = performance_monitor.analyze_performance_degradation()

        # Verify degradation detection
        assert len(degradations) == 3  # Three degraded operations

        # Find specific degradations
        api_degradation = next(d for d in degradations if d["service"] == "api_gateway")
        embedding_degradation = next(
            d for d in degradations if d["service"] == "embedding_service"
        )
        vector_db_degradation = next(
            d for d in degradations if d["service"] == "vector_db_service"
        )

        # Verify degradation analysis
        assert api_degradation["severity"] == "warning"
        assert api_degradation["degradation_percentage"] > 30  # Should be around 47%

        assert embedding_degradation["severity"] == "warning"
        assert (
            embedding_degradation["degradation_percentage"] > 50
        )  # Should be around 80%

        assert vector_db_degradation["severity"] == "critical"
        assert (
            vector_db_degradation["degradation_percentage"] > 100
        )  # Should be around 140%

        # Verify baseline comparisons
        assert api_degradation["baseline_duration"] == 0.15
        assert embedding_degradation["baseline_duration"] == 0.1
        assert vector_db_degradation["baseline_duration"] == 0.05
