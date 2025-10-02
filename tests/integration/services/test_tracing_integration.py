"""Distributed tracing integration tests.

This module tests distributed tracing functionality across service boundaries,
validating trace propagation, span hierarchy, sampling, and error tracking.
"""

import pytest

from tests.fixtures.test_data_observability import ObservabilityTestDataFactory
from tests.fixtures.test_utils_observability import (
    AsyncTestHelper,
    ObservabilityTestAssertions,
    TraceTestHelper,
)


class TestTraceSpanCreation:
    """Test basic span creation and properties."""

    def test_create_root_span(self, mock_distributed_tracer):
        """Test creating a root span with no parent."""
        tracer = mock_distributed_tracer

        span = tracer.start_span("api_gateway", "handle_request")

        assert span.service_name == "api_gateway"
        assert span.operation_name == "handle_request"
        assert span.parent_span_id is None
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.start_time is not None
        assert span.end_time is None  # Not finished yet

    def test_create_child_span(self, mock_distributed_tracer):
        """Test creating a child span with a parent."""
        tracer = mock_distributed_tracer

        parent_span = tracer.start_span("api_gateway", "handle_request")
        child_span = tracer.start_span(
            "embedding_service",
            "generate_embeddings",
            parent_span_id=parent_span.span_id,
        )

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id
        assert child_span.service_name == "embedding_service"

    def test_finish_span_with_tags(self, mock_distributed_tracer):
        """Test finishing a span and adding tags."""
        tracer = mock_distributed_tracer

        span = tracer.start_span("api_gateway", "handle_request")
        tags = {"http.method": "POST", "http.status_code": 200}

        tracer.finish_span(span, tags)

        assert span.end_time is not None
        assert span.tags == tags
        assert span.end_time > span.start_time

    def test_span_duration_calculation(self, mock_distributed_tracer):
        """Test that span duration is calculated correctly."""
        tracer = mock_distributed_tracer

        span = tracer.start_span("test_service", "test_operation")
        tracer.finish_span(span)

        duration = span.end_time - span.start_time
        assert duration >= 0  # Should be non-negative


class TestTraceHierarchy:
    """Test trace hierarchy and relationships."""

    def test_simple_trace_hierarchy(self, mock_distributed_tracer):
        """Test a simple parent-child span relationship."""
        tracer = mock_distributed_tracer

        # Create root span
        root_span = tracer.start_span("api_gateway", "handle_request")

        # Create child spans
        embedding_span = tracer.start_span(
            "embedding_service", "generate_embeddings", parent_span_id=root_span.span_id
        )
        vector_span = tracer.start_span(
            "vector_db_service", "search", parent_span_id=root_span.span_id
        )

        # Finish all spans
        tracer.finish_span(embedding_span)
        tracer.finish_span(vector_span)
        tracer.finish_span(root_span)

        # Get all spans
        spans = tracer.get_spans_for_trace(root_span.trace_id)

        # Validate hierarchy
        ObservabilityTestAssertions.assert_trace_hierarchy(
            spans, "api_gateway", ["embedding_service", "vector_db_service"]
        )

    def test_nested_trace_hierarchy(self, mock_distributed_tracer):
        """Test deeply nested span relationships."""
        tracer = mock_distributed_tracer

        # Root span
        root = tracer.start_span("api_gateway", "handle_request")

        # Level 1 children
        service_a = tracer.start_span(
            "service_a", "operation_a", parent_span_id=root.span_id
        )
        service_b = tracer.start_span(
            "service_b", "operation_b", parent_span_id=root.span_id
        )

        # Level 2 children
        service_a1 = tracer.start_span(
            "service_a1", "operation_a1", parent_span_id=service_a.span_id
        )

        # Finish spans in reverse order
        tracer.finish_span(service_a1)
        tracer.finish_span(service_a)
        tracer.finish_span(service_b)
        tracer.finish_span(root)

        # Validate structure
        spans = tracer.get_spans_for_trace(root.trace_id)
        validation = TraceTestHelper.validate_trace_structure(spans)

        assert validation["valid"], f"Invalid trace structure: {validation['issues']}"
        assert validation["span_count"] == 4
        assert validation["root_spans"] == 1

    def test_trace_duration_calculation(self, mock_distributed_tracer):
        """Test calculation of total trace duration."""
        tracer = mock_distributed_tracer

        # Create a trace with known timing
        root = tracer.start_span("test_service", "test_operation")
        tracer.finish_span(root)

        spans = [root]
        duration = TraceTestHelper.calculate_trace_duration(spans)

        assert duration >= 0
        assert duration == root.end_time - root.start_time


class TestTracePropagation:
    """Test trace context propagation across service boundaries."""

    def test_trace_id_consistency(self, mock_distributed_tracer):
        """Test that trace ID remains consistent across spans."""
        tracer = mock_distributed_tracer

        trace_id = "test-trace-123"

        span1 = tracer.start_span("service1", "op1", trace_id=trace_id)
        span2 = tracer.start_span(
            "service2", "op2", parent_span_id=span1.span_id, trace_id=trace_id
        )
        span3 = tracer.start_span(
            "service3", "op3", parent_span_id=span2.span_id, trace_id=trace_id
        )

        assert span1.trace_id == trace_id
        assert span2.trace_id == trace_id
        assert span3.trace_id == trace_id

    def test_span_id_uniqueness(self, mock_distributed_tracer):
        """Test that span IDs are unique within a trace."""
        tracer = mock_distributed_tracer

        span1 = tracer.start_span("service1", "op1")
        span2 = tracer.start_span("service2", "op2", parent_span_id=span1.span_id)
        span3 = tracer.start_span("service3", "op3", parent_span_id=span2.span_id)

        span_ids = {span1.span_id, span2.span_id, span3.span_id}
        assert len(span_ids) == 3  # All unique


class TestDistributedTracingIntegration:
    """Integration tests for complete distributed tracing scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_request_tracing(self, mock_distributed_tracer):
        """Test end-to-end request tracing across multiple services."""
        tracer = mock_distributed_tracer

        # Create realistic request trace using factory
        spans = ObservabilityTestDataFactory.create_request_trace_spans()

        # Store spans using the proper tracer method
        for span in spans:
            tracer.finish_span(span)

        # Validate the complete trace
        trace_spans = tracer.get_spans_for_trace(spans[0].trace_id)

        assert len(trace_spans) == 4  # api + embedding + vector + content

        # Validate hierarchy
        ObservabilityTestAssertions.assert_trace_hierarchy(
            trace_spans,
            "api_gateway",
            ["embedding_service", "vector_db_service", "content_intelligence"],
        )

        # Validate timing relationships
        validation = TraceTestHelper.validate_trace_structure(trace_spans)
        assert validation["valid"], f"Trace validation failed: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_concurrent_trace_operations(self, mock_distributed_tracer):
        """Test tracing with concurrent operations."""
        tracer = mock_distributed_tracer

        # Simulate concurrent service calls
        operations = [
            {"name": "embedding_call", "duration_ms": 80},
            {"name": "vector_search", "duration_ms": 45},
            {"name": "content_analysis", "duration_ms": 25},
        ]

        results = await AsyncTestHelper.run_concurrent_operations(operations)

        # Create spans based on results
        root_span = tracer.start_span("api_gateway", "handle_request")

        for result in results:
            if isinstance(result, dict):  # Successful operation
                service_name = (
                    result["operation"].replace("_call", "").replace("_", "_")
                )
                span = tracer.start_span(
                    service_name, result["operation"], parent_span_id=root_span.span_id
                )
                tracer.finish_span(span)

        tracer.finish_span(root_span)

        # Validate concurrent execution
        spans = tracer.get_spans_for_trace(root_span.trace_id)
        assert len(spans) == 4  # root + 3 concurrent operations

    @pytest.mark.asyncio
    async def test_trace_with_service_failures(self, mock_distributed_tracer):
        """Test tracing when services fail."""
        tracer = mock_distributed_tracer

        # Create trace with error spans
        spans = ObservabilityTestDataFactory.create_request_trace_spans(
            include_errors=True
        )

        # Store spans using proper tracer method
        for span in spans:
            tracer.finish_span(span)

        # Find error spans
        error_spans = [s for s in spans if s.tags.get("error") is True]

        assert len(error_spans) == 1
        assert error_spans[0].service_name == "embedding_service"
        error_message = error_spans[0].tags.get("error.message", "")
        assert "OpenAI API connection timeout" in error_message

    @pytest.mark.asyncio
    async def test_trace_sampling_simulation(self, mock_distributed_tracer):
        """Test trace sampling behavior."""
        tracer = mock_distributed_tracer

        # Create multiple traces
        traces = []
        for i in range(10):
            trace_id = f"trace-{i}"
            span = tracer.start_span(
                "test_service", "test_operation", trace_id=trace_id
            )
            tracer.finish_span(span)
            traces.append(trace_id)

        # Simulate sampling (every other trace)
        sampled_traces = traces[::2]

        # Verify sampled traces exist
        for trace_id in sampled_traces:
            spans = tracer.get_spans_for_trace(trace_id)
            assert len(spans) == 1
            assert spans[0].trace_id == trace_id
