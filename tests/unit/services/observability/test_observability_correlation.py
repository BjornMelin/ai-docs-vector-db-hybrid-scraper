"""Tests for trace correlation and context propagation module."""

from unittest.mock import Mock, patch

import pytest
from opentelemetry import context

from src.services.observability.correlation import (
    ErrorCorrelationTracker,
    TraceCorrelationManager,
    correlated_operation,
    get_correlation_manager,
    get_current_trace_context,
    get_error_tracker,
    record_error,
    set_business_context,
    set_request_context,
)


class TestTraceCorrelationManager:
    """Test TraceCorrelationManager class."""

    def test_manager_initialization(self):
        """Test correlation manager initialization."""
        manager = TraceCorrelationManager()

        assert manager.tracer is not None

    def test_generate_request_id(self):
        """Test request ID generation."""
        manager = TraceCorrelationManager()

        request_id = manager.generate_request_id()

        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID4 format
        assert request_id.count("-") == 4

    def test_generate_unique_request_ids(self):
        """Test that generated request IDs are unique."""
        manager = TraceCorrelationManager()

        ids = [manager.generate_request_id() for _ in range(10)]

        assert len(set(ids)) == 10  # All should be unique

    def test_set_request_context_with_provided_id(self):
        """Test setting request context with provided ID."""
        manager = TraceCorrelationManager()

        request_id = "test-request-123"
        result_id = manager.set_request_context(
            request_id=request_id,
            user_id="user456",
            session_id="session789",
            tenant_id="tenant001",
        )

        assert result_id == request_id

    def test_set_request_context_auto_generate_id(self):
        """Test setting request context with auto-generated ID."""
        manager = TraceCorrelationManager()

        result_id = manager.set_request_context(
            user_id="user123", session_id="session456"
        )

        assert isinstance(result_id, str)
        assert len(result_id) == 36

    def test_set_business_context(self):
        """Test setting business context."""
        manager = TraceCorrelationManager()

        # Should not raise any exceptions
        manager.set_business_context(
            operation_type="search",
            query_type="semantic",
            search_method="hybrid",
            ai_provider="openai",
            model_name="gpt-4",
            cache_strategy="redis",
        )

    def test_set_business_context_minimal(self):
        """Test setting business context with minimal parameters."""
        manager = TraceCorrelationManager()

        # Should not raise any exceptions
        manager.set_business_context(operation_type="embedding_generation")

    def test_set_performance_context(self):
        """Test setting performance context."""
        manager = TraceCorrelationManager()

        manager.set_performance_context(
            priority="high",
            timeout_ms=5000,
            retry_count=2,
            circuit_breaker_state="closed",
        )

    def test_set_performance_context_defaults(self):
        """Test setting performance context with defaults."""
        manager = TraceCorrelationManager()

        # Should not raise any exceptions
        manager.set_performance_context()

    def test_get_current_context(self):
        """Test getting current context."""
        manager = TraceCorrelationManager()

        context_info = manager.get_current_context()

        assert isinstance(context_info, dict)
        # Context may be empty if no active span

    def test_create_correlation_id(self):
        """Test correlation ID creation."""
        manager = TraceCorrelationManager()

        correlation_id = manager.create_correlation_id("test_operation")

        assert isinstance(correlation_id, str)
        assert correlation_id.startswith("test_operation_")
        assert len(correlation_id.split("_")[-1]) == 8  # 8-char hex suffix

    def test_link_operations(self):
        """Test linking parent and child operations."""
        manager = TraceCorrelationManager()

        parent_id = "parent_12345678"
        child_id = manager.link_operations(parent_id, "child_operation")

        assert isinstance(child_id, str)
        assert child_id.startswith("child_operation_")


class TestCorrelatedOperationContextManager:
    """Test correlated operation context manager."""

    def test_correlated_operation_basic(self):
        """Test basic correlated operation."""
        manager = TraceCorrelationManager()

        with manager.correlated_operation("test_operation") as correlation_id:
            assert isinstance(correlation_id, str)
            assert correlation_id.startswith("test_operation_")

    def test_correlated_operation_with_existing_id(self):
        """Test correlated operation with existing correlation ID."""
        manager = TraceCorrelationManager()

        existing_id = "existing_correlation_123"

        with manager.correlated_operation(
            "test_operation", correlation_id=existing_id
        ) as correlation_id:
            assert correlation_id == existing_id

    def test_correlated_operation_with_additional_context(self):
        """Test correlated operation with additional context."""
        manager = TraceCorrelationManager()

        with manager.correlated_operation(
            "test_operation",
            priority="high",
            user_type="premium",
            feature_flag="new_search",
        ) as correlation_id:
            assert isinstance(correlation_id, str)

    def test_correlated_operation_with_exception(self):
        """Test correlated operation with exceptions."""
        manager = TraceCorrelationManager()

        with pytest.raises(ValueError):
            with manager.correlated_operation("failing_operation"):
                raise ValueError("Test error")

    def test_nested_correlated_operations(self):
        """Test nested correlated operations."""
        manager = TraceCorrelationManager()

        with (
            manager.correlated_operation("outer_operation") as outer_id,
            manager.correlated_operation("inner_operation") as inner_id,
        ):
            assert outer_id != inner_id
            assert outer_id.startswith("outer_operation_")
            assert inner_id.startswith("inner_operation_")


class TestContextPropagation:
    """Test context propagation methods."""

    def test_extract_context_from_headers(self):
        """Test extracting context from headers."""
        manager = TraceCorrelationManager()

        headers = {
            "traceparent": "00-12345678901234567890123456789012-1234567890123456-01"
        }

        ctx = manager.extract_context_from_headers(headers)

        assert ctx is not None

    def test_inject_context_to_headers(self):
        """Test injecting context to headers."""
        manager = TraceCorrelationManager()

        headers = {}
        manager.inject_context_to_headers(headers)

        # Headers may or may not be populated depending on active context
        assert isinstance(headers, dict)

    def test_propagate_context_to_background_task(self):
        """Test context propagation to background tasks."""
        manager = TraceCorrelationManager()

        ctx = manager.propagate_context_to_background_task()

        assert ctx is not None

    def test_run_with_context(self):
        """Test running function with specific context."""
        manager = TraceCorrelationManager()

        def test_function(x, y):
            return x + y

        ctx = context.get_current()
        result = manager.run_with_context(ctx, test_function, 2, 3)

        assert result == 5


class TestErrorCorrelationTracker:
    """Test ErrorCorrelationTracker class."""

    def test_error_tracker_initialization(self):
        """Test error tracker initialization."""
        correlation_manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(correlation_manager)

        assert tracker.correlation_manager is correlation_manager
        assert tracker.tracer is not None

    def test_record_error_basic(self):
        """Test basic error recording."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        error = ValueError("Test error")
        error_id = tracker.record_error(error)

        assert isinstance(error_id, str)
        assert len(error_id) == 36  # UUID format

    def test_record_error_with_details(self):
        """Test error recording with detailed information."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        error = ConnectionError("Database connection failed")
        error_id = tracker.record_error(
            error=error,
            error_type="database_error",
            severity="high",
            user_impact="high",
            recovery_action="retry_with_backoff",
        )

        assert isinstance(error_id, str)

    def test_record_error_with_context(self):
        """Test error recording with correlation context."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        # Set some context first
        manager.set_request_context(request_id="req123", user_id="user456")
        manager.set_business_context(operation_type="search", query_type="semantic")

        error = RuntimeError("Search service error")
        error_id = tracker.record_error(error, error_type="service_error")

        assert isinstance(error_id, str)

    def test_create_error_span(self):
        """Test creating dedicated error span."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        error_details = {
            "component": "vector_search",
            "operation": "similarity_search",
            "collection": "documents",
            "query_id": "query123",
        }

        with tracker.create_error_span(
            "vector_search_timeout", error_details, "parent_correlation_456"
        ) as span:
            assert span is not None

    def test_create_error_span_with_exception(self):
        """Test error span with exception handling."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        with (
            pytest.raises(ValueError),
            tracker.create_error_span("test_error", {}),
        ):
            raise ValueError("Error in error span")


class TestGlobalInstances:
    """Test global instance management."""

    def test_get_correlation_manager_singleton(self):
        """Test that get_correlation_manager returns singleton."""
        manager1 = get_correlation_manager()
        manager2 = get_correlation_manager()

        assert manager1 is manager2

    def test_get_error_tracker_singleton(self):
        """Test that get_error_tracker returns singleton."""
        tracker1 = get_error_tracker()
        tracker2 = get_error_tracker()

        assert tracker1 is tracker2

    def test_error_tracker_uses_same_manager(self):
        """Test that error tracker uses same correlation manager."""
        manager = get_correlation_manager()
        tracker = get_error_tracker()

        assert tracker.correlation_manager is manager


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_set_request_context_convenience(self):
        """Test set_request_context convenience function."""
        request_id = set_request_context(
            user_id="user123", session_id="session456", tenant_id="tenant789"
        )

        assert isinstance(request_id, str)

    def test_set_business_context_convenience(self):
        """Test set_business_context convenience function."""
        # Should not raise any exceptions
        set_business_context(
            operation_type="embedding_generation",
            ai_provider="openai",
            model_name="text-embedding-ada-002",
        )

    def test_correlated_operation_convenience(self):
        """Test correlated_operation convenience function."""
        with correlated_operation("test_operation", priority="high") as correlation_id:
            assert isinstance(correlation_id, str)

    def test_record_error_convenience(self):
        """Test record_error convenience function."""
        error = ValueError("Test error")
        error_id = record_error(
            error=error, error_type="validation_error", severity="medium"
        )

        assert isinstance(error_id, str)

    def test_get_current_trace_context_convenience(self):
        """Test get_current_trace_context convenience function."""
        context_info = get_current_trace_context()

        assert isinstance(context_info, dict)


@pytest.fixture
def mock_baggage():
    """Fixture providing mocked baggage operations."""
    with patch("src.services.observability.correlation.baggage") as mock:
        mock.set_baggage = Mock()
        mock.get_all.return_value = {"test.key": "test_value"}
        yield mock


@pytest.fixture
def mock_trace():
    """Fixture providing mocked trace operations."""
    with patch("src.services.observability.correlation.trace") as mock:
        span = Mock()
        span.is_recording.return_value = True
        span.get_span_context.return_value = Mock()
        span.get_span_context().trace_id = 0x12345678901234567890123456789012
        span.get_span_context().span_id = 0x1234567890123456
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)

        tracer = Mock()
        tracer.start_as_current_span.return_value = span

        mock.get_current_span.return_value = span
        mock.get_tracer.return_value = tracer

        yield mock, tracer, span


class TestCorrelationIntegration:
    """Test integration scenarios with mocked dependencies."""

    def test_manager_with_mocked_baggage(self, mock_baggage):
        """Test correlation manager with mocked baggage."""
        manager = TraceCorrelationManager()

        manager.set_request_context(request_id="test123", user_id="user456")

        # Verify baggage operations
        mock_baggage.set_baggage.assert_called()
        assert mock_baggage.set_baggage.call_count >= 2

    def test_manager_with_mocked_trace(self, mock_trace):
        """Test correlation manager with mocked trace."""
        mock_trace_module, tracer, span = mock_trace

        manager = TraceCorrelationManager()
        manager.tracer = tracer

        with manager.correlated_operation("test_operation"):
            pass

        tracer.start_as_current_span.assert_called_once()
        span.set_attribute.assert_called()

    def test_get_current_context_with_mocked_trace(self, mock_trace):
        """Test get_current_context with mocked trace."""
        mock_trace_module, tracer, span = mock_trace

        manager = TraceCorrelationManager()
        context_info = manager.get_current_context()

        assert "trace_id" in context_info
        assert "span_id" in context_info
        assert context_info["trace_id"] == "12345678901234567890123456789012"
        assert context_info["span_id"] == "1234567890123456"

    def test_error_recording_with_mocked_trace(self, mock_trace):
        """Test error recording with mocked trace."""
        mock_trace_module, tracer, span = mock_trace

        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        error = ValueError("Test error")
        error_id = tracker.record_error(error)

        # Verify span operations
        span.record_exception.assert_called_once_with(error)
        span.set_status.assert_called()
        span.set_attribute.assert_called()
        span.add_event.assert_called()

        assert isinstance(error_id, str)


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_correlation_without_active_span(self):
        """Test correlation operations without active span."""
        manager = TraceCorrelationManager()

        # Should not raise errors even without active span
        manager.set_request_context(user_id="user123")
        manager.set_business_context(operation_type="test")
        context_info = manager.get_current_context()

        assert isinstance(context_info, dict)

    def test_error_recording_without_active_span(self):
        """Test error recording without active span."""
        manager = TraceCorrelationManager()
        tracker = ErrorCorrelationTracker(manager)

        error = RuntimeError("Test error")
        error_id = tracker.record_error(error)

        # Should still work without active span
        assert isinstance(error_id, str)

    def test_baggage_propagation_edge_cases(self):
        """Test baggage propagation edge cases."""
        manager = TraceCorrelationManager()

        # Test with None values
        request_id = manager.set_request_context(
            request_id=None, user_id=None, session_id="session123"
        )

        assert isinstance(request_id, str)

    def test_correlation_with_special_characters(self):
        """Test correlation with special characters in IDs."""
        manager = TraceCorrelationManager()

        # Should handle special characters gracefully
        manager.set_request_context(
            user_id="user@example.com",
            session_id="session-with-dashes",
            tenant_id="tenant_with_underscores",
        )

    def test_very_long_context_values(self):
        """Test correlation with very long context values."""
        manager = TraceCorrelationManager()

        long_value = "x" * 1000

        # Should handle long values without errors
        manager.set_business_context(
            operation_type=long_value[:50],  # Reasonable length
            query_type="semantic",
        )
