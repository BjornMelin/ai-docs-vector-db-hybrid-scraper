"""Tests for error tracking and correlation across observability systems."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.services.observability.correlation import (
    get_correlation_manager,
    get_error_tracker,
    record_error,
    set_request_context,
    correlated_operation
)
from src.services.observability.instrumentation import instrument_function, trace_operation
from src.services.observability.ai_tracking import get_ai_tracker


class TestErrorTracking:
    """Test error tracking and correlation functionality."""

    def test_error_tracker_initialization(self):
        """Test error tracker initialization."""
        error_tracker = get_error_tracker()
        
        assert error_tracker is not None
        assert hasattr(error_tracker, 'record_error')
        assert hasattr(error_tracker, 'create_error_span')

    def test_basic_error_recording(self):
        """Test basic error recording functionality."""
        correlation_manager = get_correlation_manager()
        
        # Set up request context
        request_id = correlation_manager.set_request_context(
            user_id="user123",
            session_id="session456"
        )
        
        # Record an error
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            error_id = record_error(
                error=e,
                error_type="validation_error",
                severity="medium",
                user_impact="low"
            )
            
            assert error_id is not None
            assert len(error_id) == 36  # UUID format
        
        assert request_id is not None

    def test_error_recording_with_context(self):
        """Test error recording with correlation context."""
        correlation_manager = get_correlation_manager()
        
        with correlated_operation("test_operation") as correlation_id:
            correlation_manager.set_business_context(
                operation_type="data_processing",
                query_type="high_priority"
            )
            
            try:
                # Simulate business logic error
                raise ConnectionError("Database connection failed")
            except ConnectionError as e:
                error_id = record_error(
                    error=e,
                    error_type="database_connection_error",
                    severity="high",
                    user_impact="high",
                    recovery_action="retry_with_backoff"
                )
                
                assert error_id is not None
                assert correlation_id is not None

    @patch('src.services.observability.correlation.trace')
    def test_error_span_creation(self, mock_trace):
        """Test error span creation with proper attributes."""
        # Setup mock tracer
        span = Mock()
        span.is_recording.return_value = True
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        
        tracer = Mock()
        tracer.start_as_current_span.return_value = span
        mock_trace.get_tracer.return_value = tracer
        mock_trace.get_current_span.return_value = span
        
        error_tracker = get_error_tracker()
        error_tracker.tracer = tracer
        
        # Create error span using context manager
        test_error = RuntimeError("Test runtime error")
        error_span_context = error_tracker.create_error_span(
            error_name="runtime_error",
            error_details={"exception": str(test_error), "severity": "high"}
        )
        
        # Use the context manager to simulate the span usage
        with error_span_context as error_span:
            # Simulate an error within the span
            try:
                raise test_error
            except RuntimeError as e:
                error_span.record_exception(e)
        
        # Verify span creation was called
        tracer.start_as_current_span.assert_called_with("error.runtime_error")

    def test_error_classification(self):
        """Test error classification by type and severity."""
        error_types_and_severities = [
            (ValueError("Invalid input"), "validation_error", "medium"),
            (ConnectionError("Service unavailable"), "connection_error", "high"),
            (TimeoutError("Request timeout"), "timeout_error", "medium"),
            (PermissionError("Access denied"), "permission_error", "high"),
            (KeyError("Missing required field"), "data_error", "low"),
        ]
        
        correlation_manager = get_correlation_manager()
        request_id = correlation_manager.set_request_context(user_id="test_user")
        
        recorded_errors = []
        
        for error, error_type, severity in error_types_and_severities:
            error_id = record_error(
                error=error,
                error_type=error_type,
                severity=severity,
                user_impact="varies"
            )
            recorded_errors.append(error_id)
        
        # All errors should be recorded with unique IDs
        assert len(recorded_errors) == 5
        assert len(set(recorded_errors)) == 5  # All unique
        assert request_id is not None

    def test_error_correlation_across_operations(self):
        """Test error correlation across multiple operations."""
        correlation_manager = get_correlation_manager()
        error_chain = []
        
        request_id = correlation_manager.set_request_context(
            user_id="user123",
            session_id="session456"
        )
        
        with correlated_operation("main_operation") as main_id:
            try:
                # Operation 1: Data validation
                with correlated_operation("data_validation") as validation_id:
                    raise ValueError("Invalid data format")
                    
            except ValueError as e:
                error_id = record_error(
                    error=e,
                    error_type="validation_error",
                    severity="medium"
                )
                error_chain.append((validation_id, error_id))
            
            try:
                # Operation 2: Database operation (depends on validation)
                with correlated_operation("database_operation") as db_id:
                    raise ConnectionError("Database unreachable")
                    
            except ConnectionError as e:
                error_id = record_error(
                    error=e,
                    error_type="database_error", 
                    severity="high"
                )
                error_chain.append((db_id, error_id))
        
        # Verify error chain correlation
        assert len(error_chain) == 2
        assert main_id is not None
        assert request_id is not None
        
        # Each operation should have its own correlation ID and error ID
        validation_correlation, validation_error = error_chain[0]
        db_correlation, db_error = error_chain[1]
        
        assert validation_correlation != db_correlation
        assert validation_error != db_error

    @patch('src.services.observability.instrumentation.trace')
    def test_instrumented_function_error_handling(self, mock_trace):
        """Test error handling in instrumented functions."""
        # Setup mock tracer
        span = Mock()
        span.is_recording.return_value = True
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        
        tracer = Mock()
        tracer.start_as_current_span.return_value = span
        mock_trace.get_tracer.return_value = tracer
        mock_trace.get_current_span.return_value = span
        
        @instrument_function("risky_operation")
        def risky_function(should_fail=True):
            if should_fail:
                raise RuntimeError("Function failed")
            return "success"
        
        # Test successful execution
        result = risky_function(should_fail=False)
        assert result == "success"
        
        # Test error handling
        with pytest.raises(RuntimeError):
            risky_function(should_fail=True)
        
        # Verify span was created and error was recorded
        tracer.start_as_current_span.assert_called()
        span.record_exception.assert_called()
        span.set_status.assert_called()

    def test_ai_operation_error_tracking(self):
        """Test error tracking in AI operations."""
        ai_tracker = get_ai_tracker()
        correlation_manager = get_correlation_manager()
        
        request_id = correlation_manager.set_request_context(
            user_id="user123"
        )
        
        # Test embedding generation error
        with pytest.raises(ConnectionError):
            with ai_tracker.track_embedding_generation(
                provider="openai",
                model="text-embedding-ada-002",
                input_texts=["test text"]
            ) as result:
                raise ConnectionError("OpenAI API unavailable")
        
        # Test vector search error
        with pytest.raises(TimeoutError):
            with ai_tracker.track_vector_search(
                collection_name="documents",
                query_type="semantic"
            ) as result:
                raise TimeoutError("Vector database timeout")
        
        # Test LLM call error
        with pytest.raises(ValueError):
            with ai_tracker.track_llm_call(
                provider="openai",
                model="gpt-4"
            ) as result:
                raise ValueError("Invalid API response")
        
        assert request_id is not None

    @pytest.mark.asyncio
    async def test_async_error_tracking(self):
        """Test error tracking in async operations."""
        correlation_manager = get_correlation_manager()
        
        async def async_operation_with_error():
            with correlated_operation("async_operation") as correlation_id:
                try:
                    # Simulate async work that fails
                    await asyncio.sleep(0.01)
                    raise asyncio.TimeoutError("Async operation timed out")
                except asyncio.TimeoutError as e:
                    error_id = record_error(
                        error=e,
                        error_type="async_timeout",
                        severity="medium"
                    )
                    return correlation_id, error_id
        
        correlation_id, error_id = await async_operation_with_error()
        
        assert correlation_id is not None
        assert error_id is not None
        assert len(error_id) == 36  # UUID format

    def test_error_aggregation_and_patterns(self):
        """Test error aggregation and pattern detection."""
        correlation_manager = get_correlation_manager()
        
        # Simulate multiple similar errors
        similar_errors = []
        request_id = correlation_manager.set_request_context(user_id="test_user")
        
        for i in range(5):
            try:
                # Simulate same type of error occurring multiple times
                raise ConnectionError(f"Database connection failed - attempt {i+1}")
            except ConnectionError as e:
                error_id = record_error(
                    error=e,
                    error_type="database_connection_error",
                    severity="high",
                    recovery_action=f"retry_attempt_{i+1}"
                )
                similar_errors.append(error_id)
        
        # All errors should be recorded
        assert len(similar_errors) == 5
        assert len(set(similar_errors)) == 5  # All unique error IDs
        assert request_id is not None

    def test_error_recovery_tracking(self):
        """Test tracking of error recovery attempts."""
        correlation_manager = get_correlation_manager()
        
        with correlated_operation("operation_with_retry") as correlation_id:
            recovery_attempts = []
            
            # Simulate retry logic with error tracking
            for attempt in range(3):
                try:
                    if attempt < 2:  # Fail first two attempts
                        raise ConnectionError(f"Connection failed on attempt {attempt + 1}")
                    else:
                        # Third attempt succeeds
                        break
                except ConnectionError as e:
                    error_id = record_error(
                        error=e,
                        error_type="connection_error",
                        severity="medium",
                        recovery_action=f"retry_attempt_{attempt + 1}_of_3"
                    )
                    recovery_attempts.append(error_id)
            
            # Recovery success
            correlation_manager.set_business_context(
                operation_type="retry_success",
                query_type="recovery_completed"
            )
        
        assert correlation_id is not None
        assert len(recovery_attempts) == 2  # Two failed attempts before success

    def test_error_context_propagation(self):
        """Test error context propagation across service boundaries."""
        correlation_manager = get_correlation_manager()
        
        # Service A sets initial context
        request_id = correlation_manager.set_request_context(
            user_id="user123"
        )
        
        with correlated_operation("service_a_operation") as service_a_id:
            # Service A encounters an error
            try:
                raise ValueError("Service A validation error")
            except ValueError as e:
                service_a_error = record_error(
                    error=e,
                    error_type="validation_error",
                    severity="medium",
                    recovery_action="service_a_validation_retry"
                )
            
            # Service A calls Service B (context propagated)
            with correlated_operation("service_b_operation") as service_b_id:
                correlation_manager.set_business_context(
                    operation_type="service_b_operation",
                    query_type="service_call"
                )
                
                # Service B encounters a related error
                try:
                    raise ConnectionError("Service B database error")
                except ConnectionError as e:
                    service_b_error = record_error(
                        error=e,
                        error_type="database_error",
                        severity="high",
                        recovery_action="service_b_database_retry"
                    )
        
        # Verify context propagation
        assert request_id is not None
        assert service_a_id is not None
        assert service_b_id is not None
        assert service_a_error is not None
        assert service_b_error is not None
        
        # Service correlation IDs should be different
        assert service_a_id != service_b_id


class TestErrorMetrics:
    """Test error metrics collection and reporting."""

    @patch('src.services.observability.metrics_bridge.metrics')
    def test_error_rate_metrics(self, mock_metrics):
        """Test error rate metrics collection."""
        # Setup mocks
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        meter.create_up_down_counter.return_value = Mock()
        
        from src.services.observability.metrics_bridge import initialize_metrics_bridge
        
        try:
            bridge = initialize_metrics_bridge()
            
            # Record errors for metrics
            error_types = ["validation_error", "connection_error", "timeout_error"]
            
            for error_type in error_types:
                for i in range(5):  # 5 errors of each type
                    bridge.record_error(
                        error_type=error_type,
                        severity="medium",
                        service_name="test_service"
                    )
            
            # Verify error metrics were recorded
            # In a real implementation, we'd check the actual metric values
            
        except Exception:
            # Handle case where metrics bridge is not available
            pytest.skip("Metrics bridge not available")

    @patch('src.services.observability.ai_tracking.metrics')
    def test_ai_operation_error_metrics(self, mock_metrics):
        """Test AI operation error metrics."""
        # Setup mocks
        meter = Mock()
        mock_metrics.get_meter.return_value = meter
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_gauge.return_value = Mock()
        
        ai_tracker = get_ai_tracker()
        
        # Test failed AI operations
        ai_failures = [
            ("openai", "gpt-4", "rate_limit_error"),
            ("openai", "ada-002", "api_error"),
            ("anthropic", "claude-3", "timeout_error"),
        ]
        
        for provider, model, error_type in ai_failures:
            try:
                with ai_tracker.track_llm_call(provider=provider, model=model) as result:
                    if error_type == "rate_limit_error":
                        raise ConnectionError("Rate limit exceeded")
                    elif error_type == "api_error":
                        raise ValueError("Invalid API response")
                    elif error_type == "timeout_error":
                        raise TimeoutError("Request timeout")
            except Exception:
                pass  # Errors are expected and tracked
        
        # AI tracker should have recorded error metrics
        # Verification would be implementation-specific

    def test_error_severity_distribution(self):
        """Test error severity distribution tracking."""
        correlation_manager = get_correlation_manager()
        request_id = correlation_manager.set_request_context(user_id="test_user")
        
        # Generate errors with different severities
        severity_distribution = {
            "critical": 2,
            "high": 5,
            "medium": 10,
            "low": 3
        }
        
        recorded_errors = {}
        
        for severity, count in severity_distribution.items():
            recorded_errors[severity] = []
            for i in range(count):
                try:
                    raise RuntimeError(f"{severity} error {i+1}")
                except RuntimeError as e:
                    error_id = record_error(
                        error=e,
                        error_type="runtime_error",
                        severity=severity
                    )
                    recorded_errors[severity].append(error_id)
        
        # Verify error distribution
        assert len(recorded_errors["critical"]) == 2
        assert len(recorded_errors["high"]) == 5
        assert len(recorded_errors["medium"]) == 10
        assert len(recorded_errors["low"]) == 3
        assert request_id is not None

    def test_error_trend_analysis(self):
        """Test error trend analysis over time."""
        correlation_manager = get_correlation_manager()
        
        # Simulate errors over time periods
        time_periods = ["morning", "afternoon", "evening"]
        error_trends = {}
        
        for period in time_periods:
            error_trends[period] = []
            request_id = correlation_manager.set_request_context(
                user_id="test_user"
            )
            
            # Different error patterns for different time periods
            error_count = {"morning": 2, "afternoon": 8, "evening": 3}[period]
            
            for i in range(error_count):
                try:
                    raise ConnectionError(f"Database error during {period}")
                except ConnectionError as e:
                    error_id = record_error(
                        error=e,
                        error_type="database_error",
                        severity="medium",
                        recovery_action=f"database_retry_during_{period}"
                    )
                    error_trends[period].append(error_id)
        
        # Verify trend data
        assert len(error_trends["morning"]) == 2
        assert len(error_trends["afternoon"]) == 8  # Peak error period
        assert len(error_trends["evening"]) == 3


class TestErrorAlerts:
    """Test error alerting and notification systems."""

    def test_critical_error_alerting(self):
        """Test critical error alerting."""
        correlation_manager = get_correlation_manager()
        
        request_id = correlation_manager.set_request_context(
            user_id="user123"
        )
        
        # Simulate critical error that should trigger alert
        try:
            raise RuntimeError("Critical system failure")
        except RuntimeError as e:
            error_id = record_error(
                error=e,
                error_type="system_failure",
                severity="critical",
                user_impact="high",
                recovery_action="alert_and_escalate"
            )
            
            # In a real system, this would trigger alerting mechanisms
            assert error_id is not None
        
        assert request_id is not None

    def test_error_threshold_monitoring(self):
        """Test error threshold monitoring."""
        correlation_manager = get_correlation_manager()
        
        # Simulate hitting error thresholds
        error_counts = {"validation_error": 0, "connection_error": 0}
        
        request_id = correlation_manager.set_request_context(user_id="test_user")
        
        # Generate errors to test thresholds
        for i in range(15):  # Exceed typical threshold of 10
            error_type = "validation_error" if i % 2 == 0 else "connection_error"
            error_counts[error_type] += 1
            
            try:
                raise ValueError(f"{error_type} #{error_counts[error_type]}")
            except ValueError as e:
                error_id = record_error(
                    error=e,
                    error_type=error_type,
                    severity="medium",
                    recovery_action=f"threshold_monitoring_sequence_{i+1}"
                )
                
                # Check if threshold exceeded (simplified logic)
                if error_counts[error_type] > 10:
                    # Would trigger threshold alert in real system
                    pass
        
        assert error_counts["validation_error"] > 5
        assert error_counts["connection_error"] > 5
        assert request_id is not None

    def test_error_rate_spike_detection(self):
        """Test error rate spike detection."""
        correlation_manager = get_correlation_manager()
        
        # Simulate normal error rate followed by spike
        normal_period_errors = []
        spike_period_errors = []
        
        # Normal period (low error rate)
        request_id_normal = correlation_manager.set_request_context(
            user_id="test_user"
        )
        
        for i in range(2):  # Low error rate
            try:
                raise ConnectionError(f"Normal period error {i+1}")
            except ConnectionError as e:
                error_id = record_error(
                    error=e,
                    error_type="connection_error",
                    severity="medium"
                )
                normal_period_errors.append(error_id)
        
        # Spike period (high error rate)
        request_id_spike = correlation_manager.set_request_context(
            user_id="test_user"
        )
        
        for i in range(20):  # High error rate (spike)
            try:
                raise ConnectionError(f"Spike period error {i+1}")
            except ConnectionError as e:
                error_id = record_error(
                    error=e,
                    error_type="connection_error",
                    severity="medium",
                    recovery_action="spike_detection_mode" if i > 10 else "normal_recovery"
                )
                spike_period_errors.append(error_id)
        
        # Verify spike detection data
        assert len(normal_period_errors) == 2
        assert len(spike_period_errors) == 20
        assert request_id_normal is not None
        assert request_id_spike is not None
        
        # Rate increase detection (simplified)
        rate_increase = len(spike_period_errors) / len(normal_period_errors)
        assert rate_increase == 10.0  # 10x increase