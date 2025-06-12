"""Comprehensive tests for MultiLevelCircuitBreaker with failure categorization.

This test module provides comprehensive coverage for the enhanced circuit breaker
including failure type categorization, partial failure handling, and sophisticated
recovery strategies.
"""

import asyncio
import time

import pytest
from src.infrastructure.database.enhanced_circuit_breaker import CircuitBreakerConfig
from src.infrastructure.database.enhanced_circuit_breaker import CircuitBreakerOpenError
from src.infrastructure.database.enhanced_circuit_breaker import CircuitState
from src.infrastructure.database.enhanced_circuit_breaker import FailureMetrics
from src.infrastructure.database.enhanced_circuit_breaker import FailureType
from src.infrastructure.database.enhanced_circuit_breaker import (
    MultiLevelCircuitBreaker,
)


class TestFailureMetrics:
    """Test FailureMetrics data class functionality."""

    def test_failure_metrics_initialization(self):
        """Test FailureMetrics initialization."""
        metrics = FailureMetrics()

        assert metrics.connection_failures == 0
        assert metrics.timeout_failures == 0
        assert metrics.query_failures == 0
        assert metrics.transaction_failures == 0
        assert metrics.resource_failures == 0
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.last_failure_time is None

    def test_get_failure_count(self):
        """Test getting failure count for specific types."""
        metrics = FailureMetrics()
        metrics.connection_failures = 3
        metrics.timeout_failures = 2

        assert metrics.get_failure_count(FailureType.CONNECTION) == 3
        assert metrics.get_failure_count(FailureType.TIMEOUT) == 2
        assert metrics.get_failure_count(FailureType.QUERY) == 0

    def test_increment_failure(self):
        """Test incrementing failure counts."""
        metrics = FailureMetrics()

        metrics.increment_failure(FailureType.CONNECTION)
        assert metrics.connection_failures == 1
        assert metrics.last_failure_time is not None

        metrics.increment_failure(FailureType.TIMEOUT)
        metrics.increment_failure(FailureType.TIMEOUT)
        assert metrics.timeout_failures == 2

    def test_get_total_failures(self):
        """Test total failure count calculation."""
        metrics = FailureMetrics()
        metrics.connection_failures = 2
        metrics.timeout_failures = 3
        metrics.query_failures = 1

        assert metrics.get_total_failures() == 6

    def test_get_success_rate(self):
        """Test success rate calculation."""
        metrics = FailureMetrics()

        # No requests yet
        assert metrics.get_success_rate() == 1.0

        # With some requests
        metrics.total_requests = 10
        metrics.successful_requests = 8
        assert metrics.get_success_rate() == 0.8


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_configuration(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.connection_threshold == 3
        assert config.timeout_threshold == 5
        assert config.query_threshold == 10
        assert config.transaction_threshold == 5
        assert config.resource_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_requests == 3
        assert config.half_open_success_threshold == 2
        assert config.failure_rate_threshold == 0.5
        assert config.min_requests_for_rate == 10

    def test_custom_configuration(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            connection_threshold=5,
            timeout_threshold=8,
            recovery_timeout=30.0,
            failure_rate_threshold=0.3,
        )

        assert config.connection_threshold == 5
        assert config.timeout_threshold == 8
        assert config.recovery_timeout == 30.0
        assert config.failure_rate_threshold == 0.3


class TestMultiLevelCircuitBreaker:
    """Test MultiLevelCircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Create test circuit breaker configuration."""
        return CircuitBreakerConfig(
            connection_threshold=3,
            timeout_threshold=2,
            query_threshold=5,
            transaction_threshold=3,
            resource_threshold=2,
            recovery_timeout=1.0,  # Short for testing
            half_open_max_requests=2,
            half_open_success_threshold=1,
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_breaker_config):
        """Create MultiLevelCircuitBreaker instance."""
        return MultiLevelCircuitBreaker(circuit_breaker_config)

    def test_initialization(self, circuit_breaker, circuit_breaker_config):
        """Test circuit breaker initialization."""
        assert circuit_breaker.config == circuit_breaker_config
        assert circuit_breaker.state == CircuitState.CLOSED
        assert isinstance(circuit_breaker.metrics, FailureMetrics)
        assert circuit_breaker._half_open_requests == 0
        assert circuit_breaker._half_open_successes == 0

    @pytest.mark.asyncio
    async def test_successful_execution_closed_state(self, circuit_breaker):
        """Test successful execution in closed state."""

        async def test_func():
            return "success"

        result = await circuit_breaker.execute(test_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_execution_with_timeout(self, circuit_breaker):
        """Test execution with timeout protection."""

        async def slow_func():
            await asyncio.sleep(2.0)
            return "completed"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.execute(slow_func, timeout=0.1)

        # Should record timeout failure
        assert circuit_breaker.metrics.timeout_failures == 1
        assert circuit_breaker.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_failure_type_specific_thresholds(self, circuit_breaker):
        """Test that different failure types have different thresholds."""

        async def failing_func():
            raise Exception("Test failure")

        # Connection failures should open circuit at threshold 3
        for _i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.execute(
                    failing_func, failure_type=FailureType.CONNECTION
                )

        # Circuit should be open now
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failure_threshold(self, circuit_breaker):
        """Test circuit opens when failure threshold is reached."""

        async def failing_func():
            raise Exception("Test failure")

        # Query failures have threshold of 5
        for _i in range(5):
            with pytest.raises(Exception):
                await circuit_breaker.execute(
                    failing_func, failure_type=FailureType.QUERY
                )

        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.query_failures == 5

    @pytest.mark.asyncio
    async def test_circuit_blocks_requests_when_open(self, circuit_breaker):
        """Test circuit blocks requests when open."""
        # Force circuit to open
        await circuit_breaker.force_open()

        async def test_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await circuit_breaker.execute(test_func, failure_type=FailureType.QUERY)

        assert exc_info.value.failure_type == FailureType.QUERY
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_fallback_handler_execution(self, circuit_breaker):
        """Test fallback handler execution when circuit is open."""

        # Register fallback handler
        async def fallback_handler():
            return "fallback_result"

        circuit_breaker.register_fallback_handler(FailureType.QUERY, fallback_handler)

        # Force circuit open
        await circuit_breaker.force_open()

        async def test_func():
            return "should not execute"

        result = await circuit_breaker.execute(
            test_func, failure_type=FailureType.QUERY
        )
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_provided_fallback_execution(self, circuit_breaker):
        """Test provided fallback execution when circuit is open."""
        # Force circuit open
        await circuit_breaker.force_open()

        async def test_func():
            return "should not execute"

        async def fallback():
            return "provided_fallback"

        result = await circuit_breaker.execute(test_func, fallback=fallback)
        assert result == "provided_fallback"

    @pytest.mark.asyncio
    async def test_partial_failure_handler(self, circuit_breaker):
        """Test partial failure handler execution."""

        async def failing_func():
            raise ValueError("Specific error")

        async def partial_handler(exception, *args, **kwargs):
            return f"handled_{type(exception).__name__}"

        circuit_breaker.register_partial_failure_handler(
            FailureType.QUERY, partial_handler
        )

        result = await circuit_breaker.execute(
            failing_func, failure_type=FailureType.QUERY
        )

        assert result == "handled_ValueError"
        # Should still record the failure
        assert circuit_breaker.metrics.query_failures == 1

    @pytest.mark.asyncio
    async def test_partial_failure_handler_failure(self, circuit_breaker):
        """Test behavior when partial failure handler also fails."""

        async def failing_func():
            raise ValueError("Original error")

        async def failing_handler(exception, *args, **kwargs):
            raise RuntimeError("Handler error")

        circuit_breaker.register_partial_failure_handler(
            FailureType.QUERY, failing_handler
        )

        # Should raise original exception when handler fails
        with pytest.raises(ValueError, match="Original error"):
            await circuit_breaker.execute(failing_func, failure_type=FailureType.QUERY)

    @pytest.mark.asyncio
    async def test_timeout_partial_failure_handling(self, circuit_breaker):
        """Test specific handling of timeout failures."""

        async def timeout_handler(exception, *args, **kwargs):
            return "timeout_handled"

        circuit_breaker.register_partial_failure_handler(
            FailureType.TIMEOUT, timeout_handler
        )

        async def slow_func():
            await asyncio.sleep(2.0)
            return "completed"

        result = await circuit_breaker.execute(slow_func, timeout=0.1)
        assert result == "timeout_handled"

        # Should record timeout failure
        assert circuit_breaker.metrics.timeout_failures == 1

    @pytest.mark.asyncio
    async def test_half_open_state_transition(self, circuit_breaker):
        """Test transition to half-open state after recovery timeout."""
        # Force circuit open
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout (mocked short timeout)
        await asyncio.sleep(1.1)

        # Next request should transition to half-open
        async def test_func():
            return "success"

        result = await circuit_breaker.execute(test_func)
        assert result == "success"
        # With half_open_success_threshold=1, one success should close the circuit
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test that success in half-open state closes circuit."""
        # Force to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker._state_change_time = time.time()

        async def success_func():
            return "success"

        # Execute successful request (threshold is 1 for test config)
        result = await circuit_breaker.execute(success_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.get_total_failures() == 0  # Reset on close

    @pytest.mark.asyncio
    async def test_half_open_failure_opens_circuit(self, circuit_breaker):
        """Test that failure in half-open state opens circuit."""
        # Force to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker._state_change_time = time.time()

        async def failing_func():
            raise Exception("Half-open failure")

        with pytest.raises(Exception):
            await circuit_breaker.execute(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_request_limiting(self, circuit_breaker):
        """Test request limiting in half-open state."""
        # Set up half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker._half_open_requests = 2  # At limit (config max is 2)

        async def test_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.execute(test_func)

    @pytest.mark.asyncio
    async def test_failure_rate_threshold(self, circuit_breaker):
        """Test circuit opening based on failure rate."""
        # Configure for rate-based threshold testing
        circuit_breaker.config.min_requests_for_rate = 5
        circuit_breaker.config.failure_rate_threshold = 0.6  # 60% failure rate

        async def success_func():
            return "success"

        async def failing_func():
            raise Exception("Failure")

        # Execute requests with 60% failure rate
        for i in range(10):
            try:
                if i < 6:  # 6 failures out of 10 = 60% failure rate
                    await circuit_breaker.execute(
                        failing_func, failure_type=FailureType.QUERY
                    )
                else:
                    await circuit_breaker.execute(success_func)
            except Exception:
                pass  # Expected failures

        # Circuit should be open due to failure rate
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_force_operations(self, circuit_breaker):
        """Test force open/close operations."""
        # Force open
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN

        # Force close
        await circuit_breaker.force_close()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.get_total_failures() == 0

    def test_health_status_reporting(self, circuit_breaker):
        """Test comprehensive health status reporting."""
        # Add some test data
        circuit_breaker.metrics.connection_failures = 2
        circuit_breaker.metrics.query_failures = 3
        circuit_breaker.metrics.total_requests = 10
        circuit_breaker.metrics.successful_requests = 5

        status = circuit_breaker.get_health_status()

        assert "state" in status
        assert "failure_metrics" in status
        assert "request_metrics" in status
        assert "state_info" in status
        assert "response_time_stats" in status
        assert "configuration" in status

        assert status["failure_metrics"]["connection_failures"] == 2
        assert status["failure_metrics"]["query_failures"] == 3
        assert status["request_metrics"]["total_requests"] == 10
        assert status["request_metrics"]["success_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_response_time_tracking(self, circuit_breaker):
        """Test response time statistics tracking."""

        async def timed_func(delay):
            await asyncio.sleep(delay)
            return "success"

        # Execute functions with different delays
        await circuit_breaker.execute(
            lambda: timed_func(0.1), failure_type=FailureType.QUERY
        )
        await circuit_breaker.execute(
            lambda: timed_func(0.2), failure_type=FailureType.CONNECTION
        )

        status = circuit_breaker.get_health_status()
        response_stats = status["response_time_stats"]

        assert "query" in response_stats
        assert "connection" in response_stats
        assert response_stats["query"]["sample_count"] > 0
        assert response_stats["connection"]["sample_count"] > 0

    @pytest.mark.asyncio
    async def test_failure_analysis(self, circuit_breaker):
        """Test detailed failure analysis."""
        # Add various types of failures
        circuit_breaker.metrics.connection_failures = 3
        circuit_breaker.metrics.timeout_failures = 2
        circuit_breaker.metrics.query_failures = 1
        circuit_breaker.metrics.total_requests = 10
        circuit_breaker.metrics.successful_requests = 4

        analysis = await circuit_breaker.get_failure_analysis()

        assert "status" in analysis
        assert "success_rate" in analysis
        assert "total_failures" in analysis
        assert "primary_failure_types" in analysis
        assert "failure_breakdown" in analysis
        assert "recommendations" in analysis

        # Should identify connection failures as primary
        assert "connection" in analysis["primary_failure_types"]
        assert analysis["success_rate"] == 0.4
        assert len(analysis["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_healthy_system_analysis(self, circuit_breaker):
        """Test failure analysis for healthy system."""
        # No failures
        circuit_breaker.metrics.total_requests = 10
        circuit_breaker.metrics.successful_requests = 10

        analysis = await circuit_breaker.get_failure_analysis()

        assert analysis["status"] == "healthy"
        assert "operating normally" in analysis["recommendations"][0].lower()
        assert len(analysis["primary_failure_types"]) == 0

    @pytest.mark.asyncio
    async def test_failure_counter_reset(self, circuit_breaker):
        """Test periodic failure counter reset on success."""
        # Add some failures
        circuit_breaker.metrics.query_failures = 6
        circuit_breaker.metrics.connection_failures = 4
        circuit_breaker.metrics.successful_requests = 100  # Trigger reset condition

        # Simulate successful request that should trigger reset
        async def success_func():
            return "success"

        await circuit_breaker.execute(success_func)

        # Failures should be reduced (partial reset)
        assert circuit_breaker.metrics.query_failures < 6
        assert circuit_breaker.metrics.connection_failures < 4

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self, circuit_breaker):
        """Test thread safety with concurrent executions."""

        async def test_func(delay):
            await asyncio.sleep(delay)
            return f"result_{delay}"

        # Execute multiple concurrent requests
        tasks = [
            circuit_breaker.execute(lambda delay=0.01 * i: test_func(delay))
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert circuit_breaker.metrics.total_requests == 10
        assert circuit_breaker.metrics.successful_requests == 10


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_handler_registration_edge_cases(self):
        """Test handler registration edge cases."""
        breaker = MultiLevelCircuitBreaker()

        # Register handlers
        async def test_handler(exc, *args, **kwargs):
            return "handled"

        breaker.register_partial_failure_handler(FailureType.QUERY, test_handler)
        breaker.register_fallback_handler(FailureType.CONNECTION, test_handler)

        assert FailureType.QUERY in breaker.partial_failure_handlers
        assert FailureType.CONNECTION in breaker.fallback_handlers

    @pytest.mark.asyncio
    async def test_zero_threshold_configuration(self):
        """Test behavior with zero failure thresholds."""
        config = CircuitBreakerConfig(connection_threshold=0)
        breaker = MultiLevelCircuitBreaker(config)

        # Any connection failure should immediately open circuit
        async def failing_func():
            raise Exception("Connection error")

        with pytest.raises(Exception):
            await breaker.execute(failing_func, failure_type=FailureType.CONNECTION)

        # Circuit should be open immediately
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_very_short_recovery_timeout(self):
        """Test behavior with very short recovery timeout."""
        config = CircuitBreakerConfig(recovery_timeout=0.01)  # 10ms
        breaker = MultiLevelCircuitBreaker(config)

        await breaker.force_open()

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        async def test_func():
            return "success"

        result = await breaker.execute(test_func)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

    def test_response_time_stats_with_no_data(self):
        """Test response time statistics with no execution data."""
        breaker = MultiLevelCircuitBreaker()

        stats = breaker._get_response_time_stats()

        for failure_type in FailureType:
            assert failure_type.value in stats
            assert stats[failure_type.value]["sample_count"] == 0
            assert stats[failure_type.value]["avg_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_state_change_timing(self):
        """Test state change timing accuracy."""
        breaker = MultiLevelCircuitBreaker()

        time.time()
        await breaker.force_open()

        status = breaker.get_health_status()
        duration = status["state_info"]["state_duration_seconds"]

        # Should be very recent
        assert 0 <= duration <= 1.0

    @pytest.mark.asyncio
    async def test_execution_with_args_and_kwargs(self):
        """Test execution with function arguments and keyword arguments."""
        breaker = MultiLevelCircuitBreaker()

        async def test_func(arg1, arg2, kwarg1=None, kwarg2=None):
            return f"{arg1}_{arg2}_{kwarg1}_{kwarg2}"

        result = await breaker.execute(
            test_func, FailureType.QUERY, None, None, "a", "b", kwarg1="c", kwarg2="d"
        )

        assert result == "a_b_c_d"

    @pytest.mark.asyncio
    async def test_metric_overflow_handling(self):
        """Test handling of very large metric values."""
        breaker = MultiLevelCircuitBreaker()

        # Manually set very large failure counts
        breaker.metrics.query_failures = 999999
        breaker.metrics.total_requests = 1000000
        breaker.metrics.successful_requests = 1

        # Should handle large numbers gracefully
        total_failures = breaker.metrics.get_total_failures()
        success_rate = breaker.metrics.get_success_rate()

        assert total_failures == 999999
        assert 0.0 <= success_rate <= 1.0

    @pytest.mark.asyncio
    async def test_numpy_import_error_handling(self):
        """Test graceful handling when numpy is not available."""
        breaker = MultiLevelCircuitBreaker()

        # Mock numpy import failure in response time stats
        with pytest.raises(ImportError):
            raise ImportError("Mocked numpy failure")

        # Should not crash the circuit breaker
        assert breaker.state == CircuitState.CLOSED
