"""Comprehensive tests for infrastructure shared module to improve coverage."""

import asyncio
import time

import pytest

from src.infrastructure.shared import CircuitBreaker, ClientHealth, ClientState
from src.services.errors import APIError


class TestClientState:
    """Test ClientState enumeration."""

    def test_client_state_values(self):
        """Test ClientState enum values."""
        assert ClientState.UNINITIALIZED.value == "uninitialized"
        assert ClientState.HEALTHY.value == "healthy"
        assert ClientState.DEGRADED.value == "degraded"
        assert ClientState.FAILED.value == "failed"

    def test_client_state_comparison(self):
        """Test ClientState enum comparison."""
        assert ClientState.UNINITIALIZED != ClientState.HEALTHY
        assert ClientState.HEALTHY == ClientState.HEALTHY
        assert ClientState.DEGRADED != ClientState.FAILED

    def test_client_state_string_representation(self):
        """Test ClientState string representation."""
        assert str(ClientState.HEALTHY) == "ClientState.HEALTHY"
        assert repr(ClientState.FAILED) == "<ClientState.FAILED: 'failed'>"


class TestClientHealth:
    """Test ClientHealth dataclass."""

    def test_client_health_initialization(self):
        """Test ClientHealth initialization with default values."""
        health = ClientHealth()

        assert health.state == ClientState.UNINITIALIZED
        assert health.last_success_time is None
        assert health.last_failure_time is None
        assert health.consecutive_failures == 0
        assert health._total_requests == 0
        assert health._total_failures == 0
        assert isinstance(health.metadata, dict)
        assert len(health.metadata) == 0

    def test_client_health_custom_initialization(self):
        """Test ClientHealth initialization with custom values."""
        metadata = {"test": "value", "key": 123}
        health = ClientHealth(
            state=ClientState.HEALTHY,
            consecutive_failures=5,
            _total_requests=100,
            _total_failures=10,
            metadata=metadata,
        )

        assert health.state == ClientState.HEALTHY
        assert health.consecutive_failures == 5
        assert health._total_requests == 100
        assert health._total_failures == 10
        assert health.metadata == metadata

    def test_client_health_with_timestamps(self):
        """Test ClientHealth with timestamp values."""
        current_time = time.time()
        health = ClientHealth(
            state=ClientState.DEGRADED,
            last_success_time=current_time - 100,
            last_failure_time=current_time - 50,
        )

        assert health.state == ClientState.DEGRADED
        assert health.last_success_time == current_time - 100
        assert health.last_failure_time == current_time - 50

    def test_client_health_metadata_manipulation(self):
        """Test ClientHealth metadata manipulation."""
        health = ClientHealth()

        # Add metadata
        health.metadata["connection_id"] = "conn_123"
        health.metadata["retry_count"] = 3

        assert health.metadata["connection_id"] == "conn_123"
        assert health.metadata["retry_count"] == 3

        # Update metadata
        health.metadata.update({"status": "active", "retry_count": 5})

        assert health.metadata["status"] == "active"
        assert health.metadata["retry_count"] == 5
        assert len(health.metadata) == 3


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a CircuitBreaker instance for testing."""
        return CircuitBreaker(
            failure_threshold=3, timeout_seconds=60.0, recovery_timeout=30.0
        )

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.timeout_seconds == 60.0
        assert circuit_breaker.recovery_timeout == 30.0
        assert circuit_breaker.state == ClientState.HEALTHY
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._last_failure_time is None

    def test_circuit_breaker_default_initialization(self):
        """Test CircuitBreaker with default values."""
        cb = CircuitBreaker()

        assert cb.failure_threshold == 5
        assert cb.timeout_seconds == 60.0
        assert cb.recovery_timeout == 30.0
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""

        async def test_function():
            return "success"

        result = await circuit_breaker.call(test_function)

        assert result == "success"
        assert circuit_breaker.state == ClientState.HEALTHY
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_single_failure(self, circuit_breaker):
        """Test single failure doesn't open circuit."""

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        with pytest.raises(APIError):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == ClientState.HEALTHY  # Still healthy
        assert circuit_breaker._failure_count == 1
        assert circuit_breaker._last_failure_time is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_multiple_failures_opens_circuit(
        self, circuit_breaker
    ):
        """Test multiple failures open the circuit."""

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        # Fail multiple times to open circuit
        for _ in range(3):
            with pytest.raises(APIError):
                await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == ClientState.FAILED
        assert circuit_breaker._failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_circuit_raises_immediately(
        self, circuit_breaker
    ):
        """Test that open circuit raises without calling function."""

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        async def success_function():
            return "should not be called"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(APIError):
                await circuit_breaker.call(failing_function)

        # Now circuit is open, should raise immediately
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await circuit_breaker.call(success_function)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_after_timeout(self, circuit_breaker):
        """Test circuit recovery after timeout."""

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        async def success_function():
            return "recovered"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(APIError):
                await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == ClientState.FAILED

        # Simulate timeout passage
        circuit_breaker._last_failure_time = time.time() - 61  # Beyond timeout

        # Should allow one call in half-open state
        result = await circuit_breaker.call(success_function)

        assert result == "recovered"
        assert circuit_breaker.state == ClientState.HEALTHY
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopens(self, circuit_breaker):
        """Test that failure in half-open state reopens circuit."""

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        # Open the circuit
        for _ in range(3):
            with pytest.raises(APIError):
                await circuit_breaker.call(failing_function)

        # Simulate timeout passage to enter half-open state
        circuit_breaker._last_failure_time = time.time() - 61

        # Fail again in half-open state
        with pytest.raises(APIError):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == ClientState.FAILED

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_sync_function(self, circuit_breaker):
        """Test circuit breaker with synchronous function."""

        def sync_function():
            return "sync_success"

        result = await circuit_breaker.call(sync_function)

        assert result == "sync_success"
        assert circuit_breaker.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_sync_failing_function(self, circuit_breaker):
        """Test circuit breaker with synchronous failing function."""

        def sync_failing_function():
            msg = "Sync error"
            raise APIError(msg)

        with pytest.raises(APIError):
            await circuit_breaker.call(sync_failing_function)

        assert circuit_breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_function_with_args__kwargs(self, circuit_breaker):
        """Test circuit breaker with function arguments."""

        async def function_with_args(a, b, c=None):
            return f"args: {a}, {b}, {c}"

        result = await circuit_breaker.call(function_with_args, 1, 2, c="test")

        assert result == "args: 1, 2, test"

    def test_circuit_breaker_state_transitions(self, circuit_breaker):
        """Test circuit breaker state transitions."""
        # Initial state
        assert circuit_breaker.state == ClientState.HEALTHY

        # Simulate failures
        circuit_breaker._failure_count = 2
        assert circuit_breaker.state == ClientState.HEALTHY  # Still under threshold

        circuit_breaker._failure_count = 3
        circuit_breaker._last_failure_time = time.time()
        assert circuit_breaker.state == ClientState.FAILED  # Over threshold

        # Simulate timeout recovery
        circuit_breaker._last_failure_time = time.time() - 61
        # The state would be checked when calling, but we can test the internal logic

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_calls(self, circuit_breaker):
        """Test circuit breaker with concurrent calls."""
        call_count = 0

        async def counting_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay
            return call_count

        # Run multiple concurrent calls
        tasks = [circuit_breaker.call(counting_function) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All calls should succeed
        assert len(results) == 5
        assert all(isinstance(r, int) for r in results)
        assert circuit_breaker.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_handling_edge_cases(self, circuit_breaker):
        """Test circuit breaker error handling edge cases."""

        # Test with non-APIError exception
        async def function_with_other_error():
            msg = "Not an API error"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            await circuit_breaker.call(function_with_other_error)

        # Circuit breaker should still increment failure count
        assert circuit_breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_edge_cases(self):
        """Test circuit breaker timeout edge cases."""
        # Very short timeout
        cb_short = CircuitBreaker(failure_threshold=1, timeout_seconds=0.1)

        async def failing_function():
            msg = "Test error"
            raise APIError(msg)

        # Fail once to open circuit
        with pytest.raises(APIError):
            await cb_short.call(failing_function)

        assert cb_short.state == ClientState.FAILED

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should be able to call again
        async def success_function():
            return "recovered"

        result = await cb_short.call(success_function)
        assert result == "recovered"
        assert cb_short.state == ClientState.HEALTHY


class TestCircuitBreakerIntegration:
    """Test CircuitBreaker integration scenarios."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_real_api_simulation(self):
        """Test circuit breaker with realistic API call simulation."""
        api_failure_count = 0

        async def simulated_api_call():
            nonlocal api_failure_count
            api_failure_count += 1

            if api_failure_count <= 3:
                msg = f"API failure #{api_failure_count}"
                raise APIError(msg)
            return f"API success after {api_failure_count} calls"

        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)

        # First two calls should fail and open circuit
        with pytest.raises(APIError):
            await cb.call(simulated_api_call)

        with pytest.raises(APIError):
            await cb.call(simulated_api_call)

        assert cb.state == ClientState.FAILED

        # Circuit should be open now
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await cb.call(simulated_api_call)

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should succeed after recovery
        result = await cb.call(simulated_api_call)
        assert "API success" in result
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_on_success(self):
        """Test that circuit breaker resets failure count on success."""
        cb = CircuitBreaker(failure_threshold=3)

        async def sometimes_failing_function(should_fail=True):
            if should_fail:
                msg = "Temporary failure"
                raise APIError(msg)
            return "success"

        # Fail twice (under threshold)
        for _ in range(2):
            with pytest.raises(APIError):
                await cb.call(sometimes_failing_function, True)

        assert cb._failure_count == 2
        assert cb.state == ClientState.HEALTHY

        # Succeed once
        result = await cb.call(sometimes_failing_function, False)
        assert result == "success"
        assert cb._failure_count == 0  # Should reset on success
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_different_error_types(self):
        """Test circuit breaker behavior with different error types."""
        cb = CircuitBreaker(failure_threshold=2)

        async def function_with_api_error():
            msg = "API error"
            raise APIError(msg)

        async def function_with_value_error():
            msg = "Value error"
            raise ValueError(msg)

        async def function_with_runtime_error():
            msg = "Runtime error"
            raise RuntimeError(msg)

        # All error types should increment failure count
        with pytest.raises(APIError):
            await cb.call(function_with_api_error)

        assert cb._failure_count == 1

        with pytest.raises(ValueError):
            await cb.call(function_with_value_error)

        assert cb._failure_count == 2
        assert cb.state == ClientState.FAILED  # Circuit should be open

        # Circuit should block further calls
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await cb.call(function_with_runtime_error)


class TestCircuitBreakerPerformance:
    """Test CircuitBreaker performance characteristics."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self):
        """Test circuit breaker call overhead."""
        cb = CircuitBreaker()

        async def fast_function():
            return "fast"

        # Time multiple calls to measure overhead
        start_time = time.perf_counter()

        for _ in range(100):
            result = await cb.call(fast_function)
            assert result == "fast"

        end_time = time.perf_counter()
        _total_time = end_time - start_time

        # Should complete 100 calls reasonably quickly
        assert _total_time < 1.0  # Less than 1 second for 100 calls
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_memory_usage(self):
        """Test circuit breaker doesn't accumulate memory."""
        cb = CircuitBreaker()

        async def memory_test_function():
            # Create and return some data
            return list(range(1000))

        # Run many calls
        for _ in range(50):
            result = await cb.call(memory_test_function)
            assert len(result) == 1000

        # Circuit breaker state should remain clean
        assert cb.state == ClientState.HEALTHY
        assert cb._failure_count == 0


class TestCircuitBreakerEdgeCases:
    """Test CircuitBreaker edge cases and error conditions."""

    def test_circuit_breaker_invalid_parameters(self):
        """Test circuit breaker with invalid parameters."""
        # Invalid failure threshold
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=-1)

        # Invalid timeout
        with pytest.raises(ValueError):
            CircuitBreaker(timeout_seconds=0)

        with pytest.raises(ValueError):
            CircuitBreaker(timeout_seconds=-1)

        # Invalid recovery timeout
        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=0)

        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=-1)

    @pytest.mark.asyncio
    async def test_circuit_breaker_none_function(self):
        """Test circuit breaker with None function."""
        cb = CircuitBreaker()

        with pytest.raises((TypeError, AttributeError)):
            await cb.call(None)

    @pytest.mark.asyncio
    async def test_circuit_breaker_function_that_returns_none(self):
        """Test circuit breaker with function that returns None."""
        cb = CircuitBreaker()

        async def function_returning_none():
            return None

        result = await cb.call(function_returning_none)
        assert result is None
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_function_with_complex_return(self):
        """Test circuit breaker with function returning complex data."""
        cb = CircuitBreaker()

        async def function_returning_complex():
            return {
                "data": [1, 2, 3],
                "metadata": {"count": 3, "status": "ok"},
                "nested": {"deep": {"value": "test"}},
            }

        result = await cb.call(function_returning_complex)

        assert isinstance(result, dict)
        assert result["data"] == [1, 2, 3]
        assert result["metadata"]["status"] == "ok"
        assert result["nested"]["deep"]["value"] == "test"
        assert cb.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_extremely_high_threshold(self):
        """Test circuit breaker with very high failure threshold."""
        cb = CircuitBreaker(failure_threshold=1000)

        async def failing_function():
            msg = "Always fails"
            raise APIError(msg)

        # Even many failures shouldn't open circuit
        for _ in range(50):
            with pytest.raises(APIError):
                await cb.call(failing_function)

        assert cb.state == ClientState.HEALTHY
        assert cb._failure_count == 50
