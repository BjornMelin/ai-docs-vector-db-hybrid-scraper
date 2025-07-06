"""Tests for circuit breaker implementation."""

import asyncio
import contextlib
import threading

import pytest

from src.services.functional.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
    circuit_breaker,
    create_circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_simple_mode_config(self):
        """Test simple mode configuration."""
        config = CircuitBreakerConfig.simple_mode()

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30
        assert config.enable_metrics is False
        assert config.enable_fallback is False
        assert config.enable_adaptive_timeout is False

    def test_enterprise_mode_config(self):
        """Test enterprise mode configuration."""
        config = CircuitBreakerConfig.enterprise_mode()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.enable_metrics is True
        assert config.enable_fallback is True
        assert config.enable_adaptive_timeout is True
        assert config.failure_rate_threshold == 0.4


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def simple_breaker(self):
        """Create a simple circuit breaker."""
        config = CircuitBreakerConfig.simple_mode()
        return CircuitBreaker(config)

    @pytest.fixture
    def enterprise_breaker(self):
        """Create an enterprise circuit breaker."""
        config = CircuitBreakerConfig.enterprise_mode()
        return CircuitBreaker(config)

    @pytest.mark.asyncio
    async def test_closed_state_success(self, simple_breaker):
        """Test successful operation in closed state."""

        async def success_func():
            return "success"

        result = await simple_breaker.call(success_func)
        assert result == "success"
        assert simple_breaker.state == CircuitBreakerState.CLOSED
        assert simple_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_closed_state_single_failure(self, simple_breaker):
        """Test single failure in closed state."""

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            await simple_breaker.call(failing_func)

        assert simple_breaker.state == CircuitBreakerState.CLOSED
        assert simple_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, simple_breaker):
        """Test circuit opens after failure threshold."""

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        # Fail until threshold is reached
        for _i in range(simple_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await simple_breaker.call(failing_func)

        # Circuit should now be open
        assert simple_breaker.state == CircuitBreakerState.OPEN

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await simple_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_recovery_after_timeout(self, simple_breaker):
        """Test circuit recovery after timeout."""

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        async def success_func():
            return "recovered"

        # Force circuit to open
        for _i in range(simple_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await simple_breaker.call(failing_func)

        assert simple_breaker.state == CircuitBreakerState.OPEN

        # Simulate timeout by manually adjusting last_failure_time
        simple_breaker.last_failure_time = 0

        # Should transition to half-open and then closed on success
        result = await simple_breaker.call(success_func)
        assert result == "recovered"
        assert simple_breaker.state == CircuitBreakerState.CLOSED
        assert simple_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, simple_breaker):
        """Test half-open transitions back to open on failure."""

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        # Force circuit to open
        for _i in range(simple_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await simple_breaker.call(failing_func)

        # Simulate timeout
        simple_breaker.last_failure_time = 0

        # First call should transition to half-open, then back to open on failure
        with pytest.raises(ValueError):
            await simple_breaker.call(failing_func)

        assert simple_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_enterprise_mode_failure_rate(self, enterprise_breaker):
        """Test enterprise mode failure rate monitoring."""

        async def success_func():
            return "success"

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        # Generate enough requests to trigger rate-based opening
        for i in range(15):
            try:
                if i < 8:  # 8 failures out of 15 requests = 53% failure rate
                    await enterprise_breaker.call(failing_func)
                else:
                    await enterprise_breaker.call(success_func)
            except (ValueError, CircuitBreakerError):
                # Circuit may open partway through due to failure rate
                pass

        # Should open due to high failure rate (53% > 40% threshold)
        assert enterprise_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, enterprise_breaker):
        """Test metrics tracking in enterprise mode."""

        async def success_func():
            return "success"

        async def failing_func():
            msg = "Test error"
            raise ValueError(msg)

        # Execute some operations
        await enterprise_breaker.call(success_func)
        with contextlib.suppress(ValueError):
            await enterprise_breaker.call(failing_func)

        metrics = enterprise_breaker.get_metrics()
        assert metrics["_total_requests"] == 2
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert metrics["failure_rate"] == 0.5
        assert metrics["success_rate"] == 0.5


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        config = CircuitBreakerConfig.simple_mode()

        @circuit_breaker(config)
        async def decorated_func(value):
            return f"result: {value}"

        result = await decorated_func("test")
        assert result == "result: test"

    @pytest.mark.asyncio
    async def test_decorator_failure_and_recovery(self):
        """Test decorator with failure and recovery."""
        config = CircuitBreakerConfig.simple_mode()

        call_count = 0

        @circuit_breaker(config)
        async def sometimes_failing_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                msg = "Failing"
                raise ValueError(msg)
            return "success"

        # First 3 calls should fail and open circuit
        for _i in range(3):
            with pytest.raises(ValueError):
                await sometimes_failing_func()

        # Circuit should be open, next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await sometimes_failing_func()

        # Access circuit breaker for testing
        breaker = sometimes_failing_func._circuit_breaker
        breaker.last_failure_time = 0  # Simulate timeout

        # Should recover on next call
        result = await sometimes_failing_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_metrics_access(self):
        """Test access to circuit breaker metrics through decorator."""
        config = CircuitBreakerConfig.enterprise_mode()

        @circuit_breaker(config)
        @pytest.mark.asyncio
        async def test_func():
            return "test"

        await test_func()

        # Access metrics through attached circuit breaker
        breaker = test_func._circuit_breaker
        metrics = breaker.get_metrics()
        assert metrics["_total_requests"] == 1
        assert metrics["successful_requests"] == 1


class TestCircuitBreakerFactory:
    """Test circuit breaker factory function."""

    def test_create_simple_breaker(self):
        """Test creating simple circuit breaker."""
        breaker = create_circuit_breaker("simple")
        assert breaker.config.failure_threshold == 3
        assert breaker.config.enable_metrics is False

    def test_create_enterprise_breaker(self):
        """Test creating enterprise circuit breaker."""
        breaker = create_circuit_breaker("enterprise")
        assert breaker.config.failure_threshold == 5
        assert breaker.config.enable_metrics is True

    def test_create_with_overrides(self):
        """Test creating circuit breaker with config overrides."""
        breaker = create_circuit_breaker(
            "simple",
            failure_threshold=10,
            recovery_timeout=120,
        )
        assert breaker.config.failure_threshold == 10
        assert breaker.config.recovery_timeout == 120


class TestConcurrentAccess:
    """Test circuit breaker under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test circuit breaker with concurrent operations."""
        config = CircuitBreakerConfig.simple_mode()
        breaker = CircuitBreaker(config)

        # Use thread-safe counter for concurrent operations
        counter_lock = threading.Lock()
        global_counter = 0

        @pytest.mark.asyncio
        async def test_func():
            nonlocal global_counter
            # Use thread-safe counter to ensure unique IDs
            with counter_lock:
                unique_id = global_counter
                global_counter += 1

            await asyncio.sleep(0.01)  # Simulate async work
            return f"result_{unique_id}"

        # Run multiple concurrent operations
        tasks = [breaker.call(test_func) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert len(set(results)) == 10  # All results should be unique
        assert breaker.metrics._total_requests == 10
        assert breaker.metrics.successful_requests == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Test circuit breaker with concurrent failures."""
        config = CircuitBreakerConfig.simple_mode()
        breaker = CircuitBreaker(config)

        async def failing_func():
            await asyncio.sleep(0.01)
            msg = "Concurrent failure"
            raise ValueError(msg)

        # Run concurrent failing operations
        tasks = [breaker.call(failing_func) for _ in range(5)]

        with pytest.raises(ValueError):
            await asyncio.gather(*tasks, return_exceptions=False)

        # Circuit should be open after failures
        assert breaker.state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_real_world_scenario():
    """Test a real-world scenario with varying load and failures."""
    config = CircuitBreakerConfig.enterprise_mode()
    breaker = CircuitBreaker(config)

    failure_count = 0

    async def simulated_service():
        nonlocal failure_count
        await asyncio.sleep(0.001)  # Simulate network delay

        # Simulate 50% failure rate to exceed 40% threshold
        failure_count += 1
        if failure_count % 2 == 0:
            msg = "Service unavailable"
            raise ConnectionError(msg)
        return "service_response"

    results = []
    errors = []

    # Simulate 50 requests
    for _i in range(50):
        try:
            result = await breaker.call(simulated_service)
            results.append(result)
        except (ConnectionError, CircuitBreakerError) as e:
            errors.append(type(e).__name__)

    metrics = breaker.get_metrics()

    # Verify that circuit breaker protected the system
    assert len(results) > 0  # Some requests succeeded
    assert len(errors) > 0  # Some requests failed
    assert metrics["_total_requests"] <= 50  # Circuit may have opened early
    assert "CircuitBreakerError" in errors  # Circuit breaker activated
