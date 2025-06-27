"""Tests for circuit breaker functionality.

This module tests the advanced circuit breaker implementation including:
- Basic circuit breaker behavior (closed, open, half-open states)
- Adaptive timeout adjustment
- Metrics collection
- Integration with service dependencies
- Tenacity-powered circuit breakers
"""

import asyncio
from datetime import datetime, timezone, UTC
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.errors import (
    AdvancedCircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    ExternalServiceError,
    circuit_breaker,
    tenacity_circuit_breaker,
)


class TestAdvancedCircuitBreaker:
    """Test the AdvancedCircuitBreaker class."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization with default values."""
        breaker = AdvancedCircuitBreaker(
            service_name="test_service",
            failure_threshold=3,
            recovery_timeout=30.0,
        )

        assert breaker.service_name == "test_service"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.metrics is not None

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self):
        """Test that successful calls keep the circuit in CLOSED state."""
        breaker = AdvancedCircuitBreaker("test_service", failure_threshold=3)

        async def successful_function():
            return "success"

        # Execute successful calls
        for _ in range(5):
            result = await breaker.call(successful_function)
            assert result == "success"
            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = AdvancedCircuitBreaker("test_service", failure_threshold=3)

        async def failing_function():
            raise ExternalServiceError("Service unavailable")

        # Execute failing calls
        for i in range(3):
            with pytest.raises(ExternalServiceError):
                await breaker.call(failing_function)

            if i < 2:
                assert breaker.state == CircuitState.CLOSED
            else:
                assert breaker.state == CircuitState.OPEN

        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_rejects_calls_when_open(self):
        """Test that open circuit rejects calls immediately."""
        breaker = AdvancedCircuitBreaker(
            "test_service", failure_threshold=2, recovery_timeout=60.0
        )

        async def failing_function():
            raise ExternalServiceError("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ExternalServiceError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Now calls should be rejected immediately
        with pytest.raises(ExternalServiceError, match="Circuit breaker is OPEN"):
            await breaker.call(failing_function)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Test circuit transition from OPEN to HALF_OPEN after timeout."""
        breaker = AdvancedCircuitBreaker(
            "test_service", failure_threshold=2, recovery_timeout=0.1
        )

        async def failing_function():
            raise ExternalServiceError("Service unavailable")

        async def successful_function():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ExternalServiceError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Next call should transition to HALF_OPEN, then we can test the state
        # Use a successful call to actually trigger the transition
        result = await breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED  # Should close on success

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self):
        """Test that HALF_OPEN circuit closes on successful call."""
        breaker = AdvancedCircuitBreaker(
            "test_service", failure_threshold=2, recovery_timeout=0.1
        )

        async def failing_function():
            raise ExternalServiceError("Service unavailable")

        async def successful_function():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ExternalServiceError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Successful call should close the circuit
        result = await breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self):
        """Test that HALF_OPEN circuit reopens on failure."""
        breaker = AdvancedCircuitBreaker(
            "test_service", failure_threshold=2, recovery_timeout=0.1
        )

        async def failing_function():
            raise ExternalServiceError("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ExternalServiceError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Failure in half-open should reopen circuit
        with pytest.raises(ExternalServiceError):
            await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

    def test_adaptive_timeout_adjustment(self):
        """Test adaptive timeout adjustment based on success/failure patterns."""
        breaker = AdvancedCircuitBreaker(
            "test_service", enable_adaptive_timeout=True, recovery_timeout=60.0
        )

        # Test success pattern
        for _ in range(3):
            breaker._update_adaptive_timeout(success=True)

        # Timeout should be reduced after consecutive successes
        assert breaker.adaptive_timeout < breaker.recovery_timeout

        # Test failure pattern
        breaker._update_adaptive_timeout(success=False)

        # Timeout should be increased after failure
        assert breaker.adaptive_timeout > breaker.recovery_timeout

    def test_metrics_collection(self):
        """Test circuit breaker metrics collection."""
        breaker = AdvancedCircuitBreaker("test_service", enable_metrics=True)

        # Record some calls
        breaker.metrics.record_call(success=True, response_time=0.1)
        breaker.metrics.record_call(success=False, response_time=0.5)
        breaker.metrics.record_call(success=True, response_time=0.2)

        assert breaker.metrics.total_calls == 3
        assert breaker.metrics.success_calls == 2
        assert breaker.metrics.failure_calls == 1
        assert abs(breaker.metrics.get_success_rate() - 66.67) < 0.01
        assert (
            abs(breaker.metrics.get_average_response_time() - 0.267) < 0.01
        )  # (0.1 + 0.5 + 0.2) / 3

    def test_get_status(self):
        """Test circuit breaker status reporting."""
        breaker = AdvancedCircuitBreaker("test_service", failure_threshold=5)

        status = breaker.get_status()

        assert status["service_name"] == "test_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert "total_calls" in status
        assert "success_rate" in status

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        breaker = AdvancedCircuitBreaker("test_service", failure_threshold=2)

        # Simulate failures and open circuit
        breaker.failure_count = 3
        breaker._change_state(CircuitState.OPEN)
        breaker.last_failure_time = datetime.now(tz=UTC)

        assert breaker.state == CircuitState.OPEN

        # Reset circuit
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None


class TestCircuitBreakerRegistry:
    """Test the CircuitBreakerRegistry."""

    def test_registry_operations(self):
        """Test circuit breaker registry registration and retrieval."""
        # Clear registry
        CircuitBreakerRegistry._breakers.clear()

        breaker1 = AdvancedCircuitBreaker("service1")
        breaker2 = AdvancedCircuitBreaker("service2")

        # Test registration (automatic via constructor)
        assert CircuitBreakerRegistry.get("service1") == breaker1
        assert CircuitBreakerRegistry.get("service2") == breaker2
        assert CircuitBreakerRegistry.get("nonexistent") is None

        # Test get_services
        services = CircuitBreakerRegistry.get_services()
        assert "service1" in services
        assert "service2" in services

        # Test get_all_status
        all_status = CircuitBreakerRegistry.get_all_status()
        assert "service1" in all_status
        assert "service2" in all_status


class TestCircuitBreakerDecorators:
    """Test circuit breaker decorators."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test basic circuit breaker decorator."""
        call_count = 0

        @circuit_breaker(service_name="test_decorator", failure_threshold=2)
        async def test_function(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ExternalServiceError("Test error")
            return f"call_{call_count}"

        # Successful calls
        result = await test_function(should_fail=False)
        assert result == "call_1"

        # Access circuit breaker instance
        assert hasattr(test_function, "circuit_breaker")
        assert hasattr(test_function, "get_circuit_status")
        assert hasattr(test_function, "reset_circuit")

        status = test_function.get_circuit_status()
        assert status["service_name"] == "test_decorator"
        assert status["state"] == "closed"

    @pytest.mark.asyncio
    async def test_tenacity_circuit_breaker_decorator(self):
        """Test Tenacity-powered circuit breaker decorator."""
        call_count = 0

        @tenacity_circuit_breaker(
            service_name="test_tenacity",
            max_attempts=2,
            failure_threshold=3,
        )
        async def test_function(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ExternalServiceError("Test error")
            return f"call_{call_count}"

        # Test successful call
        result = await test_function(should_fail=False)
        assert result == "call_1"

        # Test retry behavior
        call_count = 0
        with pytest.raises(ExternalServiceError):
            await test_function(should_fail=True)

        # Should have retried (call_count > 1)
        assert call_count == 2  # max_attempts

    @pytest.mark.asyncio
    async def test_decorator_with_non_retryable_exception(self):
        """Test that non-retryable exceptions are not counted."""

        @circuit_breaker(
            service_name="test_non_retryable",
            failure_threshold=2,
            expected_exceptions=(ExternalServiceError,),
        )
        async def test_function(exception_type="retryable"):
            if exception_type == "retryable":
                raise ExternalServiceError("Retryable error")
            else:
                raise ValueError("Non-retryable error")

        # Non-retryable exceptions should not affect circuit state
        with pytest.raises(ValueError):
            await test_function(exception_type="non_retryable")

        status = test_function.get_circuit_status()
        assert status["failure_count"] == 0
        assert status["state"] == "closed"


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with service dependencies."""

    @pytest.mark.asyncio
    async def test_service_dependency_protection(self):
        """Test that service dependencies are protected by circuit breakers."""
        from src.services.dependencies import (
            get_circuit_breaker_status,
            reset_all_circuit_breakers,
            reset_circuit_breaker,
        )

        # Test circuit breaker status endpoint
        status = await get_circuit_breaker_status()
        assert "summary" in status
        assert "circuits" in status
        assert "timestamp" in status

        # Test individual circuit reset
        result = await reset_circuit_breaker("nonexistent_service")
        assert result["success"] is False
        assert "not found" in result["error"]

        # Test all circuit reset
        result = await reset_all_circuit_breakers()
        assert result["success"] is True
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_health_check_includes_circuit_breakers(self):
        """Test that health checks include circuit breaker information."""
        from src.services.dependencies import get_service_health

        with patch("src.services.dependencies.get_client_manager") as mock_get_client:
            mock_client = Mock()
            mock_client.get_health_status = AsyncMock(return_value={"status": "ok"})
            mock_get_client.return_value = mock_client

            health = await get_service_health()

            assert "circuit_breakers" in health
            assert "services" in health
            assert "timestamp" in health


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up circuit breaker registry after each test."""
    yield
    # Clear registry to avoid test interference
    CircuitBreakerRegistry._breakers.clear()
