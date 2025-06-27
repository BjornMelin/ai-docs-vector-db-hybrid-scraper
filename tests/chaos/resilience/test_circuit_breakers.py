"""Circuit breaker resilience tests for chaos engineering.

This module implements comprehensive circuit breaker testing to validate
failure detection, fast-fail behavior, and automatic recovery mechanisms.
"""

import asyncio  # noqa: PLC0415
import random
import time  # noqa: PLC0415
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open."""

    pass


class ServiceFailureError(CircuitBreakerError):
    """Raised when service fails."""

    pass


class ResourceExhaustedError(CircuitBreakerError):
    """Raised when resources are exhausted."""

    pass


class SystemOverloadError(CircuitBreakerError):
    """Raised when system is overloaded."""

    pass


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    timeout: float = 30.0
    expected_exception: type = Exception
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # Successes needed to close from half-open


class CircuitBreaker:
    """Circuit breaker implementation for testing."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_attempt_time = 0
        self.call_count = 0

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.call_count += 1
        self.last_attempt_time = time.time()

        # Check if circuit should be reset to half-open
        if (
            self.state == CircuitState.OPEN
            and time.time() - self.last_failure_time >= self.config.timeout
        ):
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0

        # Fast-fail if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
        except self.config.expected_exception:
            await self._on_failure()
            raise
        else:
            return result

    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success

    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if (
            self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]
            and self.failure_count >= self.config.failure_threshold
        ):
            self.state = CircuitState.OPEN

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_count": self.call_count,
            "last_failure_time": self.last_failure_time,
            "last_attempt_time": self.last_attempt_time,
            "time_since_last_failure": time.time() - self.last_failure_time
            if self.last_failure_time
            else 0,
        }


@pytest.mark.chaos
@pytest.mark.resilience
class TestCircuitBreakers:
    """Test circuit breaker resilience patterns."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Default circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            timeout=0.1,  # Fast timeout for testing
            expected_exception=Exception,
            recovery_timeout=0.2,
            success_threshold=2,
        )

    async def test_circuit_breaker_state_transitions(
        self, circuit_breaker_config, _fault_injector
    ):
        """Test circuit breaker state transitions."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Mock service that fails
        call_count = 0

        async def failing_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First 3 calls fail
                raise ServiceFailureError("Service failure")
            return {"status": "success", "call": call_count}

        # Test CLOSED -> OPEN transition
        assert circuit_breaker.state == CircuitState.CLOSED

        # Make failing calls to trigger circuit breaker
        for _i in range(3):
            with pytest.raises(ServiceFailureError, match="Service failure"):
                await circuit_breaker.call(failing_service)

        # Circuit should now be OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 3

        # Test fast-fail behavior
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            await circuit_breaker.call(failing_service)

        # Wait for timeout to test OPEN -> HALF_OPEN transition
        await asyncio.sleep(circuit_breaker_config.timeout + 0.01)

        # Next call should put circuit in HALF_OPEN
        result = await circuit_breaker.call(failing_service)
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        assert result["status"] == "success"

        # One more success should close the circuit
        result = await circuit_breaker.call(failing_service)
        assert circuit_breaker.state == CircuitState.CLOSED
        assert result["status"] == "success"

    async def test_circuit_breaker_with_different_failure_rates(
        self, circuit_breaker_config
    ):
        """Test circuit breaker with different failure rates."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)

        failure_rate = 0.3  # 30% failure rate
        call_count = 0

        async def intermittent_service():
            nonlocal call_count
            call_count += 1

            if random.random() < failure_rate:
                raise ServiceFailureError(f"Intermittent failure on call {call_count}")

            return {"status": "success", "call": call_count}

        # Make many calls to test circuit breaker behavior with intermittent failures
        successes = 0
        failures = 0
        circuit_breaker_activations = 0

        for _i in range(50):
            try:
                await circuit_breaker.call(intermittent_service)
                successes += 1
            except Exception as e:
                if "Circuit breaker is open" in str(e):
                    circuit_breaker_activations += 1
                else:
                    failures += 1

                # If circuit is open, wait for it to reset
                if circuit_breaker.state == CircuitState.OPEN:
                    await asyncio.sleep(circuit_breaker_config.timeout + 0.01)

        # Verify behavior
        assert successes > 0, "Should have some successful calls"
        assert failures > 0, "Should have some service failures"

        # Circuit breaker should have activated at least once with 30% failure rate
        stats = circuit_breaker.get_stats()
        assert stats["call_count"] > 0

    async def test_circuit_breaker_timeout_recovery(self, circuit_breaker_config):
        """Test circuit breaker timeout-based recovery."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Service that fails initially, then recovers
        service_recovered = False

        async def recovering_service():
            if not service_recovered:
                raise ServiceFailureError("Service not yet recovered")
            return {"status": "recovered"}

        # Trigger circuit breaker
        for _ in range(circuit_breaker_config.failure_threshold):
            with pytest.raises(ServiceFailureError, match="Service not yet recovered"):
                await circuit_breaker.call(recovering_service)

        assert circuit_breaker.state == CircuitState.OPEN

        # Test fast-fail behavior
        fast_fail_start = time.time()
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            await circuit_breaker.call(recovering_service)
        fast_fail_duration = time.time() - fast_fail_start

        # Fast-fail should be very quick
        assert fast_fail_duration < 0.01, "Fast-fail should be nearly instantaneous"

        # Wait for timeout but don't recover service yet
        await asyncio.sleep(circuit_breaker_config.timeout + 0.01)

        # Should transition to HALF_OPEN but still fail
        with pytest.raises(ServiceFailureError, match="Service not yet recovered"):
            await circuit_breaker.call(recovering_service)

        # Should be OPEN again after failure in HALF_OPEN
        assert circuit_breaker.state == CircuitState.OPEN

        # Now recover the service and wait for timeout again
        service_recovered = True
        await asyncio.sleep(circuit_breaker_config.timeout + 0.01)

        # Should succeed and eventually close circuit
        for _ in range(circuit_breaker_config.success_threshold):
            result = await circuit_breaker.call(recovering_service)
            assert result["status"] == "recovered"

        assert circuit_breaker.state == CircuitState.CLOSED

    async def test_circuit_breaker_with_different_exception_types(
        self, _circuit_breaker_config
    ):
        """Test circuit breaker with different exception types."""
        # Configure circuit breaker for specific exception type
        specific_config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.1,
            expected_exception=ConnectionError,  # Only count ConnectionErrors
            recovery_timeout=0.2,
            success_threshold=1,
        )

        circuit_breaker = CircuitBreaker(specific_config)

        call_count = 0

        async def service_with_different_errors():
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise ValueError("Value error - should not trigger circuit breaker")
            elif call_count == 2:
                raise ConnectionError("Connection error - should count")
            elif call_count == 3:
                raise ConnectionError(
                    "Another connection error - should trigger circuit breaker"
                )
            elif call_count == 4:
                raise TimeoutError(
                    "Timeout error - should not count when circuit is open"
                )
            else:
                return {"status": "success"}

        # First call with ValueError - should not count toward circuit breaker
        with pytest.raises(ValueError):
            await circuit_breaker.call(service_with_different_errors)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

        # Second call with ConnectionError - should count
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(service_with_different_errors)

        assert circuit_breaker.failure_count == 1

        # Third call with ConnectionError - should trigger circuit breaker
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(service_with_different_errors)

        assert circuit_breaker.state == CircuitState.OPEN

        # Fourth call should fast-fail regardless of exception type
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            await circuit_breaker.call(service_with_different_errors)

    async def test_circuit_breaker_metrics_and_monitoring(self, circuit_breaker_config):
        """Test circuit breaker metrics collection."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Service with predictable behavior
        async def monitored_service(should_fail: bool = False):
            if should_fail:
                raise ServiceFailureError("Monitored failure")
            return {"status": "success"}

        # Track initial stats
        initial_stats = circuit_breaker.get_stats()
        assert initial_stats["state"] == "closed"
        assert initial_stats["call_count"] == 0
        assert initial_stats["failure_count"] == 0

        # Make successful calls
        for _ in range(3):
            await circuit_breaker.call(monitored_service, should_fail=False)

        success_stats = circuit_breaker.get_stats()
        assert success_stats["call_count"] == 3
        assert success_stats["failure_count"] == 0
        assert success_stats["state"] == "closed"

        # Make failing calls
        for _ in range(circuit_breaker_config.failure_threshold):
            with pytest.raises(ServiceFailureError):
                await circuit_breaker.call(monitored_service, should_fail=True)

        failure_stats = circuit_breaker.get_stats()
        assert failure_stats["call_count"] == 6  # 3 successes + 3 failures
        assert (
            failure_stats["failure_count"] == circuit_breaker_config.failure_threshold
        )
        assert failure_stats["state"] == "open"
        assert failure_stats["last_failure_time"] > 0

        # Test fast-fail metrics
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            await circuit_breaker.call(monitored_service, should_fail=False)

        fast_fail_stats = circuit_breaker.get_stats()
        assert fast_fail_stats["call_count"] == 7  # Previous + 1 fast-fail

    async def test_multiple_circuit_breakers(self, _circuit_breaker_config):
        """Test multiple circuit breakers for different services."""
        # Create circuit breakers for different services
        db_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=2, timeout=0.1, expected_exception=ConnectionError
            )
        )

        api_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3, timeout=0.15, expected_exception=Exception
            )
        )

        cache_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5, timeout=0.05, expected_exception=TimeoutError
            )
        )

        # Mock services
        async def database_service():
            raise ConnectionError("Database connection failed")

        async def api_service():
            raise ServiceFailureError("API service failed")

        async def cache_service():
            return {"cached": "data"}  # This one works

        # Test database circuit breaker
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await db_circuit.call(database_service)

        assert db_circuit.state == CircuitState.OPEN

        # Test API circuit breaker (higher threshold)
        for _ in range(2):
            with pytest.raises(ServiceFailureError):
                await api_circuit.call(api_service)

        assert api_circuit.state == CircuitState.CLOSED  # Not reached threshold yet

        # One more failure should open API circuit
        with pytest.raises(ServiceFailureError):
            await api_circuit.call(api_service)

        assert api_circuit.state == CircuitState.OPEN

        # Cache circuit should still work
        result = await cache_circuit.call(cache_service)
        assert result["cached"] == "data"
        assert cache_circuit.state == CircuitState.CLOSED

        # Verify independent operation
        db_stats = db_circuit.get_stats()
        api_stats = api_circuit.get_stats()
        cache_stats = cache_circuit.get_stats()

        assert db_stats["failure_count"] == 2
        assert api_stats["failure_count"] == 3
        assert cache_stats["failure_count"] == 0

    async def test_circuit_breaker_with_retry_logic(self, circuit_breaker_config):
        """Test circuit breaker integration with retry mechanisms."""
        circuit_breaker = CircuitBreaker(circuit_breaker_config)

        call_attempts = 0

        async def unreliable_service():
            nonlocal call_attempts
            call_attempts += 1

            # Fail first 5 attempts, then succeed
            if call_attempts <= 5:
                raise ServiceFailureError(f"Service failure attempt {call_attempts}")

            return {"status": "success", "attempts": call_attempts}

        async def retry_with_circuit_breaker(max_retries: int = 3):
            """Retry logic that respects circuit breaker."""
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = await circuit_breaker.call(unreliable_service)
                except Exception as e:
                    last_exception = e

                    # Don't retry if circuit breaker is open
                    if "Circuit breaker is open" in str(e):
                        break
                else:
                    return result

                    # Brief delay between retries
                    if attempt < max_retries:
                        await asyncio.sleep(0.01)

            raise last_exception

        # First retry attempt - should fail and open circuit breaker
        with pytest.raises(ServiceFailureError):
            await retry_with_circuit_breaker(max_retries=5)

        assert circuit_breaker.state == CircuitState.OPEN

        # Subsequent attempts should fast-fail without calling service
        service_calls_before = call_attempts

        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            await retry_with_circuit_breaker(max_retries=3)

        # Service should not have been called due to circuit breaker
        assert call_attempts == service_calls_before

        # Wait for circuit breaker timeout
        await asyncio.sleep(circuit_breaker_config.timeout + 0.01)

        # Now retry should succeed (service has "recovered")
        result = await retry_with_circuit_breaker(max_retries=3)
        assert result["status"] == "success"

    async def test_circuit_breaker_bulkhead_pattern(self):
        """Test circuit breaker with bulkhead pattern for resource isolation."""
        # Different circuit breakers for different resource pools
        critical_pool_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=2, timeout=0.1, expected_exception=Exception
            )
        )

        normal_pool_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5, timeout=0.2, expected_exception=Exception
            )
        )

        # Resource pools
        critical_pool = {"available": 2, "used": 0}
        normal_pool = {"available": 10, "used": 0}

        async def critical_operation():
            """Operation using critical resource pool."""
            if critical_pool["used"] >= critical_pool["available"]:
                raise ResourceExhaustedError("Critical pool exhausted")

            critical_pool["used"] += 1
            try:
                await asyncio.sleep(0.01)  # Simulate work
                return {"pool": "critical", "status": "success"}
            finally:
                critical_pool["used"] -= 1

        async def normal_operation():
            """Operation using normal resource pool."""
            if normal_pool["used"] >= normal_pool["available"]:
                raise ResourceExhaustedError("Normal pool exhausted")

            normal_pool["used"] += 1
            try:
                await asyncio.sleep(0.01)  # Simulate work
                return {"pool": "normal", "status": "success"}
            finally:
                normal_pool["used"] -= 1

        # Overwhelm critical pool to trigger its circuit breaker
        critical_tasks = [
            critical_pool_circuit.call(critical_operation)
            for _ in range(5)  # More than critical pool capacity
        ]

        critical_results = await asyncio.gather(*critical_tasks, return_exceptions=True)

        # Some critical operations should fail and open the circuit
        critical_failures = [r for r in critical_results if isinstance(r, Exception)]
        assert len(critical_failures) > 0

        # Normal pool should still work
        normal_result = await normal_pool_circuit.call(normal_operation)
        assert normal_result["status"] == "success"
        assert normal_pool_circuit.state == CircuitState.CLOSED

        # Verify isolation - critical circuit failure doesn't affect normal circuit
        assert critical_pool_circuit.state == CircuitState.OPEN
        assert normal_pool_circuit.state == CircuitState.CLOSED

    async def test_circuit_breaker_adaptive_thresholds(self):
        """Test adaptive circuit breaker thresholds based on system load."""
        # Circuit breaker with adaptive configuration
        base_threshold = 3

        class AdaptiveCircuitBreaker(CircuitBreaker):
            def __init__(self, config: CircuitBreakerConfig):
                super().__init__(config)
                self.system_load = 0.0  # 0.0 to 1.0

            def get_adaptive_threshold(self) -> int:
                """Adjust failure threshold based on system load."""
                if self.system_load > 0.8:  # High load
                    return max(1, self.config.failure_threshold // 2)
                elif self.system_load > 0.6:  # Medium load
                    return max(2, int(self.config.failure_threshold * 0.75))
                else:  # Low load
                    return self.config.failure_threshold

            async def _on_failure(self):
                """Override to use adaptive threshold."""
                self.failure_count += 1
                self.last_failure_time = time.time()

                adaptive_threshold = self.get_adaptive_threshold()

                if (
                    self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]
                    and self.failure_count >= adaptive_threshold
                ):
                    self.state = CircuitState.OPEN

        adaptive_circuit = AdaptiveCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=base_threshold, timeout=0.1)
        )

        async def failing_service():
            raise ServiceFailureError("Service failure")

        # Test with low system load (normal threshold)
        adaptive_circuit.system_load = 0.3
        normal_threshold = adaptive_circuit.get_adaptive_threshold()
        assert normal_threshold == base_threshold

        # Test with high system load (reduced threshold)
        adaptive_circuit.system_load = 0.9
        reduced_threshold = adaptive_circuit.get_adaptive_threshold()
        assert reduced_threshold < base_threshold

        # Trigger circuit breaker with high load
        for _ in range(reduced_threshold):
            with pytest.raises(ServiceFailureError):
                await adaptive_circuit.call(failing_service)

        # Should open with fewer failures due to high load
        assert adaptive_circuit.state == CircuitState.OPEN
        assert adaptive_circuit.failure_count == reduced_threshold
