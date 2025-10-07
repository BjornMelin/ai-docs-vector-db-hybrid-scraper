"""Circuit breaker and rate limiter stress tests for AI Documentation Vector DB.

This module tests the effectiveness of circuit breakers and rate limiters
under extreme load conditions, including trigger point validation,
recovery behavior, and cascading failure prevention.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from tests.load.conftest import LoadTestConfig, LoadTestType


class TestError(Exception):
    """Custom exception for this module."""


logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 30.0
    failure_rate_threshold: float = 0.5


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""

    _total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    state_changes: list[dict[str, Any]] = field(default_factory=list)
    rejection_count: int = 0


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.metrics = CircuitBreakerMetrics()

    async def call(self, func, *args, **_kwargs):
        """Execute function with circuit breaker protection."""
        self.metrics.total_requests += 1

        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if (
                self.last_failure_time
                and (time.time() - self.last_failure_time) > self.config.timeout
            ):
                # Transition to half-open
                self._transition_to_half_open()
            else:
                # Reject request
                msg = "Circuit breaker is open - rejecting request"
                raise TestError(msg)
                msg = "Circuit breaker is open - rejecting request"
                raise TestError(msg)

        try:
            # Execute the function
            result = await func(*args, **_kwargs)

            # Success - reset failure count
            self.failure_count = 0
            self.success_count += 1
            self.metrics.successful_requests += 1

            # Check if we should close the circuit (from half-open)
            if (
                self.state == CircuitBreakerState.HALF_OPEN
                and self.success_count >= self.config.success_threshold
            ):
                self._transition_to_closed()

        except Exception:
            # Failure - increment failure count
            self.failure_count += 1
            self.success_count = 0
            self.metrics.failed_requests += 1
            self.last_failure_time = time.time()

            # Check if we should open the circuit
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Go back to open on any failure
                self._transition_to_open()

            raise
        else:
            return result

    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.metrics.circuit_opened_count += 1
            self.metrics.state_changes.append(
                {
                    "timestamp": time.time(),
                    "from_state": self.state.value,
                    "to_state": CircuitBreakerState.OPEN.value,
                    "trigger": "failure_threshold_exceeded",
                }
            )
            logger.warning("Circuit breaker opened due to failures")

    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.metrics.state_changes.append(
            {
                "timestamp": time.time(),
                "from_state": CircuitBreakerState.OPEN.value,
                "to_state": CircuitBreakerState.HALF_OPEN.value,
                "trigger": "timeout_expired",
            }
        )
        logger.info("Circuit breaker transitioned to half-open")

    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.metrics.circuit_closed_count += 1
        self.metrics.state_changes.append(
            {
                "timestamp": time.time(),
                "from_state": CircuitBreakerState.HALF_OPEN.value,
                "to_state": CircuitBreakerState.CLOSED.value,
                "trigger": "success_threshold_met",
            }
        )
        logger.info("Circuit breaker closed - service recovered")


@dataclass
class RateLimiterConfig:
    """Rate limiter configuration."""

    max_requests: int = 100
    time_window: float = 60.0  # seconds
    burst_size: int = 10


@dataclass
class RateLimiterMetrics:
    """Rate limiter metrics."""

    _total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_rate: float = 0.0
    peak_rate: float = 0.0
    time_windows: list[dict[str, Any]] = field(default_factory=list)


class MockRateLimiter:
    """Mock rate limiter for testing."""

    def __init__(self, config: RateLimiterConfig):
        self.config = config
        self.metrics = RateLimiterMetrics()
        self.request_times = []
        self.burst_allowance = config.burst_size
        self.last_refill = time.time()

    async def allow_request(self) -> bool:
        """Check if request should be allowed."""
        current_time = time.time()
        self.metrics.total_requests += 1

        # Clean old requests outside time window
        cutoff_time = current_time - self.config.time_window
        self.request_times = [t for t in self.request_times if t > cutoff_time]

        # Refill burst allowance
        time_since_refill = current_time - self.last_refill
        refill_amount = int(
            time_since_refill * (self.config.max_requests / self.config.time_window)
        )
        if refill_amount > 0:
            self.burst_allowance = min(
                self.config.burst_size, self.burst_allowance + refill_amount
            )
            self.last_refill = current_time

        # Check rate limit
        current_requests = len(self.request_times)
        self.metrics.current_rate = (
            current_requests / self.config.time_window * 60
        )  # requests per minute
        self.metrics.peak_rate = max(self.metrics.peak_rate, self.metrics.current_rate)

        # Allow if under rate limit or burst available
        if current_requests < self.config.max_requests or self.burst_allowance > 0:
            self.request_times.append(current_time)
            self.metrics.allowed_requests += 1

            if current_requests >= self.config.max_requests:
                self.burst_allowance -= 1

            return True
        self.metrics.rejected_requests += 1
        return False


class TestCircuitBreakers:
    """Test suite for circuit breaker behavior under stress."""

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_circuit_breaker_trigger_points(self, load_test_runner):
        """Test circuit breaker trigger points under increasing failure rates."""

        # Configure circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=10.0,
            failure_rate_threshold=0.5,
        )
        circuit_breaker = MockCircuitBreaker(cb_config)

        # Mock service with configurable failure rate
        class FailureInjectingService:
            def __init__(self):
                self.failure_rate = 0.0
                self.call_count = 0

            def set_failure_rate(self, rate: float):
                self.failure_rate = max(0.0, min(1.0, rate))

            async def process_request(self, **__kwargs):
                self.call_count += 1

                # Inject failures based on failure rate
                if time.time() % 1.0 < self.failure_rate:
                    msg = f"Injected failure (rate: {self.failure_rate})"
                    raise TestError(msg)

                await asyncio.sleep(0.1)  # Simulate processing
                return {"status": "success", "call_count": self.call_count}

        failing_service = FailureInjectingService()

        # Test different failure rates
        failure_scenarios = [
            {"rate": 0.1, "duration": 30, "expected_open": False},  # Low failure rate
            {
                "rate": 0.3,
                "duration": 30,
                "expected_open": False,
            },  # Moderate failure rate
            {"rate": 0.7, "duration": 30, "expected_open": True},  # High failure rate
            {
                "rate": 0.9,
                "duration": 30,
                "expected_open": True,
            },  # Very high failure rate
        ]

        scenario_results = []

        for i, scenario in enumerate(failure_scenarios):
            logger.info(
                "Testing circuit breaker with %.1f%% failure rate",
                scenario["rate"] * 100,
            )

            # Set failure rate
            failing_service.set_failure_rate(scenario["rate"])

            # Reset circuit breaker state for clean test
            if i > 0:  # Don't reset for first scenario
                circuit_breaker.state = CircuitBreakerState.CLOSED
                circuit_breaker.failure_count = 0
                circuit_breaker.success_count = 0

            async def circuit_protected_request(**_kwargs):
                """Request protected by circuit breaker."""
                return await circuit_breaker.call(
                    failing_service.process_request, **_kwargs
                )

            # Configure test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=20,
                requests_per_second=10,
                duration_seconds=scenario["duration"],
                success_criteria={
                    "max_error_rate_percent": 80.0,  # Allow high error rates
                },
            )

            # Run test
            initial_state = circuit_breaker.state
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=circuit_protected_request,
            )
            final_state = circuit_breaker.state

            # Analyze results
            circuit_opened = circuit_breaker.metrics.circuit_opened_count > 0
            error_rate = (
                result.metrics.failed_requests / max(result.metrics.total_requests, 1)
            ) * 100

            scenario_results.append(
                {
                    "failure_rate": scenario["rate"],
                    "circuit_opened": circuit_opened,
                    "expected_open": scenario["expected_open"],
                    "error_rate": error_rate,
                    "rejection_count": circuit_breaker.metrics.rejection_count,
                    "state_changes": len(circuit_breaker.metrics.state_changes),
                    "initial_state": initial_state,
                    "final_state": final_state,
                }
            )

            logger.info(
                "Scenario %d: Failure rate %.1f%%, Circuit opened: %s, "
                "Error rate: %.2f%%, Rejections: %d",
                i + 1,
                scenario["rate"] * 100,
                circuit_opened,
                error_rate,
                circuit_breaker.metrics.rejection_count,
            )

        # Assertions
        assert len(scenario_results) == len(failure_scenarios), (
            "Not all scenarios completed"
        )

        # Verify circuit breaker behavior
        for result in scenario_results:
            if result["expected_open"]:
                assert result["circuit_opened"], (
                    f"Circuit breaker should have opened at "
                    f"{result['failure_rate']:.1%} failure rate"
                )
            else:
                # Low failure rates shouldn't trigger circuit breaker
                if result["failure_rate"] < 0.5:
                    assert not result["circuit_opened"], (
                        f"Circuit breaker opened unnecessarily at "
                        f"{result['failure_rate']:.1%} failure rate"
                    )

        # Verify progressive behavior
        high_failure_results = [r for r in scenario_results if r["failure_rate"] >= 0.7]
        assert all(r["circuit_opened"] for r in high_failure_results), (
            "Circuit breaker should open for high failure rates"
        )

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_behavior(self, load_test_runner):
        """Test circuit breaker recovery behavior after service restoration."""

        # Configure circuit breaker with shorter timeout for testing
        cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=5.0,  # Short timeout for faster testing
        )
        circuit_breaker = MockCircuitBreaker(cb_config)

        # Mock service with controllable health
        class RecoveringService:
            def __init__(self):
                self.is_healthy = True
                self.call_count = 0
                self.recovery_time = None

            def set_health(self, healthy: bool):
                if healthy and not self.is_healthy:
                    self.recovery_time = time.time()
                self.is_healthy = healthy

            async def process_request(self, **__kwargs):
                self.call_count += 1

                if not self.is_healthy:
                    msg = "Service is unhealthy"
                    raise TestError(msg)

                await asyncio.sleep(0.05)  # Fast processing when healthy
                return {"status": "success", "call_count": self.call_count}

        service = RecoveringService()

        # Recovery test phases
        phases = [
            {"name": "healthy", "duration": 10, "service_healthy": True},
            {
                "name": "failure",
                "duration": 15,
                "service_healthy": False,
            },  # Trigger circuit open
            {
                "name": "recovery_wait",
                "duration": 8,
                "service_healthy": True,
            },  # Service recovers, CB still open
            {
                "name": "recovery_test",
                "duration": 20,
                "service_healthy": True,
            },  # CB should transition to half-open and close
        ]

        phase_results = []

        async def circuit_protected_request(**_kwargs):
            """Request protected by circuit breaker."""
            return await circuit_breaker.call(service.process_request, **_kwargs)

        for phase in phases:
            logger.info(
                "Running recovery phase: %s", phase["name"]
            )

            # Set service health
            service.set_health(phase["service_healthy"])

            # Configure phase test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=10,
                requests_per_second=5,
                duration_seconds=phase["duration"],
                success_criteria={
                    "max_error_rate_percent": 90.0,  # Allow errors during failure phase
                },
            )

            # Track circuit breaker state changes during phase
            initial_state = circuit_breaker.state
            initial_metrics = {
                "_total_requests": circuit_breaker.metrics.total_requests,
                "rejections": circuit_breaker.metrics.rejection_count,
                "opened_count": circuit_breaker.metrics.circuit_opened_count,
                "closed_count": circuit_breaker.metrics.circuit_closed_count,
            }

            # Run phase
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=circuit_protected_request,
            )

            final_state = circuit_breaker.state
            final_metrics = {
                "_total_requests": circuit_breaker.metrics.total_requests,
                "rejections": circuit_breaker.metrics.rejection_count,
                "opened_count": circuit_breaker.metrics.circuit_opened_count,
                "closed_count": circuit_breaker.metrics.circuit_closed_count,
            }

            # Calculate phase-specific metrics
            phase_metrics = {
                "requests": final_metrics["_total_requests"]
                - initial_metrics["_total_requests"],
                "rejections": final_metrics["rejections"]
                - initial_metrics["rejections"],
                "state_changes": final_metrics["opened_count"]
                + final_metrics["closed_count"]
                - initial_metrics["opened_count"]
                - initial_metrics["closed_count"],
            }

            error_rate = (
                result.metrics.failed_requests / max(result.metrics.total_requests, 1)
            ) * 100

            phase_results.append(
                {
                    "phase": phase["name"],
                    "service_healthy": phase["service_healthy"],
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "error_rate": error_rate,
                    "phase_metrics": phase_metrics,
                    "result": result,
                }
            )

            logger.info(
                "Phase %s: %s -> %s, Error rate: %.2f%%, Rejections: %d",
                phase["name"],
                initial_state.value,
                final_state.value,
                error_rate,
                phase_metrics["rejections"],
            )

        # Analyze recovery behavior
        failure_phase = next(p for p in phase_results if p["phase"] == "failure")
        recovery_test_phase = next(
            p for p in phase_results if p["phase"] == "recovery_test"
        )

        # Assertions for recovery behavior
        assert failure_phase["final_state"] == CircuitBreakerState.OPEN, (
            "Circuit breaker should have opened during failure phase"
        )

        assert recovery_test_phase["final_state"] == CircuitBreakerState.CLOSED, (
            "Circuit breaker should have closed during recovery phase"
        )

        # Verify circuit breaker protected system during failure
        assert failure_phase["phase_metrics"]["rejections"] > 0, (
            "Circuit breaker should have rejected requests during failure"
        )

        # Verify successful recovery
        assert recovery_test_phase["error_rate"] < 10, (
            f"High error rate during recovery: {recovery_test_phase['error_rate']:.2f}%"
        )

        # Verify state transitions occurred
        _total_state_changes = sum(
            p["phase_metrics"]["state_changes"] for p in phase_results
        )
        assert _total_state_changes >= 2, (
            f"Expected at least 2 state changes (open -> half-open -> "
            f"closed), got {_total_state_changes}"
        )

        logger.info("Circuit breaker recovery completed successfully")
        logger.info(
            "Total state changes: %s", _total_state_changes
        )
        logger.info(
            "Final circuit breaker metrics: %s", circuit_breaker.metrics
        )


class TestRateLimiters:
    """Test suite for rate limiter behavior under stress."""

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_rate_limiter_effectiveness(self, load_test_runner):
        """Test rate limiter effectiveness under various load patterns."""

        # Configure rate limiter
        rl_config = RateLimiterConfig(
            max_requests=50,  # 50 requests per minute
            time_window=60.0,
            burst_size=10,
        )
        rate_limiter = MockRateLimiter(rl_config)

        # Mock service protected by rate limiter
        async def rate_limited_service(**__kwargs):
            """Service protected by rate limiter."""
            if not await rate_limiter.allow_request():
                msg = "Rate limit exceeded"
                raise TestError(msg)

            await asyncio.sleep(0.1)  # Simulate processing
            return {"status": "processed", "timestamp": time.time()}

        # Test different load patterns
        load_patterns = [
            {
                "name": "under_limit",
                "users": 20,
                "rps": 30,
                "duration": 60,
            },  # Under rate limit
            {
                "name": "at_limit",
                "users": 30,
                "rps": 50,
                "duration": 60,
            },  # At rate limit
            {
                "name": "over_limit",
                "users": 50,
                "rps": 80,
                "duration": 60,
            },  # Over rate limit
            {"name": "burst", "users": 100, "rps": 150, "duration": 30},  # Burst load
        ]

        pattern_results = []

        for pattern in load_patterns:
            logger.info(
                "Testing rate limiter with load pattern: %s", pattern["name"]
            )

            # Reset rate limiter for clean test
            rate_limiter.request_times.clear()
            rate_limiter.burst_allowance = rl_config.burst_size
            rate_limiter.metrics = RateLimiterMetrics()

            # Configure test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=pattern["users"],
                requests_per_second=pattern["rps"],
                duration_seconds=pattern["duration"],
                success_criteria={
                    "max_error_rate_percent": 70.0,  # Allow rate limit rejections
                },
            )

            # Run test
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=rate_limited_service,
            )

            # Analyze rate limiting effectiveness
            _total_attempted = rate_limiter.metrics.total_requests
            allowed = rate_limiter.metrics.allowed_requests
            rejected = rate_limiter.metrics.rejected_requests

            rejection_rate = (rejected / max(_total_attempted, 1)) * 100
            actual_rate = allowed / (pattern["duration"] / 60)  # Requests per minute

            pattern_results.append(
                {
                    "pattern": pattern["name"],
                    "target_rps": pattern["rps"],
                    "_total_attempted": _total_attempted,
                    "allowed": allowed,
                    "rejected": rejected,
                    "rejection_rate": rejection_rate,
                    "actual_rate": actual_rate,
                    "peak_rate": rate_limiter.metrics.peak_rate,
                    "result": result,
                }
            )

            logger.info(
                "Pattern %s: %.2f%% rejected, Actual rate: %.1f req/min, "
                "Peak rate: %.1f req/min",
                pattern["name"],
                rejection_rate,
                actual_rate,
                rate_limiter.metrics.peak_rate,
            )

        # Assertions for rate limiting effectiveness
        under_limit = next(p for p in pattern_results if p["pattern"] == "under_limit")
        over_limit = next(p for p in pattern_results if p["pattern"] == "over_limit")
        burst = next(p for p in pattern_results if p["pattern"] == "burst")

        # Under limit should have minimal rejections
        assert under_limit["rejection_rate"] < 10, (
            f"Too many rejections under rate limit: "
            f"{under_limit['rejection_rate']:.2f}%"
        )

        # Over limit should have significant rejections
        assert over_limit["rejection_rate"] > 30, (
            f"Insufficient rejections over rate limit: "
            f"{over_limit['rejection_rate']:.2f}%"
        )

        # Burst should be partially handled by burst allowance
        assert burst["rejection_rate"] > 50, (
            f"Burst load not properly limited: {burst['rejection_rate']:.2f}%"
        )

        # Rate should be enforced (actual rate should not exceed limit significantly)
        for result in pattern_results:
            if result["pattern"] != "under_limit":
                # Allow some tolerance for timing variations
                max_allowed_rate = rl_config.max_requests * 1.2  # 20% tolerance
                assert result["actual_rate"] <= max_allowed_rate, (
                    f"Rate limit exceeded: {result['actual_rate']:.1f} > "
                    f"{max_allowed_rate:.1f} req/min"
                )

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_thundering_herd_protection(self, load_test_runner):
        """Test protection against thundering herd scenarios."""

        # Configure aggressive rate limiter for thundering herd protection
        rl_config = RateLimiterConfig(
            max_requests=20,  # Very low limit
            time_window=60.0,
            burst_size=5,  # Small burst allowance
        )
        rate_limiter = MockRateLimiter(rl_config)

        # Mock expensive service (e.g., cache rebuild, database query)
        class ExpensiveService:
            def __init__(self):
                self.concurrent_executions = 0
                self.max_concurrent = 0
                self._total_executions = 0
                self.execution_times = []

            async def expensive_operation(self, **__kwargs):
                """Simulate expensive operation that shouldn't be overwhelmed."""
                self.concurrent_executions += 1
                self.max_concurrent = max(
                    self.max_concurrent, self.concurrent_executions
                )
                self._total_executions += 1

                try:
                    # Simulate expensive work (database query, cache rebuild, etc.)
                    processing_time = 2.0 + (
                        self.concurrent_executions * 0.5
                    )  # Slower with more concurrency
                    await asyncio.sleep(processing_time)

                    self.execution_times.append(processing_time)

                    return {
                        "status": "completed",
                        "execution_id": self._total_executions,
                        "processing_time": processing_time,
                        "concurrent_count": self.concurrent_executions,
                    }
                finally:
                    self.concurrent_executions -= 1

        expensive_service = ExpensiveService()

        async def rate_limited_expensive_operation(**_kwargs):
            """Expensive operation protected by rate limiter."""
            if not await rate_limiter.allow_request():
                msg = "Rate limit exceeded - protecting expensive service"
                raise TestError(msg)

            return await expensive_service.expensive_operation(**_kwargs)

        # Simulate thundering herd scenario
        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=200,  # Many simultaneous users
            requests_per_second=100,  # High request rate
            duration_seconds=60,
            success_criteria={
                "max_error_rate_percent": 80.0,  # Expect many rejections
            },
        )

        # Run thundering herd test
        logger.info("Testing thundering herd protection")
        await load_test_runner.run_load_test(
            config=config,
            target_function=rate_limited_expensive_operation,
        )

        # Analyze protection effectiveness
        _total_attempted = rate_limiter.metrics.total_requests
        rejected = rate_limiter.metrics.rejected_requests

        rejection_rate = (rejected / max(_total_attempted, 1)) * 100

        # Assertions for thundering herd protection
        assert rejection_rate > 70, (
            f"Insufficient rate limiting for thundering herd: "
            f"{rejection_rate:.2f}% rejected"
        )

        assert expensive_service.max_concurrent <= 10, (
            f"Too many concurrent executions: {expensive_service.max_concurrent}"
        )

        assert expensive_service._total_executions <= 50, (
            f"Too many expensive operations executed: "
            f"{expensive_service._total_executions}"
        )

        # Verify service performance wasn't degraded
        if expensive_service.execution_times:
            avg_execution_time = sum(expensive_service.execution_times) / len(
                expensive_service.execution_times
            )
            assert avg_execution_time < 5.0, (
                f"Expensive service performance degraded: "
                f"{avg_execution_time:.2f}s average"
            )

        logger.info("Thundering herd protection successful:")
        logger.info("  - %.2f%% requests rejected", rejection_rate)
        logger.info(
            "  - Max concurrent executions: %s", expensive_service.max_concurrent
        )
        logger.info(
            "  - Total expensive operations: %s", expensive_service._total_executions
        )
        logger.info("  - Service remained responsive")


@pytest.fixture
def circuit_breaker_config():
    """Provide circuit breaker configuration for tests."""
    return CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout=30.0,
        failure_rate_threshold=0.5,
    )


@pytest.fixture
def rate_limiter_config():
    """Provide rate limiter configuration for tests."""
    return RateLimiterConfig(
        max_requests=100,
        time_window=60.0,
        burst_size=10,
    )
