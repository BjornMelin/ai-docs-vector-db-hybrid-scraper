"""Enhanced circuit breaker with multi-level failure categorization.

This module provides advanced circuit breaker patterns with failure type categorization,
partial failure handling, and sophisticated recovery strategies.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Categories of failures for circuit breaker tracking."""

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    QUERY = "query"
    TRANSACTION = "transaction"
    RESOURCE = "resource"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class FailureMetrics:
    """Metrics for different failure types."""

    connection_failures: int = 0
    timeout_failures: int = 0
    query_failures: int = 0
    transaction_failures: int = 0
    resource_failures: int = 0

    last_failure_time: float | None = None
    total_requests: int = 0
    successful_requests: int = 0

    def get_failure_count(self, failure_type: FailureType) -> int:
        """Get failure count for specific type."""
        return getattr(self, f"{failure_type.value}_failures")

    def increment_failure(self, failure_type: FailureType) -> None:
        """Increment failure count for specific type."""
        current_count = self.get_failure_count(failure_type)
        setattr(self, f"{failure_type.value}_failures", current_count + 1)
        self.last_failure_time = time.time()

    def get_total_failures(self) -> int:
        """Get total failure count across all types."""
        return (
            self.connection_failures
            + self.timeout_failures
            + self.query_failures
            + self.transaction_failures
            + self.resource_failures
        )

    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure thresholds per type
    connection_threshold: int = 3
    timeout_threshold: int = 5
    query_threshold: int = 10
    transaction_threshold: int = 5
    resource_threshold: int = 3

    # Recovery settings
    recovery_timeout: float = 60.0
    half_open_max_requests: int = 3
    half_open_success_threshold: int = 2

    # Request timeout
    default_timeout: float | None = None

    # Failure rate threshold (alternative to count-based)
    failure_rate_threshold: float = 0.5
    min_requests_for_rate: int = 10


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, failure_type: FailureType):
        super().__init__(message)
        self.failure_type = failure_type


class MultiLevelCircuitBreaker:
    """Advanced circuit breaker with failure type categorization.

    This circuit breaker provides:
    - Separate failure tracking for different error types
    - Partial failure handlers for graceful degradation
    - Adaptive recovery strategies based on failure patterns
    - Comprehensive metrics and monitoring
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        """Initialize multi-level circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = FailureMetrics()

        # Partial failure handlers
        self.partial_failure_handlers: dict[FailureType, Callable] = {}
        self.fallback_handlers: dict[FailureType, Callable] = {}

        # Recovery tracking
        self._half_open_requests = 0
        self._half_open_successes = 0
        self._state_change_time = time.time()

        # Concurrency control
        self._lock = asyncio.Lock()

        # Performance tracking
        self._response_times: dict[FailureType, list] = {
            failure_type: [] for failure_type in FailureType
        }

    async def execute(
        self,
        func: Callable,
        failure_type: FailureType = FailureType.QUERY,
        fallback: Callable | None = None,
        timeout: float | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with multi-level circuit breaker protection.

        Args:
            func: Function to execute
            failure_type: Type of operation for failure categorization
            fallback: Optional fallback function
            timeout: Optional timeout override
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            CircuitBreakerOpenError: If circuit is open for this failure type
            Exception: Original function exceptions
        """
        async with self._lock:
            if await self._should_block_request(failure_type):
                # Try registered fallback first
                registered_fallback = self.fallback_handlers.get(failure_type)
                if registered_fallback:
                    logger.info(
                        f"Circuit breaker open for {failure_type.value}, using registered fallback"
                    )
                    return await registered_fallback(*args, **kwargs)

                # Try provided fallback
                if fallback:
                    logger.info(
                        f"Circuit breaker open for {failure_type.value}, using provided fallback"
                    )
                    return await fallback(*args, **kwargs)

                # No fallback available
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {failure_type.value} operations",
                    failure_type,
                )

        # Execute with timeout protection
        start_time = time.time()
        execution_timeout = timeout or self.config.default_timeout

        try:
            if execution_timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=execution_timeout
                )
            else:
                result = await func(*args, **kwargs)

            execution_time = time.time() - start_time
            await self._on_success(failure_type, execution_time)
            return result

        except TimeoutError as e:
            execution_time = time.time() - start_time
            await self._on_failure(FailureType.TIMEOUT, e, execution_time)

            # Try partial failure handler for timeouts
            timeout_handler = self.partial_failure_handlers.get(FailureType.TIMEOUT)
            if timeout_handler:
                logger.warning("Timeout occurred, trying partial failure handler")
                return await timeout_handler(e, *args, **kwargs)

            raise

        except Exception as e:
            execution_time = time.time() - start_time
            await self._on_failure(failure_type, e, execution_time)

            # Try partial failure handler
            partial_handler = self.partial_failure_handlers.get(failure_type)
            if partial_handler:
                logger.warning(
                    f"Failure occurred, trying partial failure handler for {failure_type.value}"
                )
                try:
                    return await partial_handler(e, *args, **kwargs)
                except Exception as handler_error:
                    logger.error(f"Partial failure handler failed: {handler_error}")

            raise

    def register_partial_failure_handler(
        self, failure_type: FailureType, handler: Callable[[Exception], Any]
    ) -> None:
        """Register handler for partial failure scenarios.

        Args:
            failure_type: Type of failure to handle
            handler: Handler function that receives the exception and original args
        """
        self.partial_failure_handlers[failure_type] = handler
        logger.info(f"Registered partial failure handler for {failure_type.value}")

    def register_fallback_handler(
        self, failure_type: FailureType, handler: Callable
    ) -> None:
        """Register fallback handler for when circuit is open.

        Args:
            failure_type: Type of failure to handle
            handler: Fallback function to execute when circuit is open
        """
        self.fallback_handlers[failure_type] = handler
        logger.info(f"Registered fallback handler for {failure_type.value}")

    async def _should_block_request(self, failure_type: FailureType) -> bool:
        """Determine if request should be blocked based on circuit state."""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return False

        elif self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            time_since_open = current_time - self._state_change_time
            if time_since_open >= self.config.recovery_timeout:
                await self._transition_to_half_open()
                return False
            return True

        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            if self._half_open_requests < self.config.half_open_max_requests:
                return False
            return True

        return False

    async def _on_success(
        self, failure_type: FailureType, execution_time: float
    ) -> None:
        """Handle successful request execution."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1

            # Track response time
            self._response_times[failure_type].append(execution_time)
            if len(self._response_times[failure_type]) > 100:
                self._response_times[failure_type].pop(0)

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1

                # Check if we should close the circuit
                if self._half_open_successes >= self.config.half_open_success_threshold:
                    await self._transition_to_closed()

            elif self.state == CircuitState.CLOSED:
                # Reset failure counters on successful requests
                await self._maybe_reset_failures()

    async def _on_failure(
        self, failure_type: FailureType, exception: Exception, execution_time: float
    ) -> None:
        """Handle failed request execution."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.increment_failure(failure_type)

            # Track response time even for failures
            self._response_times[failure_type].append(execution_time)
            if len(self._response_times[failure_type]) > 100:
                self._response_times[failure_type].pop(0)

            logger.warning(
                f"Circuit breaker failure: {failure_type.value} - {exception} "
                f"(total: {self.metrics.get_failure_count(failure_type)})"
            )

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                await self._transition_to_open()

            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if await self._should_open_circuit(failure_type):
                    await self._transition_to_open()

    async def _should_open_circuit(self, failure_type: FailureType) -> bool:
        """Determine if circuit should be opened based on failures."""
        # Check specific failure type threshold
        failure_count = self.metrics.get_failure_count(failure_type)
        threshold = getattr(self.config, f"{failure_type.value}_threshold")

        if failure_count >= threshold:
            return True

        # Check overall failure rate if we have enough requests
        if self.metrics.total_requests >= self.config.min_requests_for_rate:
            success_rate = self.metrics.get_success_rate()
            if success_rate < (1.0 - self.config.failure_rate_threshold):
                return True

        return False

    async def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self._state_change_time = time.time()
        self._half_open_requests = 0
        self._half_open_successes = 0

        logger.error(
            f"Circuit breaker opened. Total failures: {self.metrics.get_total_failures()}, "
            f"Success rate: {self.metrics.get_success_rate():.2%}"
        )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self._state_change_time = time.time()
        self._half_open_requests = 0
        self._half_open_successes = 0

        logger.info("Circuit breaker transitioning to half-open state")

    async def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self._state_change_time = time.time()

        # Reset failure metrics
        self.metrics = FailureMetrics()

        logger.info("Circuit breaker closed - service recovered")

    async def _maybe_reset_failures(self) -> None:
        """Reset failure counters periodically on successful requests."""
        # Reset every 100 successful requests or after 10 minutes
        current_time = time.time()
        time_since_last_failure = (
            current_time - self.metrics.last_failure_time
            if self.metrics.last_failure_time
            else float("inf")
        )

        should_reset = (
            self.metrics.successful_requests % 100 == 0
            or time_since_last_failure > 600  # 10 minutes
        )

        if should_reset and self.metrics.get_total_failures() > 0:
            # Partial reset - reduce failure counts but don't zero them
            reduction_factor = 0.5
            self.metrics.connection_failures = int(
                self.metrics.connection_failures * reduction_factor
            )
            self.metrics.timeout_failures = int(
                self.metrics.timeout_failures * reduction_factor
            )
            self.metrics.query_failures = int(
                self.metrics.query_failures * reduction_factor
            )
            self.metrics.transaction_failures = int(
                self.metrics.transaction_failures * reduction_factor
            )
            self.metrics.resource_failures = int(
                self.metrics.resource_failures * reduction_factor
            )

            logger.debug("Partially reset circuit breaker failure counters")

    async def force_open(self) -> None:
        """Force circuit breaker to open state."""
        async with self._lock:
            await self._transition_to_open()
            logger.warning("Circuit breaker forced to open state")

    async def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to_closed()
            logger.info("Circuit breaker forced to closed state")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status and metrics."""
        return {
            "state": self.state.value,
            "failure_metrics": {
                "connection_failures": self.metrics.connection_failures,
                "timeout_failures": self.metrics.timeout_failures,
                "query_failures": self.metrics.query_failures,
                "transaction_failures": self.metrics.transaction_failures,
                "resource_failures": self.metrics.resource_failures,
                "total_failures": self.metrics.get_total_failures(),
            },
            "request_metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "success_rate": self.metrics.get_success_rate(),
            },
            "state_info": {
                "state_duration_seconds": time.time() - self._state_change_time,
                "half_open_requests": self._half_open_requests,
                "half_open_successes": self._half_open_successes,
            },
            "response_time_stats": self._get_response_time_stats(),
            "configuration": {
                "connection_threshold": self.config.connection_threshold,
                "timeout_threshold": self.config.timeout_threshold,
                "query_threshold": self.config.query_threshold,
                "transaction_threshold": self.config.transaction_threshold,
                "resource_threshold": self.config.resource_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "failure_rate_threshold": self.config.failure_rate_threshold,
            },
        }

    def _get_response_time_stats(self) -> dict[str, dict[str, float]]:
        """Get response time statistics by failure type."""
        stats = {}

        for failure_type, times in self._response_times.items():
            if times:
                import numpy as np

                stats[failure_type.value] = {
                    "avg_ms": np.mean(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "p95_ms": np.percentile(times, 95) * 1000,
                    "sample_count": len(times),
                }
            else:
                stats[failure_type.value] = {
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p95_ms": 0.0,
                    "sample_count": 0,
                }

        return stats

    async def get_failure_analysis(self) -> dict[str, Any]:
        """Get detailed failure analysis and recommendations."""
        total_failures = self.metrics.get_total_failures()

        if total_failures == 0:
            return {
                "status": "healthy",
                "recommendations": ["System is operating normally"],
                "primary_failure_types": [],
            }

        # Analyze failure patterns
        failure_breakdown = {
            FailureType.CONNECTION: self.metrics.connection_failures,
            FailureType.TIMEOUT: self.metrics.timeout_failures,
            FailureType.QUERY: self.metrics.query_failures,
            FailureType.TRANSACTION: self.metrics.transaction_failures,
            FailureType.RESOURCE: self.metrics.resource_failures,
        }

        # Sort by failure count
        sorted_failures = sorted(
            failure_breakdown.items(), key=lambda x: x[1], reverse=True
        )

        primary_failures = [
            failure_type.value for failure_type, count in sorted_failures if count > 0
        ][:3]  # Top 3 failure types

        # Generate recommendations
        recommendations = []

        if self.metrics.connection_failures > 0:
            recommendations.append(
                "Check database connectivity and connection pool settings"
            )

        if self.metrics.timeout_failures > 0:
            recommendations.append(
                "Review query performance and increase timeout thresholds"
            )

        if self.metrics.query_failures > 0:
            recommendations.append("Analyze query patterns and optimize slow queries")

        if self.metrics.transaction_failures > 0:
            recommendations.append(
                "Review transaction isolation levels and deadlock handling"
            )

        if self.metrics.resource_failures > 0:
            recommendations.append("Monitor system resources (CPU, memory, disk)")

        success_rate = self.metrics.get_success_rate()
        if success_rate < 0.9:
            recommendations.append(
                f"Overall success rate is low ({success_rate:.1%}) - investigate system health"
            )

        return {
            "status": "degraded" if success_rate < 0.95 else "warning",
            "success_rate": success_rate,
            "total_failures": total_failures,
            "primary_failure_types": primary_failures,
            "failure_breakdown": {k.value: v for k, v in failure_breakdown.items()},
            "recommendations": recommendations,
            "circuit_state": self.state.value,
        }
