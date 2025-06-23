
"""Circuit breaker implementation for resilient service patterns.

Provides configurable circuit breaker patterns with simple/enterprise modes.
Based on modern async patterns with FastAPI integration.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from functools import wraps
from typing import Any
from typing import TypeVar

from fastapi import Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Supports both simple and enterprise modes based on environment.
    """

    # Core settings
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    test_request_timeout: int = 30  # seconds

    # Enterprise features (enabled based on environment)
    enable_metrics: bool = True
    enable_fallback: bool = True
    enable_adaptive_timeout: bool = False

    # Exception handling
    monitored_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )

    # Metrics
    failure_rate_threshold: float = 0.5  # 50% failure rate
    min_requests_for_rate: int = 10

    @classmethod
    def simple_mode(cls) -> "CircuitBreakerConfig":
        """Create simple circuit breaker configuration (50 lines equivalent)."""
        return cls(
            failure_threshold=3,
            recovery_timeout=30,
            test_request_timeout=10,
            enable_metrics=False,
            enable_fallback=False,
            enable_adaptive_timeout=False,
        )

    @classmethod
    def enterprise_mode(cls) -> "CircuitBreakerConfig":
        """Create enterprise circuit breaker configuration with advanced features."""
        return cls(
            failure_threshold=5,
            recovery_timeout=60,
            test_request_timeout=30,
            enable_metrics=True,
            enable_fallback=True,
            enable_adaptive_timeout=True,
            min_requests_for_rate=20,
            failure_rate_threshold=0.4,  # More sensitive
        )


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics tracking."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    state_changes: int = 0
    last_failure_time: float = 0
    average_response_time: float = 0

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class CircuitBreaker:
    """Async circuit breaker with configurable complexity modes."""

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_test_time = 0
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        logger.info(
            f"Circuit breaker initialized: {config.failure_threshold} threshold, "
            f"{config.recovery_timeout}s recovery"
        )

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails and circuit allows
        """
        async with self._lock:
            # Check if circuit should be opened based on metrics
            if self.config.enable_metrics and self._should_open_circuit():
                await self._open_circuit()

            # Handle different states
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    self.metrics.blocked_requests += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Retry after "
                        f"{self.config.recovery_timeout - (time.time() - self.last_failure_time):.0f}s"
                    )

            elif self.state == CircuitBreakerState.HALF_OPEN:
                if time.time() - self.last_test_time > self.config.test_request_timeout:
                    await self._open_circuit()
                    self.metrics.blocked_requests += 1
                    raise CircuitBreakerError("Circuit breaker test timeout")

        # Execute function
        start_time = time.time()
        try:
            self.metrics.total_requests += 1
            result = await func(*args, **kwargs)

            # Success - reset failure count and close circuit if needed
            execution_time = time.time() - start_time
            await self._record_success(execution_time)

            return result

        except self.config.monitored_exceptions:
            execution_time = time.time() - start_time
            await self._record_failure(execution_time)
            raise

    async def _record_success(self, execution_time: float) -> None:
        """Record successful execution."""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.failure_count = 0

            # Update average response time
            if self.config.enable_metrics:
                total_time = (
                    self.metrics.average_response_time
                    * (self.metrics.total_requests - 1)
                    + execution_time
                )
                self.metrics.average_response_time = (
                    total_time / self.metrics.total_requests
                )

            # Close circuit if in half-open state
            if self.state == CircuitBreakerState.HALF_OPEN:
                await self._close_circuit()

    async def _record_failure(self, execution_time: float) -> None:
        """Record failed execution."""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.metrics.last_failure_time = self.last_failure_time

            # Update average response time
            if self.config.enable_metrics:
                total_time = (
                    self.metrics.average_response_time
                    * (self.metrics.total_requests - 1)
                    + execution_time
                )
                self.metrics.average_response_time = (
                    total_time / self.metrics.total_requests
                )

            # Check if circuit should open
            if self.failure_count >= self.config.failure_threshold or (
                self.config.enable_metrics and self._should_open_circuit()
            ):
                await self._open_circuit()

    def _should_open_circuit(self) -> bool:
        """Check if circuit should open based on failure rate."""
        if not self.config.enable_metrics:
            return False

        if self.metrics.total_requests < self.config.min_requests_for_rate:
            return False

        return self.metrics.failure_rate >= self.config.failure_rate_threshold

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    async def _open_circuit(self) -> None:
        """Transition to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.metrics.state_changes += 1
            logger.warning(
                f"Circuit breaker OPEN: {self.failure_count} failures, "
                f"failure_rate={self.metrics.failure_rate:.2%}"
            )

    async def _close_circuit(self) -> None:
        """Transition to CLOSED state."""
        if self.state != CircuitBreakerState.CLOSED:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.metrics.state_changes += 1
            logger.info(f"Circuit breaker CLOSED: recovered from {old_state.value}")

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_test_time = time.time()
        self.metrics.state_changes += 1
        logger.info("Circuit breaker HALF_OPEN: testing recovery")

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics.

        Returns:
            Dictionary with current metrics
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "failure_rate": self.metrics.failure_rate,
            "success_rate": self.metrics.success_rate,
            "average_response_time": self.metrics.average_response_time,
            "state_changes": self.metrics.state_changes,
        }


class CircuitBreakerError(Exception):
    """Circuit breaker specific exception."""

    pass


def circuit_breaker(config: CircuitBreakerConfig | None = None):
    """Decorator for circuit breaker functionality.

    Args:
        config: Circuit breaker configuration

    Returns:
        Decorated function with circuit breaker protection
    """
    if config is None:
        config = CircuitBreakerConfig.simple_mode()

    breaker = CircuitBreaker(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(func, *args, **kwargs)

        # Attach circuit breaker for metrics access
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


def create_circuit_breaker(mode: str = "simple", **kwargs: Any) -> CircuitBreaker:
    """Factory function for creating circuit breakers.

    Args:
        mode: "simple" or "enterprise"
        **kwargs: Additional configuration overrides

    Returns:
        Configured CircuitBreaker instance
    """
    if mode == "simple":
        config = CircuitBreakerConfig.simple_mode()
    elif mode == "enterprise":
        config = CircuitBreakerConfig.enterprise_mode()
    else:
        config = CircuitBreakerConfig()

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return CircuitBreaker(config)


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for circuit breaker protection."""

    def __init__(self, app, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker middleware.

        Args:
            app: FastAPI application
            config: Circuit breaker configuration
        """
        super().__init__(app)
        self.config = config or CircuitBreakerConfig.simple_mode()
        self.circuit_breaker = CircuitBreaker(self.config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with circuit breaker protection.

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        try:
            return await self.circuit_breaker.call(call_next, request)
        except CircuitBreakerError as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=str(e))


# Convenience function for middleware setup
def circuit_breaker_middleware(app, mode: str = "simple", **kwargs: Any) -> None:
    """Add circuit breaker middleware to FastAPI app.

    Args:
        app: FastAPI application
        mode: "simple" or "enterprise"
        **kwargs: Additional configuration overrides
    """
    if mode == "simple":
        config = CircuitBreakerConfig.simple_mode()
    elif mode == "enterprise":
        config = CircuitBreakerConfig.enterprise_mode()
    else:
        config = CircuitBreakerConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    app.add_middleware(CircuitBreakerMiddleware, config=config)
