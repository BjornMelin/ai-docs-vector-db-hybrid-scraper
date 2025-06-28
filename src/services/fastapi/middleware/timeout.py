"""Timeout and circuit breaker middleware for production resilience.

This middleware provides request timeouts and circuit breaker patterns
to prevent cascading failures and improve system resilience.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class TimeoutConfig:
    """Timeout middleware configuration."""

    enabled: bool = True
    request_timeout: float = 30.0
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    state: CircuitState = CircuitState.CLOSED
    half_open_calls: int = 0


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Request timeout middleware with circuit breaker pattern.

    Features:
    - Configurable request timeouts
    - Circuit breaker for endpoint resilience
    - Graceful timeout handling
    - Circuit state monitoring and recovery
    """

    def __init__(self, app: Callable, config: TimeoutConfig):
        """Initialize timeout middleware.

        Args:
            app: ASGI application
            config: Timeout configuration

        """
        super().__init__(app)
        self.config = config

        # Circuit breaker state per endpoint
        self._circuit_stats: dict[str, CircuitBreakerStats] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with timeout and circuit breaker logic."""
        if not self.config.enabled:
            return await call_next(request)

        endpoint = self._get_endpoint_key(request)

        # Check circuit breaker state
        if self.config.enable_circuit_breaker:
            circuit_response = self._check_circuit_breaker(endpoint)
            if circuit_response:
                return circuit_response

        # Execute request with timeout
        try:
            response = await self._execute_with_timeout(request, call_next)

            # Record success for circuit breaker
            if self.config.enable_circuit_breaker:
                self._record_success(endpoint)

            return response

        except TimeoutError:
            # Handle timeout
            logger.warning(
                "Request timeout",
                extra={
                    "endpoint": endpoint,
                    "timeout": self.config.request_timeout,
                    "method": request.method,
                    "path": request.url.path,
                },
            )

            # Record failure for circuit breaker
            if self.config.enable_circuit_breaker:
                self._record_failure(endpoint)

            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "timeout": self.config.request_timeout,
                },
            )

        except Exception:
            # Record failure for circuit breaker
            if self.config.enable_circuit_breaker:
                self._record_failure(endpoint)

            # Re-raise the exception
            raise

    async def _execute_with_timeout(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Execute request with timeout protection.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response

        Raises:
            asyncio.TimeoutError: If request times out

        """
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=self.config.request_timeout
            )
        except TimeoutError:
            # Log timeout details
            logger.warning(
                "Request execution timeout",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "timeout": self.config.request_timeout,
                },
            )
            raise

    def _get_endpoint_key(self, request: Request) -> str:
        """Generate endpoint key for circuit breaker tracking.

        Args:
            request: HTTP request

        Returns:
            Endpoint identifier string

        """
        # Use method + path pattern for grouping
        return f"{request.method}:{request.url.path}"

    def _check_circuit_breaker(self, endpoint: str) -> Response | None:
        """Check circuit breaker state and return early response if needed.

        Args:
            endpoint: Endpoint identifier

        Returns:
            Response if circuit is open, None if request should proceed

        """
        stats = self._circuit_stats.get(endpoint, CircuitBreakerStats())
        current_time = time.time()

        if stats.state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            if current_time - stats.last_failure_time >= self.config.recovery_timeout:
                stats.state = CircuitState.HALF_OPEN
                stats.half_open_calls = 0
                logger.info(f"Circuit breaker entering HALF_OPEN state for {endpoint}")
            else:
                # Circuit is still open, reject request
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Service temporarily unavailable",
                        "circuit_state": "open",
                    },
                )

        elif stats.state == CircuitState.HALF_OPEN:
            # Limit calls in half-open state
            if stats.half_open_calls >= self.config.half_open_max_calls:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Service testing recovery",
                        "circuit_state": "half_open",
                    },
                )
            stats.half_open_calls += 1

        # Store updated stats
        self._circuit_stats[endpoint] = stats
        return None

    def _record_success(self, endpoint: str) -> None:
        """Record successful request for circuit breaker.

        Args:
            endpoint: Endpoint identifier

        """
        stats = self._circuit_stats.get(endpoint, CircuitBreakerStats())
        stats.success_count += 1
        stats.last_success_time = time.time()

        if stats.state == CircuitState.HALF_OPEN:
            # Successful call in half-open state - close the circuit
            stats.state = CircuitState.CLOSED
            stats.failure_count = 0
            stats.half_open_calls = 0
            logger.info(
                f"Circuit breaker CLOSED for {endpoint} after successful recovery"
            )

        elif stats.state == CircuitState.CLOSED:
            # Reset failure count on success
            stats.failure_count = max(0, stats.failure_count - 1)

        self._circuit_stats[endpoint] = stats

    def _record_failure(self, endpoint: str) -> None:
        """Record failed request for circuit breaker.

        Args:
            endpoint: Endpoint identifier

        """
        stats = self._circuit_stats.get(endpoint, CircuitBreakerStats())
        stats.failure_count += 1
        stats.last_failure_time = time.time()

        if stats.state == CircuitState.HALF_OPEN:
            # Failure in half-open state - open the circuit again
            stats.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker re-OPENED for {endpoint} after failure in half-open state"
            )

        elif stats.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if stats.failure_count >= self.config.failure_threshold:
                stats.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPENED for {endpoint} after {stats.failure_count} failures"
                )

        self._circuit_stats[endpoint] = stats

    def get_circuit_stats(self) -> dict[str, dict]:
        """Get current circuit breaker statistics for monitoring.

        Returns:
            Dictionary of endpoint statistics

        """
        return {
            endpoint: {
                "state": stats.state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "last_failure_time": stats.last_failure_time,
                "last_success_time": stats.last_success_time,
                "half_open_calls": stats.half_open_calls,
            }
            for endpoint, stats in self._circuit_stats.items()
        }

    def reset_circuit(self, endpoint: str) -> bool:
        """Manually reset circuit breaker for an endpoint.

        Args:
            endpoint: Endpoint identifier

        Returns:
            True if circuit was reset, False if endpoint not found

        """
        if endpoint in self._circuit_stats:
            self._circuit_stats[endpoint] = CircuitBreakerStats()
            logger.info(f"Manually reset circuit breaker for {endpoint}")
            return True
        return False


class BulkheadMiddleware(BaseHTTPMiddleware):
    """Bulkhead pattern middleware for resource isolation.

    Simple implementation that limits concurrent requests per endpoint
    to prevent resource exhaustion.
    """

    def __init__(self, app: Callable, max_concurrent: int = 10):
        """Initialize bulkhead middleware.

        Args:
            app: ASGI application
            max_concurrent: Maximum concurrent requests per endpoint

        """
        super().__init__(app)
        self.max_concurrent = max_concurrent
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with concurrency limiting."""
        endpoint = f"{request.method}:{request.url.path}"

        # Get or create semaphore for this endpoint
        if endpoint not in self._semaphores:
            self._semaphores[endpoint] = asyncio.Semaphore(self.max_concurrent)

        semaphore = self._semaphores[endpoint]

        # Try to acquire semaphore
        if semaphore.locked():
            # Too many concurrent requests
            return JSONResponse(
                status_code=503,
                content={"error": "Too many concurrent requests for this endpoint"},
            )

        async with semaphore:
            return await call_next(request)


# Export middleware classes
__all__ = ["BulkheadMiddleware", "CircuitState", "TimeoutMiddleware"]
