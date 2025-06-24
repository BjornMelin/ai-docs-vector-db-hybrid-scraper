"""Shared infrastructure components to avoid circular imports.

This module contains classes that are used by multiple infrastructure components
to prevent circular import dependencies.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from src.services.errors import APIError


logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Client connection state enumeration.

    Values:
        UNINITIALIZED: Client not yet initialized or connected
        HEALTHY: Client is connected and operating normally
        DEGRADED: Client is experiencing issues but partially functional
        FAILED: Client has failed and is not operational
    """

    UNINITIALIZED = "uninitialized"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class ClientHealth:
    """Client health status tracking.

    Attributes:
        state: Current connection state of the client
        last_check: Unix timestamp of last health check
        last_error: Description of the last error encountered, if any
        consecutive_failures: Number of consecutive failures for this client
    """

    state: ClientState
    last_check: float
    last_error: str | None = None
    consecutive_failures: int = 0


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = ClientState.HEALTHY
        self._half_open_attempts = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ClientState:
        """Get current circuit state.

        Returns:
            ClientState: Current state of the circuit breaker:
                - HEALTHY: Normal operation
                - DEGRADED: Half-open state, testing recovery
                - FAILED: Circuit open, rejecting requests
        """
        if (
            self._state == ClientState.FAILED
            and self._last_failure_time
            and time.time() - self._last_failure_time > self.recovery_timeout
        ):
            return ClientState.DEGRADED  # Half-open
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Any: Result from the executed function

        Raises:
            APIError: If circuit breaker is open or half-open test fails
            Exception: Any exception raised by the executed function
        """
        async with self._lock:
            current_state = self.state

            if current_state == ClientState.FAILED:
                raise APIError("Circuit breaker is open")

            if current_state == ClientState.DEGRADED:
                if self._half_open_attempts >= self.half_open_requests:
                    self._state = ClientState.FAILED
                    raise APIError("Circuit breaker is open (half-open test failed)")
                self._half_open_attempts += 1

        try:
            result = await func(*args, **kwargs)
            # Success - reset the circuit
            async with self._lock:
                self._failure_count = 0
                self._half_open_attempts = 0
                self._state = ClientState.HEALTHY
            return result

        except Exception:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()

                if self._failure_count >= self.failure_threshold:
                    self._state = ClientState.FAILED
                    logger.exception(
                        f"Circuit breaker opened after {self._failure_count} failures"
                    )

            raise
