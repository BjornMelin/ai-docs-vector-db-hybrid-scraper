"""
Timeout and circuit breaker middleware.

- AnyIO-based request timeout.
- Circuit breaker via `purgatory` to prevent cascades.

References:
- Purgatory (async-friendly, pluggable storage).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import anyio
from purgatory import AsyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from purgatory.service._async.circuitbreaker import AsyncCircuitBreaker


class CircuitState(str, Enum):
    """Expose states without leaking library types."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    """Timeout + breaker configuration."""

    enabled: bool = True
    request_timeout: float = 30.0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Apply a per-request timeout and route through circuit breaker."""

    def __init__(self, app: Callable, *, config: TimeoutConfig) -> None:
        super().__init__(app)
        self._cfg = config
        self._factory = AsyncCircuitBreakerFactory(
            default_threshold=config.failure_threshold,
            default_ttl=config.recovery_timeout,
        )
        self._breaker: AsyncCircuitBreaker | None = None
        self._breaker_name = "fastapi_request_timeout"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._cfg.enabled:
            return await call_next(request)

        breaker = await self._ensure_breaker()

        # Reject immediately if breaker is open.
        if breaker.context.state == "opened":
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable", "circuit": "open"},
            )

        async def _guarded() -> Response:
            return await call_next(request)

        try:
            # Half-open allows limited probes internally.
            async with breaker:
                with anyio.fail_after(self._cfg.request_timeout):
                    return await _guarded()
        except TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "timeout": self._cfg.request_timeout,
                },
            )
        except OpenedState:
            return JSONResponse(
                status_code=503,
                content={"error": "Circuit rejecting requests", "circuit": "open"},
            )

    async def _ensure_breaker(self) -> AsyncCircuitBreaker:
        if self._breaker is None:
            self._breaker = await self._factory.get_breaker(
                self._breaker_name,
                threshold=self._cfg.failure_threshold,
                ttl=self._cfg.recovery_timeout,
            )
        return self._breaker


class BulkheadMiddleware(BaseHTTPMiddleware):
    """Per-endpoint concurrency limiter using AnyIO capacity limiters."""

    def __init__(self, app: Callable, *, max_concurrent: int = 10) -> None:
        super().__init__(app)
        self._max = max_concurrent
        self._locks: dict[str, anyio.CapacityLimiter] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        key = f"{request.method}:{request.url.path}"
        limiter = self._locks.setdefault(key, anyio.CapacityLimiter(self._max))
        async with limiter:
            return await call_next(request)


__all__ = ["TimeoutMiddleware", "BulkheadMiddleware", "CircuitState", "TimeoutConfig"]
