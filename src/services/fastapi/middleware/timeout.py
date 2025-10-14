"""Timeout and circuit breaker middleware."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import anyio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


try:
    from purgatory.domain.model import OpenedState  # type: ignore[import]
except ModuleNotFoundError:
    OpenedState = type(
        "OpenedState",
        (Exception,),
        {"__doc__": "Fallback OpenedState when purgatory is unavailable."},
    )


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from purgatory.service._async.circuitbreaker import AsyncCircuitBreaker  # type: ignore  # noqa: I001
    from src.services.circuit_breaker.circuit_breaker_manager import (
        CircuitBreakerManager,
    )
else:  # pragma: no cover - runtime degrade when purgatory absent
    AsyncCircuitBreaker = object  # type: ignore[assignment]
    CircuitBreakerManager = object  # type: ignore[assignment]


class CircuitBreakerResolver(Protocol):
    """Protocol for async factory returning a circuit breaker manager."""

    async def __call__(self) -> CircuitBreakerManager:  # pragma: no cover - typing only
        ...


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


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Apply a per-request timeout and route through circuit breaker."""

    def __init__(
        self,
        app: Callable,
        *,
        config: TimeoutConfig,
        manager: CircuitBreakerManager | None = None,
        manager_resolver: CircuitBreakerResolver | None = None,
    ) -> None:
        super().__init__(app)
        self._cfg = config
        self._manager = manager
        self._manager_resolver = manager_resolver
        self._breaker: AsyncCircuitBreaker | None = None
        self._breaker_lock = asyncio.Lock()
        self._manager_lock = asyncio.Lock()
        self._breaker_name = "fastapi_request_timeout"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Execute the request within timeout and circuit breaker constraints."""
        if not self._cfg.enabled:
            return await call_next(request)

        manager = await self._resolve_manager()
        if manager is None:
            return await self._dispatch_without_breaker(request, call_next)

        return await self._dispatch_with_breaker(manager, request, call_next)

    async def _resolve_manager(self) -> CircuitBreakerManager | None:
        """Return an injected manager or lazily resolve via resolver."""
        if self._manager is not None:
            return self._manager

        if self._manager_resolver is None:
            return None

        async with self._manager_lock:
            if self._manager is None:
                self._manager = await self._manager_resolver()
        return self._manager

    async def _ensure_breaker(
        self, manager: CircuitBreakerManager
    ) -> AsyncCircuitBreaker:
        """Return the cached circuit breaker instance for FastAPI requests."""
        if self._breaker is not None:
            return self._breaker

        async with self._breaker_lock:
            if self._breaker is None:
                self._breaker = await manager.get_breaker(
                    self._breaker_name,
                    threshold=self._cfg.failure_threshold,
                    ttl=self._cfg.recovery_timeout,
                )
        if self._breaker is None:  # pragma: no cover - defensive
            msg = "Failed to initialize circuit breaker."
            raise RuntimeError(msg)
        return self._breaker

    async def _dispatch_without_breaker(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Fallback path enforcing only the request timeout."""
        response: Response | None = None
        with anyio.move_on_after(self._cfg.request_timeout) as scope:
            response = await call_next(request)
        if scope.cancel_called:
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "timeout": self._cfg.request_timeout,
                },
            )
        if response is None:  # pragma: no cover - defensive
            raise RuntimeError("Timeout middleware returned no response")
        return response

    async def _dispatch_with_breaker(
        self,
        manager: CircuitBreakerManager,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Route the request through the circuit breaker when available."""
        breaker = await self._ensure_breaker(manager)

        if breaker.context.state == "opened":
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable", "circuit": "open"},
            )

        async def _guarded() -> Response:
            return await call_next(request)

        try:
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


__all__ = ["BulkheadMiddleware", "CircuitState", "TimeoutConfig", "TimeoutMiddleware"]
