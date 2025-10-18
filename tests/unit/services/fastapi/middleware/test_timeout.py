"""Tests for timeout middleware behavior."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.services.fastapi.middleware.timeout import (
    BulkheadMiddleware,
    TimeoutConfig,
    TimeoutMiddleware,
)


class CircuitBreakerManagerProtocol(Protocol):
    """Typed subset representing the breaker manager contract."""

    async def get_breaker(self, *args: Any, **kwargs: Any) -> Any:
        """Return an async context manager for breaker operations."""


if TYPE_CHECKING:  # pragma: no cover - typing hint only
    from src.services.circuit_breaker.circuit_breaker_manager import (
        CircuitBreakerManager as _RealCircuitBreakerManager,
    )

    CircuitBreakerManager = _RealCircuitBreakerManager
else:  # pragma: no cover - runtime fallback when optional dep missing
    CircuitBreakerManager = CircuitBreakerManagerProtocol


class DummyContext:
    """Minimal breaker context for testing."""

    def __init__(self, state: str = "closed") -> None:
        """Initialize breaker context with state."""
        self.state = state


class DummyBreaker:
    """Async-compatible circuit breaker double."""

    def __init__(self, state: str = "closed") -> None:
        """Initialize breaker with context."""
        self.context = DummyContext(state)

    async def __aenter__(self) -> DummyBreaker:
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit async context."""
        return


class DummyManager:
    """In-memory manager returning a shared dummy breaker."""

    def __init__(self) -> None:
        """Initialize with a single shared breaker."""
        self.breaker = DummyBreaker()

    async def get_breaker(self, *_args, **_kwargs) -> DummyBreaker:
        """Return the shared dummy breaker."""
        return self.breaker


@pytest.fixture(name="dummy_manager")
def _dummy_manager() -> DummyManager:
    """Fixture providing a reusable dummy circuit breaker manager."""
    return DummyManager()


def _build_request() -> Request:
    """Build a test request."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "client": ("127.0.0.1", 1234),
    }
    return Request(scope)


@pytest.mark.anyio
async def test_timeout_middleware_passes_through(dummy_manager: DummyManager) -> None:
    """Middleware should forward responses when the circuit is closed."""
    middleware = TimeoutMiddleware(
        lambda *_: None,
        config=TimeoutConfig(),
        manager=cast("CircuitBreakerManager", dummy_manager),
    )

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 200
    assert isinstance(response, JSONResponse)
    payload = json.loads(bytes(response.body))
    assert payload == {"ok": True}


@pytest.mark.anyio
async def test_timeout_middleware_returns_open_response(
    dummy_manager: DummyManager,
) -> None:
    """Middleware should short-circuit when breaker reports open."""
    middleware = TimeoutMiddleware(
        lambda *_: None,
        config=TimeoutConfig(),
        manager=cast("CircuitBreakerManager", dummy_manager),
    )
    dummy_manager.breaker.context.state = "opened"

    async def call_next(_request: Request) -> Response:
        raise AssertionError("call_next should not run")

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 503
    payload = json.loads(bytes(response.body))
    assert payload["circuit"] == "open"


@pytest.mark.anyio
async def test_timeout_middleware_times_out(dummy_manager: DummyManager) -> None:
    """Middleware should emit timeout response on TimeoutError."""
    middleware = TimeoutMiddleware(
        lambda *_: None,
        config=TimeoutConfig(request_timeout=0.1),
        manager=cast("CircuitBreakerManager", dummy_manager),
    )

    async def call_next(_request: Request) -> Response:
        raise TimeoutError

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 504
    payload = json.loads(bytes(response.body))
    assert payload["error"] == "Request timeout"


@pytest.mark.anyio
async def test_timeout_middleware_timeout_only_fallback() -> None:
    """Middleware should still enforce timeout when no manager is injected."""
    middleware = TimeoutMiddleware(
        lambda *_: None, config=TimeoutConfig(request_timeout=0.01)
    )

    async def call_next(_request: Request) -> Response:
        await anyio.sleep(0.05)
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 504
    payload = json.loads(bytes(response.body))
    assert payload["error"] == "Request timeout"


@pytest.mark.anyio
async def test_timeout_middleware_disabled_returns_directly(
    dummy_manager: DummyManager,
) -> None:
    """Middleware should bypass timeout entirely when disabled."""
    middleware = TimeoutMiddleware(
        lambda *_: None,
        config=TimeoutConfig(enabled=False),
        manager=cast("CircuitBreakerManager", dummy_manager),
    )

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_timeout_middleware_resolver_used_once(
    dummy_manager: DummyManager,
) -> None:
    """Resolver should only execute until cached manager is initialized."""
    calls = 0

    async def _resolver() -> CircuitBreakerManager:
        nonlocal calls
        calls += 1
        return cast("CircuitBreakerManager", dummy_manager)

    middleware = TimeoutMiddleware(
        lambda *_: None,
        config=TimeoutConfig(),
        manager_resolver=_resolver,
    )

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    await middleware.dispatch(_build_request(), call_next)
    await middleware.dispatch(_build_request(), call_next)

    assert calls == 1


@pytest.mark.anyio
async def test_bulkhead_middleware_enforces_concurrency_limit() -> None:
    """Bulkhead middleware should cap concurrent requests per endpoint."""
    middleware = BulkheadMiddleware(lambda *_: None, max_concurrent=1)

    active = 0
    peak = 0

    async def call_next(_request: Request) -> Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await anyio.sleep(0)
        active -= 1
        return JSONResponse({"ok": True})

    async def execute_request() -> None:
        response = await middleware.dispatch(_build_request(), call_next)
        assert response.status_code == 200

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(execute_request)
        task_group.start_soon(execute_request)

    assert peak == 1
