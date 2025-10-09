"""Tests for timeout middleware behavior."""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.services.fastapi.middleware import timeout
from src.services.fastapi.middleware.timeout import (
    BulkheadMiddleware,
    TimeoutConfig,
    TimeoutMiddleware,
)


class DummyContext:
    def __init__(self, state: str = "closed") -> None:
        """Initialize dummy context with state."""

        self.state = state


class DummyBreaker:
    def __init__(self, state: str = "closed") -> None:
        """Initialize dummy breaker with state."""

        self.context = DummyContext(state)

    async def __aenter__(self) -> DummyBreaker:
        """Enter async context."""

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple
        """Exit async context."""

        return None


class DummyManager:
    """In-memory manager returning a shared dummy breaker."""

    def __init__(self) -> None:
        self.breaker = DummyBreaker()

    async def get_breaker(self, *_args, **_kwargs) -> DummyBreaker:
        """Return the shared dummy breaker."""

        return self.breaker


@pytest.fixture(name="patched_manager")
def _patched_manager(monkeypatch: pytest.MonkeyPatch) -> DummyManager:
    """Fixture to patch shared circuit breaker manager with dummy implementation."""

    manager = DummyManager()

    async def _get_manager() -> DummyManager:
        return manager

    monkeypatch.setattr(timeout, "PURGATORY_AVAILABLE", True)
    monkeypatch.setattr(
        "src.services.circuit_breaker.provider.get_circuit_breaker_manager",
        _get_manager,
    )
    return manager


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
async def test_timeout_middleware_passes_through(patched_manager: DummyManager) -> None:
    """Test that timeout middleware passes through when circuit closed."""

    middleware = TimeoutMiddleware(lambda *_: None, config=TimeoutConfig())

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_timeout_middleware_returns_open_response(
    patched_manager: DummyManager,
) -> None:
    """Test that timeout middleware returns open response when circuit open."""

    middleware = TimeoutMiddleware(lambda *_: None, config=TimeoutConfig())
    patched_manager.breaker.context.state = "opened"

    async def call_next(_request: Request) -> Response:
        raise AssertionError("call_next should not run")

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 503
    payload = json.loads(bytes(response.body))
    assert payload["circuit"] == "open"


@pytest.mark.anyio
async def test_timeout_middleware_times_out(patched_manager: DummyManager) -> None:
    """Test that timeout middleware times out requests."""

    middleware = TimeoutMiddleware(
        lambda *_: None, config=TimeoutConfig(request_timeout=0.1)
    )

    async def call_next(_request: Request) -> Response:
        raise TimeoutError

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 504
    payload = json.loads(bytes(response.body))
    assert payload["error"] == "Request timeout"


@pytest.mark.anyio
async def test_timeout_middleware_disabled_returns_directly(
    patched_manager: DummyManager,
) -> None:
    """Test that disabled timeout middleware returns directly."""

    middleware = TimeoutMiddleware(lambda *_: None, config=TimeoutConfig(enabled=False))

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(_build_request(), call_next)
    assert response.status_code == 200


@pytest.mark.anyio
async def test_bulkhead_middleware_tracks_limiters() -> None:
    """Test that bulkhead middleware tracks limiters."""

    middleware = BulkheadMiddleware(lambda *_: None, max_concurrent=1)

    async def call_next(_request: Request) -> Response:
        return JSONResponse({"ok": True})

    request = _build_request()
    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    assert middleware._locks  # pylint: disable=protected-access
    assert "GET:/" in middleware._locks  # pylint: disable=protected-access
