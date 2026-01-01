"""Tests for security middleware behaviours."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from limits import parse as parse_rate_limit
from slowapi.errors import RateLimitExceeded  # type: ignore[import-not-found]
from slowapi.middleware import SlowAPIMiddleware  # type: ignore[import-not-found]
from slowapi.wrappers import Limit  # type: ignore[import-not-found]
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from src.config.security.config import SecurityConfig
from src.services.fastapi.middleware.security import (
    SecurityMiddleware,
    _rate_limited,
    _resolve_storage_uri,
    enable_global_rate_limit,
)


@pytest.fixture(name="app")
def fixture_app() -> FastAPI:
    """Return a FastAPI application wired with a simple health endpoint."""
    application = FastAPI()

    @application.get("/health")
    def health_check() -> PlainTextResponse:
        """Return a plain text response indicating the application is healthy."""
        return PlainTextResponse("ok")

    return application


def test_security_middleware_injects_headers(app: FastAPI) -> None:
    """SecurityMiddleware applies default and custom headers."""
    app.add_middleware(SecurityMiddleware, extra_headers={"X-Test-Header": "1"})

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Test-Header"] == "1"


def test_enable_global_rate_limit_disabled(app: FastAPI) -> None:
    """When disabled, rate limiting returns None and clears state."""
    config = SecurityConfig(enable_rate_limiting=False)
    limiter = enable_global_rate_limit(app, config=config)

    assert limiter is None
    assert getattr(app.state, "limiter", None) is None


def test_enable_global_rate_limit_enabled(app: FastAPI) -> None:
    """When enabled, rate limiting attaches middleware and limiter state."""
    config = SecurityConfig(
        enable_rate_limiting=True,
        default_rate_limit=5,
        rate_limit_window=60,
    )
    limiter = enable_global_rate_limit(app, config=config)

    assert limiter is not None
    assert app.state.limiter is limiter
    middleware_classes = {middleware.cls for middleware in app.user_middleware}
    assert SlowAPIMiddleware in middleware_classes


def test_resolve_storage_uri_with_password() -> None:
    """Redis passwords are injected into the Redis URI when configured."""
    credential = "dummy-token"
    config = SecurityConfig(
        redis_url="redis://redis.example.com:6379/0",
        redis_password=credential,
    )

    uri = _resolve_storage_uri(config)
    assert uri == "redis://:dummy-token@redis.example.com:6379/0"


def test_rate_limited_handler_formats_response() -> None:
    """_rate_limited converts SlowAPI exceptions into JSON responses."""
    limit = Limit(
        limit=parse_rate_limit("5/minute"),
        key_func=lambda *_args, **_kwargs: "127.0.0.1",
        scope="global",
        per_method=False,
        methods=None,
        error_message=None,
        exempt_when=None,
        cost=1,
        override_defaults=False,
    )
    exc = RateLimitExceeded(limit)
    response = asyncio.run(_rate_limited(cast(Request, object()), exc))

    assert response.status_code == 429
    body = bytes(response.body)
    payload = json.loads(body.decode("utf-8"))
    assert payload == {"error": "Rate limit exceeded"}
