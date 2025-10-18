"""Tests for security middleware and rate limiting helpers."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request as StarletteRequest

from src.config.security import SecurityConfig
from src.services.fastapi.middleware.security import (
    SecurityMiddleware,
    _format_default_limit,
    _rate_limited,
    _resolve_storage_uri,
    enable_global_rate_limit,
)


def test_security_middleware_applies_default_headers() -> None:
    """Test that SecurityMiddleware applies default headers."""
    app = FastAPI()
    app.add_middleware(SecurityMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"ok": "true"}

    response = TestClient(app).get("/ping")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_security_middleware_respects_extra_headers() -> None:
    """Test that extra headers are added to defaults."""
    app = FastAPI()
    app.add_middleware(
        SecurityMiddleware,
        extra_headers={"Strict-Transport-Security": "max-age=3600"},
    )

    @app.get("/check")
    async def _check() -> dict[str, str]:
        return {"ok": "true"}

    response = TestClient(app).get("/check")

    assert response.headers["Strict-Transport-Security"] == "max-age=3600"


def test_enable_global_rate_limit_enforces_limits() -> None:
    """Test that rate limiting is enforced."""
    app = FastAPI()
    limiter = enable_global_rate_limit(
        app,
        config=SecurityConfig(default_rate_limit=1, rate_limit_window=60),
    )
    assert limiter is not None

    @app.get("/limited")
    @limiter.limit("1/minute")
    async def _limited(request: Request) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    ok_response = client.get("/limited")
    assert ok_response.status_code == 200

    limited_response = client.get("/limited")
    assert limited_response.status_code == 429
    assert limited_response.json()["error"] == "Rate limit exceeded"


def test_enable_global_rate_limit_respects_disabled_flag() -> None:
    """Rate limiting is skipped when disabled in configuration."""
    app = FastAPI()
    limiter = enable_global_rate_limit(
        app,
        config=SecurityConfig(enable_rate_limiting=False),
    )

    assert limiter is None
    assert getattr(app.state, "limiter", None) is None


def test_format_default_limit_humanizes_window() -> None:
    """Default limit formatter should select human readable granularity."""
    minute_config = SecurityConfig(default_rate_limit=5, rate_limit_window=60)
    assert _format_default_limit(minute_config) == "5/minute"

    custom_config = SecurityConfig(default_rate_limit=5, rate_limit_window=45)
    assert _format_default_limit(custom_config) == "5/45 second"


def test_resolve_storage_uri_handles_password_injection() -> None:
    """Redis credentials should be merged into URI when password provided."""
    base_config = SecurityConfig(redis_url="redis://localhost:6379/0")
    assert _resolve_storage_uri(base_config) == "redis://localhost:6379/0"

    credential = "example-value"
    password_config = SecurityConfig(
        redis_url="redis://user@localhost:6380/1",
        redis_password=credential,
    )
    assert (
        _resolve_storage_uri(password_config)
        == "redis://user:example-value@localhost:6380/1"
    )

    embedded_config = SecurityConfig(
        redis_url="redis://user:existing@localhost:6380/1",
        redis_password="".join(["ignore", "d"]),
    )
    assert (
        _resolve_storage_uri(embedded_config)
        == "redis://user:existing@localhost:6380/1"
    )


@pytest.mark.asyncio()
async def test_rate_limited_merges_headers() -> None:
    """Rate limit handler should merge retry headers into response."""
    exc = RateLimitExceeded.__new__(RateLimitExceeded)
    cast(Any, exc).retry_after = 7
    cast(Any, exc).headers = {"X-RateLimit-Remaining": "0"}

    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = StarletteRequest(scope)

    response = await _rate_limited(request, exc)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert json.loads(bytes(response.body)) == {"error": "Rate limit exceeded"}


@pytest.mark.asyncio()
async def test_rate_limited_rejects_unexpected_exception() -> None:
    """Non SlowAPI exceptions should raise informative errors."""
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = StarletteRequest(scope)

    with pytest.raises(TypeError):
        await _rate_limited(request, Exception("boom"))
