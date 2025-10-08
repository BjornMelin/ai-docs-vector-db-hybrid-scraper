"""Tests for security middleware and rate limiting helpers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.services.fastapi.middleware.security import (
    SecurityMiddleware,
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
    limiter = enable_global_rate_limit(app, default_limit="1/minute")

    @app.get("/limited")
    @limiter.limit("1/minute")
    async def _limited(request: Request) -> dict[str, str]:  # noqa: ARG001
        return {"ok": "true"}

    client = TestClient(app)
    ok_response = client.get("/limited")
    assert ok_response.status_code == 200

    limited_response = client.get("/limited")
    assert limited_response.status_code == 429
    assert limited_response.json()["error"] == "Rate limit exceeded"
