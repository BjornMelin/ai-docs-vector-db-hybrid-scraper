"""Tests for the FastAPI security middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.security.config import SecurityConfig
from src.services.fastapi.middleware.security import SecurityMiddleware


def _build_test_app(config: SecurityConfig) -> TestClient:
    app = FastAPI()
    app.add_middleware(SecurityMiddleware, config=config)

    @app.get("/ping")
    async def ping() -> dict[str, bool]:  # pragma: no cover - exercised via client
        return {"ok": True}

    return TestClient(app)


def test_security_middleware_injects_headers() -> None:
    """Security middleware should inject the configured security headers."""

    client = _build_test_app(SecurityConfig(enable_rate_limiting=False))
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert "Strict-Transport-Security" in response.headers
    assert "Content-Security-Policy" in response.headers


def test_security_middleware_enforces_rate_limit() -> None:
    """Enabling rate limiting should throttle repeated client requests."""

    config = SecurityConfig(
        enable_rate_limiting=True,
        default_rate_limit=1,
        rate_limit_window=60,
    )
    client = _build_test_app(config)

    ok_response = client.get("/ping")
    assert ok_response.status_code == 200

    limited_response = client.get("/ping")
    assert limited_response.status_code == 429
    assert limited_response.json()["error"] == "Rate limit exceeded"
