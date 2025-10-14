"""Integration tests for global API rate limiting."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.config.security import SecurityConfig
from src.services.fastapi.middleware.security import enable_global_rate_limit


def _create_app(config: SecurityConfig) -> FastAPI:
    """Construct a FastAPI app with configured rate limiting."""
    app = FastAPI()
    limiter = enable_global_rate_limit(app, config=config)
    assert limiter is not None, "Limiter should be initialized when enabled"

    @app.get("/limited")
    async def _limited_endpoint(request: Request) -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_rate_limiting_enforces_configured_threshold() -> None:
    """Requests exceeding the configured limit receive HTTP 429 responses."""
    app = _create_app(
        SecurityConfig(
            default_rate_limit=2,
            rate_limit_window=1,
        )
    )
    client = TestClient(app)

    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200

    blocked = client.get("/limited")
    assert blocked.status_code == 429
    assert "Rate limit exceeded" in blocked.json().get("error", "")
