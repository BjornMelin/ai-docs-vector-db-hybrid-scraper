"""Tests for performance middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.fastapi.middleware.performance import (
    PerformanceMiddleware,
    setup_prometheus,
)


def test_performance_middleware_sets_header() -> None:
    """Test that PerformanceMiddleware sets response time header."""

    app = FastAPI()
    app.add_middleware(PerformanceMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/ping")

    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
    # Ensure header contains a float value.
    float(response.headers["X-Response-Time"])


def test_setup_prometheus_adds_metrics_route() -> None:
    """Test that setup_prometheus adds metrics route."""

    app = FastAPI()
    inst = setup_prometheus(app)

    assert inst is not None
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/metrics" in paths
