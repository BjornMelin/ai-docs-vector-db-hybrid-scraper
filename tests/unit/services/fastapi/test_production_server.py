"""Lifecycle tests for the standalone production health surface."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.routing import Route

from src.config import Settings
from src.config.models import Environment, MonitoringConfig
from src.services.fastapi import production_server


@pytest.mark.asyncio
async def test_health_endpoint_uses_owned_qdrant_and_reports_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production endpoint must not omit an unavailable configured Qdrant."""
    settings = Settings(
        environment=Environment.TESTING,
        monitoring=MonitoringConfig(include_system_metrics=False),
    )
    client = AsyncMock()
    client.get_collections.side_effect = ConnectionError("qdrant unavailable")
    container = SimpleNamespace(qdrant_client=lambda: client)
    lifecycle_events: list[str] = []

    @asynccontextmanager
    async def fake_container_session(*, settings: Settings):
        assert settings is production.config
        lifecycle_events.append("acquire")
        try:
            yield container
        finally:
            lifecycle_events.append("release")

    monkeypatch.setattr(production_server.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        production_server,
        "container_session",
        fake_container_session,
    )
    production = production_server.ProductionMCPServer(settings)
    production.startup = AsyncMock()
    production.shutdown = AsyncMock()
    app = production.create_app()
    health_route = next(
        route
        for route in app.routes
        if isinstance(route, Route) and route.path == "/health"
    )

    async with production.lifespan(app):
        response = await health_route.endpoint(None)
        payload = json.loads(response.body)

        assert response.status_code == 503
        assert payload["checks"]["qdrant"]["status"] == "unhealthy"
        assert payload["status"] == "unhealthy"

    assert lifecycle_events == ["acquire", "release"]
    production.startup.assert_awaited_once_with()
    production.shutdown.assert_awaited_once_with()
    assert production._health_manager is None  # pylint: disable=protected-access
