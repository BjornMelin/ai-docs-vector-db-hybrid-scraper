"""Integration tests for the unified MCP server lifespan."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import unified_mcp_server


def _lifespan_context() -> AbstractAsyncContextManager[None]:
    lifespan_attr = getattr(unified_mcp_server, "managed_lifespan", None)
    if lifespan_attr is None:
        msg = "Unified MCP server does not expose a managed lifespan context."
        raise AttributeError(msg)
    return lifespan_attr(unified_mcp_server.mcp)


@pytest.mark.asyncio
async def test_lifespan_initializes_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan context should initialize services and register tools."""

    monkeypatch.setattr(unified_mcp_server, "validate_configuration", MagicMock())

    config = SimpleNamespace(
        app_name="Test App",
        version="1.2.3",
        environment=SimpleNamespace(value="testing"),
        log_level=SimpleNamespace(value="INFO"),
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            enable_local_cache=False,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=False,
            include_system_metrics=False,
            system_metrics_interval=60,
            enable_health_checks=False,
        ),
        observability=SimpleNamespace(
            enabled=True,
            service_name="test-service",
            service_version="0.0.1",
            otlp_endpoint="http://localhost:4317",
            otlp_headers={},
            otlp_insecure=True,
            track_ai_operations=False,
            track_costs=False,
            instrument_fastapi=False,
            instrument_httpx=False,
            console_exporter=False,
        ),
    )
    monkeypatch.setattr(unified_mcp_server, "get_settings", lambda: config)

    client_manager = AsyncMock()
    client_manager.cleanup = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "ClientManager", lambda: client_manager)

    register_mock = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "register_all_tools", register_mock)

    initialize_observability = MagicMock(return_value=True)
    monkeypatch.setattr(
        unified_mcp_server,
        "initialize_observability",
        initialize_observability,
    )

    async with _lifespan_context():
        pass

    client_manager.initialize.assert_awaited_once()
    client_manager.cleanup.assert_awaited_once()
    register_mock.assert_awaited_once_with(unified_mcp_server.mcp, client_manager)
    initialize_observability.assert_called_once_with(config)


@pytest.mark.asyncio
async def test_lifespan_initializes_observability_with_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observability should be initialized even when monitoring features are on."""

    monkeypatch.setattr(unified_mcp_server, "validate_configuration", MagicMock())

    config = SimpleNamespace(
        app_name="Test App",
        version="1.2.3",
        environment=SimpleNamespace(value="testing"),
        log_level=SimpleNamespace(value="INFO"),
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            enable_local_cache=True,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=1,
            enable_health_checks=True,
        ),
        observability=SimpleNamespace(
            enabled=True,
            service_name="test-service",
            service_version="0.0.1",
            otlp_endpoint="http://localhost:4317",
            otlp_headers={},
            otlp_insecure=False,
            track_ai_operations=True,
            track_costs=True,
            instrument_fastapi=True,
            instrument_httpx=True,
            console_exporter=True,
        ),
    )
    monkeypatch.setattr(unified_mcp_server, "get_settings", lambda: config)

    client_manager = AsyncMock()
    client_manager.cleanup = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "ClientManager", lambda: client_manager)

    register_mock = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "register_all_tools", register_mock)

    initialize_observability = MagicMock(return_value=True)
    monkeypatch.setattr(
        unified_mcp_server,
        "initialize_observability",
        initialize_observability,
    )

    async with _lifespan_context():
        await asyncio.sleep(0.01)

    client_manager.initialize.assert_awaited_once()
    client_manager.cleanup.assert_awaited_once()
    register_mock.assert_awaited_once_with(unified_mcp_server.mcp, client_manager)
    initialize_observability.assert_called_once_with(config)
