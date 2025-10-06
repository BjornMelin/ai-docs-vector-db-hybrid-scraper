"""Integration tests for the unified MCP server lifespan."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import unified_mcp_server


@pytest.mark.asyncio
async def test_lifespan_initializes_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan context should initialize services and register tools."""

    monkeypatch.setattr(unified_mcp_server, "validate_configuration", MagicMock())

    config = SimpleNamespace(
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            enable_local_cache=False,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=False,
            include_system_metrics=False,
            system_metrics_interval=60,
        ),
    )
    monkeypatch.setattr(unified_mcp_server, "get_config", lambda: config)

    client_manager = AsyncMock()
    client_manager.cleanup = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "ClientManager", lambda: client_manager)

    register_mock = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "register_all_tools", register_mock)

    monkeypatch.setattr(
        unified_mcp_server,
        "initialize_monitoring_system",
        lambda _config, _qdrant, _redis: (None, None),
    )

    async with unified_mcp_server.lifespan():
        pass

    client_manager.initialize.assert_awaited_once()
    client_manager.cleanup.assert_awaited_once()
    register_mock.assert_awaited_once_with(unified_mcp_server.mcp, client_manager)


@pytest.mark.asyncio
async def test_lifespan_cancels_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Background monitoring tasks should be cancelled on shutdown."""

    monkeypatch.setattr(unified_mcp_server, "validate_configuration", MagicMock())

    config = SimpleNamespace(
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            enable_local_cache=True,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=1,
        ),
    )
    monkeypatch.setattr(unified_mcp_server, "get_config", lambda: config)

    client_manager = AsyncMock()
    client_manager.cleanup = AsyncMock()
    cache_dependency = AsyncMock(return_value=object())
    monkeypatch.setattr(unified_mcp_server, "get_cache_manager", cache_dependency)
    monkeypatch.setattr(unified_mcp_server, "ClientManager", lambda: client_manager)

    register_mock = AsyncMock()
    monkeypatch.setattr(unified_mcp_server, "register_all_tools", register_mock)

    health_manager = AsyncMock()
    metrics_registry = AsyncMock()
    monkeypatch.setattr(
        unified_mcp_server,
        "initialize_monitoring_system",
        lambda _config, _qdrant, _redis: (metrics_registry, health_manager),
    )

    monitor_tasks: list[asyncio.Task] = []

    async def _fake_health_checks(*_args, **_kwargs):
        try:
            await asyncio.sleep(3600)
        finally:
            current = asyncio.current_task()
            if current is not None:
                monitor_tasks.append(current)

    monkeypatch.setattr(
        unified_mcp_server,
        "run_periodic_health_checks",
        _fake_health_checks,
    )
    monkeypatch.setattr(
        unified_mcp_server,
        "update_system_metrics_periodically",
        _fake_health_checks,
    )
    monkeypatch.setattr(
        unified_mcp_server,
        "update_cache_metrics_periodically",
        _fake_health_checks,
    )
    monkeypatch.setattr(unified_mcp_server, "setup_fastmcp_monitoring", MagicMock())

    async with unified_mcp_server.lifespan():
        await asyncio.sleep(0.01)

    assert monitor_tasks, "Expected monitoring tasks to be started"
    for task in monitor_tasks:
        assert task.cancelled()
