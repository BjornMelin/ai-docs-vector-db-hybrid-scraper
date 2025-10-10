"""Integration tests for the unified MCP server lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import unified_mcp_server


def _lifespan_context() -> AbstractAsyncContextManager[None]:
    lifespan_attr = getattr(unified_mcp_server, "lifespan", None)
    if lifespan_attr is not None:
        callable_lifespan = cast(
            Callable[[], AbstractAsyncContextManager[None]], lifespan_attr
        )
        return callable_lifespan()

    mcp_runtime: Any = unified_mcp_server.mcp
    lifespan_factory = getattr(mcp_runtime, "lifespan", None)
    if lifespan_factory is None:
        msg = "FastMCP instance does not expose a lifespan context."
        raise AttributeError(msg)
    # pylint: disable=not-callable
    return cast(Callable[[], AbstractAsyncContextManager[None]], lifespan_factory)()


def _stub_container() -> SimpleNamespace:
    """Create a container stub exposing required service providers."""

    vector_service = MagicMock(name="vector_service")
    cache_manager = MagicMock(name="cache_manager")
    browser_manager = MagicMock(name="browser_manager")
    content_service = MagicMock(name="content_service")
    project_storage = MagicMock(name="project_storage")
    embedding_manager = MagicMock(name="embedding_manager")
    qdrant_client = MagicMock(name="qdrant_client")

    return SimpleNamespace(
        qdrant_client=lambda: qdrant_client,
        vector_store_service=lambda: vector_service,
        cache_manager=lambda: cache_manager,
        browser_manager=lambda: browser_manager,
        content_intelligence_service=lambda: content_service,
        project_storage=lambda: project_storage,
        embedding_manager=lambda: embedding_manager,
    )


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
    monkeypatch.setattr(unified_mcp_server, "get_settings", lambda: config)

    container = _stub_container()
    initialize_stub = AsyncMock(return_value=container)
    shutdown_stub = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_container",
        initialize_stub,
    )
    monkeypatch.setattr(
        "src.unified_mcp_server.shutdown_container",
        shutdown_stub,
    )

    register_mock = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.register_all_tools",
        register_mock,
    )

    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_monitoring_system",
        lambda _config, _qdrant, _redis: None,
    )

    async with _lifespan_context():
        pass

    initialize_stub.assert_awaited_once_with(config)
    shutdown_stub.assert_awaited_once()
    register_mock.assert_awaited_once()
    await_args = register_mock.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["vector_service"] is container.vector_store_service()
    assert kwargs["cache_manager"] is container.cache_manager()
    assert kwargs["embedding_manager"] is container.embedding_manager()


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
            enable_health_checks=True,
        ),
    )
    monkeypatch.setattr(unified_mcp_server, "get_settings", lambda: config)

    container = _stub_container()
    initialize_stub = AsyncMock(return_value=container)
    shutdown_stub = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_container",
        initialize_stub,
    )
    monkeypatch.setattr(
        "src.unified_mcp_server.shutdown_container",
        shutdown_stub,
    )

    register_mock = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.register_all_tools",
        register_mock,
    )

    health_manager = SimpleNamespace(config=SimpleNamespace(enabled=True))
    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_monitoring_system",
        lambda _config, _qdrant, _redis: health_manager,
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
        "src.unified_mcp_server.run_periodic_health_checks",
        _fake_health_checks,
    )
    monkeypatch.setattr(
        "src.unified_mcp_server.setup_fastmcp_monitoring",
        MagicMock(),
    )

    async with _lifespan_context():
        await asyncio.sleep(0.01)

    assert monitor_tasks, "Expected monitoring tasks to be started"
    for task in monitor_tasks:
        assert task.cancelled()
