"""Integration tests for the unified MCP server lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import unified_mcp_server


def _lifespan_context() -> AbstractAsyncContextManager[None]:
    """Return the managed lifespan context for the unified MCP server."""

    lifespan_factory = getattr(unified_mcp_server, "managed_lifespan", None)
    if lifespan_factory is None:
        msg = "Unified MCP server does not expose a managed lifespan context."
        raise AttributeError(msg)
    return cast(
        Callable[[Any], AbstractAsyncContextManager[None]],
        lifespan_factory,
    )(unified_mcp_server.mcp)


def _stub_container() -> SimpleNamespace:
    """Return a container stub exposing the providers used during startup."""

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
        app_name="Test App",
        version="1.2.3",
        environment=SimpleNamespace(value="testing"),
        log_level=SimpleNamespace(value="INFO"),
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
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

    container = _stub_container()

    @asynccontextmanager
    async def _container_context(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield container

    monkeypatch.setattr(
        "src.unified_mcp_server.container_session",
        _container_context,
    )

    register_mock = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.register_all_tools",
        register_mock,
    )

    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_monitoring_system",
        MagicMock(return_value=None),
    )

    async with _lifespan_context():
        pass

    register_mock.assert_awaited_once()
    await_args = register_mock.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["vector_service"] is container.vector_store_service()
    assert kwargs["cache_manager"] is container.cache_manager()
    assert kwargs["embedding_manager"] is container.embedding_manager()


@pytest.mark.asyncio
async def test_lifespan_enables_monitoring_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health monitoring should start when configuration enables it."""

    monkeypatch.setattr(unified_mcp_server, "validate_configuration", MagicMock())

    config = SimpleNamespace(
        app_name="Test App",
        version="1.2.3",
        environment=SimpleNamespace(value="testing"),
        log_level=SimpleNamespace(value="INFO"),
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=1,
            enable_health_checks=True,
        ),
        observability=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(unified_mcp_server, "get_settings", lambda: config)

    container = _stub_container()

    @asynccontextmanager
    async def _container_context(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield container

    monkeypatch.setattr(
        "src.unified_mcp_server.container_session",
        _container_context,
    )

    register_mock = AsyncMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.register_all_tools",
        register_mock,
    )

    health_manager = SimpleNamespace(config=SimpleNamespace(enabled=True))
    monkeypatch.setattr(
        "src.unified_mcp_server.initialize_monitoring_system",
        MagicMock(return_value=health_manager),
    )

    created_tasks: list[asyncio.Task[Any]] = []
    original_create_task = asyncio.create_task

    def _track_create_task(coro: Any, *args: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro, *args, **kwargs)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "src.unified_mcp_server.asyncio.create_task",
        _track_create_task,
    )

    async def _fake_health_checks(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "src.unified_mcp_server.run_periodic_health_checks",
        _fake_health_checks,
    )

    monitoring_mock = MagicMock()
    monkeypatch.setattr(
        "src.unified_mcp_server.setup_fastmcp_monitoring",
        monitoring_mock,
    )

    async with _lifespan_context():
        await asyncio.sleep(0)

    register_mock.assert_awaited_once()
    monitoring_mock.assert_called_once_with(
        unified_mcp_server.mcp, config, health_manager
    )
    assert created_tasks, "Expected a monitoring task to be scheduled."
    for task in created_tasks:
        assert task.cancelled()
