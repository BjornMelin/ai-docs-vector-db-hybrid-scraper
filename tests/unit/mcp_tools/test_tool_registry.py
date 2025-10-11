"""Tests for tool registry wiring under dependency-injector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.mcp_tools import tool_registry


@pytest.fixture()
def fake_mcp() -> MagicMock:
    """Provide a fake FastMCP application."""

    app = MagicMock()
    app.tool.side_effect = lambda *args, **kwargs: (lambda fn: fn)
    return app


@pytest.fixture()
def stub_services() -> dict[str, Any]:
    """Expose stubbed service dependencies."""

    return {
        "vector_service": MagicMock(name="vector_service"),
        "cache_manager": MagicMock(name="cache_manager"),
        "crawl_manager": MagicMock(name="crawl_manager"),
        "content_intelligence_service": MagicMock(name="content_service"),
        "project_storage": MagicMock(name="project_storage"),
        "embedding_manager": MagicMock(name="embedding_manager"),
        "health_manager": MagicMock(name="health_manager"),
    }


@pytest.mark.asyncio
async def test_register_all_tools_invokes_registrars_once(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: MagicMock,
    stub_services: dict[str, Any],
) -> None:
    """Each tool module should be invoked exactly once with expected args."""

    calls: dict[str, SimpleNamespace] = {}

    def _capture(name: str):
        def _record(*args: Any, **kwargs: Any) -> None:
            calls[name] = SimpleNamespace(args=args, kwargs=kwargs)

        return _record

    monkeypatch.setattr(
        tool_registry.tools.retrieval,
        "register_tools",
        _capture("retrieval"),
    )
    monkeypatch.setattr(
        tool_registry.tools.documents,
        "register_tools",
        _capture("documents"),
    )
    monkeypatch.setattr(
        tool_registry.tools.embeddings,
        "register_tools",
        _capture("embeddings"),
    )
    monkeypatch.setattr(
        tool_registry.tools.lightweight_scrape,
        "register_tools",
        _capture("lightweight_scrape"),
    )
    monkeypatch.setattr(
        tool_registry.tools.collection_management,
        "register_tools",
        _capture("collection_management"),
    )
    monkeypatch.setattr(
        tool_registry.tools.projects,
        "register_tools",
        _capture("projects"),
    )
    monkeypatch.setattr(
        tool_registry.tools.payload_indexing,
        "register_tools",
        _capture("payload_indexing"),
    )
    monkeypatch.setattr(
        tool_registry.tools.analytics,
        "register_tools",
        _capture("analytics"),
    )
    monkeypatch.setattr(
        tool_registry.tools.cache,
        "register_tools",
        _capture("cache"),
    )
    monkeypatch.setattr(
        tool_registry.tools.content_intelligence,
        "register_tools",
        _capture("content_intelligence"),
    )
    monkeypatch.setattr(
        tool_registry.tools.system_health,
        "register_tools",
        _capture("system_health"),
    )
    monkeypatch.setattr(
        tool_registry.tools.web_search,
        "register_tools",
        _capture("web_search"),
    )
    monkeypatch.setattr(
        tool_registry.tools.cost_estimation,
        "register_tools",
        _capture("cost_estimation"),
    )

    await tool_registry.register_all_tools(fake_mcp, **stub_services)

    expected_modules = {
        "retrieval",
        "documents",
        "embeddings",
        "lightweight_scrape",
        "collection_management",
        "projects",
        "payload_indexing",
        "analytics",
        "cache",
        "content_intelligence",
        "system_health",
        "web_search",
        "cost_estimation",
    }

    assert set(calls) == expected_modules
    assert all(namespace.args[0] is fake_mcp for namespace in calls.values())
    assert (
        calls["retrieval"].kwargs["vector_service"] is stub_services["vector_service"]
    )
    assert calls["documents"].kwargs["cache_manager"] is stub_services["cache_manager"]
    assert (
        calls["system_health"].kwargs["health_manager"]
        is stub_services["health_manager"]
    )


@pytest.mark.asyncio
async def test_register_all_tools_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: MagicMock,
    stub_services: dict[str, Any],
) -> None:
    """Exceptions from registrars should surface and stop registration."""

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        tool_registry.tools.retrieval,
        "register_tools",
        _fail,
    )
    called = False

    def _documents(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        tool_registry.tools.documents,
        "register_tools",
        _documents,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await tool_registry.register_all_tools(fake_mcp, **stub_services)

    assert called is False


@pytest.mark.asyncio
async def test_register_all_tools_logs_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: MagicMock,
    stub_services: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Registry should log a concise summary after successful registration."""

    for module in [
        tool_registry.tools.retrieval,
        tool_registry.tools.documents,
        tool_registry.tools.embeddings,
        tool_registry.tools.lightweight_scrape,
        tool_registry.tools.collection_management,
        tool_registry.tools.projects,
        tool_registry.tools.payload_indexing,
        tool_registry.tools.analytics,
        tool_registry.tools.cache,
        tool_registry.tools.content_intelligence,
        tool_registry.tools.system_health,
        tool_registry.tools.web_search,
        tool_registry.tools.cost_estimation,
    ]:
        monkeypatch.setattr(module, "register_tools", lambda *args, **kwargs: None)

    caplog.set_level("INFO", tool_registry.__name__)

    await tool_registry.register_all_tools(fake_mcp, **stub_services)

    assert "Registered MCP tools" in caplog.text


@pytest.mark.asyncio
async def test_register_all_tools_skips_system_health_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: MagicMock,
    stub_services: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """System health tools should be skipped when no manager is provided."""

    services = {**stub_services, "health_manager": None}

    for module in [
        tool_registry.tools.retrieval,
        tool_registry.tools.documents,
        tool_registry.tools.embeddings,
        tool_registry.tools.lightweight_scrape,
        tool_registry.tools.collection_management,
        tool_registry.tools.projects,
        tool_registry.tools.payload_indexing,
        tool_registry.tools.analytics,
        tool_registry.tools.cache,
        tool_registry.tools.content_intelligence,
        tool_registry.tools.web_search,
        tool_registry.tools.cost_estimation,
    ]:
        monkeypatch.setattr(module, "register_tools", lambda *args, **kwargs: None)

    system_health_mock = MagicMock()
    monkeypatch.setattr(
        tool_registry.tools.system_health, "register_tools", system_health_mock
    )

    caplog.set_level("INFO", tool_registry.__name__)

    await tool_registry.register_all_tools(fake_mcp, **services)

    system_health_mock.assert_not_called()
    assert "Skipping system health tool registration" in caplog.text
