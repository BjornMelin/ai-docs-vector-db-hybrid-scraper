"""Tests for tool registry wiring."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, cast

import pytest


if "src.infrastructure.client_manager" not in sys.modules:
    client_manager_stub = types.ModuleType("src.infrastructure.client_manager")

    class ClientManager:  # pragma: no cover - import shim for tests
        """Minimal client manager placeholder."""

        async def initialize(self) -> None:  # pragma: no cover - unused placeholder
            return None

    client_manager_stub.ClientManager = ClientManager  # type: ignore[attr-defined]
    sys.modules["src.infrastructure.client_manager"] = client_manager_stub


if "src.services.crawling" not in sys.modules:
    crawling_stub = types.ModuleType("src.services.crawling")

    async def crawl_page(
        *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:  # pragma: no cover
        return {}

    crawling_stub.crawl_page = crawl_page  # type: ignore[attr-defined]
    sys.modules["src.services.crawling"] = crawling_stub


if "src.mcp_tools.tools" not in sys.modules:
    tools_stub = types.ModuleType("src.mcp_tools.tools")

    for module_name in [
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
    ]:
        full_name = f"src.mcp_tools.tools.{module_name}"
        submodule = types.ModuleType(full_name)

        def register_tools(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
            return None

        submodule.register_tools = register_tools  # type: ignore[attr-defined]
        setattr(tools_stub, module_name, submodule)
        sys.modules[full_name] = submodule

    sys.modules["src.mcp_tools.tools"] = tools_stub


from src.mcp_tools import tool_registry

from .conftest import FakeClientManager, FakeMCP


@pytest.mark.asyncio
async def test_register_all_tools_invokes_pipeline_in_order(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: FakeMCP,
    fake_client_manager: FakeClientManager,
) -> None:
    """All registrars execute once in the defined pipeline order."""

    calls: list[str] = []

    stubbed_pipeline = []

    for module_name, _ in tool_registry._REGISTRATION_PIPELINE:

        def stub(mcp: Any, manager: Any, *, _name: str = module_name) -> None:
            assert mcp is fake_mcp
            assert manager is fake_client_manager
            calls.append(_name)

        stubbed_pipeline.append((module_name, stub))

    monkeypatch.setattr(tool_registry, "_REGISTRATION_PIPELINE", stubbed_pipeline)

    await tool_registry.register_all_tools(
        cast(Any, fake_mcp), cast(Any, fake_client_manager)
    )

    expected_order = [name for name, _ in tool_registry._REGISTRATION_PIPELINE]
    assert calls == expected_order


@pytest.mark.asyncio
async def test_register_all_tools_propagates_registration_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: FakeMCP,
    fake_client_manager: FakeClientManager,
) -> None:
    """Exceptions from a registrar bubble up and halt subsequent registrations."""

    calls: list[str] = []
    names = [name for name, _ in tool_registry._REGISTRATION_PIPELINE]

    def fail_stub(mcp: Any, manager: Any) -> None:
        assert mcp is fake_mcp
        assert manager is fake_client_manager
        calls.append(names[1])
        raise RuntimeError("boom")

    def first_stub(mcp: Any, manager: Any) -> None:
        assert mcp is fake_mcp
        assert manager is fake_client_manager
        calls.append(names[0])

    stubbed_pipeline = [(names[0], first_stub), (names[1], fail_stub)]

    for module_name in names[2:]:

        def later_stub(mcp: Any, manager: Any, *, _name: str = module_name) -> None:
            calls.append(_name)

        stubbed_pipeline.append((module_name, later_stub))

    monkeypatch.setattr(tool_registry, "_REGISTRATION_PIPELINE", stubbed_pipeline)

    with pytest.raises(RuntimeError, match="boom"):
        await tool_registry.register_all_tools(
            cast(Any, fake_mcp), cast(Any, fake_client_manager)
        )

    assert calls == names[:2]


@pytest.mark.asyncio
async def test_register_all_tools_logs_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp: FakeMCP,
    fake_client_manager: FakeClientManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Registry reports the number of modules registered via logging."""

    stubbed_pipeline = []

    for module_name, _ in tool_registry._REGISTRATION_PIPELINE:

        def stub(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial
            return None

        stubbed_pipeline.append((module_name, stub))

    monkeypatch.setattr(tool_registry, "_REGISTRATION_PIPELINE", stubbed_pipeline)

    caplog.set_level(logging.INFO, tool_registry.__name__)
    await tool_registry.register_all_tools(
        cast(Any, fake_mcp), cast(Any, fake_client_manager)
    )

    summary = f"Registered {len(tool_registry._REGISTRATION_PIPELINE)} MCP tool modules"
    assert summary in caplog.text
