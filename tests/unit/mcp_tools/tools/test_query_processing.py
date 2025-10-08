"""Tests for the simplified MCP query processing tools."""

# pylint: disable=duplicate-code

from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.contracts.retrieval import SearchRecord


def _load_tools_module(monkeypatch: pytest.MonkeyPatch):
    """Load the query processing tools module with stubbed dependencies."""

    stub_responses = types.ModuleType("src.mcp_tools.models.responses")

    class StubSearchResult(dict):  # pragma: no cover - simple holder
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)

    stub_responses.SearchResult = StubSearchResult  # type: ignore[attr-defined]

    stub_models = types.ModuleType("src.mcp_tools.models")
    stub_models.responses = stub_responses  # type: ignore[attr-defined]

    stub_package = types.ModuleType("src.mcp_tools")
    stub_package.models = stub_models  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "src.mcp_tools", stub_package)
    monkeypatch.setitem(sys.modules, "src.mcp_tools.models", stub_models)
    monkeypatch.setitem(sys.modules, "src.mcp_tools.models.responses", stub_responses)

    module_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "mcp_tools"
        / "tools"
        / "query_processing_tools.py"
    )
    if not module_path.exists():
        pytest.skip("query_processing_tools module not present in this build")
    spec = spec_from_file_location("qp_tools", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class MockMCP:
    """Minimal MCP server stub that collects registered tools."""

    def __init__(self) -> None:
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class MockContext:
    """Collects log output emitted by tools during tests."""

    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    async def debug(self, message: str) -> None:  # pragma: no cover - simple store
        self.debug_messages.append(message)

    async def info(self, message: str) -> None:  # pragma: no cover - simple store
        self.info_messages.append(message)

    async def error(self, message: str) -> None:  # pragma: no cover - simple store
        self.error_messages.append(message)


@pytest.fixture
def tools_module(monkeypatch: pytest.MonkeyPatch):
    """Provide the query processing tools module with stubbed dependencies."""

    return _load_tools_module(monkeypatch)


@pytest.mark.asyncio
async def test_tool_registration(monkeypatch, tools_module) -> None:
    """Registering tools should expose the search_documents tool."""

    mcp = MockMCP()
    client_manager = Mock()

    async def _fake_get_orchestrator(_: Mock) -> AsyncMock:
        orchestrator = AsyncMock()
        orchestrator.search.return_value = SimpleNamespace(results=[])
        return orchestrator

    monkeypatch.setattr(
        tools_module,
        "_get_orchestrator",
        AsyncMock(side_effect=_fake_get_orchestrator),
    )

    tools_module.register_tools(mcp, client_manager)

    assert "search_documents" in mcp.tools
    assert callable(mcp.tools["search_documents"])


@pytest.mark.asyncio
async def test_search_documents_invokes_orchestrator(monkeypatch, tools_module) -> None:
    """The search tool should execute the orchestrator and convert results."""

    mcp = MockMCP()
    client_manager = Mock()
    ctx = MockContext()

    orchestrator = AsyncMock()
    orchestrator.search.return_value = SimpleNamespace(
        records=[SearchRecord(id="1", content="Hello", score=0.75)]
    )
    monkeypatch.setattr(
        tools_module, "_get_orchestrator", AsyncMock(return_value=orchestrator)
    )

    tools_module.register_tools(mcp, client_manager)
    search_tool = mcp.tools["search_documents"]

    search_request_cls = tools_module.SearchToolRequest
    request = search_request_cls(query="hello", limit=5)
    results = await search_tool(request, ctx)

    orchestrator.search.assert_awaited_once()
    assert len(results) == 1
    assert results[0].content == "Hello"
    assert ctx.info_messages  # ensure logs were emitted
