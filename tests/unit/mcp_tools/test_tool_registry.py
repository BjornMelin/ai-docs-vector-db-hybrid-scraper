"""Tests for the unified MCP tool registry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools import tool_registry


def _make_module(
    name: str, attr: str, registered: list[str], *, raises: Exception | None = None
):
    """Construct a simple namespace with a recording register function.

    Args:
        name: Canonical module label to append when registration succeeds.
        attr: Registrar attribute exposed by the module under test.
        registered: Mutable sequence capturing successful registrations.
        raises: Optional exception instance for simulating failure paths.

    Returns:
        SimpleNamespace: Proxy module exposing the expected registration attribute.
    """

    def register(mcp, client_manager):
        """Record the registration invocation or raise for failure simulation."""

        del mcp, client_manager
        if raises is not None:
            raise raises
        registered.append(name)

    return SimpleNamespace(**{attr: register})


def _build_modules(registered: list[str]):
    """Assemble the minimal modules namespace used by the registry tests.

    Args:
        registered: Shared list used to collect registration order.

    Returns:
        SimpleNamespace: Collection of mocked tool modules.
    """

    return SimpleNamespace(
        search=_make_module("search", "register_tools", registered),
        documents=_make_module("documents", "register_tools", registered),
        embeddings=_make_module("embeddings", "register_tools", registered),
        lightweight_scrape=_make_module(
            "lightweight_scrape", "register_tools", registered
        ),
        collections=_make_module("collections", "register_tools", registered),
        projects=_make_module("projects", "register_tools", registered),
        search_tools=_make_module("search_tools", "register_tools", registered),
        query_processing=_make_module("query_processing", "register_tools", registered),
        filtering_tools=_make_module(
            "filtering_tools", "register_filtering_tools", registered
        ),
        query_processing_tools=_make_module(
            "query_processing_tools",
            "register_query_processing_tools",
            registered,
        ),
        payload_indexing=_make_module("payload_indexing", "register_tools", registered),
        analytics=_make_module("analytics", "register_tools", registered),
        cache=_make_module("cache", "register_tools", registered),
        utilities=_make_module("utilities", "register_tools", registered),
        content_intelligence=_make_module(
            "content_intelligence", "register_tools", registered
        ),
        agentic_rag=_make_module(
            "agentic_rag",
            "register_tools",
            registered,
            raises=ImportError("missing optional dependency"),
        ),
    )


@pytest.mark.asyncio
async def test_register_all_tools_invokes_each_module(monkeypatch):
    """register_all_tools should forward registration to every module exactly once."""
    registered: list[str] = []
    modules = _build_modules(registered)

    monkeypatch.setattr(tool_registry, "tools", modules)

    mcp = MagicMock()
    mcp.tool = MagicMock(return_value=lambda func: func)
    client_manager = AsyncMock()

    await tool_registry.register_all_tools(mcp, client_manager)

    expected = [
        "search",
        "documents",
        "embeddings",
        "lightweight_scrape",
        "collections",
        "projects",
        "search_tools",
        "query_processing",
        "filtering_tools",
        "query_processing_tools",
        "payload_indexing",
        "analytics",
        "cache",
        "utilities",
        "content_intelligence",
    ]
    assert registered[: len(expected)] == expected
    assert "agentic_rag" not in registered  # optional module failure is tolerated


@pytest.mark.asyncio
async def test_register_all_tools_propagates_required_module_failure(monkeypatch):
    """Required module failures should bubble up as runtime errors."""

    registered: list[str] = []
    modules = _build_modules(registered)
    modules.documents = _make_module(
        "documents",
        "register_tools",
        registered,
        raises=RuntimeError("documents failure"),
    )
    monkeypatch.setattr(tool_registry, "tools", modules)

    mcp = MagicMock()
    mcp.tool = MagicMock(return_value=lambda func: func)
    client_manager = AsyncMock()

    with pytest.raises(RuntimeError, match="documents failure"):
        await tool_registry.register_all_tools(mcp, client_manager)

    assert registered == ["search"]
