"""Tests for the unified MCP tool registry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools import tool_registry


@pytest.mark.asyncio
async def test_register_all_tools_invokes_each_module(monkeypatch):
    """register_all_tools should forward registration to every module exactly once."""
    registered = []

    def make_module(name: str, attr: str, raises: bool = False):
        def register(mcp, client_manager):  # noqa: D401 - simple recording closure
            if raises:
                raise ImportError("missing optional dependency")
            registered.append(name)

        return SimpleNamespace(**{attr: register})

    modules = SimpleNamespace(
        search=make_module("search", "register_tools"),
        documents=make_module("documents", "register_tools"),
        embeddings=make_module("embeddings", "register_tools"),
        lightweight_scrape=make_module("lightweight_scrape", "register_tools"),
        collections=make_module("collections", "register_tools"),
        projects=make_module("projects", "register_tools"),
        search_tools=make_module("search_tools", "register_tools"),
        query_processing=make_module("query_processing", "register_tools"),
        filtering_tools=make_module("filtering_tools", "register_filtering_tools"),
        query_processing_tools=make_module(
            "query_processing_tools", "register_query_processing_tools"
        ),
        payload_indexing=make_module("payload_indexing", "register_tools"),
        analytics=make_module("analytics", "register_tools"),
        cache=make_module("cache", "register_tools"),
        utilities=make_module("utilities", "register_tools"),
        content_intelligence=make_module("content_intelligence", "register_tools"),
        agentic_rag=make_module("agentic_rag", "register_tools", raises=True),
    )

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
