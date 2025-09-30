"""Tests for the unified MCP tool registry."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools import tool_registry


@pytest.mark.asyncio
async def test_register_all_tools_invokes_each_module(monkeypatch, build_tool_modules):
    """register_all_tools should forward registration to every module exactly once."""
    registered: list[str] = []
    modules = build_tool_modules(registered)

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
async def test_register_all_tools_propagates_required_module_failure(
    monkeypatch, build_tool_modules
):
    """Required module failures should bubble up as runtime errors."""

    registered: list[str] = []
    modules = build_tool_modules(
        registered,
        overrides={
            "documents": {"raises": RuntimeError("documents failure")},
        },
    )
    monkeypatch.setattr(tool_registry, "tools", modules)

    mcp = MagicMock()
    mcp.tool = MagicMock(return_value=lambda func: func)
    client_manager = AsyncMock()

    with pytest.raises(RuntimeError, match="documents failure"):
        await tool_registry.register_all_tools(mcp, client_manager)

    assert registered == ["search"]
