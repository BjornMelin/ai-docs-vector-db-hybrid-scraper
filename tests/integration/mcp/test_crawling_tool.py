"""Integration-style tests for the crawling MCP tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.tools import crawling as crawling_module
from src.services.browser.unified_manager import UnifiedScrapingRequest


ToolCallable = Callable[..., Awaitable[dict[str, Any]]]


class StubMCP:
    """Minimal FastMCP stub capturing registered tools."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolCallable] = {}

    def tool(
        self, *_args: Any, **_kwargs: Any
    ) -> Callable[[ToolCallable], ToolCallable]:
        def decorator(func: ToolCallable) -> ToolCallable:
            self.tools[func.__name__] = func
            return func

        return decorator


@pytest.mark.asyncio
async def test_enhanced_crawl_delegates_to_router() -> None:
    """enhanced_5_tier_crawl should pass a UnifiedScrapingRequest to the manager."""

    mcp = StubMCP()
    manager = MagicMock()
    manager.scrape_url = AsyncMock(
        return_value={"success": True, "provider": "playwright"}
    )

    crawling_module.register_tools(mcp, crawl_manager=manager)
    tool = mcp.tools["enhanced_5_tier_crawl"]
    ctx = AsyncMock()

    result = await tool(
        url="https://app.example.com/dashboard",
        tier="playwright",
        interaction_required=True,
        custom_actions=[{"action": "click"}],
        timeout_ms=1234,
        ctx=ctx,
    )

    manager.scrape_url.assert_awaited_once()
    args, _ = manager.scrape_url.await_args
    request = args[0]
    assert isinstance(request, UnifiedScrapingRequest)
    assert request.tier == "playwright"
    assert request.interaction_required is True
    assert request.timeout == 1234
    assert request.custom_actions == [{"action": "click"}]
    assert result["provider"] == "playwright"


@pytest.mark.asyncio
async def test_enhanced_crawl_validates_url() -> None:
    """Invalid URLs should raise ValueError before delegating."""

    mcp = StubMCP()
    manager = MagicMock()
    manager.scrape_url = AsyncMock()

    crawling_module.register_tools(mcp, crawl_manager=manager)
    tool = mcp.tools["enhanced_5_tier_crawl"]

    with pytest.raises(ValueError):
        await tool(url="invalid-url", ctx=None)

    manager.scrape_url.assert_not_called()
