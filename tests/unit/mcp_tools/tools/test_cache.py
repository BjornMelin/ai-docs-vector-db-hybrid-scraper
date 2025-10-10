"""Comprehensive test suite for MCP cache tools."""

# pylint: disable=duplicate-code

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.models.responses import CacheClearResponse, CacheStatsResponse
from src.mcp_tools.tools.cache import register_tools


class TestCacheTools:
    """Test suite for cache MCP tools."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager with async methods."""

        cache_manager = MagicMock()
        cache_manager.clear_all = AsyncMock(return_value=50)
        cache_manager.clear_pattern = AsyncMock(return_value=50)
        cache_manager.get_stats = AsyncMock(
            return_value={
                "hit_rate": 0.92,
                "size": 1500,
                "total_requests": 25000,
                "vendor_metric": 0.88,
            }
        )
        return cache_manager

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""

        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.error = AsyncMock()
        ctx.warning = AsyncMock()
        return ctx

    def _register_tools(
        self, cache_manager: Any
    ) -> tuple[MagicMock, dict[str, Callable[..., Awaitable[Any]]]]:
        mock_mcp = MagicMock()
        registered_tools: dict[str, Callable[..., Awaitable[Any]]] = {}

        def capture(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture
        register_tools(mock_mcp, cache_manager=cache_manager)
        return mock_mcp, registered_tools

    @pytest.mark.asyncio()
    async def test_clear_cache_all(self, mock_cache_manager, mock_context):
        """Clearing all cache entries should succeed and log progress."""

        _, tools = self._register_tools(mock_cache_manager)
        clear_cache = tools["clear_cache"]

        result = await clear_cache(pattern=None, ctx=mock_context)

        assert isinstance(result, CacheClearResponse)
        assert result.status == "success"
        assert result.cleared_count == 50
        assert result.pattern is None
        mock_cache_manager.clear_all.assert_awaited_once()
        mock_context.info.assert_called()

    @pytest.mark.asyncio()
    async def test_clear_cache_pattern(self, mock_cache_manager, mock_context):
        """Clearing a specific pattern should call ``clear_pattern``."""

        _, tools = self._register_tools(mock_cache_manager)
        clear_cache = tools["clear_cache"]

        result = await clear_cache(pattern="search:*", ctx=mock_context)

        assert isinstance(result, CacheClearResponse)
        assert result.pattern == "search:*"
        mock_cache_manager.clear_pattern.assert_awaited_once_with("search:*")
        mock_context.info.assert_called()

    @pytest.mark.asyncio()
    async def test_get_cache_stats(self, mock_cache_manager, mock_context):
        """Retrieving cache stats should normalise data and log context."""

        _, tools = self._register_tools(mock_cache_manager)
        get_cache_stats = tools["get_cache_stats"]

        result = await get_cache_stats(ctx=mock_context)

        assert isinstance(result, CacheStatsResponse)
        assert result.hit_rate == 0.92
        assert result.size == 1500
        assert result.total_requests == 25000
        mock_cache_manager.get_stats.assert_awaited_once()
        mock_context.info.assert_called()

    @pytest.mark.asyncio()
    async def test_cache_error_handling(self, mock_cache_manager, mock_context):
        """Errors from the cache manager should bubble up with logging."""

        mock_cache_manager.clear_all.side_effect = RuntimeError("cache offline")
        _, tools = self._register_tools(mock_cache_manager)
        clear_cache = tools["clear_cache"]

        with pytest.raises(RuntimeError, match="cache offline"):
            await clear_cache(pattern=None, ctx=mock_context)

        mock_context.error.assert_called()

    @pytest.mark.asyncio()
    async def test_cache_stats_error_handling(self, mock_cache_manager, mock_context):
        """Stats failures should propagate and emit error logs."""

        mock_cache_manager.get_stats.side_effect = RuntimeError("stats offline")
        _, tools = self._register_tools(mock_cache_manager)
        get_cache_stats = tools["get_cache_stats"]

        with pytest.raises(RuntimeError, match="stats offline"):
            await get_cache_stats(ctx=mock_context)

        mock_context.error.assert_called()

    def test_tool_registration(self, mock_cache_manager):
        """Both cache tools should be registered."""

        mock_mcp, _ = self._register_tools(mock_cache_manager)
        assert mock_mcp.tool.call_count == 2

    @pytest.mark.asyncio()
    async def test_context_logging_integration(
        self, mock_cache_manager, mock_context
    ) -> None:
        """Both tools should emit informational logs via context."""

        _, tools = self._register_tools(mock_cache_manager)
        clear_cache = tools["clear_cache"]
        get_cache_stats = tools["get_cache_stats"]

        await clear_cache(pattern="test:*", ctx=mock_context)
        await get_cache_stats(ctx=mock_context)

        assert mock_context.info.call_count >= 2

    @pytest.mark.asyncio()
    async def test_cache_manager_interactions(
        self, mock_cache_manager, mock_context
    ) -> None:
        """Verify the cache manager methods are invoked as expected."""

        _, tools = self._register_tools(mock_cache_manager)
        clear_cache = tools["clear_cache"]
        get_cache_stats = tools["get_cache_stats"]

        await clear_cache(pattern=None, ctx=mock_context)
        await get_cache_stats(ctx=mock_context)

        mock_cache_manager.clear_all.assert_awaited_once()
        mock_cache_manager.get_stats.assert_awaited_once()
