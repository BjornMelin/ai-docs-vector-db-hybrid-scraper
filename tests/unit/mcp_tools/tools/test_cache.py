"""Comprehensive test suite for MCP cache tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.mcp_tools.models.responses import CacheClearResponse
from src.mcp_tools.models.responses import CacheStatsResponse


class TestCacheTools:
    """Test suite for cache MCP tools."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create a mock client manager with cache service."""
        mock_manager = MagicMock()

        # Mock cache manager
        mock_cache = AsyncMock()
        mock_cache.clear_all.return_value = 50  # cleared 50 items
        mock_cache.clear_pattern.return_value = 50  # cleared 50 items
        mock_cache.get_stats.return_value = {
            "hit_rate": 0.92,
            "size": 1500,
            "total_requests": 25000,
        }
        mock_manager.get_cache_manager = AsyncMock(return_value=mock_cache)

        return mock_manager

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.debug = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.warning = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, mock_client_manager, mock_context):
        """Test clearing all cache."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]

        result = await clear_cache(pattern=None, ctx=mock_context)

        assert isinstance(result, CacheClearResponse)
        assert result.status == "success"
        assert result.cleared_count == 50
        assert result.pattern is None

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_clear_cache_pattern(self, mock_client_manager, mock_context):
        """Test clearing cache with pattern."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]

        result = await clear_cache(pattern="search:*", ctx=mock_context)

        assert isinstance(result, CacheClearResponse)
        assert result.status == "success"
        assert result.cleared_count == 50
        assert result.pattern == "search:*"

        # Verify pattern-specific logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_client_manager, mock_context):
        """Test getting cache statistics."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_cache_stats = registered_tools["get_cache_stats"]

        result = await get_cache_stats(ctx=mock_context)

        assert isinstance(result, CacheStatsResponse)
        assert result.hit_rate == 0.92
        assert result.size == 1500
        assert result.total_requests == 25000

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, mock_client_manager, mock_context):
        """Test cache error handling."""
        from src.mcp_tools.tools.cache import register_tools

        # Make cache manager raise an exception
        mock_cache = AsyncMock()
        mock_cache.clear_all.side_effect = Exception("Cache service unavailable")
        mock_client_manager.get_cache_manager = AsyncMock(return_value=mock_cache)

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]

        # Should raise the exception after logging
        with pytest.raises(Exception, match="Cache service unavailable"):
            await clear_cache(pattern=None, ctx=mock_context)

        # Error should be logged
        mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_cache_stats_error_handling(self, mock_client_manager, mock_context):
        """Test cache stats error handling."""
        from src.mcp_tools.tools.cache import register_tools

        # Make cache manager raise an exception
        mock_cache = AsyncMock()
        mock_cache.get_stats.side_effect = Exception("Stats unavailable")
        mock_client_manager.get_cache_manager = AsyncMock(return_value=mock_cache)

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_cache_stats = registered_tools["get_cache_stats"]

        # Should raise the exception after logging
        with pytest.raises(Exception, match="Stats unavailable"):
            await get_cache_stats(ctx=mock_context)

        # Error should be logged
        mock_context.error.assert_called()

    def test_cache_clear_response_validation(self):
        """Test cache clear response model validation."""
        response = CacheClearResponse(
            status="success", cleared_count=25, pattern="test:*"
        )

        assert response.status == "success"
        assert response.cleared_count == 25
        assert response.pattern == "test:*"

        # Test without pattern
        response_no_pattern = CacheClearResponse(
            status="success", cleared_count=100, pattern=None
        )

        assert response_no_pattern.status == "success"
        assert response_no_pattern.cleared_count == 100
        assert response_no_pattern.pattern is None

    def test_cache_stats_response_validation(self):
        """Test cache stats response model validation."""
        response = CacheStatsResponse(hit_rate=0.95, size=2000, total_requests=50000)

        assert response.hit_rate == 0.95
        assert response.size == 2000
        assert response.total_requests == 50000

    @pytest.mark.asyncio
    async def test_context_logging_integration(self, mock_client_manager, mock_context):
        """Test that context logging is properly integrated."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]
        get_cache_stats = registered_tools["get_cache_stats"]

        # Test clear cache logging
        await clear_cache(pattern="test:*", ctx=mock_context)
        assert mock_context.info.call_count >= 1

        # Reset and test stats logging
        mock_context.reset_mock()
        await get_cache_stats(ctx=mock_context)
        assert mock_context.info.call_count >= 1

    def test_tool_registration(self, mock_client_manager):
        """Test that cache tools are properly registered."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        register_tools(mock_mcp, mock_client_manager)

        # Should have registered 2 tools
        assert mock_mcp.tool.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_manager_interactions(self, mock_client_manager, mock_context):
        """Test proper interaction with cache manager."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]
        get_cache_stats = registered_tools["get_cache_stats"]

        # Test cache manager is retrieved
        await clear_cache(pattern=None, ctx=mock_context)
        mock_client_manager.get_cache_manager.assert_called()

        # Test stats call
        await get_cache_stats(ctx=mock_context)
        cache_manager = await mock_client_manager.get_cache_manager()
        cache_manager.get_stats.assert_called()

    @pytest.mark.asyncio
    async def test_different_clear_patterns(self, mock_client_manager, mock_context):
        """Test clearing cache with different patterns."""
        from src.mcp_tools.tools.cache import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        clear_cache = registered_tools["clear_cache"]

        # Test different patterns
        patterns = ["search:*", "embedding:*", "vector:*", None]

        for pattern in patterns:
            result = await clear_cache(pattern=pattern, ctx=mock_context)
            assert isinstance(result, CacheClearResponse)
            assert result.pattern == pattern
