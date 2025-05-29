"""Comprehensive tests for cache tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from fastmcp import Context
from src.infrastructure.client_manager import ClientManager
from src.mcp.tools import cache


class TestCacheTools:
    """Test cache tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager."""
        cm = Mock(spec=ClientManager)

        # Mock cache_manager
        cache_mgr = AsyncMock()
        cache_mgr.get_stats = AsyncMock(
            return_value={
                "total_keys": 100,
                "memory_usage": "50MB",
                "hit_rate": 0.85,
                "cache_size": 100,
                "eviction_count": 10,
            }
        )
        cache_mgr.clear = AsyncMock(return_value=50)

        cm.cache_manager = cache_mgr
        return cm

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        ctx = Mock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance that captures registered tools."""
        mcp = Mock()
        mcp._tools = {}

        def tool_decorator(func=None, **kwargs):
            def wrapper(f):
                mcp._tools[f.__name__] = f
                return f

            return wrapper if func is None else wrapper(func)

        mcp.tool = tool_decorator
        return mcp

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that cache tools are registered correctly."""
        # Register tools
        cache.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "get_cache_stats" in mock_mcp._tools
        assert "clear_cache" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_mcp, mock_client_manager, mock_context):
        """Test get_cache_stats functionality."""
        # Register tools
        cache.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        stats_func = mock_mcp._tools["get_cache_stats"]

        # Call the function
        result = await stats_func()

        # Verify results
        assert result["total_keys"] == 100
        assert result["memory_usage"] == "50MB"
        assert result["hit_rate"] == 0.85
        assert result["cache_size"] == 100
        assert result["eviction_count"] == 10

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, mock_mcp, mock_client_manager, mock_context):
        """Test clear_cache with no pattern (clear all)."""
        # Mock clear_all method
        mock_client_manager.cache_manager.clear_all = AsyncMock(return_value=50)

        # Register tools
        cache.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        clear_func = mock_mcp._tools["clear_cache"]

        # Call the function
        result = await clear_func()

        # Verify results
        assert result["status"] == "success"
        assert result["cleared_count"] == 50
        assert result["pattern"] is None

        # Verify cache was cleared
        mock_client_manager.cache_manager.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_with_pattern(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test clear_cache with specific pattern."""
        # Mock clear_pattern method
        mock_client_manager.cache_manager.clear_pattern = AsyncMock(return_value=50)

        # Register tools
        cache.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        clear_func = mock_mcp._tools["clear_cache"]

        # Call the function with pattern
        result = await clear_func(pattern="embeddings:*")

        # Verify results
        assert result["status"] == "success"
        assert result["cleared_count"] == 50
        assert result["pattern"] == "embeddings:*"

        # Verify cache was cleared with pattern
        mock_client_manager.cache_manager.clear_pattern.assert_called_once_with(
            "embeddings:*"
        )

    @pytest.mark.asyncio
    async def test_clear_cache_error_handling(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test clear_cache error handling."""
        # Configure mock to raise error
        mock_client_manager.cache_manager.clear_all = AsyncMock(
            side_effect=Exception("Cache clear failed")
        )

        # Register tools
        cache.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        clear_func = mock_mcp._tools["clear_cache"]

        # Call the function - the actual implementation raises the exception
        with pytest.raises(Exception, match="Cache clear failed"):
            await clear_func()
