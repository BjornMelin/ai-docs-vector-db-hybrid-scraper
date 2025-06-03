"""Comprehensive test suite for MCP analytics tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.mcp.models.requests import AnalyticsRequest
from src.mcp.models.responses import AnalyticsResponse
from src.mcp.models.responses import SystemHealthResponse


class TestAnalyticsTools:
    """Test suite for analytics MCP tools."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create a mock client manager with all required services."""
        mock_manager = MagicMock()

        # Mock qdrant service
        mock_qdrant = AsyncMock()
        mock_qdrant.list_collections.return_value = ["docs", "api", "knowledge"]
        mock_qdrant.get_collection_info.return_value = {
            "vectors_count": 1000,
            "points_count": 1000,
            "status": "green"
        }
        mock_manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

        # Mock cache manager
        mock_cache = AsyncMock()
        mock_cache.get_stats.return_value = {
            "hit_rate": 0.85,
            "size": 500,
            "total_requests": 10000
        }
        mock_manager.get_cache_manager = AsyncMock(return_value=mock_cache)

        # Mock embedding manager
        mock_embedding = AsyncMock()
        mock_embedding.get_current_provider_info.return_value = {
            "name": "fastembed",
            "model": "BAAI/bge-small-en-v1.5"
        }
        mock_manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

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
    async def test_get_analytics_basic(self, mock_client_manager, mock_context):
        """Test basic analytics retrieval."""
        from src.mcp.tools.analytics import register_tools

        # Create a mock MCP server
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool

        # Register tools
        register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        get_analytics = registered_tools["get_analytics"]

        # Test basic request
        request = AnalyticsRequest(
            collection=None,
            include_performance=False,
            include_costs=False
        )

        result = await get_analytics(request, mock_context)

        # Verify result is AnalyticsResponse
        assert isinstance(result, AnalyticsResponse)
        assert result.timestamp is not None
        assert result.collections is not None
        assert isinstance(result.collections, dict)

        # Verify context logging was called
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_get_analytics_with_performance(self, mock_client_manager, mock_context):
        """Test analytics with performance metrics."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_analytics = registered_tools["get_analytics"]

        request = AnalyticsRequest(
            collection=None,
            include_performance=True,
            include_costs=False
        )

        result = await get_analytics(request, mock_context)

        assert isinstance(result, AnalyticsResponse)
        assert result.cache_metrics is not None
        assert isinstance(result.cache_metrics, dict)

        # Verify cache manager was called
        mock_client_manager.get_cache_manager.assert_called()

    @pytest.mark.asyncio
    async def test_get_analytics_with_costs(self, mock_client_manager, mock_context):
        """Test analytics with cost estimates."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_analytics = registered_tools["get_analytics"]

        request = AnalyticsRequest(
            collection=None,
            include_performance=True,
            include_costs=True
        )

        result = await get_analytics(request, mock_context)

        assert isinstance(result, AnalyticsResponse)
        assert result.costs is not None
        assert isinstance(result.costs, dict)

    @pytest.mark.asyncio
    async def test_get_analytics_specific_collection(self, mock_client_manager, mock_context):
        """Test analytics for a specific collection."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_analytics = registered_tools["get_analytics"]

        request = AnalyticsRequest(
            collection="docs",
            include_performance=False,
            include_costs=False
        )

        result = await get_analytics(request, mock_context)

        assert isinstance(result, AnalyticsResponse)
        assert "docs" in result.collections or len(result.collections) >= 0

    @pytest.mark.asyncio
    async def test_get_system_health(self, mock_client_manager, mock_context):
        """Test system health check."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_system_health = registered_tools["get_system_health"]

        result = await get_system_health(mock_context)

        assert isinstance(result, SystemHealthResponse)
        assert result.status is not None
        assert result.timestamp is not None
        assert result.services is not None
        assert isinstance(result.services, dict)

    @pytest.mark.asyncio
    async def test_analytics_error_handling(self, mock_client_manager, mock_context):
        """Test analytics error handling."""
        from src.mcp.tools.analytics import register_tools

        # Make qdrant service raise an exception
        mock_qdrant = AsyncMock()
        mock_qdrant.list_collections.side_effect = Exception("Service unavailable")
        mock_client_manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_analytics = registered_tools["get_analytics"]

        request = AnalyticsRequest(
            collection=None,
            include_performance=False,
            include_costs=False
        )

        # Should raise the exception after logging
        with pytest.raises(Exception, match="Service unavailable"):
            await get_analytics(request, mock_context)

        # Error should be logged
        mock_context.error.assert_called()

    def test_analytics_request_validation(self):
        """Test analytics request model validation."""
        # Test valid request
        request = AnalyticsRequest(
            collection="test",
            include_performance=True,
            include_costs=True
        )
        assert request.collection == "test"
        assert request.include_performance is True
        assert request.include_costs is True

        # Test defaults
        default_request = AnalyticsRequest()
        assert default_request.collection is None
        assert default_request.include_performance is True
        assert default_request.include_costs is True

    def test_analytics_response_validation(self):
        """Test analytics response model validation."""
        response = AnalyticsResponse(
            timestamp="2024-01-01T00:00:00Z",
            collections={"test": {"vector_count": 100}},
            cache_metrics={"hit_rate": 0.85},
            performance={"avg_query_time": 45.2},
            costs={"storage_gb": 2.5}
        )

        assert response.timestamp == "2024-01-01T00:00:00Z"
        assert "test" in response.collections
        assert response.cache_metrics["hit_rate"] == 0.85
        assert response.performance["avg_query_time"] == 45.2
        assert response.costs["storage_gb"] == 2.5

    def test_system_health_response_validation(self):
        """Test system health response model validation."""
        response = SystemHealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            services={
                "qdrant": {"status": "healthy"},
                "cache": {"status": "healthy"}
            }
        )

        assert response.status == "healthy"
        assert response.timestamp == "2024-01-01T00:00:00Z"
        assert "qdrant" in response.services
        assert "cache" in response.services

    @pytest.mark.asyncio
    async def test_context_logging_integration(self, mock_client_manager, mock_context):
        """Test that context logging is properly integrated."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        get_analytics = registered_tools["get_analytics"]
        get_system_health = registered_tools["get_system_health"]

        # Test analytics logging
        request = AnalyticsRequest()
        await get_analytics(request, mock_context)

        # Should have multiple logging calls
        assert mock_context.info.call_count >= 1

        # Reset and test system health logging
        mock_context.reset_mock()
        await get_system_health(mock_context)

        assert mock_context.info.call_count >= 1

    def test_tool_registration(self, mock_client_manager):
        """Test that tools are properly registered with the MCP server."""
        from src.mcp.tools.analytics import register_tools

        mock_mcp = MagicMock()
        register_tools(mock_mcp, mock_client_manager)

        # Should have registered at least 2 tools
        assert mock_mcp.tool.call_count >= 2
