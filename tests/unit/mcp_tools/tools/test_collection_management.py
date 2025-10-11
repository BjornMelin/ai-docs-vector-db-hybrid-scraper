"""Comprehensive test suite for MCP collections tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.models.responses import CollectionInfo, CollectionOperationResponse
from src.mcp_tools.tools.collection_management import register_tools


class TestCollectionsTools:
    """Test suite for collections MCP tools."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create a mock client manager with collections service."""
        mock_manager = MagicMock()

        # Mock vector store service
        mock_vector = AsyncMock()
        mock_vector.list_collections.return_value = [
            "docs",
            "api",
            "knowledge",
        ]

        async def mock_stats(_name: str) -> dict[str, Any]:
            return {
                "points_count": 1000,
                "indexed_vectors": 950,
                "vectors": {"size": 384, "distance": "cosine"},
            }

        mock_vector.collection_stats.side_effect = mock_stats
        mock_vector.delete_collection = AsyncMock()

        # Mock cache manager
        mock_cache = AsyncMock()
        mock_cache.clear.return_value = 10  # cleared items
        mock_manager.get_vector_store_service = AsyncMock(return_value=mock_vector)
        mock_manager.get_cache_manager = AsyncMock(return_value=mock_cache)
        mock_manager.cache_mock = mock_cache
        mock_manager.cache_manager = mock_cache
        mock_manager.vector_service = mock_vector

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
    async def test_list_collections(self, mock_client_manager, mock_context):
        """Test listing collections."""

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        list_collections = registered_tools["list_collections"]

        result = await list_collections(ctx=mock_context)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, CollectionInfo) for item in result)
        collection_names = [item.name for item in result]
        assert "docs" in collection_names
        assert "api" in collection_names
        assert "knowledge" in collection_names

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_optimize_collection(self, mock_client_manager, mock_context):
        """Test optimizing a collection."""

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        optimize_collection = registered_tools["optimize_collection"]

        result = await optimize_collection(collection_name="docs", ctx=mock_context)

        assert isinstance(result, CollectionOperationResponse)
        assert result.status == "optimized"
        assert result.collection == "docs"

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_client_manager, mock_context):
        """Test deleting a collection."""

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        delete_collection = registered_tools["delete_collection"]
        mock_vector = await mock_client_manager.get_vector_store_service()

        result = await delete_collection(
            collection_name="old_collection", ctx=mock_context
        )

        assert isinstance(result, CollectionOperationResponse)
        assert result.status == "deleted"
        assert result.collection == "old_collection"

        mock_context.info.assert_called()
        mock_vector.delete_collection.assert_awaited_once_with("old_collection")
        mock_client_manager.get_cache_manager.assert_awaited_once()
        mock_client_manager.cache_mock.clear_pattern.assert_awaited_once_with(
            "*:old_collection:*"
        )

    @pytest.mark.asyncio
    async def test_delete_collection_missing_methods(
        self, mock_client_manager, mock_context
    ):
        """Ensure an explicit error surfaces when delete/drop are unavailable."""

        mock_vector = await mock_client_manager.get_vector_store_service()
        mock_vector.delete_collection = None

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        delete_collection = registered_tools["delete_collection"]

        result = await delete_collection(
            collection_name="missing_methods", ctx=mock_context
        )

        assert result.status == "error"
        assert result.message
        assert "Vector service does not expose" in result.message

    @pytest.mark.asyncio
    async def test_collections_error_handling(self, mock_client_manager, mock_context):
        """Test collections error handling."""

        # Make qdrant service raise an exception
        mock_vector = AsyncMock()
        mock_vector.list_collections.side_effect = Exception("Service unavailable")
        mock_client_manager.get_vector_store_service = AsyncMock(
            return_value=mock_vector
        )

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        list_collections = registered_tools["list_collections"]

        # Should raise the exception after logging
        with pytest.raises(Exception, match="Service unavailable"):
            await list_collections(ctx=mock_context)

        # Error should be logged
        mock_context.error.assert_called()

    def test_collection_info_validation(self):
        """Test collection info model validation."""
        collection_info = CollectionInfo(
            name="test_collection",
            vectors_count=1000,
            points_count=1000,
            status="green",
        )

        assert collection_info.name == "test_collection"
        assert collection_info.vectors_count == 1000
        assert collection_info.points_count == 1000
        assert collection_info.status == "green"

    def test_collection_operation_response_validation(self):
        """Test collection operation response model validation."""

        operation_response = CollectionOperationResponse(
            status="deleted", collection="test_collection"
        )

        assert operation_response.status == "deleted"
        assert operation_response.collection == "test_collection"

    @pytest.mark.asyncio
    async def test_context_logging_integration(self, mock_client_manager, mock_context):
        """Test that context logging is properly integrated."""

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        # Test each tool has logging
        tools_to_test = [
            ("list_collections", []),
            ("delete_collection", ["old"]),
            ("optimize_collection", ["test"]),
        ]

        for tool_name, args in tools_to_test:
            mock_context.reset_mock()
            tool = registered_tools[tool_name]

            if args:
                await tool(*args, ctx=mock_context)
            else:
                await tool(ctx=mock_context)

            assert mock_context.info.call_count >= 1, f"Tool {tool_name} should log"

    def test_tool_registration(self, mock_client_manager):
        """Test that collection tools are properly registered."""

        mock_mcp = MagicMock()
        register_tools(
            mock_mcp,
            vector_service=mock_client_manager,
            cache_manager=mock_client_manager.cache_manager,
        )

        # Should have registered 3 tools
        assert mock_mcp.tool.call_count == 3
