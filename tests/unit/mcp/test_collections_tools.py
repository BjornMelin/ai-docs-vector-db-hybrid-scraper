"""Comprehensive tests for collections tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from fastmcp import Context
from src.infrastructure.client_manager import ClientManager
from src.mcp.tools import collections


class TestCollectionsTools:
    """Test collections tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager with necessary attributes."""
        cm = Mock(spec=ClientManager)

        # Mock qdrant_service
        qdrant_service = AsyncMock()
        qdrant_service.list_collections = AsyncMock(
            return_value=["collection1", "collection2"]
        )
        qdrant_service.get_collection_info = AsyncMock(
            return_value=Mock(
                vectors_count=100,
                indexed_vectors_count=100,
                points_count=100,
                segments_count=1,
                config=Mock(
                    params=Mock(
                        vectors=Mock(size=1536, distance="Cosine"),
                        shard_number=1,
                        replication_factor=1,
                    ),
                    hnsw_config=Mock(m=16, ef_construct=200, full_scan_threshold=10000),
                    optimizer_config=Mock(
                        deleted_threshold=0.2, vacuum_min_vector_number=1000
                    ),
                    quantization_config=None,
                ),
                status="green",
            )
        )
        qdrant_service.delete_collection = AsyncMock(return_value=True)

        # Mock cache_manager
        cache_manager = AsyncMock()
        cache_manager.clear = AsyncMock()

        # Set attributes
        cm.qdrant_service = qdrant_service
        cm.cache_manager = cache_manager

        return cm

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        ctx = Mock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
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
        """Test that collection tools are registered correctly."""
        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "list_collections" in mock_mcp._tools
        assert "delete_collection" in mock_mcp._tools
        assert "optimize_collection" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_mcp, mock_client_manager, mock_context):
        """Test list_collections functionality."""
        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        list_func = mock_mcp._tools["list_collections"]

        # Call the function (no ctx parameter)
        result = await list_func()

        # Verify results
        assert len(result) == 2
        assert result[0]["name"] == "collection1"
        assert result[0]["vectors_count"] == 100
        assert result[0]["indexed_vectors_count"] == 100
        assert result[0]["config"]["size"] == 1536
        assert result[0]["config"]["distance"] == "Cosine"

        # Verify service was called
        mock_client_manager.qdrant_service.list_collections.assert_called_once()
        assert mock_client_manager.qdrant_service.get_collection_info.call_count == 2

    @pytest.mark.asyncio
    async def test_list_collections_with_error(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test list_collections with error getting info."""
        # Configure mock to fail on second collection
        mock_client_manager.qdrant_service.get_collection_info = AsyncMock(
            side_effect=[
                Mock(
                    vectors_count=100,
                    indexed_vectors_count=100,
                    config=Mock(
                        params=Mock(vectors=Mock(size=1536, distance="Cosine"))
                    ),
                ),
                Exception("Failed to get info"),
            ]
        )

        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        list_func = mock_mcp._tools["list_collections"]

        # Call the function
        result = await list_func()

        # Verify results
        assert len(result) == 2
        assert result[0]["name"] == "collection1"
        assert result[0]["vectors_count"] == 100
        assert result[1]["name"] == "collection2"
        assert "error" in result[1]
        assert "Failed to get info" in result[1]["error"]

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_mcp, mock_client_manager, mock_context):
        """Test delete_collection functionality."""
        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        delete_func = mock_mcp._tools["delete_collection"]

        # Call the function
        result = await delete_func(collection_name="old_collection")

        # Verify results
        assert result["status"] == "deleted"
        assert result["collection"] == "old_collection"

        # Verify service was called
        mock_client_manager.qdrant_service.delete_collection.assert_called_once_with(
            "old_collection"
        )

        # Verify cache was cleared for this collection
        mock_client_manager.cache_manager.clear.assert_called_once_with(
            pattern="*:old_collection:*"
        )

    @pytest.mark.asyncio
    async def test_delete_collection_error(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test delete_collection error handling."""
        # Configure mock to raise exception
        mock_client_manager.qdrant_service.delete_collection = AsyncMock(
            side_effect=Exception("Deletion failed")
        )

        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        delete_func = mock_mcp._tools["delete_collection"]

        # Call the function
        result = await delete_func(collection_name="fail_collection")

        # Verify error handling
        assert result["status"] == "error"
        assert "Deletion failed" in result["message"]

    @pytest.mark.asyncio
    async def test_optimize_collection(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test optimize_collection functionality."""
        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        optimize_func = mock_mcp._tools["optimize_collection"]

        # Call the function
        result = await optimize_func(collection_name="collection1")

        # Verify results
        assert result["status"] == "optimized"
        assert result["collection"] == "collection1"
        assert result["vectors_count"] == 100
        assert result["indexed_vectors_count"] == 100

        # Verify service was called
        mock_client_manager.qdrant_service.get_collection_info.assert_called_once_with(
            "collection1"
        )

    @pytest.mark.asyncio
    async def test_optimize_collection_error(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test optimize_collection error handling."""
        # Configure mock to raise exception
        mock_client_manager.qdrant_service.get_collection_info = AsyncMock(
            side_effect=Exception("Collection not found")
        )

        # Register tools
        collections.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        optimize_func = mock_mcp._tools["optimize_collection"]

        # Call the function
        result = await optimize_func(collection_name="nonexistent")

        # Verify error handling
        assert result["status"] == "error"
        assert "Collection not found" in result["message"]
