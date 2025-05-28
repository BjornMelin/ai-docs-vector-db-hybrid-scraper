"""Tests for individual MCP tool modules."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp.tools import collections
from src.mcp.tools import embeddings
from src.mcp.tools import search


class TestSearchTools:
    """Test search tool module."""

    @pytest.fixture
    def mcp(self):
        """Create mock MCP instance."""
        mcp = Mock(spec=FastMCP)
        mcp.tool = Mock(return_value=lambda f: f)
        return mcp

    @pytest.fixture
    def client_manager(self):
        """Create mock client manager."""
        return Mock(spec=ClientManager)

    def test_register_search_tools(self, mcp, client_manager):
        """Test that search tools are registered correctly."""
        # Register tools
        search.register_tools(mcp, client_manager)

        # Verify tool decorator was called
        assert mcp.tool.call_count >= 2  # search_documents and search_similar

    @pytest.mark.asyncio
    async def test_search_documents_caching(self, mcp, client_manager):
        """Test search_documents with cache hit."""
        # Setup mocks
        cache_manager = AsyncMock()
        cache_manager.get = AsyncMock(
            return_value=[
                {
                    "content": "cached",
                    "metadata": {},
                    "score": 0.9,
                    "id": "1",
                    "collection": "test",
                }
            ]
        )

        # Mock the CacheManager constructor
        with patch(
            "src.mcp.tools.search.CacheManager", return_value=cache_manager
        ):
            # Register tools
            search.register_tools(mcp, client_manager)

            # Get the registered function
            search_func = mcp.tool.return_value.call_args[0][0]

            # Create mock request and context
            from src.config.enums import SearchStrategy
            from src.mcp.models.requests import SearchRequest

            request = SearchRequest(
                query="test query",
                collection="test",
                strategy=SearchStrategy.DENSE,
                limit=10,
            )

            ctx = AsyncMock()

            # Call the function
            results = await search_func(request, ctx)

            # Verify cache was checked
            cache_manager.get.assert_called_once()
            assert len(results) == 1
            assert results[0].content == "cached"


class TestEmbeddingTools:
    """Test embedding tool module."""

    @pytest.fixture
    def mcp(self):
        """Create mock MCP instance."""
        mcp = Mock(spec=FastMCP)
        mcp.tool = Mock(return_value=lambda f: f)
        return mcp

    @pytest.fixture
    def client_manager(self):
        """Create mock client manager."""
        return Mock(spec=ClientManager)

    def test_register_embedding_tools(self, mcp, client_manager):
        """Test that embedding tools are registered correctly."""
        # Register tools
        embeddings.register_tools(mcp, client_manager)

        # Verify tool decorator was called
        assert (
            mcp.tool.call_count >= 2
        )  # generate_embeddings and list_embedding_providers

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, mcp, client_manager):
        """Test successful embedding generation."""
        # Setup mocks
        embedding_manager = AsyncMock()
        embedding_result = Mock()
        embedding_result.embeddings = [[0.1, 0.2, 0.3]]
        embedding_result.sparse_embeddings = None
        embedding_result.dimensions = 3
        embedding_result.model = "text-embedding-3-small"
        embedding_result.total_tokens = 10

        embedding_manager.generate_embeddings = AsyncMock(return_value=embedding_result)
        embedding_manager.get_current_provider_info = Mock(
            return_value={"name": "openai", "model": "text-embedding-3-small"}
        )

        # Mock the EmbeddingManager constructor
        with pytest.mock.patch(
            "src.mcp.tools.embeddings.EmbeddingManager", return_value=embedding_manager
        ):
            # Register tools
            embeddings.register_tools(mcp, client_manager)

            # Get the registered function
            embed_func = mcp.tool.return_value.call_args[0][0]

            # Create mock request
            from src.mcp.models.requests import EmbeddingRequest

            request = EmbeddingRequest(
                texts=["test text"], model="text-embedding-3-small"
            )

            # Call the function
            result = await embed_func(request)

            # Verify result
            assert result["count"] == 1
            assert result["dimensions"] == 3
            assert result["provider"] == "openai"
            assert result["model"] == "text-embedding-3-small"


class TestCollectionTools:
    """Test collection tool module."""

    @pytest.fixture
    def mcp(self):
        """Create mock MCP instance."""
        mcp = Mock(spec=FastMCP)
        mcp.tool = Mock(return_value=lambda f: f)
        return mcp

    @pytest.fixture
    def client_manager(self):
        """Create mock client manager."""
        # Create a more complete mock
        cm = Mock(spec=ClientManager)

        # Mock qdrant_service
        qdrant_service = AsyncMock()
        qdrant_service.list_collections = AsyncMock(
            return_value=["collection1", "collection2"]
        )
        qdrant_service.get_collection_info = AsyncMock(
            return_value=Mock(vectors_count=100, indexed_vectors_count=100)
        )

        # Mock cache_manager
        cache_manager = AsyncMock()
        cache_manager.clear = AsyncMock()

        # Set attributes
        cm.qdrant_service = qdrant_service
        cm.cache_manager = cache_manager

        return cm

    def test_register_collection_tools(self, mcp, client_manager):
        """Test that collection tools are registered correctly."""
        # Register tools
        collections.register_tools(mcp, client_manager)

        # Verify tool decorator was called
        assert (
            mcp.tool.call_count >= 3
        )  # list_collections, delete_collection, optimize_collection
