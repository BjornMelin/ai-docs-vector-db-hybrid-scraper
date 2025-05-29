"""Comprehensive tests for search tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import Context
from src.config.enums import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp.models.requests import SearchRequest
from src.mcp.models.responses import SearchResult
from src.mcp.tools import search


class TestSearchTools:
    """Test search tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager with complete configuration."""
        cm = Mock(spec=ClientManager)

        # Add config attribute with necessary structure
        config = Mock()
        config.cache = Mock(
            enable_caching=True,
            enable_local_cache=True,
            enable_dragonfly_cache=False,
            dragonfly_url="redis://localhost:6379",
            local_max_size=1000,
            local_max_memory_mb=100,
        )
        config.qdrant = Mock(
            url="http://localhost:6333", api_key=None, timeout=30.0, prefer_grpc=False
        )
        config.openai = Mock(api_key="test-key")
        config.embedding = Mock(provider="openai", model="text-embedding-3-small")
        config.performance = Mock(cache_ttl=3600)
        cm.config = config

        # Add unified_config attribute for service initialization
        unified_config = Mock()
        unified_config.cache = config.cache
        unified_config.qdrant = config.qdrant
        unified_config.openai = config.openai
        unified_config.embedding = config.embedding
        unified_config.performance = config.performance
        cm.unified_config = unified_config

        # Add required services
        cm.cache_manager = AsyncMock()
        cm.embedding_manager = AsyncMock()
        cm.qdrant_service = AsyncMock()

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
        """Test that search tools are registered correctly."""
        # Register tools
        search.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "search_documents" in mock_mcp._tools
        assert "search_similar" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_search_documents_basic(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test basic search_documents functionality."""
        # Setup mocks
        with (
            patch("src.mcp.tools.search.CacheManager") as MockCacheManager,
            patch("src.mcp.tools.search.EmbeddingManager") as MockEmbeddingManager,
            patch("src.mcp.tools.search.QdrantService") as MockQdrantService,
        ):
            # Configure mocks
            mock_cache = AsyncMock()
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            MockCacheManager.return_value = mock_cache

            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                return_value=Mock(embeddings=[[0.1, 0.2, 0.3]], sparse_embeddings=None)
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            mock_qdrant = AsyncMock()
            mock_qdrant.initialize = AsyncMock()
            mock_qdrant.hybrid_search = AsyncMock(
                return_value=[
                    {
                        "id": "1",
                        "score": 0.9,
                        "payload": {
                            "content": "Test content",
                            "metadata": {"source": "test.com"},
                        },
                    }
                ]
            )
            MockQdrantService.return_value = mock_qdrant

            # Register tools
            search.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            search_func = mock_mcp._tools["search_documents"]

            # Create request
            request = SearchRequest(
                query="test query",
                collection="test_collection",
                strategy=SearchStrategy.DENSE,
                limit=10,
            )

            # Call the function
            results = await search_func(request, mock_context)

            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].content == "Test content"
            assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_search_documents_with_cache(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test search_documents with cache hit."""
        # Setup mocks
        with patch("src.mcp.tools.search.CacheManager") as MockCacheManager:
            # Configure cache to return cached results
            mock_cache = AsyncMock()
            mock_cache.get = AsyncMock(
                return_value=[
                    {
                        "content": "Cached content",
                        "metadata": {"source": "cache"},
                        "score": 0.95,
                        "id": "cached-1",
                        "collection": "test",
                    }
                ]
            )
            MockCacheManager.return_value = mock_cache

            # Register tools
            search.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            search_func = mock_mcp._tools["search_documents"]

            # Create request
            request = SearchRequest(
                query="cached query",
                collection="test_collection",
                strategy=SearchStrategy.DENSE,
                limit=5,
            )

            # Call the function
            results = await search_func(request, mock_context)

            # Verify cached results were returned
            assert len(results) == 1
            assert results[0].content == "Cached content"
            assert results[0].score == 0.95

            # Verify cache was checked
            mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar(self, mock_mcp, mock_client_manager, mock_context):
        """Test search_similar functionality."""
        # Setup mocks
        with patch("src.mcp.tools.search.QdrantService") as MockQdrantService:
            mock_qdrant = AsyncMock()
            mock_qdrant.initialize = AsyncMock()
            mock_qdrant.retrieve = AsyncMock(
                return_value=[
                    Mock(vector=[0.1, 0.2, 0.3], payload={"content": "Original doc"})
                ]
            )
            mock_qdrant.hybrid_search = AsyncMock(
                return_value=[
                    {
                        "id": "2",
                        "score": 0.85,
                        "payload": {
                            "content": "Similar doc",
                            "metadata": {"similarity": "high"},
                        },
                    }
                ]
            )
            MockQdrantService.return_value = mock_qdrant

            # Register tools
            search.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            similar_func = mock_mcp._tools["search_similar"]

            # Call the function with correct parameters
            result = await similar_func(
                query_id="doc-123",
                collection="test_collection",
                limit=5,
                ctx=mock_context,
            )

            # Verify results
            assert len(result) == 1
            assert result[0].content == "Similar doc"
            assert result[0].score == 0.85

    @pytest.mark.asyncio
    async def test_search_documents_hybrid_strategy(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test search_documents with hybrid strategy."""
        with (
            patch("src.mcp.tools.search.CacheManager") as MockCacheManager,
            patch("src.mcp.tools.search.EmbeddingManager") as MockEmbeddingManager,
            patch("src.mcp.tools.search.QdrantService") as MockQdrantService,
        ):
            # Configure mocks
            MockCacheManager.return_value = AsyncMock(get=AsyncMock(return_value=None))

            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                return_value=Mock(
                    embeddings=[[0.1, 0.2, 0.3]],
                    sparse_embeddings=[[0.0, 0.5, 0.0, 0.8]],
                )
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            mock_qdrant = AsyncMock()
            mock_qdrant.initialize = AsyncMock()
            mock_qdrant.hybrid_search = AsyncMock(
                return_value=[
                    {
                        "id": "hybrid-1",
                        "score": 0.92,
                        "payload": {
                            "content": "Hybrid result",
                            "metadata": {"type": "hybrid"},
                        },
                    }
                ]
            )
            MockQdrantService.return_value = mock_qdrant

            # Register tools
            search.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            search_func = mock_mcp._tools["search_documents"]

            # Create request with hybrid strategy
            request = SearchRequest(
                query="hybrid search",
                collection="test_collection",
                strategy=SearchStrategy.HYBRID,
                limit=10,
            )

            # Call the function
            results = await search_func(request, mock_context)

            # Verify hybrid search was used
            assert len(results) == 1
            assert results[0].content == "Hybrid result"

            # Verify sparse embeddings were generated
            mock_embedding_mgr.generate_embeddings.assert_called_with(
                texts=["hybrid search"], generate_sparse=True, model=None
            )

    @pytest.mark.asyncio
    async def test_search_documents_error_handling(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test search_documents error handling."""
        with (
            patch("src.mcp.tools.search.CacheManager") as MockCacheManager,
            patch("src.mcp.tools.search.EmbeddingManager") as MockEmbeddingManager,
            patch("src.mcp.tools.search.QdrantService") as MockQdrantService,
        ):
            # Configure mocks
            MockCacheManager.return_value = AsyncMock(get=AsyncMock(return_value=None))

            # Configure embedding manager to raise error
            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                side_effect=Exception("Embedding error")
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Configure QdrantService
            mock_qdrant = AsyncMock()
            mock_qdrant.initialize = AsyncMock()
            MockQdrantService.return_value = mock_qdrant

            # Register tools
            search.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            search_func = mock_mcp._tools["search_documents"]

            # Create request
            request = SearchRequest(
                query="error test",
                collection="test_collection",
                strategy=SearchStrategy.DENSE,
                limit=10,
            )

            # Call should raise error
            with pytest.raises(Exception, match="Embedding error"):
                await search_func(request, mock_context)
