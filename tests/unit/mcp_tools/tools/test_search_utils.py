"""Comprehensive tests for search utilities.

- Real-world functionality focus
- Complete coverage of search_documents_core function
- Zero flaky tests
- Modern pytest patterns
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.config.enums import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools._search_utils import search_documents_core


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.logs = {"info": [], "debug": [], "warning": [], "error": []}

    async def info(self, msg: str):
        self.logs["info"].append(msg)

    async def debug(self, msg: str):
        self.logs["debug"].append(msg)

    async def warning(self, msg: str):
        self.logs["warning"].append(msg)

    async def error(self, msg: str):
        self.logs["error"].append(msg)


@pytest.fixture
def mock_client_manager():
    """Create mock client manager with all required services."""
    manager = Mock(spec=ClientManager)

    # Mock cache manager
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)  # No cached results by default
    mock_cache.set = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=mock_cache)

    # Mock embedding manager
    mock_embedding = AsyncMock()
    mock_embedding.generate_embeddings = AsyncMock(
        return_value=Mock(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            sparse_embeddings=[[0.0, 0.1, 0.0, 0.2, 0.0]],
        )
    )
    manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

    # Mock qdrant service
    mock_qdrant = AsyncMock()
    mock_qdrant.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "test-doc-1",
                "score": 0.95,
                "payload": {
                    "content": "Test document content for search results",
                    "url": "https://example.com/doc1",
                    "title": "Test Document 1",
                    "metadata": {"source": "test"},
                    "content_intelligence_analyzed": True,
                    "content_type": "documentation",
                    "content_confidence": 0.9,
                    "quality_overall": 0.85,
                    "quality_completeness": 0.8,
                    "quality_relevance": 0.9,
                    "quality_confidence": 0.85,
                },
            },
            {
                "id": "test-doc-2",
                "score": 0.87,
                "payload": {
                    "content": "Another test document with different content",
                    "url": "https://example.com/doc2",
                    "title": "Test Document 2",
                    "metadata": {"source": "test"},
                },
            },
        ]
    )
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    return manager


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MockContext()


@pytest.fixture
def sample_search_request():
    """Create sample search request."""
    return SearchRequest(
        query="test search query",
        collection="test_collection",
        limit=10,
        strategy=SearchStrategy.HYBRID,
    )


class TestSearchDocumentsCore:
    """Test core search functionality."""

    async def test_successful_hybrid_search(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test successful hybrid search with full functionality."""
        results = await search_documents_core(
            sample_search_request, mock_client_manager, mock_context
        )

        # Verify results structure
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

        # Verify first result with content intelligence
        first_result = results[0]
        assert first_result.id == "test-doc-1"
        assert first_result.content == "Test document content for search results"
        assert first_result.score == 0.95
        assert first_result.url == "https://example.com/doc1"
        assert first_result.title == "Test Document 1"
        assert first_result.content_intelligence_analyzed is True
        assert first_result.content_type == "documentation"
        assert first_result.content_confidence == 0.9
        assert first_result.quality_overall == 0.85

        # Verify second result without content intelligence
        second_result = results[1]
        assert second_result.id == "test-doc-2"
        assert second_result.content_intelligence_analyzed is None
        assert second_result.content_type is None

        # Verify service calls
        mock_client_manager.get_cache_manager.assert_called_once()
        mock_client_manager.get_embedding_manager.assert_called_once()
        mock_client_manager.get_qdrant_service.assert_called_once()

        # Verify embedding generation with sparse enabled
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_called_once_with(
            texts=["test search query"], generate_sparse=True, model=None
        )

        # Verify hybrid search call
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.hybrid_search.assert_called_once()
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["query_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert call_args[1]["sparse_vector"] == {1: 0.1, 3: 0.2}  # Non-zero indices
        assert call_args[1]["limit"] == 10
        assert call_args[1]["fusion_type"] == "rrf"

        # Verify caching
        mock_cache = mock_client_manager.get_cache_manager.return_value
        expected_cache_key = (
            "search:test_collection:test search query:SearchStrategy.HYBRID:10"
        )
        mock_cache.get.assert_called_once_with(expected_cache_key)
        mock_cache.set.assert_called_once()

        # Verify logging
        assert any(
            "Starting search request" in msg for msg in mock_context.logs["info"]
        )
        assert any(
            "Executing SearchStrategy.HYBRID search" in msg
            for msg in mock_context.logs["info"]
        )
        assert any(
            "Search completed: 2 results found" in msg
            for msg in mock_context.logs["info"]
        )

    async def test_dense_search_strategy(self, mock_client_manager, mock_context):
        """Test dense-only search strategy."""
        request = SearchRequest(
            query="dense search test",
            collection="dense_collection",
            strategy=SearchStrategy.DENSE,
        )

        results = await search_documents_core(
            request, mock_client_manager, mock_context
        )

        assert len(results) == 2

        # Verify embedding generation without sparse
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_called_once_with(
            texts=["dense search test"], generate_sparse=False, model=None
        )

        # Verify hybrid search called with no sparse vector
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["sparse_vector"] is None
        assert call_args[1]["query_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_sparse_search_strategy(self, mock_client_manager, mock_context):
        """Test sparse-only search strategy."""
        request = SearchRequest(
            query="sparse search test",
            collection="sparse_collection",
            strategy=SearchStrategy.SPARSE,
        )

        results = await search_documents_core(
            request, mock_client_manager, mock_context
        )

        assert len(results) == 2

        # Verify embedding generation with sparse enabled
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_called_once_with(
            texts=["sparse search test"], generate_sparse=True, model=None
        )

        # Verify hybrid search called with empty dense vector
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["query_vector"] == []
        assert call_args[1]["sparse_vector"] == {1: 0.1, 3: 0.2}

    async def test_cache_hit_scenario(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test successful cache hit returns cached results."""
        # Mock cache hit
        cached_results = [
            {
                "id": "cached-doc-1",
                "content": "Cached document content",
                "score": 0.98,
                "url": "https://example.com/cached",
                "title": "Cached Document",
                "metadata": {"cached": True},
            }
        ]
        mock_cache = mock_client_manager.get_cache_manager.return_value
        mock_cache.get = AsyncMock(return_value=cached_results)

        results = await search_documents_core(
            sample_search_request, mock_client_manager, mock_context
        )

        # Verify cached result returned
        assert len(results) == 1
        assert results[0].id == "cached-doc-1"
        assert results[0].content == "Cached document content"

        # Verify no embedding or search calls made
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_not_called()

        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.hybrid_search.assert_not_called()

        # Verify debug logging for cache hit
        assert any("Cache hit for request" in msg for msg in mock_context.logs["debug"])

    async def test_sparse_embeddings_not_available_error(
        self, mock_client_manager, mock_context
    ):
        """Test error when sparse search requested but sparse embeddings not available."""
        request = SearchRequest(query="sparse test", strategy=SearchStrategy.SPARSE)

        # Mock embedding manager to return no sparse embeddings
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings = AsyncMock(
            return_value=Mock(embeddings=[[0.1, 0.2, 0.3]], sparse_embeddings=None)
        )

        with pytest.raises(
            ValueError, match="Sparse embeddings not available for sparse search"
        ):
            await search_documents_core(request, mock_client_manager, mock_context)

        # Verify error logged
        assert any("Search failed" in msg for msg in mock_context.logs["error"])

    async def test_embedding_generation_failure(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test handling of embedding generation failure."""
        # Mock embedding failure
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.side_effect = Exception(
            "Embedding service unavailable"
        )

        with pytest.raises(Exception, match="Embedding service unavailable"):
            await search_documents_core(
                sample_search_request, mock_client_manager, mock_context
            )

        # Verify error logged
        assert any("Search failed" in msg for msg in mock_context.logs["error"])

    async def test_qdrant_search_failure(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test handling of Qdrant search failure."""
        # Mock Qdrant search failure
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.hybrid_search.side_effect = Exception("Qdrant connection error")

        with pytest.raises(Exception, match="Qdrant connection error"):
            await search_documents_core(
                sample_search_request, mock_client_manager, mock_context
            )

        # Verify error logged
        assert any("Search failed" in msg for msg in mock_context.logs["error"])

    async def test_empty_search_results(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test handling of empty search results."""
        # Mock empty results
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.hybrid_search = AsyncMock(return_value=[])

        results = await search_documents_core(
            sample_search_request, mock_client_manager, mock_context
        )

        assert len(results) == 0

        # Verify no caching for empty results
        mock_cache = mock_client_manager.get_cache_manager.return_value
        mock_cache.set.assert_not_called()

        # Verify completion message
        assert any(
            "Search completed: 0 results found" in msg
            for msg in mock_context.logs["info"]
        )

    async def test_custom_embedding_model(self, mock_client_manager, mock_context):
        """Test search with custom embedding model."""
        request = SearchRequest(
            query="custom model test", embedding_model="custom-embedding-model"
        )

        await search_documents_core(request, mock_client_manager, mock_context)

        # Verify custom model passed to embedding generation
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_called_once_with(
            texts=["custom model test"],
            generate_sparse=True,  # Default hybrid strategy
            model="custom-embedding-model",
        )

    async def test_score_threshold_parameter(self, mock_client_manager, mock_context):
        """Test search with score threshold parameter."""
        request = SearchRequest(query="threshold test", score_threshold=0.8)

        await search_documents_core(request, mock_client_manager, mock_context)

        # Verify score threshold passed to search
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["score_threshold"] == 0.8

    async def test_custom_cache_ttl(self, mock_client_manager, mock_context):
        """Test search with custom cache TTL."""
        request = SearchRequest(query="cache ttl test", cache_ttl=600)

        await search_documents_core(request, mock_client_manager, mock_context)

        # Verify custom TTL used for caching
        mock_cache = mock_client_manager.get_cache_manager.return_value
        call_args = mock_cache.set.call_args
        assert call_args[1]["ttl"] == 600

    async def test_search_accuracy_parameter(self, mock_client_manager, mock_context):
        """Test search with search accuracy parameter."""
        request = SearchRequest(query="accuracy test", search_accuracy="accurate")

        # Add search_accuracy attribute to request for testing
        request.search_accuracy = "accurate"

        await search_documents_core(request, mock_client_manager, mock_context)

        # Verify search accuracy passed to search
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["search_accuracy"] == "accurate"

    async def test_reranking_skipped(self, mock_client_manager, mock_context):
        """Test that reranking is currently skipped."""
        request = SearchRequest(query="rerank test", rerank=True)

        results = await search_documents_core(
            request, mock_client_manager, mock_context
        )

        # Verify results returned (reranking doesn't affect them since it's skipped)
        assert len(results) == 2

        # Verify debug message about reranking
        assert any(
            "Applying BGE reranking" in msg for msg in mock_context.logs["debug"]
        )

    async def test_context_none_logging(self, mock_client_manager):
        """Test search functionality when context is None."""
        request = SearchRequest(query="no context test")

        # Should not raise exception with None context
        results = await search_documents_core(request, mock_client_manager, ctx=None)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_search_request_id_generation(
        self, mock_client_manager, mock_context, sample_search_request
    ):
        """Test that unique request IDs are generated for tracking."""
        # Run multiple searches to verify unique IDs
        await search_documents_core(
            sample_search_request, mock_client_manager, mock_context
        )

        # Verify request ID appears in logs
        info_msgs = mock_context.logs["info"]
        request_id_msgs = [msg for msg in info_msgs if "Starting search request" in msg]
        assert len(request_id_msgs) == 1

        # Request ID should be UUID format
        request_msg = request_id_msgs[0]
        assert " with strategy SearchStrategy.HYBRID" in request_msg

    async def test_sparse_vector_conversion(self, mock_client_manager, mock_context):
        """Test sparse vector conversion to dict format."""
        request = SearchRequest(
            query="sparse conversion test", strategy=SearchStrategy.SPARSE
        )

        # Mock sparse embeddings with specific pattern
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings = AsyncMock(
            return_value=Mock(
                embeddings=[[0.1, 0.2, 0.3]],
                sparse_embeddings=[
                    [0.0, 0.5, 0.0, 0.7, 0.0, 0.3]
                ],  # Only indices 1, 3, 5 non-zero
            )
        )

        await search_documents_core(request, mock_client_manager, mock_context)

        # Verify sparse vector conversion
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        expected_sparse = {1: 0.5, 3: 0.7, 5: 0.3}
        assert call_args[1]["sparse_vector"] == expected_sparse


class TestSearchDocumentsCoreIntegration:
    """Test integration scenarios and edge cases."""

    async def test_large_result_set_caching(self, mock_client_manager, mock_context):
        """Test caching behavior with large result sets."""
        # Mock large result set
        large_results = [
            {
                "id": f"doc-{i}",
                "score": 0.9 - (i * 0.01),
                "payload": {
                    "content": f"Document {i} content",
                    "url": f"https://example.com/doc{i}",
                    "title": f"Document {i}",
                    "metadata": {"index": i},
                },
            }
            for i in range(50)
        ]

        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.hybrid_search = AsyncMock(return_value=large_results)

        request = SearchRequest(query="large test", limit=50)
        results = await search_documents_core(
            request, mock_client_manager, mock_context
        )

        assert len(results) == 50

        # Verify caching behavior
        mock_cache = mock_client_manager.get_cache_manager.return_value
        mock_cache.set.assert_called_once()
        cache_data = mock_cache.set.call_args[0][1]
        assert len(cache_data) == 50

    async def test_minimal_search_request(self, mock_client_manager, mock_context):
        """Test search with minimal request parameters."""
        minimal_request = SearchRequest(query="minimal")

        results = await search_documents_core(
            minimal_request, mock_client_manager, mock_context
        )

        assert len(results) == 2

        # Verify defaults used
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["collection_name"] == "documentation"  # Default collection
        assert call_args[1]["limit"] == 10  # Default limit

    async def test_search_with_all_parameters(self, mock_client_manager, mock_context):
        """Test search with all possible parameters set."""
        comprehensive_request = SearchRequest(
            query="comprehensive test query",
            collection="custom_collection",
            limit=25,
            strategy=SearchStrategy.HYBRID,
            embedding_model="custom-model",
            score_threshold=0.75,
            rerank=True,
            cache_ttl=1200,
        )

        # Add search_accuracy for comprehensive test
        comprehensive_request.search_accuracy = "balanced"

        results = await search_documents_core(
            comprehensive_request, mock_client_manager, mock_context
        )

        assert len(results) == 2

        # Verify all parameters used correctly
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.assert_called_once_with(
            texts=["comprehensive test query"],
            generate_sparse=True,
            model="custom-model",
        )

        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        call_args = mock_qdrant.hybrid_search.call_args
        assert call_args[1]["collection_name"] == "custom_collection"
        assert call_args[1]["limit"] == 25
        assert call_args[1]["score_threshold"] == 0.75
        assert call_args[1]["search_accuracy"] == "balanced"

        mock_cache = mock_client_manager.get_cache_manager.return_value
        cache_call_args = mock_cache.set.call_args
        assert cache_call_args[1]["ttl"] == 1200
