"""Tests for MCP search utilities."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.config.enums import SearchStrategy, SearchAccuracy
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools._search_utils import search_documents_core

if TYPE_CHECKING:
    from fastmcp import Context
else:
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def mock_client_manager():
    """Create a mock client manager with all required services."""
    manager = Mock()
    
    # Mock cache manager
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)  # No cache hit by default
    mock_cache.set = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=mock_cache)
    
    # Mock embedding manager
    mock_embedding = AsyncMock()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_embedding_result.sparse_embeddings = None
    mock_embedding.generate_embeddings = AsyncMock(return_value=mock_embedding_result)
    manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)
    
    # Mock Qdrant service
    mock_qdrant = AsyncMock()
    mock_qdrant.hybrid_search = AsyncMock(return_value=[])
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)
    
    return manager


@pytest.mark.asyncio
async def test_search_documents_core_basic(mock_client_manager, mock_context):
    """Test basic search functionality."""
    # Setup request
    request = SearchRequest(
        query="test query",
        collection="documentation",
        limit=10,
        strategy=SearchStrategy.DENSE,
    )
    
    # Setup mock results
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.return_value = [
        {
            "id": "doc1",
            "score": 0.95,
            "payload": {
                "content": "Test content 1",
                "metadata": {"type": "documentation"},
            },
        },
        {
            "id": "doc2",
            "score": 0.85,
            "payload": {
                "content": "Test content 2",
                "metadata": {"type": "guide"},
            },
        },
    ]
    
    # Execute search
    results = await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify results
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].id == "doc1"
    assert results[0].content == "Test content 1"
    assert results[0].score == 0.95
    assert results[0].collection == "documentation"
    
    # Verify service calls
    mock_qdrant.hybrid_search.assert_called_once_with(
        collection_name="documentation",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=None,
        limit=10,
        score_threshold=0.0,
        fusion_type="rrf",
        search_accuracy="balanced",
    )
    
    # Verify caching
    mock_cache = await mock_client_manager.get_cache_manager()
    mock_cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_search_documents_core_with_cache_hit(mock_client_manager, mock_context):
    """Test search with cache hit."""
    request = SearchRequest(
        query="cached query",
        collection="documentation",
        limit=5,
    )
    
    # Setup cache hit
    cached_results = [
        {
            "id": "cached1",
            "content": "Cached content",
            "score": 0.9,
            "metadata": {},
            "collection": "documentation",
        }
    ]
    mock_cache = await mock_client_manager.get_cache_manager()
    mock_cache.get.return_value = cached_results
    
    # Execute search
    results = await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify cached results returned
    assert len(results) == 1
    assert results[0].id == "cached1"
    assert results[0].content == "Cached content"
    
    # Verify no embedding or search calls
    mock_embedding = await mock_client_manager.get_embedding_manager()
    mock_embedding.generate_embeddings.assert_not_called()
    
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.assert_not_called()


@pytest.mark.asyncio
async def test_search_documents_core_hybrid_search(mock_client_manager, mock_context):
    """Test hybrid search with dense and sparse vectors."""
    request = SearchRequest(
        query="hybrid search test",
        collection="docs",
        strategy=SearchStrategy.HYBRID,
        limit=3,
    )
    
    # Setup embedding result with sparse embeddings
    mock_embedding = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_embedding_result.sparse_embeddings = [[0.0, 0.5, 0.0, 0.8, 0.0]]
    mock_embedding.generate_embeddings.return_value = mock_embedding_result
    
    # Setup search results
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.return_value = [
        {
            "id": "hybrid1",
            "score": 0.92,
            "payload": {"content": "Hybrid result", "metadata": {}},
        }
    ]
    
    # Execute search
    results = await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify hybrid search was called with both vectors
    mock_qdrant.hybrid_search.assert_called_once_with(
        collection_name="docs",
        query_vector=[0.1, 0.2, 0.3],
        sparse_vector={1: 0.5, 3: 0.8},  # Non-zero indices
        limit=3,
        score_threshold=0.0,
        fusion_type="rrf",
        search_accuracy="balanced",
    )
    
    assert len(results) == 1
    assert results[0].content == "Hybrid result"


@pytest.mark.asyncio
async def test_search_documents_core_sparse_only(mock_client_manager, mock_context):
    """Test sparse-only search strategy."""
    request = SearchRequest(
        query="sparse search",
        collection="docs",
        strategy=SearchStrategy.SPARSE,
    )
    
    # Setup sparse embeddings
    mock_embedding = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2]]
    mock_embedding_result.sparse_embeddings = [[0.0, 0.3, 0.7, 0.0]]
    mock_embedding.generate_embeddings.return_value = mock_embedding_result
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify sparse-only search
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.assert_called_once_with(
        collection_name="docs",
        query_vector=[],  # Empty dense vector for sparse-only
        sparse_vector={1: 0.3, 2: 0.7},
        limit=10,
        score_threshold=0.0,
        fusion_type="rrf",
        search_accuracy="balanced",
    )


@pytest.mark.asyncio
async def test_search_documents_core_sparse_not_available(mock_client_manager, mock_context):
    """Test error when sparse embeddings not available for sparse search."""
    request = SearchRequest(
        query="sparse search",
        collection="docs",
        strategy=SearchStrategy.SPARSE,
    )
    
    # Setup without sparse embeddings
    mock_embedding = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2]]
    mock_embedding_result.sparse_embeddings = None
    mock_embedding.generate_embeddings.return_value = mock_embedding_result
    
    # Should raise error
    with pytest.raises(ValueError, match="Sparse embeddings not available"):
        await search_documents_core(request, mock_client_manager, mock_context)


@pytest.mark.asyncio
async def test_search_documents_core_with_filters(mock_client_manager, mock_context):
    """Test search with metadata filters."""
    request = SearchRequest(
        query="filtered search",
        collection="docs",
        filters={"type": "api", "version": "v2"},
        score_threshold=0.7,
    )
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify score threshold was passed
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    _, kwargs = mock_qdrant.hybrid_search.call_args
    assert kwargs["score_threshold"] == 0.7


@pytest.mark.asyncio
async def test_search_documents_core_with_custom_ttl(mock_client_manager, mock_context):
    """Test search with custom cache TTL."""
    request = SearchRequest(
        query="custom ttl",
        collection="docs",
        cache_ttl=600,  # 10 minutes
    )
    
    # Setup search results
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.return_value = [
        {
            "id": "ttl1",
            "score": 0.8,
            "payload": {"content": "TTL test", "metadata": {}},
        }
    ]
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify cache was set with custom TTL
    mock_cache = await mock_client_manager.get_cache_manager()
    _, kwargs = mock_cache.set.call_args
    assert kwargs["ttl"] == 600


@pytest.mark.asyncio
async def test_search_documents_core_with_search_accuracy(mock_client_manager, mock_context):
    """Test search with custom search accuracy."""
    request = SearchRequest(
        query="accurate search",
        collection="docs",
        search_accuracy=SearchAccuracy.HIGH,
    )
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify search accuracy was passed
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    _, kwargs = mock_qdrant.hybrid_search.call_args
    assert kwargs["search_accuracy"] == SearchAccuracy.HIGH


@pytest.mark.asyncio
async def test_search_documents_core_without_context(mock_client_manager):
    """Test search without context parameter."""
    request = SearchRequest(
        query="no context",
        collection="docs",
    )
    
    # Execute search without context
    results = await search_documents_core(request, mock_client_manager, ctx=None)
    
    # Should work normally
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_documents_core_error_handling(mock_client_manager, mock_context):
    """Test error handling during search."""
    request = SearchRequest(
        query="error test",
        collection="docs",
    )
    
    # Setup service to raise error
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.side_effect = Exception("Search service error")
    
    # Should propagate error
    with pytest.raises(Exception, match="Search service error"):
        await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify error was logged
    mock_context.error.assert_called_once()


@pytest.mark.asyncio
async def test_search_documents_core_empty_results(mock_client_manager, mock_context):
    """Test handling of empty search results."""
    request = SearchRequest(
        query="no results",
        collection="empty",
    )
    
    # Setup empty results
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.return_value = []
    
    # Execute search
    results = await search_documents_core(request, mock_client_manager, mock_context)
    
    # Should return empty list
    assert results == []
    
    # Should not cache empty results
    mock_cache = await mock_client_manager.get_cache_manager()
    mock_cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_search_documents_core_with_reranking(mock_client_manager, mock_context):
    """Test search with reranking enabled."""
    request = SearchRequest(
        query="rerank test",
        collection="docs",
        enable_reranking=True,
        rerank=True,
    )
    
    # Setup search results
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant.hybrid_search.return_value = [
        {
            "id": f"doc{i}",
            "score": 0.9 - i * 0.1,
            "payload": {"content": f"Content {i}", "metadata": {}},
        }
        for i in range(3)
    ]
    
    # Execute search
    results = await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify results (reranking is currently a no-op in the implementation)
    assert len(results) == 3
    assert results[0].score == 0.9
    
    # Verify debug message about reranking
    mock_context.debug.assert_any_call("Applying BGE reranking...")


@pytest.mark.asyncio
async def test_search_documents_core_request_id_tracking(mock_client_manager, mock_context):
    """Test that request ID is generated and tracked."""
    request = SearchRequest(
        query="tracking test",
        collection="docs",
    )
    
    # Mock uuid4
    test_uuid = "test-uuid-1234"
    with patch("src.mcp_tools.tools._search_utils.uuid4", return_value=test_uuid):
        await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify request ID was used in logging
    mock_context.info.assert_any_call(
        f"Starting search request {test_uuid} with strategy {SearchStrategy.HYBRID}"
    )


@pytest.mark.asyncio
async def test_search_documents_core_long_query_truncation(mock_client_manager, mock_context):
    """Test that long queries are truncated in debug logs."""
    long_query = "a" * 100  # Very long query
    request = SearchRequest(
        query=long_query,
        collection="docs",
    )
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify query was truncated in debug log
    mock_context.debug.assert_any_call(f"Generating embeddings for query: {long_query[:50]}...")


@pytest.mark.asyncio
async def test_search_documents_core_custom_embedding_model(mock_client_manager, mock_context):
    """Test search with custom embedding model."""
    request = SearchRequest(
        query="custom model",
        collection="docs",
        embedding_model="custom-model-v2",
    )
    
    # Execute search
    await search_documents_core(request, mock_client_manager, mock_context)
    
    # Verify custom model was passed to embedding manager
    mock_embedding = await mock_client_manager.get_embedding_manager()
    _, kwargs = mock_embedding.generate_embeddings.call_args
    assert kwargs["model"] == "custom-model-v2"