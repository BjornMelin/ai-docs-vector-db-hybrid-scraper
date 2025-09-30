"""Tests for the search document core utilities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from src.config import SearchStrategy
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.tools._search_utils import search_documents_core


@pytest.fixture
async def mock_client_manager():
    """Builds a minimal async client manager stub for search tests."""
    cache = AsyncMock()
    cache.get = AsyncMock()
    cache.set = AsyncMock()

    embeddings = AsyncMock()
    embeddings.generate_embeddings = AsyncMock()

    qdrant = AsyncMock()
    qdrant.hybrid_search = AsyncMock()

    manager = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=cache)
    manager.get_embedding_manager = AsyncMock(return_value=embeddings)
    manager.get_qdrant_service = AsyncMock(return_value=qdrant)
    return manager


@pytest.fixture
def ctx():
    """Provide a minimal context with async logging hooks."""
    return SimpleNamespace(
        info=AsyncMock(),
        debug=AsyncMock(),
        warning=AsyncMock(),
        error=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_search_documents_core_returns_cached_results(mock_client_manager, ctx):
    """Cached payloads should be returned without hitting downstream services."""
    cached = [{"id": "1", "content": "cached", "metadata": {}, "score": 0.9}]
    cache = await mock_client_manager.get_cache_manager()
    cache.get.return_value = cached

    request = SearchRequest(query="cached query")

    results = await search_documents_core(request, mock_client_manager, ctx)

    assert [result.content for result in results] == ["cached"]
    (
        await mock_client_manager.get_embedding_manager()
    ).generate_embeddings.assert_not_called()
    (await mock_client_manager.get_qdrant_service()).hybrid_search.assert_not_called()
    cache.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_documents_core_executes_search_and_caches(
    mock_client_manager, ctx
):
    """A cache miss should trigger embedding generation, search, and caching."""
    cache = await mock_client_manager.get_cache_manager()
    cache.get.return_value = None

    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.generate_embeddings.return_value = SimpleNamespace(
        embeddings=[[0.1, 0.2]],
        sparse_embeddings=None,
    )

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.hybrid_search.return_value = [
        {"id": UUID(int=1), "score": 0.5, "payload": {"content": "match"}}
    ]

    request = SearchRequest(query="fresh query", limit=1)

    results = await search_documents_core(request, mock_client_manager, ctx)

    assert len(results) == 1
    assert results[0].content == "match"
    embedding_manager.generate_embeddings.assert_awaited_once()
    qdrant.hybrid_search.assert_awaited_once()
    cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_documents_core_returns_empty_results_on_cache_miss(
    mock_client_manager, ctx
):
    """Hybrid search should return an empty list when Qdrant finds no matches."""
    cache = await mock_client_manager.get_cache_manager()
    cache.get.return_value = None

    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.generate_embeddings.return_value = SimpleNamespace(
        embeddings=[[0.3, 0.4]],
        sparse_embeddings=None,
    )

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.hybrid_search.return_value = []

    request = SearchRequest(query="no results expected", limit=3)

    results = await search_documents_core(request, mock_client_manager, ctx)

    assert results == []
    cache.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_documents_core_sparse_without_sparse_embeddings(
    mock_client_manager, ctx
):
    """Sparse search without sparse embeddings should raise an informative error."""
    cache = await mock_client_manager.get_cache_manager()
    cache.get.return_value = None

    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.generate_embeddings.return_value = SimpleNamespace(
        embeddings=[[0.1]],
        sparse_embeddings=[],
    )

    request = SearchRequest(query="need sparse", strategy=SearchStrategy.SPARSE)

    with pytest.raises(ValueError, match="Sparse embeddings not available"):
        await search_documents_core(request, mock_client_manager, ctx)
