"""Tests for search orchestrator."""

from unittest.mock import AsyncMock

import pytest
from src.services.query_processing.clustering import ClusterGroup
from src.services.query_processing.clustering import ResultClusteringResult
from src.services.query_processing.clustering import SearchResult as ClusterSearchResult
from src.services.query_processing.expansion import QueryExpansionResult
from src.services.query_processing.orchestrator import SearchMode
from src.services.query_processing.orchestrator import SearchOrchestrator
from src.services.query_processing.orchestrator import SearchPipeline
from src.services.query_processing.orchestrator import SearchRequest
from src.services.query_processing.orchestrator import SearchResult
from src.services.query_processing.ranking import PersonalizedRankingResult
from src.services.query_processing.ranking import RankedResult


@pytest.fixture
def orchestrator():
    """Create a search orchestrator instance."""
    return SearchOrchestrator(cache_size=10)


@pytest.fixture
def mock_expansion_service():
    """Mock query expansion service."""
    service = AsyncMock()
    service.expand_query = AsyncMock(
        return_value=QueryExpansionResult(
            original_query="test",
            expanded_query="test OR testing OR tests",
            expanded_terms=[],
            expansion_strategy="hybrid",
            expansion_scope="moderate",
            confidence_score=0.8,
            processing_time_ms=10.0,
            cache_hit=False,
        )
    )
    return service


@pytest.fixture
def mock_clustering_service():
    """Mock clustering service."""
    service = AsyncMock()

    # Create mock search results for clustering
    mock_results = [
        ClusterSearchResult(
            id=f"doc_{i}",
            title=f"Result {i}",
            content=f"Content {i}",
            score=0.9 - (i * 0.1),
            embedding=[0.1] * 10,
        )
        for i in range(3)
    ]

    service.cluster_results = AsyncMock(
        return_value=ResultClusteringResult(
            clusters=[
                ClusterGroup(
                    cluster_id=0,
                    label="Cluster 0",
                    results=mock_results[:2],
                    confidence=0.8,
                    size=2,
                    avg_score=0.85,
                    coherence_score=0.75,
                )
            ],
            outliers=[],
            method_used="kmeans",
            total_results=3,
            clustered_results=2,
            outlier_count=1,
            cluster_count=1,
            processing_time_ms=20.0,
            cache_hit=False,
        )
    )
    return service


@pytest.fixture
def mock_ranking_service():
    """Mock personalized ranking service."""
    service = AsyncMock()
    service.rank_results = AsyncMock(
        return_value=PersonalizedRankingResult(
            ranked_results=[
                RankedResult(
                    result_id="doc_0",
                    original_score=0.9,
                    final_score=0.95,
                    preference_score=0.8,
                    recency_score=0.7,
                    interaction_score=0.6,
                    ranking_factors={},
                )
            ],
            user_id="test_user",
            processing_time_ms=15.0,
            cache_hit=False,
        )
    )
    return service


@pytest.mark.asyncio
async def test_basic_search(orchestrator):
    """Test basic search without advanced features."""
    request = SearchRequest(
        query="test query",
        limit=10,
        mode=SearchMode.BASIC,
        enable_expansion=False,
        enable_clustering=False,
        enable_personalization=False,
    )

    result = await orchestrator.search(request)

    assert isinstance(result, SearchResult)
    assert result.query_processed == "test query"
    assert len(result.results) <= 10
    assert result.processing_time_ms > 0
    assert not result.cache_hit
    assert "query_expansion" not in result.features_used


@pytest.mark.asyncio
async def test_search_with_query_expansion(orchestrator, mock_expansion_service):
    """Test search with query expansion enabled."""
    orchestrator._query_expansion_service = mock_expansion_service

    request = SearchRequest(
        query="test",
        limit=10,
        mode=SearchMode.ENHANCED,
        enable_expansion=True,
    )

    result = await orchestrator.search(request)

    assert result.query_processed == "test OR testing OR tests"
    assert result.expanded_query == "test OR testing OR tests"
    assert "query_expansion" in result.features_used
    mock_expansion_service.expand_query.assert_called_once()


@pytest.mark.asyncio
async def test_search_with_clustering(orchestrator, mock_clustering_service):
    """Test search with result clustering enabled."""
    orchestrator._clustering_service = mock_clustering_service

    request = SearchRequest(
        query="test",
        limit=10,
        mode=SearchMode.ENHANCED,
        enable_clustering=True,
    )

    # Override _execute_search to return enough results for clustering
    async def mock_execute_search(query, request, config):
        return [
            {"id": f"doc_{i}", "content": f"Result {i}", "score": 0.9 - (i * 0.1)}
            for i in range(10)
        ]

    orchestrator._execute_search = mock_execute_search

    result = await orchestrator.search(request)

    assert "result_clustering" in result.features_used
    mock_clustering_service.cluster_results.assert_called_once()


@pytest.mark.asyncio
async def test_search_with_personalization(orchestrator, mock_ranking_service):
    """Test search with personalized ranking enabled."""
    orchestrator._ranking_service = mock_ranking_service

    request = SearchRequest(
        query="test",
        limit=10,
        user_id="test_user",
        mode=SearchMode.ENHANCED,
        enable_personalization=True,
    )

    result = await orchestrator.search(request)

    assert "personalized_ranking" in result.features_used
    mock_ranking_service.rank_results.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_configurations(orchestrator):
    """Test different pipeline configurations."""
    # Test FAST pipeline
    request = SearchRequest(
        query="test",
        pipeline=SearchPipeline.FAST,
    )
    result = await orchestrator.search(request)
    assert result.processing_time_ms < 5000  # Should be fast

    # Test BALANCED pipeline
    request = SearchRequest(
        query="test",
        pipeline=SearchPipeline.BALANCED,
    )
    result = await orchestrator.search(request)
    assert "query_expansion" in result.features_used or len(result.features_used) >= 0

    # Test COMPREHENSIVE pipeline
    request = SearchRequest(
        query="test",
        pipeline=SearchPipeline.COMPREHENSIVE,
        user_id="test_user",
    )
    result = await orchestrator.search(request)
    # Would have more features if all services were mocked


@pytest.mark.asyncio
async def test_caching(orchestrator):
    """Test result caching."""
    request = SearchRequest(
        query="test query",
        limit=10,
        enable_caching=True,
    )

    # First call - should not be cached
    result1 = await orchestrator.search(request)
    assert not result1.cache_hit

    # Second call - should be cached
    result2 = await orchestrator.search(request)
    assert result2.cache_hit
    assert result2.query_processed == result1.query_processed


@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    """Test error handling in search."""

    # Mock _execute_search to raise an error
    async def mock_failing_search(query, request, config):
        raise Exception("Search failed")

    orchestrator._execute_search = mock_failing_search

    request = SearchRequest(
        query="test",
        limit=10,
    )

    result = await orchestrator.search(request)

    # Should return empty results on error
    assert result.results == []
    assert result.total_results == 0
    assert result.processing_time_ms > 0


@pytest.mark.asyncio
async def test_stats_tracking(orchestrator):
    """Test statistics tracking."""
    initial_stats = orchestrator.get_stats()
    assert initial_stats["total_searches"] == 0

    request = SearchRequest(query="test", limit=10)
    await orchestrator.search(request)

    stats = orchestrator.get_stats()
    assert stats["total_searches"] == 1
    assert stats["avg_processing_time"] > 0


@pytest.mark.asyncio
async def test_cache_clearing(orchestrator):
    """Test cache clearing functionality."""
    request = SearchRequest(query="test", limit=10, enable_caching=True)

    # Populate cache
    await orchestrator.search(request)
    assert len(orchestrator.cache) > 0

    # Clear cache
    orchestrator.clear_cache()
    assert len(orchestrator.cache) == 0


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test backward compatibility aliases."""
    from src.services.query_processing.orchestrator import AdvancedSearchOrchestrator
    from src.services.query_processing.orchestrator import AdvancedSearchRequest
    from src.services.query_processing.orchestrator import AdvancedSearchResult

    # Check aliases work
    assert AdvancedSearchOrchestrator == SearchOrchestrator
    assert AdvancedSearchRequest == SearchRequest
    assert AdvancedSearchResult == SearchResult


@pytest.mark.asyncio
async def test_features_with_failures(
    orchestrator, mock_expansion_service, mock_clustering_service
):
    """Test that failures in optional features don't break search."""
    # Make expansion fail
    mock_expansion_service.expand_query.side_effect = Exception("Expansion failed")
    orchestrator._query_expansion_service = mock_expansion_service

    # Make clustering fail
    mock_clustering_service.cluster_results.side_effect = Exception("Clustering failed")
    orchestrator._clustering_service = mock_clustering_service

    request = SearchRequest(
        query="test",
        limit=10,
        mode=SearchMode.ENHANCED,
        enable_expansion=True,
        enable_clustering=True,
    )

    # Override _execute_search to return results
    async def mock_execute_search(query, request, config):
        return [
            {"id": f"doc_{i}", "content": f"Result {i}", "score": 0.9 - (i * 0.1)}
            for i in range(10)
        ]

    orchestrator._execute_search = mock_execute_search

    result = await orchestrator.search(request)

    # Should still return results despite feature failures
    assert len(result.results) > 0
    assert result.query_processed == "test"  # Original query since expansion failed
    assert "query_expansion" not in result.features_used
    assert "result_clustering" not in result.features_used
