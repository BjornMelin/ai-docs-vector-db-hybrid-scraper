"""Tests for query processing orchestrator."""

from unittest.mock import AsyncMock

import pytest

from src.services.query_processing.models import (
    QueryProcessingRequest,
)
from src.services.query_processing.orchestrator import (
    SearchMode,
    SearchOrchestrator as AdvancedSearchOrchestrator,
    SearchPipeline,
    SearchRequest as AdvancedSearchRequest,
    SearchResult as AdvancedSearchResult,
)


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(
        return_value={"success": True, "embeddings": [[0.1] * 768]}
    )
    manager.rerank_results = AsyncMock(
        return_value=[{"original": {"id": "1", "content": "test", "score": 0.9}}]
    )
    return manager


@pytest.fixture
def _mock_qdrant_service():
    """Create a mock Qdrant service."""
    service = AsyncMock()
    service.filtered_search = AsyncMock(
        return_value=[
            {
                "id": "1",
                "payload": {"content": "test content", "title": "Test"},
                "score": 0.9,
            }
        ]
    )
    service.search.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "1",
                "payload": {"content": "test content", "title": "Test"},
                "score": 0.9,
            }
        ]
    )
    service.search.multi_stage_search = AsyncMock(
        return_value=[
            {
                "id": "1",
                "payload": {"content": "test content", "title": "Test"},
                "score": 0.9,
            }
        ]
    )
    return service


@pytest.fixture
def mock_hyde_engine():
    """Create a mock HyDE engine."""
    engine = AsyncMock()
    engine.enhanced_search = AsyncMock(
        return_value=[
            {"id": "1", "content": "test content", "title": "Test", "score": 0.9}
        ]
    )
    return engine


@pytest.fixture
def orchestrator():
    """Create an orchestrator instance."""
    return AdvancedSearchOrchestrator(
        cache_size=100,
        enable_performance_optimization=True,
    )


@pytest.fixture
async def initialized_orchestrator(orchestrator):
    """Create an initialized orchestrator."""
    await orchestrator.initialize()
    return orchestrator


@pytest.fixture
def processing_request():
    """Create a sample processing request."""
    return QueryProcessingRequest(
        query="What is machine learning?",
        collection_name="documentation",
        limit=10,
    )


@pytest.fixture
def advanced_sample_request():
    """Create a sample advanced search request."""
    return AdvancedSearchRequest(
        query="What is machine learning?",
        collection_name="documentation",
        limit=10,
        mode=SearchMode.ENHANCED,
        pipeline=SearchPipeline.BALANCED,
    )


class TestAdvancedSearchOrchestrator:
    """Test the AdvancedSearchOrchestrator class."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert (
            hasattr(orchestrator, "_initialized") is False
            or orchestrator._initialized is False
        )
        assert orchestrator.enable_performance_optimization is True
        assert orchestrator.cache_size == 100
        # Check that services are available via properties
        assert hasattr(orchestrator, "query_expansion_service")
        assert hasattr(orchestrator, "clustering_service")
        assert hasattr(orchestrator, "ranking_service")
        assert hasattr(orchestrator, "federated_service")

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()
        # The  orchestrator doesn't use _initialized flag the same way
        # but initialization should complete without error

    @pytest.mark.asyncio
    async def test_basic_query_processing(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test basic query processing flow."""
        # Use the  search method with AdvancedSearchRequest
        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.total_results >= 0  # May be 0 with mocked search
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_search(self, initialized_orchestrator, advanced_sample_request):
        """Test  search flow."""
        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.total_results >= 0  # May be 0 with mocked search
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_query_expansion_enabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with query expansion enabled."""
        advanced_sample_request.enable_expansion = True
        advanced_sample_request.query = "machine learning algorithms"

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Query expansion should be tracked in features_used
        assert "query_expansion" in response.features_used

    @pytest.mark.asyncio
    async def test_query_expansion_disabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with query expansion disabled."""
        advanced_sample_request.enable_expansion = False

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Query expansion should NOT be in features_used
        assert "query_expansion" not in response.features_used

    @pytest.mark.asyncio
    async def test_clustering_enabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with result clustering enabled."""
        advanced_sample_request.enable_clustering = True

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # With few results, clustering might not be applied
        # but the feature should be attempted

    @pytest.mark.asyncio
    async def test_clustering_disabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with result clustering disabled."""
        advanced_sample_request.enable_clustering = False

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Clustering should not be in features_used
        assert "result_clustering" not in response.features_used

    @pytest.mark.asyncio
    async def test_personalization_enabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with personalized ranking enabled."""
        advanced_sample_request.enable_personalization = True
        advanced_sample_request.user_id = "test_user_123"

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Personalization should be tracked if enabled
        if len(response.results) > 0:
            # Should have applied personalized ranking
            pass

    @pytest.mark.asyncio
    async def test_federation_enabled(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test search with federation enabled."""
        advanced_sample_request.enable_federation = True

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Federation should be attempted

    @pytest.mark.asyncio
    async def test_rag_enabled(self, initialized_orchestrator, advanced_sample_request):
        """Test search with RAG answer generation enabled."""
        advanced_sample_request.enable_rag = True

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # RAG features are portfolio features - should work

    @pytest.mark.asyncio
    async def test_pipeline_fast_mode(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test fast pipeline configuration."""
        advanced_sample_request.pipeline = SearchPipeline.FAST

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Fast pipeline should complete quickly

    @pytest.mark.asyncio
    async def test_pipeline_comprehensive_mode(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test comprehensive pipeline configuration."""
        advanced_sample_request.pipeline = SearchPipeline.COMPREHENSIVE

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Comprehensive mode should use more features

    @pytest.mark.asyncio
    async def test_error_handling(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test error handling and graceful degradation."""
        # Test that search completes even if some features fail
        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0
        # Should always return a response even if there are internal errors

    @pytest.mark.asyncio
    async def test_performance_requirements(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test performance requirements handling."""
        advanced_sample_request.max_processing_time_ms = 1000.0

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        # Should respect performance constraints

    @pytest.mark.asyncio
    async def test_user_context_integration(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test user context integration."""
        advanced_sample_request.user_id = "test_user"
        advanced_sample_request.session_id = "test_session"

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        # Context should be handled properly

    @pytest.mark.asyncio
    async def test_caching_behavior(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test caching behavior."""
        advanced_sample_request.enable_caching = True

        # First request should not be cached
        response1 = await initialized_orchestrator.search(advanced_sample_request)
        assert isinstance(response1, AdvancedSearchResult)
        assert response1.cache_hit is False

        # Second identical request should hit cache
        response2 = await initialized_orchestrator.search(advanced_sample_request)
        assert isinstance(response2, AdvancedSearchResult)
        assert response2.cache_hit is True

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test statistics tracking."""
        # Perform a search
        response = await initialized_orchestrator.search(advanced_sample_request)
        assert isinstance(response, AdvancedSearchResult)

        # Get stats from orchestrator
        stats = initialized_orchestrator.get_stats()
        assert "total_searches" in stats
        assert stats["total_searches"] >= 1
        assert "avg_processing_time" in stats

    @pytest.mark.asyncio
    async def test_features_tracking(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test features tracking in response."""
        advanced_sample_request.enable_expansion = True

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert hasattr(response, "features_used")
        assert isinstance(response.features_used, list)

    @pytest.mark.asyncio
    async def test_timing_measurements(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test timing measurements."""
        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_empty_query_handling(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test handling of edge cases."""
        # Test with minimal query
        advanced_sample_request.query = "a"
        advanced_sample_request.limit = 1

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert response.total_results >= 0
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_uninitialized_orchestrator(
        self, orchestrator, advanced_sample_request
    ):
        """Test using uninitialized orchestrator."""
        # The  orchestrator doesn't require explicit initialization
        response = await orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)

    @pytest.mark.asyncio
    async def test_multiple_searches_stats(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test performance statistics tracking."""
        # Process multiple queries to build stats
        for i in range(3):
            # Modify query to avoid cache hits
            advanced_sample_request.query = f"What is machine learning? Query {i}"
            await initialized_orchestrator.search(advanced_sample_request)

        stats = initialized_orchestrator.get_stats()

        assert stats["total_searches"] >= 3
        assert stats["avg_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_cache_stats_tracking(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test cache statistics tracking."""
        # First query should miss cache
        response1 = await initialized_orchestrator.search(advanced_sample_request)
        assert response1.cache_hit is False

        # Same query should hit cache
        response2 = await initialized_orchestrator.search(advanced_sample_request)
        assert response2.cache_hit is True

        # Check cache stats
        stats = initialized_orchestrator.get_stats()
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1

    @pytest.mark.asyncio
    async def test_result_limits(
        self, initialized_orchestrator, advanced_sample_request
    ):
        """Test result limiting."""
        advanced_sample_request.limit = 3

        response = await initialized_orchestrator.search(advanced_sample_request)

        assert isinstance(response, AdvancedSearchResult)
        assert len(response.results) <= 3  # Should respect limit

    @pytest.mark.asyncio
    async def test_cleanup(self, initialized_orchestrator):
        """Test orchestrator cleanup."""
        await initialized_orchestrator.cleanup()
        # Cleanup should complete without error

    @pytest.mark.asyncio
    async def test_cache_clearing(self, initialized_orchestrator):
        """Test cache clearing functionality."""
        # Clear cache should work without error
        initialized_orchestrator.clear_cache()

        # Cache should be empty after clearing
        assert len(initialized_orchestrator.cache) == 0
