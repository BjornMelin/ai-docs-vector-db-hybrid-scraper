"""Tests for the  hybrid search service.

This module contains comprehensive tests for the AdvancedHybridSearchService
including query classification, model selection, adaptive fusion, and A/B testing.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ABTestVariant, Config, ModelType, OptimizationStrategy, QueryType
from src.models.vector_search import (
    ABTestConfig,
    FusionConfig,
    HybridSearchRequest,
    HybridSearchResponse,
    ModelSelectionStrategy,
    QueryClassification,
    SearchAccuracy,
    SecureSearchParamsModel,
)
from src.services.errors import QdrantServiceError
from src.services.query_processing.models import (
    QueryComplexity,
)
from src.services.query_processing.orchestrator import (
    SearchMode,
    SearchPipeline,
    SearchResult as AdvancedSearchResult,
)
from src.services.vector_db.hybrid_search import HybridSearchService


class TestAdvancedHybridSearchService:
    """Test suite for HybridSearchService."""

    @pytest.fixture
    def mock_client(self):
        """Mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self):
        """Mock unified configuration."""
        config = MagicMock(spec=Config)
        config.embedding_cost_budget = 1000.0
        return config

    @pytest.fixture
    def mock_qdrant_search(self):
        """Mock QdrantSearch service."""
        mock_search = AsyncMock()
        mock_search.filtered_search.return_value = [
            {"id": "doc1", "score": 0.9, "payload": {"title": "Test Document 1"}},
            {"id": "doc2", "score": 0.8, "payload": {"title": "Test Document 2"}},
        ]
        mock_search.hybrid_search.return_value = [
            {"id": "doc1", "score": 0.95, "payload": {"title": "Test Document 1"}},
            {"id": "doc3", "score": 0.85, "payload": {"title": "Test Document 3"}},
        ]
        return mock_search

    @pytest.fixture
    def service(self, mock_client, mock_config, mock_qdrant_search):
        """Create HybridSearchService instance."""
        return HybridSearchService(mock_client, mock_config, mock_qdrant_search)

    @pytest.fixture
    def sample_request(self):
        """Create sample  hybrid search request."""
        return HybridSearchRequest(
            query="How to implement async functions in Python?",
            collection_name="test_collection",
            limit=5,
            search_params=SecureSearchParamsModel(
                accuracy_level=SearchAccuracy.BALANCED, hnsw_ef=64
            ),
            fusion_config=FusionConfig(algorithm="rrf"),
            enable_query_classification=True,
            enable_model_selection=True,
            enable_adaptive_fusion=True,
            enable_splade=True,
            user_id="test_user",  # Test data
            session_id="test_session",  # Test data
        )

    @pytest.fixture
    def sample_query_classification(self):
        """Create sample query classification."""
        return QueryClassification(
            query_type=QueryType.CODE,
            complexity_level=QueryComplexity.MODERATE,
            domain="programming",
            programming_language="python",
            is_multimodal=False,
            confidence=0.85,
            features={"has_code_keywords": True, "query_length": 8},
        )

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization."""
        assert service.client is not None
        assert service.config is not None
        assert service.qdrant_search is not None
        assert service.query_classifier is not None
        assert service.adaptive_fusion_tuner is not None
        assert service.model_selector is not None
        assert service.splade_provider is not None
        assert service.enable_fallback is True
        assert service.fallback_timeout_ms == 5000

    @pytest.mark.asyncio
    async def test_service_initialization_process(self, service):
        """Test service initialization process."""
        with patch.object(service.splade_provider, "initialize") as mock_init:
            mock_init.return_value = None
            await service.initialize()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_basic_hybrid_search(self, service, sample_request):
        """Test basic  hybrid search functionality."""

        # Mock all components including the orchestrator
        with (
            patch.object(service.query_classifier, "classify_query") as mock_classify,
            patch.object(service.model_selector, "select_optimal_model") as mock_model,
            patch.object(
                service.splade_provider, "generate_sparse_vector"
            ) as mock_splade,
            patch.object(
                service.adaptive_fusion_tuner, "compute_adaptive_weights"
            ) as mock_fusion,
            patch.object(service.orchestrator, "search") as mock_orchestrator,
        ):
            # Setup mocks
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.CODE,
                complexity_level=QueryComplexity.MODERATE,
                domain="programming",
                programming_language="python",
                is_multimodal=False,
                confidence=0.85,
                features={},
            )

            mock_model.return_value = ModelSelectionStrategy(
                primary_model="text-embedding-3-small",
                model_type=ModelType.GENERAL_PURPOSE,
                fallback_models=[],
                selection_rationale="Optimal for code documentation",
                expected_performance=0.85,
                cost_efficiency=0.9,
                query_classification=mock_classify.return_value,
            )
            mock_splade.return_value = {1: 0.8, 2: 0.6, 3: 0.4}
            mock_fusion.return_value = MagicMock(
                dense_weight=0.7,
                sparse_weight=0.3,
                effectiveness_score=MagicMock(hybrid_effectiveness=0.85),
            )

            # Mock orchestrator response

            mock_orchestrator.return_value = AdvancedSearchResult(
                results=[
                    {
                        "id": f"result_{i}",
                        "score": 0.9 - i * 0.1,
                        "payload": {
                            "id": f"result_{i}",
                            "title": f"Search Result {i}",
                            "content": f"Content for result {i} matching query: "
                            f"{sample_request.query}",
                            "score": 0.9 - i * 0.1,
                            "content_type": "documentation" if i % 2 == 0 else "code",
                            "published_date": "2024-01-01T00:00:00Z",
                            "metadata": {
                                "source": "mock_search",
                                "processing_stage": "core_search",
                            },
                            "final_rank": i + 1,
                            "pipeline": "fast",
                            "processing_timestamp": "2025-06-12T20:29:48.864882",
                        },
                        "vector": None,
                    }
                    for i in range(4)
                ],
                _total_results=4,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                query_processed="How to implement async functions in Python?",
                stage_results=[],
                _total_processing_time_ms=150.0,
                quality_score=0.85,
                diversity_score=0.7,
                relevance_score=0.8,
                features_used=["adaptive_fusion", "query_classification"],
                search_metadata={
                    "vector_generation_time_ms": 50.0,
                    "cache_hit": False,
                    "fusion_weights": {"dense": 0.7, "sparse": 0.3},
                    "effectiveness_score": 0.85,
                },
                optimizations_applied=["adaptive_fusion", "query_classification"],
                success=True,
            )

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert len(response.results) > 0
            assert response.query_classification is not None
            assert response.model_selection is not None
            assert response.fusion_weights is not None
            assert response.optimization_applied is True
            assert response.retrieval_metrics is not None

    @pytest.mark.asyncio
    async def test_search_with_query_classification_disabled(
        self, service, sample_request
    ):
        """Test search with query classification disabled."""
        sample_request.enable_query_classification = False

        response = await service.hybrid_search(sample_request)

        assert isinstance(response, HybridSearchResponse)
        assert response.query_classification is None
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_with_model_selection_disabled(self, service, sample_request):
        """Test search with model selection disabled."""
        sample_request.enable_model_selection = False

        response = await service.hybrid_search(sample_request)

        assert isinstance(response, HybridSearchResponse)
        assert response.model_selection is None

    @pytest.mark.asyncio
    async def test_search_with_splade_disabled(self, service, sample_request):
        """Test search with SPLADE disabled."""
        sample_request.enable_splade = False

        response = await service.hybrid_search(sample_request)

        assert isinstance(response, HybridSearchResponse)
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_with_adaptive_fusion_disabled(self, service, sample_request):
        """Test search with adaptive fusion disabled."""
        sample_request.enable_adaptive_fusion = False

        response = await service.hybrid_search(sample_request)

        assert isinstance(response, HybridSearchResponse)
        assert response.fusion_weights is None
        # Other optimizations (query classification, model selection) may still be
        # applied
        # so optimization_applied can be True even when adaptive fusion is disabled

    @pytest.mark.asyncio
    async def test_ab_test_assignment(self, service, sample_request):
        """Test A/B test variant assignment."""
        sample_request.ab_test_config = ABTestConfig(
            experiment_name="test_experiment",  # Test data
            variants=[ABTestVariant.CONTROL, ABTestVariant.RRF_OPTIMIZED],
            traffic_allocation={"control": 0.5, "rrf_optimized": 0.5},
        )

        response = await service.hybrid_search(sample_request)

        assert response.ab_test_variant in [
            ABTestVariant.CONTROL,
            ABTestVariant.RRF_OPTIMIZED,
        ]

    @pytest.mark.asyncio
    async def test_ab_test_consistent_assignment(self, service, sample_request):
        """Test that A/B test assignment is consistent for same user."""
        sample_request.ab_test_config = ABTestConfig(
            experiment_name="test_experiment",  # Test data
            variants=[ABTestVariant.CONTROL, ABTestVariant.RRF_OPTIMIZED],
            traffic_allocation={"control": 0.5, "rrf_optimized": 0.5},
        )

        # Run multiple times with same user ID
        variants = []
        for _ in range(5):
            response = await service.hybrid_search(sample_request)
            variants.append(response.ab_test_variant)

        # All variants should be the same for consistent user experience
        assert len(set(variants)) == 1

    @pytest.mark.asyncio
    async def test_query_classification_timeout(self, service, sample_request):
        """Test query classification timeout handling."""
        with patch.object(service.query_classifier, "classify_query") as mock_classify:
            # Simulate timeout
            mock_classify.side_effect = TimeoutError()

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert response.query_classification is None
            assert len(response.results) > 0  # Should fallback gracefully

    @pytest.mark.asyncio
    async def test_model_selection_timeout(self, service, sample_request):
        """Test model selection timeout handling."""
        with (
            patch.object(service.query_classifier, "classify_query") as mock_classify,
            patch.object(service.model_selector, "select_optimal_model") as mock_model,
        ):
            mock_classify.return_value = MagicMock()
            mock_model.side_effect = TimeoutError()

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert response.model_selection is None
            assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_splade_generation_timeout(self, service, sample_request):
        """Test SPLADE generation timeout handling."""
        with patch.object(
            service.splade_provider, "generate_sparse_vector"
        ) as mock_splade:
            mock_splade.side_effect = TimeoutError()

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_adaptive_fusion_error_handling(self, service, sample_request):
        """Test adaptive fusion error handling."""
        with (
            patch.object(service.query_classifier, "classify_query") as mock_classify,
            patch.object(
                service.adaptive_fusion_tuner, "compute_adaptive_weights"
            ) as mock_fusion,
        ):
            mock_classify.return_value = MagicMock()
            mock_fusion.side_effect = Exception("Fusion error")

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert response.fusion_weights is None
            assert response.effectiveness_score is None
            assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_fallback_search_on_error(self, service, sample_request):
        """Test fallback search when the orchestrator fails."""
        with patch.object(service.orchestrator, "search") as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Orchestrator error")

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert response.fallback_reason is not None
            # Fallback reason should mention failure/error
            assert any(
                word in response.fallback_reason.lower()
                for word in ["error", "failed", "failure"]
            )
            # When all search attempts fail, results may be empty
            assert isinstance(response.results, list)

    @pytest.mark.asyncio
    async def test_fallback_disabled_error_propagation(self, service, sample_request):
        """Test error propagation when fallback is disabled."""
        service.enable_fallback = False

        with patch.object(service.orchestrator, "search") as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Orchestrator error")

            with pytest.raises(
                QdrantServiceError, match="Advanced hybrid search failed"
            ):
                await service.hybrid_search(sample_request)

    @pytest.mark.asyncio
    async def test_search_metrics_calculation(self, service, sample_request):
        """Test search metrics calculation."""
        response = await service.hybrid_search(sample_request)

        assert response.retrieval_metrics is not None
        assert response.retrieval_metrics._total_time_ms > 0
        assert response.retrieval_metrics.results_count == len(response.results)
        assert (
            response.retrieval_metrics.hnsw_ef_used
            == sample_request.search_params.hnsw_ef
        )

    # Note: _generate_dense_vector and _execute_search_strategies methods were removed
    # when the service was refactored to use the AdvancedSearchOrchestrator.
    # These implementation details are now handled by the orchestrator.

    # Note: _apply_weighted_fusion method was also removed when the service
    # was refactored to use the AdvancedSearchOrchestrator.
    # Weighted fusion is now handled by the orchestrator's fusion algorithms.

    @pytest.mark.asyncio
    async def test_performance_statistics(self, service):
        """Test performance statistics retrieval."""
        # Add some mock metrics
        service.search_metrics["test1"] = MagicMock(_total_time_ms=100.0)
        service.search_metrics["test2"] = MagicMock(_total_time_ms=200.0)

        with (
            patch.object(
                service.adaptive_fusion_tuner, "get_performance_stats"
            ) as mock_stats,
            patch.object(service.splade_provider, "get_cache_stats") as mock_cache,
        ):
            mock_stats.return_value = {"_total_queries": 10}
            mock_cache.return_value = {"cache_size": 5}

            stats = service.get_performance_statistics()

            assert stats["_total_searches"] == 2
            assert stats["average_search_time"] == 150.0
            assert "fusion_tuner_stats" in stats
            assert "splade_cache_stats" in stats

    @pytest.mark.asyncio
    async def test_user_feedback_processing(self, service):
        """Test user feedback processing."""
        query_id = str(uuid.uuid4())
        feedback = {"satisfaction": 0.8, "clicked": True, "dwell_time": 45}

        # This should not raise an exception
        await service.update_with_user_feedback(query_id, feedback)

    @pytest.mark.asyncio
    async def test_search_for_learning_storage(self, service, sample_request):
        """Test storage of search results for learning."""
        query_id = str(uuid.uuid4())
        response = HybridSearchResponse(
            results=[],
            retrieval_metrics={
                "_total_time_ms": 100.0,
                "results_count": 0,
                "search_time_ms": 50.0,
                "query_vector_time_ms": 25.0,
                "filtered_count": 0,
                "cache_hit": False,
                "hnsw_ef_used": 64,
            },
        )

        with patch.object(service.model_selector, "update_performance_history"):
            await service._store_search_for_learning(query_id, sample_request, response)

            assert query_id in service.search_metrics

    @pytest.mark.asyncio
    async def test_format_search_results(self, service):
        """Test search results formatting."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.9,
                "payload": {"title": "Doc 1"},
                "vector": [0.1, 0.2],
            },
            {"id": "doc2", "score": 0.8, "payload": {"title": "Doc 2"}},
        ]

        formatted = service._format_search_results(raw_results)

        assert len(formatted) == 2
        assert formatted[0].id == "doc1"
        assert formatted[0].score == 0.9
        assert formatted[0].payload == {"title": "Doc 1"}
        assert formatted[0].vector == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, service, sample_request):
        """Test handling of empty search results."""

        with patch.object(service.orchestrator, "search") as mock_orchestrator:
            mock_orchestrator.return_value = AdvancedSearchResult(
                results=[],
                _total_results=0,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                query_processed="How to implement async functions in Python?",
                stage_results=[],
                _total_processing_time_ms=100.0,
                quality_score=0.0,
                diversity_score=0.0,
                relevance_score=0.0,
                features_used=[],
                search_metadata={},
                optimizations_applied=[],
                success=True,
            )

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert len(response.results) == 0
            assert response.retrieval_metrics.results_count == 0

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self, service, sample_request):
        """Test handling of large result sets."""

        # Mock large result set limited by orchestrator
        large_results = [
            {
                "id": f"doc{i}",
                "score": 0.9 - i * 0.01,
                "payload": {"title": f"Doc {i}"},
                "vector": None,
            }
            for i in range(sample_request.limit)  # Orchestrator respects the limit
        ]

        with patch.object(service.orchestrator, "search") as mock_orchestrator:
            mock_orchestrator.return_value = AdvancedSearchResult(
                results=large_results,
                _total_results=sample_request.limit,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                query_processed="How to implement async functions in Python?",
                stage_results=[],
                _total_processing_time_ms=150.0,
                quality_score=0.85,
                diversity_score=0.7,
                relevance_score=0.8,
                features_used=[],
                search_metadata={},
                optimizations_applied=[],
                success=True,
            )

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert (
                len(response.results) == sample_request.limit
            )  # Should be limited to request limit

    @pytest.mark.parametrize(
        "query_type",
        [
            QueryType.CODE,
            QueryType.DOCUMENTATION,
            QueryType.CONCEPTUAL,
            QueryType.API_REFERENCE,
            QueryType.TROUBLESHOOTING,
            QueryType.MULTIMODAL,
        ],
    )
    @pytest.mark.asyncio
    async def test_different_query_types(self, service, sample_request, query_type):
        """Test search with different query types."""
        with patch.object(service.query_classifier, "classify_query") as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=query_type,
                complexity_level=QueryComplexity.MODERATE,
                domain="general",
                programming_language=None,
                is_multimodal=False,
                confidence=0.8,
                features={},
            )

            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
            assert response.query_classification.query_type == query_type

    @pytest.mark.parametrize(
        "optimization_strategy",
        [
            OptimizationStrategy.SPEED_OPTIMIZED,
            OptimizationStrategy.QUALITY_OPTIMIZED,
            OptimizationStrategy.COST_OPTIMIZED,
            OptimizationStrategy.BALANCED,
        ],
    )
    @pytest.mark.asyncio
    async def test_different_optimization_strategies(
        self, service, sample_request, optimization_strategy
    ):
        """Test search with different optimization strategies."""
        with patch.object(service.model_selector, "select_optimal_model") as mock_model:
            mock_model.return_value = MagicMock()

            # This would require extending the request model, but tests the concept
            response = await service.hybrid_search(sample_request)

            assert isinstance(response, HybridSearchResponse)
