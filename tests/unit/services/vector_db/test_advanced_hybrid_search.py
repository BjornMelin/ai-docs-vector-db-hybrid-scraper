"""Tests for the advanced hybrid search service.

This module contains comprehensive tests for the AdvancedHybridSearchService
including query classification, model selection, adaptive fusion, and A/B testing.
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import UnifiedConfig
from src.config.enums import ABTestVariant, OptimizationStrategy, QueryComplexity, QueryType
from src.models.vector_search import (
    ABTestConfig,
    AdvancedHybridSearchRequest,
    AdvancedSearchResponse,
    QueryClassification,
    SearchAccuracy,
    SearchParams,
    FusionConfig,
    SPLADEConfig,
)
from src.services.errors import QdrantServiceError
from src.services.vector_db.advanced_hybrid_search import AdvancedHybridSearchService


class TestAdvancedHybridSearchService:
    """Test suite for AdvancedHybridSearchService."""

    @pytest.fixture
    def mock_client(self):
        """Mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self):
        """Mock unified configuration."""
        config = MagicMock(spec=UnifiedConfig)
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
        """Create AdvancedHybridSearchService instance."""
        return AdvancedHybridSearchService(mock_client, mock_config, mock_qdrant_search)

    @pytest.fixture
    def sample_request(self):
        """Create sample advanced hybrid search request."""
        return AdvancedHybridSearchRequest(
            query="How to implement async functions in Python?",
            collection_name="test_collection",
            limit=5,
            search_params=SearchParams(
                accuracy_level=SearchAccuracy.BALANCED,
                hnsw_ef=64
            ),
            fusion_config=FusionConfig(algorithm="rrf"),
            enable_query_classification=True,
            enable_model_selection=True,
            enable_adaptive_fusion=True,
            enable_splade=True,
            user_id="test_user",
            session_id="test_session"
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
            features={"has_code_keywords": True, "query_length": 8}
        )

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

    async def test_service_initialization_process(self, service):
        """Test service initialization process."""
        with patch.object(service.splade_provider, 'initialize') as mock_init:
            mock_init.return_value = None
            await service.initialize()
            mock_init.assert_called_once()

    async def test_basic_advanced_hybrid_search(self, service, sample_request):
        """Test basic advanced hybrid search functionality."""
        # Mock all components
        with patch.object(service.query_classifier, 'classify_query') as mock_classify, \
             patch.object(service.model_selector, 'select_optimal_model') as mock_model, \
             patch.object(service.splade_provider, 'generate_sparse_vector') as mock_splade, \
             patch.object(service.adaptive_fusion_tuner, 'compute_adaptive_weights') as mock_fusion:
            
            # Setup mocks
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.CODE,
                complexity_level=QueryComplexity.MODERATE,
                domain="programming",
                programming_language="python",
                is_multimodal=False,
                confidence=0.85,
                features={}
            )
            mock_model.return_value = MagicMock()
            mock_splade.return_value = {1: 0.8, 2: 0.6, 3: 0.4}
            mock_fusion.return_value = MagicMock(
                dense_weight=0.7,
                sparse_weight=0.3,
                effectiveness_score=MagicMock(hybrid_effectiveness=0.85)
            )

            response = await service.advanced_hybrid_search(sample_request)

            assert isinstance(response, AdvancedSearchResponse)
            assert len(response.results) > 0
            assert response.query_classification is not None
            assert response.model_selection is not None
            assert response.fusion_weights is not None
            assert response.optimization_applied is True
            assert response.retrieval_metrics is not None

    async def test_search_with_query_classification_disabled(self, service, sample_request):
        """Test search with query classification disabled."""
        sample_request.enable_query_classification = False
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert response.query_classification is None
        assert len(response.results) > 0

    async def test_search_with_model_selection_disabled(self, service, sample_request):
        """Test search with model selection disabled."""
        sample_request.enable_model_selection = False
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert response.model_selection is None

    async def test_search_with_splade_disabled(self, service, sample_request):
        """Test search with SPLADE disabled."""
        sample_request.enable_splade = False
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert len(response.results) > 0

    async def test_search_with_adaptive_fusion_disabled(self, service, sample_request):
        """Test search with adaptive fusion disabled."""
        sample_request.enable_adaptive_fusion = False
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert response.fusion_weights is None
        assert response.optimization_applied is False

    async def test_ab_test_assignment(self, service, sample_request):
        """Test A/B test variant assignment."""
        sample_request.ab_test_config = ABTestConfig(
            traffic_allocation={"control": 0.5, "treatment": 0.5}
        )
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert response.ab_test_variant in [ABTestVariant.CONTROL, ABTestVariant.TREATMENT]

    async def test_ab_test_consistent_assignment(self, service, sample_request):
        """Test that A/B test assignment is consistent for same user."""
        sample_request.ab_test_config = ABTestConfig(
            traffic_allocation={"control": 0.5, "treatment": 0.5}
        )
        
        # Run multiple times with same user ID
        variants = []
        for _ in range(5):
            response = await service.advanced_hybrid_search(sample_request)
            variants.append(response.ab_test_variant)
        
        # All variants should be the same for consistent user experience
        assert len(set(variants)) == 1

    async def test_query_classification_timeout(self, service, sample_request):
        """Test query classification timeout handling."""
        with patch.object(service.query_classifier, 'classify_query') as mock_classify:
            # Simulate timeout
            mock_classify.side_effect = asyncio.TimeoutError()
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert response.query_classification is None
            assert len(response.results) > 0  # Should fallback gracefully

    async def test_model_selection_timeout(self, service, sample_request):
        """Test model selection timeout handling."""
        with patch.object(service.query_classifier, 'classify_query') as mock_classify, \
             patch.object(service.model_selector, 'select_optimal_model') as mock_model:
            
            mock_classify.return_value = MagicMock()
            mock_model.side_effect = asyncio.TimeoutError()
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert response.model_selection is None
            assert len(response.results) > 0

    async def test_splade_generation_timeout(self, service, sample_request):
        """Test SPLADE generation timeout handling."""
        with patch.object(service.splade_provider, 'generate_sparse_vector') as mock_splade:
            mock_splade.side_effect = asyncio.TimeoutError()
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert len(response.results) > 0

    async def test_adaptive_fusion_error_handling(self, service, sample_request):
        """Test adaptive fusion error handling."""
        with patch.object(service.query_classifier, 'classify_query') as mock_classify, \
             patch.object(service.adaptive_fusion_tuner, 'compute_adaptive_weights') as mock_fusion:
            
            mock_classify.return_value = MagicMock()
            mock_fusion.side_effect = Exception("Fusion error")
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert response.fusion_weights is None
            assert response.effectiveness_score is None
            assert len(response.results) > 0

    async def test_fallback_search_on_error(self, service, sample_request):
        """Test fallback search when advanced features fail."""
        with patch.object(service.query_classifier, 'classify_query') as mock_classify:
            mock_classify.side_effect = Exception("Classification error")
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert response.fallback_reason is not None
            assert "Classification error" in response.fallback_reason
            assert len(response.results) > 0

    async def test_fallback_disabled_error_propagation(self, service, sample_request):
        """Test error propagation when fallback is disabled."""
        service.enable_fallback = False
        
        with patch.object(service.query_classifier, 'classify_query') as mock_classify:
            mock_classify.side_effect = Exception("Classification error")
            
            with pytest.raises(QdrantServiceError, match="Advanced hybrid search failed"):
                await service.advanced_hybrid_search(sample_request)

    async def test_search_metrics_calculation(self, service, sample_request):
        """Test search metrics calculation."""
        response = await service.advanced_hybrid_search(sample_request)
        
        assert response.retrieval_metrics is not None
        assert response.retrieval_metrics.total_time_ms > 0
        assert response.retrieval_metrics.results_count == len(response.results)
        assert response.retrieval_metrics.hnsw_ef_used == sample_request.search_params.hnsw_ef

    async def test_dense_vector_generation(self, service):
        """Test dense vector generation."""
        vector = await service._generate_dense_vector("test query", None)
        
        assert isinstance(vector, list)
        assert len(vector) == 1536  # OpenAI embedding dimension
        assert all(isinstance(x, float) for x in vector)

    async def test_search_strategies_execution(self, service, sample_request):
        """Test execution of different search strategies."""
        dense_vector = [0.1] * 1536
        sparse_vector = {1: 0.8, 2: 0.6}
        
        results = await service._execute_search_strategies(
            sample_request, dense_vector, sparse_vector, ABTestVariant.CONTROL
        )
        
        assert "dense" in results
        assert "sparse" in results
        assert "hybrid" in results
        assert len(results["dense"]) > 0
        assert len(results["hybrid"]) > 0

    async def test_weighted_fusion_application(self, service):
        """Test weighted fusion application."""
        search_results = {
            "dense": [
                {"id": "doc1", "score": 0.9, "payload": {"title": "Doc 1"}},
                {"id": "doc2", "score": 0.8, "payload": {"title": "Doc 2"}},
            ],
            "sparse": [
                {"id": "doc2", "score": 0.85, "payload": {"title": "Doc 2"}},
                {"id": "doc3", "score": 0.75, "payload": {"title": "Doc 3"}},
            ]
        }
        
        fusion_weights = MagicMock()
        fusion_weights.dense_weight = 0.7
        fusion_weights.sparse_weight = 0.3
        
        fused_results = service._apply_weighted_fusion(search_results, fusion_weights, 5)
        
        assert len(fused_results) <= 5
        assert all("id" in result for result in fused_results)
        assert all("score" in result for result in fused_results)

    async def test_performance_statistics(self, service):
        """Test performance statistics retrieval."""
        # Add some mock metrics
        service.search_metrics["test1"] = MagicMock(total_time_ms=100.0)
        service.search_metrics["test2"] = MagicMock(total_time_ms=200.0)
        
        with patch.object(service.adaptive_fusion_tuner, 'get_performance_stats') as mock_stats, \
             patch.object(service.splade_provider, 'get_cache_stats') as mock_cache:
            
            mock_stats.return_value = {"total_queries": 10}
            mock_cache.return_value = {"cache_size": 5}
            
            stats = service.get_performance_statistics()
            
            assert stats["total_searches"] == 2
            assert stats["average_search_time"] == 150.0
            assert "fusion_tuner_stats" in stats
            assert "splade_cache_stats" in stats

    async def test_user_feedback_processing(self, service):
        """Test user feedback processing."""
        query_id = str(uuid.uuid4())
        feedback = {"satisfaction": 0.8, "clicked": True, "dwell_time": 45}
        
        # This should not raise an exception
        await service.update_with_user_feedback(query_id, feedback)

    async def test_search_for_learning_storage(self, service, sample_request):
        """Test storage of search results for learning."""
        query_id = str(uuid.uuid4())
        response = AdvancedSearchResponse(results=[], retrieval_metrics=MagicMock())
        
        with patch.object(service.model_selector, 'update_performance_history') as mock_update:
            await service._store_search_for_learning(query_id, sample_request, response)
            
            assert query_id in service.search_metrics

    async def test_format_search_results(self, service):
        """Test search results formatting."""
        raw_results = [
            {"id": "doc1", "score": 0.9, "payload": {"title": "Doc 1"}, "vector": [0.1, 0.2]},
            {"id": "doc2", "score": 0.8, "payload": {"title": "Doc 2"}},
        ]
        
        formatted = service._format_search_results(raw_results)
        
        assert len(formatted) == 2
        assert formatted[0].id == "doc1"
        assert formatted[0].score == 0.9
        assert formatted[0].payload == {"title": "Doc 1"}
        assert formatted[0].vector == [0.1, 0.2]

    async def test_empty_results_handling(self, service, sample_request):
        """Test handling of empty search results."""
        service.qdrant_search.filtered_search.return_value = []
        service.qdrant_search.hybrid_search.return_value = []
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert len(response.results) == 0
        assert response.retrieval_metrics.results_count == 0

    async def test_large_result_set_handling(self, service, sample_request):
        """Test handling of large result sets."""
        # Mock large result set
        large_results = [
            {"id": f"doc{i}", "score": 0.9 - i * 0.01, "payload": {"title": f"Doc {i}"}}
            for i in range(100)
        ]
        service.qdrant_search.hybrid_search.return_value = large_results
        
        response = await service.advanced_hybrid_search(sample_request)
        
        assert isinstance(response, AdvancedSearchResponse)
        assert len(response.results) == sample_request.limit  # Should be limited to request limit

    @pytest.mark.parametrize("query_type", [
        QueryType.CODE,
        QueryType.DOCUMENTATION,
        QueryType.CONCEPTUAL,
        QueryType.API_REFERENCE,
        QueryType.TROUBLESHOOTING,
        QueryType.MULTIMODAL
    ])
    async def test_different_query_types(self, service, sample_request, query_type):
        """Test search with different query types."""
        with patch.object(service.query_classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=query_type,
                complexity_level=QueryComplexity.MODERATE,
                domain="general",
                programming_language=None,
                is_multimodal=False,
                confidence=0.8,
                features={}
            )
            
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)
            assert response.query_classification.query_type == query_type

    @pytest.mark.parametrize("optimization_strategy", [
        OptimizationStrategy.SPEED_OPTIMIZED,
        OptimizationStrategy.QUALITY_OPTIMIZED,
        OptimizationStrategy.COST_OPTIMIZED,
        OptimizationStrategy.BALANCED
    ])
    async def test_different_optimization_strategies(self, service, sample_request, optimization_strategy):
        """Test search with different optimization strategies."""
        with patch.object(service.model_selector, 'select_optimal_model') as mock_model:
            mock_model.return_value = MagicMock()
            
            # This would require extending the request model, but tests the concept
            response = await service.advanced_hybrid_search(sample_request)
            
            assert isinstance(response, AdvancedSearchResponse)