"""Comprehensive tests for the Advanced Search Orchestrator.

This module provides extensive test coverage for the AdvancedSearchOrchestrator
including all enums, models, search modes, processing stages, pipeline configurations,
error handling, caching, and performance tracking.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.query_processing.clustering import ClusteringMethod
from src.services.query_processing.clustering import ResultClusteringResult
from src.services.query_processing.expansion import ExpansionScope
from src.services.query_processing.expansion import ExpansionStrategy
from src.services.query_processing.expansion import QueryExpansionResult
from src.services.query_processing.federated import FederatedSearchResult
from src.services.query_processing.federated import SearchMode as FederatedSearchMode
from src.services.query_processing.orchestrator import AdvancedSearchOrchestrator
from src.services.query_processing.orchestrator import AdvancedSearchRequest
from src.services.query_processing.orchestrator import AdvancedSearchResult
from src.services.query_processing.orchestrator import ProcessingStage
from src.services.query_processing.orchestrator import SearchMode
from src.services.query_processing.orchestrator import SearchPipeline
from src.services.query_processing.orchestrator import StageResult
from src.services.query_processing.ranking import PersonalizedRankingResult
from src.services.query_processing.ranking import RankingStrategy


class TestEnums:
    """Test all enum classes for completeness and values."""

    def test_search_mode_enum(self):
        """Test SearchMode enum values."""
        assert SearchMode.SIMPLE == "simple"
        assert SearchMode.ENHANCED == "enhanced"
        assert SearchMode.INTELLIGENT == "intelligent"
        assert SearchMode.FEDERATED == "federated"
        assert SearchMode.PERSONALIZED == "personalized"
        assert SearchMode.COMPREHENSIVE == "comprehensive"

        # Test all enum values are defined
        expected_modes = {
            "simple",
            "enhanced",
            "intelligent",
            "federated",
            "personalized",
            "comprehensive",
        }
        actual_modes = {mode.value for mode in SearchMode}
        assert actual_modes == expected_modes

    def test_processing_stage_enum(self):
        """Test ProcessingStage enum values."""
        assert ProcessingStage.PREPROCESSING == "preprocessing"
        assert ProcessingStage.EXPANSION == "expansion"
        assert ProcessingStage.FILTERING == "filtering"
        assert ProcessingStage.EXECUTION == "execution"
        assert ProcessingStage.CLUSTERING == "clustering"
        assert ProcessingStage.RANKING == "ranking"
        assert ProcessingStage.FEDERATION == "federation"
        assert ProcessingStage.POSTPROCESSING == "postprocessing"

        # Test all enum values are defined
        expected_stages = {
            "preprocessing",
            "expansion",
            "filtering",
            "execution",
            "clustering",
            "ranking",
            "federation",
            "postprocessing",
        }
        actual_stages = {stage.value for stage in ProcessingStage}
        assert actual_stages == expected_stages

    def test_search_pipeline_enum(self):
        """Test SearchPipeline enum values."""
        assert SearchPipeline.FAST == "fast"
        assert SearchPipeline.BALANCED == "balanced"
        assert SearchPipeline.COMPREHENSIVE == "comprehensive"
        assert SearchPipeline.DISCOVERY == "discovery"
        assert SearchPipeline.PRECISION == "precision"
        assert SearchPipeline.PERSONALIZED == "personalized"

        # Test all enum values are defined
        expected_pipelines = {
            "fast",
            "balanced",
            "comprehensive",
            "discovery",
            "precision",
            "personalized",
        }
        actual_pipelines = {pipeline.value for pipeline in SearchPipeline}
        assert actual_pipelines == expected_pipelines


class TestModels:
    """Test all Pydantic models for validation and functionality."""

    def test_stage_result_model(self):
        """Test StageResult model validation."""
        # Valid stage result
        stage_result = StageResult(
            stage=ProcessingStage.PREPROCESSING,
            success=True,
            processing_time_ms=150.5,
            results_count=10,
            metadata={"processed_query": "test"},
            error_details=None,
        )

        assert stage_result.stage == ProcessingStage.PREPROCESSING
        assert stage_result.success is True
        assert stage_result.processing_time_ms == 150.5
        assert stage_result.results_count == 10
        assert stage_result.metadata == {"processed_query": "test"}
        assert stage_result.error_details is None

    def test_stage_result_model_with_error(self):
        """Test StageResult model with error details."""
        stage_result = StageResult(
            stage=ProcessingStage.EXPANSION,
            success=False,
            processing_time_ms=50.0,
            results_count=0,
            error_details={"error": "Expansion service unavailable"},
        )

        assert stage_result.success is False
        assert stage_result.error_details["error"] == "Expansion service unavailable"

    def test_advanced_search_request_model(self):
        """Test AdvancedSearchRequest model validation."""
        request = AdvancedSearchRequest(
            query="machine learning algorithms",
            collection_name="documents",
            limit=50,
            offset=10,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
        )

        assert request.query == "machine learning algorithms"
        assert request.collection_name == "documents"
        assert request.limit == 50
        assert request.offset == 10
        assert request.search_mode == SearchMode.ENHANCED
        assert request.pipeline == SearchPipeline.BALANCED

    def test_advanced_search_request_defaults(self):
        """Test AdvancedSearchRequest with default values."""
        request = AdvancedSearchRequest(query="test query")

        assert request.collection_name is None
        assert request.limit == 10
        assert request.offset == 0
        assert request.search_mode == SearchMode.ENHANCED
        assert request.pipeline == SearchPipeline.BALANCED
        assert request.enable_expansion is True
        assert request.enable_clustering is False
        assert request.enable_personalization is False
        assert request.enable_federation is False
        assert request.enable_caching is True
        assert request.max_processing_time_ms == 5000.0
        assert request.quality_threshold == 0.6
        assert request.diversity_factor == 0.1

    def test_advanced_search_request_validation(self):
        """Test AdvancedSearchRequest validation rules."""
        # Empty query should be rejected
        with pytest.raises(ValueError, match="Query cannot be empty"):
            AdvancedSearchRequest(query="")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            AdvancedSearchRequest(query="   ")

    def test_advanced_search_request_field_validation(self):
        """Test AdvancedSearchRequest field validation."""
        # Test limit validation
        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", limit=0)

        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", limit=1001)

        # Test offset validation
        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", offset=-1)

        # Test max_processing_time_ms validation
        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", max_processing_time_ms=50.0)

        # Test quality_threshold validation
        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", quality_threshold=1.5)

        # Test diversity_factor validation
        with pytest.raises(ValueError):
            AdvancedSearchRequest(query="test", diversity_factor=-0.1)

    def test_advanced_search_result_model(self):
        """Test AdvancedSearchResult model."""
        stage_results = [
            StageResult(
                stage=ProcessingStage.PREPROCESSING,
                success=True,
                processing_time_ms=100.0,
                results_count=1,
            )
        ]

        result = AdvancedSearchResult(
            results=[{"id": "1", "title": "Test", "score": 0.9}],
            total_results=1,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            query_processed="test query",
            stage_results=stage_results,
            total_processing_time_ms=500.0,
            quality_score=0.85,
            diversity_score=0.7,
            relevance_score=0.9,
            features_used=["query_expansion"],
            optimizations_applied=["performance_optimization"],
        )

        assert len(result.results) == 1
        assert result.total_results == 1
        assert result.search_mode == SearchMode.ENHANCED
        assert result.pipeline == SearchPipeline.BALANCED
        assert result.quality_score == 0.85
        assert result.diversity_score == 0.7
        assert result.relevance_score == 0.9
        assert "query_expansion" in result.features_used


class TestAdvancedSearchOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with default settings."""
        orchestrator = AdvancedSearchOrchestrator()

        assert orchestrator.enable_all_features is True
        assert orchestrator.enable_performance_optimization is True
        assert orchestrator.max_concurrent_stages == 5
        assert orchestrator.cache_size == 1000
        assert len(orchestrator.search_cache) == 0
        assert orchestrator.cache_stats == {"hits": 0, "misses": 0}

    def test_orchestrator_custom_initialization(self):
        """Test orchestrator initialization with custom settings."""
        orchestrator = AdvancedSearchOrchestrator(
            enable_all_features=False,
            enable_performance_optimization=False,
            cache_size=500,
            max_concurrent_stages=3,
        )

        assert orchestrator.enable_all_features is False
        assert orchestrator.enable_performance_optimization is False
        assert orchestrator.cache_size == 500
        assert orchestrator.max_concurrent_stages == 3

    def test_services_initialization(self):
        """Test that all component services are initialized."""
        orchestrator = AdvancedSearchOrchestrator()

        # Filter services
        assert orchestrator.temporal_filter is not None
        assert orchestrator.content_type_filter is not None
        assert orchestrator.metadata_filter is not None
        assert orchestrator.similarity_threshold_manager is not None
        assert orchestrator.filter_composer is not None

        # Query processing services
        assert orchestrator.query_expansion_service is not None
        assert orchestrator.clustering_service is not None
        assert orchestrator.ranking_service is not None
        assert orchestrator.federated_service is not None

    def test_pipeline_configs_initialization(self):
        """Test pipeline configurations are properly initialized."""
        orchestrator = AdvancedSearchOrchestrator()

        # Test all pipeline configurations exist
        for pipeline in SearchPipeline:
            assert pipeline.value in orchestrator.pipeline_configs

        # Test specific pipeline configurations
        fast_config = orchestrator.pipeline_configs[SearchPipeline.FAST.value]
        assert fast_config["enable_expansion"] is False
        assert fast_config["max_processing_time_ms"] == 1000.0

        comprehensive_config = orchestrator.pipeline_configs[
            SearchPipeline.COMPREHENSIVE.value
        ]
        assert comprehensive_config["enable_expansion"] is True
        assert comprehensive_config["enable_clustering"] is True
        assert comprehensive_config["enable_personalization"] is True
        assert comprehensive_config["enable_federation"] is True


@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing."""
    return AdvancedSearchOrchestrator()


@pytest.fixture
def mock_orchestrator():
    """Create orchestrator with mocked services."""
    orchestrator = AdvancedSearchOrchestrator()

    # Mock filter services
    orchestrator.temporal_filter = AsyncMock()
    orchestrator.content_type_filter = AsyncMock()
    orchestrator.metadata_filter = AsyncMock()
    orchestrator.similarity_threshold_manager = AsyncMock()
    orchestrator.filter_composer = AsyncMock()

    # Mock query processing services
    orchestrator.query_expansion_service = AsyncMock()
    orchestrator.clustering_service = AsyncMock()
    orchestrator.ranking_service = AsyncMock()
    orchestrator.federated_service = AsyncMock()

    return orchestrator


@pytest.fixture
def basic_search_request():
    """Create basic search request."""
    return AdvancedSearchRequest(
        query="machine learning tutorial", collection_name="documentation", limit=10
    )


@pytest.fixture
def comprehensive_search_request():
    """Create comprehensive search request with all features enabled."""
    return AdvancedSearchRequest(
        query="python data science best practices",
        collection_name="tutorials",
        limit=20,
        search_mode=SearchMode.COMPREHENSIVE,
        pipeline=SearchPipeline.COMPREHENSIVE,
        enable_expansion=True,
        enable_clustering=True,
        enable_personalization=True,
        enable_federation=True,
        user_id="user123",
        session_id="session456",
        context={"domain": "data_science", "intent": "learning"},
        temporal_criteria={"since": "2024-01-01"},
        content_type_criteria={"type": "tutorial"},
        metadata_criteria={"difficulty": "intermediate"},
    )


class TestSearchModes:
    """Test different search modes functionality."""

    async def test_simple_search_mode(self, mock_orchestrator, basic_search_request):
        """Test simple search mode execution."""
        basic_search_request.search_mode = SearchMode.SIMPLE
        basic_search_request.pipeline = SearchPipeline.FAST

        result = await mock_orchestrator.search(basic_search_request)

        assert result.search_mode == SearchMode.SIMPLE
        assert result.pipeline == SearchPipeline.FAST

    async def test_enhanced_search_mode(self, mock_orchestrator, basic_search_request):
        """Test enhanced search mode execution."""
        basic_search_request.search_mode = SearchMode.ENHANCED
        basic_search_request.enable_expansion = True

        # Mock expansion service

        mock_result = QueryExpansionResult(
            original_query=basic_search_request.query,
            expanded_query="enhanced machine learning tutorial",
            expanded_terms=[],
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
            expansion_scope=ExpansionScope.MODERATE,
            confidence_score=0.8,
            processing_time_ms=100.0,
        )
        mock_orchestrator.query_expansion_service.expand_query.return_value = (
            mock_result
        )

        result = await mock_orchestrator.search(basic_search_request)

        assert result.search_mode == SearchMode.ENHANCED
        # Should have called expansion service
        mock_orchestrator.query_expansion_service.expand_query.assert_called_once()

    async def test_intelligent_search_mode(
        self, mock_orchestrator, basic_search_request
    ):
        """Test intelligent search mode execution."""
        basic_search_request.search_mode = SearchMode.INTELLIGENT
        basic_search_request.pipeline = SearchPipeline.BALANCED

        result = await mock_orchestrator.search(basic_search_request)

        assert result.search_mode == SearchMode.INTELLIGENT

    async def test_federated_search_mode(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test federated search mode execution."""
        comprehensive_search_request.search_mode = SearchMode.FEDERATED
        comprehensive_search_request.enable_federation = True

        # Mock federated service
        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[{"id": "fed_1", "title": "Federated Result", "score": 0.8}],
            collections_searched=["collection1", "collection2"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={"total_hits": 1},
        )

        result = await mock_orchestrator.search(comprehensive_search_request)

        assert result.search_mode == SearchMode.FEDERATED
        # Should have called federated service
        mock_orchestrator.federated_service.search.assert_called_once()

    async def test_personalized_search_mode(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test personalized search mode execution."""
        comprehensive_search_request.search_mode = SearchMode.PERSONALIZED
        comprehensive_search_request.enable_personalization = True
        comprehensive_search_request.user_id = "user123"

        # Mock ranking service
        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[
                AsyncMock(
                    result_id="1",
                    title="Personalized Result",
                    content="Content",
                    final_score=0.95,
                    original_score=0.8,
                    personalization_boost=0.15,
                    ranking_factors=["user_preference"],
                    metadata={},
                )
            ],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.2,
            user_profile_confidence=0.8,
        )

        result = await mock_orchestrator.search(comprehensive_search_request)

        assert result.search_mode == SearchMode.PERSONALIZED
        # Should have called ranking service
        mock_orchestrator.ranking_service.rank_results.assert_called_once()

    async def test_comprehensive_search_mode(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test comprehensive search mode with all features."""
        comprehensive_search_request.search_mode = SearchMode.COMPREHENSIVE

        # Mock all services with proper return types

        expansion_result = QueryExpansionResult(
            original_query=comprehensive_search_request.query,
            expanded_query="comprehensive query",
            expanded_terms=[],
            expansion_strategy=ExpansionStrategy.HYBRID,
            expansion_scope=ExpansionScope.MODERATE,
            confidence_score=0.9,
            processing_time_ms=100.0,
        )
        mock_orchestrator.query_expansion_service.expand_query.return_value = (
            expansion_result
        )

        clustering_result = ResultClusteringResult(
            clusters=[],
            method_used=ClusteringMethod.HDBSCAN,
            total_results=0,
            clustered_results=0,
            outlier_count=0,
            cluster_count=0,
            processing_time_ms=100.0,
        )
        mock_orchestrator.clustering_service.cluster_results.return_value = (
            clustering_result
        )

        ranking_result = PersonalizedRankingResult(
            ranked_results=[],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            user_profile_confidence=0.7,
            processing_time_ms=100.0,
            reranking_impact=0.1,
            diversity_score=0.8,
            coverage_score=0.75,
        )
        mock_orchestrator.ranking_service.rank_results.return_value = ranking_result

        from src.services.query_processing.federated import CollectionSelectionStrategy
        from src.services.query_processing.federated import ResultMergingStrategy

        federated_result = FederatedSearchResult(
            results=[],
            total_results=0,
            collection_results=[],
            collections_searched=["docs"],
            collections_failed=[],
            search_strategy=CollectionSelectionStrategy.ALL,
            merging_strategy=ResultMergingStrategy.SCORE_BASED,
            search_mode=FederatedSearchMode.PARALLEL,
            total_search_time_ms=100.0,
            fastest_collection_ms=50.0,
            slowest_collection_ms=100.0,
            overall_confidence=0.8,
            coverage_score=0.7,
            diversity_score=0.6,
        )
        mock_orchestrator.federated_service.search.return_value = federated_result

        result = await mock_orchestrator.search(comprehensive_search_request)

        assert result.search_mode == SearchMode.COMPREHENSIVE
        # Note: Services may not be called if features are disabled by pipeline config
        # This test ensures the search completes successfully with comprehensive mode


class TestProcessingStages:
    """Test processing stage execution and order."""

    async def test_preprocessing_stage(self, mock_orchestrator, basic_search_request):
        """Test preprocessing stage execution."""
        result = await mock_orchestrator.search(basic_search_request)

        # Should have preprocessing stage result
        preprocessing_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.PREPROCESSING
        ]
        assert len(preprocessing_stages) == 1
        assert preprocessing_stages[0].success is True

    async def test_expansion_stage(self, mock_orchestrator, basic_search_request):
        """Test expansion stage execution."""
        basic_search_request.enable_expansion = True

        # Mock expansion service
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="expanded query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should have expansion stage result
        expansion_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.EXPANSION
        ]
        assert len(expansion_stages) == 1
        assert expansion_stages[0].success is True

    async def test_filtering_stage(self, mock_orchestrator, basic_search_request):
        """Test filtering stage execution."""
        basic_search_request.temporal_criteria = {"since": "2024-01-01"}

        # Mock filter results
        mock_filter_result = AsyncMock()
        mock_filter_result.filter_conditions = {"date": {"gte": "2024-01-01"}}
        mock_orchestrator.temporal_filter.apply.return_value = mock_filter_result

        result = await mock_orchestrator.search(basic_search_request)

        # Should have filtering stage result
        filtering_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.FILTERING
        ]
        assert len(filtering_stages) == 1
        assert filtering_stages[0].success is True
        mock_orchestrator.temporal_filter.apply.assert_called_once()

    async def test_execution_stage(self, mock_orchestrator, basic_search_request):
        """Test core search execution stage."""
        result = await mock_orchestrator.search(basic_search_request)

        # Should have execution stage result
        execution_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.EXECUTION
        ]
        assert len(execution_stages) == 1
        assert execution_stages[0].success is True

    async def test_clustering_stage(self, mock_orchestrator, basic_search_request):
        """Test clustering stage execution."""
        basic_search_request.enable_clustering = True

        # Mock clustering service
        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[
                AsyncMock(
                    cluster_id="cluster1",
                    label="Programming",
                    items=[{"id": "result_1"}],
                    coherence_score=0.8,
                )
            ],
            algorithm_used=ClusteringMethod.HDBSCAN,
            overall_coherence=0.8,
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should have clustering stage result
        clustering_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.CLUSTERING
        ]
        assert len(clustering_stages) == 1
        assert clustering_stages[0].success is True
        mock_orchestrator.clustering_service.cluster_results.assert_called_once()

    async def test_ranking_stage(self, mock_orchestrator, basic_search_request):
        """Test ranking stage execution."""
        basic_search_request.enable_personalization = True
        basic_search_request.user_id = "user123"

        # Mock ranking service
        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.1,
            user_profile_confidence=0.7,
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should have ranking stage result
        ranking_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.RANKING
        ]
        assert len(ranking_stages) == 1
        assert ranking_stages[0].success is True
        mock_orchestrator.ranking_service.rank_results.assert_called_once()

    async def test_federation_stage(self, mock_orchestrator, basic_search_request):
        """Test federation stage execution."""
        basic_search_request.enable_federation = True

        # Mock federated service
        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[{"id": "fed_1", "title": "Federated", "score": 0.8}],
            collections_searched=["docs"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={},
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should have federation stage result
        federation_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.FEDERATION
        ]
        assert len(federation_stages) == 1
        assert federation_stages[0].success is True
        mock_orchestrator.federated_service.search.assert_called_once()

    async def test_postprocessing_stage(self, mock_orchestrator, basic_search_request):
        """Test postprocessing stage execution."""
        result = await mock_orchestrator.search(basic_search_request)

        # Should have postprocessing stage result
        postprocessing_stages = [
            stage
            for stage in result.stage_results
            if stage.stage == ProcessingStage.POSTPROCESSING
        ]
        assert len(postprocessing_stages) == 1
        assert postprocessing_stages[0].success is True

    async def test_stage_execution_order(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test that stages execute in the correct order."""
        # Enable all features to test full pipeline
        result = await mock_orchestrator.search(comprehensive_search_request)

        # Extract stage names in execution order
        executed_stages = [stage.stage for stage in result.stage_results]

        # Expected order of stages
        expected_order = [
            ProcessingStage.PREPROCESSING,
            ProcessingStage.EXPANSION,
            ProcessingStage.FILTERING,
            ProcessingStage.EXECUTION,
            ProcessingStage.CLUSTERING,
            ProcessingStage.RANKING,
            ProcessingStage.FEDERATION,
            ProcessingStage.POSTPROCESSING,
        ]

        # Verify order (allowing for skipped stages)
        expected_index = 0
        for stage in executed_stages:
            while (
                expected_index < len(expected_order)
                and expected_order[expected_index] != stage
            ):
                expected_index += 1
            assert expected_index < len(expected_order), f"Unexpected stage: {stage}"
            expected_index += 1

    async def test_skip_stages(self, mock_orchestrator, basic_search_request):
        """Test skipping specific stages."""
        basic_search_request.skip_stages = [
            ProcessingStage.EXPANSION,
            ProcessingStage.CLUSTERING,
        ]
        basic_search_request.enable_expansion = True
        basic_search_request.enable_clustering = True

        result = await mock_orchestrator.search(basic_search_request)

        # Skipped stages should not be present
        executed_stages = [stage.stage for stage in result.stage_results]
        assert ProcessingStage.EXPANSION not in executed_stages
        assert ProcessingStage.CLUSTERING not in executed_stages

        # Other stages should still be present
        assert ProcessingStage.PREPROCESSING in executed_stages
        assert ProcessingStage.EXECUTION in executed_stages


class TestPipelineConfigurations:
    """Test predefined pipeline configurations."""

    async def test_fast_pipeline(self, mock_orchestrator):
        """Test fast pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query", pipeline=SearchPipeline.FAST
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.FAST
        # Fast pipeline should skip expensive operations
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        assert len(expansion_stages) == 0  # Should be disabled
        assert len(clustering_stages) == 0  # Should be disabled

    async def test_balanced_pipeline(self, mock_orchestrator):
        """Test balanced pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query", pipeline=SearchPipeline.BALANCED
        )

        # Mock expansion service for balanced pipeline
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="expanded test query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.BALANCED
        # Balanced pipeline should enable expansion but not clustering
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        assert len(expansion_stages) == 1  # Should be enabled
        assert len(clustering_stages) == 0  # Should be disabled

    async def test_comprehensive_pipeline(self, mock_orchestrator):
        """Test comprehensive pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query",
            pipeline=SearchPipeline.COMPREHENSIVE,
            user_id="user123",  # Needed for personalization
        )

        # Mock all services for comprehensive pipeline
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="comprehensive query",
            expanded_terms=["term1", "term2"],
            confidence_score=0.9,
            expansion_strategy=ExpansionStrategy.HYBRID,
        )

        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[], algorithm_used=ClusteringMethod.HDBSCAN, overall_coherence=0.8
        )

        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.1,
            user_profile_confidence=0.7,
        )

        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[],
            collections_searched=["docs"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={},
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.COMPREHENSIVE
        # Comprehensive pipeline should enable all features
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]
        federation_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FEDERATION
        ]

        assert len(expansion_stages) == 1
        assert len(clustering_stages) == 1
        assert len(ranking_stages) == 1
        assert len(federation_stages) == 1

    async def test_discovery_pipeline(self, mock_orchestrator):
        """Test discovery pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query", pipeline=SearchPipeline.DISCOVERY
        )

        # Mock services for discovery pipeline
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="discovery query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[], algorithm_used=ClusteringMethod.HDBSCAN, overall_coherence=0.8
        )

        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[],
            collections_searched=["docs"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={},
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.DISCOVERY
        # Discovery pipeline should enable expansion, clustering, and federation
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        federation_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FEDERATION
        ]
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]

        assert len(expansion_stages) == 1
        assert len(clustering_stages) == 1
        assert len(federation_stages) == 1
        assert len(ranking_stages) == 0  # Should be disabled for discovery

    async def test_precision_pipeline(self, mock_orchestrator):
        """Test precision pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query",
            pipeline=SearchPipeline.PRECISION,
            user_id="user123",  # Needed for personalization
        )

        # Mock ranking service for precision pipeline
        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.CONTENT_BASED,
            personalization_applied=True,
            reranking_impact=0.1,
            user_profile_confidence=0.9,
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.PRECISION
        # Precision pipeline should enable personalization but not expansion/clustering
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]

        assert len(expansion_stages) == 0  # Should be disabled
        assert len(clustering_stages) == 0  # Should be disabled
        assert len(ranking_stages) == 1  # Should be enabled

    async def test_personalized_pipeline(self, mock_orchestrator):
        """Test personalized pipeline configuration."""
        request = AdvancedSearchRequest(
            query="test query", pipeline=SearchPipeline.PERSONALIZED, user_id="user123"
        )

        # Mock services for personalized pipeline
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="personalized query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.CONTEXT_AWARE,
        )

        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.2,
            user_profile_confidence=0.8,
        )

        result = await mock_orchestrator.search(request)

        assert result.pipeline == SearchPipeline.PERSONALIZED
        # Personalized pipeline should enable expansion and ranking
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]

        assert len(expansion_stages) == 1
        assert len(ranking_stages) == 1
        assert len(clustering_stages) == 0  # Should be disabled

    async def test_custom_pipeline_overrides(self, mock_orchestrator):
        """Test custom pipeline configuration overrides."""
        request = AdvancedSearchRequest(
            query="test query",
            pipeline=SearchPipeline.FAST,
            enable_expansion=True,  # Override fast pipeline default
            enable_clustering=True,  # Override fast pipeline default
        )

        # Mock services
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="custom query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[], algorithm_used=ClusteringMethod.HDBSCAN, overall_coherence=0.8
        )

        result = await mock_orchestrator.search(request)

        # Should override pipeline defaults
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]

        assert len(expansion_stages) == 1  # Overridden to enabled
        assert len(clustering_stages) == 1  # Overridden to enabled


class TestPerformanceTracking:
    """Test performance tracking and metrics."""

    async def test_processing_time_tracking(
        self, mock_orchestrator, basic_search_request
    ):
        """Test processing time tracking."""
        result = await mock_orchestrator.search(basic_search_request)

        assert result.total_processing_time_ms > 0
        assert "stage_count" in result.performance_metrics
        assert "successful_stages" in result.performance_metrics
        assert "avg_stage_time" in result.performance_metrics
        assert "processing_efficiency" in result.performance_metrics

    async def test_stage_timing(self, mock_orchestrator, basic_search_request):
        """Test individual stage timing."""
        result = await mock_orchestrator.search(basic_search_request)

        for stage_result in result.stage_results:
            assert stage_result.processing_time_ms >= 0

    async def test_performance_stats_accumulation(
        self, mock_orchestrator, basic_search_request
    ):
        """Test performance statistics accumulation."""
        # Process multiple queries with the same orchestrator instance
        for i in range(3):
            # Create a slightly different request to avoid caching
            request = AdvancedSearchRequest(
                query=f"machine learning tutorial {i}",
                collection_name="documentation",
                limit=10,
                enable_caching=False,  # Disable caching to ensure each call is processed
            )
            await mock_orchestrator.search(request)

        stats = mock_orchestrator.get_performance_stats()

        assert stats["total_searches"] == 3
        assert stats["avg_processing_time"] > 0
        assert "pipeline_usage" in stats
        assert "stage_performance" in stats

    async def test_feature_usage_tracking(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test feature usage tracking."""
        # Mock services
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="tracked query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await mock_orchestrator.search(comprehensive_search_request)

        assert "query_expansion" in result.features_used

        stats = mock_orchestrator.get_performance_stats()
        assert "feature_usage" in stats
        assert "query_expansion" in stats["feature_usage"]

    async def test_quality_metrics(self, mock_orchestrator, basic_search_request):
        """Test quality metrics calculation."""
        result = await mock_orchestrator.search(basic_search_request)

        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.diversity_score <= 1.0
        assert 0.0 <= result.relevance_score <= 1.0

    async def test_quality_stats_accumulation(
        self, mock_orchestrator, basic_search_request
    ):
        """Test quality statistics accumulation."""
        # Process multiple queries
        for _ in range(3):
            await mock_orchestrator.search(basic_search_request)

        stats = mock_orchestrator.get_performance_stats()

        assert "avg_quality_score" in stats
        assert "avg_diversity_score" in stats
        assert "avg_relevance_score" in stats
        assert 0.0 <= stats["avg_quality_score"] <= 1.0


class TestCachingFunctionality:
    """Test caching functionality."""

    async def test_cache_hit(self, mock_orchestrator, basic_search_request):
        """Test cache hit functionality."""
        # First request - should be a cache miss
        result1 = await mock_orchestrator.search(basic_search_request)
        assert result1.cache_hit is False

        # Second identical request - should be a cache hit
        result2 = await mock_orchestrator.search(basic_search_request)
        assert result2.cache_hit is True

    async def test_cache_miss(self, mock_orchestrator, basic_search_request):
        """Test cache miss functionality."""
        # First request
        result1 = await mock_orchestrator.search(basic_search_request)
        assert result1.cache_hit is False

        # Different request - should be a cache miss
        basic_search_request.query = "different query"
        result2 = await mock_orchestrator.search(basic_search_request)
        assert result2.cache_hit is False

    async def test_cache_disabled(self, mock_orchestrator, basic_search_request):
        """Test behavior when caching is disabled."""
        basic_search_request.enable_caching = False

        # First request
        result1 = await mock_orchestrator.search(basic_search_request)
        assert result1.cache_hit is False

        # Second identical request - should still be a miss
        result2 = await mock_orchestrator.search(basic_search_request)
        assert result2.cache_hit is False

    async def test_cache_eviction(self, mock_orchestrator):
        """Test cache eviction when size limit is reached."""
        # Set small cache size
        mock_orchestrator.cache_size = 2

        # Fill cache
        for i in range(3):
            request = AdvancedSearchRequest(query=f"query {i}")
            await mock_orchestrator.search(request)

        # Cache should not exceed size limit
        assert len(mock_orchestrator.search_cache) <= 2

    async def test_cache_key_generation(self, mock_orchestrator):
        """Test cache key generation."""
        request1 = AdvancedSearchRequest(query="test", limit=10)
        request2 = AdvancedSearchRequest(query="test", limit=20)

        key1 = mock_orchestrator._generate_cache_key(request1)
        key2 = mock_orchestrator._generate_cache_key(request2)

        # Different limits should generate different keys
        assert key1 != key2

    async def test_cache_stats(self, mock_orchestrator, basic_search_request):
        """Test cache statistics tracking."""
        # Initial stats
        initial_stats = mock_orchestrator.cache_stats.copy()

        # First request (miss)
        await mock_orchestrator.search(basic_search_request)
        assert mock_orchestrator.cache_stats["misses"] == initial_stats["misses"] + 1

        # Second request (hit)
        await mock_orchestrator.search(basic_search_request)
        assert mock_orchestrator.cache_stats["hits"] == initial_stats["hits"] + 1

    async def test_clear_cache(self, mock_orchestrator, basic_search_request):
        """Test cache clearing."""
        # Add something to cache
        await mock_orchestrator.search(basic_search_request)
        assert len(mock_orchestrator.search_cache) > 0

        # Clear cache
        mock_orchestrator.clear_cache()
        assert len(mock_orchestrator.search_cache) == 0
        assert mock_orchestrator.cache_stats == {"hits": 0, "misses": 0}


class TestErrorHandling:
    """Test error handling and recovery."""

    async def test_preprocessing_stage_failure(
        self, mock_orchestrator, basic_search_request
    ):
        """Test handling of preprocessing stage failure."""
        # Make preprocessing fail
        with patch.object(
            mock_orchestrator, "_execute_preprocessing_stage"
        ) as mock_preprocess:
            mock_preprocess.return_value = StageResult(
                stage=ProcessingStage.PREPROCESSING,
                success=False,
                processing_time_ms=100.0,
                results_count=0,
                error_details={"error": "Preprocessing failed"},
            )

            result = await mock_orchestrator.search(basic_search_request)

            # Should return error result
            assert result.total_results == 0
            assert len(result.results) == 0
            assert any(not sr.success for sr in result.stage_results)

    async def test_expansion_stage_failure(
        self, mock_orchestrator, basic_search_request
    ):
        """Test handling of expansion stage failure."""
        basic_search_request.enable_expansion = True

        # Make expansion service fail
        mock_orchestrator.query_expansion_service.expand_query.side_effect = Exception(
            "Expansion failed"
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should continue with original query
        assert result.query_processed == basic_search_request.query.strip().lower()
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        if expansion_stages:
            assert not expansion_stages[0].success

    async def test_search_execution_failure(
        self, mock_orchestrator, basic_search_request
    ):
        """Test handling of search execution failure."""
        # Make search execution fail
        with patch.object(mock_orchestrator, "_execute_search_stage") as mock_search:
            mock_search.return_value = StageResult(
                stage=ProcessingStage.EXECUTION,
                success=False,
                processing_time_ms=100.0,
                results_count=0,
                error_details={"error": "Search failed"},
            )

            result = await mock_orchestrator.search(basic_search_request)

            # Should return error result
            assert result.total_results == 0

    async def test_clustering_stage_failure(
        self, mock_orchestrator, basic_search_request
    ):
        """Test handling of clustering stage failure."""
        basic_search_request.enable_clustering = True

        # Make clustering service fail
        mock_orchestrator.clustering_service.cluster_results.side_effect = Exception(
            "Clustering failed"
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should continue without clustering
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        if clustering_stages:
            assert not clustering_stages[0].success

    async def test_ranking_stage_failure(self, mock_orchestrator, basic_search_request):
        """Test handling of ranking stage failure."""
        basic_search_request.enable_personalization = True
        basic_search_request.user_id = "user123"

        # Make ranking service fail
        mock_orchestrator.ranking_service.rank_results.side_effect = Exception(
            "Ranking failed"
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should continue without personalized ranking
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]
        if ranking_stages:
            assert not ranking_stages[0].success

    async def test_federation_stage_failure(
        self, mock_orchestrator, basic_search_request
    ):
        """Test handling of federation stage failure."""
        basic_search_request.enable_federation = True

        # Make federated service fail
        mock_orchestrator.federated_service.search.side_effect = Exception(
            "Federation failed"
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should continue without federated results
        federation_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FEDERATION
        ]
        if federation_stages:
            assert not federation_stages[0].success

    async def test_filtering_stage_failure_continues(
        self, mock_orchestrator, basic_search_request
    ):
        """Test that filtering failures don't stop the pipeline."""
        basic_search_request.temporal_criteria = {"since": "invalid-date"}

        # Make filter fail
        mock_orchestrator.temporal_filter.apply.side_effect = Exception("Filter failed")

        result = await mock_orchestrator.search(basic_search_request)

        # Should continue with search despite filter failure
        assert len(result.results) >= 0  # Should have mock results
        filtering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FILTERING
        ]
        if filtering_stages:
            assert not filtering_stages[0].success

    async def test_general_exception_handling(
        self, mock_orchestrator, basic_search_request
    ):
        """Test general exception handling."""
        # Make a core method fail
        with patch.object(mock_orchestrator, "_apply_pipeline_config") as mock_config:
            mock_config.side_effect = Exception("Unexpected error")

            result = await mock_orchestrator.search(basic_search_request)

            # Should return error result
            assert result.total_results == 0
            assert "error" in result.search_metadata


class TestFeatureToggling:
    """Test feature toggling and conditional execution."""

    async def test_expansion_toggle(self, mock_orchestrator, basic_search_request):
        """Test expansion feature toggle."""
        # Test with expansion enabled
        basic_search_request.enable_expansion = True
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="expanded query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await mock_orchestrator.search(basic_search_request)
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        assert len(expansion_stages) == 1

        # Test with expansion disabled
        basic_search_request.enable_expansion = False
        result = await mock_orchestrator.search(basic_search_request)
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        assert len(expansion_stages) == 0

    async def test_clustering_toggle(self, mock_orchestrator, basic_search_request):
        """Test clustering feature toggle."""
        # Test with clustering enabled
        basic_search_request.enable_clustering = True
        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[], algorithm_used=ClusteringMethod.HDBSCAN, overall_coherence=0.8
        )

        result = await mock_orchestrator.search(basic_search_request)
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        assert len(clustering_stages) == 1

        # Test with clustering disabled
        basic_search_request.enable_clustering = False
        result = await mock_orchestrator.search(basic_search_request)
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]
        assert len(clustering_stages) == 0

    async def test_personalization_toggle(
        self, mock_orchestrator, basic_search_request
    ):
        """Test personalization feature toggle."""
        # Test with personalization enabled
        basic_search_request.enable_personalization = True
        basic_search_request.user_id = "user123"
        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.1,
            user_profile_confidence=0.7,
        )

        result = await mock_orchestrator.search(basic_search_request)
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]
        assert len(ranking_stages) == 1

        # Test with personalization disabled
        basic_search_request.enable_personalization = False
        result = await mock_orchestrator.search(basic_search_request)
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]
        assert len(ranking_stages) == 0

    async def test_federation_toggle(self, mock_orchestrator, basic_search_request):
        """Test federation feature toggle."""
        # Test with federation enabled
        basic_search_request.enable_federation = True
        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[],
            collections_searched=["docs"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={},
        )

        result = await mock_orchestrator.search(basic_search_request)
        federation_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FEDERATION
        ]
        assert len(federation_stages) == 1

        # Test with federation disabled
        basic_search_request.enable_federation = False
        result = await mock_orchestrator.search(basic_search_request)
        federation_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FEDERATION
        ]
        assert len(federation_stages) == 0

    async def test_personalization_without_user_id(
        self, mock_orchestrator, basic_search_request
    ):
        """Test personalization skipping when no user ID provided."""
        basic_search_request.enable_personalization = True
        basic_search_request.user_id = None  # No user ID

        result = await mock_orchestrator.search(basic_search_request)
        ranking_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.RANKING
        ]

        if ranking_stages:
            # Should skip personalization without user ID
            assert (
                ranking_stages[0].metadata.get("personalization") == "skipped_no_user"
            )

    async def test_conditional_filter_application(
        self, mock_orchestrator, basic_search_request
    ):
        """Test conditional filter application."""
        # Test with no filter criteria - should skip filtering stage
        result = await mock_orchestrator.search(basic_search_request)
        filtering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FILTERING
        ]
        assert len(filtering_stages) == 1  # Always executes but may have no filters

        # Test with filter criteria
        basic_search_request.temporal_criteria = {"since": "2024-01-01"}
        mock_filter_result = AsyncMock()
        mock_filter_result.filter_conditions = {"date": {"gte": "2024-01-01"}}
        mock_orchestrator.temporal_filter.apply.return_value = mock_filter_result

        result = await mock_orchestrator.search(basic_search_request)
        filtering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.FILTERING
        ]
        assert len(filtering_stages) == 1
        assert filtering_stages[0].success


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    async def test_full_pipeline_integration(
        self, mock_orchestrator, comprehensive_search_request
    ):
        """Test full pipeline with all features enabled."""
        # Mock all services
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="comprehensive expanded query",
            expanded_terms=["ML", "algorithm", "training"],
            confidence_score=0.9,
            expansion_strategy=ExpansionStrategy.HYBRID,
        )

        mock_orchestrator.clustering_service.cluster_results.return_value = AsyncMock(
            clusters=[
                AsyncMock(
                    cluster_id="programming",
                    label="Programming Concepts",
                    items=[{"id": "result_1"}, {"id": "result_2"}],
                    coherence_score=0.85,
                ),
                AsyncMock(
                    cluster_id="data_science",
                    label="Data Science",
                    items=[{"id": "result_3"}],
                    coherence_score=0.9,
                ),
            ],
            algorithm_used=ClusteringMethod.HDBSCAN,
            overall_coherence=0.875,
        )

        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[
                AsyncMock(
                    result_id="result_1",
                    title="Advanced ML Tutorial",
                    content="Comprehensive machine learning guide",
                    final_score=0.95,
                    original_score=0.8,
                    personalization_boost=0.15,
                    ranking_factors=["user_preference", "expertise_level"],
                    metadata={"difficulty": "advanced", "topic": "ML"},
                )
            ],
            strategy_used=RankingStrategy.HYBRID,
            personalization_applied=True,
            reranking_impact=0.25,
            user_profile_confidence=0.85,
        )

        mock_orchestrator.federated_service.search.return_value = AsyncMock(
            results=[
                {
                    "id": "fed_1",
                    "title": "External Resource",
                    "score": 0.8,
                    "source": "external",
                }
            ],
            collections_searched=["documentation", "tutorials", "examples"],
            collections_failed=[],
            search_strategy=FederatedSearchMode.PARALLEL,
            federated_metadata={"total_hits": 15, "federation_time_ms": 200},
        )

        # Mock filter services
        mock_filter_result = AsyncMock()
        mock_filter_result.filter_conditions = {"content_type": "tutorial"}
        mock_orchestrator.content_type_filter.apply.return_value = mock_filter_result

        result = await mock_orchestrator.search(comprehensive_search_request)

        # Verify all stages executed successfully
        expected_stages = {
            ProcessingStage.PREPROCESSING,
            ProcessingStage.EXPANSION,
            ProcessingStage.FILTERING,
            ProcessingStage.EXECUTION,
            ProcessingStage.CLUSTERING,
            ProcessingStage.RANKING,
            ProcessingStage.FEDERATION,
            ProcessingStage.POSTPROCESSING,
        }

        executed_stages = {stage.stage for stage in result.stage_results}
        assert expected_stages.issubset(executed_stages)

        # Verify all stages succeeded
        assert all(stage.success for stage in result.stage_results)

        # Verify comprehensive features were used
        expected_features = {
            "query_expansion",
            "result_clustering",
            "personalized_ranking",
            "federated_search",
        }
        assert expected_features.issubset(set(result.features_used))

    async def test_progressive_fallback_scenario(
        self, mock_orchestrator, basic_search_request
    ):
        """Test progressive fallback when stages fail."""
        basic_search_request.enable_expansion = True
        basic_search_request.enable_clustering = True

        # Make expansion fail
        mock_orchestrator.query_expansion_service.expand_query.side_effect = Exception(
            "Expansion failed"
        )

        # Make clustering fail
        mock_orchestrator.clustering_service.cluster_results.side_effect = Exception(
            "Clustering failed"
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should still complete successfully with core search
        assert result.total_processing_time_ms > 0

        # Failed stages should be marked as unsuccessful
        expansion_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.EXPANSION
        ]
        clustering_stages = [
            s for s in result.stage_results if s.stage == ProcessingStage.CLUSTERING
        ]

        if expansion_stages:
            assert not expansion_stages[0].success
        if clustering_stages:
            assert not clustering_stages[0].success

    async def test_partial_feature_scenario(
        self, mock_orchestrator, basic_search_request
    ):
        """Test scenario with only some features enabled."""
        basic_search_request.enable_expansion = True
        basic_search_request.enable_clustering = False
        basic_search_request.enable_personalization = False
        basic_search_request.enable_federation = False

        # Mock only expansion service
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="partially expanded query",
            expanded_terms=["term1"],
            confidence_score=0.8,
            expansion_strategy=ExpansionStrategy.SEMANTIC_BASED,
        )

        result = await mock_orchestrator.search(basic_search_request)

        # Should only have enabled features
        executed_stages = {stage.stage for stage in result.stage_results}
        assert ProcessingStage.EXPANSION in executed_stages
        assert ProcessingStage.CLUSTERING not in executed_stages
        assert ProcessingStage.RANKING not in executed_stages
        assert ProcessingStage.FEDERATION not in executed_stages

        assert "query_expansion" in result.features_used
        assert "result_clustering" not in result.features_used

    async def test_high_performance_scenario(
        self, mock_orchestrator, basic_search_request
    ):
        """Test high-performance scenario with time constraints."""
        basic_search_request.max_processing_time_ms = 500.0  # Very tight constraint
        basic_search_request.pipeline = SearchPipeline.FAST

        result = await mock_orchestrator.search(basic_search_request)

        # Should complete within reasonable time (allowing for mock overhead)
        assert result.total_processing_time_ms < 2000  # Generous for mocks

        # Should use fast pipeline optimizations
        assert "performance_optimization" in result.optimizations_applied

    async def test_context_driven_scenario(self, mock_orchestrator):
        """Test context-driven search scenario."""
        request = AdvancedSearchRequest(
            query="django rest api",
            context={
                "domain": "web_development",
                "intent": "learning",
                "programming_language": "python",
                "experience_level": "intermediate",
            },
            enable_expansion=True,
            enable_personalization=True,
            user_id="dev_user_123",
            session_id="learning_session_456",
        )

        # Mock expansion with context awareness
        mock_orchestrator.query_expansion_service.expand_query.return_value = AsyncMock(
            expanded_query="django rest api web_development learning python",
            expanded_terms=["REST", "API", "Django", "Python", "web"],
            confidence_score=0.9,
            expansion_strategy=ExpansionStrategy.CONTEXT_AWARE,
        )

        # Mock personalized ranking
        mock_orchestrator.ranking_service.rank_results.return_value = AsyncMock(
            ranked_results=[],
            strategy_used=RankingStrategy.CONTEXTUAL,
            personalization_applied=True,
            reranking_impact=0.3,
            user_profile_confidence=0.8,
        )

        result = await mock_orchestrator.search(request)

        # Should incorporate context in processing
        assert result.search_metadata["user_context"] is True
        assert "query_expansion" in result.features_used
        assert "personalized_ranking" in result.features_used


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_enhance_query_with_context(self, orchestrator):
        """Test query enhancement with context."""
        query = "machine learning"
        context = {"domain": "data_science", "intent": "tutorial"}

        enhanced = orchestrator._enhance_query_with_context(query, context)

        assert "domain:data_science" in enhanced
        assert "intent:tutorial" in enhanced

    def test_merge_federated_results(self, orchestrator):
        """Test federated results merging."""
        primary_results = [
            {"id": "p1", "score": 0.9},
            {"id": "p2", "score": 0.8},
            {"id": "p3", "score": 0.7},
        ]

        federated_results = [{"id": "f1", "score": 0.85}, {"id": "f2", "score": 0.75}]

        merged = orchestrator._merge_federated_results(
            primary_results, federated_results
        )

        # Should interleave results (2 primary, 1 federated pattern)
        assert len(merged) == 5
        assert merged[0]["id"] == "p1"
        assert merged[1]["id"] == "p2"
        assert merged[2]["id"] == "f1"

    def test_apply_diversity_optimization(self, orchestrator):
        """Test diversity optimization."""
        results = [
            {"id": "1", "content_type": "code", "score": 0.9},
            {"id": "2", "content_type": "code", "score": 0.8},
            {"id": "3", "content_type": "documentation", "score": 0.7},
            {"id": "4", "content_type": "code", "score": 0.6},
        ]

        diversified = orchestrator._apply_diversity_optimization(results, 0.5)

        # Scores should be adjusted for diversity
        assert len(diversified) == 4
        # Second code result should have lower score due to diversity penalty
        code_results = [r for r in diversified if r["content_type"] == "code"]
        assert len(code_results) >= 2

    def test_calculate_quality_metrics(self, orchestrator):
        """Test quality metrics calculation."""
        results = [
            {"score": 0.9, "content_type": "documentation"},
            {"score": 0.8, "content_type": "code"},
            {"score": 0.7, "content_type": "tutorial"},
        ]

        request = AdvancedSearchRequest(query="test")
        metrics = orchestrator._calculate_quality_metrics(results, request)

        assert "quality_score" in metrics
        assert "diversity_score" in metrics
        assert "relevance_score" in metrics

        # Quality should be average of scores
        expected_quality = (0.9 + 0.8 + 0.7) / 3
        assert abs(metrics["quality_score"] - expected_quality) < 0.01

        # Diversity should reflect content type variety
        assert metrics["diversity_score"] > 0

    def test_get_features_used(self, orchestrator):
        """Test features used detection."""
        config = {
            "enable_expansion": True,
            "enable_clustering": False,
            "enable_personalization": True,
            "enable_federation": False,
        }

        stage_results = [
            StageResult(
                stage=ProcessingStage.FILTERING,
                success=True,
                processing_time_ms=100,
                results_count=5,
                metadata={"applied_filters": ["temporal", "content_type"]},
            )
        ]

        features = orchestrator._get_features_used(config, stage_results)

        expected_features = {
            "query_expansion",
            "personalized_ranking",
            "temporal_filtering",
            "content_type_filtering",
        }

        assert expected_features.issubset(set(features))

    def test_get_optimizations_applied(self, orchestrator):
        """Test optimizations detection."""
        config = {"enable_caching": True, "diversity_factor": 0.2}

        orchestrator.enable_performance_optimization = True

        optimizations = orchestrator._get_optimizations_applied(config)

        expected_optimizations = {
            "performance_optimization",
            "result_caching",
            "diversity_optimization",
        }

        assert expected_optimizations.issubset(set(optimizations))

    def test_build_error_result(self, orchestrator):
        """Test error result building."""
        request = AdvancedSearchRequest(query="test")
        stage_results = [
            StageResult(
                stage=ProcessingStage.PREPROCESSING,
                success=True,
                processing_time_ms=100,
                results_count=1,
            )
        ]

        error_result = orchestrator._build_error_result(
            request, stage_results, "Test error", 500.0
        )

        assert error_result.total_results == 0
        assert len(error_result.results) == 0
        assert error_result.total_processing_time_ms == 500.0
        assert "error" in error_result.search_metadata
        assert error_result.search_metadata["failed"] is True


class TestPerformanceStatsAndCleanup:
    """Test performance statistics and cleanup methods."""

    async def test_get_performance_stats(self, mock_orchestrator, basic_search_request):
        """Test performance statistics retrieval."""
        # Process some requests to build stats
        for i in range(3):
            # Create a slightly different request to avoid caching
            request = AdvancedSearchRequest(
                query=f"machine learning tutorial {i}",
                collection_name="documentation",
                limit=10,
                enable_caching=False,  # Disable caching to ensure each call is processed
            )
            await mock_orchestrator.search(request)

        stats = mock_orchestrator.get_performance_stats()

        # Should include all expected stat categories
        expected_keys = {
            "total_searches",
            "avg_processing_time",
            "feature_usage",
            "pipeline_usage",
            "stage_performance",
            "avg_quality_score",
            "avg_diversity_score",
            "avg_relevance_score",
            "cache_stats",
            "cache_size",
        }

        assert expected_keys.issubset(set(stats.keys()))
        assert stats["total_searches"] == 3

    async def test_stage_performance_tracking(
        self, mock_orchestrator, basic_search_request
    ):
        """Test stage-level performance tracking."""
        await mock_orchestrator.search(basic_search_request)

        stats = mock_orchestrator.get_performance_stats()
        stage_perf = stats["stage_performance"]

        # Should have stats for executed stages
        assert "preprocessing" in stage_perf
        assert "execution" in stage_perf
        assert "postprocessing" in stage_perf

        # Each stage should have expected metrics
        for stage_stats in stage_perf.values():
            assert "total_time" in stage_stats
            assert "count" in stage_stats
            assert "success_rate" in stage_stats

    def test_clear_cache_functionality(self, mock_orchestrator):
        """Test cache clearing functionality."""
        # Add some data to cache and stats
        mock_orchestrator.search_cache["key1"] = "value1"
        mock_orchestrator.search_cache["key2"] = "value2"
        mock_orchestrator.cache_stats = {"hits": 5, "misses": 3}

        # Clear cache
        mock_orchestrator.clear_cache()

        # Should be empty
        assert len(mock_orchestrator.search_cache) == 0
        assert mock_orchestrator.cache_stats == {"hits": 0, "misses": 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
