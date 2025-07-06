"""Comprehensive tests for the simplified SearchOrchestrator.

This test suite provides extensive coverage for the SearchOrchestrator including:
- Enums, models, and configuration
- Core search functionality
- Feature integration (expansion, clustering, ranking, RAG)
- Pipeline configurations
- Caching and performance tracking
- Error handling and edge cases
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.query_processing.orchestrator import (
    SearchMode,
    SearchOrchestrator,
    SearchPipeline,
    SearchRequest,
    SearchResult,
)


@pytest.fixture
def orchestrator():
    """Create a SearchOrchestrator instance for testing."""
    return SearchOrchestrator(
        cache_size=100,
        enable_performance_optimization=True,
    )


@pytest.fixture
def basic_request():
    """Create a basic search request."""
    return SearchRequest(
        query="What is machine learning?",
        collection_name="documentation",
        limit=10,
    )


@pytest.fixture
def comprehensive_request():
    """Create a comprehensive search request with all features enabled."""
    return SearchRequest(
        query="Python web development best practices",
        collection_name="programming",
        limit=20,
        mode=SearchMode.FULL,
        pipeline=SearchPipeline.COMPREHENSIVE,
        enable_expansion=True,
        enable_clustering=True,
        enable_personalization=True,
        enable_federation=True,
        enable_rag=True,
        user_id="test_user_123",
        session_id="session_456",
    )


class TestEnums:
    """Test all enum classes for correct values."""

    def test_search_mode_enum(self):
        """Test SearchMode enum values."""
        assert SearchMode.BASIC == "basic"
        assert SearchMode.ENHANCED == "enhanced"
        assert SearchMode.FULL == "full"

        # Test all enum values are defined
        expected_modes = {"basic", "enhanced", "full"}
        actual_modes = {mode.value for mode in SearchMode}
        assert actual_modes == expected_modes

    def test_search_pipeline_enum(self):
        """Test SearchPipeline enum values."""
        assert SearchPipeline.FAST == "fast"
        assert SearchPipeline.BALANCED == "balanced"
        assert SearchPipeline.COMPREHENSIVE == "comprehensive"

        # Test all enum values are defined
        expected_pipelines = {"fast", "balanced", "comprehensive"}
        actual_pipelines = {pipeline.value for pipeline in SearchPipeline}
        assert actual_pipelines == expected_pipelines


class TestModels:
    """Test Pydantic models for validation and functionality."""

    def test_search_request_validation(self):
        """Test SearchRequest model validation."""
        # Valid request
        request = SearchRequest(
            query="test query",
            collection_name="test_collection",
            limit=5,
            offset=10,
        )
        assert request.query == "test query"
        assert request.collection_name == "test_collection"
        assert request.limit == 5
        assert request.offset == 10
        assert request.mode == SearchMode.ENHANCED  # Default
        assert request.pipeline == SearchPipeline.BALANCED  # Default

    def test_search_request_defaults(self):
        """Test SearchRequest default values."""
        request = SearchRequest(query="test")
        assert request.collection_name is None
        assert request.limit == 10
        assert request.offset == 0
        assert request.mode == SearchMode.ENHANCED
        assert request.pipeline == SearchPipeline.BALANCED
        assert request.enable_expansion is True
        assert request.enable_clustering is False
        assert request.enable_personalization is False
        assert request.enable_federation is False
        assert request.enable_rag is False
        assert request.enable_caching is True
        assert request.max_processing_time_ms == 5000.0

    def test_search_request_validation_limits(self):
        """Test SearchRequest validation limits."""
        # Test limit boundaries
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)  # Below minimum

        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=1001)  # Above maximum

        # Test offset boundaries
        with pytest.raises(ValueError):
            SearchRequest(query="test", offset=-1)  # Below minimum

        # Valid boundaries
        request_min = SearchRequest(query="test", limit=1, offset=0)
        assert request_min.limit == 1
        assert request_min.offset == 0

        request_max = SearchRequest(query="test", limit=1000)
        assert request_max.limit == 1000

    def test_search_result_validation(self):
        """Test SearchResult model validation."""
        result = SearchResult(
            results=[{"id": "1", "content": "test"}],
            _total_results=1,
            query_processed="test query",
            processing_time_ms=500.0,
            expanded_query="expanded test query",
            features_used=["query_expansion"],
            generated_answer="Test answer",
            answer_confidence=0.8,
            cache_hit=True,
        )

        assert len(result.results) == 1
        assert result._total_results == 1
        assert result.query_processed == "test query"
        assert result.processing_time_ms == 500.0
        assert result.expanded_query == "expanded test query"
        assert "query_expansion" in result.features_used
        assert result.generated_answer == "Test answer"
        assert result.answer_confidence == 0.8
        assert result.cache_hit is True

    def test_search_result_defaults(self):
        """Test SearchResult default values."""
        result = SearchResult(
            results=[],
            _total_results=0,
            query_processed="test",
            processing_time_ms=100.0,
        )

        assert result.results == []
        assert result._total_results == 0
        assert result.expanded_query is None
        assert result.clusters is None
        assert result.features_used == []
        assert result.generated_answer is None
        assert result.answer_confidence is None
        assert result.answer_sources is None
        assert result.answer_metrics is None
        assert result.cache_hit is False


class TestSearchOrchestrator:
    """Test the SearchOrchestrator class initialization and configuration."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.enable_performance_optimization is True
        assert orchestrator.cache_size == 100
        assert isinstance(orchestrator.cache, dict)
        assert len(orchestrator.cache) == 0

        # Check stats initialization
        expected_stats = {
            "_total_searches": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        assert orchestrator.stats == expected_stats

        # Check pipeline configurations
        assert SearchPipeline.FAST in orchestrator.pipeline_configs
        assert SearchPipeline.BALANCED in orchestrator.pipeline_configs
        assert SearchPipeline.COMPREHENSIVE in orchestrator.pipeline_configs

    def test_pipeline_configurations(self, orchestrator):
        """Test pipeline configurations are correct."""
        fast_config = orchestrator.pipeline_configs[SearchPipeline.FAST]
        assert fast_config["enable_expansion"] is False
        assert fast_config["enable_clustering"] is False
        assert fast_config["enable_personalization"] is False
        assert fast_config["max_processing_time_ms"] == 1000.0

        balanced_config = orchestrator.pipeline_configs[SearchPipeline.BALANCED]
        assert balanced_config["enable_expansion"] is True
        assert balanced_config["enable_clustering"] is False
        assert balanced_config["enable_personalization"] is False
        assert balanced_config["max_processing_time_ms"] == 3000.0

        comprehensive_config = orchestrator.pipeline_configs[
            SearchPipeline.COMPREHENSIVE
        ]
        assert comprehensive_config["enable_expansion"] is True
        assert comprehensive_config["enable_clustering"] is True
        assert comprehensive_config["enable_personalization"] is True
        assert comprehensive_config["max_processing_time_ms"] == 10000.0

    def test_lazy_service_loading(self, orchestrator):
        """Test lazy loading of services."""
        # Services should be None initially
        assert orchestrator._query_expansion_service is None
        assert orchestrator._clustering_service is None
        assert orchestrator._ranking_service is None
        assert orchestrator._federated_service is None
        assert orchestrator._rag_generator is None

        # Accessing properties should initialize services
        expansion_service = orchestrator.query_expansion_service
        assert expansion_service is not None
        assert orchestrator._query_expansion_service is expansion_service

        clustering_service = orchestrator.clustering_service
        assert clustering_service is not None
        assert orchestrator._clustering_service is clustering_service

        ranking_service = orchestrator.ranking_service
        assert ranking_service is not None
        assert orchestrator._ranking_service is ranking_service

        federated_service = orchestrator.federated_service
        assert federated_service is not None
        assert orchestrator._federated_service is federated_service

        rag_generator = orchestrator.rag_generator
        assert rag_generator is not None
        assert orchestrator._rag_generator is rag_generator

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()
        # No specific initialization behavior to test in current implementation
        # This mainly verifies the method exists and doesn't raise errors

    @pytest.mark.asyncio
    async def test_cleanup(self, orchestrator):
        """Test orchestrator cleanup."""
        # Add some data to cache
        orchestrator.cache["test_key"] = "test_value"
        assert len(orchestrator.cache) == 1

        # Mock RAG generator
        orchestrator._rag_generator = Mock()
        orchestrator._rag_generator._initialized = True
        orchestrator._rag_generator.cleanup = AsyncMock()

        await orchestrator.cleanup()

        # Cache should be cleared
        assert len(orchestrator.cache) == 0

        # RAG generator cleanup should be called
        orchestrator._rag_generator.cleanup.assert_called_once()


class TestCoreSearchFunctionality:
    """Test core search functionality."""

    @pytest.mark.asyncio
    async def test_basic_search(self, orchestrator, basic_request):
        """Test basic search without advanced features."""
        result = await orchestrator.search(basic_request)

        assert isinstance(result, SearchResult)
        # Allow for query processing differences (case, punctuation)
        assert (
            result.query_processed.replace("?", "").replace(".", "").strip().lower()
            == basic_request.query.replace("?", "").replace(".", "").strip().lower()
        )
        assert result.processing_time_ms > 0
        assert len(result.results) <= basic_request.limit
        assert result._total_results >= 0
        assert result.cache_hit is False

    @pytest.mark.asyncio
    async def test_search_with_mock_results(self, orchestrator, basic_request):
        """Test search returns expected mock results."""
        result = await orchestrator.search(basic_request)

        # Should have mock results from _execute_search
        assert len(result.results) == basic_request.limit
        assert result._total_results == 20  # Mock returns 20 _total results

        # Check result structure
        first_result = result.results[0]
        assert "id" in first_result
        assert "content" in first_result
        assert "score" in first_result
        assert "metadata" in first_result
        assert first_result["id"] == "doc_0"

    @pytest.mark.asyncio
    async def test_search_modes(self, orchestrator):
        """Test different search modes."""
        # Basic mode
        basic_request = SearchRequest(
            query="test", mode=SearchMode.BASIC, enable_expansion=True
        )
        result = await orchestrator.search(basic_request)
        # Expansion should be disabled in basic mode even if requested
        assert "query_expansion" not in result.features_used

        # Enhanced mode
        enhanced_request = SearchRequest(
            query="test", mode=SearchMode.ENHANCED, enable_expansion=True
        )
        # Mock the private attribute instead of the property
        mock_service = Mock()
        mock_service.expand_query = AsyncMock(
            return_value=Mock(expanded_query="expanded test")
        )
        orchestrator._query_expansion_service = mock_service

        result = await orchestrator.search(enhanced_request)
        assert result.expanded_query == "expanded test"
        assert "query_expansion" in result.features_used

    @pytest.mark.asyncio
    async def test_search_with_limit_and_offset(self, orchestrator):
        """Test search respects limit and offset parameters."""
        request = SearchRequest(query="test", limit=5, offset=0)
        result = await orchestrator.search(request)

        assert len(result.results) == 5
        assert result.results[0]["id"] == "doc_0"  # First result

    @pytest.mark.asyncio
    async def test_search_error_handling(self, orchestrator, basic_request):
        """Test search error handling."""
        # Mock _execute_search to raise an exception
        with patch.object(
            orchestrator, "_execute_search", side_effect=Exception("Search failed")
        ):
            result = await orchestrator.search(basic_request)

            # Should return minimal result on error
            assert isinstance(result, SearchResult)
            assert result.results == []
            assert result._total_results == 0
            assert result.query_processed == basic_request.query
            assert result.processing_time_ms > 0


class TestFeatureIntegration:
    """Test integration of various search features."""

    @pytest.mark.asyncio
    async def test_query_expansion_feature(self, orchestrator):
        """Test query expansion feature."""
        request = SearchRequest(
            query="ML algorithms", mode=SearchMode.ENHANCED, enable_expansion=True
        )

        # Mock the expansion service
        mock_service = Mock()
        mock_expansion_result = Mock()
        mock_expansion_result.expanded_query = "ML machine learning algorithms"
        mock_service.expand_query = AsyncMock(return_value=mock_expansion_result)
        orchestrator._query_expansion_service = mock_service

        result = await orchestrator.search(request)

        assert result.expanded_query == "ML machine learning algorithms"
        assert "query_expansion" in result.features_used
        mock_service.expand_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_expansion_failure(self, orchestrator):
        """Test query expansion failure handling."""
        request = SearchRequest(
            query="test", mode=SearchMode.ENHANCED, enable_expansion=True
        )

        # Mock expansion service to raise exception
        mock_service = Mock()
        mock_service.expand_query = AsyncMock(side_effect=Exception("Expansion failed"))
        orchestrator._query_expansion_service = mock_service

        result = await orchestrator.search(request)

        # Search should continue without expansion
        assert result.expanded_query is None
        assert "query_expansion" not in result.features_used
        assert len(result.results) > 0  # Search should still succeed

    @pytest.mark.asyncio
    async def test_clustering_feature(self, orchestrator):
        """Test result clustering feature."""
        request = SearchRequest(
            query="test",
            enable_clustering=True,
            enable_expansion=False,  # Disable expansion to focus on clustering
        )

        # Mock _execute_search to return results with required fields for clustering
        with patch.object(orchestrator, "_execute_search") as mock_search:
            mock_search.return_value = [
                {
                    "id": "doc_0",
                    "title": "Test Document 0",
                    "content": "content 0",
                    "score": 0.9,
                },
                {
                    "id": "doc_1",
                    "title": "Test Document 1",
                    "content": "content 1",
                    "score": 0.8,
                },
                {
                    "id": "doc_2",
                    "title": "Test Document 2",
                    "content": "content 2",
                    "score": 0.7,
                },
                {
                    "id": "doc_3",
                    "title": "Test Document 3",
                    "content": "content 3",
                    "score": 0.6,
                },
                {
                    "id": "doc_4",
                    "title": "Test Document 4",
                    "content": "content 4",
                    "score": 0.5,
                },
                {
                    "id": "doc_5",
                    "title": "Test Document 5",
                    "content": "content 5",
                    "score": 0.4,
                },
            ]

            # Mock clustering service
            mock_service = Mock()
            mock_cluster = Mock()
            mock_cluster.cluster_id = "cluster_1"
            mock_cluster.label = "Technical Documentation"
            mock_cluster.results = [Mock(id="doc_0"), Mock(id="doc_1")]

            mock_clustering_result = Mock()
            mock_clustering_result.clusters = [mock_cluster]
            mock_service.cluster_results = AsyncMock(
                return_value=mock_clustering_result
            )
            orchestrator._clustering_service = mock_service

            result = await orchestrator.search(request)

            assert "result_clustering" in result.features_used
            # Check that cluster information was added to results
            assert result.results[0]["cluster_id"] == "cluster_1"
            assert result.results[0]["cluster_label"] == "Technical Documentation"

    @pytest.mark.asyncio
    async def test_clustering_with_insufficient_results(self, orchestrator):
        """Test clustering is skipped with insufficient results."""
        request = SearchRequest(
            query="test",
            limit=3,  # Less than 5 results
            enable_clustering=True,
        )

        # Mock to return only 3 results
        with patch.object(orchestrator, "_execute_search") as mock_search:
            mock_search.return_value = [
                {"id": "doc_0", "content": "result 0", "score": 0.9},
                {"id": "doc_1", "content": "result 1", "score": 0.8},
                {"id": "doc_2", "content": "result 2", "score": 0.7},
            ]

            result = await orchestrator.search(request)

            # Clustering should be skipped
            assert "result_clustering" not in result.features_used

    @pytest.mark.asyncio
    async def test_personalized_ranking_feature(self, orchestrator):
        """Test personalized ranking feature."""
        request = SearchRequest(
            query="test",
            enable_personalization=True,
            enable_expansion=False,  # Disable expansion to focus on ranking
            user_id="test_user",
        )

        # Mock _execute_search to return results with required fields for ranking
        with patch.object(orchestrator, "_execute_search") as mock_search:
            mock_search.return_value = [
                {
                    "id": "doc_0",
                    "title": "Test Document 0",
                    "content": "content 0",
                    "score": 0.9,
                },
                {
                    "id": "doc_1",
                    "title": "Test Document 1",
                    "content": "content 1",
                    "score": 0.8,
                },
            ]

            # Mock ranking service
            mock_service = Mock()
            mock_ranked = Mock()
            mock_ranked.result_id = "doc_0"
            mock_ranked.final_score = 0.95

            mock_ranking_result = Mock()
            mock_ranking_result.ranked_results = [mock_ranked]
            mock_service.rank_results = AsyncMock(return_value=mock_ranking_result)
            orchestrator._ranking_service = mock_service

            result = await orchestrator.search(request)

            assert "personalized_ranking" in result.features_used
            # Check that personalized score was added
            assert result.results[0]["personalized_score"] == 0.95

    @pytest.mark.asyncio
    async def test_personalized_ranking_without_user_id(self, orchestrator):
        """Test personalized ranking is skipped without user ID."""
        request = SearchRequest(
            query="test",
            enable_personalization=True,
            user_id=None,  # No user ID
        )

        result = await orchestrator.search(request)

        # Ranking should be skipped
        assert "personalized_ranking" not in result.features_used

    @patch("src.config.get_config")
    @pytest.mark.asyncio
    async def test_rag_feature_enabled(self, mock_get_config, orchestrator):
        """Test RAG answer generation feature."""
        # Mock config
        mock_config = Mock()
        mock_config.rag.enable_rag = True
        mock_config.rag.max_results_for_context = 5
        mock_config.rag.min_confidence_threshold = 0.7
        mock_get_config.return_value = mock_config

        request = SearchRequest(query="What is Python?", enable_rag=True)

        # Mock RAG generator
        mock_rag = Mock()
        mock_rag._initialized = True
        mock_rag.initialize = AsyncMock()

        mock_source = Mock()
        mock_source.source_id = "doc_0"
        mock_source.title = "Python Documentation"
        mock_source.url = "https://docs.python.org"
        mock_source.relevance_score = 0.9
        mock_source.excerpt = "Python is a programming language"

        mock_rag_result = Mock()
        mock_rag_result.answer = "Python is a high-level programming language."
        mock_rag_result.confidence_score = 0.85
        mock_rag_result.sources = [mock_source]
        mock_rag_result.metrics = Mock()
        mock_rag_result.metrics.model_dump.return_value = {"tokens": 100}

        mock_rag.generate_answer = AsyncMock(return_value=mock_rag_result)
        orchestrator._rag_generator = mock_rag

        result = await orchestrator.search(request)

        assert "rag_answer_generation" in result.features_used
        assert result.generated_answer == "Python is a high-level programming language."
        assert result.answer_confidence == 0.85
        assert len(result.answer_sources) == 1
        assert result.answer_sources[0]["source_id"] == "doc_0"
        assert result.answer_metrics == {"tokens": 100}

    @patch("src.config.get_config")
    @pytest.mark.asyncio
    async def test_rag_low_confidence_filtering(self, mock_get_config, orchestrator):
        """Test RAG answers are filtered by confidence threshold."""
        # Mock config with high confidence threshold
        mock_config = Mock()
        mock_config.rag.enable_rag = True
        mock_config.rag.max_results_for_context = 5
        mock_config.rag.min_confidence_threshold = 0.8  # High threshold
        mock_get_config.return_value = mock_config

        request = SearchRequest(query="What is Python?", enable_rag=True)

        # Mock RAG generator with low confidence
        mock_rag = Mock()
        mock_rag._initialized = True
        mock_rag.initialize = AsyncMock()

        mock_rag_result = Mock()
        mock_rag_result.answer = "Low confidence answer"
        mock_rag_result.confidence_score = 0.6  # Below threshold
        mock_rag_result.sources = []
        mock_rag_result.metrics = None

        mock_rag.generate_answer = AsyncMock(return_value=mock_rag_result)
        orchestrator._rag_generator = mock_rag

        result = await orchestrator.search(request)

        # RAG feature should not be in features_used due to low confidence
        assert "rag_answer_generation" not in result.features_used
        assert result.generated_answer is None
        assert result.answer_confidence is None

    @pytest.mark.asyncio
    async def test_rag_failure_handling(self, orchestrator):
        """Test RAG failure doesn't break search."""
        request = SearchRequest(query="test", enable_rag=True)

        # Mock RAG generator to raise exception
        mock_rag = Mock()
        mock_rag._initialized = True
        mock_rag.initialize = AsyncMock()
        mock_rag.generate_answer = AsyncMock(side_effect=Exception("RAG failed"))
        orchestrator._rag_generator = mock_rag

        result = await orchestrator.search(request)

        # Search should continue without RAG
        assert "rag_answer_generation" not in result.features_used
        assert result.generated_answer is None
        assert len(result.results) > 0  # Search should still succeed

    @pytest.mark.asyncio
    async def test_federated_search_feature(self, orchestrator):
        """Test federated search feature."""
        # Mock federated search service
        mock_federated_service = Mock()
        mock_fed_result = Mock()
        mock_fed_result.results = [
            {
                "id": "fed_doc_1",
                "title": "Federated Document 1",
                "content": "Content from collection A",
                "score": 0.95,
                "metadata": {"source": "collection_a"},
                "collection": "collection_a",
            },
            {
                "id": "fed_doc_2",
                "title": "Federated Document 2",
                "content": "Content from collection B",
                "score": 0.88,
                "metadata": {"source": "collection_b"},
                "collection": "collection_b",
            },
        ]
        mock_federated_service.search = AsyncMock(return_value=mock_fed_result)
        orchestrator._federated_service = mock_federated_service

        request = SearchRequest(
            query="federated search test",
            enable_federation=True,
            collection_name="target_collection",
        )

        result = await orchestrator.search(request)

        # Should use federated results
        assert len(result.results) == 2
        assert result.results[0]["id"] == "fed_doc_1"
        assert result.results[0]["title"] == "Federated Document 1"
        assert result.results[0]["collection"] == "collection_a"
        assert result.results[1]["collection"] == "collection_b"

        # Verify federated service was called
        mock_federated_service.search.assert_called_once()
        fed_request = mock_federated_service.search.call_args[0][0]
        assert fed_request.query == "federated search test"
        assert fed_request.target_collections == ["target_collection"]

    @pytest.mark.asyncio
    async def test_federated_search_failure_fallback(self, orchestrator):
        """Test federated search failure falls back to regular search."""
        # Mock federated service to fail
        mock_federated_service = Mock()
        mock_federated_service.search = AsyncMock(
            side_effect=Exception("Federation failed")
        )
        orchestrator._federated_service = mock_federated_service

        request = SearchRequest(query="test federation failure", enable_federation=True)

        result = await orchestrator.search(request)

        # Should fall back to mock results
        assert len(result.results) > 0
        assert result.results[0]["id"].startswith("doc_")  # Mock result format
        assert "Document" in result.results[0]["title"]  # Mock result format


class TestPipelineConfiguration:
    """Test pipeline configuration application."""

    def test_apply_pipeline_config_fast(self, orchestrator):
        """Test applying fast pipeline configuration."""
        # Create request with only required fields to test pure pipeline config
        request = SearchRequest(
            query="test",
            pipeline=SearchPipeline.FAST,
            # Don't set any optional fields to get pure pipeline behavior
        )

        config = orchestrator._apply_pipeline_config(request)

        # Should use pipeline defaults, not model defaults
        assert config["enable_expansion"] is False
        assert config["enable_clustering"] is False
        assert config["enable_personalization"] is False
        assert config["max_processing_time_ms"] == 1000.0

    def test_apply_pipeline_config_balanced(self, orchestrator):
        """Test applying balanced pipeline configuration."""
        request = SearchRequest(query="test", pipeline=SearchPipeline.BALANCED)

        config = orchestrator._apply_pipeline_config(request)

        assert config["enable_expansion"] is True
        assert config["enable_clustering"] is False
        assert config["enable_personalization"] is False
        assert config["max_processing_time_ms"] == 3000.0

    def test_apply_pipeline_config_comprehensive(self, orchestrator):
        """Test applying comprehensive pipeline configuration."""
        request = SearchRequest(query="test", pipeline=SearchPipeline.COMPREHENSIVE)

        config = orchestrator._apply_pipeline_config(request)

        assert config["enable_expansion"] is True
        assert config["enable_clustering"] is True
        assert config["enable_personalization"] is True
        assert config["max_processing_time_ms"] == 10000.0

    def test_apply_pipeline_config_with_overrides(self, orchestrator):
        """Test pipeline configuration with explicit request overrides."""
        request = SearchRequest(
            query="test",
            pipeline=SearchPipeline.FAST,  # Fast disables expansion
            enable_expansion=True,  # But explicitly enable it
            enable_clustering=True,  # Enable clustering too
            max_processing_time_ms=2000.0,  # Custom timeout
        )

        config = orchestrator._apply_pipeline_config(request)

        # Request overrides should take precedence
        assert config["enable_expansion"] is True
        assert config["enable_clustering"] is True
        assert config["enable_personalization"] is False  # Still from pipeline
        assert config["max_processing_time_ms"] == 2000.0


class TestCachingFunctionality:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_miss_and_hit(self, orchestrator, basic_request):
        """Test cache miss followed by cache hit."""
        # First search should miss cache
        result1 = await orchestrator.search(basic_request)
        assert result1.cache_hit is False
        assert orchestrator.stats["cache_misses"] == 1
        assert orchestrator.stats["cache_hits"] == 0

        # Second identical search should hit cache
        result2 = await orchestrator.search(basic_request)
        assert result2.cache_hit is True
        assert orchestrator.stats["cache_misses"] == 1
        assert orchestrator.stats["cache_hits"] == 1

        # Results should be identical
        assert result1.query_processed == result2.query_processed
        assert result1._total_results == result2._total_results

    @pytest.mark.asyncio
    async def test_cache_disabled(self, orchestrator):
        """Test search with caching disabled."""
        request = SearchRequest(query="test", enable_caching=False)

        result1 = await orchestrator.search(request)
        result2 = await orchestrator.search(request)

        # Both should be cache misses
        assert result1.cache_hit is False
        assert result2.cache_hit is False
        assert len(orchestrator.cache) == 0

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, orchestrator):
        """Test cache key generation."""
        request1 = SearchRequest(query="test", limit=10, user_id="user1")
        request2 = SearchRequest(
            query="test", limit=20, user_id="user1"
        )  # Different limit
        request3 = SearchRequest(
            query="test", limit=10, user_id="user2"
        )  # Different user

        key1 = orchestrator._get_cache_key(request1)
        key2 = orchestrator._get_cache_key(request2)
        key3 = orchestrator._get_cache_key(request3)

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, orchestrator):
        """Test cache respects size limit."""
        orchestrator.cache_size = 2  # Small cache for testing

        # Fill cache beyond limit
        for i in range(5):
            request = SearchRequest(query=f"test query {i}")
            await orchestrator.search(request)

        # Cache should not exceed size limit
        assert len(orchestrator.cache) <= orchestrator.cache_size

    def test_clear_cache(self, orchestrator):
        """Test cache clearing."""
        # Add items to cache
        orchestrator.cache["key1"] = "value1"
        orchestrator.cache["key2"] = "value2"
        assert len(orchestrator.cache) == 2

        orchestrator.clear_cache()
        assert len(orchestrator.cache) == 0


class TestStatisticsAndPerformance:
    """Test statistics and performance tracking."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, orchestrator, basic_request):
        """Test statistics are properly tracked."""
        initial_stats = orchestrator.get_stats()
        assert initial_stats["_total_searches"] == 0
        assert initial_stats["avg_processing_time"] == 0.0

        # Perform searches
        await orchestrator.search(basic_request)
        await orchestrator.search(basic_request)

        stats = orchestrator.get_stats()
        assert stats["_total_searches"] == 2
        assert stats["avg_processing_time"] > 0
        assert stats["cache_hits"] == 1  # Second search hits cache
        assert stats["cache_misses"] == 1  # First search misses cache

    @pytest.mark.asyncio
    async def test_average_processing_time_calculation(self, orchestrator):
        """Test average processing time calculation."""
        # Perform multiple searches
        for i in range(3):
            request = SearchRequest(
                query=f"test {i}"
            )  # Different queries to avoid cache
            await orchestrator.search(request)

        stats = orchestrator.get_stats()
        assert stats["_total_searches"] == 3
        assert stats["avg_processing_time"] > 0

    def test_get_stats_returns_copy(self, orchestrator):
        """Test get_stats returns a copy, not reference."""
        stats1 = orchestrator.get_stats()
        stats2 = orchestrator.get_stats()

        # Modify one copy
        stats1["_total_searches"] = 999

        # Other copy should be unchanged
        assert stats2["_total_searches"] != 999
        assert orchestrator.stats["_total_searches"] != 999


class TestUtilityMethods:
    """Test utility methods."""

    def test_apply_ranking(self, orchestrator):
        """Test applying ranking results to search results."""
        results = [
            {"id": "doc_1", "content": "Result 1", "score": 0.8},
            {"id": "doc_2", "content": "Result 2", "score": 0.9},
            {"id": "doc_3", "content": "Result 3", "score": 0.7},
        ]

        # Mock ranking result that reorders results
        mock_ranking_result = Mock()
        mock_ranked_1 = Mock()
        mock_ranked_1.result_id = "doc_2"
        mock_ranked_1.final_score = 0.95
        mock_ranked_2 = Mock()
        mock_ranked_2.result_id = "doc_1"
        mock_ranked_2.final_score = 0.85
        mock_ranked_3 = Mock()
        mock_ranked_3.result_id = "doc_3"
        mock_ranked_3.final_score = 0.75

        mock_ranking_result.ranked_results = [
            mock_ranked_1,
            mock_ranked_2,
            mock_ranked_3,
        ]

        ranked_results = orchestrator._apply_ranking(results, mock_ranking_result)

        # Results should be reordered and have personalized scores
        assert len(ranked_results) == 3
        assert ranked_results[0]["id"] == "doc_2"
        assert ranked_results[0]["personalized_score"] == 0.95
        assert ranked_results[1]["id"] == "doc_1"
        assert ranked_results[1]["personalized_score"] == 0.85

    def test_apply_ranking_with_missing_results(self, orchestrator):
        """Test applying ranking with some results missing from ranking."""
        results = [
            {"id": "doc_1", "content": "Result 1", "score": 0.8},
            {"id": "doc_2", "content": "Result 2", "score": 0.9},
        ]

        # Ranking result only includes one result
        mock_ranking_result = Mock()
        mock_ranked = Mock()
        mock_ranked.result_id = "doc_1"
        mock_ranked.final_score = 0.95
        mock_ranking_result.ranked_results = [mock_ranked]

        ranked_results = orchestrator._apply_ranking(results, mock_ranking_result)

        # Should include all original results
        assert len(ranked_results) == 2
        assert any(r["id"] == "doc_1" for r in ranked_results)
        assert any(r["id"] == "doc_2" for r in ranked_results)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, orchestrator):
        """Test search with empty query."""
        request = SearchRequest(query="")
        result = await orchestrator.search(request)

        assert isinstance(result, SearchResult)
        assert result.query_processed == ""

    @pytest.mark.asyncio
    async def test_very_long_query(self, orchestrator):
        """Test search with very long query."""
        long_query = "a" * 1000  # 1000 character query
        request = SearchRequest(query=long_query)
        result = await orchestrator.search(request)

        assert isinstance(result, SearchResult)
        assert result.query_processed == long_query

    @pytest.mark.asyncio
    async def test_zero_limit(self, _orchestrator):
        """Test search with zero limit should fail validation."""
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_with_all_features_failing(self, orchestrator):
        """Test search continues when all optional features fail."""
        request = SearchRequest(
            query="test",
            enable_expansion=True,
            enable_clustering=True,
            enable_personalization=True,
            enable_rag=True,
            user_id="test_user",
        )

        # Mock all services to fail
        mock_expansion = Mock()
        mock_expansion.expand_query = AsyncMock(
            side_effect=Exception("Expansion failed")
        )
        orchestrator._query_expansion_service = mock_expansion

        mock_clustering = Mock()
        mock_clustering.cluster_results = AsyncMock(
            side_effect=Exception("Clustering failed")
        )
        orchestrator._clustering_service = mock_clustering

        mock_ranking = Mock()
        mock_ranking.rank_results = AsyncMock(side_effect=Exception("Ranking failed"))
        orchestrator._ranking_service = mock_ranking

        mock_rag = Mock()
        mock_rag._initialized = True
        mock_rag.initialize = AsyncMock()
        mock_rag.generate_answer = AsyncMock(side_effect=Exception("RAG failed"))
        orchestrator._rag_generator = mock_rag

        result = await orchestrator.search(request)

        # Search should still succeed with basic results
        assert isinstance(result, SearchResult)
        assert len(result.results) > 0
        assert result.features_used == []  # No features should be marked as used

    @pytest.mark.asyncio
    async def test_search_with_network_timeout_simulation(self, orchestrator):
        """Test search behavior under timeout conditions."""
        request = SearchRequest(
            query="test",
            max_processing_time_ms=1.0,  # Very short timeout
        )

        # The current implementation doesn't enforce timeouts,
        # but this tests the parameter is accepted
        result = await orchestrator.search(request)
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_comprehensive_feature_integration(
        self, orchestrator, comprehensive_request
    ):
        """Test all features working together successfully."""
        # Mock all services to succeed
        with patch("src.config.get_config") as mock_get_config:
            # Mock federated search to return results that will be processed by other features
            # Need more than 5 results for clustering to trigger
            mock_federated_service = Mock()
            mock_fed_result = Mock()
            mock_fed_result.results = [
                {
                    "id": f"doc_{i}",
                    "title": f"Comprehensive Test Document {i}",
                    "content": f"Test content for comprehensive integration {i}",
                    "score": 0.9 - (i * 0.1),
                    "metadata": {"source": "test_collection"},
                }
                for i in range(8)  # 8 results to trigger clustering (> 5)
            ]
            mock_federated_service.search = AsyncMock(return_value=mock_fed_result)
            orchestrator._federated_service = mock_federated_service
            # Mock expansion
            mock_expansion = Mock()
            mock_expansion_result = Mock()
            mock_expansion_result.expanded_query = "expanded query"
            mock_expansion.expand_query = AsyncMock(return_value=mock_expansion_result)
            orchestrator._query_expansion_service = mock_expansion

            # Mock clustering
            mock_clustering = Mock()
            mock_cluster = Mock()
            mock_cluster.cluster_id = "cluster_1"
            mock_cluster.label = "Technical"
            mock_cluster.results = [Mock(id="doc_0")]
            mock_clustering_result = Mock()
            mock_clustering_result.clusters = [mock_cluster]
            mock_clustering.cluster_results = AsyncMock(
                return_value=mock_clustering_result
            )
            orchestrator._clustering_service = mock_clustering

            # Mock ranking
            mock_ranking = Mock()
            mock_ranked = Mock()
            mock_ranked.result_id = "doc_0"
            mock_ranked.final_score = 0.95
            mock_ranking_result = Mock()
            mock_ranking_result.ranked_results = [mock_ranked]
            mock_ranking.rank_results = AsyncMock(return_value=mock_ranking_result)
            orchestrator._ranking_service = mock_ranking

            # Mock RAG
            mock_config = Mock()
            mock_config.rag.enable_rag = True
            mock_config.rag.max_results_for_context = 5
            mock_config.rag.min_confidence_threshold = 0.7
            mock_get_config.return_value = mock_config

            mock_rag = Mock()
            mock_rag._initialized = True
            mock_rag.initialize = AsyncMock()
            mock_rag_result = Mock()
            mock_rag_result.answer = "Comprehensive answer"
            mock_rag_result.confidence_score = 0.9
            mock_rag_result.sources = []
            mock_rag_result.metrics = None
            mock_rag.generate_answer = AsyncMock(return_value=mock_rag_result)
            orchestrator._rag_generator = mock_rag

            result = await orchestrator.search(comprehensive_request)

            # All features should be used
            expected_features = {
                "query_expansion",
                "result_clustering",
                "personalized_ranking",
                "rag_answer_generation",
            }
            assert set(result.features_used) == expected_features

            # Check all feature results
            assert result.expanded_query == "expanded query"
            assert result.results[0]["cluster_id"] == "cluster_1"
            assert result.results[0]["personalized_score"] == 0.95
            assert result.generated_answer == "Comprehensive answer"
            assert result.answer_confidence == 0.9
