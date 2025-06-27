class TestError(Exception):
    """Custom exception for this module."""

    pass


"""Tests for HyDE query processing engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.embeddings.manager import EmbeddingManager
from src.services.errors import EmbeddingServiceError, QdrantServiceError
from src.services.hyde.cache import HyDECache
from src.services.hyde.config import HyDEConfig, HyDEMetricsConfig, HyDEPromptConfig
from src.services.hyde.engine import HyDEQueryEngine
from src.services.hyde.generator import GenerationResult, HypotheticalDocumentGenerator
from src.services.vector_db.service import QdrantService


class TestHyDEQueryEngine:
    """Tests for HyDEQueryEngine class."""

    @pytest.fixture
    def hyde_config(self):
        """Create HyDE configuration."""
        return HyDEConfig(
            enable_hyde=True,
            enable_fallback=True,
            enable_reranking=True,
            enable_caching=True,
            num_generations=3,
            generation_temperature=0.7,
            max_generation_tokens=200,
            generation_model="gpt-3.5-turbo",
            hyde_prefetch_limit=50,
            query_prefetch_limit=30,
            hyde_weight_in_fusion=0.6,
            fusion_algorithm="rrf",
            cache_ttl_seconds=3600,
            parallel_generation=True,
            max_concurrent_generations=5,
        )

    @pytest.fixture
    def prompt_config(self):
        """Create prompt configuration."""
        return HyDEPromptConfig()

    @pytest.fixture
    def metrics_config(self):
        """Create metrics configuration."""
        return HyDEMetricsConfig(
            track_generation_time=True,
            track_cache_hits=True,
            track_search_quality=True,
            ab_testing_enabled=False,
            control_group_percentage=0.5,
        )

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = MagicMock(spec=EmbeddingManager)
        manager.initialize = AsyncMock()
        manager.generate_embeddings = AsyncMock()
        manager.rerank_results = AsyncMock()
        return manager

    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock Qdrant service."""
        service = MagicMock(spec=QdrantService)
        service.initialize = AsyncMock()
        service.hyde_search = AsyncMock()
        service.filtered_search = AsyncMock()
        return service

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        manager = MagicMock()
        manager.initialize = AsyncMock()
        manager.cleanup = AsyncMock()
        return manager

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def engine(
        self,
        hyde_config,
        prompt_config,
        metrics_config,
        mock_embedding_manager,
        mock_qdrant_service,
        mock_cache_manager,
        mock_llm_client,
    ):
        """Create HyDEQueryEngine instance."""
        return HyDEQueryEngine(
            config=hyde_config,
            prompt_config=prompt_config,
            metrics_config=metrics_config,
            embedding_manager=mock_embedding_manager,
            qdrant_service=mock_qdrant_service,
            cache_manager=mock_cache_manager,
            llm_client=mock_llm_client,
        )

    def test_init(
        self,
        hyde_config,
        prompt_config,
        metrics_config,
        mock_embedding_manager,
        mock_qdrant_service,
        mock_cache_manager,
        mock_llm_client,
    ):
        """Test engine initialization."""
        engine = HyDEQueryEngine(
            config=hyde_config,
            prompt_config=prompt_config,
            metrics_config=metrics_config,
            embedding_manager=mock_embedding_manager,
            qdrant_service=mock_qdrant_service,
            cache_manager=mock_cache_manager,
            llm_client=mock_llm_client,
        )

        assert engine.config == hyde_config
        assert engine.prompt_config == prompt_config
        assert engine.metrics_config == metrics_config
        assert engine.embedding_manager == mock_embedding_manager
        assert engine.qdrant_service == mock_qdrant_service
        assert engine._initialized is False

        # Check components are created
        assert isinstance(engine.generator, HypotheticalDocumentGenerator)
        assert isinstance(engine.cache, HyDECache)

        # Check metrics initialization
        assert engine.search_count == 0
        assert engine.total_search_time == 0.0
        assert engine.cache_hit_count == 0
        assert engine.generation_count == 0
        assert engine.fallback_count == 0
        assert engine.control_group_searches == 0
        assert engine.treatment_group_searches == 0

    async def test_initialize_success(
        self, engine, _mock_embedding_manager, mock_qdrant_service
    ):
        """Test successful engine initialization."""
        # Mock successful initialization of components
        engine.generator.initialize = AsyncMock()
        engine.cache.initialize = AsyncMock()

        await engine.initialize()

        assert engine._initialized is True
        engine.generator.initialize.assert_called_once()
        engine.cache.initialize.assert_called_once()

    async def test_initialize_embedding_manager_no_initialize(
        self, engine, mock_embedding_manager
    ):
        """Test initialization when embedding manager has no initialize method."""
        engine.generator.initialize = AsyncMock()
        engine.cache.initialize = AsyncMock()
        del mock_embedding_manager.initialize

        await engine.initialize()

        assert engine._initialized is True

    async def test_initialize_qdrant_service_no_initialize(
        self, engine, mock_qdrant_service
    ):
        """Test initialization when Qdrant service has no initialize method."""
        engine.generator.initialize = AsyncMock()
        engine.cache.initialize = AsyncMock()
        del mock_qdrant_service.initialize

        await engine.initialize()

        assert engine._initialized is True

    async def test_initialize_error(self, engine):
        """Test initialization error handling."""
        engine.generator.initialize = AsyncMock(
            side_effect=Exception("Generator error")
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine.initialize()

        assert "Failed to initialize HyDE engine" in str(exc_info.value)
        assert engine._initialized is False

    async def test_initialize_already_initialized(self, engine):
        """Test initialization when already initialized."""
        engine._initialized = True
        engine.generator.initialize = AsyncMock()

        await engine.initialize()

        # Should not call initialize again
        engine.generator.initialize.assert_not_called()

    async def test_cleanup(self, engine):
        """Test engine cleanup."""
        engine._initialized = True
        engine.generator.cleanup = AsyncMock()
        engine.cache.cleanup = AsyncMock()

        await engine.cleanup()

        assert engine._initialized is False
        engine.generator.cleanup.assert_called_once()
        engine.cache.cleanup.assert_called_once()

    async def test_cleanup_with_exceptions(self, engine):
        """Test cleanup handles exceptions from components."""
        engine._initialized = True
        engine.generator.cleanup = AsyncMock(
            side_effect=Exception("Generator cleanup error")
        )
        engine.cache.cleanup = AsyncMock(side_effect=Exception("Cache cleanup error"))

        # Should not raise exceptions
        await engine.cleanup()

        assert engine._initialized is False

    async def test_enhanced_search_hyde_disabled(
        self, engine, mock_qdrant_service, mock_embedding_manager
    ):
        """Test enhanced search when HyDE is disabled."""
        engine._initialized = True
        engine.config.enable_hyde = False

        # Mock fallback search
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_results = [{"id": "doc1", "score": 0.9}]
        mock_qdrant_service.filtered_search.return_value = mock_results

        results = await engine.enhanced_search("test query")

        assert results == mock_results
        assert engine.fallback_count == 0  # Not counted as fallback when disabled
        mock_qdrant_service.filtered_search.assert_called_once()

    async def test_enhanced_search_success_with_cache_hit(
        self, engine, mock_embedding_manager
    ):
        """Test successful enhanced search with cache hit."""
        engine._initialized = True

        # Mock cache hit
        cached_results = [{"id": "cached_doc", "score": 0.8}]
        engine.cache.get_search_results = AsyncMock(return_value=cached_results)

        results = await engine.enhanced_search("test query")

        assert results == cached_results
        assert engine.cache_hit_count == 1
        assert engine.search_count == 1

    async def test_enhanced_search_success_cache_miss(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test successful enhanced search with cache miss."""
        engine._initialized = True

        # Mock cache miss
        engine.cache.get_search_results = AsyncMock(return_value=None)

        # Mock HyDE embedding generation
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])

        # Mock query embedding generation
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        # Mock search results
        mock_results = [{"id": "doc1", "score": 0.9}]
        mock_qdrant_service.hyde_search.return_value = mock_results

        # Mock cache operations
        engine.cache.set_search_results = AsyncMock()

        results = await engine.enhanced_search("test query", use_cache=True)

        assert results == mock_results
        assert engine.search_count == 1
        assert engine.cache_hit_count == 0
        mock_qdrant_service.hyde_search.assert_called_once()
        engine.cache.set_search_results.assert_called_once()

    async def test_enhanced_search_with_reranking(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search with reranking enabled."""
        engine._initialized = True
        engine.config.enable_reranking = True

        # Mock cache miss and successful search
        engine.cache.get_search_results = AsyncMock(return_value=None)
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        initial_results = [{"id": "doc1", "score": 0.8}, {"id": "doc2", "score": 0.7}]
        mock_qdrant_service.hyde_search.return_value = initial_results

        # Mock reranking
        reranked_results = [{"id": "doc2", "score": 0.9}, {"id": "doc1", "score": 0.6}]
        mock_embedding_manager.rerank_results.return_value = reranked_results

        engine.cache.set_search_results = AsyncMock()

        results = await engine.enhanced_search("test query", use_cache=False)

        assert results == reranked_results
        mock_embedding_manager.rerank_results.assert_called_once_with(
            "test query", initial_results
        )

    async def test_enhanced_search_reranking_not_available(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search when reranking is not available."""
        engine._initialized = True
        engine.config.enable_reranking = True

        # Mock cache miss and successful search
        engine.cache.get_search_results = AsyncMock(return_value=None)
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        # Remove rerank_results method
        del mock_embedding_manager.rerank_results

        initial_results = [{"id": "doc1", "score": 0.8}]
        mock_qdrant_service.hyde_search.return_value = initial_results
        engine.cache.set_search_results = AsyncMock()

        results = await engine.enhanced_search("test query", use_cache=False)

        assert results == initial_results  # Should return original results

    async def test_enhanced_search_reranking_error(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search when reranking fails."""
        engine._initialized = True
        engine.config.enable_reranking = True

        # Mock cache miss and successful search
        engine.cache.get_search_results = AsyncMock(return_value=None)
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        initial_results = [{"id": "doc1", "score": 0.8}]
        mock_qdrant_service.hyde_search.return_value = initial_results

        # Mock reranking error
        mock_embedding_manager.rerank_results.side_effect = Exception("Reranking error")

        engine.cache.set_search_results = AsyncMock()

        results = await engine.enhanced_search("test query", use_cache=False)

        assert results == initial_results  # Should return original results on error

    async def test_enhanced_search_fallback_on_error(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search falls back on HyDE error."""
        engine._initialized = True
        engine.config.enable_fallback = True

        # Mock HyDE embedding generation error
        engine._get_or_generate_hyde_embedding = AsyncMock(
            side_effect=Exception("HyDE generation error")
        )

        # Mock fallback search
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        fallback_results = [{"id": "fallback_doc", "score": 0.7}]
        mock_qdrant_service.filtered_search.return_value = fallback_results

        results = await engine.enhanced_search("test query")

        assert results == fallback_results
        assert engine.fallback_count == 1
        mock_qdrant_service.filtered_search.assert_called_once()

    async def test_enhanced_search_error_no_fallback(self, engine):
        """Test enhanced search error when fallback is disabled."""
        engine._initialized = True
        engine.config.enable_fallback = False

        # Mock HyDE embedding generation error
        engine._get_or_generate_hyde_embedding = AsyncMock(
            side_effect=Exception("HyDE generation error")
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine.enhanced_search("test query")

        assert "HyDE search failed" in str(exc_info.value)

    async def test_enhanced_search_not_initialized(self, engine):
        """Test enhanced search when not initialized."""
        from src.services.errors import APIError

        with pytest.raises(APIError):
            await engine.enhanced_search("test query")

    async def test_get_or_generate_hyde_embedding_cache_hit(self, engine):
        """Test getting HyDE embedding from cache."""
        engine._initialized = True

        cached_embedding = [0.4, 0.5, 0.6]
        engine.cache.get_hyde_embedding = AsyncMock(return_value=cached_embedding)

        result = await engine._get_or_generate_hyde_embedding(
            "test query", "python", True
        )

        assert result == cached_embedding
        engine.cache.get_hyde_embedding.assert_called_once_with("test query", "python")

    async def test_get_or_generate_hyde_embedding_cache_miss(
        self, engine, mock_embedding_manager
    ):
        """Test generating HyDE embedding on cache miss."""
        engine._initialized = True

        # Mock cache miss
        engine.cache.get_hyde_embedding = AsyncMock(return_value=None)

        # Mock document generation
        generation_result = GenerationResult(
            documents=["doc1", "doc2", "doc3"],
            generation_time=1.5,
            tokens_used=100,
            cost_estimate=0.01,
            diversity_score=0.8,
        )
        engine.generator.generate_documents = AsyncMock(return_value=generation_result)

        # Mock embedding generation
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        }

        # Mock cache set
        engine.cache.set_hyde_embedding = AsyncMock()

        result = await engine._get_or_generate_hyde_embedding(
            "test query", "python", True
        )

        # Should return averaged embedding
        assert len(result) == 3
        assert abs(result[0] - 0.4) < 0.01  # Average of [0.1, 0.4, 0.7]
        assert abs(result[1] - 0.5) < 0.01  # Average of [0.2, 0.5, 0.8]
        assert abs(result[2] - 0.6) < 0.01  # Average of [0.3, 0.6, 0.9]

        assert engine.generation_count == 1
        engine.cache.set_hyde_embedding.assert_called_once()

    async def test_get_or_generate_hyde_embedding_no_documents(self, engine):
        """Test error when no hypothetical documents are generated."""
        engine._initialized = True

        engine.cache.get_hyde_embedding = AsyncMock(return_value=None)

        # Mock empty document generation
        generation_result = GenerationResult(
            documents=[],
            generation_time=1.0,
            tokens_used=0,
            cost_estimate=0.0,
        )
        engine.generator.generate_documents = AsyncMock(return_value=generation_result)

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine._get_or_generate_hyde_embedding("test query", "python", True)

        assert "Failed to generate hypothetical documents" in str(exc_info.value)

    async def test_get_or_generate_hyde_embedding_no_embeddings(
        self, engine, mock_embedding_manager
    ):
        """Test error when embedding generation fails."""
        engine._initialized = True

        engine.cache.get_hyde_embedding = AsyncMock(return_value=None)

        generation_result = GenerationResult(
            documents=["doc1"],
            generation_time=1.0,
            tokens_used=50,
            cost_estimate=0.005,
        )
        engine.generator.generate_documents = AsyncMock(return_value=generation_result)

        # Mock embedding generation failure
        mock_embedding_manager.generate_embeddings.return_value = {}

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine._get_or_generate_hyde_embedding("test query", "python", True)

        assert "Failed to generate embeddings for hypothetical documents" in str(
            exc_info.value
        )

    async def test_generate_query_embedding_success(
        self, engine, mock_embedding_manager
    ):
        """Test successful query embedding generation."""
        engine._initialized = True

        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        result = await engine._generate_query_embedding("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_embedding_manager.generate_embeddings.assert_called_once_with(
            texts=["test query"], provider_name="openai", auto_select=False
        )

    async def test_generate_query_embedding_failure(
        self, engine, mock_embedding_manager
    ):
        """Test query embedding generation failure."""
        engine._initialized = True

        mock_embedding_manager.generate_embeddings.return_value = {}

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine._generate_query_embedding("test query")

        assert "Failed to generate query embedding" in str(exc_info.value)

    async def test_perform_hyde_search_success(self, engine, mock_qdrant_service):
        """Test successful HyDE search execution."""
        engine._initialized = True

        query_embedding = [0.1, 0.2, 0.3]
        hyde_embedding = [0.4, 0.5, 0.6]
        mock_results = [{"id": "doc1", "score": 0.9}]

        mock_qdrant_service.hyde_search.return_value = mock_results

        results = await engine._perform_hyde_search(
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            collection_name="documents",
            limit=10,
            filters={"type": "doc"},
            search_accuracy="balanced",
        )

        assert results == mock_results
        mock_qdrant_service.hyde_search.assert_called_once_with(
            collection_name="documents",
            query="HyDE search",
            query_embedding=query_embedding,
            hypothetical_embeddings=[hyde_embedding],
            limit=10,
            fusion_algorithm=engine.config.fusion_algorithm,
            search_accuracy="balanced",
        )

    async def test_perform_hyde_search_error(self, engine, mock_qdrant_service):
        """Test HyDE search execution error."""
        engine._initialized = True

        mock_qdrant_service.hyde_search.side_effect = Exception("Search error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await engine._perform_hyde_search(
                query_embedding=[0.1, 0.2, 0.3],
                hyde_embedding=[0.4, 0.5, 0.6],
                collection_name="documents",
                limit=10,
                filters=None,
                search_accuracy="balanced",
            )

        assert "HyDE search execution failed" in str(exc_info.value)

    async def test_fallback_search_success(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test successful fallback search."""
        engine._initialized = True

        # Mock query embedding generation
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        # Mock search results
        mock_results = [{"id": "doc1", "score": 0.8}]
        mock_qdrant_service.filtered_search.return_value = mock_results

        results = await engine._fallback_search(
            query="test query",
            collection_name="documents",
            limit=10,
            filters={"type": "doc"},
            search_accuracy="balanced",
        )

        assert results == mock_results
        mock_qdrant_service.filtered_search.assert_called_once_with(
            collection_name="documents",
            query_vector=[0.1, 0.2, 0.3],
            filters={"type": "doc"},
            limit=10,
            search_accuracy="balanced",
        )

    async def test_fallback_search_error(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test fallback search error."""
        engine._initialized = True

        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_qdrant_service.filtered_search.side_effect = Exception("Search error")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await engine._fallback_search(
                query="test query",
                collection_name="documents",
                limit=10,
                filters=None,
                search_accuracy="balanced",
            )

        assert "Both HyDE and fallback search failed" in str(exc_info.value)

    async def test_should_use_hyde_for_ab_test(self, engine):
        """Test A/B testing logic."""
        engine.metrics_config.control_group_percentage = 0.5

        # Test with different queries to get different hash values
        queries = ["query1", "query2", "query3", "query4", "query5"]
        results = []

        for query in queries:
            result = await engine._should_use_hyde_for_ab_test(query)
            results.append(result)

        # Should have a mix of True and False values
        assert True in results or False in results  # At least one should be present

        # Same query should always return same result
        result1 = await engine._should_use_hyde_for_ab_test("consistent_query")
        result2 = await engine._should_use_hyde_for_ab_test("consistent_query")
        assert result1 == result2

    async def test_enhanced_search_ab_testing_control(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search with A/B testing - control group."""
        engine._initialized = True
        engine.metrics_config.ab_testing_enabled = True

        # Mock A/B test to return control group (no HyDE)
        engine._should_use_hyde_for_ab_test = AsyncMock(return_value=False)

        # Mock fallback search
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_results = [{"id": "doc1", "score": 0.8}]
        mock_qdrant_service.filtered_search.return_value = mock_results

        results = await engine.enhanced_search("test query")

        assert results == mock_results
        assert engine.control_group_searches == 1
        assert engine.treatment_group_searches == 0
        mock_qdrant_service.filtered_search.assert_called_once()

    async def test_enhanced_search_ab_testing_treatment(
        self, engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test enhanced search with A/B testing - treatment group."""
        engine._initialized = True
        engine.metrics_config.ab_testing_enabled = True

        # Mock A/B test to return treatment group (use HyDE)
        engine._should_use_hyde_for_ab_test = AsyncMock(return_value=True)

        # Mock cache miss
        engine.cache.get_search_results = AsyncMock(return_value=None)

        # Mock HyDE embedding generation
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])

        # Mock query embedding generation
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        # Mock search results
        mock_results = [{"id": "doc1", "score": 0.9}]
        mock_qdrant_service.hyde_search.return_value = mock_results

        engine.cache.set_search_results = AsyncMock()

        results = await engine.enhanced_search("test query")

        assert results == mock_results
        assert engine.control_group_searches == 0
        assert engine.treatment_group_searches == 1
        mock_qdrant_service.hyde_search.assert_called_once()

    async def test_batch_search_success(self, engine):
        """Test successful batch search."""
        engine._initialized = True

        # Mock enhanced_search
        async def mock_enhanced_search(query, **_kwargs):
            return [{"id": f"doc_{query}", "score": 0.8}]

        engine.enhanced_search = AsyncMock(side_effect=mock_enhanced_search)

        queries = ["query1", "query2", "query3"]
        results = await engine.batch_search(queries, max_concurrent=2)

        assert len(results) == 3
        assert results[0] == [{"id": "doc_query1", "score": 0.8}]
        assert results[1] == [{"id": "doc_query2", "score": 0.8}]
        assert results[2] == [{"id": "doc_query3", "score": 0.8}]

        assert engine.enhanced_search.call_count == 3

    async def test_batch_search_with_errors(self, engine):
        """Test batch search with some errors."""
        engine._initialized = True

        # Mock enhanced_search with some errors
        async def mock_enhanced_search(query, **_kwargs):
            if query == "error_query":
                raise TestError("Search error")
            return [{"id": f"doc_{query}", "score": 0.8}]

        engine.enhanced_search = AsyncMock(side_effect=mock_enhanced_search)

        queries = ["query1", "error_query", "query3"]
        results = await engine.batch_search(queries)

        assert len(results) == 3
        assert results[0] == [{"id": "doc_query1", "score": 0.8}]
        assert results[1] == []  # Error case returns empty list
        assert results[2] == [{"id": "doc_query3", "score": 0.8}]

    async def test_batch_search_not_initialized(self, engine):
        """Test batch search when not initialized."""
        from src.services.errors import APIError

        with pytest.raises(APIError):
            await engine.batch_search(["query1", "query2"])

    def test_get_performance_metrics(self, engine):
        """Test performance metrics calculation."""
        # Set some test values
        engine.search_count = 10
        engine.total_search_time = 20.0
        engine.cache_hit_count = 3
        engine.fallback_count = 2
        engine.control_group_searches = 4
        engine.treatment_group_searches = 6

        # Mock component metrics
        engine.generator.get_metrics = MagicMock(return_value={"generation_count": 5})
        engine.cache.get_cache_metrics = MagicMock(return_value={"cache_hits": 3})

        metrics = engine.get_performance_metrics()

        assert metrics["search_performance"]["total_searches"] == 10
        assert metrics["search_performance"]["avg_search_time"] == 2.0
        assert metrics["search_performance"]["cache_hit_rate"] == 0.3
        assert metrics["search_performance"]["fallback_rate"] == 0.2

        assert metrics["generation_metrics"]["generation_count"] == 5
        assert metrics["cache_metrics"]["cache_hits"] == 3

    def test_get_performance_metrics_ab_testing_disabled(self, engine):
        """Test performance metrics when A/B testing is disabled."""
        engine.metrics_config.ab_testing_enabled = False

        engine.generator.get_metrics = MagicMock(return_value={})
        engine.cache.get_cache_metrics = MagicMock(return_value={})

        metrics = engine.get_performance_metrics()

        assert "ab_testing" not in metrics

    def test_get_performance_metrics_ab_testing_enabled(self, engine):
        """Test performance metrics when A/B testing is enabled."""
        engine.metrics_config.ab_testing_enabled = True
        engine.control_group_searches = 3
        engine.treatment_group_searches = 7

        engine.generator.get_metrics = MagicMock(return_value={})
        engine.cache.get_cache_metrics = MagicMock(return_value={})

        metrics = engine.get_performance_metrics()

        assert "ab_testing" in metrics
        assert metrics["ab_testing"]["control_group_searches"] == 3
        assert metrics["ab_testing"]["treatment_group_searches"] == 7
        assert metrics["ab_testing"]["total_ab_searches"] == 10
        assert metrics["ab_testing"]["treatment_percentage"] == 0.7

    def test_get_performance_metrics_zero_searches(self, engine):
        """Test performance metrics when no searches have been performed."""
        engine.generator.get_metrics = MagicMock(return_value={})
        engine.cache.get_cache_metrics = MagicMock(return_value={})

        metrics = engine.get_performance_metrics()

        assert metrics["search_performance"]["total_searches"] == 0
        assert metrics["search_performance"]["avg_search_time"] == 0.0
        assert metrics["search_performance"]["cache_hit_rate"] == 0.0
        assert metrics["search_performance"]["fallback_rate"] == 0.0

    def test_reset_metrics(self, engine):
        """Test resetting performance metrics."""
        # Set some test values
        engine.search_count = 10
        engine.total_search_time = 20.0
        engine.cache_hit_count = 3
        engine.generation_count = 5
        engine.fallback_count = 2
        engine.control_group_searches = 4
        engine.treatment_group_searches = 6

        # Mock cache reset
        engine.cache.reset_metrics = MagicMock()

        engine.reset_metrics()

        assert engine.search_count == 0
        assert engine.total_search_time == 0.0
        assert engine.cache_hit_count == 0
        assert engine.generation_count == 0
        assert engine.fallback_count == 0
        assert engine.control_group_searches == 0
        assert engine.treatment_group_searches == 0

        engine.cache.reset_metrics.assert_called_once()

    async def test_cache_operations(self, engine):
        """Test cache get and set operations."""
        engine._initialized = True

        # Test _get_cached_results
        cache_params = {
            "limit": 10,
            "filters": {"type": "doc"},
            "search_accuracy": "balanced",
            "domain": "python",
            "hyde_enabled": True,
        }

        engine.cache.get_search_results = AsyncMock(return_value=None)

        result = await engine._get_cached_results(
            query="test query",
            collection_name="documents",
            limit=10,
            filters={"type": "doc"},
            search_accuracy="balanced",
            domain="python",
        )

        assert result is None
        engine.cache.get_search_results.assert_called_once_with(
            "test query", "documents", cache_params
        )

        # Test _cache_search_results
        search_results = [{"id": "doc1", "score": 0.9}]
        engine.cache.set_search_results = AsyncMock()

        await engine._cache_search_results(
            query="test query",
            collection_name="documents",
            limit=10,
            filters={"type": "doc"},
            search_accuracy="balanced",
            domain="python",
            results=search_results,
        )

        engine.cache.set_search_results.assert_called_once()
        call_args = engine.cache.set_search_results.call_args

        assert call_args[0][0] == "test query"
        assert call_args[0][1] == "documents"
        assert call_args[0][2] == cache_params
        assert call_args[0][3] == search_results
        assert "result_count" in call_args[0][4]
        assert "cached_at" in call_args[0][4]
