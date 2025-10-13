"""Unit tests for the HyDE query engine."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.cache.embedding_cache import EmbeddingCache
from src.services.cache.search_cache import SearchResultCache
from src.services.embeddings.manager import EmbeddingManager
from src.services.errors import APIError, EmbeddingServiceError
from src.services.hyde.config import HyDEConfig, HyDEMetricsConfig, HyDEPromptConfig
from src.services.hyde.engine import HyDEQueryEngine
from src.services.hyde.generator import GenerationResult, HypotheticalDocumentGenerator
from src.services.vector_db.service import VectorStoreService


class TestError(Exception):
    """Custom exception for this module."""


def _vector_match(
    *,
    doc_id: str = "doc1",
    score: float = 0.9,
    content: str = "example",
    title: str = "Example",
    url: str = "https://example.com",
    collection: str = "documents",
    metadata: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Construct a lightweight vector match object for adapter stubs."""

    return SimpleNamespace(
        id=doc_id,
        score=score,
        content=content,
        title=title,
        url=url,
        collection=collection,
        metadata=metadata or {},
        normalized_score=None,
        raw_score=None,
    )


class TestHyDEQueryEngine:
    """Tests for HyDEQueryEngine class."""

    @pytest.fixture
    def hyde_config(self) -> HyDEConfig:
        """Create HyDE configuration."""

        return HyDEConfig()

    @pytest.fixture
    def prompt_config(self) -> HyDEPromptConfig:
        """Create prompt configuration."""

        return HyDEPromptConfig()

    @pytest.fixture
    def metrics_config(self) -> HyDEMetricsConfig:
        """Create metrics configuration."""

        return HyDEMetricsConfig(
            track_generation_time=True,
            track_cache_hits=True,
            track_search_quality=True,
            ab_testing_enabled=False,
            control_group_percentage=0.5,
        )

    @pytest.fixture
    def mock_embedding_manager(self) -> EmbeddingManager:
        """Create mock embedding manager."""

        manager = MagicMock(spec=EmbeddingManager)
        manager.initialize = AsyncMock()
        manager.generate_embeddings = AsyncMock(
            return_value={"embeddings": [[0.1, 0.2, 0.3]]}
        )
        manager.rerank_results = AsyncMock()
        return manager

    @pytest.fixture
    def mock_vector_store(self) -> VectorStoreService:
        """Create mock vector store service."""

        service = MagicMock(spec=VectorStoreService)
        service.initialize = AsyncMock()
        service.search_vector = AsyncMock()
        service.hybrid_search = AsyncMock()
        service.search_documents = AsyncMock()
        return service

    @pytest.fixture
    def mock_embedding_cache(self) -> EmbeddingCache:
        """Create mock embedding cache."""

        cache = MagicMock(spec=EmbeddingCache)
        cache.get_embedding = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def mock_search_cache(self) -> SearchResultCache:
        """Create mock search cache."""

        cache = MagicMock(spec=SearchResultCache)
        cache.get_search_results = AsyncMock(return_value=None)
        cache.set_search_results = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def engine(
        self,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        metrics_config: HyDEMetricsConfig,
        mock_embedding_manager: EmbeddingManager,
        mock_vector_store: VectorStoreService,
        mock_embedding_cache: EmbeddingCache,
        mock_search_cache: SearchResultCache,
    ) -> HyDEQueryEngine:
        """Create HyDEQueryEngine instance wired with mocks."""

        engine = HyDEQueryEngine(
            config=hyde_config,
            prompt_config=prompt_config,
            metrics_config=metrics_config,
            embedding_manager=mock_embedding_manager,
            vector_store=mock_vector_store,
            embedding_cache=mock_embedding_cache,
            search_cache=mock_search_cache,
            openai_api_key="sk-test",
        )
        mock_generator = MagicMock(spec=HypotheticalDocumentGenerator)
        mock_generator.initialize = AsyncMock()
        mock_generator.cleanup = AsyncMock()
        mock_generator.generate_documents = AsyncMock(
            return_value=GenerationResult(
                documents=["doc1", "doc2"],
                generation_time=0.1,
                tokens_used=10,
                cost_estimate=0.01,
                diversity_score=0.5,
            )
        )
        mock_generator.get_metrics = MagicMock(return_value={"generated": 1})
        engine.generator = mock_generator
        return engine

    def test_init(
        self,
        engine: HyDEQueryEngine,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        metrics_config: HyDEMetricsConfig,
        mock_embedding_manager: EmbeddingManager,
        mock_vector_store: VectorStoreService,
        mock_embedding_cache: EmbeddingCache,
        mock_search_cache: SearchResultCache,
    ) -> None:
        """Test engine initialization wiring."""

        assert engine.config == hyde_config
        assert engine.prompt_config == prompt_config
        assert engine.metrics_config == metrics_config
        assert engine.embedding_manager == mock_embedding_manager
        assert engine.vector_store == mock_vector_store
        assert engine.embedding_cache is mock_embedding_cache
        assert engine.search_cache is mock_search_cache
        assert engine.embedding_cache_hit_count == 0
        assert engine.cache_hit_count == 0

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        engine: HyDEQueryEngine,
        mock_embedding_manager: EmbeddingManager,
        mock_vector_store: VectorStoreService,
    ) -> None:
        """Test successful engine initialization."""

        await engine.initialize()

        assert engine._initialized is True
        engine.generator.initialize.assert_called_once()
        mock_embedding_manager.initialize.assert_called_once()
        mock_vector_store.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, engine: HyDEQueryEngine) -> None:
        """Test engine cleanup."""

        engine._initialized = True
        await engine.cleanup()

        assert engine._initialized is False
        engine.generator.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_search_cache_hit(
        self,
        engine: HyDEQueryEngine,
        mock_search_cache: SearchResultCache,
        mock_vector_store: VectorStoreService,
    ) -> None:
        """Search returns cached results when available."""

        engine._initialized = True
        cached_results = [{"id": "cached_doc", "score": 0.8}]
        mock_search_cache.get_search_results.return_value = cached_results
        engine._get_or_generate_hyde_embedding = AsyncMock()

        results = await engine.enhanced_search("test query")

        assert results == cached_results
        assert engine.cache_hit_count == 1
        mock_search_cache.get_search_results.assert_called_once()
        mock_vector_store.search_vector.assert_not_called()
        engine._get_or_generate_hyde_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_enhanced_search_cache_miss_caches_results(
        self,
        engine: HyDEQueryEngine,
        mock_search_cache: SearchResultCache,
        mock_vector_store: VectorStoreService,
    ) -> None:
        """Search caches results when no cached entry exists."""

        engine._initialized = True
        mock_search_cache.get_search_results.return_value = None
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        mock_vector_store.search_vector.return_value = [
            _vector_match(doc_id="doc1", score=0.8)
        ]

        results = await engine.enhanced_search("test query", limit=5)

        assert len(results) == 1
        mock_vector_store.search_vector.assert_called_once()
        mock_search_cache.set_search_results.assert_called_once()
        args, kwargs = mock_search_cache.set_search_results.call_args
        assert kwargs["query"] == "test query"
        assert kwargs["limit"] == 5
        assert kwargs["search_type"] == "hyde"
        assert kwargs["params"]["hyde_enabled"] is True

    @pytest.mark.asyncio
    async def test_enhanced_search_with_reranking(
        self,
        engine: HyDEQueryEngine,
        mock_search_cache: SearchResultCache,
        mock_vector_store: VectorStoreService,
        mock_embedding_manager: EmbeddingManager,
    ) -> None:
        """Search applies reranking when enabled."""

        engine._initialized = True
        engine.config.enable_reranking = True
        mock_search_cache.get_search_results.return_value = None
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        initial_results = [
            _vector_match(doc_id="doc1", score=0.8),
            _vector_match(doc_id="doc2", score=0.7),
        ]
        mock_vector_store.search_vector.return_value = initial_results
        mock_embedding_manager.rerank_results.return_value = [
            {"id": "doc2", "score": 0.9},
            {"id": "doc1", "score": 0.6},
        ]

        results = await engine.enhanced_search("test query", use_cache=False)

        assert results[0]["id"] == "doc2"
        mock_embedding_manager.rerank_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_search_failure_with_fallback(
        self,
        engine: HyDEQueryEngine,
    ) -> None:
        """Search falls back to standard search when HyDE fails."""

        engine._initialized = True
        engine.config.enable_fallback = True
        engine._get_or_generate_hyde_embedding = AsyncMock(
            side_effect=RuntimeError("generation failure")
        )
        engine._fallback_search = AsyncMock(
            return_value=[{"id": "fallback", "score": 0.1}]
        )

        results = await engine.enhanced_search("test query")

        assert results[0]["id"] == "fallback"
        engine._fallback_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_search_failure_without_fallback(
        self,
        engine: HyDEQueryEngine,
    ) -> None:
        """Search raises when fallback disabled and HyDE fails."""

        engine._initialized = True
        engine.config.enable_fallback = False
        engine._get_or_generate_hyde_embedding = AsyncMock(
            side_effect=RuntimeError("generation failure")
        )

        with pytest.raises(EmbeddingServiceError):
            await engine.enhanced_search("test query")

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_cache_hit(
        self,
        engine: HyDEQueryEngine,
        mock_embedding_cache: EmbeddingCache,
    ) -> None:
        """Embeddings are returned from cache when available."""

        mock_embedding_cache.get_embedding.return_value = [0.1, 0.2, 0.3]
        embedding = await engine._get_or_generate_hyde_embedding(
            "test query", None, True
        )

        assert embedding == [0.1, 0.2, 0.3]
        assert engine.embedding_cache_hit_count == 1
        engine.generator.generate_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_generates_and_caches(
        self,
        engine: HyDEQueryEngine,
        mock_embedding_cache: EmbeddingCache,
        mock_embedding_manager: EmbeddingManager,
    ) -> None:
        """Embeddings are generated and stored when cache misses."""

        mock_embedding_cache.get_embedding.return_value = None
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.5, 0.6, 0.7]]
        }

        embedding = await engine._get_or_generate_hyde_embedding(
            "test query", "docs", True
        )

        assert pytest.approx(embedding[0], rel=1e-9) == 0.5
        mock_embedding_cache.set_embedding.assert_called_once()
        args, kwargs = mock_embedding_cache.set_embedding.call_args
        assert "docs" in kwargs["text"]
        assert kwargs["ttl"] == engine.config.cache_ttl_seconds

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_no_documents(
        self,
        engine: HyDEQueryEngine,
    ) -> None:
        """Raises when generator returns no documents."""

        engine.generator.generate_documents.return_value = GenerationResult(
            documents=[],
            generation_time=0.0,
            tokens_used=0,
            cost_estimate=0.0,
            diversity_score=0.0,
        )

        with pytest.raises(EmbeddingServiceError):
            await engine._get_or_generate_hyde_embedding("test", None, True)

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_missing_embeddings(
        self,
        engine: HyDEQueryEngine,
        mock_embedding_manager: EmbeddingManager,
    ) -> None:
        """Raises when embedding generation result lacks embeddings."""

        mock_embedding_manager.generate_embeddings.return_value = {"provider": "openai"}

        with pytest.raises(EmbeddingServiceError):
            await engine._get_or_generate_hyde_embedding("test", None, False)

    def test_get_performance_metrics(self, engine: HyDEQueryEngine) -> None:
        """Metrics include cache and generation summaries."""

        engine.search_count = 2
        engine.cache_hit_count = 1
        engine.embedding_cache_hit_count = 1
        engine.fallback_count = 1
        engine.total_search_time = 5.0

        metrics = engine.get_performance_metrics()

        assert metrics["search_performance"]["total_searches"] == 2
        assert metrics["cache_metrics"]["embedding_cache_hits"] == 1
        assert metrics["cache_metrics"]["result_cache_hits"] == 1

    def test_reset_metrics(self, engine: HyDEQueryEngine) -> None:
        """Resetting metrics clears counters."""

        engine.search_count = 5
        engine.cache_hit_count = 2
        engine.embedding_cache_hit_count = 1
        engine.total_search_time = 3.0

        engine.reset_metrics()

        assert engine.search_count == 0
        assert engine.cache_hit_count == 0
        assert engine.embedding_cache_hit_count == 0
        assert engine.total_search_time == 0.0

    @pytest.mark.asyncio
    async def test_enhanced_search_ab_testing_control(
        self,
        engine: HyDEQueryEngine,
    ) -> None:
        """AB testing respects control group routing."""

        engine._initialized = True
        engine.metrics_config.ab_testing_enabled = True
        engine._should_use_hyde_for_ab_test = AsyncMock(return_value=False)
        engine._fallback_search = AsyncMock(return_value=[{"id": "fallback"}])

        results = await engine.enhanced_search("test query", force_hyde=False)

        assert results == [{"id": "fallback"}]
        assert engine.control_group_searches == 1
        engine._fallback_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_search_handles_qdrant_error(
        self,
        engine: HyDEQueryEngine,
        mock_search_cache: SearchResultCache,
        mock_vector_store: VectorStoreService,
    ) -> None:
        """Vector store errors raise QdrantServiceError."""

        engine._initialized = True
        mock_search_cache.get_search_results.return_value = None
        engine._get_or_generate_hyde_embedding = AsyncMock(return_value=[0.4, 0.5, 0.6])
        mock_vector_store.search_vector.side_effect = Exception("qdrant down")

        with pytest.raises(EmbeddingServiceError):
            await engine.enhanced_search("test query", use_cache=False)

    @pytest.mark.asyncio
    async def test_fallback_search_error(
        self,
        engine: HyDEQueryEngine,
        mock_embedding_manager: EmbeddingManager,
        mock_vector_store: VectorStoreService,
    ) -> None:
        """Fallback search propagates failures."""

        mock_embedding_manager.generate_embeddings.side_effect = APIError("boom")

        with pytest.raises(EmbeddingServiceError):
            await engine._fallback_search("query", "documents", 5, None, "balanced")
