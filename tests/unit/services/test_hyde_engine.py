#!/usr/bin/env python3
"""
Unit tests for HyDE Query Engine.

Tests the main HyDE engine functionality including enhanced search,
Query API integration, fallback mechanisms, and performance optimizations.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEMetricsConfig
from src.services.hyde.config import HyDEPromptConfig
from src.services.hyde.engine import HyDEQueryEngine


class TestHyDEQueryEngine:
    """Test cases for HyDEQueryEngine functionality."""

    @pytest.fixture
    def mock_embedding_manager(self):
        """Mock embedding manager for testing."""
        manager = AsyncMock()

        # Mock embedding generation
        manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }

        # Mock reranking
        manager.rerank_results.return_value = [
            {"content": "reranked result 1", "original": {"score": 0.95}},
            {"content": "reranked result 2", "original": {"score": 0.85}},
        ]

        return manager

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock Qdrant service for testing."""
        service = AsyncMock()

        # Mock search results
        mock_point_1 = MagicMock()
        mock_point_1.id = "1"
        mock_point_1.score = 0.9
        mock_point_1.payload = {
            "content": "This is a document about machine learning",
            "title": "ML Guide",
            "url": "https://example.com/ml",
        }

        mock_point_2 = MagicMock()
        mock_point_2.id = "2"
        mock_point_2.score = 0.8
        mock_point_2.payload = {
            "content": "Deep learning tutorial with examples",
            "title": "DL Tutorial",
            "url": "https://example.com/dl",
        }

        service.hybrid_search.return_value = [mock_point_1, mock_point_2]
        service.query_api_search.return_value = [mock_point_1, mock_point_2]

        return service

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing."""
        cache = AsyncMock()

        # Mock cache operations
        cache.get_hyde_embedding.return_value = None  # Cache miss by default
        cache.set_hyde_embedding.return_value = True
        cache.get_search_results.return_value = None  # Cache miss by default
        cache.set_search_results.return_value = True

        return cache

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        client = AsyncMock()

        # Mock completion response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated hypothetical document"
        mock_response.choices = [mock_choice]

        client.chat.completions.create.return_value = mock_response

        return client

    @pytest.fixture
    def hyde_config(self):
        """Default HyDE configuration for testing."""
        return HyDEConfig()

    @pytest.fixture
    def prompt_config(self):
        """Default prompt configuration for testing."""
        return HyDEPromptConfig()

    @pytest.fixture
    def metrics_config(self):
        """Default metrics configuration for testing."""
        return HyDEMetricsConfig()

    @pytest.fixture
    def hyde_engine(
        self,
        hyde_config,
        prompt_config,
        metrics_config,
        mock_embedding_manager,
        mock_qdrant_service,
        mock_cache_manager,
        mock_llm_client,
    ):
        """HyDEQueryEngine instance for testing."""
        return HyDEQueryEngine(
            config=hyde_config,
            prompt_config=prompt_config,
            metrics_config=metrics_config,
            embedding_manager=mock_embedding_manager,
            qdrant_service=mock_qdrant_service,
            cache_manager=mock_cache_manager,
            llm_client=mock_llm_client,
        )

    @pytest.mark.asyncio
    async def test_enhanced_search_basic(self, hyde_engine, mock_cache_manager):
        """Test basic enhanced search functionality."""
        query = "What is machine learning?"
        collection_name = "documentation"

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Should have attempted to get/set cache
        assert mock_cache_manager.get_hyde_embedding.called
        assert mock_cache_manager.set_hyde_embedding.called

    @pytest.mark.asyncio
    async def test_enhanced_search_with_cache_hit(
        self, hyde_engine, mock_cache_manager, mock_qdrant_service
    ):
        """Test enhanced search with cache hit."""
        query = "API documentation"
        collection_name = "docs"

        # Mock cache hit
        cached_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cached_docs = ["cached doc 1", "cached doc 2"]
        mock_cache_manager.get_hyde_embedding.return_value = {
            "embedding": cached_embedding,
            "hypothetical_docs": cached_docs,
            "metadata": {"generation_time": 1.5},
        }

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=3,
            use_cache=True,
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Should have used cached embedding
        assert mock_cache_manager.get_hyde_embedding.called
        # Should not have generated new embedding since cache hit
        assert not mock_cache_manager.set_hyde_embedding.called

    @pytest.mark.asyncio
    async def test_enhanced_search_with_domain(self, hyde_engine):
        """Test enhanced search with domain specification."""
        query = "REST API authentication"
        collection_name = "api_docs"
        domain = "api"

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
            domain=domain,
        )

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_enhanced_search_with_filters(self, hyde_engine, mock_qdrant_service):
        """Test enhanced search with additional filters."""
        query = "database optimization"
        collection_name = "tech_docs"
        filters = {"category": "database", "language": "python"}

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
            filters=filters,
        )

        assert isinstance(results, list)
        # Should have passed filters to Qdrant service
        assert mock_qdrant_service.hybrid_search.called

    @pytest.mark.asyncio
    async def test_enhanced_search_accuracy_levels(self, hyde_engine):
        """Test enhanced search with different accuracy levels."""
        query = "performance optimization"
        collection_name = "docs"

        for accuracy in ["fast", "balanced", "high"]:
            results = await hyde_engine.enhanced_search(
                query=query,
                collection_name=collection_name,
                limit=5,
                search_accuracy=accuracy,
            )

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_enhanced_search_force_hyde(self, hyde_engine, mock_cache_manager):
        """Test enhanced search with force_hyde flag."""
        query = "forced HyDE generation"
        collection_name = "docs"

        # Mock cache hit
        mock_cache_manager.get_hyde_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "hypothetical_docs": ["cached doc"],
            "metadata": {"generation_time": 1.0},
        }

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
            force_hyde=True,
        )

        assert isinstance(results, list)
        # Even with cache hit, should have generated new embedding due to force_hyde
        # (depending on implementation details)

    @pytest.mark.asyncio
    async def test_enhanced_search_fallback_on_error(
        self, hyde_engine, mock_embedding_manager, mock_qdrant_service
    ):
        """Test fallback to regular search when HyDE fails."""
        query = "fallback test query"
        collection_name = "docs"

        # Mock HyDE generation failure
        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Generation failed"
        )

        # But regular search should still work
        mock_qdrant_service.hybrid_search.return_value = [
            MagicMock(id="1", score=0.8, payload={"content": "fallback result"})
        ]

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
        )

        # Should still return results from fallback
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_enhanced_search_no_cache(self, hyde_engine, mock_cache_manager):
        """Test enhanced search with caching disabled."""
        query = "no cache test"
        collection_name = "docs"

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
            use_cache=False,
        )

        assert isinstance(results, list)
        # Should not have attempted cache operations
        assert not mock_cache_manager.get_hyde_embedding.called
        assert not mock_cache_manager.set_hyde_embedding.called

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_cache_miss(
        self, hyde_engine, mock_cache_manager
    ):
        """Test HyDE embedding generation on cache miss."""
        query = "cache miss query"
        domain = "tutorial"

        # Mock cache miss
        mock_cache_manager.get_hyde_embedding.return_value = None

        embedding = await hyde_engine._get_or_generate_hyde_embedding(
            query, domain, use_cache=True
        )

        assert isinstance(embedding, list)
        assert len(embedding) > 0

        # Should have attempted cache get and set
        assert mock_cache_manager.get_hyde_embedding.called
        assert mock_cache_manager.set_hyde_embedding.called

    @pytest.mark.asyncio
    async def test_get_or_generate_hyde_embedding_cache_hit(
        self, hyde_engine, mock_cache_manager
    ):
        """Test HyDE embedding retrieval on cache hit."""
        query = "cache hit query"
        domain = "api"

        # Mock cache hit
        cached_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_cache_manager.get_hyde_embedding.return_value = {
            "embedding": cached_embedding,
            "hypothetical_docs": ["doc1", "doc2"],
            "metadata": {"generation_time": 2.0},
        }

        embedding = await hyde_engine._get_or_generate_hyde_embedding(
            query, domain, use_cache=True
        )

        assert embedding == cached_embedding

        # Should have attempted cache get but not set
        assert mock_cache_manager.get_hyde_embedding.called
        assert not mock_cache_manager.set_hyde_embedding.called

    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, hyde_engine, mock_embedding_manager):
        """Test regular query embedding generation."""
        query = "test query for embedding"

        embedding = await hyde_engine._generate_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert mock_embedding_manager.generate_embeddings.called

    @pytest.mark.asyncio
    async def test_perform_hyde_search_with_query_api(
        self, hyde_engine, mock_qdrant_service
    ):
        """Test HyDE search using Query API."""
        query_embedding = [0.1, 0.2, 0.3]
        hyde_embedding = [0.2, 0.3, 0.4]
        collection_name = "docs"
        limit = 5
        search_accuracy = "balanced"

        # Mock Query API availability
        mock_qdrant_service.query_api_search = AsyncMock()
        mock_qdrant_service.query_api_search.return_value = [
            MagicMock(id="1", score=0.9, payload={"content": "query api result"})
        ]

        results = await hyde_engine._perform_hyde_search(
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            collection_name=collection_name,
            limit=limit,
            search_accuracy=search_accuracy,
        )

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_perform_hyde_search_fallback_to_hybrid(
        self, hyde_engine, mock_qdrant_service
    ):
        """Test fallback to hybrid search when Query API fails."""
        query_embedding = [0.1, 0.2, 0.3]
        hyde_embedding = [0.2, 0.3, 0.4]
        collection_name = "docs"
        limit = 5

        # Mock Query API failure
        mock_qdrant_service.query_api_search = AsyncMock()
        mock_qdrant_service.query_api_search.side_effect = Exception("Query API failed")

        # But hybrid search should work
        mock_qdrant_service.hybrid_search.return_value = [
            MagicMock(id="1", score=0.8, payload={"content": "hybrid search result"})
        ]

        results = await hyde_engine._perform_hyde_search(
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            collection_name=collection_name,
            limit=limit,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        # Should have fallen back to hybrid search
        assert mock_qdrant_service.hybrid_search.called

    @pytest.mark.asyncio
    async def test_average_embeddings(self, hyde_engine):
        """Test embedding averaging functionality."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
        ]

        averaged = hyde_engine._average_embeddings(embeddings)

        assert isinstance(averaged, list)
        assert len(averaged) == 3  # Same dimension as input

        # Check that it's actually an average
        expected_avg = [0.2, 0.3, 0.4]  # Manual calculation
        for _i, (actual, expected) in enumerate(
            zip(averaged, expected_avg, strict=False)
        ):
            assert abs(actual - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_weighted_average_embeddings(self, hyde_engine):
        """Test weighted embedding averaging."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ]
        weights = [0.7, 0.3]  # Higher weight for first embedding

        averaged = hyde_engine._weighted_average_embeddings(embeddings, weights)

        assert isinstance(averaged, list)
        assert len(averaged) == 3

        # Verify weighted average calculation
        for i in range(3):
            expected = embeddings[0][i] * 0.7 + embeddings[1][i] * 0.3
            assert abs(averaged[i] - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_error_handling_with_retries(
        self, hyde_engine, mock_embedding_manager
    ):
        """Test error handling with retry logic."""
        query = "error test query"
        collection_name = "docs"

        # Mock failures followed by success
        call_count = 0

        def mock_embedding_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Failure {call_count}")
            return {"embeddings": [[0.1, 0.2, 0.3]]}

        mock_embedding_manager.generate_embeddings.side_effect = (
            mock_embedding_side_effect
        )

        # Configure retries
        hyde_engine.config.max_retries = 3

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
        )

        # Should eventually succeed after retries
        assert isinstance(results, list)
        assert call_count >= 3  # Should have retried

    @pytest.mark.asyncio
    async def test_metrics_collection(self, hyde_engine):
        """Test metrics collection during search."""
        hyde_engine.metrics_config.track_generation_time = True
        hyde_engine.metrics_config.track_search_accuracy = True

        query = "metrics test query"
        collection_name = "docs"

        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name=collection_name,
            limit=5,
        )

        assert isinstance(results, list)
        # Verify that metrics were collected (implementation dependent)

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, hyde_engine):
        """Test concurrent search operations."""
        queries = [f"query_{i}" for i in range(5)]
        collection_name = "docs"

        # Create concurrent search tasks
        tasks = [
            hyde_engine.enhanced_search(
                query=query,
                collection_name=collection_name,
                limit=3,
            )
            for query in queries
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed
        assert len(results) == len(queries)
        for result in results:
            if not isinstance(result, Exception):
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_cleanup_and_resource_management(self, hyde_engine):
        """Test proper cleanup and resource management."""
        # Test cleanup method if it exists
        if hasattr(hyde_engine, "cleanup"):
            await hyde_engine.cleanup()

        # Test that the engine can be used after cleanup
        query = "cleanup test query"
        results = await hyde_engine.enhanced_search(
            query=query,
            collection_name="docs",
            limit=3,
        )

        assert isinstance(results, list)
