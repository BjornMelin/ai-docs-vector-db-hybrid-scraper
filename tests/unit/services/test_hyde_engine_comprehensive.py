"""Comprehensive tests for HyDE query engine."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.errors import EmbeddingServiceError
from src.services.errors import QdrantServiceError
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEMetricsConfig
from src.services.hyde.config import HyDEPromptConfig
from src.services.hyde.engine import HyDEQueryEngine


@pytest.fixture
def mock_config():
    """Create mock HyDE configuration."""
    return HyDEConfig(
        enable_hyde=True,
        enable_fallback=True,
        enable_ab_testing=True,
        ab_testing_ratio=0.5,
        cache_hypothetical_docs=True,
        cache_search_results=True,
        enable_query_api=True,
        max_concurrent_searches=5,
        search_timeout_seconds=30,
    )


@pytest.fixture
def mock_prompt_config():
    """Create mock prompt configuration."""
    return HyDEPromptConfig(
        default_template="Write a document that answers: {query}",
        domain_templates={
            "api": "Write an API documentation that explains: {query}",
            "tutorial": "Write a tutorial that covers: {query}",
        },
        document_count=3,
        max_tokens=150,
        temperature=0.7,
    )


@pytest.fixture
def mock_metrics_config():
    """Create mock metrics configuration."""
    return HyDEMetricsConfig(
        track_performance=True,
        track_cache_stats=True,
        track_generation_metrics=True,
        metrics_window_minutes=60,
        alert_thresholds={
            "search_latency_p95": 1000,
            "cache_hit_rate": 0.5,
            "generation_failure_rate": 0.1,
        },
    )


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    manager.get_batch_embeddings = AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    return manager


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    service.hybrid_search = AsyncMock(
        return_value=[
            {"id": "doc1", "score": 0.95, "payload": {"content": "Result 1"}},
            {"id": "doc2", "score": 0.87, "payload": {"content": "Result 2"}},
        ]
    )
    service.search = AsyncMock(
        return_value=[
            {"id": "doc1", "score": 0.95},
            {"id": "doc2", "score": 0.87},
        ]
    )
    return service


@pytest.fixture
def mock_cache_manager():
    """Create mock cache manager."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.get = AsyncMock(return_value=None)
    manager.set = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value="This is a hypothetical document about the query."
    )
    return client


@pytest.fixture
def hyde_engine(
    mock_config,
    mock_prompt_config,
    mock_metrics_config,
    mock_embedding_manager,
    mock_qdrant_service,
    mock_cache_manager,
    mock_llm_client,
):
    """Create HyDE query engine for testing."""
    return HyDEQueryEngine(
        config=mock_config,
        prompt_config=mock_prompt_config,
        metrics_config=mock_metrics_config,
        embedding_manager=mock_embedding_manager,
        qdrant_service=mock_qdrant_service,
        cache_manager=mock_cache_manager,
        llm_client=mock_llm_client,
    )


class TestHyDEEngineInitialization:
    """Test HyDE engine initialization."""

    def test_engine_initialization(
        self,
        hyde_engine,
        mock_config,
        mock_prompt_config,
        mock_metrics_config,
        mock_embedding_manager,
        mock_qdrant_service,
    ):
        """Test basic engine initialization."""
        assert hyde_engine.config == mock_config
        assert hyde_engine.prompt_config == mock_prompt_config
        assert hyde_engine.metrics_config == mock_metrics_config
        assert hyde_engine.embedding_manager == mock_embedding_manager
        assert hyde_engine.qdrant_service == mock_qdrant_service
        assert hyde_engine._initialized is False

        # Check performance tracking initialization
        assert hyde_engine.search_count == 0
        assert hyde_engine.total_search_time == 0.0
        assert hyde_engine.cache_hit_count == 0
        assert hyde_engine.generation_count == 0
        assert hyde_engine.fallback_count == 0

        # Check A/B testing initialization
        assert hyde_engine.control_group_searches == 0
        assert hyde_engine.treatment_group_searches == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, hyde_engine):
        """Test successful initialization."""
        await hyde_engine.initialize()

        assert hyde_engine._initialized is True
        # Verify all components were initialized
        hyde_engine.generator.initialize.assert_called_once()
        hyde_engine.cache.initialize.assert_called_once()
        hyde_engine.embedding_manager.initialize.assert_called_once()
        hyde_engine.qdrant_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_component_failure(self, hyde_engine):
        """Test initialization when a component fails."""
        hyde_engine.generator.initialize.side_effect = Exception(
            "Generator init failed"
        )

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize HyDE engine"
        ):
            await hyde_engine.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, hyde_engine):
        """Test that initialization is idempotent."""
        await hyde_engine.initialize()
        await hyde_engine.initialize()  # Second call

        # Should only initialize once
        hyde_engine.generator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, hyde_engine):
        """Test engine cleanup."""
        hyde_engine._initialized = True

        await hyde_engine.cleanup()

        # Verify all components were cleaned up
        hyde_engine.generator.cleanup.assert_called_once()
        hyde_engine.cache.cleanup.assert_called_once()
        hyde_engine.embedding_manager.cleanup.assert_called_once()
        hyde_engine.qdrant_service.cleanup.assert_called_once()
        assert hyde_engine._initialized is False


class TestHyDESearch:
    """Test HyDE search operations."""

    @pytest.mark.asyncio
    async def test_search_with_cache_hit(self, hyde_engine):
        """Test search with cached results."""
        hyde_engine._initialized = True

        # Mock cache hit
        cached_results = [
            {"id": "cached1", "score": 0.9, "payload": {"content": "Cached result"}}
        ]
        hyde_engine.cache.get_cached_search_results = AsyncMock(
            return_value={"results": cached_results, "metadata": {"cached": True}}
        )

        query = "machine learning algorithms"
        collection = "documentation"

        results = await hyde_engine.search(
            query=query,
            collection=collection,
            limit=10,
        )

        assert results == cached_results
        assert hyde_engine.cache_hit_count == 1
        # Should not call generator or Qdrant for cache hit
        hyde_engine.generator.generate_hypothetical_documents.assert_not_called()
        hyde_engine.qdrant_service.hybrid_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_with_cache_miss(self, hyde_engine):
        """Test search with cache miss - full HyDE pipeline."""
        hyde_engine._initialized = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock hypothetical document generation
        hypothetical_docs = [
            "Machine learning is a subset of AI...",
            "Algorithms in ML include supervised learning...",
            "Neural networks are fundamental to deep learning...",
        ]
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=hypothetical_docs
        )

        # Mock embedding generation
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        # Mock search results
        search_results = [
            {"id": "doc1", "score": 0.95, "payload": {"content": "ML tutorial"}},
            {"id": "doc2", "score": 0.87, "payload": {"content": "Algorithm guide"}},
        ]
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(
            return_value=search_results
        )

        # Mock cache set
        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        query = "machine learning algorithms"
        collection = "documentation"

        results = await hyde_engine.search(
            query=query,
            collection=collection,
            limit=10,
        )

        assert results == search_results
        assert hyde_engine.search_count == 1
        assert hyde_engine.generation_count == 1

        # Verify full pipeline was executed
        hyde_engine.generator.generate_hypothetical_documents.assert_called_once()
        hyde_engine.embedding_manager.get_embedding.assert_called()
        hyde_engine.qdrant_service.hybrid_search.assert_called_once()
        hyde_engine.cache.cache_search_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_domain_specific_generation(self, hyde_engine):
        """Test search with domain-specific document generation."""
        hyde_engine._initialized = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock domain-specific generation
        api_docs = [
            "GET /api/users - Returns list of users",
            "POST /api/users - Creates a new user",
            "Authentication via Bearer token required",
        ]
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=api_docs
        )

        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(return_value=[])
        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        query = "user management API"
        collection = "api_docs"
        domain = "api"

        await hyde_engine.search(
            query=query,
            collection=collection,
            domain=domain,
            limit=10,
        )

        # Verify domain was passed to generator
        call_args = hyde_engine.generator.generate_hypothetical_documents.call_args
        assert call_args[1]["domain"] == domain

    @pytest.mark.asyncio
    async def test_search_with_fallback_on_generation_failure(self, hyde_engine):
        """Test search fallback when HyDE generation fails."""
        hyde_engine._initialized = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock generation failure
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            side_effect=Exception("Generation failed")
        )

        # Mock fallback search
        fallback_results = [
            {"id": "fallback1", "score": 0.8, "payload": {"content": "Direct search"}}
        ]
        hyde_engine.qdrant_service.search = AsyncMock(return_value=fallback_results)
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        query = "test query"
        collection = "docs"

        results = await hyde_engine.search(query=query, collection=collection)

        assert results == fallback_results
        assert hyde_engine.fallback_count == 1

        # Verify fallback search was used
        hyde_engine.qdrant_service.search.assert_called_once()
        hyde_engine.qdrant_service.hybrid_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, hyde_engine):
        """Test search timeout handling."""
        hyde_engine._initialized = True
        hyde_engine.config.search_timeout_seconds = 0.1  # Very short timeout

        # Mock slow operations
        async def slow_generation(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            return ["Slow doc"]

        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            side_effect=slow_generation
        )
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        hyde_engine.qdrant_service.search = AsyncMock(return_value=[])

        query = "timeout test"
        collection = "docs"

        # Should timeout and fall back
        await hyde_engine.search(query=query, collection=collection)

        assert hyde_engine.fallback_count == 1
        # Should use direct search as fallback
        hyde_engine.qdrant_service.search.assert_called_once()


class TestBatchSearch:
    """Test batch search operations."""

    @pytest.mark.asyncio
    async def test_batch_search_basic(self, hyde_engine):
        """Test basic batch search functionality."""
        hyde_engine._initialized = True

        queries = [
            {"query": "ML algorithms", "collection": "tutorials"},
            {"query": "API design", "collection": "api_docs"},
            {"query": "database optimization", "collection": "guides"},
        ]

        # Mock individual search results
        search_results = [
            [{"id": "ml1", "score": 0.9}],
            [{"id": "api1", "score": 0.8}],
            [{"id": "db1", "score": 0.85}],
        ]

        # Mock search method
        hyde_engine.search = AsyncMock(side_effect=search_results)

        results = await hyde_engine.batch_search(queries, max_concurrent=2)

        assert len(results) == len(queries)
        assert results == search_results
        assert hyde_engine.search.call_count == len(queries)

    @pytest.mark.asyncio
    async def test_batch_search_with_concurrency_control(self, hyde_engine):
        """Test batch search with concurrency control."""
        hyde_engine._initialized = True

        queries = [{"query": f"query_{i}", "collection": "docs"} for i in range(10)]

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0

        async def mock_search(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return [{"id": "result", "score": 0.9}]

        hyde_engine.search = AsyncMock(side_effect=mock_search)

        await hyde_engine.batch_search(queries, max_concurrent=3)

        # Should not exceed concurrency limit
        assert max_concurrent_seen <= 3

    @pytest.mark.asyncio
    async def test_batch_search_error_handling(self, hyde_engine):
        """Test batch search error handling."""
        hyde_engine._initialized = True

        queries = [
            {"query": "good query", "collection": "docs"},
            {"query": "bad query", "collection": "docs"},
            {"query": "another good query", "collection": "docs"},
        ]

        # Mock one search failing
        async def mock_search(*args, **kwargs):
            if "bad query" in args[0]:
                raise Exception("Search failed")
            return [{"id": "result", "score": 0.9}]

        hyde_engine.search = AsyncMock(side_effect=mock_search)

        results = await hyde_engine.batch_search(queries, return_exceptions=True)

        assert len(results) == len(queries)
        assert isinstance(results[1], Exception)  # Failed search
        assert isinstance(results[0], list)  # Successful search
        assert isinstance(results[2], list)  # Successful search


class TestABTesting:
    """Test A/B testing functionality."""

    @pytest.mark.asyncio
    async def test_ab_testing_assignment(self, hyde_engine):
        """Test A/B testing group assignment."""
        hyde_engine._initialized = True
        hyde_engine.config.ab_testing_ratio = 0.5

        # Mock search methods
        hyde_engine.search = AsyncMock(return_value=[{"id": "hyde", "score": 0.9}])
        hyde_engine._direct_search = AsyncMock(
            return_value=[{"id": "direct", "score": 0.8}]
        )

        # Run multiple searches to test distribution
        for i in range(20):
            query = f"test query {i}"
            await hyde_engine.search_with_ab_testing(
                query=query, collection="docs", user_id=f"user_{i}"
            )

        # Both groups should have some searches
        assert hyde_engine.control_group_searches > 0
        assert hyde_engine.treatment_group_searches > 0

        # Total should equal number of searches
        total = (
            hyde_engine.control_group_searches + hyde_engine.treatment_group_searches
        )
        assert total == 20

    def test_ab_testing_group_determination(self, hyde_engine):
        """Test A/B testing group determination logic."""
        hyde_engine.config.ab_testing_ratio = 0.3

        # Test deterministic assignment based on user_id
        user_id = "test_user_123"
        group1 = hyde_engine._determine_ab_group(user_id)
        group2 = hyde_engine._determine_ab_group(user_id)
        assert group1 == group2  # Should be consistent

        # Test distribution over many users
        treatment_count = 0
        total_users = 1000

        for i in range(total_users):
            if hyde_engine._determine_ab_group(f"user_{i}") == "treatment":
                treatment_count += 1

        treatment_ratio = treatment_count / total_users
        # Should be close to configured ratio (within 5%)
        assert abs(treatment_ratio - 0.3) < 0.05

    @pytest.mark.asyncio
    async def test_ab_testing_metrics_collection(self, hyde_engine):
        """Test A/B testing metrics collection."""
        hyde_engine._initialized = True

        # Mock search methods with different performance
        async def mock_hyde_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slower HyDE
            return [{"id": "hyde", "score": 0.95}]

        async def mock_direct_search(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate faster direct
            return [{"id": "direct", "score": 0.85}]

        hyde_engine.search = AsyncMock(side_effect=mock_hyde_search)
        hyde_engine._direct_search = AsyncMock(side_effect=mock_direct_search)

        # Force treatment group
        with patch.object(hyde_engine, "_determine_ab_group", return_value="treatment"):
            results = await hyde_engine.search_with_ab_testing(
                query="test", collection="docs", user_id="user1"
            )

        assert results[0]["id"] == "hyde"
        assert hyde_engine.treatment_group_searches == 1

        # Check performance tracking
        assert hyde_engine.total_search_time > 0

    @pytest.mark.asyncio
    async def test_ab_testing_disabled(self, hyde_engine):
        """Test behavior when A/B testing is disabled."""
        hyde_engine._initialized = True
        hyde_engine.config.enable_ab_testing = False

        hyde_engine.search = AsyncMock(return_value=[{"id": "hyde", "score": 0.9}])

        results = await hyde_engine.search_with_ab_testing(
            query="test", collection="docs", user_id="user1"
        )

        # Should always use HyDE when A/B testing disabled
        assert results[0]["id"] == "hyde"
        hyde_engine.search.assert_called_once()

        # No A/B metrics should be tracked
        assert hyde_engine.control_group_searches == 0
        assert hyde_engine.treatment_group_searches == 0


class TestPerformanceMetrics:
    """Test performance metrics and monitoring."""

    def test_get_performance_stats(self, hyde_engine):
        """Test getting performance statistics."""
        # Set up some metrics
        hyde_engine.search_count = 100
        hyde_engine.total_search_time = 50.0
        hyde_engine.cache_hit_count = 30
        hyde_engine.generation_count = 70
        hyde_engine.fallback_count = 5

        hyde_engine.control_group_searches = 45
        hyde_engine.treatment_group_searches = 55

        stats = hyde_engine.get_performance_stats()

        assert stats["total_searches"] == 100
        assert stats["average_search_time"] == 0.5  # 50.0 / 100
        assert stats["cache_hit_rate"] == 0.3  # 30 / 100
        assert stats["generation_rate"] == 0.7  # 70 / 100
        assert stats["fallback_rate"] == 0.05  # 5 / 100

        assert stats["ab_testing"]["control_searches"] == 45
        assert stats["ab_testing"]["treatment_searches"] == 55
        assert stats["ab_testing"]["total_ab_searches"] == 100

    def test_reset_performance_stats(self, hyde_engine):
        """Test resetting performance statistics."""
        # Set some stats
        hyde_engine.search_count = 100
        hyde_engine.total_search_time = 50.0
        hyde_engine.cache_hit_count = 30
        hyde_engine.generation_count = 70
        hyde_engine.fallback_count = 5
        hyde_engine.control_group_searches = 45
        hyde_engine.treatment_group_searches = 55

        hyde_engine.reset_performance_stats()

        assert hyde_engine.search_count == 0
        assert hyde_engine.total_search_time == 0.0
        assert hyde_engine.cache_hit_count == 0
        assert hyde_engine.generation_count == 0
        assert hyde_engine.fallback_count == 0
        assert hyde_engine.control_group_searches == 0
        assert hyde_engine.treatment_group_searches == 0

    @pytest.mark.asyncio
    async def test_performance_tracking_during_search(self, hyde_engine):
        """Test performance tracking during actual search."""
        hyde_engine._initialized = True

        # Mock cache miss to trigger full search
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock components with delays
        async def mock_generation(*args, **kwargs):
            await asyncio.sleep(0.01)
            return ["Hypothetical doc"]

        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            side_effect=mock_generation
        )
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(return_value=[])
        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        initial_search_count = hyde_engine.search_count
        initial_generation_count = hyde_engine.generation_count

        await hyde_engine.search(query="test", collection="docs")

        # Verify metrics were updated
        assert hyde_engine.search_count == initial_search_count + 1
        assert hyde_engine.generation_count == initial_generation_count + 1
        assert hyde_engine.total_search_time > 0


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, hyde_engine):
        """Test search when engine not initialized."""
        with pytest.raises(EmbeddingServiceError, match="HyDE engine not initialized"):
            await hyde_engine.search(query="test", collection="docs")

    @pytest.mark.asyncio
    async def test_search_qdrant_service_error(self, hyde_engine):
        """Test search when Qdrant service fails."""
        hyde_engine._initialized = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock generation success
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=["Doc"]
        )
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        # Mock Qdrant failure
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(
            side_effect=QdrantServiceError("Qdrant connection failed")
        )

        # Should fall back to direct search
        hyde_engine.qdrant_service.search = AsyncMock(
            return_value=[{"id": "fallback", "score": 0.8}]
        )

        results = await hyde_engine.search(query="test", collection="docs")

        assert results[0]["id"] == "fallback"
        assert hyde_engine.fallback_count == 1

    @pytest.mark.asyncio
    async def test_batch_search_partial_failures(self, hyde_engine):
        """Test batch search with partial failures."""
        hyde_engine._initialized = True

        queries = [
            {"query": "success1", "collection": "docs"},
            {"query": "failure", "collection": "docs"},
            {"query": "success2", "collection": "docs"},
        ]

        # Mock search with one failure
        async def mock_search(*args, **kwargs):
            if "failure" in args[0]:
                raise Exception("Search failed")
            return [{"id": "success", "score": 0.9}]

        hyde_engine.search = AsyncMock(side_effect=mock_search)

        # Should handle exceptions gracefully
        results = await hyde_engine.batch_search(queries, return_exceptions=True)

        assert len(results) == 3
        assert isinstance(results[0], list)  # Success
        assert isinstance(results[1], Exception)  # Failure
        assert isinstance(results[2], list)  # Success


class TestQueryAPIIntegration:
    """Test Query API integration features."""

    @pytest.mark.asyncio
    async def test_query_api_prefetch_fusion(self, hyde_engine):
        """Test Query API prefetch and fusion."""
        hyde_engine._initialized = True
        hyde_engine.config.enable_query_api = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock hypothetical documents
        hypothetical_docs = ["Doc 1", "Doc 2", "Doc 3"]
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=hypothetical_docs
        )

        # Mock embeddings for each document
        doc_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        hyde_engine.embedding_manager.get_batch_embeddings = AsyncMock(
            return_value=doc_embeddings
        )

        # Mock Query API search
        hyde_engine.qdrant_service.multi_stage_search = AsyncMock(
            return_value=[{"id": "fusion_result", "score": 0.95}]
        )

        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        results = await hyde_engine.search_with_query_api(
            query="test query", collection="docs", fusion_algorithm="rrf"
        )

        assert results[0]["id"] == "fusion_result"

        # Verify Query API was used
        hyde_engine.qdrant_service.multi_stage_search.assert_called_once()
        call_args = hyde_engine.qdrant_service.multi_stage_search.call_args
        assert call_args[1]["fusion_algorithm"] == "rrf"

    @pytest.mark.asyncio
    async def test_query_api_fallback_to_hybrid(self, hyde_engine):
        """Test fallback to hybrid search when Query API fails."""
        hyde_engine._initialized = True
        hyde_engine.config.enable_query_api = True

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock generation and embeddings
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=["Doc"]
        )
        hyde_engine.embedding_manager.get_batch_embeddings = AsyncMock(
            return_value=[[0.1, 0.2, 0.3]]
        )

        # Mock Query API failure
        hyde_engine.qdrant_service.multi_stage_search = AsyncMock(
            side_effect=Exception("Query API failed")
        )

        # Mock hybrid search success
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(
            return_value=[{"id": "hybrid_result", "score": 0.85}]
        )

        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        results = await hyde_engine.search_with_query_api(
            query="test query", collection="docs"
        )

        assert results[0]["id"] == "hybrid_result"
        assert hyde_engine.fallback_count == 1

        # Verify hybrid search was used as fallback
        hyde_engine.qdrant_service.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_api_disabled(self, hyde_engine):
        """Test behavior when Query API is disabled."""
        hyde_engine._initialized = True
        hyde_engine.config.enable_query_api = False

        # Mock cache miss
        hyde_engine.cache.get_cached_search_results = AsyncMock(return_value=None)

        # Mock standard HyDE components
        hyde_engine.generator.generate_hypothetical_documents = AsyncMock(
            return_value=["Doc"]
        )
        hyde_engine.embedding_manager.get_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        hyde_engine.qdrant_service.hybrid_search = AsyncMock(
            return_value=[{"id": "standard_result", "score": 0.9}]
        )
        hyde_engine.cache.cache_search_results = AsyncMock(return_value=True)

        results = await hyde_engine.search_with_query_api(
            query="test query", collection="docs"
        )

        assert results[0]["id"] == "standard_result"

        # Should use standard hybrid search, not Query API
        hyde_engine.qdrant_service.hybrid_search.assert_called_once()
        hyde_engine.qdrant_service.multi_stage_search.assert_not_called()
