"""Tests for HyDE caching implementation."""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.services.errors import EmbeddingServiceError
from src.services.hyde.cache import HyDECache
from src.services.hyde.config import HyDEConfig
from src.services.hyde.generator import GenerationResult


class TestHyDECache:
    """Tests for HyDECache class."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        manager = MagicMock()
        manager.initialize = AsyncMock()
        manager.cleanup = AsyncMock()
        manager.get = AsyncMock()
        manager.set = AsyncMock(return_value=True)
        manager.delete = AsyncMock(return_value=True)
        return manager

    @pytest.fixture
    def hyde_config(self):
        """Create HyDE configuration."""
        return HyDEConfig(
            cache_ttl_seconds=3600,
            cache_hypothetical_docs=True,
            cache_prefix="test_hyde",
        )

    @pytest.fixture
    def cache(self, hyde_config, mock_cache_manager):
        """Create HyDECache instance."""
        return HyDECache(config=hyde_config, cache_manager=mock_cache_manager)

    def test_init(self, hyde_config, mock_cache_manager):
        """Test cache initialization."""
        cache = HyDECache(config=hyde_config, cache_manager=mock_cache_manager)

        assert cache.config == hyde_config
        assert cache.cache_manager == mock_cache_manager
        assert cache._initialized is False

        # Check metrics initialization
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert cache.cache_sets == 0
        assert cache.cache_errors == 0

        # Check cache key prefixes
        assert cache.embedding_prefix == "test_hyde:embedding"
        assert cache.documents_prefix == "test_hyde:documents"
        assert cache.results_prefix == "test_hyde:results"

    async def test_initialize_success(self, cache, mock_cache_manager):
        """Test successful cache initialization."""
        # Mock successful test
        mock_cache_manager.get.return_value = "test_value"

        await cache.initialize()

        assert cache._initialized is True
        mock_cache_manager.initialize.assert_called_once()
        mock_cache_manager.set.assert_called_once()
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.delete.assert_called_once()

    async def test_initialize_no_cache_manager_initialize(
        self, cache, mock_cache_manager
    ):
        """Test initialization when cache manager has no initialize method."""
        # Remove initialize method from mock
        del mock_cache_manager.initialize
        mock_cache_manager.get.return_value = "test_value"

        await cache.initialize()

        assert cache._initialized is True

    async def test_initialize_test_failure(self, cache, mock_cache_manager):
        """Test initialization failure when cache test fails."""
        mock_cache_manager.get.return_value = "wrong_value"

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await cache.initialize()

        assert "Cache test failed" in str(exc_info.value)
        assert cache._initialized is False

    async def test_initialize_already_initialized(self, cache, mock_cache_manager):
        """Test initialization when already initialized."""
        cache._initialized = True

        await cache.initialize()

        # Should not call cache manager methods again
        mock_cache_manager.initialize.assert_not_called()
        mock_cache_manager.set.assert_not_called()

    async def test_initialize_error(self, cache, mock_cache_manager):
        """Test initialization error handling."""
        mock_cache_manager.initialize.side_effect = Exception("Cache error")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await cache.initialize()

        assert "Failed to initialize HyDE cache" in str(exc_info.value)
        assert cache._initialized is False

    async def test_cleanup(self, cache, mock_cache_manager):
        """Test cache cleanup."""
        cache._initialized = True

        await cache.cleanup()

        assert cache._initialized is False
        mock_cache_manager.cleanup.assert_called_once()

    async def test_cleanup_no_cleanup_method(self, cache, mock_cache_manager):
        """Test cleanup when cache manager has no cleanup method."""
        cache._initialized = True
        del mock_cache_manager.cleanup

        await cache.cleanup()

        assert cache._initialized is False

    async def test_get_hyde_embedding_cache_hit_dict_format(
        self, cache, mock_cache_manager
    ):
        """Test getting HyDE embedding from cache - dict format."""
        cache._initialized = True

        # Mock cached data in dict format
        cached_data = {
            "embedding": [0.1, 0.2, 0.3],
            "query": "test query",
            "timestamp": time.time(),
        }
        mock_cache_manager.get.return_value = cached_data

        result = await cache.get_hyde_embedding("test query", "python")

        assert result == [0.1, 0.2, 0.3]
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0

    async def test_get_hyde_embedding_cache_hit_binary_format(
        self, cache, mock_cache_manager
    ):
        """Test getting HyDE embedding from cache - binary format."""
        cache._initialized = True

        # Mock cached data in binary format
        embedding_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cached_data = {
            "embedding": embedding_array.tobytes(),
            "embedding_shape": embedding_array.shape,
            "embedding_dtype": str(embedding_array.dtype),
        }
        mock_cache_manager.get.return_value = cached_data

        result = await cache.get_hyde_embedding("test query", "python")

        assert len(result) == 3
        assert abs(result[0] - 0.1) < 0.001
        assert abs(result[1] - 0.2) < 0.001
        assert abs(result[2] - 0.3) < 0.001
        assert cache.cache_hits == 1

    async def test_get_hyde_embedding_cache_hit_list_format(
        self, cache, mock_cache_manager
    ):
        """Test getting HyDE embedding from cache - direct list format."""
        cache._initialized = True

        # Mock cached data as direct list
        mock_cache_manager.get.return_value = [0.1, 0.2, 0.3]

        result = await cache.get_hyde_embedding("test query", "python")

        assert result == [0.1, 0.2, 0.3]
        assert cache.cache_hits == 1

    async def test_get_hyde_embedding_cache_miss(self, cache, mock_cache_manager):
        """Test getting HyDE embedding - cache miss."""
        cache._initialized = True
        mock_cache_manager.get.return_value = None

        result = await cache.get_hyde_embedding("test query", "python")

        assert result is None
        assert cache.cache_hits == 0
        assert cache.cache_misses == 1

    async def test_get_hyde_embedding_invalid_format(self, cache, mock_cache_manager):
        """Test getting HyDE embedding with invalid cache format."""
        cache._initialized = True
        mock_cache_manager.get.return_value = "invalid_format"

        result = await cache.get_hyde_embedding("test query", "python")

        assert result is None
        assert cache.cache_misses == 1

    async def test_get_hyde_embedding_error(self, cache, mock_cache_manager):
        """Test getting HyDE embedding with cache error."""
        cache._initialized = True
        mock_cache_manager.get.side_effect = Exception("Cache error")

        result = await cache.get_hyde_embedding("test query", "python")

        assert result is None
        assert cache.cache_errors == 1

    async def test_get_hyde_embedding_not_initialized(self, cache):
        """Test getting HyDE embedding when not initialized."""
        from src.services.errors import APIError

        with pytest.raises(APIError):
            await cache.get_hyde_embedding("test query")

    async def test_set_hyde_embedding_success(self, cache, mock_cache_manager):
        """Test setting HyDE embedding successfully."""
        cache._initialized = True

        embedding = [0.1, 0.2, 0.3]
        hypothetical_docs = ["doc1", "doc2"]
        metadata = {"generation_time": 1.5, "tokens_used": 100}

        result = await cache.set_hyde_embedding(
            query="test query",
            embedding=embedding,
            hypothetical_docs=hypothetical_docs,
            generation_metadata=metadata,
            domain="python",
        )

        assert result is True
        assert cache.cache_sets == 1
        mock_cache_manager.set.assert_called_once()

        # Check the cached data structure
        call_args = mock_cache_manager.set.call_args
        cache_key = call_args[0][0]
        cache_data = call_args[0][1]
        ttl = call_args[1]["ttl"]

        assert "test_hyde:embedding:" in cache_key
        assert cache_data["query"] == "test query"
        assert cache_data["domain"] == "python"
        assert cache_data["hypothetical_docs"] == hypothetical_docs
        assert cache_data["metadata"] == metadata
        assert ttl == cache.config.cache_ttl_seconds

    async def test_set_hyde_embedding_no_hypothetical_docs(
        self, cache, mock_cache_manager
    ):
        """Test setting HyDE embedding without caching hypothetical docs."""
        cache._initialized = True
        cache.config.cache_hypothetical_docs = False

        embedding = [0.1, 0.2, 0.3]
        hypothetical_docs = ["doc1", "doc2"]

        result = await cache.set_hyde_embedding(
            query="test query",
            embedding=embedding,
            hypothetical_docs=hypothetical_docs,
        )

        assert result is True

        # Check that hypothetical docs are not cached
        call_args = mock_cache_manager.set.call_args
        cache_data = call_args[0][1]
        assert cache_data["hypothetical_docs"] == []

    async def test_set_hyde_embedding_error(self, cache, mock_cache_manager):
        """Test setting HyDE embedding with cache error."""
        cache._initialized = True
        mock_cache_manager.set.side_effect = Exception("Cache error")

        result = await cache.set_hyde_embedding(
            query="test query",
            embedding=[0.1, 0.2, 0.3],
            hypothetical_docs=["doc1"],
        )

        assert result is False
        assert cache.cache_errors == 1

    async def test_get_hypothetical_documents_success(self, cache, mock_cache_manager):
        """Test getting hypothetical documents successfully."""
        cache._initialized = True

        cached_docs = ["doc1", "doc2", "doc3"]
        mock_cache_manager.get.return_value = cached_docs

        result = await cache.get_hypothetical_documents("test query", "python")

        assert result == cached_docs
        assert cache.cache_hits == 1

    async def test_get_hypothetical_documents_disabled(self, cache, mock_cache_manager):
        """Test getting hypothetical documents when caching disabled."""
        cache.config.cache_hypothetical_docs = False

        result = await cache.get_hypothetical_documents("test query", "python")

        assert result is None
        mock_cache_manager.get.assert_not_called()

    async def test_get_hypothetical_documents_miss(self, cache, mock_cache_manager):
        """Test getting hypothetical documents - cache miss."""
        cache._initialized = True
        mock_cache_manager.get.return_value = None

        result = await cache.get_hypothetical_documents("test query", "python")

        assert result is None
        assert cache.cache_misses == 1

    async def test_get_hypothetical_documents_error(self, cache, mock_cache_manager):
        """Test getting hypothetical documents with error."""
        cache._initialized = True
        mock_cache_manager.get.side_effect = Exception("Cache error")

        result = await cache.get_hypothetical_documents("test query", "python")

        assert result is None
        assert cache.cache_errors == 1

    async def test_set_hypothetical_documents_success(self, cache, mock_cache_manager):
        """Test setting hypothetical documents successfully."""
        cache._initialized = True

        documents = ["doc1", "doc2"]
        generation_result = GenerationResult(
            documents=documents,
            generation_time=1.5,
            tokens_used=100,
            cost_estimate=0.01,
            diversity_score=0.8,
        )

        result = await cache.set_hypothetical_documents(
            query="test query",
            documents=documents,
            generation_result=generation_result,
            domain="python",
        )

        assert result is True
        assert cache.cache_sets == 1

        # Check cached data
        call_args = mock_cache_manager.set.call_args
        cache_data = call_args[0][1]
        assert cache_data["documents"] == documents
        assert cache_data["generation_time"] == 1.5
        assert cache_data["tokens_used"] == 100
        assert cache_data["diversity_score"] == 0.8

    async def test_set_hypothetical_documents_disabled(self, cache, mock_cache_manager):
        """Test setting hypothetical documents when caching disabled."""
        cache.config.cache_hypothetical_docs = False

        documents = ["doc1", "doc2"]
        generation_result = GenerationResult(
            documents=documents,
            generation_time=1.5,
            tokens_used=100,
            cost_estimate=0.01,
        )

        result = await cache.set_hypothetical_documents(
            query="test query",
            documents=documents,
            generation_result=generation_result,
        )

        assert result is True  # Returns True even when disabled
        mock_cache_manager.set.assert_not_called()

    async def test_set_hypothetical_documents_error(self, cache, mock_cache_manager):
        """Test setting hypothetical documents with error."""
        cache._initialized = True
        mock_cache_manager.set.side_effect = Exception("Cache error")

        documents = ["doc1"]
        generation_result = GenerationResult(
            documents=documents,
            generation_time=1.0,
            tokens_used=50,
            cost_estimate=0.005,
        )

        result = await cache.set_hypothetical_documents(
            query="test query",
            documents=documents,
            generation_result=generation_result,
        )

        assert result is False
        assert cache.cache_errors == 1

    async def test_get_search_results_success(self, cache, mock_cache_manager):
        """Test getting search results successfully."""
        cache._initialized = True

        cached_results = [{"id": "doc1", "score": 0.9}]
        mock_cache_manager.get.return_value = cached_results

        search_params = {"limit": 10, "filters": {}, "hyde_enabled": True}
        result = await cache.get_search_results(
            "test query", "documents", search_params
        )

        assert result == cached_results
        assert cache.cache_hits == 1

    async def test_get_search_results_miss(self, cache, mock_cache_manager):
        """Test getting search results - cache miss."""
        cache._initialized = True
        mock_cache_manager.get.return_value = None

        search_params = {"limit": 10, "filters": {}}
        result = await cache.get_search_results(
            "test query", "documents", search_params
        )

        assert result is None
        assert cache.cache_misses == 1

    async def test_get_search_results_error(self, cache, mock_cache_manager):
        """Test getting search results with error."""
        cache._initialized = True
        mock_cache_manager.get.side_effect = Exception("Cache error")

        search_params = {"limit": 10}
        result = await cache.get_search_results(
            "test query", "documents", search_params
        )

        assert result is None
        assert cache.cache_errors == 1

    async def test_set_search_results_success(self, cache, mock_cache_manager):
        """Test setting search results successfully."""
        cache._initialized = True

        results = [{"id": "doc1", "score": 0.9}]
        search_params = {"limit": 10, "filters": {}}
        search_metadata = {"result_count": 1}

        result = await cache.set_search_results(
            query="test query",
            collection_name="documents",
            search_params=search_params,
            results=results,
            search_metadata=search_metadata,
        )

        assert result is True
        assert cache.cache_sets == 1

        # Check TTL is shorter for search results
        call_args = mock_cache_manager.set.call_args
        ttl = call_args[1]["ttl"]
        assert ttl <= cache.config.cache_ttl_seconds // 2

    async def test_set_search_results_error(self, cache, mock_cache_manager):
        """Test setting search results with error."""
        cache._initialized = True
        mock_cache_manager.set.side_effect = Exception("Cache error")

        result = await cache.set_search_results(
            query="test query",
            collection_name="documents",
            search_params={},
            results=[],
        )

        assert result is False
        assert cache.cache_errors == 1

    async def test_warm_cache_success(self, cache, mock_cache_manager):
        """Test cache warming with some cached and some missing queries."""
        cache._initialized = True

        # First query has cached embedding, second doesn't
        mock_cache_manager.get.side_effect = [[0.1, 0.2, 0.3], None]

        queries = ["cached query", "uncached query"]
        results = await cache.warm_cache(queries, "python")

        assert results["cached query"] is True
        assert results["uncached query"] is False
        assert len(results) == 2

    async def test_warm_cache_error(self, cache, mock_cache_manager):
        """Test cache warming with errors."""
        cache._initialized = True
        mock_cache_manager.get.side_effect = Exception("Cache error")

        queries = ["query1", "query2"]
        results = await cache.warm_cache(queries)

        assert results["query1"] is False
        assert results["query2"] is False

    async def test_invalidate_query_success(self, cache, mock_cache_manager):
        """Test query invalidation successfully."""
        cache._initialized = True

        result = await cache.invalidate_query("test query", "python")

        assert result is True
        assert mock_cache_manager.delete.call_count == 2  # embedding + documents

    async def test_invalidate_query_partial_success(self, cache, mock_cache_manager):
        """Test query invalidation with partial success."""
        cache._initialized = True
        mock_cache_manager.delete.side_effect = [
            True,
            False,
        ]  # First succeeds, second fails

        result = await cache.invalidate_query("test query", "python")

        assert result is True  # At least one succeeded

    async def test_invalidate_query_error(self, cache, mock_cache_manager):
        """Test query invalidation with error."""
        cache._initialized = True
        mock_cache_manager.delete.side_effect = Exception("Cache error")

        result = await cache.invalidate_query("test query")

        assert result is False

    def test_get_embedding_cache_key(self, cache):
        """Test embedding cache key generation."""
        key1 = cache._get_embedding_cache_key("test query", "python")
        key2 = cache._get_embedding_cache_key("test query", "python")
        key3 = cache._get_embedding_cache_key("test query", "javascript")
        key4 = cache._get_embedding_cache_key("different query", "python")

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        assert key1 != key3
        assert key1 != key4

        # Keys should have correct prefix
        assert key1.startswith("test_hyde:embedding:")

    def test_get_documents_cache_key(self, cache):
        """Test documents cache key generation."""
        key1 = cache._get_documents_cache_key("test query", "python")
        key2 = cache._get_documents_cache_key("test query", None)

        assert key1 != key2
        assert key1.startswith("test_hyde:documents:")

    def test_get_results_cache_key(self, cache):
        """Test results cache key generation."""
        params1 = {"limit": 10, "filters": {}}
        params2 = {"limit": 20, "filters": {}}

        key1 = cache._get_results_cache_key("test query", "documents", params1)
        key2 = cache._get_results_cache_key("test query", "documents", params2)

        assert key1 != key2
        assert key1.startswith("test_hyde:results:")

        # Same params should produce same key
        key3 = cache._get_results_cache_key("test query", "documents", params1)
        assert key1 == key3

    def test_get_results_cache_key_deterministic(self, cache):
        """Test that results cache key is deterministic for same parameters."""
        # Order of parameters shouldn't matter due to sort_keys=True
        params1 = {"limit": 10, "filters": {"type": "doc"}, "accuracy": "high"}
        params2 = {"accuracy": "high", "filters": {"type": "doc"}, "limit": 10}

        key1 = cache._get_results_cache_key("query", "collection", params1)
        key2 = cache._get_results_cache_key("query", "collection", params2)

        assert key1 == key2

    def test_get_cache_metrics(self, cache):
        """Test cache metrics calculation."""
        # Set some test values
        cache.cache_hits = 8
        cache.cache_misses = 2
        cache.cache_sets = 5
        cache.cache_errors = 1

        metrics = cache.get_cache_metrics()

        assert metrics["cache_hits"] == 8
        assert metrics["cache_misses"] == 2
        assert metrics["cache_sets"] == 5
        assert metrics["cache_errors"] == 1
        assert metrics["total_requests"] == 10
        assert metrics["hit_rate"] == 0.8
        assert metrics["error_rate"] == 0.1

    def test_get_cache_metrics_zero_requests(self, cache):
        """Test cache metrics when no requests have been made."""
        metrics = cache.get_cache_metrics()

        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0
        assert metrics["cache_sets"] == 0
        assert metrics["cache_errors"] == 0
        assert metrics["total_requests"] == 0
        assert metrics["hit_rate"] == 0.0
        assert metrics["error_rate"] == 0.0

    def test_reset_metrics(self, cache):
        """Test resetting cache metrics."""
        # Set some values
        cache.cache_hits = 5
        cache.cache_misses = 3
        cache.cache_sets = 2
        cache.cache_errors = 1

        cache.reset_metrics()

        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert cache.cache_sets == 0
        assert cache.cache_errors == 0

    async def test_embedding_cache_key_with_none_domain(
        self, cache, mock_cache_manager
    ):
        """Test embedding cache key generation with None domain."""
        cache._initialized = True
        mock_cache_manager.get.return_value = None

        await cache.get_hyde_embedding("test query", None)

        # Should use "general" as default domain
        call_args = mock_cache_manager.get.call_args
        cache_key = call_args[0][0]
        assert "test_hyde:embedding:" in cache_key

    async def test_binary_embedding_storage_and_retrieval(
        self, cache, mock_cache_manager
    ):
        """Test that embeddings are stored in binary format and retrieved correctly."""
        cache._initialized = True

        # Test storage
        embedding = [0.1, 0.2, 0.3, 0.4]
        await cache.set_hyde_embedding(
            query="test query",
            embedding=embedding,
            hypothetical_docs=["doc1"],
        )

        # Check that set was called with binary data
        call_args = mock_cache_manager.set.call_args
        cache_data = call_args[0][1]

        assert isinstance(cache_data["embedding"], bytes)
        assert "embedding_shape" in cache_data
        assert "embedding_dtype" in cache_data

        # Test retrieval
        mock_cache_manager.get.return_value = cache_data
        retrieved = await cache.get_hyde_embedding("test query")

        assert len(retrieved) == 4
        assert abs(retrieved[0] - 0.1) < 0.001
        assert abs(retrieved[1] - 0.2) < 0.001
        assert abs(retrieved[2] - 0.3) < 0.001
        assert abs(retrieved[3] - 0.4) < 0.001
