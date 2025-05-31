"""Comprehensive tests for HyDE cache service."""

import asyncio
import json
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.errors import EmbeddingServiceError
from src.services.hyde.cache import HyDECache
from src.services.hyde.config import HyDEConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = MagicMock()
    config.hyde = HyDEConfig(
        cache_hypothetical_docs=True,
        cache_prefix="hyde_test",
        cache_ttl_embeddings=3600,
        cache_ttl_documents=1800,
        cache_ttl_results=900,
        enable_compression=True,
        compression_threshold=1024,
    )
    return config


@pytest.fixture
def mock_cache_manager():
    """Create mock cache manager."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.get = AsyncMock(return_value=None)
    manager.set = AsyncMock(return_value=True)
    manager.delete = AsyncMock(return_value=True)
    manager.exists = AsyncMock(return_value=False)
    manager.scan_keys = AsyncMock(return_value=[])
    manager.get_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    return manager


@pytest.fixture
def hyde_cache(mock_config, mock_cache_manager):
    """Create HyDECache instance for testing."""
    return HyDECache(config=mock_config.hyde, cache_manager=mock_cache_manager)


class TestHyDECacheInitialization:
    """Test HyDE cache initialization."""

    def test_cache_initialization(self, hyde_cache, mock_config, mock_cache_manager):
        """Test basic cache initialization."""
        assert hyde_cache.config is not None
        assert hyde_cache.cache_manager == mock_cache_manager
        assert hyde_cache._initialized is False
        assert hyde_cache.cache_hits == 0
        assert hyde_cache.cache_misses == 0
        assert hyde_cache.cache_sets == 0
        assert hyde_cache.cache_errors == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, hyde_cache, mock_cache_manager):
        """Test successful cache initialization."""
        await hyde_cache.initialize()

        assert hyde_cache._initialized is True
        mock_cache_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, hyde_cache, mock_cache_manager):
        """Test cache initialization failure."""
        mock_cache_manager.initialize.side_effect = Exception("Cache connection failed")

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize HyDE cache"
        ):
            await hyde_cache.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, hyde_cache, mock_cache_manager):
        """Test that initialization is idempotent."""
        await hyde_cache.initialize()
        await hyde_cache.initialize()  # Second call

        # Should only initialize once
        mock_cache_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, hyde_cache, mock_cache_manager):
        """Test cache cleanup."""
        hyde_cache._initialized = True

        await hyde_cache.cleanup()

        mock_cache_manager.cleanup.assert_called_once()
        assert hyde_cache._initialized is False


class TestEmbeddingCaching:
    """Test embedding caching operations."""

    @pytest.mark.asyncio
    async def test_cache_embedding_basic(self, hyde_cache, mock_cache_manager):
        """Test basic embedding caching."""
        hyde_cache._initialized = True

        query = "machine learning algorithms"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"generation_time": 1.5, "model": "gpt-4"}

        success = await hyde_cache.cache_embedding(
            query=query,
            embedding=embedding,
            metadata=metadata,
        )

        assert success is True
        mock_cache_manager.set.assert_called_once()

        # Verify cache key and data structure
        call_args = mock_cache_manager.set.call_args
        cache_key = call_args[0][0]
        cache_data = call_args[0][1]

        assert "hyde_test:embedding:" in cache_key
        assert isinstance(cache_data, dict)
        assert "embedding" in cache_data
        assert "metadata" in cache_data
        assert cache_data["metadata"]["generation_time"] == 1.5

    @pytest.mark.asyncio
    async def test_get_cached_embedding_hit(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached embedding - cache hit."""
        hyde_cache._initialized = True

        query = "database optimization"
        cached_embedding = [0.1, 0.2, 0.3]
        cached_data = {
            "embedding": cached_embedding,
            "metadata": {"generation_time": 2.0, "model": "gpt-4"},
            "timestamp": int(time.time()),
        }

        mock_cache_manager.get.return_value = cached_data

        result = await hyde_cache.get_cached_embedding(query)

        assert result is not None
        assert result["embedding"] == cached_embedding
        assert result["metadata"]["generation_time"] == 2.0
        assert hyde_cache.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_cached_embedding_miss(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached embedding - cache miss."""
        hyde_cache._initialized = True

        mock_cache_manager.get.return_value = None

        result = await hyde_cache.get_cached_embedding("missing_query")

        assert result is None
        assert hyde_cache.cache_misses == 1

    @pytest.mark.asyncio
    async def test_cache_embedding_with_binary_serialization(
        self, hyde_cache, mock_cache_manager
    ):
        """Test embedding caching with binary serialization."""
        hyde_cache._initialized = True

        query = "large embedding test"
        # Large embedding that should trigger binary serialization
        large_embedding = [float(i) for i in range(1536)]

        with patch.object(hyde_cache, "_should_compress", return_value=True):
            success = await hyde_cache.cache_embedding(
                query=query,
                embedding=large_embedding,
                metadata={"size": len(large_embedding)},
            )

        assert success is True
        call_args = mock_cache_manager.set.call_args
        cache_data = call_args[0][1]

        # Should contain binary embedding data
        assert "embedding_binary" in cache_data or "embedding" in cache_data

    @pytest.mark.asyncio
    async def test_cache_embedding_with_ttl(self, hyde_cache, mock_cache_manager):
        """Test embedding caching with TTL."""
        hyde_cache._initialized = True

        query = "ttl test"
        embedding = [0.1, 0.2, 0.3]
        custom_ttl = 7200

        success = await hyde_cache.cache_embedding(
            query=query,
            embedding=embedding,
            ttl=custom_ttl,
        )

        assert success is True

        # Verify TTL was passed
        call_args = mock_cache_manager.set.call_args
        if len(call_args[1]) > 0 and "ttl" in call_args[1]:
            assert call_args[1]["ttl"] == custom_ttl


class TestDocumentCaching:
    """Test hypothetical document caching."""

    @pytest.mark.asyncio
    async def test_cache_hypothetical_documents(self, hyde_cache, mock_cache_manager):
        """Test caching hypothetical documents."""
        hyde_cache._initialized = True

        query = "API documentation"
        hypothetical_docs = [
            "This is a comprehensive API guide...",
            "Here's how to use our REST endpoints...",
            "Authentication is handled via API keys...",
        ]
        metadata = {
            "generation_model": "gpt-4",
            "generation_time": 2.3,
            "prompt_version": "v1.2",
        }

        success = await hyde_cache.cache_hypothetical_documents(
            query=query,
            documents=hypothetical_docs,
            metadata=metadata,
        )

        assert success is True
        mock_cache_manager.set.assert_called_once()

        call_args = mock_cache_manager.set.call_args
        cache_key = call_args[0][0]
        cache_data = call_args[0][1]

        assert "hyde_test:documents:" in cache_key
        assert cache_data["documents"] == hypothetical_docs
        assert cache_data["metadata"]["generation_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_cached_documents_hit(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached documents - cache hit."""
        hyde_cache._initialized = True

        query = "tutorials"
        cached_docs = ["Tutorial 1", "Tutorial 2", "Tutorial 3"]
        cached_data = {
            "documents": cached_docs,
            "metadata": {"generation_time": 1.8},
            "timestamp": int(time.time()),
        }

        mock_cache_manager.get.return_value = cached_data

        result = await hyde_cache.get_cached_documents(query)

        assert result is not None
        assert result["documents"] == cached_docs
        assert hyde_cache.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_cached_documents_miss(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached documents - cache miss."""
        hyde_cache._initialized = True

        mock_cache_manager.get.return_value = None

        result = await hyde_cache.get_cached_documents("missing_query")

        assert result is None
        assert hyde_cache.cache_misses == 1


class TestSearchResultsCaching:
    """Test search results caching."""

    @pytest.mark.asyncio
    async def test_cache_search_results(self, hyde_cache, mock_cache_manager):
        """Test caching search results."""
        hyde_cache._initialized = True

        query = "vector database tutorial"
        collection = "documentation"
        search_params = {"limit": 10, "strategy": "hyde"}
        results = [
            {"id": "doc1", "content": "Vector databases...", "score": 0.95},
            {"id": "doc2", "content": "Tutorial on...", "score": 0.87},
        ]
        metadata = {
            "search_time": 0.15,
            "total_found": 2,
            "search_strategy": "hyde",
        }

        success = await hyde_cache.cache_search_results(
            query=query,
            collection=collection,
            search_params=search_params,
            results=results,
            metadata=metadata,
        )

        assert success is True
        mock_cache_manager.set.assert_called_once()

        call_args = mock_cache_manager.set.call_args
        cache_key = call_args[0][0]
        cache_data = call_args[0][1]

        assert "hyde_test:results:" in cache_key
        assert cache_data["results"] == results
        assert cache_data["metadata"]["search_time"] == 0.15

    @pytest.mark.asyncio
    async def test_get_cached_search_results_hit(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached search results - cache hit."""
        hyde_cache._initialized = True

        query = "cached search"
        collection = "docs"
        search_params = {"limit": 5}

        cached_results = [{"id": "cached1", "score": 0.9}]
        cached_data = {
            "results": cached_results,
            "metadata": {"search_time": 0.12},
            "timestamp": int(time.time()),
        }

        mock_cache_manager.get.return_value = cached_data

        result = await hyde_cache.get_cached_search_results(
            query=query,
            collection=collection,
            search_params=search_params,
        )

        assert result is not None
        assert result["results"] == cached_results
        assert hyde_cache.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_cached_search_results_miss(self, hyde_cache, mock_cache_manager):
        """Test retrieving cached search results - cache miss."""
        hyde_cache._initialized = True

        mock_cache_manager.get.return_value = None

        result = await hyde_cache.get_cached_search_results(
            query="missing",
            collection="docs",
            search_params={},
        )

        assert result is None
        assert hyde_cache.cache_misses == 1


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_embedding_cache_key_generation(self, hyde_cache):
        """Test embedding cache key generation."""
        query = "test query with spaces and symbols!@#"
        domain = "api_docs"

        cache_key = hyde_cache._generate_embedding_key(query, domain)

        assert "hyde_test:embedding:" in cache_key
        assert cache_key.count(":") >= 2  # At least prefix and query parts
        # Should normalize the query
        assert " " not in cache_key.split(":")[-1]

    def test_document_cache_key_generation(self, hyde_cache):
        """Test document cache key generation."""
        query = "hypothetical documents test"
        generation_params = {"model": "gpt-4", "temperature": 0.7}

        cache_key = hyde_cache._generate_document_key(query, generation_params)

        assert "hyde_test:documents:" in cache_key
        # Should include normalized query and params hash
        assert len(cache_key.split(":")) >= 3

    def test_search_results_cache_key_generation(self, hyde_cache):
        """Test search results cache key generation."""
        query = "search test"
        collection = "documentation"
        search_params = {"limit": 10, "strategy": "hyde", "filters": {"type": "api"}}

        cache_key = hyde_cache._generate_search_key(query, collection, search_params)

        assert "hyde_test:results:" in cache_key
        assert "documentation" in cache_key
        # Should include params hash for uniqueness
        assert len(cache_key) > len("hyde_test:results:search_test:documentation")

    def test_cache_key_consistency(self, hyde_cache):
        """Test cache key generation consistency."""
        query = "consistency test"
        params = {"param1": "value1", "param2": "value2"}

        # Generate same key multiple times
        key1 = hyde_cache._generate_document_key(query, params)
        key2 = hyde_cache._generate_document_key(query, params)
        key3 = hyde_cache._generate_document_key(query, params)

        assert key1 == key2 == key3

    def test_cache_key_uniqueness(self, hyde_cache):
        """Test cache key uniqueness for different inputs."""
        base_query = "test query"

        # Different domains should produce different keys
        key1 = hyde_cache._generate_embedding_key(base_query, "domain1")
        key2 = hyde_cache._generate_embedding_key(base_query, "domain2")
        assert key1 != key2

        # Different params should produce different keys
        key3 = hyde_cache._generate_document_key(base_query, {"model": "gpt-3.5"})
        key4 = hyde_cache._generate_document_key(base_query, {"model": "gpt-4"})
        assert key3 != key4


class TestCacheInvalidation:
    """Test cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_invalidate_query_cache(self, hyde_cache, mock_cache_manager):
        """Test invalidating all cache entries for a query."""
        hyde_cache._initialized = True

        query = "invalidation test"
        pattern_keys = [
            "hyde_test:embedding:invalidation_test",
            "hyde_test:documents:invalidation_test:abc123",
            "hyde_test:results:invalidation_test:docs:def456",
        ]

        mock_cache_manager.scan_keys.return_value = pattern_keys
        mock_cache_manager.delete.return_value = True

        deleted_count = await hyde_cache.invalidate_query_cache(query)

        assert deleted_count == len(pattern_keys)
        mock_cache_manager.scan_keys.assert_called()
        assert mock_cache_manager.delete.call_count == len(pattern_keys)

    @pytest.mark.asyncio
    async def test_invalidate_by_domain(self, hyde_cache, mock_cache_manager):
        """Test invalidating cache entries by domain."""
        hyde_cache._initialized = True

        domain = "api_docs"
        domain_keys = [
            "hyde_test:embedding:query1:api_docs",
            "hyde_test:embedding:query2:api_docs",
        ]

        mock_cache_manager.scan_keys.return_value = domain_keys

        deleted_count = await hyde_cache.invalidate_by_domain(domain)

        assert deleted_count == len(domain_keys)
        # Should scan for keys with domain pattern
        scan_call_args = mock_cache_manager.scan_keys.call_args[0][0]
        assert "api_docs" in scan_call_args

    @pytest.mark.asyncio
    async def test_clear_all_hyde_cache(self, hyde_cache, mock_cache_manager):
        """Test clearing all HyDE cache entries."""
        hyde_cache._initialized = True

        all_hyde_keys = [
            "hyde_test:embedding:query1",
            "hyde_test:documents:query2:hash1",
            "hyde_test:results:query3:collection:hash2",
            "other:cache:entry",  # Should not be deleted
        ]

        mock_cache_manager.scan_keys.return_value = all_hyde_keys

        deleted_count = await hyde_cache.clear_all()

        # Should only delete HyDE keys
        assert deleted_count == 3  # All except 'other:cache:entry'
        scan_call_args = mock_cache_manager.scan_keys.call_args[0][0]
        assert "hyde_test:" in scan_call_args


class TestCacheStatistics:
    """Test cache statistics and monitoring."""

    def test_get_cache_stats(self, hyde_cache, mock_cache_manager):
        """Test getting cache statistics."""
        # Set up cache metrics
        hyde_cache.cache_hits = 150
        hyde_cache.cache_misses = 50
        hyde_cache.cache_sets = 200
        hyde_cache.cache_errors = 5

        mock_cache_manager.get_stats.return_value = {
            "hits": 500,
            "misses": 100,
            "memory_usage": 1024000,
        }

        stats = hyde_cache.get_cache_stats()

        assert stats["hyde_cache_hits"] == 150
        assert stats["hyde_cache_misses"] == 50
        assert stats["hyde_hit_rate"] == 0.75  # 150 / (150 + 50)
        assert stats["hyde_cache_sets"] == 200
        assert stats["hyde_cache_errors"] == 5
        assert "underlying_cache_stats" in stats

    def test_reset_cache_stats(self, hyde_cache):
        """Test resetting cache statistics."""
        # Set some stats
        hyde_cache.cache_hits = 100
        hyde_cache.cache_misses = 50
        hyde_cache.cache_sets = 120
        hyde_cache.cache_errors = 3

        hyde_cache.reset_stats()

        assert hyde_cache.cache_hits == 0
        assert hyde_cache.cache_misses == 0
        assert hyde_cache.cache_sets == 0
        assert hyde_cache.cache_errors == 0

    @pytest.mark.asyncio
    async def test_cache_performance_metrics(self, hyde_cache, mock_cache_manager):
        """Test cache performance tracking."""
        hyde_cache._initialized = True

        # Simulate cache operations
        mock_cache_manager.get.return_value = None  # Miss
        await hyde_cache.get_cached_embedding("query1")

        mock_cache_manager.get.return_value = {"embedding": [0.1, 0.2]}  # Hit
        await hyde_cache.get_cached_embedding("query2")

        await hyde_cache.cache_embedding("query3", [0.3, 0.4])

        # Check metrics
        assert hyde_cache.cache_hits == 1
        assert hyde_cache.cache_misses == 1
        assert hyde_cache.cache_sets == 1


class TestErrorHandling:
    """Test error handling in cache operations."""

    @pytest.mark.asyncio
    async def test_cache_operation_error_handling(self, hyde_cache, mock_cache_manager):
        """Test error handling for cache operations."""
        hyde_cache._initialized = True

        # Mock cache error
        mock_cache_manager.get.side_effect = Exception("Cache connection error")

        # Should handle error gracefully and return None
        result = await hyde_cache.get_cached_embedding("test_query")

        assert result is None
        assert hyde_cache.cache_errors == 1

    @pytest.mark.asyncio
    async def test_cache_set_error_handling(self, hyde_cache, mock_cache_manager):
        """Test error handling for cache set operations."""
        hyde_cache._initialized = True

        mock_cache_manager.set.side_effect = Exception("Cache write error")

        # Should handle error gracefully and return False
        success = await hyde_cache.cache_embedding(
            query="test",
            embedding=[0.1, 0.2, 0.3],
        )

        assert success is False
        assert hyde_cache.cache_errors == 1

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, hyde_cache):
        """Test operations when cache not initialized."""
        # Don't initialize cache

        with pytest.raises(EmbeddingServiceError, match="HyDE cache not initialized"):
            await hyde_cache.get_cached_embedding("test")

        with pytest.raises(EmbeddingServiceError, match="HyDE cache not initialized"):
            await hyde_cache.cache_embedding("test", [0.1, 0.2])


class TestDataSerialization:
    """Test data serialization and compression."""

    def test_embedding_serialization(self, hyde_cache):
        """Test embedding serialization to binary format."""
        original_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Test serialization
        binary_data = hyde_cache._serialize_embedding(original_embedding)
        assert isinstance(binary_data, bytes)

        # Test deserialization
        restored_embedding = hyde_cache._deserialize_embedding(binary_data)
        assert isinstance(restored_embedding, list)
        assert len(restored_embedding) == len(original_embedding)

        # Verify values are approximately equal (accounting for float precision)
        for orig, restored in zip(original_embedding, restored_embedding, strict=False):
            assert abs(orig - restored) < 1e-6

    def test_should_compress_logic(self, hyde_cache):
        """Test compression decision logic."""
        # Small data should not be compressed
        small_data = {"small": "data"}
        assert hyde_cache._should_compress(small_data) is False

        # Large data should be compressed
        large_data = {"large": "x" * 2000}
        assert hyde_cache._should_compress(large_data) is True

    def test_data_compression(self, hyde_cache):
        """Test data compression and decompression."""
        # Large data that should be compressed
        large_data = {
            "embedding": [float(i) for i in range(1536)],
            "metadata": {"description": "x" * 1000},
        }

        # Test compression
        compressed = hyde_cache._compress_data(large_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(json.dumps(large_data))

        # Test decompression
        decompressed = hyde_cache._decompress_data(compressed)
        assert decompressed == large_data


class TestConcurrentOperations:
    """Test concurrent cache operations."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, hyde_cache, mock_cache_manager):
        """Test concurrent cache operations for thread safety."""
        hyde_cache._initialized = True

        queries = [f"query_{i}" for i in range(10)]

        # Create concurrent set operations
        tasks = []
        for i, query in enumerate(queries):
            task = hyde_cache.cache_embedding(
                query=query,
                embedding=[float(i), float(i + 1)],
                metadata={"index": i},
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        assert all(
            result is True or isinstance(result, Exception) for result in results
        )

        # Should have made multiple calls
        assert mock_cache_manager.set.call_count >= len(queries)

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self, hyde_cache, mock_cache_manager):
        """Test concurrent get operations."""
        hyde_cache._initialized = True

        queries = [f"query_{i}" for i in range(5)]

        # Mock different cache responses
        def mock_get_side_effect(key):
            if "query_0" in key:
                return {"embedding": [0.1, 0.2]}
            elif "query_1" in key:
                return {"embedding": [0.3, 0.4]}
            else:
                return None

        mock_cache_manager.get.side_effect = mock_get_side_effect

        # Create concurrent get operations
        tasks = [hyde_cache.get_cached_embedding(query) for query in queries]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify results
        assert results[0] is not None  # query_0 should hit
        assert results[1] is not None  # query_1 should hit
        assert all(r is None for r in results[2:])  # Others should miss

        # Check metrics
        assert hyde_cache.cache_hits == 2
        assert hyde_cache.cache_misses == 3


class TestCacheWarming:
    """Test cache warming functionality."""

    @pytest.mark.asyncio
    async def test_warm_embedding_cache(self, hyde_cache, mock_cache_manager):
        """Test warming embedding cache with precomputed data."""
        hyde_cache._initialized = True

        queries_embeddings = {
            "getting started": [0.1, 0.2, 0.3],
            "API reference": [0.4, 0.5, 0.6],
            "tutorials": [0.7, 0.8, 0.9],
        }

        warmed_count = await hyde_cache.warm_embedding_cache(queries_embeddings)

        assert warmed_count == len(queries_embeddings)
        assert mock_cache_manager.set.call_count == len(queries_embeddings)

    @pytest.mark.asyncio
    async def test_warm_document_cache(self, hyde_cache, mock_cache_manager):
        """Test warming document cache with precomputed data."""
        hyde_cache._initialized = True

        queries_documents = {
            "API guide": ["API doc 1", "API doc 2"],
            "tutorial": ["Tutorial doc 1", "Tutorial doc 2"],
        }

        warmed_count = await hyde_cache.warm_document_cache(queries_documents)

        assert warmed_count == len(queries_documents)
        assert mock_cache_manager.set.call_count == len(queries_documents)

    @pytest.mark.asyncio
    async def test_warm_cache_with_generator(self, hyde_cache, mock_cache_manager):
        """Test cache warming with embedding generator function."""
        hyde_cache._initialized = True

        queries = ["query1", "query2", "query3"]

        # Mock embedding generator
        async def mock_embedding_generator(query):
            return [0.1 * len(query), 0.2 * len(query), 0.3 * len(query)]

        warmed_count = await hyde_cache.warm_cache_with_generator(
            queries=queries,
            embedding_generator=mock_embedding_generator,
        )

        assert warmed_count == len(queries)
        assert mock_cache_manager.set.call_count == len(queries)
