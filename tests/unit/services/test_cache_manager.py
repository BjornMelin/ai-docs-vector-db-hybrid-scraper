"""Comprehensive tests for CacheManager service."""

import asyncio
from enum import Enum
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.base import CacheBackend
from src.services.cache.manager import CacheManager
from src.services.errors import CacheError


class CacheType(Enum):
    """Test cache types."""

    EMBEDDINGS = "embeddings"
    CRAWL_RESULTS = "crawl_results"
    QUERY_RESULTS = "query_results"


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = {
        "dragonfly_url": "redis://localhost:6379",
        "enable_local_cache": True,
        "enable_distributed_cache": True,
        "local_max_size": 1000,
        "local_max_memory_mb": 512,
        "distributed_ttl_seconds": {
            CacheType.EMBEDDINGS: 86400,
            CacheType.CRAWL_RESULTS: 3600,
            CacheType.QUERY_RESULTS: 7200,
        },
    }
    return config


@pytest.fixture
def manager(mock_config):
    """Create CacheManager instance for testing."""
    return CacheManager(**mock_config)


@pytest.fixture
def mock_local_cache():
    """Create mock local cache."""
    cache = AsyncMock(spec=CacheBackend)
    cache.initialize = AsyncMock()
    cache.cleanup = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    cache.clear = AsyncMock()
    cache.get_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    return cache


@pytest.fixture
def mock_dragonfly_cache():
    """Create mock Dragonfly cache."""
    cache = AsyncMock(spec=CacheBackend)
    cache.initialize = AsyncMock()
    cache.cleanup = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    cache.clear = AsyncMock()
    cache.get_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    return cache


@pytest.fixture
def mock_embedding_cache():
    """Create mock embedding cache."""
    cache = AsyncMock()
    cache.get_embedding = AsyncMock(return_value=None)
    cache.set_embedding = AsyncMock(return_value=True)
    cache.invalidate_embedding = AsyncMock(return_value=True)
    cache.get_batch_embeddings = AsyncMock(return_value=[])
    cache.set_batch_embeddings = AsyncMock(return_value=True)
    cache.get_stats = MagicMock(
        return_value={
            "embedding_hits": 0,
            "embedding_misses": 0,
            "embedding_hit_rate": 0.0,
        }
    )
    return cache


@pytest.fixture
def mock_search_cache():
    """Create mock search results cache."""
    cache = AsyncMock()
    cache.get_search_results = AsyncMock(return_value=None)
    cache.set_search_results = AsyncMock(return_value=True)
    cache.invalidate_search_results = AsyncMock(return_value=True)
    cache.get_stats = MagicMock(
        return_value={
            "search_hits": 0,
            "search_misses": 0,
        }
    )
    return cache


class TestCacheManagerInitialization:
    """Test cache manager initialization."""

    def test_manager_initialization(self, manager):
        """Test basic manager initialization."""
        assert manager._initialized is False
        assert manager._local_cache is None
        assert manager._dragonfly_cache is None
        assert manager._embedding_cache is None
        assert manager._search_cache is None
        assert manager._enable_local is True
        assert manager._enable_distributed is True

    @pytest.mark.asyncio
    async def test_initialize_all_caches(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test initialization with all caches enabled."""
        with (
            patch(
                "src.services.cache.manager.LocalCache", return_value=mock_local_cache
            ),
            patch(
                "src.services.cache.manager.DragonflyCache",
                return_value=mock_dragonfly_cache,
            ),
            patch("src.services.cache.manager.EmbeddingCache"),
            patch("src.services.cache.manager.SearchResultsCache"),
        ):
            await manager.initialize()

        assert manager._initialized is True
        assert manager._local_cache is not None
        assert manager._dragonfly_cache is not None
        assert manager._embedding_cache is not None
        assert manager._search_cache is not None

        # Verify caches were initialized
        mock_local_cache.initialize.assert_called_once()
        mock_dragonfly_cache.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_local_only(self, manager, mock_local_cache):
        """Test initialization with only local cache."""
        manager._enable_distributed = False

        with patch(
            "src.services.cache.manager.LocalCache", return_value=mock_local_cache
        ):
            await manager.initialize()

        assert manager._initialized is True
        assert manager._local_cache is not None
        assert manager._dragonfly_cache is None
        mock_local_cache.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_distributed_only(self, manager, mock_dragonfly_cache):
        """Test initialization with only distributed cache."""
        manager._enable_local = False

        with (
            patch(
                "src.services.cache.manager.DragonflyCache",
                return_value=mock_dragonfly_cache,
            ),
            patch("src.services.cache.manager.EmbeddingCache"),
            patch("src.services.cache.manager.SearchResultsCache"),
        ):
            await manager.initialize()

        assert manager._initialized is True
        assert manager._local_cache is None
        assert manager._dragonfly_cache is not None
        mock_dragonfly_cache.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_caches(self, manager):
        """Test initialization with no caches enabled."""
        manager._enable_local = False
        manager._enable_distributed = False

        with pytest.raises(CacheError, match="No cache backends enabled"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, manager, mock_local_cache):
        """Test that initialization is idempotent."""
        with patch(
            "src.services.cache.manager.LocalCache", return_value=mock_local_cache
        ):
            await manager.initialize()
            await manager.initialize()  # Second call

        # Should only initialize once
        mock_local_cache.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, manager, mock_local_cache, mock_dragonfly_cache):
        """Test cleanup of all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._embedding_cache = AsyncMock()
        manager._search_cache = AsyncMock()
        manager._initialized = True

        await manager.close()

        mock_local_cache.cleanup.assert_called_once()
        mock_dragonfly_cache.cleanup.assert_called_once()
        assert manager._initialized is False


class TestBasicCacheOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_get_not_initialized(self, manager):
        """Test get operation when not initialized."""
        with pytest.raises(CacheError, match="Cache manager not initialized"):
            await manager.get("test_key", CacheType.EMBEDDINGS)

    @pytest.mark.asyncio
    async def test_get_local_hit(self, manager, mock_local_cache, mock_dragonfly_cache):
        """Test get operation with local cache hit."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Local cache returns value
        mock_local_cache.get.return_value = {"data": "local_value"}

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result == {"data": "local_value"}
        mock_local_cache.get.assert_called_once_with("embeddings:test_key")
        # Should not check distributed cache on local hit
        mock_dragonfly_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_distributed_hit(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test get operation with distributed cache hit."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Local cache miss, distributed hit
        mock_local_cache.get.return_value = None
        mock_dragonfly_cache.get.return_value = {"data": "distributed_value"}

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result == {"data": "distributed_value"}
        mock_local_cache.get.assert_called_once()
        mock_dragonfly_cache.get.assert_called_once()

        # Should write back to local cache
        mock_local_cache.set.assert_called_once_with(
            "embeddings:test_key", {"data": "distributed_value"}, ttl=None
        )

    @pytest.mark.asyncio
    async def test_get_cache_miss(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test get operation with cache miss."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Both caches miss
        mock_local_cache.get.return_value = None
        mock_dragonfly_cache.get.return_value = None

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result is None
        mock_local_cache.get.assert_called_once()
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_all_caches(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test set operation to all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        data = {"key": "value"}
        success = await manager.set("test_key", data, CacheType.EMBEDDINGS, ttl=3600)

        assert success is True

        # Should set in both caches
        mock_local_cache.set.assert_called_once_with(
            "embeddings:test_key", data, ttl=3600
        )
        mock_dragonfly_cache.set.assert_called_once_with(
            "embeddings:test_key", data, ttl=3600
        )

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test set operation with default TTL."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        data = {"key": "value"}
        success = await manager.set("test_key", data, CacheType.CRAWL_RESULTS)

        assert success is True

        # Should use default TTL for cache type
        expected_ttl = manager._distributed_ttl[CacheType.CRAWL_RESULTS]
        mock_dragonfly_cache.set.assert_called_once_with(
            "crawl_results:test_key", data, ttl=expected_ttl
        )

    @pytest.mark.asyncio
    async def test_delete_from_all_caches(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test delete operation from all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        success = await manager.delete("test_key", CacheType.QUERY_RESULTS)

        assert success is True

        # Should delete from both caches
        mock_local_cache.delete.assert_called_once_with("query_results:test_key")
        mock_dragonfly_cache.delete.assert_called_once_with("query_results:test_key")

    @pytest.mark.asyncio
    async def test_exists_in_any_cache(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test exists operation checking all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Test when exists in local
        mock_local_cache.exists.return_value = True
        assert await manager.exists("key1", CacheType.EMBEDDINGS) is True

        # Test when exists only in distributed
        mock_local_cache.exists.return_value = False
        mock_dragonfly_cache.exists.return_value = True
        assert await manager.exists("key2", CacheType.EMBEDDINGS) is True

        # Test when doesn't exist anywhere
        mock_local_cache.exists.return_value = False
        mock_dragonfly_cache.exists.return_value = False
        assert await manager.exists("key3", CacheType.EMBEDDINGS) is False


class TestSpecializedCaches:
    """Test specialized cache operations."""

    @pytest.mark.asyncio
    async def test_embedding_cache_operations(self, manager, mock_embedding_cache):
        """Test embedding cache specific operations."""
        manager._embedding_cache = mock_embedding_cache
        manager._initialized = True

        # Test get embedding
        mock_embedding_cache.get_embedding.return_value = [0.1, 0.2, 0.3]

        embedding = await manager.get_embedding(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
        )

        assert embedding == [0.1, 0.2, 0.3]
        mock_embedding_cache.get_embedding.assert_called_once()

        # Test set embedding
        await manager.set_embedding(
            text="Test text",
            model="text-embedding-3-small",
            embedding=[0.1, 0.2, 0.3],
            provider="openai",
            dimensions=3,
        )

        mock_embedding_cache.set_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_cache_operations(self, manager, mock_search_cache):
        """Test search cache specific operations."""
        manager._search_cache = mock_search_cache
        manager._initialized = True

        # Test get search results
        mock_results = [{"id": "1", "score": 0.9}]
        mock_search_cache.get_search_results.return_value = mock_results

        results = await manager.get_search_results(
            query="test query", collection="docs", filters={"type": "api"}
        )

        assert results == mock_results
        mock_search_cache.get_search_results.assert_called_once()

        # Test set search results
        await manager.set_search_results(
            query="test query",
            collection="docs",
            results=mock_results,
            filters={"type": "api"},
        )

        mock_search_cache.set_search_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_specialized_cache_not_available(self, manager):
        """Test error when specialized cache not available."""
        manager._initialized = True
        manager._embedding_cache = None

        with pytest.raises(CacheError, match="Embedding cache not available"):
            await manager.get_embedding(
                text="Test", provider="openai", model="model", dimensions=1536
            )


class TestCacheStatistics:
    """Test cache statistics and monitoring."""

    def test_get_stats_all_caches(
        self,
        manager,
        mock_local_cache,
        mock_dragonfly_cache,
        mock_embedding_cache,
        mock_search_cache,
    ):
        """Test getting statistics from all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._embedding_cache = mock_embedding_cache
        manager._search_cache = mock_search_cache
        manager._initialized = True

        # Set up mock stats
        mock_local_cache.get_stats.return_value = {
            "hits": 100,
            "misses": 50,
            "hit_rate": 0.667,
        }
        mock_dragonfly_cache.get_stats.return_value = {
            "hits": 200,
            "misses": 100,
            "hit_rate": 0.667,
        }

        stats = manager.get_stats()

        assert "local_cache" in stats
        assert "distributed_cache" in stats
        assert "embedding_cache" in stats
        assert "search_cache" in stats
        assert stats["local_cache"]["hits"] == 100
        assert stats["distributed_cache"]["hits"] == 200

    def test_get_stats_partial_caches(self, manager, mock_local_cache):
        """Test getting statistics with only some caches enabled."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = None
        manager._initialized = True

        stats = manager.get_stats()

        assert "local_cache" in stats
        assert stats["distributed_cache"] is None
        assert stats["embedding_cache"] is None


class TestClearOperations:
    """Test cache clearing operations."""

    @pytest.mark.asyncio
    async def test_clear_all_caches(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test clearing all caches."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        await manager.clear_all()

        mock_local_cache.clear.assert_called_once()
        mock_dragonfly_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_type(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test clearing specific cache type."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Mock scan_keys to return some keys
        mock_local_cache.scan_keys = AsyncMock(
            return_value=["embeddings:key1", "embeddings:key2"]
        )
        mock_dragonfly_cache.scan_keys = AsyncMock(
            return_value=["embeddings:key3", "embeddings:key4"]
        )

        await manager.clear_cache_type(CacheType.EMBEDDINGS)

        # Should scan for keys and delete them
        mock_local_cache.scan_keys.assert_called_once_with("embeddings:*")
        mock_dragonfly_cache.scan_keys.assert_called_once_with("embeddings:*")
        assert mock_local_cache.delete.call_count == 2
        assert mock_dragonfly_cache.delete.call_count == 2


class TestCacheWarming:
    """Test cache warming functionality."""

    @pytest.mark.asyncio
    async def test_warm_embeddings_cache(self, manager, mock_embedding_cache):
        """Test warming embedding cache with precomputed data."""
        manager._embedding_cache = mock_embedding_cache
        manager._initialized = True

        warmup_data = {
            "Common query 1": [0.1, 0.2, 0.3],
            "Common query 2": [0.4, 0.5, 0.6],
        }

        success = await manager.warm_embeddings_cache(
            embeddings_map=warmup_data,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
        )

        assert success is True
        mock_embedding_cache.set_batch_embeddings.assert_called_once_with(
            embeddings_map=warmup_data,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
            ttl=None,
        )

    @pytest.mark.asyncio
    async def test_warm_search_cache(self, manager, mock_search_cache):
        """Test warming search cache with common queries."""
        manager._search_cache = mock_search_cache
        manager._initialized = True

        common_queries = [
            {
                "query": "getting started",
                "collection": "docs",
                "results": [{"id": "1", "score": 0.95}],
            },
            {
                "query": "api reference",
                "collection": "docs",
                "results": [{"id": "2", "score": 0.92}],
            },
        ]

        for query_data in common_queries:
            await manager.warm_search_cache(
                query=query_data["query"],
                collection=query_data["collection"],
                results=query_data["results"],
            )

        assert mock_search_cache.set_search_results.call_count == 2


class TestErrorHandling:
    """Test error handling in cache operations."""

    @pytest.mark.asyncio
    async def test_get_with_local_error(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test get operation when local cache errors."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Local cache throws error
        mock_local_cache.get.side_effect = Exception("Local cache error")
        mock_dragonfly_cache.get.return_value = {"data": "fallback_value"}

        # Should fall back to distributed cache
        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result == {"data": "fallback_value"}
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_partial_failure(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test set operation when one cache fails."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Local succeeds, distributed fails
        mock_local_cache.set.return_value = True
        mock_dragonfly_cache.set.side_effect = Exception("Distributed cache error")

        # Should still return success if at least one cache succeeds
        success = await manager.set("test_key", {"data": "value"}, CacheType.EMBEDDINGS)

        assert success is True

    @pytest.mark.asyncio
    async def test_set_with_all_failures(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test set operation when all caches fail."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Both caches fail
        mock_local_cache.set.side_effect = Exception("Local error")
        mock_dragonfly_cache.set.side_effect = Exception("Distributed error")

        success = await manager.set("test_key", {"data": "value"}, CacheType.EMBEDDINGS)

        assert success is False


class TestConcurrency:
    """Test concurrent cache operations."""

    @pytest.mark.asyncio
    async def test_concurrent_gets(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test concurrent get operations."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Simulate different cache responses
        async def mock_get_local(key):
            await asyncio.sleep(0.01)  # Simulate latency
            if "key1" in key:
                return {"data": "local1"}
            return None

        async def mock_get_distributed(key):
            await asyncio.sleep(0.02)  # Simulate higher latency
            if "key2" in key:
                return {"data": "distributed2"}
            return None

        mock_local_cache.get.side_effect = mock_get_local
        mock_dragonfly_cache.get.side_effect = mock_get_distributed

        # Execute concurrent gets
        tasks = [
            manager.get("key1", CacheType.EMBEDDINGS),
            manager.get("key2", CacheType.EMBEDDINGS),
            manager.get("key3", CacheType.EMBEDDINGS),
        ]

        results = await asyncio.gather(*tasks)

        assert results[0] == {"data": "local1"}
        assert results[1] == {"data": "distributed2"}
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_concurrent_sets(
        self, manager, mock_local_cache, mock_dragonfly_cache
    ):
        """Test concurrent set operations."""
        manager._local_cache = mock_local_cache
        manager._dragonfly_cache = mock_dragonfly_cache
        manager._initialized = True

        # Execute concurrent sets
        tasks = [
            manager.set(f"key{i}", {"data": f"value{i}"}, CacheType.EMBEDDINGS)
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert all(results)  # All should succeed
        assert mock_local_cache.set.call_count == 5
        assert mock_dragonfly_cache.set.call_count == 5
