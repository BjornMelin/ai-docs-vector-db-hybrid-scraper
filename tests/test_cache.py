"""Tests for intelligent caching layer."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.services.cache.local_cache import LocalCache
from src.services.cache.manager import CacheManager, CacheType
from src.services.cache.metrics import CacheMetrics
from src.services.cache.dragonfly_cache import DragonflyCache


class TestLocalCache:
    """Test local LRU cache implementation."""

    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LocalCache(max_size=10, default_ttl=60)

        # Test set and get
        assert await cache.set("key1", "value1") is True
        assert await cache.get("key1") == "value1"

        # Test exists
        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

        # Test delete
        assert await cache.delete("key1") is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LocalCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LocalCache(max_size=3, default_ttl=60)

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_memory_limit(self):
        """Test memory limit enforcement."""
        cache = LocalCache(max_size=100, max_memory_mb=0.001)  # 1KB limit

        # Add large value
        large_value = "x" * 2000  # 2KB
        result = await cache.set("key1", large_value)
        assert result is False  # Should fail due to memory limit

        # Add small value
        small_value = "x" * 100
        result = await cache.set("key2", small_value)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test cache clearing."""
        cache = LocalCache(max_size=10, default_ttl=60)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestDragonflyCache:
    """Test DragonflyDB cache implementation."""

    @pytest.fixture
    async def dragonfly_cache(self):
        """Create DragonflyDB cache with mocked client."""
        cache = DragonflyCache(redis_url="redis://localhost:6379", default_ttl=60)
        # Mock Redis client
        cache._client = AsyncMock()
        return cache

    @pytest.mark.asyncio
    async def test_basic_operations(self, dragonfly_cache):
        """Test basic cache operations."""
        # Mock Redis responses
        dragonfly_cache._client.get.return_value = b'"value1"'
        dragonfly_cache._client.set.return_value = True
        dragonfly_cache._client.exists.return_value = True
        dragonfly_cache._client.delete.return_value = 1

        # Test set and get
        assert await dragonfly_cache.set("key1", "value1") is True
        assert await dragonfly_cache.get("key1") == "value1"

        # Test exists
        assert await dragonfly_cache.exists("key1") is True

        # Test delete
        assert await dragonfly_cache.delete("key1") is True

    @pytest.mark.asyncio
    async def test_batch_operations(self, dragonfly_cache):
        """Test batch get operations."""
        # Mock mget response
        dragonfly_cache._client.mget.return_value = [
            b'"value1"',
            None,
            b'"value3"',
        ]

        results = await dragonfly_cache.mget(["key1", "key2", "key3"])

        assert results == ["value1", None, "value3"]

    @pytest.mark.asyncio
    async def test_error_handling(self, dragonfly_cache):
        """Test error handling."""
        from redis.exceptions import RedisError
        
        # Mock Redis error
        dragonfly_cache._client.get.side_effect = RedisError("Redis connection error")

        # Should return None on error
        result = await dragonfly_cache.get("key1")
        assert result is None


class TestCacheManager:
    """Test cache manager with tiered caching."""

    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager with mocked caches."""
        manager = CacheManager(
            dragonfly_url="redis://localhost:6379",
            enable_local_cache=True,
            enable_distributed_cache=True,
            key_prefix="test:",
        )

        # Mock caches
        manager._local_cache = AsyncMock()
        manager._distributed_cache = AsyncMock()

        return manager

    @pytest.mark.asyncio
    async def test_get_with_local_hit(self, cache_manager):
        """Test get with local cache hit."""
        cache_manager._local_cache.get.return_value = "value1"

        result = await cache_manager.get("key1")

        assert result == "value1"
        # Key is hashed, so just check that get was called once
        cache_manager._local_cache.get.assert_called_once()
        cache_manager._distributed_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_with_redis_hit(self, cache_manager):
        """Test get with Redis hit and local miss."""
        cache_manager._local_cache.get.return_value = None
        cache_manager._distributed_cache.get.return_value = "value1"

        result = await cache_manager.get("key1")

        assert result == "value1"
        # Key is hashed, so just check that get was called once
        cache_manager._local_cache.get.assert_called_once()
        cache_manager._distributed_cache.get.assert_called_once()
        # Should write back to local cache
        cache_manager._local_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_to_both_tiers(self, cache_manager):
        """Test set writes to both cache tiers."""
        cache_manager._local_cache.set.return_value = True
        cache_manager._distributed_cache.set.return_value = True

        result = await cache_manager.set("key1", "value1", ttl=60)

        assert result is True
        cache_manager._local_cache.set.assert_called_once()
        cache_manager._distributed_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        # Mock the specialized caches to avoid AttributeError
        cache_manager._embedding_cache = AsyncMock()
        cache_manager._search_cache = AsyncMock()
        cache_manager._embedding_cache.get_stats.return_value = {"hits": 0, "misses": 0}
        cache_manager._search_cache.get_stats.return_value = {"hits": 0, "misses": 0}
        
        # Test stats retrieval
        stats = await cache_manager.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_performance_stats(self, cache_manager):
        """Test performance statistics."""
        # Mock the metrics to avoid AttributeError
        cache_manager._metrics = AsyncMock()
        cache_manager._metrics.get_summary.return_value = {}
        
        # Test performance stats
        perf_stats = await cache_manager.get_performance_stats()
        assert isinstance(perf_stats, dict)

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_manager):
        """Test cache clearing functionality."""
        # Test clear all
        result = await cache_manager.clear()
        assert isinstance(result, bool)
        
        # Test clear specific type
        result = await cache_manager.clear(CacheType.EMBEDDINGS)
        assert isinstance(result, bool)


class TestCacheMetrics:
    """Test cache metrics tracking."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()

        # Record some hits and misses
        metrics.record_hit("embeddings", "local", 0.01)
        metrics.record_hit("embeddings", "local", 0.01)
        metrics.record_miss("embeddings", 0.02)

        summary = metrics.get_summary()
        assert summary["embeddings"]["local"]["hits"] == 2
        assert summary["embeddings"]["total"]["misses"] == 1

    def test_size_tracking(self):
        """Test cache size tracking."""
        metrics = CacheMetrics()

        # Record operations
        metrics.record_set("embeddings", 0.01, True)
        metrics.record_set("embeddings", 0.02, False)

        summary = metrics.get_summary()
        assert summary["embeddings"]["total"]["sets"] == 1
        assert summary["embeddings"]["total"]["errors"] == 1

    def test_operation_tracking(self):
        """Test operation count tracking."""
        metrics = CacheMetrics()

        # Record various operations
        metrics.record_set("embeddings", 0.01, True)
        metrics.record_hit("embeddings", "local", 0.01)
        metrics.record_miss("embeddings", 0.02)

        summary = metrics.get_summary()
        assert summary["embeddings"]["total"]["sets"] == 1
        assert summary["embeddings"]["local"]["hits"] == 1
        assert summary["embeddings"]["total"]["misses"] == 1
