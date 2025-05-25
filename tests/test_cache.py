"""Tests for intelligent caching layer."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import PropertyMock

import pytest
from src.services.cache.local_cache import LocalCache
from src.services.cache.manager import CacheManager
from src.services.cache.metrics import CacheMetrics
from src.services.cache.redis_cache import RedisCache


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


class TestRedisCache:
    """Test Redis cache implementation."""

    @pytest.fixture
    async def redis_cache(self):
        """Create Redis cache with mocked client."""
        cache = RedisCache(url="redis://localhost:6379", default_ttl=60)
        # Mock Redis client
        cache._client = AsyncMock()
        cache._initialized = True
        return cache

    @pytest.mark.asyncio
    async def test_basic_operations(self, redis_cache):
        """Test basic cache operations."""
        # Mock Redis responses
        redis_cache._client.get.return_value = b'{"data": "value1"}'
        redis_cache._client.set.return_value = True
        redis_cache._client.exists.return_value = True
        redis_cache._client.delete.return_value = 1

        # Test set and get
        assert await redis_cache.set("key1", "value1") is True
        assert await redis_cache.get("key1") == "value1"

        # Test exists
        assert await redis_cache.exists("key1") is True

        # Test delete
        assert await redis_cache.delete("key1") is True

    @pytest.mark.asyncio
    async def test_batch_operations(self, redis_cache):
        """Test batch get operations."""
        # Mock mget response
        redis_cache._client.mget.return_value = [
            b'{"data": "value1"}',
            None,
            b'{"data": "value3"}',
        ]

        results = await redis_cache.mget(["key1", "key2", "key3"])

        assert results == ["value1", None, "value3"]

    @pytest.mark.asyncio
    async def test_error_handling(self, redis_cache):
        """Test error handling."""
        # Mock Redis error
        redis_cache._client.get.side_effect = Exception("Redis error")

        # Should return None on error
        result = await redis_cache.get("key1")
        assert result is None


class TestCacheManager:
    """Test cache manager with tiered caching."""

    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager with mocked caches."""
        from src.config.models import UnifiedConfig, CacheConfig

        config = UnifiedConfig(
            cache=CacheConfig(
                enable_caching=True,
                enable_local_cache=True,
                enable_redis_cache=True,
            )
        )
        manager = CacheManager(config)

        # Mock caches
        manager._local_cache = AsyncMock()
        manager._redis_cache = AsyncMock()
        manager._initialized = True

        return manager

    @pytest.mark.asyncio
    async def test_get_with_local_hit(self, cache_manager):
        """Test get with local cache hit."""
        cache_manager._local_cache.get.return_value = "value1"

        result = await cache_manager.get("key1")

        assert result == "value1"
        cache_manager._local_cache.get.assert_called_once_with("key1")
        cache_manager._redis_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_with_redis_hit(self, cache_manager):
        """Test get with Redis hit and local miss."""
        cache_manager._local_cache.get.return_value = None
        cache_manager._redis_cache.get.return_value = "value1"

        result = await cache_manager.get("key1")

        assert result == "value1"
        cache_manager._local_cache.get.assert_called_once_with("key1")
        cache_manager._redis_cache.get.assert_called_once_with("key1")
        # Should write back to local cache
        cache_manager._local_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_to_both_tiers(self, cache_manager):
        """Test set writes to both cache tiers."""
        cache_manager._local_cache.set.return_value = True
        cache_manager._redis_cache.set.return_value = True

        result = await cache_manager.set("key1", "value1", ttl=60)

        assert result is True
        cache_manager._local_cache.set.assert_called_once()
        cache_manager._redis_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_key_patterns(self, cache_manager):
        """Test cache key generation for different data types."""
        # Test embedding key
        key = cache_manager.get_embedding_key(["text1", "text2"])
        assert key.startswith("embed:")

        # Test crawl key
        key = cache_manager.get_crawl_key("https://example.com")
        assert key.startswith("crawl:")

        # Test search key
        key = cache_manager.get_search_key("query", {"filter": "value"})
        assert key.startswith("search:")

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, cache_manager):
        """Test metrics tracking."""
        # Simulate cache hit
        cache_manager._local_cache.get.return_value = "value1"
        await cache_manager.get("key1")

        metrics = cache_manager.get_metrics()
        assert "local" in metrics
        assert metrics["local"]["requests"] > 0

    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_manager):
        """Test cache warming functionality."""
        # Mock common queries
        cache_manager._redis_cache.scan_iter = AsyncMock(
            return_value=["search:common1", "search:common2"]
        )
        cache_manager._redis_cache.get.return_value = "cached_result"

        await cache_manager.warm_cache(["common_query"])

        # Should prefetch common queries
        assert cache_manager._redis_cache.get.called


class TestCacheMetrics:
    """Test cache metrics tracking."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()

        # Record some hits and misses
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()

        stats = metrics.get_stats()
        assert stats["hit_rate"] == 0.67  # 2/3 ≈ 0.67

    def test_size_tracking(self):
        """Test cache size tracking."""
        metrics = CacheMetrics()

        metrics.update_size(100)
        metrics.update_memory_usage(1024 * 1024)  # 1MB

        stats = metrics.get_stats()
        assert stats["size"] == 100
        assert stats["memory_mb"] == 1.0

    def test_operation_tracking(self):
        """Test operation count tracking."""
        metrics = CacheMetrics()

        metrics.record_set()
        metrics.record_delete()
        metrics.record_eviction()

        stats = metrics.get_stats()
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["evictions"] == 1