"""Tests for intelligent caching layer."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

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

        # Test size
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        assert await cache.size() == 2

        # Test clear
        assert await cache.clear() == 2
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LocalCache(max_size=3)

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it most recently used
        await cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still present
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"  # Still present
        assert await cache.get("key4") == "value4"  # New key

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LocalCache(default_ttl=1)  # 1 second TTL

        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_memory_limit(self):
        """Test memory-based eviction."""
        cache = LocalCache(max_size=100, max_memory_mb=0.001)  # 1KB limit

        # Try to add large value
        large_value = "x" * 2000  # 2KB
        assert await cache.set("key1", large_value) is False  # Too large

        # Add smaller values
        small_value = "x" * 100
        assert await cache.set("key2", small_value) is True

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LocalCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Generate some hits and misses
        asyncio.run(cache.set("key1", "value1"))
        asyncio.run(cache.get("key1"))  # Hit
        asyncio.run(cache.get("key2"))  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestRedisCache:
    """Test Redis cache implementation."""

    @pytest.mark.asyncio
    async def test_serialization(self):
        """Test value serialization/deserialization."""
        cache = RedisCache()

        # Test various data types
        test_data = {
            "string": "test string",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "nested": {"list": [1, {"key": "value"}]},
        }

        serialized = cache._serialize(test_data)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data

    @pytest.mark.asyncio
    async def test_compression(self):
        """Test value compression."""
        cache = RedisCache(enable_compression=True, compression_threshold=100)

        # Small value (no compression)
        small_value = "small"
        serialized = cache._serialize(small_value)
        assert not serialized.startswith(b"Z:")

        # Large value (compressed)
        large_value = "x" * 1000
        serialized = cache._serialize(large_value)
        assert serialized.startswith(b"Z:")
        assert len(serialized) < len(large_value)

    @pytest.mark.asyncio
    @patch("redis.asyncio.Redis")
    async def test_batch_operations(self, mock_redis_class):
        """Test batch get/set operations."""
        # Setup mock
        mock_client = AsyncMock()
        mock_redis_class.return_value = mock_client
        mock_client.ping.return_value = True

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_client.pipeline.return_value.__aenter__.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [b'{"value": 1}', None, b'{"value": 3}']

        cache = RedisCache()
        cache._client = mock_client

        # Test get_many
        results = await cache.get_many(["key1", "key2", "key3"])
        assert results == {
            "key1": {"value": 1},
            "key2": None,
            "key3": {"value": 3},
        }


class TestCacheManager:
    """Test two-tier cache manager."""

    @pytest.mark.asyncio
    async def test_embedding_cache_key(self):
        """Test embedding cache key generation."""
        manager = CacheManager(enable_local_cache=False, enable_redis_cache=False)

        key = manager._make_embedding_key(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        # Key should be deterministic
        assert key.startswith("emb:")
        assert ":openai:" in key
        assert ":text-embedding-3-small:" in key
        assert ":1536" in key

        # Same input should generate same key
        key2 = manager._make_embedding_key(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        )
        assert key == key2

        # Different text should generate different key
        key3 = manager._make_embedding_key(
            text="Different text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        )
        assert key != key3

    @pytest.mark.asyncio
    async def test_two_tier_caching(self):
        """Test two-tier cache behavior."""
        # Create manager with mocked caches
        manager = CacheManager()
        manager.local_cache = AsyncMock(spec=LocalCache)
        manager.redis_cache = AsyncMock(spec=RedisCache)

        # Mock cache responses
        manager.local_cache.get.return_value = None  # L1 miss
        manager.redis_cache.get.return_value = [1.0, 2.0, 3.0]  # L2 hit

        # Get embedding
        result = await manager.get_embedding(
            text="test",
            provider="openai",
            model="test-model",
            dimensions=3,
        )

        # Should populate L1 cache from L2
        assert result == [1.0, 2.0, 3.0]
        manager.local_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_invalidation_pattern(self):
        """Test pattern-based cache invalidation."""
        manager = CacheManager()
        manager.redis_cache = AsyncMock(spec=RedisCache)

        # Mock Redis client
        mock_client = AsyncMock()
        manager.redis_cache.client = mock_client

        # Mock scan_iter to return some keys
        mock_client.scan_iter.return_value.__aiter__.return_value = [
            b"aidocs:crawl:hash1:firecrawl",
            b"aidocs:crawl:hash2:firecrawl",
        ]

        # Invalidate pattern
        await manager.invalidate_pattern("crawl:*:firecrawl")

        # Should have deleted matching keys
        assert mock_client.delete.call_count == 2


class TestCacheMetrics:
    """Test cache metrics collection."""

    def test_basic_metrics(self):
        """Test basic metrics recording."""
        metrics = CacheMetrics()

        # Record some operations
        metrics.record_hit("embeddings", "local", 10.5)
        metrics.record_hit("embeddings", "redis", 25.3)
        metrics.record_miss("embeddings", 50.0)
        metrics.record_set("embeddings", 15.0, True)
        metrics.record_set("embeddings", 20.0, False)

        # Get summary
        summary = metrics.get_summary()

        assert summary["embeddings"]["local"]["hits"] == 1
        assert summary["embeddings"]["redis"]["hits"] == 1
        assert summary["embeddings"]["total"]["misses"] == 1
        assert summary["embeddings"]["total"]["sets"] == 1
        assert summary["embeddings"]["total"]["errors"] == 1

        # Check hit rate calculation
        assert (
            summary["embeddings"]["total"]["hit_rate"] == 0.0
        )  # 0 hits in total stats


class TestCacheIntegration:
    """Integration tests with embedding manager."""

    @pytest.mark.asyncio
    async def test_embedding_manager_cache_integration(self):
        """Test cache integration with embedding manager."""
        from src.services.config import APIConfig
        from src.services.embeddings.manager import EmbeddingManager

        # Create config with caching enabled
        config = APIConfig(
            enable_caching=True,
            enable_local_cache=True,
            enable_redis_cache=False,  # Disable Redis for testing
            openai_api_key="sk-test",
        )

        # Create manager
        manager = EmbeddingManager(config)

        # Mock the embedding provider
        mock_provider = AsyncMock()
        mock_provider.generate_embeddings.return_value = [[1.0, 2.0, 3.0]]
        mock_provider.cost_per_token = 0.00002
        mock_provider.model_name = "test-model"
        manager.providers = {"test": mock_provider}
        manager._initialized = True

        # First call - cache miss
        result1 = await manager.generate_embeddings(
            texts=["test text"],
            provider_name="test",
            auto_select=False,
        )
        assert result1["cache_hit"] is False
        mock_provider.generate_embeddings.assert_called_once()

        # Second call - cache hit
        mock_provider.generate_embeddings.reset_mock()
        result2 = await manager.generate_embeddings(
            texts=["test text"],
            provider_name="test",
            auto_select=False,
        )
        assert result2["cache_hit"] is True
        assert result2["embeddings"] == [[1.0, 2.0, 3.0]]
        mock_provider.generate_embeddings.assert_not_called()  # Not called due to cache

        # Cleanup
        await manager.cleanup()
