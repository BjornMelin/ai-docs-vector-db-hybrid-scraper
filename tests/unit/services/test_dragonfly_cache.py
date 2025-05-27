"""Tests for DragonflyDB cache implementation."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.cache.dragonfly_cache import DragonflyCache


class TestDragonflyCache:
    """Test suite for DragonflyCache."""

    @pytest.fixture
    def cache_config(self):
        """Basic cache configuration."""
        return {
            "redis_url": "redis://localhost:6379",
            "key_prefix": "test:",
            "compression_threshold": 1024,
            "max_connections": 50,
            "max_retries": 3,
        }

    @pytest.fixture
    def cache(self, cache_config):
        """Create DragonflyCache instance."""
        return DragonflyCache(**cache_config)

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.redis_url == "redis://localhost:6379"
        assert cache.key_prefix == "test:"
        assert cache.compression_threshold == 1024
        assert cache.max_connections == 50
        assert cache.max_retries == 3

    def test_key_generation(self, cache):
        """Test key generation with prefix."""
        assert cache._get_key("mykey") == "test:mykey"
        assert cache._get_key("another") == "test:another"

    def test_serialization(self, cache):
        """Test data serialization and deserialization."""
        # Test various data types
        test_cases = [
            "string_value",
            {"dict": "value", "number": 42},
            [1, 2, 3, "list"],
            42,
            3.14159,
            True,
            None,
        ]

        for test_data in test_cases:
            serialized = cache._serialize(test_data)
            assert isinstance(serialized, bytes)

            deserialized = cache._deserialize(serialized)
            assert deserialized == test_data

    def test_compression(self, cache):
        """Test compression for large values."""
        # Small value - no compression
        small_data = "small"
        serialized = cache._serialize(small_data)
        assert not serialized.startswith(b"zstd:")

        # Large value - should be compressed
        large_data = "x" * 2000  # Larger than compression_threshold
        serialized = cache._serialize(large_data)
        assert serialized.startswith(b"zstd:")

        # Verify decompression works
        deserialized = cache._deserialize(serialized)
        assert deserialized == large_data

    @pytest.mark.asyncio
    async def test_get_success(self, cache):
        """Test successful get operation."""
        mock_client = AsyncMock()
        mock_client.get.return_value = json.dumps("test_value").encode()

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("test_key")
            assert result == "test_value"
            mock_client.get.assert_called_once_with("test:test_key")

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        """Test cache miss (key not found)."""
        mock_client = AsyncMock()
        mock_client.get.return_value = None

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("missing_key")
            assert result is None
            mock_client.get.assert_called_once_with("test:missing_key")

    @pytest.mark.asyncio
    async def test_set_success(self, cache):
        """Test successful set operation."""
        mock_client = AsyncMock()
        mock_client.set.return_value = True

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.set("test_key", "test_value", ttl=300)
            assert result is True
            mock_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_success(self, cache):
        """Test successful delete operation."""
        mock_client = AsyncMock()
        mock_client.delete.return_value = 1

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.delete("test_key")
            assert result is True
            mock_client.delete.assert_called_once_with("test:test_key")

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test key existence check."""
        mock_client = AsyncMock()
        mock_client.exists.return_value = 1

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.exists("test_key")
            assert result is True
            mock_client.exists.assert_called_once_with("test:test_key")

    @pytest.mark.asyncio
    async def test_ttl(self, cache):
        """Test TTL retrieval."""
        mock_client = AsyncMock()
        mock_client.ttl.return_value = 300

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ttl("test_key")
            assert result == 300
            mock_client.ttl.assert_called_once_with("test:test_key")

    @pytest.mark.asyncio
    async def test_size(self, cache):
        """Test cache size calculation."""
        mock_client = AsyncMock()
        mock_client.dbsize.return_value = 100

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.size()
            assert result == 100
            mock_client.dbsize.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test cache clearing."""
        mock_client = AsyncMock()
        mock_client.flushdb.return_value = True

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.clear()
            assert result is True
            mock_client.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_mget_basic(self, cache):
        """Test multi-get operation."""
        mock_client = AsyncMock()
        mock_client.mget.return_value = [
            json.dumps("value1").encode(),
            None,
            json.dumps("value3").encode(),
        ]

        with patch.object(cache, "_get_client", return_value=mock_client):
            keys = ["key1", "key2", "key3"]
            result = await cache.mget(keys)
            expected = {"key1": "value1", "key2": None, "key3": "value3"}
            assert result == expected

    @pytest.mark.asyncio
    async def test_batch_operations(self, cache):
        """Test batch set and get operations."""
        # Mock for batch set
        mock_client = AsyncMock()
        mock_pipe = AsyncMock()
        mock_pipe.execute.return_value = [True, True, True]

        # Configure pipeline context manager
        mock_client.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_client.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(cache, "_get_client", return_value=mock_client):
            # Test batch set
            data = {"key1": "value1", "key2": "value2", "key3": "value3"}
            result = await cache.set_many(data, ttl=300)
            assert result is True

    @pytest.mark.asyncio
    async def test_scan_keys(self, cache):
        """Test key scanning."""
        mock_client = AsyncMock()

        # Mock scan_iter to return keys
        async def mock_scan_iter(match=None, count=None):
            keys = [b"test:key1", b"test:key2", b"test:key3"]
            for key in keys:
                yield key

        mock_client.scan_iter.return_value = mock_scan_iter()

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.scan_keys("key*")
            assert "key1" in result
            assert "key2" in result
            assert "key3" in result

    @pytest.mark.asyncio
    async def test_error_handling(self, cache):
        """Test error handling in cache operations."""
        from redis.exceptions import RedisError

        mock_client = AsyncMock()
        mock_client.get.side_effect = RedisError("Connection failed")

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("test_key")
            assert result is None  # Should return None on error

    @pytest.mark.asyncio
    async def test_memory_usage(self, cache):
        """Test memory usage calculation."""
        mock_client = AsyncMock()
        mock_client.memory_usage.return_value = 1024

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get_memory_usage("test_key")
            assert result == 1024

    @pytest.mark.asyncio
    async def test_close(self, cache):
        """Test cache cleanup."""
        mock_pool = AsyncMock()
        cache.pool = mock_pool

        await cache.close()
        mock_pool.disconnect.assert_called_once()

    def test_performance_optimizations(self, cache):
        """Test DragonflyDB-specific performance optimizations."""
        # Verify higher connection pool is configured
        assert cache.max_connections >= 50

        # Verify compression is enabled for large values
        assert cache.compression_threshold > 0

        # Verify retry logic is configured
        assert cache.max_retries >= 3

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache):
        """Test concurrent cache operations."""
        mock_client = AsyncMock()
        mock_client.get.return_value = json.dumps("test_value").encode()

        with patch.object(cache, "_get_client", return_value=mock_client):
            # Execute multiple concurrent operations
            tasks = [cache.get(f"key_{i}") for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All operations should succeed
            assert len(results) == 10
            assert all(result == "test_value" for result in results)

    @pytest.mark.asyncio
    async def test_integration_patterns(self, cache):
        """Test common integration patterns."""
        mock_client = AsyncMock()

        # Test cache-aside pattern
        mock_client.get.return_value = None  # Cache miss
        mock_client.set.return_value = True

        with patch.object(cache, "_get_client", return_value=mock_client):
            # Simulate cache-aside pattern
            cached_value = await cache.get("user:123")
            if cached_value is None:
                # Simulate database fetch
                fresh_value = {"id": 123, "name": "John"}
                await cache.set("user:123", fresh_value, ttl=3600)
                cached_value = fresh_value

            assert cached_value == {"id": 123, "name": "John"}
