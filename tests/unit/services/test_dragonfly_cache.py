"""Tests for DragonflyCache implementation."""

import json
import zlib
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from redis.exceptions import ConnectionError
from redis.exceptions import TimeoutError
from src.services.cache.dragonfly_cache import DragonflyCache


class TestDragonflyInitialization:
    """Test DragonflyCache initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        cache = DragonflyCache()

        assert cache.redis_url == "redis://localhost:6379"
        assert cache.key_prefix == ""
        assert cache.enable_compression is True
        assert cache.compression_threshold == 1024
        assert cache._client is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        cache = DragonflyCache(
            redis_url="redis://custom:6379",
            key_prefix="test:",
            max_connections=100,
            enable_compression=False,
            compression_threshold=2048,
        )

        assert cache.redis_url == "redis://custom:6379"
        assert cache.key_prefix == "test:"
        assert cache.enable_compression is False
        assert cache.compression_threshold == 2048

    @pytest.mark.asyncio
    async def test_client_lazy_initialization(self):
        """Test lazy client initialization."""
        cache = DragonflyCache()

        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_client

            # First access should create client
            client = await cache.client
            assert client == mock_client
            mock_client.ping.assert_called_once()

            # Second access should return same client
            client2 = await cache.client
            assert client2 == mock_client
            assert mock_client.ping.call_count == 1  # Still only called once

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing cache connection."""
        cache = DragonflyCache()

        # Mock client and pool
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        cache._client = mock_client

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        cache.pool = mock_pool

        await cache.close()

        mock_client.aclose.assert_called_once()
        mock_pool.aclose.assert_called_once()
        assert cache._client is None


class TestDragonflyOperations:
    """Test DragonflyCache operations."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.exists = AsyncMock(return_value=0)
        client.expire = AsyncMock(return_value=True)
        client.ttl = AsyncMock(return_value=-2)
        client.flushdb = AsyncMock()
        client.dbsize = AsyncMock(return_value=100)
        client.info = AsyncMock(return_value={"db0": {"keys": 100, "expires": 10}})
        client.ping = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def cache_with_client(self, mock_redis_client):
        """Create cache with mocked client."""
        cache = DragonflyCache()
        cache._client = mock_redis_client
        return cache

    @pytest.mark.asyncio
    async def test_get_key_not_found(self, cache_with_client):
        """Test get operation when key doesn't exist."""
        result = await cache_with_client.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_key_found(self, cache_with_client):
        """Test get operation when key exists."""
        cache = cache_with_client
        test_data = {"key": "value", "number": 42}

        # Mock uncompressed data
        cache._client.get.return_value = json.dumps(test_data).encode()

        result = await cache.get("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_get_with_compression(self, cache_with_client):
        """Test get operation with compressed data."""
        cache = cache_with_client
        test_data = {"data": "x" * 2000}  # Large data

        # Mock compressed data with prefix
        json_str = json.dumps(test_data, separators=(",", ":"))
        compressed = b"Z:" + zlib.compress(json_str.encode())
        cache._client.get.return_value = compressed

        result = await cache.get("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_set_basic(self, cache_with_client):
        """Test basic set operation."""
        cache = cache_with_client
        test_data = {"key": "value"}

        result = await cache.set("test_key", test_data)
        assert result is True

        # Verify Redis set was called
        cache._client.set.assert_called_once()
        call_args = cache._client.set.call_args
        assert call_args[0][0] == "test_key"  # No prefix by default

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache_with_client):
        """Test set operation with TTL."""
        cache = cache_with_client
        test_data = {"key": "value"}
        ttl = 3600

        await cache.set("test_key", test_data, ttl=ttl)

        call_args = cache._client.set.call_args
        assert call_args[1]["ex"] == ttl

    @pytest.mark.asyncio
    async def test_set_with_compression(self, cache_with_client):
        """Test set operation with compression for large data."""
        cache = cache_with_client
        # Large data that should trigger compression
        test_data = {"data": "x" * 2000}

        await cache.set("test_key", test_data)

        # Verify data was compressed
        call_args = cache._client.set.call_args
        stored_data = call_args[0][1]

        # Should have compression prefix
        assert stored_data.startswith(b"Z:")
        # Should be compressed (smaller than JSON)
        assert len(stored_data) < len(json.dumps(test_data).encode())

    @pytest.mark.asyncio
    async def test_set_without_compression_small_data(self, cache_with_client):
        """Test set operation doesn't compress small data."""
        cache = cache_with_client
        test_data = {"key": "small"}

        await cache.set("test_key", test_data)

        call_args = cache._client.set.call_args
        stored_data = call_args[0][1]

        # Should be plain JSON (with compact separators)
        assert stored_data == json.dumps(test_data, separators=(",", ":")).encode()

    @pytest.mark.asyncio
    async def test_delete_existing_key(self, cache_with_client):
        """Test delete operation for existing key."""
        cache = cache_with_client
        cache._client.delete.return_value = 1  # 1 key deleted

        result = await cache.delete("test_key")
        assert result is True

        cache._client.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache_with_client):
        """Test delete operation for non-existent key."""
        cache = cache_with_client
        cache._client.delete.return_value = 0  # No keys deleted

        result = await cache.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_key_present(self, cache_with_client):
        """Test exists operation when key is present."""
        cache = cache_with_client
        cache._client.exists.return_value = 1

        result = await cache.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_key_absent(self, cache_with_client):
        """Test exists operation when key is absent."""
        cache = cache_with_client
        cache._client.exists.return_value = 0

        result = await cache.exists("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_expire_key(self, cache_with_client):
        """Test setting expiration on a key."""
        cache = cache_with_client

        result = await cache.expire("test_key", 3600)
        assert result is True

        cache._client.expire.assert_called_once_with("test_key", 3600)

    @pytest.mark.asyncio
    async def test_ttl_key_with_expiry(self, cache_with_client):
        """Test getting TTL of key with expiry."""
        cache = cache_with_client
        cache._client.ttl.return_value = 3600

        result = await cache.ttl("test_key")
        assert result == 3600

    @pytest.mark.asyncio
    async def test_ttl_key_without_expiry(self, cache_with_client):
        """Test getting TTL of key without expiry."""
        cache = cache_with_client
        cache._client.ttl.return_value = -1  # No expiry

        result = await cache.ttl("test_key")
        assert result == 0  # max(0, -1) returns 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_with_client):
        """Test clearing entire cache."""
        cache = cache_with_client

        result = await cache.clear()

        cache._client.flushdb.assert_called_once()
        assert result == -1  # Unknown count when flushing entire DB

    @pytest.mark.asyncio
    async def test_size(self, cache_with_client):
        """Test getting cache size without prefix."""
        cache = cache_with_client

        # Mock info response
        cache._client.info.return_value = {"db0": {"keys": 100, "expires": 10}}

        size = await cache.size()

        assert size == 100
        cache._client.info.assert_called_once_with("keyspace")


class TestDragonflyBatchOperations:
    """Test batch operations using actual DragonflyCache methods."""

    @pytest.fixture
    def cache_with_pipeline(self):
        """Create cache with pipeline support."""
        cache = DragonflyCache()

        # Mock client with pipeline support
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.get = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.delete = MagicMock()
        mock_pipeline.mset = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=None)

        mock_client.pipeline = MagicMock(return_value=mock_pipeline)
        mock_client.delete = AsyncMock(return_value=2)
        mock_client.exists = AsyncMock(return_value=0)
        mock_client.mget = AsyncMock(return_value=[])
        cache._client = mock_client

        return cache, mock_pipeline

    @pytest.mark.asyncio
    async def test_get_many(self, cache_with_pipeline):
        """Test get_many operation."""
        cache, pipeline = cache_with_pipeline
        keys = ["key1", "key2", "key3"]

        # Mock pipeline results
        pipeline.execute.return_value = [
            json.dumps({"data": 1}).encode(),
            None,  # key2 not found
            json.dumps({"data": 3}).encode(),
        ]

        results = await cache.get_many(keys)

        assert len(results) == 3
        assert results["key1"] == {"data": 1}
        assert results["key2"] is None
        assert results["key3"] == {"data": 3}

    @pytest.mark.asyncio
    async def test_set_many(self, cache_with_pipeline):
        """Test set_many operation."""
        cache, pipeline = cache_with_pipeline
        items = {
            "key1": {"data": 1},
            "key2": {"data": 2},
            "key3": {"data": 3},
        }

        pipeline.execute.return_value = [True, True, True]

        results = await cache.set_many(items, ttl=3600)

        assert results["key1"] is True
        assert results["key2"] is True
        assert results["key3"] is True
        assert pipeline.setex.call_count == 3

    @pytest.mark.asyncio
    async def test_delete_many(self, cache_with_pipeline):
        """Test delete_many operation."""
        cache, pipeline = cache_with_pipeline
        keys = ["key1", "key2", "key3"]

        # Mock that 2 keys were deleted
        cache._client.delete.return_value = 2

        results = await cache.delete_many(keys)

        # With partial deletion, it should check each key
        assert cache._client.exists.call_count == 3

    @pytest.mark.asyncio
    async def test_mget(self, cache_with_pipeline):
        """Test mget operation."""
        cache, pipeline = cache_with_pipeline
        keys = ["key1", "key2", "key3"]

        # Mock mget results
        cache._client.mget.return_value = [
            json.dumps({"data": 1}).encode(),
            None,
            json.dumps({"data": 3}).encode(),
        ]

        results = await cache.mget(keys)

        assert len(results) == 3
        assert results[0] == {"data": 1}
        assert results[1] is None
        assert results[2] == {"data": 3}

    @pytest.mark.asyncio
    async def test_mset(self, cache_with_pipeline):
        """Test mset operation."""
        cache, pipeline = cache_with_pipeline
        mapping = {
            "key1": {"data": 1},
            "key2": {"data": 2},
        }

        pipeline.execute.return_value = [True, True, True]

        result = await cache.mset(mapping, ttl=3600)

        assert result is True
        assert pipeline.mset.call_count == 1
        assert pipeline.expire.call_count == 2  # One for each key


class TestDragonflyErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def cache_with_failing_client(self):
        """Create cache with client that fails."""
        cache = DragonflyCache()
        client = MagicMock()
        cache._client = client
        return cache, client

    @pytest.mark.asyncio
    async def test_get_connection_error(self, cache_with_failing_client):
        """Test get operation with connection error."""
        cache, client = cache_with_failing_client
        client.get.side_effect = ConnectionError("Connection lost")

        # Should return None on error, not raise
        result = await cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_timeout_error(self, cache_with_failing_client):
        """Test set operation with timeout."""
        cache, client = cache_with_failing_client
        client.set.side_effect = TimeoutError("Operation timeout")

        # Should return False on error, not raise
        result = await cache.set("test_key", {"data": "value"})
        assert result is False

    @pytest.mark.asyncio
    async def test_operation_lazy_client_failure(self):
        """Test operations when lazy client initialization fails."""
        cache = DragonflyCache()

        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("Cannot connect")
            mock_redis_class.return_value = mock_client

            # Should return None on error, not raise
            result = await cache.get("test_key")
            assert result is None


class TestDragonflyKeyManagement:
    """Test key management utilities."""

    def test_make_key_with_prefix(self):
        """Test key generation with prefix."""
        cache = DragonflyCache(key_prefix="myapp:")

        key = cache._make_key("user:123")
        assert key == "myapp:user:123"

    def test_make_key_empty_prefix(self):
        """Test key generation with empty prefix."""
        cache = DragonflyCache(key_prefix="")

        key = cache._make_key("user:123")
        assert key == "user:123"

    @pytest.mark.asyncio
    async def test_scan_keys(self):
        """Test scanning keys with pattern."""
        cache = DragonflyCache()
        mock_client = MagicMock()

        # Mock scan_iter method to yield results
        async def mock_scan_iter(match, count):
            for key in [b"user:1", b"user:2"]:
                if b"user" in key:
                    yield key

        mock_client.scan_iter = mock_scan_iter
        cache._client = mock_client

        keys = await cache.scan_keys("user:*")

        assert len(keys) == 2
        assert "user:1" in keys
        assert "user:2" in keys


class TestDragonflyOptimizations:
    """Test DragonflyDB-specific optimizations."""

    @pytest.fixture
    def cache_with_client(self, mock_redis_client):
        """Create cache with mocked client."""
        cache = DragonflyCache()
        cache._client = mock_redis_client
        return cache

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = MagicMock()
        client.set = AsyncMock(return_value=True)
        client.memory_usage = AsyncMock(return_value=1024)
        return client

    @pytest.mark.asyncio
    async def test_memory_usage(self, cache_with_client):
        """Test getting memory usage of a key."""
        cache = cache_with_client

        usage = await cache.get_memory_usage("test_key")

        assert usage == 1024
        cache._client.memory_usage.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_compression_efficiency(self, cache_with_client):
        """Test compression efficiency for large data."""
        cache = cache_with_client

        # Create highly compressible data
        test_data = {"data": "a" * 10000}

        await cache.set("test_key", test_data)

        call_args = cache._client.set.call_args
        stored_data = call_args[0][1]

        # Should have compression prefix
        assert stored_data.startswith(b"Z:")
        # Should achieve significant compression
        compression_ratio = len(stored_data) / len(json.dumps(test_data).encode())
        assert compression_ratio < 0.1  # Less than 10% of original size
