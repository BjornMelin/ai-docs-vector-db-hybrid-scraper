"""Tests for dragonfly cache module."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from redis.exceptions import RedisError
from src.services.cache.dragonfly_cache import DragonflyCache


class TestDragonflyCache:
    """Test the DragonflyCache class."""

    @pytest.fixture
    def mock_redis_pool(self):
        """Create a mock Redis connection pool."""
        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        return mock_pool

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.aclose = AsyncMock()
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = 1
        mock_client.flushdb.return_value = True
        mock_client.info.return_value = {"db0": {"keys": 100}}
        mock_client.ttl.return_value = 3600
        mock_client.expire.return_value = True
        mock_client.mget.return_value = []
        mock_client.mset.return_value = True
        mock_client.memory_usage.return_value = 256

        # Mock scan_iter
        async def mock_scan_iter(match=None, count=100):
            if match and "*" in match:
                yield b"test:key1"
                yield b"test:key2"

        mock_client.scan_iter = mock_scan_iter

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.get = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, True])
        mock_pipeline.mset = MagicMock()
        mock_pipeline.expire = MagicMock()

        # Set up pipeline context manager
        mock_pipeline_ctx = AsyncMock()
        mock_pipeline_ctx.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_client.pipeline.return_value = mock_pipeline_ctx

        return mock_client

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    def test_dragonfly_cache_initialization_full(
        self, mock_pool_from_url, mock_redis_pool
    ):
        """Test DragonflyCache initialization with all parameters."""
        mock_pool_from_url.return_value = mock_redis_pool

        cache = DragonflyCache(
            redis_url="redis://test:6379",
            default_ttl=7200,
            max_connections=100,
            socket_timeout=10.0,
            socket_connect_timeout=10.0,
            socket_keepalive=False,
            retry_on_timeout=False,
            max_retries=5,
            key_prefix="test:",
            enable_compression=False,
            compression_threshold=2048,
        )

        assert cache.redis_url == "redis://test:6379"
        assert cache.default_ttl == 7200
        assert cache.key_prefix == "test:"
        assert cache.enable_compression is False
        assert cache.compression_threshold == 2048
        assert cache._client is None

        # Verify connection pool configuration
        mock_pool_from_url.assert_called_once()
        call_args = mock_pool_from_url.call_args
        assert call_args[0][0] == "redis://test:6379"
        assert call_args[1]["max_connections"] == 100
        assert call_args[1]["socket_timeout"] == 10.0

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    def test_dragonfly_cache_initialization_defaults(
        self, mock_pool_from_url, mock_redis_pool
    ):
        """Test DragonflyCache initialization with defaults."""
        mock_pool_from_url.return_value = mock_redis_pool

        cache = DragonflyCache()

        assert cache.redis_url == "redis://localhost:6379"
        assert cache.default_ttl == 3600
        assert cache.key_prefix == ""
        assert cache.enable_compression is True
        assert cache.compression_threshold == 1024

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_client_property_lazy_initialization(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test client property lazy initialization."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        # Client should be None initially
        assert cache._client is None

        # First access should initialize
        client = await cache.client
        assert client is mock_redis_client
        assert cache._client is mock_redis_client

        # Second access should return same client
        client2 = await cache.client
        assert client2 is mock_redis_client

        # Redis class should be called only once
        mock_redis_cls.assert_called_once()
        mock_redis_client.ping.assert_called_once()

    def test_make_key_with_prefix(self):
        """Test key generation with prefix."""
        cache = DragonflyCache(key_prefix="test:")

        key = cache._make_key("mykey")
        assert key == "test:mykey"

    def test_make_key_without_prefix(self):
        """Test key generation without prefix."""
        cache = DragonflyCache(key_prefix="")

        key = cache._make_key("mykey")
        assert key == "mykey"

    def test_serialize_simple_data(self):
        """Test serialization of simple data types."""
        cache = DragonflyCache(enable_compression=False)

        # Test string
        result = cache._serialize("hello")
        expected = json.dumps(
            "hello", separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert result == expected

        # Test number
        result = cache._serialize(42)
        expected = json.dumps(42, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        assert result == expected

        # Test dict
        data = {"key": "value", "number": 123}
        result = cache._serialize(data)
        expected = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        assert result == expected

    def test_serialize_with_compression(self):
        """Test serialization with compression."""
        cache = DragonflyCache(enable_compression=True, compression_threshold=10)

        # Large data that should be compressed
        large_data = "x" * 1000
        result = cache._serialize(large_data)

        # Should have compression marker
        assert result.startswith(b"Z:")
        # Should be smaller than uncompressed
        uncompressed = json.dumps(
            large_data, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert len(result) < len(uncompressed)

    def test_serialize_no_compression_below_threshold(self):
        """Test serialization with no compression below threshold."""
        cache = DragonflyCache(enable_compression=True, compression_threshold=1000)

        # Small data below threshold
        small_data = "hello"
        result = cache._serialize(small_data)

        # Should not have compression marker
        assert not result.startswith(b"Z:")
        expected = json.dumps(
            small_data, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert result == expected

    def test_deserialize_simple_data(self):
        """Test deserialization of simple data."""
        cache = DragonflyCache(enable_compression=False)

        # Test string
        data = json.dumps("hello", separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        result = cache._deserialize(data)
        assert result == "hello"

        # Test number
        data = json.dumps(42, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        result = cache._deserialize(data)
        assert result == 42

        # Test dict
        original = {"key": "value", "number": 123}
        data = json.dumps(original, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        result = cache._deserialize(data)
        assert result == original

    def test_deserialize_compressed_data(self):
        """Test deserialization of compressed data."""
        cache = DragonflyCache(enable_compression=True, compression_threshold=10)

        # Create compressed data
        original = "x" * 1000
        serialized = cache._serialize(original)

        # Deserialize should return original
        result = cache._deserialize(serialized)
        assert result == original

    def test_deserialize_empty_data(self):
        """Test deserialization of empty data."""
        cache = DragonflyCache()

        result = cache._deserialize(b"")
        assert result is None

        result = cache._deserialize(None)
        assert result is None

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test successful get operation."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        # Mock serialized data
        test_data = {"key": "value"}
        serialized = json.dumps(
            test_data, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        mock_redis_client.get.return_value = serialized

        result = await cache.get("test_key")

        assert result == test_data
        mock_redis_client.get.assert_called_once_with("test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_not_found(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get operation when key not found."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.get.return_value = None

        cache = DragonflyCache()

        result = await cache.get("nonexistent_key")

        assert result is None

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_with_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get operation with key prefix."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache(key_prefix="app:")

        await cache.get("test_key")

        mock_redis_client.get.assert_called_once_with("app:test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.get.side_effect = RedisError("Connection failed")

        cache = DragonflyCache()

        result = await cache.get("test_key")

        assert result is None

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_set_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test successful set operation."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.set.return_value = True

        cache = DragonflyCache(default_ttl=3600)
        test_data = {"key": "value"}

        result = await cache.set("test_key", test_data)

        assert result is True
        mock_redis_client.set.assert_called_once()
        call_args = mock_redis_client.set.call_args
        assert call_args[0][0] == "test_key"  # key
        assert call_args[1]["ex"] == 3600  # TTL

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test set operation with custom TTL."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        await cache.set("test_key", "value", ttl=7200)

        call_args = mock_redis_client.set.call_args
        assert call_args[1]["ex"] == 7200

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_set_with_conditional_flags(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test set operation with conditional flags."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        # Test nx flag (only set if not exists)
        await cache.set("test_key", "value", nx=True)
        call_args = mock_redis_client.set.call_args
        assert call_args[1]["nx"] is True

        # Test xx flag (only set if exists)
        await cache.set("test_key", "value", xx=True)
        call_args = mock_redis_client.set.call_args
        assert call_args[1]["xx"] is True

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_set_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test set operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.set.side_effect = RedisError("Connection failed")

        cache = DragonflyCache()

        result = await cache.set("test_key", "value")

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test successful delete operation."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.return_value = 1

        cache = DragonflyCache()

        result = await cache.delete("test_key")

        assert result is True
        mock_redis_client.delete.assert_called_once_with("test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete operation when key not found."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.return_value = 0

        cache = DragonflyCache()

        result = await cache.delete("nonexistent_key")

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.side_effect = RedisError("Connection failed")

        cache = DragonflyCache()

        result = await cache.delete("test_key")

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_exists_true(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test exists operation returning True."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.exists.return_value = 1

        cache = DragonflyCache()

        result = await cache.exists("test_key")

        assert result is True
        mock_redis_client.exists.assert_called_once_with("test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_exists_false(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test exists operation returning False."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.exists.return_value = 0

        cache = DragonflyCache()

        result = await cache.exists("nonexistent_key")

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_exists_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test exists operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.exists.side_effect = RedisError("Connection failed")

        cache = DragonflyCache()

        result = await cache.exists("test_key")

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_clear_with_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test clear operation with prefix."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        # Mock scan_iter to return specific keys
        async def mock_scan_iter(match=None, count=100):
            if match == "test:*":
                yield b"test:key1"
                yield b"test:key2"

        mock_redis_client.scan_iter = mock_scan_iter

        cache = DragonflyCache(key_prefix="test:")

        result = await cache.clear()

        assert result == 2
        # Should call delete for each key found
        assert mock_redis_client.delete.call_count == 2

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_clear_without_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test clear operation without prefix (flushdb)."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache(key_prefix="")

        result = await cache.clear()

        assert result == -1  # Unknown count for flushdb
        mock_redis_client.flushdb.assert_called_once()

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_clear_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test clear operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.flushdb.side_effect = RedisError("Connection failed")

        cache = DragonflyCache(key_prefix="")

        result = await cache.clear()

        assert result == 0

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_size_with_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test size operation with prefix."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        # Mock scan_iter to return specific keys
        async def mock_scan_iter(match=None, count=100):
            if match == "test:*":
                yield b"test:key1"
                yield b"test:key2"
                yield b"test:key3"

        mock_redis_client.scan_iter = mock_scan_iter

        cache = DragonflyCache(key_prefix="test:")

        result = await cache.size()

        assert result == 3

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_size_without_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test size operation without prefix."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.info.return_value = {"db0": {"keys": 150}}

        cache = DragonflyCache(key_prefix="")

        result = await cache.size()

        assert result == 150
        mock_redis_client.info.assert_called_once_with("keyspace")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_size_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test size operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.info.side_effect = RedisError("Connection failed")

        cache = DragonflyCache(key_prefix="")

        result = await cache.size()

        assert result == 0

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_close_operation(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test close operation."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        # Initialize client
        await cache.client

        # Close should close client and pool
        await cache.close()

        assert cache._client is None
        mock_redis_client.aclose.assert_called_once()
        mock_redis_pool.aclose.assert_called_once()

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_close_without_client(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test close operation without initialized client."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        cache = DragonflyCache()

        # Close without initializing client
        await cache.close()

        mock_redis_client.aclose.assert_not_called()
        mock_redis_pool.aclose.assert_called_once()

    # TODO: Add pipeline tests for get_many, set_many, mset operations
    # These tests are currently skipped due to complex async context manager mocking

    # TODO: Add set_many tests - currently skipped due to pipeline mocking complexity

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_many_all_deleted(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete_many operation with all keys deleted."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.return_value = 3  # All 3 keys deleted

        cache = DragonflyCache()
        keys = ["key1", "key2", "key3"]

        result = await cache.delete_many(keys)

        expected = {"key1": True, "key2": True, "key3": True}
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_many_none_deleted(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete_many operation with no keys deleted."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.return_value = 0  # No keys deleted

        cache = DragonflyCache()
        keys = ["key1", "key2", "key3"]

        result = await cache.delete_many(keys)

        expected = {"key1": False, "key2": False, "key3": False}
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_many_partial_deleted(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete_many operation with partial deletion."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.return_value = 2  # Only 2 of 3 keys deleted

        # Mock exists calls for individual checks
        exists_calls = [False, True, False]  # key1 deleted, key2 exists, key3 deleted
        mock_redis_client.exists.side_effect = exists_calls

        cache = DragonflyCache()
        keys = ["key1", "key2", "key3"]

        result = await cache.delete_many(keys)

        expected = {"key1": True, "key2": False, "key3": True}
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_delete_many_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test delete_many operation with error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.delete.side_effect = RedisError("Delete failed")

        cache = DragonflyCache()
        keys = ["key1", "key2"]

        result = await cache.delete_many(keys)

        expected = {"key1": False, "key2": False}
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_mget_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test mget operation successfully."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        test_data1 = json.dumps("value1", separators=(",", ":")).encode("utf-8")
        test_data2 = json.dumps("value2", separators=(",", ":")).encode("utf-8")
        mock_redis_client.mget.return_value = [test_data1, None, test_data2]

        cache = DragonflyCache()
        keys = ["key1", "key2", "key3"]

        result = await cache.mget(keys)

        expected = ["value1", None, "value2"]
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_mget_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test mget operation with error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.mget.side_effect = RedisError("MGET failed")

        cache = DragonflyCache()
        keys = ["key1", "key2"]

        result = await cache.mget(keys)

        expected = [None, None]
        assert result == expected

    # TODO: Add mset tests - currently skipped due to pipeline mocking complexity

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_ttl_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test ttl operation successfully."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.ttl.return_value = 3600

        cache = DragonflyCache()

        result = await cache.ttl("test_key")

        assert result == 3600
        mock_redis_client.ttl.assert_called_once_with("test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_ttl_negative_value(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test ttl operation with negative value."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.ttl.return_value = -1  # No TTL

        cache = DragonflyCache()

        result = await cache.ttl("test_key")

        assert result == 0  # Should return 0 for negative values

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_ttl_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test ttl operation with error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.ttl.side_effect = RedisError("TTL failed")

        cache = DragonflyCache()

        result = await cache.ttl("test_key")

        assert result == 0

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_expire_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test expire operation successfully."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.expire.return_value = True

        cache = DragonflyCache()

        result = await cache.expire("test_key", 7200)

        assert result is True
        mock_redis_client.expire.assert_called_once_with("test_key", 7200)

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_expire_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test expire operation with error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.expire.side_effect = RedisError("Expire failed")

        cache = DragonflyCache()

        result = await cache.expire("test_key", 7200)

        assert result is False

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_scan_keys_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test scan_keys operation successfully."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        # Mock scan_iter to return bytes
        async def mock_scan_iter(match=None, count=100):
            if match == "test:*":
                yield b"test:key1"
                yield b"test:key2"

        mock_redis_client.scan_iter = mock_scan_iter

        cache = DragonflyCache(key_prefix="test:")

        result = await cache.scan_keys("*")

        expected = ["key1", "key2"]
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_scan_keys_without_prefix(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test scan_keys operation without prefix."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client

        async def mock_scan_iter(match=None, count=100):
            if match == "user:*":
                yield b"user:1"
                yield b"user:2"

        mock_redis_client.scan_iter = mock_scan_iter

        cache = DragonflyCache(key_prefix="")

        result = await cache.scan_keys("user:*")

        expected = ["user:1", "user:2"]
        assert result == expected

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @pytest.mark.asyncio
    async def test_scan_keys_with_error(self, mock_pool_from_url, mock_redis_pool):
        """Test scan_keys operation with error."""
        mock_pool_from_url.return_value = mock_redis_pool

        cache = DragonflyCache()

        class MockScanIterError:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RedisError("Scan failed")

        mock_client = AsyncMock()
        mock_client.scan_iter = MagicMock(return_value=MockScanIterError())

        # Set the _client directly
        cache._client = mock_client

        result = await cache.scan_keys("*")

        assert result == []

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_memory_usage_success(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get_memory_usage operation successfully."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.memory_usage.return_value = 1024

        cache = DragonflyCache()

        result = await cache.get_memory_usage("test_key")

        assert result == 1024
        mock_redis_client.memory_usage.assert_called_once_with("test_key")

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_memory_usage_not_supported(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get_memory_usage when not supported."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        # AttributeError when memory_usage command not available
        mock_redis_client.memory_usage.side_effect = AttributeError("Not supported")

        cache = DragonflyCache()

        result = await cache.get_memory_usage("test_key")

        assert result == 0

    @patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url")
    @patch("src.services.cache.dragonfly_cache.redis.Redis")
    @pytest.mark.asyncio
    async def test_get_memory_usage_with_error(
        self, mock_redis_cls, mock_pool_from_url, mock_redis_pool, mock_redis_client
    ):
        """Test get_memory_usage operation with Redis error."""
        mock_pool_from_url.return_value = mock_redis_pool
        mock_redis_cls.return_value = mock_redis_client
        mock_redis_client.memory_usage.side_effect = RedisError("Memory usage failed")

        cache = DragonflyCache()

        result = await cache.get_memory_usage("test_key")

        assert result == 0

    def test_retry_strategy_configuration(self):
        """Test retry strategy configuration."""
        with patch(
            "src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url"
        ) as mock_pool:
            mock_pool.return_value = MagicMock()

            # Test with retry enabled
            DragonflyCache(retry_on_timeout=True, max_retries=5)

            # Verify retry configuration was passed to pool
            call_args = mock_pool.call_args
            retry_param = call_args[1]["retry"]
            assert retry_param is not None

    def test_retry_strategy_disabled(self):
        """Test retry strategy when disabled."""
        with patch(
            "src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url"
        ) as mock_pool:
            mock_pool.return_value = MagicMock()

            # Test with retry disabled
            DragonflyCache(retry_on_timeout=False)

            # Verify no retry configuration
            call_args = mock_pool.call_args
            retry_param = call_args[1]["retry"]
            assert retry_param is None

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        cache = DragonflyCache(enable_compression=False)

        test_cases = [
            "simple string",
            42,
            3.14159,
            True,
            None,
            ["list", "of", "items"],
            {"dict": "value", "nested": {"key": "value"}},
            {"unicode": "测试数据"},
        ]

        for original in test_cases:
            serialized = cache._serialize(original)
            deserialized = cache._deserialize(serialized)
            assert deserialized == original

    def test_compression_roundtrip(self):
        """Test compression and decompression roundtrip."""
        cache = DragonflyCache(enable_compression=True, compression_threshold=10)

        # Large data that should be compressed
        large_data = {"key": "x" * 1000, "other": "data"}

        serialized = cache._serialize(large_data)
        deserialized = cache._deserialize(serialized)

        assert deserialized == large_data
        assert serialized.startswith(b"Z:")  # Should be compressed
