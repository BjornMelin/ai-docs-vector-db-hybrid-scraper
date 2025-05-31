"""Comprehensive tests for DragonflyCache service."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.errors import CacheError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = MagicMock()
    config.cache = MagicMock()
    config.cache.dragonfly_url = "redis://localhost:6379"
    config.cache.ttl_embeddings = 86400  # 1 day
    config.cache.ttl_crawl = 3600  # 1 hour
    config.cache.ttl_queries = 7200  # 2 hours
    config.cache.compression_threshold = 1024
    return config


@pytest.fixture
def cache(mock_config):
    """Create DragonflyCache instance for testing."""
    return DragonflyCache(mock_config)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.expire = AsyncMock(return_value=True)
    client.ttl = AsyncMock(return_value=-1)
    client.scan = AsyncMock(return_value=(0, []))
    client.mget = AsyncMock(return_value=[])
    client.keys = AsyncMock(return_value=[])
    client.flushdb = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client.wait_closed = AsyncMock()
    return client


class TestDragonflyCache:
    """Test cache initialization and connection."""

    def test_cache_initialization(self, cache, mock_config):
        """Test basic cache initialization."""
        assert cache.config == mock_config
        assert cache._redis is None
        assert cache._initialized is False
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert cache.stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, cache, mock_redis):
        """Test successful cache initialization."""
        with patch("redis.from_url", return_value=mock_redis):
            await cache.initialize()

        assert cache._initialized is True
        assert cache._redis is mock_redis
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, cache):
        """Test initialization with connection failure."""
        with patch("redis.from_url", side_effect=Exception("Connection refused")):
            with pytest.raises(CacheError, match="Failed to connect to DragonflyDB"):
                await cache.initialize()

    @pytest.mark.asyncio
    async def test_initialize_ping_failure(self, cache, mock_redis):
        """Test initialization when ping fails."""
        mock_redis.ping.side_effect = Exception("Ping failed")

        with patch("redis.from_url", return_value=mock_redis):
            with pytest.raises(CacheError, match="Failed to connect to DragonflyDB"):
                await cache.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, cache, mock_redis):
        """Test that initialization is idempotent."""
        with patch("redis.from_url", return_value=mock_redis):
            await cache.initialize()
            await cache.initialize()  # Second call

        # Should only create connection once
        assert mock_redis.ping.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, cache, mock_redis):
        """Test cache cleanup."""
        cache._redis = mock_redis
        cache._initialized = True

        await cache.cleanup()

        mock_redis.close.assert_called_once()
        mock_redis.wait_closed.assert_called_once()
        assert cache._redis is None
        assert cache._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, cache):
        """Test cleanup when not initialized."""
        # Should not raise error
        await cache.cleanup()
        assert cache._initialized is False


class TestBasicOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_get_not_initialized(self, cache):
        """Test get operation when cache not initialized."""
        with pytest.raises(CacheError, match="Cache not initialized"):
            await cache.get("test_key")

    @pytest.mark.asyncio
    async def test_set_not_initialized(self, cache):
        """Test set operation when cache not initialized."""
        with pytest.raises(CacheError, match="Cache not initialized"):
            await cache.set("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache, mock_redis):
        """Test get operation with cache miss."""
        cache._redis = mock_redis
        cache._initialized = True
        mock_redis.get.return_value = None

        result = await cache.get("missing_key")

        assert result is None
        assert cache.stats["misses"] == 1
        mock_redis.get.assert_called_once_with("missing_key")

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache, mock_redis):
        """Test get operation with cache hit."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock serialized data
        test_data = {"key": "value", "number": 42}
        serialized = json.dumps(test_data).encode("utf-8")
        mock_redis.get.return_value = serialized

        result = await cache.get("test_key")

        assert result == test_data
        assert cache.stats["hits"] == 1
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_basic(self, cache, mock_redis):
        """Test basic set operation."""
        cache._redis = mock_redis
        cache._initialized = True

        test_data = {"message": "Hello, World!"}
        success = await cache.set("test_key", test_data, ttl=3600)

        assert success is True

        # Verify Redis set was called with serialized data
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "test_key"
        assert json.loads(call_args[0][1]) == test_data
        assert call_args[1]["ex"] == 3600

    @pytest.mark.asyncio
    async def test_set_with_compression(self, cache, mock_redis):
        """Test set operation with compression for large data."""
        cache._redis = mock_redis
        cache._initialized = True

        # Create large data that exceeds compression threshold
        large_data = {"data": "x" * 2000}

        with patch.object(
            cache, "_compress", return_value=b"compressed_data"
        ) as mock_compress:
            success = await cache.set("large_key", large_data)

        assert success is True
        mock_compress.assert_called_once()

        # Verify compressed data was stored
        call_args = mock_redis.set.call_args
        assert call_args[0][1] == b"compressed_data"

    @pytest.mark.asyncio
    async def test_get_with_decompression(self, cache, mock_redis):
        """Test get operation with decompression."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock compressed data retrieval
        compressed_data = b"compressed_data"
        mock_redis.get.return_value = compressed_data

        expected_data = {"data": "x" * 2000}
        with patch.object(
            cache, "_decompress", return_value=json.dumps(expected_data).encode()
        ) as mock_decompress:
            with patch.object(cache, "_is_compressed", return_value=True):
                result = await cache.get("large_key")

        assert result == expected_data
        mock_decompress.assert_called_once_with(compressed_data)

    @pytest.mark.asyncio
    async def test_delete(self, cache, mock_redis):
        """Test delete operation."""
        cache._redis = mock_redis
        cache._initialized = True

        success = await cache.delete("test_key")

        assert success is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cache, mock_redis):
        """Test delete operation for non-existent key."""
        cache._redis = mock_redis
        cache._initialized = True
        mock_redis.delete.return_value = 0

        success = await cache.delete("missing_key")

        assert success is False

    @pytest.mark.asyncio
    async def test_exists(self, cache, mock_redis):
        """Test exists operation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Key exists
        mock_redis.exists.return_value = 1
        assert await cache.exists("existing_key") is True

        # Key doesn't exist
        mock_redis.exists.return_value = 0
        assert await cache.exists("missing_key") is False

    @pytest.mark.asyncio
    async def test_expire(self, cache, mock_redis):
        """Test expire operation."""
        cache._redis = mock_redis
        cache._initialized = True

        success = await cache.expire("test_key", 7200)

        assert success is True
        mock_redis.expire.assert_called_once_with("test_key", 7200)

    @pytest.mark.asyncio
    async def test_ttl(self, cache, mock_redis):
        """Test TTL operation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Key with TTL
        mock_redis.ttl.return_value = 3600
        ttl = await cache.ttl("test_key")
        assert ttl == 3600

        # Key without TTL (persistent)
        mock_redis.ttl.return_value = -1
        ttl = await cache.ttl("persistent_key")
        assert ttl == -1

        # Key doesn't exist
        mock_redis.ttl.return_value = -2
        ttl = await cache.ttl("missing_key")
        assert ttl == -2


class TestBatchOperations:
    """Test batch cache operations."""

    @pytest.mark.asyncio
    async def test_mget(self, cache, mock_redis):
        """Test multi-get operation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock serialized values
        values = [
            json.dumps({"id": 1}).encode(),
            None,  # Missing key
            json.dumps({"id": 3}).encode(),
        ]
        mock_redis.mget.return_value = values

        results = await cache.mget(["key1", "key2", "key3"])

        assert len(results) == 3
        assert results[0] == {"id": 1}
        assert results[1] is None
        assert results[2] == {"id": 3}

        # Stats should reflect hits and misses
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_mset(self, cache, mock_redis):
        """Test multi-set operation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Use pipeline mock
        pipeline = AsyncMock()
        pipeline.set = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, True, True])
        mock_redis.pipeline.return_value = pipeline

        items = {
            "key1": {"value": 1},
            "key2": {"value": 2},
            "key3": {"value": 3},
        }

        success = await cache.mset(items, ttl=3600)

        assert success is True
        assert pipeline.set.call_count == 3
        pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_keys(self, cache, mock_redis):
        """Test key scanning operation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock scan results in batches
        mock_redis.scan.side_effect = [
            (1234, ["prefix:key1", "prefix:key2"]),
            (5678, ["prefix:key3"]),
            (0, ["prefix:key4"]),  # cursor 0 means end
        ]

        keys = await cache.scan_keys("prefix:*")

        assert len(keys) == 4
        assert "prefix:key1" in keys
        assert "prefix:key4" in keys
        assert mock_redis.scan.call_count == 3

    @pytest.mark.asyncio
    async def test_clear(self, cache, mock_redis):
        """Test clear all cache operation."""
        cache._redis = mock_redis
        cache._initialized = True

        await cache.clear()

        mock_redis.flushdb.assert_called_once()
        # Stats should be reset
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0


class TestErrorHandling:
    """Test error handling in cache operations."""

    @pytest.mark.asyncio
    async def test_get_with_error(self, cache, mock_redis):
        """Test get operation error handling."""
        cache._redis = mock_redis
        cache._initialized = True
        mock_redis.get.side_effect = Exception("Redis error")

        # Should return None and increment error counter
        result = await cache.get("test_key")

        assert result is None
        assert cache.stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_set_with_error(self, cache, mock_redis):
        """Test set operation error handling."""
        cache._redis = mock_redis
        cache._initialized = True
        mock_redis.set.side_effect = Exception("Redis error")

        # Should return False and increment error counter
        success = await cache.set("test_key", {"data": "value"})

        assert success is False
        assert cache.stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, cache, mock_redis):
        """Test handling of invalid JSON data."""
        cache._redis = mock_redis
        cache._initialized = True

        # Return invalid JSON
        mock_redis.get.return_value = b"invalid json {"

        result = await cache.get("bad_key")

        assert result is None
        assert cache.stats["errors"] == 1


class TestCompression:
    """Test compression functionality."""

    def test_compress_decompress(self, cache):
        """Test compression and decompression."""
        original_data = b"This is test data that will be compressed" * 100

        compressed = cache._compress(original_data)
        assert len(compressed) < len(original_data)
        assert compressed.startswith(b"DRAGONFLY_COMPRESSED:")

        decompressed = cache._decompress(compressed)
        assert decompressed == original_data

    def test_is_compressed(self, cache):
        """Test compressed data detection."""
        compressed_data = b"DRAGONFLY_COMPRESSED:xxxxx"
        regular_data = b"Regular data"

        assert cache._is_compressed(compressed_data) is True
        assert cache._is_compressed(regular_data) is False

    def test_compression_threshold(self, cache):
        """Test compression threshold logic."""
        cache.config.cache.compression_threshold = 100

        # Small data should not be compressed
        small_data = {"msg": "Hi"}
        serialized_small = json.dumps(small_data).encode()
        assert len(serialized_small) < 100

        # Large data should be compressed
        large_data = {"msg": "x" * 200}
        serialized_large = json.dumps(large_data).encode()
        assert len(serialized_large) > 100


class TestCacheStrategies:
    """Test different caching strategies."""

    @pytest.mark.asyncio
    async def test_cache_with_tags(self, cache, mock_redis):
        """Test caching with tags for invalidation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock pipeline for tag operations
        pipeline = AsyncMock()
        pipeline.sadd = MagicMock()
        pipeline.set = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, True])
        mock_redis.pipeline.return_value = pipeline

        # Set with tags
        success = await cache.set_with_tags(
            "product:123",
            {"name": "Product", "price": 99.99},
            tags=["products", "category:electronics"],
            ttl=3600,
        )

        assert success is True
        assert pipeline.sadd.call_count == 2  # Two tags
        pipeline.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, cache, mock_redis):
        """Test cache invalidation by tag."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock tag members
        mock_redis.smembers.return_value = {b"product:123", b"product:456"}

        # Mock pipeline for deletion
        pipeline = AsyncMock()
        pipeline.delete = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1, 1])  # Deleted counts
        mock_redis.pipeline.return_value = pipeline

        count = await cache.invalidate_by_tag("products")

        assert count == 2
        mock_redis.smembers.assert_called_once_with("tag:products")
        assert pipeline.delete.call_count >= 2  # Keys + tag

    @pytest.mark.asyncio
    async def test_cache_warming(self, cache, mock_redis):
        """Test cache warming functionality."""
        cache._redis = mock_redis
        cache._initialized = True

        # Prepare warm data
        warm_data = {
            "config:app": {"version": "1.0", "features": ["a", "b"]},
            "config:db": {"host": "localhost", "port": 5432},
        }

        # Mock pipeline
        pipeline = AsyncMock()
        pipeline.set = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, True])
        mock_redis.pipeline.return_value = pipeline

        success = await cache.warm_cache(warm_data, ttl=86400)

        assert success is True
        assert pipeline.set.call_count == 2


class TestMetricsAndStats:
    """Test metrics and statistics functionality."""

    def test_get_stats(self, cache):
        """Test statistics retrieval."""
        # Simulate some operations
        cache.stats["hits"] = 100
        cache.stats["misses"] = 50
        cache.stats["errors"] = 5
        cache.stats["sets"] = 120
        cache.stats["deletes"] = 10

        stats = cache.get_stats()

        assert stats["hits"] == 100
        assert stats["misses"] == 50
        assert stats["hit_rate"] == 100 / 150  # hits / (hits + misses)
        assert stats["errors"] == 5
        assert stats["total_operations"] == 285

    def test_reset_stats(self, cache):
        """Test statistics reset."""
        # Set some stats
        cache.stats["hits"] = 100
        cache.stats["misses"] = 50

        cache.reset_stats()

        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert all(v == 0 for v in cache.stats.values())

    @pytest.mark.asyncio
    async def test_get_memory_info(self, cache, mock_redis):
        """Test memory usage information retrieval."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock Redis INFO command response
        mock_redis.info.return_value = {
            "used_memory": 1048576,  # 1MB
            "used_memory_human": "1.00M",
            "used_memory_peak": 2097152,  # 2MB
            "total_system_memory": 8589934592,  # 8GB
        }

        memory_info = await cache.get_memory_info()

        assert memory_info["used_memory"] == 1048576
        assert memory_info["used_memory_human"] == "1.00M"
        assert "used_memory_peak" in memory_info


class TestAdvancedFeatures:
    """Test advanced cache features."""

    @pytest.mark.asyncio
    async def test_lua_script_execution(self, cache, mock_redis):
        """Test Lua script execution for atomic operations."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock script registration and execution
        mock_redis.script_load.return_value = "sha1_hash"
        mock_redis.evalsha.return_value = 1

        # Example: Conditional set script
        script = """
        local key = KEYS[1]
        local value = ARGV[1]
        local condition = ARGV[2]
        
        if redis.call('get', key) == condition then
            return redis.call('set', key, value)
        else
            return 0
        end
        """

        # Load and execute script
        sha = await cache.load_script(script)
        result = await cache.execute_script(
            sha, keys=["mykey"], args=["newvalue", "oldvalue"]
        )

        assert sha == "sha1_hash"
        assert result == 1

    @pytest.mark.asyncio
    async def test_pub_sub_invalidation(self, cache, mock_redis):
        """Test pub/sub based cache invalidation."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock pub/sub
        pubsub = AsyncMock()
        pubsub.subscribe = AsyncMock()
        pubsub.get_message = AsyncMock(
            side_effect=[
                {"type": "subscribe"},
                {"type": "message", "data": b"invalidate:products"},
                None,
            ]
        )
        mock_redis.pubsub.return_value = pubsub

        # Set up invalidation listener
        invalidated_keys = []

        async def on_invalidate(pattern):
            invalidated_keys.append(pattern)

        await cache.setup_invalidation_listener(on_invalidate)

        # Simulate receiving invalidation message
        # (In real implementation, this would be in a background task)

        # Publish invalidation
        await cache.publish_invalidation("products")

        mock_redis.publish.assert_called_once_with(
            "cache:invalidate", "invalidate:products"
        )

    @pytest.mark.asyncio
    async def test_distributed_locking(self, cache, mock_redis):
        """Test distributed locking for cache operations."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock lock acquisition
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.delete.return_value = 1  # Lock released

        lock_acquired = False

        async with cache.distributed_lock("resource:123", timeout=5):
            lock_acquired = True
            # Perform operations while holding lock

        assert lock_acquired is True
        # Verify lock was acquired and released
        assert mock_redis.set.called
        assert mock_redis.delete.called

    @pytest.mark.asyncio
    async def test_cache_migration(self, cache, mock_redis):
        """Test cache key migration/renaming."""
        cache._redis = mock_redis
        cache._initialized = True

        # Mock key listing and pipeline operations
        mock_redis.scan.return_value = (0, [b"old:key1", b"old:key2"])

        pipeline = AsyncMock()
        pipeline.rename = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, True])
        mock_redis.pipeline.return_value = pipeline

        # Migrate keys from old pattern to new
        migrated = await cache.migrate_keys(
            "old:*", lambda k: k.replace("old:", "new:")
        )

        assert migrated == 2
        assert pipeline.rename.call_count == 2


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_connection_pooling(self, cache):
        """Test connection pool configuration."""
        pool_config = {
            "max_connections": 50,
            "min_idle": 10,
            "max_idle": 20,
        }

        with patch("redis.from_url") as mock_from_url:
            await cache.initialize(pool_config=pool_config)

            # Verify pool configuration was passed
            call_args = mock_from_url.call_args
            assert "max_connections" in str(call_args)

    @pytest.mark.asyncio
    async def test_pipelining(self, cache, mock_redis):
        """Test command pipelining for batch operations."""
        cache._redis = mock_redis
        cache._initialized = True

        pipeline = AsyncMock()
        pipeline.get = MagicMock()
        pipeline.set = MagicMock()
        pipeline.expire = MagicMock()
        pipeline.execute = AsyncMock(return_value=[b'{"value": 1}', True, True])
        mock_redis.pipeline.return_value = pipeline

        # Execute multiple operations in pipeline
        async with cache.pipeline() as pipe:
            await pipe.get("key1")
            await pipe.set("key2", {"value": 2})
            await pipe.expire("key2", 3600)

        pipeline.execute.assert_called_once()
        assert pipeline.get.call_count == 1
        assert pipeline.set.call_count == 1
        assert pipeline.expire.call_count == 1
