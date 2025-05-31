"""Comprehensive tests for LocalCache service."""

import asyncio
import time

import pytest
from src.services.cache.local_cache import CacheEntry
from src.services.cache.local_cache import LocalCache


@pytest.fixture
def cache():
    """Create LocalCache instance for testing."""
    return LocalCache(max_size=100, default_ttl=300, max_memory_mb=10)


@pytest.fixture
def small_cache():
    """Create small LocalCache for testing eviction."""
    return LocalCache(max_size=3, default_ttl=60, max_memory_mb=1)


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = CacheEntry(value="test_value")
        assert entry.value == "test_value"
        assert entry.expires_at is None

    def test_entry_with_expiration(self):
        """Test entry with expiration time."""
        expires_at = time.time() + 100
        entry = CacheEntry(value="test_value", expires_at=expires_at)
        assert entry.value == "test_value"
        assert entry.expires_at == expires_at

    def test_is_expired_no_expiration(self):
        """Test is_expired when no expiration set."""
        entry = CacheEntry(value="test_value")
        assert entry.is_expired() is False

    def test_is_expired_future(self):
        """Test is_expired with future expiration."""
        entry = CacheEntry(value="test_value", expires_at=time.time() + 100)
        assert entry.is_expired() is False

    def test_is_expired_past(self):
        """Test is_expired with past expiration."""
        entry = CacheEntry(value="test_value", expires_at=time.time() - 100)
        assert entry.is_expired() is True


class TestLocalCacheInitialization:
    """Test cache initialization."""

    def test_cache_initialization(self, cache):
        """Test basic cache initialization."""
        assert cache.max_size == 100
        assert cache.default_ttl == 300
        assert cache.max_memory_bytes == 10 * 1024 * 1024
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_default_values(self):
        """Test cache with default values."""
        cache = LocalCache()
        assert cache.max_size == 1000
        assert cache.default_ttl == 300
        assert cache.max_memory_bytes == 100 * 1024 * 1024

    def test_cache_no_ttl(self):
        """Test cache without default TTL."""
        cache = LocalCache(default_ttl=None)
        assert cache.default_ttl is None

    @pytest.mark.asyncio
    async def test_initialize_cleanup(self, cache):
        """Test initialize and cleanup methods."""
        await cache.initialize()
        # LocalCache doesn't need initialization, but method should work
        assert True

        await cache.cleanup()
        # Should clear cache
        assert len(cache._cache) == 0


class TestBasicOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        # Set value
        success = await cache.set("key1", "value1")
        assert success is True

        # Get value
        result = await cache.get("key1")
        assert result == "value1"
        assert cache._hits == 1

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Test getting missing key."""
        result = await cache.get("missing_key")
        assert result is None
        assert cache._misses == 1

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache):
        """Test set with custom TTL."""
        success = await cache.set("key1", "value1", ttl=1)
        assert success is True

        # Should exist immediately
        result = await cache.get("key1")
        assert result == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_none_ttl(self, cache):
        """Test set with None TTL (no expiration)."""
        success = await cache.set("key1", "value1", ttl=None)
        assert success is True

        # Should not expire
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_overwrites_existing(self, cache):
        """Test that set overwrites existing values."""
        await cache.set("key1", "value1")
        await cache.set("key1", "value2")

        result = await cache.get("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test delete operation."""
        await cache.set("key1", "value1")

        # Delete existing key
        success = await cache.delete("key1")
        assert success is True

        # Verify deleted
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_missing_key(self, cache):
        """Test deleting missing key."""
        success = await cache.delete("missing_key")
        assert success is False

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test exists operation."""
        await cache.set("key1", "value1")

        # Check existing key
        exists = await cache.exists("key1")
        assert exists is True

        # Check missing key
        exists = await cache.exists("missing_key")
        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_expired_key(self, cache):
        """Test exists with expired key."""
        await cache.set("key1", "value1", ttl=0.1)

        # Should exist immediately
        assert await cache.exists("key1") is True

        # Wait for expiration
        await asyncio.sleep(0.2)
        assert await cache.exists("key1") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clear operation."""
        # Add multiple items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Clear cache
        await cache.clear()

        # Verify empty
        assert len(cache._cache) == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None


class TestLRUEviction:
    """Test LRU eviction behavior."""

    @pytest.mark.asyncio
    async def test_lru_eviction(self, small_cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")

        # Access key1 to make it recently used
        await small_cache.get("key1")

        # Add new item - should evict key2 (least recently used)
        await small_cache.set("key4", "value4")

        # Check eviction
        assert await small_cache.get("key1") == "value1"  # Still present
        assert await small_cache.get("key2") is None  # Evicted
        assert await small_cache.get("key3") == "value3"  # Still present
        assert await small_cache.get("key4") == "value4"  # New item

    @pytest.mark.asyncio
    async def test_lru_update_moves_to_end(self, small_cache):
        """Test that updating a key moves it to end (most recent)."""
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")

        # Update key1
        await small_cache.set("key1", "new_value1")

        # Add new item - should evict key2
        await small_cache.set("key4", "value4")

        assert await small_cache.get("key1") == "new_value1"
        assert await small_cache.get("key2") is None  # Evicted
        assert await small_cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_get_updates_lru_order(self, small_cache):
        """Test that get operation updates LRU order."""
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")

        # Access key1 multiple times
        await small_cache.get("key1")
        await small_cache.get("key1")

        # Add new item - should evict key2 (not key1)
        await small_cache.set("key4", "value4")

        assert await small_cache.get("key1") == "value1"
        assert await small_cache.get("key2") is None


class TestMemoryManagement:
    """Test memory management features."""

    @pytest.mark.asyncio
    async def test_memory_estimation(self, cache):
        """Test memory usage estimation."""
        # Add items of different types
        await cache.set("str_key", "a" * 1000)
        await cache.set("list_key", [1, 2, 3, 4, 5])
        await cache.set("dict_key", {"a": 1, "b": 2, "c": 3})

        memory_usage = cache._estimate_memory_usage()
        assert memory_usage > 0
        assert memory_usage < cache.max_memory_bytes

    @pytest.mark.asyncio
    async def test_memory_based_eviction(self, cache):
        """Test eviction based on memory limit."""
        cache.max_memory_bytes = 5000  # Small limit

        # Add large items
        large_value = "x" * 1000
        for i in range(10):
            await cache.set(f"key{i}", large_value)

        # Should have evicted some items to stay under memory limit
        assert len(cache._cache) < 10
        memory_usage = cache._estimate_memory_usage()
        assert memory_usage <= cache.max_memory_bytes * 1.1  # Allow 10% overhead

    def test_get_size_of_different_types(self, cache):
        """Test size estimation for different types."""
        # Test various types
        assert cache._get_size(None) > 0
        assert cache._get_size(42) > 0
        assert cache._get_size(3.14) > 0
        assert cache._get_size("hello") > 0
        assert cache._get_size([1, 2, 3]) > 0
        assert cache._get_size({"a": 1}) > 0
        assert cache._get_size({1, 2, 3}) > 0
        assert cache._get_size((1, 2, 3)) > 0

        # String size should increase with length
        small_str_size = cache._get_size("a")
        large_str_size = cache._get_size("a" * 1000)
        assert large_str_size > small_str_size

    def test_get_size_of_nested_structures(self, cache):
        """Test size estimation for nested structures."""
        nested_dict = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3, {"level4": "value"}]
                }
            }
        }
        size = cache._get_size(nested_dict)
        assert size > 0

        # Custom object
        class CustomObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = [1, 2, 3]

        obj = CustomObj()
        obj_size = cache._get_size(obj)
        assert obj_size > 0


class TestExpirationHandling:
    """Test TTL and expiration handling."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, cache):
        """Test cleanup of expired entries."""
        # Add entries with different TTLs
        await cache.set("key1", "value1", ttl=0.1)
        await cache.set("key2", "value2", ttl=10)
        await cache.set("key3", "value3", ttl=None)  # No expiration

        # Wait for first to expire
        await asyncio.sleep(0.2)

        # Cleanup expired
        cache._cleanup_expired()

        # Check results
        assert "key1" not in cache._cache
        assert "key2" in cache._cache
        assert "key3" in cache._cache

    @pytest.mark.asyncio
    async def test_get_removes_expired(self, cache):
        """Test that get removes expired entries."""
        await cache.set("key1", "value1", ttl=0.1)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Get should return None and remove entry
        result = await cache.get("key1")
        assert result is None
        assert "key1" not in cache._cache

    @pytest.mark.asyncio
    async def test_exists_removes_expired(self, cache):
        """Test that exists removes expired entries."""
        await cache.set("key1", "value1", ttl=0.1)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Exists should return False and remove entry
        exists = await cache.exists("key1")
        assert exists is False
        assert "key1" not in cache._cache

    @pytest.mark.asyncio
    async def test_periodic_cleanup_disabled(self, cache):
        """Test behavior when cleanup is disabled."""
        # Cleanup is called on operations, not periodically
        await cache.set("key1", "value1", ttl=0.1)
        await asyncio.sleep(0.2)

        # Entry still in cache until accessed
        assert "key1" in cache._cache

        # Access triggers cleanup
        await cache.get("key1")
        assert "key1" not in cache._cache


class TestStatistics:
    """Test cache statistics."""

    def test_get_stats(self, cache):
        """Test statistics retrieval."""
        cache._hits = 100
        cache._misses = 50
        cache._evictions = 10
        cache._expired_evictions = 5

        # Add some entries
        for i in range(5):
            cache._cache[f"key{i}"] = CacheEntry(value=f"value{i}")

        stats = cache.get_stats()

        assert stats["hits"] == 100
        assert stats["misses"] == 50
        assert stats["hit_rate"] == pytest.approx(0.667, rel=0.01)
        assert stats["entries"] == 5
        assert stats["evictions"] == 10
        assert stats["expired_evictions"] == 5
        assert stats["memory_usage_mb"] > 0
        assert stats["memory_usage_percent"] > 0

    def test_get_stats_no_requests(self, cache):
        """Test statistics with no requests."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["entries"] == 0


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_sets(self, cache):
        """Test concurrent set operations."""
        async def set_value(key, value):
            await cache.set(key, value)

        # Create concurrent tasks
        tasks = []
        for i in range(10):
            tasks.append(set_value(f"key{i}", f"value{i}"))

        await asyncio.gather(*tasks)

        # Verify all values set
        for i in range(10):
            assert await cache.get(f"key{i}") == f"value{i}"

    @pytest.mark.asyncio
    async def test_concurrent_get_set(self, cache):
        """Test concurrent get and set operations."""
        await cache.set("shared_key", 0)

        async def increment():
            for _ in range(10):
                value = await cache.get("shared_key") or 0
                await cache.set("shared_key", value + 1)
                await asyncio.sleep(0.001)  # Small delay

        # Run multiple incrementers concurrently
        tasks = [increment() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Final value should be consistent
        final_value = await cache.get("shared_key")
        assert isinstance(final_value, int)
        assert final_value >= 10  # At least some increments

    @pytest.mark.asyncio
    async def test_lock_prevents_race_conditions(self, cache):
        """Test that lock prevents race conditions."""
        results = []

        async def critical_section(n):
            async with cache._lock:
                # Simulate critical work
                results.append(n)
                await asyncio.sleep(0.01)
                results.append(n)

        # Run concurrently
        tasks = [critical_section(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Check that operations completed atomically
        for i in range(5):
            first_idx = results.index(i)
            second_idx = results.index(i, first_idx + 1)
            assert second_idx == first_idx + 1  # No interleaving


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_set_none_value(self, cache):
        """Test setting None as value."""
        success = await cache.set("key1", None)
        assert success is True

        result = await cache.get("key1")
        assert result is None

        # But key should exist
        exists = await cache.exists("key1")
        assert exists is True

    @pytest.mark.asyncio
    async def test_empty_string_key(self, cache):
        """Test empty string as key."""
        success = await cache.set("", "value")
        assert success is True

        result = await cache.get("")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_large_value(self, cache):
        """Test storing large values."""
        large_value = "x" * 1_000_000  # 1MB string
        success = await cache.set("large_key", large_value)
        assert success is True

        result = await cache.get("large_key")
        assert result == large_value

    @pytest.mark.asyncio
    async def test_max_size_zero(self):
        """Test cache with max_size=0."""
        cache = LocalCache(max_size=0)

        # Should not store anything
        success = await cache.set("key1", "value1")
        assert success is True  # Operation succeeds

        result = await cache.get("key1")
        assert result is None  # But nothing stored

    @pytest.mark.asyncio
    async def test_negative_ttl(self, cache):
        """Test negative TTL (already expired)."""
        success = await cache.set("key1", "value1", ttl=-1)
        assert success is True

        # Should be immediately expired
        result = await cache.get("key1")
        assert result is None


class TestBatchOperations:
    """Test batch operations that LocalCache might support."""

    @pytest.mark.asyncio
    async def test_get_many(self, cache):
        """Test getting multiple keys at once."""
        # Set up test data
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # LocalCache might not have batch operations by default
        # But we can test sequential gets
        keys = ["key1", "key2", "missing", "key3"]
        results = []
        for key in keys:
            results.append(await cache.get(key))

        assert results == ["value1", "value2", None, "value3"]

    @pytest.mark.asyncio
    async def test_set_many(self, cache):
        """Test setting multiple keys at once."""
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        # Sequential sets
        for key, value in items.items():
            await cache.set(key, value)

        # Verify all set
        for key, expected_value in items.items():
            assert await cache.get(key) == expected_value

    @pytest.mark.asyncio
    async def test_delete_many(self, cache):
        """Test deleting multiple keys at once."""
        # Set up test data
        keys = ["key1", "key2", "key3"]
        for key in keys:
            await cache.set(key, f"value_{key}")

        # Delete all
        for key in keys:
            await cache.delete(key)

        # Verify all deleted
        for key in keys:
            assert await cache.get(key) is None


class TestScanOperations:
    """Test key scanning/pattern matching if supported."""

    @pytest.mark.asyncio
    async def test_scan_keys_simple(self, cache):
        """Test basic key scanning."""
        # Add test data
        await cache.set("user:1", "John")
        await cache.set("user:2", "Jane")
        await cache.set("product:1", "Widget")
        await cache.set("product:2", "Gadget")

        # Scan for user keys
        user_keys = await cache.scan_keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

    @pytest.mark.asyncio
    async def test_scan_keys_wildcard(self, cache):
        """Test wildcard pattern matching."""
        await cache.set("test_key_1", "value1")
        await cache.set("test_key_2", "value2")
        await cache.set("other_key", "value3")

        # Pattern with wildcard
        keys = await cache.scan_keys("test_*")
        assert len(keys) == 2
        assert "test_key_1" in keys
        assert "test_key_2" in keys
        assert "other_key" not in keys

    @pytest.mark.asyncio
    async def test_scan_keys_all(self, cache):
        """Test scanning all keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Get all keys
        all_keys = await cache.scan_keys("*")
        assert len(all_keys) == 3
        assert set(all_keys) == {"key1", "key2", "key3"}

    @pytest.mark.asyncio
    async def test_scan_excludes_expired(self, cache):
        """Test that scan excludes expired keys."""
        await cache.set("key1", "value1", ttl=0.1)
        await cache.set("key2", "value2", ttl=10)

        # Wait for first to expire
        await asyncio.sleep(0.2)

        # Scan should only return non-expired
        keys = await cache.scan_keys("*")
        assert len(keys) == 1
        assert "key2" in keys
        assert "key1" not in keys


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_expire_not_supported(self, cache):
        """Test expire method (not supported in local cache)."""
        await cache.set("key1", "value1")

        # LocalCache doesn't support changing TTL after set
        success = await cache.expire("key1", 60)
        assert success is False

    @pytest.mark.asyncio
    async def test_ttl_not_supported(self, cache):
        """Test ttl method (not supported in local cache)."""
        await cache.set("key1", "value1", ttl=60)

        # LocalCache doesn't track remaining TTL
        ttl = await cache.ttl("key1")
        assert ttl == -1  # Not supported

    def test_pattern_match(self, cache):
        """Test pattern matching utility."""
        # Test exact match
        assert cache._pattern_match("test", "test") is True
        assert cache._pattern_match("test", "other") is False

        # Test wildcard patterns
        assert cache._pattern_match("test*", "test123") is True
        assert cache._pattern_match("test*", "other") is False
        assert cache._pattern_match("*test", "mytest") is True
        assert cache._pattern_match("*test*", "mytesting") is True

        # Test ? wildcard
        assert cache._pattern_match("te?t", "test") is True
        assert cache._pattern_match("te?t", "tent") is True
        assert cache._pattern_match("te?t", "toast") is False

    @pytest.mark.asyncio
    async def test_mget_not_implemented(self, cache):
        """Test mget raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await cache.mget(["key1", "key2"])

    @pytest.mark.asyncio
    async def test_mset_not_implemented(self, cache):
        """Test mset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await cache.mset({"key1": "value1"})


class TestMemorySafety:
    """Test memory safety and resource management."""

    @pytest.mark.asyncio
    async def test_memory_limit_enforced(self):
        """Test that memory limit is enforced."""
        # Create cache with tiny memory limit
        cache = LocalCache(max_size=1000, max_memory_mb=0.001)  # 1KB

        # Try to add large values
        large_value = "x" * 1000  # 1KB per value

        # Add values until memory limit
        added = 0
        for i in range(100):
            success = await cache.set(f"key{i}", large_value)
            if success:
                added += 1

        # Should have stopped adding due to memory limit
        assert added < 100
        assert len(cache._cache) < 100

        # Memory usage should be close to limit
        memory_usage = cache._estimate_memory_usage()
        assert memory_usage <= cache.max_memory_bytes * 1.5  # Allow some overhead

    @pytest.mark.asyncio
    async def test_eviction_frees_memory(self, small_cache):
        """Test that eviction properly frees memory."""
        # Fill cache
        for i in range(3):
            await small_cache.set(f"key{i}", f"value{i}" * 100)

        initial_memory = small_cache._estimate_memory_usage()

        # Force eviction by adding new item
        await small_cache.set("key3", "small")

        final_memory = small_cache._estimate_memory_usage()

        # Memory should decrease after evicting large value
        assert final_memory < initial_memory

    def test_circular_reference_handling(self, cache):
        """Test handling of circular references in size calculation."""
        # Create circular reference
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2

        # Should handle without infinite recursion
        size = cache._get_size(obj1, seen=set())
        assert size > 0
        assert size < 10000  # Reasonable size

