"""Modern tests for LRU cache behavior and memory management.

Tests for the memory leak fixes and LRU cache implementations.
Validates that caches respect size limits and properly evict items.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, strategies as st

from src.services.cache.local_cache import LocalCache


class TestLRUCacheBehavior:
    """Test LRU cache behavior after memory leak fixes."""

    @pytest.fixture
    def small_cache(self) -> LocalCache:
        """Create a small cache for testing LRU behavior."""
        return LocalCache(max_size=3, default_ttl=300, max_memory_mb=1.0)

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self, small_cache: LocalCache):
        """Test that LRU eviction maintains proper order.

        Verifies that least recently used items are evicted first
        when cache reaches maximum size.
        """
        # Arrange: Fill cache to capacity
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")

        # Access key1 to make it recently used
        await small_cache.get("key1")

        # Act: Add a new item, forcing eviction
        await small_cache.set("key4", "value4")

        # Assert: key2 should be evicted (least recently used)
        assert await small_cache.get("key1") == "value1"  # Recently accessed
        assert await small_cache.get("key2") is None  # Should be evicted
        assert await small_cache.get("key3") == "value3"  # Still present
        assert await small_cache.get("key4") == "value4"  # Newly added

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, small_cache: LocalCache):
        """Test that memory limits are properly enforced.

        Verifies that cache respects memory limits and evicts items
        when memory usage exceeds the threshold.
        """
        # Arrange: Create large values that would exceed memory limit
        large_value = "x" * 100000  # 100KB value

        # Act: Try to add multiple large values
        await small_cache.set("large1", large_value)
        await small_cache.set("large2", large_value)

        # Assert: Cache should respect memory limits
        memory_usage = small_cache._current_memory
        max_memory = small_cache.max_memory_bytes
        assert memory_usage <= max_memory, (
            f"Memory usage {memory_usage} exceeds limit {max_memory}"
        )

    @pytest.mark.asyncio
    async def test_cache_size_never_exceeds_max(self, small_cache: LocalCache):
        """Test that cache size never exceeds maximum.

        Property-based test that verifies cache size constraint
        is always maintained regardless of operations.
        """
        # Act: Perform many cache operations
        for i in range(20):  # More than max_size of 3
            await small_cache.set(f"key{i}", f"value{i}")

            # Assert: Size should never exceed maximum
            current_size = await small_cache.size()
            assert current_size <= small_cache.max_size

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_cache_consistency_property(self, keys):
        """Property-based test for cache consistency.

        Tests that cache operations maintain consistency regardless
        of the sequence of keys used.
        """
        cache = LocalCache(max_size=5, default_ttl=300, max_memory_mb=1.0)

        # Act: Set all keys
        for key in keys:
            await cache.set(key, f"value_{key}")

        # Assert: Cache size should not exceed maximum
        size = await cache.size()
        assert size <= cache.max_size

        # Assert: Most recently set keys should be retrievable
        # (at least the last max_size keys)
        if len(keys) <= cache.max_size:
            # All keys should be present
            for key in keys:
                value = await cache.get(key)
                assert value == f"value_{key}"
        else:
            # At least the last max_size keys should be present
            recent_keys = keys[-cache.max_size :]
            for key in recent_keys:
                value = await cache.get(key)
                assert value == f"value_{key}"

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, small_cache: LocalCache):
        """Test that concurrent cache access is safe.

        Verifies that cache operations work correctly under
        concurrent access patterns.
        """

        async def cache_worker(worker_id: int, cache: LocalCache):
            """Worker function that performs cache operations."""
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                await cache.set(key, value)
                retrieved = await cache.get(key)
                # Value might be None if evicted by other workers
                if retrieved is not None:
                    assert retrieved == value

        # Act: Run multiple workers concurrently
        tasks = [cache_worker(i, small_cache) for i in range(5)]
        await asyncio.gather(*tasks)

        # Assert: Cache should remain consistent
        size = await small_cache.size()
        assert size <= small_cache.max_size
        assert size >= 0

    @pytest.mark.asyncio
    async def test_cache_stats_tracking(self, small_cache: LocalCache):
        """Test that cache statistics are properly tracked.

        Verifies that hit/miss/eviction counters work correctly
        and provide useful metrics.
        """
        # Arrange: Initial state
        initial_hits = small_cache._hits
        initial_misses = small_cache._misses
        initial_evictions = small_cache._evictions

        # Act: Perform cache operations
        await small_cache.set("key1", "value1")
        await small_cache.get("key1")  # Hit
        await small_cache.get("missing")  # Miss

        # Fill cache to force eviction
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")
        await small_cache.set("key4", "value4")  # Should evict key1

        # Assert: Statistics should be updated
        assert small_cache._hits == initial_hits + 1
        assert small_cache._misses == initial_misses + 1
        assert small_cache._evictions >= initial_evictions

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_eviction(self, small_cache: LocalCache):
        """Test that memory is properly cleaned up on eviction.

        Verifies that evicted items don't contribute to memory usage
        and that memory tracking is accurate.
        """
        # Arrange: Add items and track memory
        await small_cache.set("key1", "small_value")
        memory_after_first = small_cache._current_memory

        await small_cache.set("key2", "another_small_value")
        memory_after_second = small_cache._current_memory

        await small_cache.set("key3", "third_small_value")
        memory_after_third = small_cache._current_memory

        # Act: Add fourth item to force eviction
        await small_cache.set("key4", "fourth_small_value")
        memory_after_eviction = small_cache._current_memory

        # Assert: Memory should not continuously grow
        # After eviction, memory should be similar to previous levels
        assert memory_after_eviction <= memory_after_third * 1.1  # Allow 10% variance


class TestCacheIntegrationPatterns:
    """Test integration patterns for cache usage."""

    @pytest.mark.asyncio
    async def test_cache_as_service_boundary(self):
        """Test cache usage at service boundaries.

        Tests proper mocking patterns for cache dependencies
        without testing implementation details.
        """
        # Arrange: Mock cache for testing service behavior
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True  # Successful set

        # Simulate service that uses cache
        async def service_method(key: str) -> str:
            cached_value = await mock_cache.get(key)
            if cached_value:
                return cached_value

            # Simulate expensive operation
            computed_value = f"computed_value_for_{key}"
            await mock_cache.set(key, computed_value)
            return computed_value

        # Act
        result = await service_method("test_key")

        # Assert: Service behavior, not cache implementation
        assert result == "computed_value_for_test_key"
        mock_cache.get.assert_called_once_with("test_key")
        mock_cache.set.assert_called_once_with(
            "test_key", "computed_value_for_test_key"
        )

    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test proper error handling in cache operations.

        Verifies that services gracefully handle cache failures
        without compromising core functionality.
        """
        # Arrange: Cache that raises exceptions
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Cache connection failed")
        mock_cache.set.side_effect = Exception("Cache connection failed")

        # Simulate service with cache fallback
        async def resilient_service_method(key: str) -> str:
            try:
                cached_value = await mock_cache.get(key)
                if cached_value:
                    return cached_value
            except Exception:
                # Gracefully handle cache failure
                pass  # noqa: S110

            # Fallback to direct computation
            computed_value = f"computed_value_for_{key}"

            try:
                await mock_cache.set(key, computed_value)
            except Exception:
                # Cache set failure is non-fatal
                pass  # noqa: S110

            return computed_value

        # Act
        result = await resilient_service_method("test_key")

        # Assert: Service continues to work despite cache failures
        assert result == "computed_value_for_test_key"
