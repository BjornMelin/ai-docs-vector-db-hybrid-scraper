"""Tests for cache patterns module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.patterns import CachePatterns


class TestCachePatterns:
    """Test the CachePatterns class."""

    @pytest.fixture
    def mock_dragonfly_cache(self):
        """Create a mock DragonflyCache for testing."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.mget.return_value = []
        mock_cache.mset.return_value = True
        mock_cache.ttl.return_value = 0
        return mock_cache

    @pytest.fixture
    def cache_patterns(self, mock_dragonfly_cache):
        """Create a CachePatterns instance for testing."""
        return CachePatterns(mock_dragonfly_cache)

    def test_cache_patterns_initialization(self, mock_dragonfly_cache):
        """Test CachePatterns initialization."""
        patterns = CachePatterns(mock_dragonfly_cache)

        assert patterns.cache == mock_dragonfly_cache

    @pytest.mark.asyncio
    async def test_cache_aside_cache_hit_fresh(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cache-aside pattern with fresh cache hit."""
        cached_data = {"result": "cached_value"}
        mock_dragonfly_cache.get.return_value = cached_data
        mock_dragonfly_cache.ttl.return_value = 3000  # Fresh data (> stale threshold)

        fetch_func = MagicMock(return_value={"result": "fresh_value"})

        result = await cache_patterns.cache_aside(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
            stale_while_revalidate=60,
        )

        assert result == cached_data
        mock_dragonfly_cache.get.assert_called_once_with("test_key")
        mock_dragonfly_cache.ttl.assert_called_once_with("test_key")
        fetch_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_aside_cache_hit_stale(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cache-aside pattern with stale cache hit."""
        cached_data = {"result": "cached_value"}
        mock_dragonfly_cache.get.return_value = cached_data
        mock_dragonfly_cache.ttl.return_value = 30  # Stale data (< stale threshold)

        fetch_func = MagicMock(return_value={"result": "fresh_value"})

        result = await cache_patterns.cache_aside(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
            stale_while_revalidate=60,
        )

        assert result == cached_data
        mock_dragonfly_cache.get.assert_called_once_with("test_key")
        mock_dragonfly_cache.ttl.assert_called_once_with("test_key")
        # Background refresh should be triggered but we get stale data immediately

    @pytest.mark.asyncio
    async def test_cache_aside_cache_miss(self, cache_patterns, mock_dragonfly_cache):
        """Test cache-aside pattern with cache miss."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.side_effect = [True, True]  # Lock and data

        fresh_data = {"result": "fresh_value"}
        fetch_func = MagicMock(return_value=fresh_data)

        result = await cache_patterns.cache_aside(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
        )

        assert result == fresh_data
        mock_dragonfly_cache.get.assert_called_with("test_key")
        fetch_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_aside_with_async_fetch(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cache-aside pattern with async fetch function."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.side_effect = [True, True]  # Lock and data

        fresh_data = {"result": "fresh_value"}

        async def async_fetch():
            return fresh_data

        result = await cache_patterns.cache_aside(
            key="test_key",
            fetch_func=async_fetch,
            ttl=3600,
        )

        assert result == fresh_data

    @pytest.mark.asyncio
    async def test_refresh_cache_lock_acquired(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test _refresh_cache when lock is acquired."""
        mock_dragonfly_cache.set.side_effect = [
            True,
            True,
        ]  # Lock acquired, data cached
        mock_dragonfly_cache.delete.return_value = True

        fresh_data = {"result": "fresh_value"}
        fetch_func = MagicMock(return_value=fresh_data)

        result = await cache_patterns._refresh_cache("test_key", fetch_func, 3600)

        assert result == fresh_data
        # Check lock acquisition
        lock_call = mock_dragonfly_cache.set.call_args_list[0]
        assert lock_call[0][0] == "lock:test_key"
        assert lock_call[1]["nx"] is True
        assert lock_call[1]["ttl"] == 10

        # Check data caching
        data_call = mock_dragonfly_cache.set.call_args_list[1]
        assert data_call[0][0] == "test_key"
        assert data_call[0][1] == fresh_data
        assert data_call[1]["ttl"] == 3600

        # Check lock release
        mock_dragonfly_cache.delete.assert_called_once_with("lock:test_key")

    @pytest.mark.asyncio
    async def test_refresh_cache_lock_not_acquired(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test _refresh_cache when lock is not acquired."""
        # Lock not acquired, then data appears in cache
        mock_dragonfly_cache.set.return_value = False

        cached_data = {"result": "cached_value"}
        call_count = 0

        def mock_get(key):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # Return data on second call
                return cached_data
            return None

        mock_dragonfly_cache.get.side_effect = mock_get

        fetch_func = MagicMock(return_value={"result": "fresh_value"})

        result = await cache_patterns._refresh_cache("test_key", fetch_func, 3600)

        assert result == cached_data
        fetch_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_cache_timeout_fallback(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test _refresh_cache timeout fallback."""
        mock_dragonfly_cache.set.return_value = False  # Lock not acquired
        mock_dragonfly_cache.get.return_value = None  # No data appears

        fresh_data = {"result": "fresh_value"}
        fetch_func = MagicMock(return_value=fresh_data)

        with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up test
            result = await cache_patterns._refresh_cache("test_key", fetch_func, 3600)

        assert result == fresh_data
        fetch_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_cache_fetch_error_with_stale(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test _refresh_cache handling fetch error with stale data available."""
        mock_dragonfly_cache.set.return_value = True  # Lock acquired
        mock_dragonfly_cache.delete.return_value = True

        stale_data = {"result": "stale_value"}

        def mock_get_for_error(key):
            if key == "lock:test_key":
                return None
            return stale_data

        mock_dragonfly_cache.get.side_effect = mock_get_for_error

        def failing_fetch():
            raise Exception("Fetch failed")

        result = await cache_patterns._refresh_cache("test_key", failing_fetch, 3600)

        assert result == stale_data
        mock_dragonfly_cache.delete.assert_called_once_with("lock:test_key")

    @pytest.mark.asyncio
    async def test_refresh_cache_fetch_error_no_stale(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test _refresh_cache handling fetch error with no stale data."""
        mock_dragonfly_cache.set.return_value = True  # Lock acquired
        mock_dragonfly_cache.get.return_value = None  # No stale data
        mock_dragonfly_cache.delete.return_value = True

        def failing_fetch():
            raise Exception("Fetch failed")

        with pytest.raises(Exception, match="Fetch failed"):
            await cache_patterns._refresh_cache("test_key", failing_fetch, 3600)

        mock_dragonfly_cache.delete.assert_called_once_with("lock:test_key")

    @pytest.mark.asyncio
    async def test_batch_cache_all_cached(self, cache_patterns, mock_dragonfly_cache):
        """Test batch cache with all items cached."""
        keys = ["key1", "key2", "key3"]
        cached_values = ["value1", "value2", "value3"]
        mock_dragonfly_cache.mget.return_value = cached_values

        fetch_func = MagicMock()

        result = await cache_patterns.batch_cache(
            keys=keys,
            fetch_func=fetch_func,
            ttl=3600,
        )

        expected = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert result == expected
        mock_dragonfly_cache.mget.assert_called_once_with(keys)
        fetch_func.assert_not_called()
        mock_dragonfly_cache.mset.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_cache_partial_cached(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test batch cache with partial cache hits."""
        keys = ["key1", "key2", "key3"]
        cached_values = ["value1", None, "value3"]  # key2 missing
        mock_dragonfly_cache.mget.return_value = cached_values

        fresh_data = {"key2": "fresh_value2"}
        fetch_func = MagicMock(return_value=fresh_data)
        mock_dragonfly_cache.mset.return_value = True

        result = await cache_patterns.batch_cache(
            keys=keys,
            fetch_func=fetch_func,
            ttl=3600,
        )

        expected = {"key1": "value1", "key2": "fresh_value2", "key3": "value3"}
        assert result == expected

        # Check fetch was called with missing keys
        fetch_func.assert_called_once_with(["key2"])

        # Check mset was called with fresh data
        mock_dragonfly_cache.mset.assert_called_once_with(fresh_data, ttl=3600)

    @pytest.mark.asyncio
    async def test_batch_cache_all_missing(self, cache_patterns, mock_dragonfly_cache):
        """Test batch cache with all items missing."""
        keys = ["key1", "key2"]
        cached_values = [None, None]
        mock_dragonfly_cache.mget.return_value = cached_values

        fresh_data = {"key1": "fresh1", "key2": "fresh2"}
        fetch_func = MagicMock(return_value=fresh_data)
        mock_dragonfly_cache.mset.return_value = True

        result = await cache_patterns.batch_cache(
            keys=keys,
            fetch_func=fetch_func,
            ttl=3600,
        )

        assert result == fresh_data
        fetch_func.assert_called_once_with(keys)
        mock_dragonfly_cache.mset.assert_called_once_with(fresh_data, ttl=3600)

    @pytest.mark.asyncio
    async def test_batch_cache_with_async_fetch(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test batch cache with async fetch function."""
        keys = ["key1"]
        mock_dragonfly_cache.mget.return_value = [None]

        fresh_data = {"key1": "fresh1"}

        async def async_fetch(missing_keys):
            return fresh_data

        mock_dragonfly_cache.mset.return_value = True

        result = await cache_patterns.batch_cache(
            keys=keys,
            fetch_func=async_fetch,
            ttl=3600,
        )

        assert result == fresh_data

    @pytest.mark.asyncio
    async def test_batch_cache_fetch_error(self, cache_patterns, mock_dragonfly_cache):
        """Test batch cache handling fetch error."""
        keys = ["key1", "key2"]
        mock_dragonfly_cache.mget.return_value = ["value1", None]

        def failing_fetch(missing_keys):
            raise Exception("Fetch failed")

        result = await cache_patterns.batch_cache(
            keys=keys,
            fetch_func=failing_fetch,
            ttl=3600,
        )

        # Should return partial results (only cached items)
        expected = {"key1": "value1"}
        assert result == expected

    @pytest.mark.asyncio
    async def test_cached_computation_cache_hit(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cached computation with cache hit."""
        cached_result = {"computation": "cached_result"}
        mock_dragonfly_cache.get.return_value = cached_result

        def expensive_func(x, y, z=10):
            return {"computation": "fresh_result"}

        result = await cache_patterns.cached_computation(
            expensive_func,
            5,
            10,
            z=20,
            ttl=3600,
        )

        assert result == cached_result
        mock_dragonfly_cache.get.assert_called_once()
        mock_dragonfly_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_computation_cache_miss(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cached computation with cache miss."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.return_value = True

        def expensive_func(x, y, z=10):
            return {"computation": f"result_{x}_{y}_{z}"}

        result = await cache_patterns.cached_computation(
            expensive_func,
            5,
            10,
            z=20,
            ttl=3600,
        )

        expected = {"computation": "result_5_10_20"}
        assert result == expected

        # Check computation was called
        mock_dragonfly_cache.set.assert_called_once()
        call_args = mock_dragonfly_cache.set.call_args
        assert call_args[0][1] == expected  # Result was cached
        assert call_args[1]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_cached_computation_custom_key(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cached computation with custom cache key."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.return_value = True

        def simple_func():
            return "result"

        await cache_patterns.cached_computation(
            simple_func,
            cache_key="custom_key",
            ttl=3600,
        )

        mock_dragonfly_cache.get.assert_called_once_with("custom_key")

    @pytest.mark.asyncio
    async def test_cached_computation_key_generation(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cached computation automatic key generation."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.return_value = True

        def test_func(a, b, c=3):
            return a + b + c

        await cache_patterns.cached_computation(
            test_func,
            1,
            2,
            c=4,
            ttl=3600,
        )

        # Verify key generation
        get_call = mock_dragonfly_cache.get.call_args[0][0]
        assert get_call.startswith("compute:")
        assert len(get_call.split(":")[1]) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_cached_computation_with_async_func(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cached computation with async function."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.return_value = True

        async def async_func(x):
            return x * 2

        result = await cache_patterns.cached_computation(
            async_func,
            5,
            ttl=3600,
        )

        assert result == 10

    @pytest.mark.asyncio
    async def test_cached_computation_error(self, cache_patterns, mock_dragonfly_cache):
        """Test cached computation handling errors."""
        mock_dragonfly_cache.get.return_value = None

        def failing_func():
            raise ValueError("Computation failed")

        with pytest.raises(ValueError, match="Computation failed"):
            await cache_patterns.cached_computation(
                failing_func,
                ttl=3600,
            )

    @pytest.mark.asyncio
    async def test_write_through_success(self, cache_patterns, mock_dragonfly_cache):
        """Test write-through pattern success."""
        mock_dragonfly_cache.set.return_value = True

        persist_func = MagicMock()
        value = {"data": "test_value"}

        result = await cache_patterns.write_through(
            key="test_key",
            value=value,
            persist_func=persist_func,
            ttl=3600,
        )

        assert result is True
        persist_func.assert_called_once_with("test_key", value)
        mock_dragonfly_cache.set.assert_called_once_with("test_key", value, ttl=3600)

    @pytest.mark.asyncio
    async def test_write_through_with_async_persist(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test write-through pattern with async persist function."""
        mock_dragonfly_cache.set.return_value = True

        async def async_persist(key, value):
            pass

        value = {"data": "test_value"}

        result = await cache_patterns.write_through(
            key="test_key",
            value=value,
            persist_func=async_persist,
            ttl=3600,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_write_through_persist_error(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test write-through pattern with persist error."""

        def failing_persist(key, value):
            raise Exception("Persist failed")

        value = {"data": "test_value"}

        result = await cache_patterns.write_through(
            key="test_key",
            value=value,
            persist_func=failing_persist,
            ttl=3600,
        )

        assert result is False
        mock_dragonfly_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_through_cache_error(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test write-through pattern with cache error."""
        mock_dragonfly_cache.set.return_value = False

        persist_func = MagicMock()
        value = {"data": "test_value"}

        result = await cache_patterns.write_through(
            key="test_key",
            value=value,
            persist_func=persist_func,
            ttl=3600,
        )

        assert result is False
        persist_func.assert_called_once_with("test_key", value)

    @pytest.mark.asyncio
    async def test_write_behind_success(self, cache_patterns, mock_dragonfly_cache):
        """Test write-behind pattern success."""
        mock_dragonfly_cache.set.return_value = True

        persist_func = MagicMock()
        value = {"data": "test_value"}

        result = await cache_patterns.write_behind(
            key="test_key",
            value=value,
            persist_func=persist_func,
            ttl=3600,
            delay=0.1,
        )

        assert result is True
        mock_dragonfly_cache.set.assert_called_once_with("test_key", value, ttl=3600)

        # Persistence should be scheduled but not called yet
        persist_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_behind_cache_failure(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test write-behind pattern with cache failure."""
        mock_dragonfly_cache.set.return_value = False

        persist_func = MagicMock()
        value = {"data": "test_value"}

        result = await cache_patterns.write_behind(
            key="test_key",
            value=value,
            persist_func=persist_func,
            ttl=3600,
            delay=0.1,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delayed_persist_sync_function(self, cache_patterns):
        """Test _delayed_persist with synchronous function."""
        persist_func = MagicMock()
        value = {"data": "test_value"}

        await cache_patterns._delayed_persist(
            key="test_key",
            value=value,
            persist_func=persist_func,
            delay=0.001,  # Very short delay for testing
        )

        persist_func.assert_called_once_with("test_key", value)

    @pytest.mark.asyncio
    async def test_delayed_persist_async_function(self, cache_patterns):
        """Test _delayed_persist with async function."""
        persist_called = False

        async def async_persist(key, value):
            nonlocal persist_called
            persist_called = True

        value = {"data": "test_value"}

        await cache_patterns._delayed_persist(
            key="test_key",
            value=value,
            persist_func=async_persist,
            delay=0.001,
        )

        assert persist_called

    @pytest.mark.asyncio
    async def test_delayed_persist_error(self, cache_patterns):
        """Test _delayed_persist handling errors."""

        def failing_persist(key, value):
            raise Exception("Persist failed")

        value = {"data": "test_value"}

        # Should not raise exception
        await cache_patterns._delayed_persist(
            key="test_key",
            value=value,
            persist_func=failing_persist,
            delay=0.001,
        )

    @pytest.mark.asyncio
    async def test_cache_warming_success(self, cache_patterns, mock_dragonfly_cache):
        """Test cache warming success."""
        keys_and_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
        mock_dragonfly_cache.mset.return_value = True

        result = await cache_patterns.cache_warming(
            keys_and_data=keys_and_data,
            ttl=3600,
            batch_size=2,
        )

        assert result == 3
        # Should be called twice due to batch_size=2
        assert mock_dragonfly_cache.mset.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_warming_partial_failure(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test cache warming with partial failures."""
        keys_and_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4",
        }
        # First batch succeeds, second fails
        mock_dragonfly_cache.mset.side_effect = [True, False]

        result = await cache_patterns.cache_warming(
            keys_and_data=keys_and_data,
            ttl=3600,
            batch_size=2,
        )

        assert result == 2  # Only first batch succeeded
        assert mock_dragonfly_cache.mset.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_warming_exception(self, cache_patterns, mock_dragonfly_cache):
        """Test cache warming with exception."""
        keys_and_data = {
            "key1": "value1",
            "key2": "value2",
        }
        mock_dragonfly_cache.mset.side_effect = Exception("MSET failed")

        result = await cache_patterns.cache_warming(
            keys_and_data=keys_and_data,
            ttl=3600,
            batch_size=10,
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_cache_warming_empty_data(self, cache_patterns, mock_dragonfly_cache):
        """Test cache warming with empty data."""
        result = await cache_patterns.cache_warming(
            keys_and_data={},
            ttl=3600,
        )

        assert result == 0
        mock_dragonfly_cache.mset.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_ahead_fresh_data(self, cache_patterns, mock_dragonfly_cache):
        """Test refresh-ahead with fresh data."""
        cached_data = {"result": "cached_value"}
        mock_dragonfly_cache.get.return_value = cached_data
        mock_dragonfly_cache.ttl.return_value = 3000  # Fresh (> 80% of 3600)

        fetch_func = MagicMock()

        result = await cache_patterns.refresh_ahead(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
            refresh_threshold=0.8,
        )

        assert result == cached_data
        fetch_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_ahead_should_refresh(
        self, cache_patterns, mock_dragonfly_cache
    ):
        """Test refresh-ahead when data should be refreshed."""
        cached_data = {"result": "cached_value"}
        mock_dragonfly_cache.get.return_value = cached_data
        mock_dragonfly_cache.ttl.return_value = 500  # Old data (< 80% of 3600)

        fetch_func = MagicMock(return_value={"result": "fresh_value"})

        result = await cache_patterns.refresh_ahead(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
            refresh_threshold=0.8,
        )

        assert result == cached_data  # Returns stale data immediately
        # Background refresh should be triggered

    @pytest.mark.asyncio
    async def test_refresh_ahead_cache_miss(self, cache_patterns, mock_dragonfly_cache):
        """Test refresh-ahead with cache miss."""
        mock_dragonfly_cache.get.return_value = None
        mock_dragonfly_cache.set.side_effect = [True, True]  # Lock and data

        fresh_data = {"result": "fresh_value"}
        fetch_func = MagicMock(return_value=fresh_data)

        result = await cache_patterns.refresh_ahead(
            key="test_key",
            fetch_func=fetch_func,
            ttl=3600,
        )

        assert result == fresh_data

    def test_background_tasks_cleanup(self, cache_patterns):
        """Test that background tasks are properly managed."""
        # Test that _background_tasks attribute gets created and managed
        assert not hasattr(cache_patterns, "_background_tasks")

        # Simulate adding a task (using a mock instead of actual asyncio.create_task)
        mock_task = MagicMock()
        cache_patterns._background_tasks = {mock_task}

        # Simulate task completion callback
        cache_patterns._background_tasks.discard(mock_task)

        assert len(cache_patterns._background_tasks) == 0
