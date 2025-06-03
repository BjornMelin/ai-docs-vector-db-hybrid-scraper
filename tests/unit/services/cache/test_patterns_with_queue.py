"""Tests for CachePatterns with task queue integration."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.cache.patterns import CachePatterns


class TestCachePatternsWithTaskQueue:
    """Test CachePatterns with task queue integration."""

    @pytest.fixture
    def cache(self):
        """Create mock DragonflyCache."""
        return AsyncMock(spec=DragonflyCache)

    @pytest.fixture
    def task_queue_manager(self):
        """Create mock task queue manager."""
        return AsyncMock()

    @pytest.fixture
    def cache_patterns(self, cache, task_queue_manager):
        """Create CachePatterns with task queue."""
        return CachePatterns(cache, task_queue_manager)

    @pytest.mark.asyncio
    async def test_write_behind_with_queue(self, cache_patterns, cache, task_queue_manager):
        """Test write_behind uses task queue."""
        # Setup
        cache.set = AsyncMock(return_value=True)
        task_queue_manager.enqueue = AsyncMock(return_value="job_789")

        # Create a persist function
        def persist_func(key, value):
            pass

        # Execute
        result = await cache_patterns.write_behind(
            key="test_key",
            value={"data": "test"},
            persist_func=persist_func,
            ttl=3600,
            delay=5.0
        )

        # Verify
        assert result is True
        cache.set.assert_called_once_with("test_key", {"data": "test"}, ttl=3600)

        task_queue_manager.enqueue.assert_called_once_with(
            "persist_cache",
            key="test_key",
            value={"data": "test"},
            persist_func_module=persist_func.__module__,
            persist_func_name="persist_func",
            delay=5.0,
            _delay=5,  # Integer seconds
        )

    @pytest.mark.asyncio
    async def test_write_behind_queue_failure_fallback(self, cache_patterns, cache, task_queue_manager):
        """Test write_behind falls back when queue fails."""
        # Setup
        cache.set = AsyncMock(return_value=True)
        task_queue_manager.enqueue = AsyncMock(return_value=None)  # Queue failure

        persist_calls = []

        def persist_func(key, value):
            persist_calls.append((key, value))

        # Execute
        result = await cache_patterns.write_behind(
            key="test_key",
            value={"data": "test"},
            persist_func=persist_func,
            delay=0.001  # Very short delay for test
        )

        # Verify
        assert result is True

        # Should have attempted to queue
        task_queue_manager.enqueue.assert_called_once()

        # Should have fallen back to direct persist
        assert len(persist_calls) == 1
        assert persist_calls[0] == ("test_key", {"data": "test"})

    @pytest.mark.asyncio
    async def test_write_behind_no_queue(self, cache):
        """Test write_behind without task queue."""
        # Create patterns without task queue
        cache_patterns = CachePatterns(cache, task_queue_manager=None)
        cache.set = AsyncMock(return_value=True)

        persist_calls = []

        def persist_func(key, value):
            persist_calls.append((key, value))

        # Execute
        result = await cache_patterns.write_behind(
            key="test_key",
            value={"data": "test"},
            persist_func=persist_func,
            delay=0.001  # Very short delay for test
        )

        # Verify
        assert result is True

        # Wait a bit for async persistence
        await asyncio.sleep(0.01)

        # Should have persisted directly
        assert len(persist_calls) == 1
        assert persist_calls[0] == ("test_key", {"data": "test"})

    @pytest.mark.asyncio
    async def test_write_behind_async_persist_func(self, cache_patterns, cache, task_queue_manager):
        """Test write_behind with async persist function."""
        # Setup
        cache.set = AsyncMock(return_value=True)
        task_queue_manager.enqueue = AsyncMock(return_value="job_999")

        async def async_persist_func(key, value):
            pass

        # Execute
        result = await cache_patterns.write_behind(
            key="test_key",
            value={"data": "test"},
            persist_func=async_persist_func,
            delay=10.0
        )

        # Verify
        assert result is True

        task_queue_manager.enqueue.assert_called_once_with(
            "persist_cache",
            key="test_key",
            value={"data": "test"},
            persist_func_module=async_persist_func.__module__,
            persist_func_name="async_persist_func",
            delay=10.0,
            _delay=10,
        )

    @pytest.mark.asyncio
    async def test_write_behind_cache_failure(self, cache_patterns, cache, task_queue_manager):
        """Test write_behind when cache set fails."""
        # Setup
        cache.set = AsyncMock(return_value=False)

        def persist_func(key, value):
            pass

        # Execute
        result = await cache_patterns.write_behind(
            key="test_key",
            value={"data": "test"},
            persist_func=persist_func,
        )

        # Verify
        assert result is False

        # Should not have attempted to queue since cache failed
        task_queue_manager.enqueue.assert_not_called()
