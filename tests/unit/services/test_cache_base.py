"""Tests for cache base module."""

from abc import ABC

import pytest
from src.services.cache.base import CacheInterface


class MockCache(CacheInterface[str]):
    """Mock cache implementation for testing."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}
        self.is_closed = False

    async def get(self, key: str) -> str | None:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        return self.data.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        self.data[key] = value
        self.ttls[key] = ttl
        return True

    async def delete(self, key: str) -> bool:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        if key in self.data:
            del self.data[key]
            del self.ttls[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        return key in self.data

    async def clear(self) -> int:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        count = len(self.data)
        self.data.clear()
        self.ttls.clear()
        return count

    async def size(self) -> int:
        if self.is_closed:
            raise RuntimeError("Cache is closed")
        return len(self.data)

    async def close(self) -> None:
        self.is_closed = True
        self.data.clear()
        self.ttls.clear()


class FailingCache(CacheInterface[str]):
    """Cache implementation that always fails for testing."""

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        return False

    async def delete(self, key: str) -> bool:
        return False

    async def exists(self, key: str) -> bool:
        return False

    async def clear(self) -> int:
        return 0

    async def size(self) -> int:
        return 0

    async def close(self) -> None:
        pass


class TestCacheInterface:
    """Test the CacheInterface abstract base class."""

    def test_cache_interface_is_abstract(self):
        """Test that CacheInterface is an abstract base class."""
        assert issubclass(CacheInterface, ABC)

        with pytest.raises(TypeError):
            CacheInterface()

    def test_cache_interface_generic(self):
        """Test that CacheInterface is properly generic."""
        cache_int = CacheInterface[int]
        cache_str = CacheInterface[str]

        assert cache_int != cache_str

    @pytest.mark.asyncio
    async def test_mock_cache_basic_operations(self):
        """Test basic cache operations with mock implementation."""
        cache = MockCache()

        # Test initial state
        assert await cache.size() == 0
        assert await cache.get("key1") is None
        assert not await cache.exists("key1")

        # Test set operation
        result = await cache.set("key1", "value1")
        assert result is True
        assert await cache.size() == 1
        assert await cache.exists("key1")

        # Test get operation
        value = await cache.get("key1")
        assert value == "value1"

        # Test delete operation
        result = await cache.delete("key1")
        assert result is True
        assert await cache.size() == 0
        assert not await cache.exists("key1")

        # Test delete non-existent key
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_mock_cache_ttl_handling(self):
        """Test TTL handling in mock cache."""
        cache = MockCache()

        # Test setting with TTL
        await cache.set("key1", "value1", ttl=300)
        assert cache.ttls["key1"] == 300

        # Test setting without TTL
        await cache.set("key2", "value2")
        assert cache.ttls["key2"] is None

    @pytest.mark.asyncio
    async def test_mock_cache_clear_operation(self):
        """Test cache clear operation."""
        cache = MockCache()

        # Add multiple items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert await cache.size() == 3

        # Clear cache
        cleared = await cache.clear()
        assert cleared == 3
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_mock_cache_close_operation(self):
        """Test cache close operation."""
        cache = MockCache()

        # Add some data
        await cache.set("key1", "value1")
        assert await cache.size() == 1

        # Close cache
        await cache.close()
        assert cache.is_closed

        # Operations should fail after close
        with pytest.raises(RuntimeError, match="Cache is closed"):
            await cache.get("key1")

        with pytest.raises(RuntimeError, match="Cache is closed"):
            await cache.set("key2", "value2")

    @pytest.mark.asyncio
    async def test_get_many_default_implementation(self):
        """Test the default get_many implementation."""
        cache = MockCache()

        # Set up test data
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Test get_many
        results = await cache.get_many(["key1", "key2", "key3"])

        expected = {"key1": "value1", "key2": "value2", "key3": None}
        assert results == expected

    @pytest.mark.asyncio
    async def test_get_many_empty_keys(self):
        """Test get_many with empty key list."""
        cache = MockCache()

        results = await cache.get_many([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_set_many_default_implementation(self):
        """Test the default set_many implementation."""
        cache = MockCache()

        items = {"key1": "value1", "key2": "value2", "key3": "value3"}

        # Test set_many
        results = await cache.set_many(items, ttl=300)

        expected = {"key1": True, "key2": True, "key3": True}
        assert results == expected

        # Verify data was set
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

        # Verify TTL was set
        assert cache.ttls["key1"] == 300
        assert cache.ttls["key2"] == 300
        assert cache.ttls["key3"] == 300

    @pytest.mark.asyncio
    async def test_set_many_empty_items(self):
        """Test set_many with empty items dict."""
        cache = MockCache()

        results = await cache.set_many({})
        assert results == {}

    @pytest.mark.asyncio
    async def test_set_many_with_failing_operations(self):
        """Test set_many with some failing operations."""
        cache = FailingCache()

        items = {"key1": "value1", "key2": "value2"}

        results = await cache.set_many(items)

        expected = {"key1": False, "key2": False}
        assert results == expected

    @pytest.mark.asyncio
    async def test_delete_many_default_implementation(self):
        """Test the default delete_many implementation."""
        cache = MockCache()

        # Set up test data
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Test delete_many
        results = await cache.delete_many(["key1", "key2", "key3"])

        expected = {
            "key1": True,
            "key2": True,
            "key3": False,  # Doesn't exist
        }
        assert results == expected

        # Verify data was deleted
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_delete_many_empty_keys(self):
        """Test delete_many with empty key list."""
        cache = MockCache()

        results = await cache.delete_many([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_delete_many_with_failing_operations(self):
        """Test delete_many with some failing operations."""
        cache = FailingCache()

        results = await cache.delete_many(["key1", "key2"])

        expected = {"key1": False, "key2": False}
        assert results == expected

    @pytest.mark.asyncio
    async def test_batch_operations_consistency(self):
        """Test that batch operations maintain consistency."""
        cache = MockCache()

        # Test data
        items = {"key1": "value1", "key2": "value2", "key3": "value3"}

        # Set multiple items
        set_results = await cache.set_many(items, ttl=600)
        assert all(set_results.values())

        # Get multiple items
        get_results = await cache.get_many(list(items.keys()))
        for key, value in items.items():
            assert get_results[key] == value

        # Delete multiple items
        delete_results = await cache.delete_many(list(items.keys()))
        assert all(delete_results.values())

        # Verify all items are gone
        final_get_results = await cache.get_many(list(items.keys()))
        assert all(value is None for value in final_get_results.values())

    @pytest.mark.asyncio
    async def test_cache_interface_method_signatures(self):
        """Test that all abstract methods have correct signatures."""
        cache = MockCache()

        # Test method signatures match interface
        assert callable(cache.get)
        assert callable(cache.set)
        assert callable(cache.delete)
        assert callable(cache.exists)
        assert callable(cache.clear)
        assert callable(cache.size)
        assert callable(cache.close)
        assert callable(cache.get_many)
        assert callable(cache.set_many)
        assert callable(cache.delete_many)

    @pytest.mark.asyncio
    async def test_cache_type_safety(self):
        """Test type safety of cache operations."""
        cache = MockCache()  # MockCache[str]

        # Should work with strings
        await cache.set("key", "string_value")
        value = await cache.get("key")
        assert isinstance(value, str)
        assert value == "string_value"

    def test_cache_interface_inheritance(self):
        """Test that implementations properly inherit from CacheInterface."""
        cache = MockCache()

        assert isinstance(cache, CacheInterface)
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "delete")
        assert hasattr(cache, "exists")
        assert hasattr(cache, "clear")
        assert hasattr(cache, "size")
        assert hasattr(cache, "close")
        assert hasattr(cache, "get_many")
        assert hasattr(cache, "set_many")
        assert hasattr(cache, "delete_many")
