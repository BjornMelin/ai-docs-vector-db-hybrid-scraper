"""Tests for CacheManager with current implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.manager import CacheManager
from src.services.cache.manager import CacheType


class TestCacheManagerInitialization:
    """Test cache manager initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default configuration."""
        manager = CacheManager()

        assert manager.enable_local_cache is True
        assert manager.enable_distributed_cache is True
        assert manager.key_prefix == "aidocs:"
        assert manager.enable_specialized_caches is True

        # Check default TTLs
        assert manager.distributed_ttl_seconds[CacheType.EMBEDDINGS] == 86400 * 7
        assert manager.distributed_ttl_seconds[CacheType.CRAWL_RESULTS] == 3600
        assert manager.distributed_ttl_seconds[CacheType.QUERY_RESULTS] == 3600

    def test_initialization_local_only(self):
        """Test initialization with only local cache."""
        manager = CacheManager(
            enable_local_cache=True,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
        )

        assert manager._local_cache is not None
        assert manager._distributed_cache is None
        assert manager._embedding_cache is None
        assert manager._search_cache is None

    def test_initialization_distributed_only(self):
        """Test initialization with only distributed cache."""
        with patch("src.services.cache.manager.DragonflyCache") as mock_dragonfly:
            manager = CacheManager(
                enable_local_cache=False,
                enable_distributed_cache=True,
                dragonfly_url="redis://localhost:6379",
            )

            assert manager._local_cache is None
            assert manager._distributed_cache is not None
            mock_dragonfly.assert_called_once()

    def test_initialization_with_specialized_caches(self):
        """Test initialization with specialized caches."""
        with patch("src.services.cache.manager.DragonflyCache") as mock_dragonfly:
            manager = CacheManager(
                enable_distributed_cache=True,
                enable_specialized_caches=True,
                dragonfly_url="redis://localhost:6379",
            )

            assert manager._embedding_cache is not None
            assert manager._search_cache is not None

    def test_initialization_no_caches_enabled(self):
        """Test initialization with no caches enabled."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
        )

        assert manager._local_cache is None
        assert manager._distributed_cache is None
        assert manager._embedding_cache is None
        assert manager._search_cache is None


class TestCacheManagerOperations:
    """Test cache manager operations."""

    @pytest.fixture
    def mock_local_cache(self):
        """Create mock local cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock()
        cache.get_stats = MagicMock(
            return_value={
                "hits": 100,
                "misses": 50,
                "size": 1000,
            }
        )
        return cache

    @pytest.fixture
    def mock_dragonfly_cache(self):
        """Create mock DragonflyDB cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock()
        cache.get_stats = AsyncMock(
            return_value={
                "hits": 500,
                "misses": 100,
                "size": 5000,
            }
        )
        cache.initialize = AsyncMock()
        cache.cleanup = AsyncMock()
        return cache

    @pytest.fixture
    def manager_with_mocks(self, mock_local_cache, mock_dragonfly_cache):
        """Create manager with mocked caches."""
        manager = CacheManager()
        manager._local_cache = mock_local_cache
        manager._distributed_cache = mock_dragonfly_cache
        manager._embedding_cache = None
        manager._search_cache = None
        return manager

    @pytest.mark.asyncio
    async def test_get_with_local_cache_hit(self, manager_with_mocks):
        """Test get operation with local cache hit."""
        manager = manager_with_mocks
        manager._local_cache.get.return_value = {"data": "cached_value"}

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result == {"data": "cached_value"}
        # Check that get was called with the hashed key
        call_args = manager._local_cache.get.call_args[0]
        assert call_args[0].startswith("aidocs:embeddings:")
        manager._distributed_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_with_distributed_cache_hit(self, manager_with_mocks):
        """Test get operation with distributed cache hit."""
        manager = manager_with_mocks
        manager._local_cache.get.return_value = None
        manager._distributed_cache.get.return_value = {"data": "distributed_value"}

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result == {"data": "distributed_value"}
        manager._local_cache.get.assert_called_once()
        manager._distributed_cache.get.assert_called_once()
        # Should also update local cache
        manager._local_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, manager_with_mocks):
        """Test get operation with cache miss."""
        manager = manager_with_mocks
        manager._local_cache.get.return_value = None
        manager._distributed_cache.get.return_value = None

        result = await manager.get("test_key", CacheType.EMBEDDINGS)

        assert result is None
        manager._local_cache.get.assert_called_once()
        manager._distributed_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_to_both_caches(self, manager_with_mocks):
        """Test set operation updates both caches."""
        manager = manager_with_mocks
        data = {"data": "test_value"}

        result = await manager.set("test_key", data, CacheType.EMBEDDINGS)

        assert result is True
        manager._local_cache.set.assert_called_once()
        manager._distributed_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, manager_with_mocks):
        """Test set operation with custom TTL."""
        manager = manager_with_mocks
        data = {"data": "test_value"}
        custom_ttl = 7200

        await manager.set("test_key", data, CacheType.EMBEDDINGS, ttl=custom_ttl)

        # Check distributed cache was called with custom TTL
        call_args = manager._distributed_cache.set.call_args
        assert call_args[1]["ttl"] == custom_ttl

    @pytest.mark.asyncio
    async def test_delete_from_both_caches(self, manager_with_mocks):
        """Test delete operation removes from both caches."""
        manager = manager_with_mocks

        result = await manager.delete("test_key")

        assert result is True
        # Check that delete was called with the hashed key
        local_call_args = manager._local_cache.delete.call_args[0]
        dist_call_args = manager._distributed_cache.delete.call_args[0]
        # Both should be called with same hashed key
        assert local_call_args[0] == dist_call_args[0]
        assert local_call_args[0].startswith("aidocs:crawl_results:")

    @pytest.mark.asyncio
    async def test_clear_all_caches(self, manager_with_mocks):
        """Test clearing all caches."""
        manager = manager_with_mocks

        await manager.clear()

        manager._local_cache.clear.assert_called_once()
        manager._distributed_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, manager_with_mocks):
        """Test getting cache statistics."""
        manager = manager_with_mocks
        # Mock the required methods
        manager._local_cache.size = AsyncMock(return_value=1000)
        manager._local_cache.get_memory_usage = MagicMock(return_value=50000)
        manager._local_cache.max_size = 1000
        manager._local_cache.max_memory_mb = 100.0

        manager._distributed_cache.size = AsyncMock(return_value=5000)
        manager._distributed_cache.redis_url = "redis://localhost:6379"
        manager._distributed_cache.enable_compression = True
        manager._distributed_cache.max_connections = 50

        stats = await manager.get_stats()

        assert "local" in stats["manager"]["enabled_layers"]
        assert "dragonfly" in stats["manager"]["enabled_layers"]
        assert stats["local"]["size"] == 1000
        assert stats["local"]["memory_usage"] == 50000
        assert stats["dragonfly"]["size"] == 5000
        assert stats["dragonfly"]["compression"] is True

    @pytest.mark.asyncio
    async def test_operation_with_no_caches(self):
        """Test operations when no caches are enabled."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
        )

        # Get should return None (default)
        result = await manager.get("test_key", CacheType.EMBEDDINGS)
        assert result is None

        # Get with custom default should return the default
        result = await manager.get(
            "test_key", CacheType.EMBEDDINGS, default="custom_default"
        )
        assert result == "custom_default"

        # Set should return True (success even with no cache)
        result = await manager.set("test_key", {"data": "value"}, CacheType.EMBEDDINGS)
        assert result is True

        # Delete should return True (no-op)
        result = await manager.delete("test_key")
        assert result is True


class TestCacheManagerKeyGeneration:
    """Test cache key generation."""

    def test_get_cache_key_basic(self):
        """Test basic cache key generation."""
        manager = CacheManager(key_prefix="test_prefix:")

        key = manager._get_cache_key("test_key", CacheType.EMBEDDINGS)

        assert key.startswith("test_prefix:embeddings:")
        assert len(key) > len("test_prefix:embeddings:")

    def test_get_cache_key_consistent(self):
        """Test cache key generation is consistent."""
        manager = CacheManager()

        key1 = manager._get_cache_key("same_key", CacheType.EMBEDDINGS)
        key2 = manager._get_cache_key("same_key", CacheType.EMBEDDINGS)

        assert key1 == key2

    def test_get_cache_key_different_keys(self):
        """Test cache key differs for different keys."""
        manager = CacheManager()

        key1 = manager._get_cache_key("key1", CacheType.EMBEDDINGS)
        key2 = manager._get_cache_key("key2", CacheType.EMBEDDINGS)

        assert key1 != key2


class TestCacheManagerSpecializedCaches:
    """Test specialized cache integration."""

    @pytest.fixture
    def manager_with_specialized(self):
        """Create manager with specialized caches."""
        with patch("src.services.cache.manager.DragonflyCache"):
            manager = CacheManager(
                enable_distributed_cache=True,
                enable_specialized_caches=True,
            )

            # Mock specialized caches
            manager._embedding_cache = MagicMock()
            manager._search_cache = MagicMock()

            return manager

    def test_embedding_cache_property(self, manager_with_specialized):
        """Test embedding cache property."""
        assert manager_with_specialized.embedding_cache is not None

    def test_search_cache_property(self, manager_with_specialized):
        """Test search cache property."""
        assert manager_with_specialized.search_cache is not None

    def test_no_specialized_caches(self):
        """Test accessing specialized caches when disabled."""
        manager = CacheManager(enable_specialized_caches=False)

        assert manager.embedding_cache is None
        assert manager.search_cache is None


class TestCacheManagerErrorHandling:
    """Test error handling in cache manager."""

    @pytest.fixture
    def mock_local_cache(self):
        """Create mock local cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock()
        return cache

    @pytest.fixture
    def mock_dragonfly_cache(self):
        """Create mock DragonflyDB cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.clear = AsyncMock()
        return cache

    @pytest.fixture
    def manager_with_mocks(self, mock_local_cache, mock_dragonfly_cache):
        """Create manager with mocked caches."""
        manager = CacheManager()
        manager._local_cache = mock_local_cache
        manager._distributed_cache = mock_dragonfly_cache
        manager._embedding_cache = None
        manager._search_cache = None
        return manager

    @pytest.mark.asyncio
    async def test_get_handles_cache_errors(self, manager_with_mocks):
        """Test get operation handles cache errors gracefully."""
        manager = manager_with_mocks
        manager._local_cache.get.side_effect = Exception("Cache error")

        # Should not raise, but return None
        result = await manager.get("test_key", CacheType.EMBEDDINGS)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_handles_cache_errors(self, manager_with_mocks):
        """Test set operation handles cache errors gracefully."""
        manager = manager_with_mocks
        manager._local_cache.set.side_effect = Exception("Cache error")

        # Should return False when exception occurs
        result = await manager.set("test_key", {"data": "value"}, CacheType.EMBEDDINGS)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_handles_cache_errors(self, manager_with_mocks):
        """Test delete operation handles cache errors gracefully."""
        manager = manager_with_mocks
        manager._local_cache.delete.side_effect = Exception("Cache error")

        # Should return False when exception occurs
        result = await manager.delete("test_key")
        assert result is False
