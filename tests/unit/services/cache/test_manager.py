"""Tests for cache manager module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.enums import CacheType
from src.services.cache.manager import CacheManager


class TestCacheType:
    """Test the CacheType enum."""

    def test_cache_type_values(self):
        """Test CacheType enum values."""
        assert CacheType.EMBEDDINGS.value == "embeddings"
        assert CacheType.CRAWL.value == "crawl"
        assert CacheType.SEARCH.value == "search"
        assert CacheType.HYDE.value == "hyde"

    def test_cache_type_members(self):
        """Test CacheType enum members."""
        assert len(CacheType) == 4
        assert CacheType.EMBEDDINGS in CacheType
        assert CacheType.CRAWL in CacheType
        assert CacheType.SEARCH in CacheType
        assert CacheType.HYDE in CacheType


class TestCacheManager:
    """Test the CacheManager class."""

    @pytest.fixture
    def manager_config(self):
        """Default cache manager configuration."""
        return {
            "dragonfly_url": "redis://localhost:6379",
            "enable_local_cache": True,
            "enable_distributed_cache": True,
            "local_max_size": 1000,
            "local_max_memory_mb": 100.0,
            "local_ttl_seconds": 300,
            "key_prefix": "test:",
            "enable_metrics": True,
            "enable_specialized_caches": True,
        }

    @patch("src.services.cache.manager.LocalCache")
    @patch("src.services.cache.manager.DragonflyCache")
    @patch("src.services.cache.manager.EmbeddingCache")
    @patch("src.services.cache.manager.SearchResultCache")
    @patch("src.services.cache.manager.CacheMetrics")
    def test_manager_initialization_full(
        self,
        mock_metrics,
        mock_search_cache,
        mock_embedding_cache,
        mock_dragonfly_cache,
        mock_local_cache,
        manager_config,
    ):
        """Test full cache manager initialization."""
        manager = CacheManager(**manager_config)

        assert manager.enable_local_cache is True
        assert manager.enable_distributed_cache is True
        assert manager.key_prefix == "test:"
        assert manager.enable_specialized_caches is True

        # Verify cache instances created
        mock_local_cache.assert_called_once()
        mock_dragonfly_cache.assert_called_once()
        mock_embedding_cache.assert_called_once()
        mock_search_cache.assert_called_once()
        mock_metrics.assert_called_once()

    @patch("src.services.cache.manager.LocalCache")
    @patch("src.services.cache.manager.DragonflyCache")
    def test_manager_initialization_minimal(
        self, mock_dragonfly_cache, mock_local_cache
    ):
        """Test minimal cache manager initialization."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_metrics=False,
            enable_specialized_caches=False,
        )

        assert manager.enable_local_cache is False
        assert manager.enable_distributed_cache is False
        assert manager.enable_specialized_caches is False

        # Verify no cache instances created
        mock_local_cache.assert_not_called()
        mock_dragonfly_cache.assert_not_called()
        assert manager._local_cache is None
        assert manager._distributed_cache is None
        assert manager._embedding_cache is None
        assert manager._search_cache is None
        assert manager._metrics is None

    @patch("src.services.cache.manager.LocalCache")
    @patch("src.services.cache.manager.DragonflyCache")
    @patch("src.services.cache.manager.EmbeddingCache")
    @patch("src.services.cache.manager.SearchResultCache")
    @patch("src.services.cache.manager.CacheMetrics")
    def test_manager_properties(
        self,
        mock_metrics,
        mock_search_cache,
        mock_embedding_cache,
        mock_dragonfly_cache,
        mock_local_cache,
        manager_config,
    ):
        """Test cache manager properties."""
        manager = CacheManager(**manager_config)

        assert manager.local_cache is not None
        assert manager.distributed_cache is not None
        assert manager.embedding_cache is not None
        assert manager.search_cache is not None
        assert manager.metrics is not None

    def test_default_ttl_configuration(self):
        """Test default TTL configuration."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
        )

        expected_ttls = {
            CacheType.EMBEDDINGS: 86400 * 7,  # 7 days
            CacheType.CRAWL: 3600,  # 1 hour
            CacheType.SEARCH: 3600,  # 1 hour
            CacheType.HYDE: 3600,  # 1 hour
        }

        assert manager.distributed_ttl_seconds == expected_ttls

    def test_custom_ttl_configuration(self):
        """Test custom TTL configuration."""
        custom_ttls = {
            CacheType.EMBEDDINGS: 43200,  # 12 hours
            CacheType.CRAWL: 1800,  # 30 minutes
            CacheType.SEARCH: 900,  # 15 minutes
            CacheType.HYDE: 1200,  # 20 minutes
        }

        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
            distributed_ttl_seconds=custom_ttls,
        )

        assert manager.distributed_ttl_seconds == custom_ttls

    def test_cache_key_generation(self):
        """Test cache key generation."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
            key_prefix="test:",
        )

        key = manager._get_cache_key("some_key", CacheType.EMBEDDINGS)

        # Should contain prefix and cache type
        assert key.startswith("test:embeddings:")
        # Should contain hash of the key
        assert len(key.split(":")) == 3
        assert len(key.split(":")[-1]) == 12  # MD5 hash first 12 chars

    @pytest.mark.asyncio
    async def test_get_local_cache_hit(self):
        """Test get operation with local cache hit."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.get.return_value = "cached_value"
            mock_local_cache_cls.return_value = mock_local_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(
                enable_distributed_cache=False, enable_specialized_caches=False
            )

            # Test get operation
            result = await manager.get("test_key", CacheType.CRAWL)

            assert result == "cached_value"
            mock_local_cache.get.assert_called_once()
            mock_metrics.record_hit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_distributed_cache_hit(self):
        """Test get operation with distributed cache hit."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.get.return_value = None  # Local cache miss
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.get.return_value = "distributed_value"
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(enable_specialized_caches=False)

            # Test get operation
            result = await manager.get("test_key", CacheType.CRAWL)

            assert result == "distributed_value"
            mock_local_cache.get.assert_called_once()
            mock_distributed_cache.get.assert_called_once()
            mock_local_cache.set.assert_called_once()  # Should populate L1
            mock_metrics.record_hit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_miss(self):
        """Test get operation with cache miss."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.get.return_value = None
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.get.return_value = None
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(enable_specialized_caches=False)

            # Test get operation with default
            result = await manager.get("test_key", CacheType.CRAWL, "default_value")

            assert result == "default_value"
            mock_metrics.record_miss.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_with_exception(self):
        """Test get operation with exception handling."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.get.side_effect = Exception("Cache error")
            mock_local_cache_cls.return_value = mock_local_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(
                enable_distributed_cache=False, enable_specialized_caches=False
            )

            # Test get operation
            result = await manager.get("test_key", CacheType.CRAWL, "default")

            assert result == "default"
            mock_metrics.record_miss.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_operation(self):
        """Test set operation."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.set.return_value = True
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(enable_specialized_caches=False)

            # Test set operation
            result = await manager.set("test_key", "test_value", CacheType.EMBEDDINGS)

            assert result is True
            mock_local_cache.set.assert_called_once()
            mock_distributed_cache.set.assert_called_once()
            mock_metrics.record_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        """Test set operation with custom TTL."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.set.return_value = True
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_specialized_caches=False, enable_metrics=False
            )

            # Test set operation with custom TTL
            await manager.set("test_key", "test_value", CacheType.EMBEDDINGS, ttl=600)

            # Verify TTL values
            local_call = mock_local_cache.set.call_args
            distributed_call = mock_distributed_cache.set.call_args

            assert local_call[1]["ttl"] == 300  # min(600, 300) for local
            assert distributed_call[1]["ttl"] == 600  # custom TTL for distributed

    @pytest.mark.asyncio
    async def test_set_with_exception(self):
        """Test set operation with exception handling."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.set.side_effect = Exception("Set error")
            mock_local_cache_cls.return_value = mock_local_cache

            mock_metrics = MagicMock()
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(
                enable_distributed_cache=False, enable_specialized_caches=False
            )

            # Test set operation
            result = await manager.set("test_key", "test_value")

            assert result is False
            mock_metrics.record_set.assert_called_once_with(
                CacheType.CRAWL, pytest.approx(0, abs=100), False
            )

    @pytest.mark.asyncio
    async def test_delete_operation(self):
        """Test delete operation."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.delete.return_value = True
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_specialized_caches=False, enable_metrics=False
            )

            # Test delete operation
            result = await manager.delete("test_key", CacheType.EMBEDDINGS)

            assert result is True
            mock_local_cache.delete.assert_called_once()
            mock_distributed_cache.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_with_exception(self):
        """Test delete operation with exception handling."""
        with patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls:
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.delete.side_effect = Exception("Delete error")
            mock_local_cache_cls.return_value = mock_local_cache

            manager = CacheManager(
                enable_distributed_cache=False,
                enable_specialized_caches=False,
                enable_metrics=False,
            )

            # Test delete operation
            result = await manager.delete("test_key")

            assert result is False

    @pytest.mark.asyncio
    async def test_clear_all_caches(self):
        """Test clearing all caches."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_specialized_caches=False, enable_metrics=False
            )

            # Test clear operation
            result = await manager.clear()

            assert result is True
            mock_local_cache.clear.assert_called_once()
            mock_distributed_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_specific_cache_type(self):
        """Test clearing specific cache type."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.scan_keys.return_value = ["key1", "key2"]
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_specialized_caches=False,
                enable_metrics=False,
                key_prefix="test:",
            )

            # Test clear specific cache type
            result = await manager.clear(CacheType.EMBEDDINGS)

            assert result is True
            mock_distributed_cache.scan_keys.assert_called_once_with(
                "test:embeddings:*"
            )
            assert mock_distributed_cache.delete.call_count == 2
            assert mock_local_cache.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_with_exception(self):
        """Test clear operation with exception handling."""
        with patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls:
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.clear.side_effect = Exception("Clear error")
            mock_local_cache_cls.return_value = mock_local_cache

            manager = CacheManager(
                enable_distributed_cache=False,
                enable_specialized_caches=False,
                enable_metrics=False,
            )

            # Test clear operation
            result = await manager.clear()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting cache statistics."""
        with (
            patch("src.services.cache.manager.LocalCache") as mock_local_cache_cls,
            patch(
                "src.services.cache.manager.DragonflyCache"
            ) as mock_dragonfly_cache_cls,
            patch(
                "src.services.cache.manager.EmbeddingCache"
            ) as mock_embedding_cache_cls,
            patch(
                "src.services.cache.manager.SearchResultCache"
            ) as mock_search_cache_cls,
            patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls,
        ):
            # Setup mocks
            mock_local_cache = AsyncMock()
            mock_local_cache.size.return_value = 100
            mock_local_cache.get_memory_usage.return_value = 50.0
            mock_local_cache.max_size = 1000
            mock_local_cache.max_memory_mb = 100.0
            mock_local_cache_cls.return_value = mock_local_cache

            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.size.return_value = 500
            mock_distributed_cache.redis_url = "redis://localhost:6379"
            mock_distributed_cache.enable_compression = True
            mock_distributed_cache.max_connections = 50
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            mock_embedding_cache = AsyncMock()
            mock_embedding_cache.get_stats.return_value = {"embedding_stats": "data"}
            mock_embedding_cache_cls.return_value = mock_embedding_cache

            mock_search_cache = AsyncMock()
            mock_search_cache.get_stats.return_value = {"search_stats": "data"}
            mock_search_cache_cls.return_value = mock_search_cache

            mock_metrics = MagicMock()
            mock_metrics.get_summary.return_value = {"metrics": "data"}
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager()

            # Test get stats
            stats = await manager.get_stats()

            assert "manager" in stats
            assert "local" in stats
            assert "dragonfly" in stats
            assert "embedding_cache" in stats
            assert "search_cache" in stats
            assert "metrics" in stats

            assert stats["manager"]["enabled_layers"] == ["local", "dragonfly"]
            assert stats["local"]["size"] == 100
            assert stats["dragonfly"]["size"] == 500

    @pytest.mark.asyncio
    async def test_close_operation(self):
        """Test close operation."""
        with patch(
            "src.services.cache.manager.DragonflyCache"
        ) as mock_dragonfly_cache_cls:
            # Setup mocks
            mock_distributed_cache = AsyncMock()
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_local_cache=False,
                enable_specialized_caches=False,
                enable_metrics=False,
            )

            # Test close operation
            await manager.close()

            mock_distributed_cache.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_exception(self):
        """Test close operation with exception handling."""
        with patch(
            "src.services.cache.manager.DragonflyCache"
        ) as mock_dragonfly_cache_cls:
            # Setup mocks
            mock_distributed_cache = AsyncMock()
            mock_distributed_cache.close.side_effect = Exception("Close error")
            mock_dragonfly_cache_cls.return_value = mock_distributed_cache

            manager = CacheManager(
                enable_local_cache=False,
                enable_specialized_caches=False,
                enable_metrics=False,
            )

            # Test close operation (should not raise)
            await manager.close()

    @pytest.mark.asyncio
    async def test_get_embedding_direct(self):
        """Test direct embedding cache access."""
        with (
            patch("src.services.cache.manager.DragonflyCache"),
            patch(
                "src.services.cache.manager.EmbeddingCache"
            ) as mock_embedding_cache_cls,
        ):
            # Setup mocks
            mock_embedding_cache = AsyncMock()
            mock_embedding_cache.get_embedding.return_value = [0.1, 0.2, 0.3]
            mock_embedding_cache_cls.return_value = mock_embedding_cache

            manager = CacheManager(enable_local_cache=False, enable_metrics=False)

            # Test direct embedding access
            result = await manager.get_embedding_direct("hash123", "model1")

            assert result == [0.1, 0.2, 0.3]
            mock_embedding_cache.get_embedding.assert_called_once_with(
                "hash123", "model1"
            )

    @pytest.mark.asyncio
    async def test_get_embedding_direct_disabled(self):
        """Test direct embedding cache access when disabled."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
            enable_metrics=False,
        )

        # Test direct embedding access when disabled
        result = await manager.get_embedding_direct("hash123", "model1")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_search_results_direct(self):
        """Test direct search result cache access."""
        with (
            patch("src.services.cache.manager.DragonflyCache"),
            patch(
                "src.services.cache.manager.SearchResultCache"
            ) as mock_search_cache_cls,
        ):
            # Setup mocks
            mock_search_cache = AsyncMock()
            mock_search_cache.set_search_results.return_value = True
            mock_search_cache_cls.return_value = mock_search_cache

            manager = CacheManager(enable_local_cache=False, enable_metrics=False)

            # Test direct search result setting
            results = [{"id": 1, "score": 0.9}]
            result = await manager.set_search_results_direct(
                "query_hash", "collection1", results, 3600
            )

            assert result is True
            mock_search_cache.set_search_results.assert_called_once_with(
                "query_hash", "collection1", results, 3600
            )

    @pytest.mark.asyncio
    async def test_set_search_results_direct_disabled(self):
        """Test direct search result cache access when disabled."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
            enable_metrics=False,
        )

        # Test direct search result setting when disabled
        result = await manager.set_search_results_direct(
            "query_hash", "collection1", [], 3600
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_performance_stats(self):
        """Test getting performance statistics."""
        with patch("src.services.cache.manager.CacheMetrics") as mock_metrics_cls:
            # Setup mocks
            mock_metrics = MagicMock()
            mock_metrics.get_hit_rates.return_value = {"hit_rate": 0.85}
            mock_metrics.get_latency_stats.return_value = {"avg_latency": 10.5}
            mock_metrics.get_operation_counts.return_value = {"total_ops": 1000}
            mock_metrics_cls.return_value = mock_metrics

            manager = CacheManager(
                enable_local_cache=False,
                enable_distributed_cache=False,
                enable_specialized_caches=False,
            )

            # Test performance stats
            stats = await manager.get_performance_stats()

            assert stats["hit_rates"] == {"hit_rate": 0.85}
            assert stats["latency_stats"] == {"avg_latency": 10.5}
            assert stats["operation_counts"] == {"total_ops": 1000}

    @pytest.mark.asyncio
    async def test_get_performance_stats_disabled(self):
        """Test getting performance statistics when metrics disabled."""
        manager = CacheManager(
            enable_local_cache=False,
            enable_distributed_cache=False,
            enable_specialized_caches=False,
            enable_metrics=False,
        )

        # Test performance stats when metrics disabled
        stats = await manager.get_performance_stats()

        assert stats == {}
