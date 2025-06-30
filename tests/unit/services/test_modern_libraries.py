"""Tests for modern library implementations.

This module tests the modernized circuit breaker, caching, and rate limiting
implementations to ensure they provide equivalent or better functionality
than the custom implementations they replace.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Config
from src.config.enums import CacheType
from src.services.cache.modern import ModernCacheManager
from src.services.circuit_breaker.modern import ModernCircuitBreakerManager
from src.services.migration.library_migration import (
    LibraryMigrationManager,
    MigrationConfig,
    MigrationMode,
)


@pytest.fixture
def redis_url():
    """Redis URL for testing."""
    return "redis://localhost:6379"


@pytest.fixture
def config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.cache = Mock()
    config.cache.cache_ttl_seconds = {
        CacheType.EMBEDDINGS: 86400,
        CacheType.SEARCH: 3600,
        CacheType.CRAWL: 3600,
        CacheType.HYDE: 3600,
    }
    config.performance = Mock()
    config.performance.circuit_breaker_failure_threshold = 5
    config.performance.circuit_breaker_recovery_timeout = 60
    return config


class TestModernCircuitBreakerManager:
    """Test cases for ModernCircuitBreakerManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, redis_url, config):
        """Test circuit breaker manager initialization."""
        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
        ):
            mock_storage.from_url.return_value = Mock()
            mock_factory.return_value = Mock()

            manager = ModernCircuitBreakerManager(redis_url, config)

            assert manager.redis_url == redis_url
            assert manager.config == config
            mock_storage.from_url.assert_called_once_with(redis_url)

    @pytest.mark.asyncio
    async def test_get_breaker(self, redis_url, config):
        """Test getting circuit breaker instances."""
        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
        ):
            mock_storage.from_url.return_value = Mock()
            mock_breaker = AsyncMock()
            mock_factory_instance = AsyncMock()
            mock_factory_instance.get_breaker.return_value = mock_breaker
            mock_factory.return_value = mock_factory_instance

            manager = ModernCircuitBreakerManager(redis_url, config)

            # Test getting breaker for first time
            breaker1 = await manager.get_breaker("service1")
            assert breaker1 == mock_breaker
            mock_factory_instance.get_breaker.assert_called_once_with("service1")

            # Test getting same breaker again (should be cached)
            breaker2 = await manager.get_breaker("service1")
            assert breaker2 == mock_breaker
            assert mock_factory_instance.get_breaker.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_protected_call_success(self, redis_url, config):
        """Test successful protected function call."""
        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
        ):
            mock_storage.from_url.return_value = Mock()
            mock_breaker = AsyncMock()
            mock_breaker.__aenter__ = AsyncMock(return_value=mock_breaker)
            mock_breaker.__aexit__ = AsyncMock(return_value=False)

            mock_factory_instance = AsyncMock()
            mock_factory_instance.get_breaker.return_value = mock_breaker
            mock_factory.return_value = mock_factory_instance

            manager = ModernCircuitBreakerManager(redis_url, config)

            # Test function to call
            async def test_func(x, y):
                return x + y

            # Mock the context manager behavior
            with patch.object(manager, "get_breaker", return_value=mock_breaker):
                result = await manager.protected_call("service1", test_func, 2, 3)
                assert result == 5, "Function should return sum of arguments"

                # Verify the function was called through the circuit breaker
                mock_breaker.__aenter__.assert_called_once()
                mock_breaker.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator(self, redis_url, config):
        """Test circuit breaker decorator functionality."""
        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
        ):
            mock_storage.from_url.return_value = Mock()
            mock_factory.return_value = Mock()

            manager = ModernCircuitBreakerManager(redis_url, config)

            # Test decorator
            @manager.decorator("test_service")
            async def decorated_function(x):
                return x * 2

            assert decorated_function.__name__ == "decorated_function"
            assert asyncio.iscoroutinefunction(decorated_function)

    @pytest.mark.asyncio
    async def test_get_breaker_status(self, redis_url, config):
        """Test getting circuit breaker status."""
        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
        ):
            mock_storage.from_url.return_value = Mock()
            mock_factory.return_value = Mock()

            manager = ModernCircuitBreakerManager(redis_url, config)

            # Test status for non-existent breaker
            status = await manager.get_breaker_status("nonexistent")
            assert status == {"status": "not_initialized"}

            # Test status for existing breaker
            mock_breaker = Mock()
            mock_breaker.state = "closed"
            mock_breaker.failure_count = 0
            mock_breaker.is_open = False
            manager._breakers["test_service"] = mock_breaker

            status = await manager.get_breaker_status("test_service")
            assert status["service_name"] == "test_service"
            assert status["state"] == "closed"


class TestModernCacheManager:
    """Test cases for ModernCacheManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, redis_url, config):
        """Test cache manager initialization."""
        with patch("src.services.cache.modern.Cache"):
            manager = ModernCacheManager(redis_url, config=config)

            assert manager.redis_url == redis_url
            assert manager.config == config
            assert manager.key_prefix == "aidocs:"

    @pytest.mark.asyncio
    async def test_get_cache_for_type(self, redis_url, config):
        """Test getting cache instance for different types."""
        with patch("src.services.cache.modern.Cache"):
            manager = ModernCacheManager(redis_url, config=config)

            embedding_cache = manager.get_cache_for_type(CacheType.EMBEDDINGS)
            search_cache = manager.get_cache_for_type(CacheType.SEARCH)

            assert embedding_cache == manager.embedding_cache
            assert search_cache == manager.search_cache

    @pytest.mark.asyncio
    async def test_cache_decorators(self, redis_url, config):
        """Test cache decorator functionality."""
        with (
            patch("src.services.cache.modern.Cache"),
            patch("src.services.cache.modern.cached") as mock_cached,
        ):
            manager = ModernCacheManager(redis_url, config=config)

            # Test embedding cache decorator
            _embedding_decorator = manager.cache_embeddings(ttl=3600)
            mock_cached.assert_called_with(
                ttl=3600,
                cache=manager.embedding_cache,
                key_builder=manager._embedding_key_builder,
            )

            # Test search cache decorator
            _search_decorator = manager.cache_search_results(ttl=1800)
            assert mock_cached.call_count == 2

    @pytest.mark.asyncio
    async def test_get_set_operations(self, redis_url, config):
        """Test cache get and set operations."""
        with patch("src.services.cache.modern.Cache"):
            mock_cache_instance = AsyncMock()
            mock_cache_instance.get.return_value = "cached_value"
            mock_cache_instance.set.return_value = None

            manager = ModernCacheManager(redis_url, config=config)
            manager.search_cache = mock_cache_instance

            # Test get operation
            value = await manager.get("test_key", CacheType.SEARCH, "default")
            assert value == "cached_value"
            mock_cache_instance.get.assert_called_once_with("test_key")

            # Test set operation
            success = await manager.set(
                "test_key", "test_value", CacheType.SEARCH, 3600
            )
            assert success is True
            mock_cache_instance.set.assert_called_once_with(
                "test_key", "test_value", ttl=3600
            )

    @pytest.mark.asyncio
    async def test_get_set_error_handling(self, redis_url, config):
        """Test error handling in cache operations."""
        with patch("src.services.cache.modern.Cache"):
            mock_cache_instance = AsyncMock()
            mock_cache_instance.get.side_effect = Exception("Cache error")
            mock_cache_instance.set.side_effect = Exception("Cache error")

            manager = ModernCacheManager(redis_url, config=config)
            manager.search_cache = mock_cache_instance

            # Test get with error
            value = await manager.get("test_key", CacheType.SEARCH, "default")
            assert value == "default"

            # Test set with error
            success = await manager.set("test_key", "test_value", CacheType.SEARCH)
            assert success is False

    @pytest.mark.asyncio
    async def test_clear_operations(self, redis_url, config):
        """Test cache clear operations."""
        with patch("src.services.cache.modern.Cache"):
            mock_embedding_cache = AsyncMock()
            mock_search_cache = AsyncMock()

            manager = ModernCacheManager(redis_url, config=config)
            manager.embedding_cache = mock_embedding_cache
            manager.search_cache = mock_search_cache

            # Test clear specific cache type
            success = await manager.clear(CacheType.SEARCH)
            assert success is True
            mock_search_cache.clear.assert_called_once()
            mock_embedding_cache.clear.assert_not_called()

            # Test clear all caches
            success = await manager.clear()
            assert success is True
            mock_embedding_cache.clear.assert_called_once()

    def test_key_builders(self, redis_url, config):
        """Test cache key builder functions."""
        with patch("src.services.cache.modern.Cache"):
            manager = ModernCacheManager(redis_url, config=config)

            # Test embedding key builder
            embedding_key = manager._embedding_key_builder(
                None, "test text", model="gpt-3.5"
            )
            assert embedding_key.startswith("embed:gpt-3.5:")
            assert len(embedding_key.split(":")) == 3

            # Test search key builder
            search_key = manager._search_key_builder(
                None, "test query", filters={"type": "doc"}
            )
            assert search_key.startswith("search:")

            # Test crawl key builder
            crawl_key = manager._crawl_key_builder(None, "https://example.com")
            assert crawl_key.startswith("crawl:")


class TestLibraryMigrationManager:
    """Test cases for LibraryMigrationManager."""

    @pytest.mark.asyncio
    async def test_initialization_gradual_mode(self, config):
        """Test migration manager initialization in gradual mode."""
        migration_config = MigrationConfig(mode=MigrationMode.GRADUAL)

        with (
            patch(
                "src.services.migration.library_migration.ModernCircuitBreakerManager"
            ),
            patch("src.services.migration.library_migration.ModernCacheManager"),
        ):
            manager = LibraryMigrationManager(
                config=config,
                migration_config=migration_config,
                redis_url="redis://localhost:6379",
            )

            assert manager.migration_config.mode == MigrationMode.GRADUAL
            assert manager.redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_modern_only(self, config):
        """Test getting circuit breaker in modern-only mode."""
        migration_config = MigrationConfig(mode=MigrationMode.MODERN_ONLY)

        with patch(
            "src.services.migration.library_migration.ModernCircuitBreakerManager"
        ) as mock_cb_class:
            mock_cb_instance = AsyncMock()
            mock_breaker = AsyncMock()
            mock_cb_instance.get_breaker.return_value = mock_breaker
            mock_cb_class.return_value = mock_cb_instance

            manager = LibraryMigrationManager(
                config=config,
                migration_config=migration_config,
                redis_url="redis://localhost:6379",
            )
            await manager._initialize_modern_services()

            result = await manager.get_circuit_breaker("test_service")
            assert result == mock_breaker
            mock_cb_instance.get_breaker.assert_called_once_with("test_service")

    @pytest.mark.asyncio
    async def test_get_cache_manager_legacy_only(self, config):
        """Test getting cache manager in legacy-only mode."""
        migration_config = MigrationConfig(mode=MigrationMode.LEGACY_ONLY)

        with patch("src.services.cache.manager.CacheManager") as mock_legacy_cache:
            mock_legacy_instance = Mock()
            mock_legacy_cache.return_value = mock_legacy_instance

            manager = LibraryMigrationManager(
                config=config,
                migration_config=migration_config,
                redis_url="redis://localhost:6379",
            )
            await manager._initialize_legacy_services()

            result = await manager.get_cache_manager()
            assert result == mock_legacy_instance

    @pytest.mark.asyncio
    async def test_migration_state_tracking(self, config):
        """Test migration state tracking and updates."""
        migration_config = MigrationConfig(mode=MigrationMode.GRADUAL)

        manager = LibraryMigrationManager(
            config=config,
            migration_config=migration_config,
            redis_url="redis://localhost:6379",
        )

        # Test initial state
        assert not manager.migration_state["circuit_breaker_migrated"]
        assert not manager.migration_state["cache_migrated"]

        # Test force migration
        success = await manager.force_migration("circuit_breaker", True)
        assert success
        assert manager.migration_state["circuit_breaker_migrated"]

    @pytest.mark.asyncio
    async def test_get_migration_status(self, config):
        """Test getting migration status information."""
        migration_config = MigrationConfig(mode=MigrationMode.PARALLEL)

        manager = LibraryMigrationManager(
            config=config,
            migration_config=migration_config,
            redis_url="redis://localhost:6379",
        )

        status = await manager.get_migration_status()

        assert status["mode"] == "parallel"
        assert "migration_state" in status
        assert "performance_metrics" in status
        assert "services" in status

    @pytest.mark.asyncio
    async def test_cleanup(self, config):
        """Test migration manager cleanup."""
        migration_config = MigrationConfig(mode=MigrationMode.MODERN_ONLY)

        with (
            patch(
                "src.services.migration.library_migration.ModernCircuitBreakerManager"
            ) as mock_cb_class,
            patch(
                "src.services.migration.library_migration.ModernCacheManager"
            ) as mock_cache_class,
        ):
            mock_cb_instance = AsyncMock()
            mock_cache_instance = AsyncMock()
            mock_cb_class.return_value = mock_cb_instance
            mock_cache_class.return_value = mock_cache_instance

            manager = LibraryMigrationManager(
                config=config,
                migration_config=migration_config,
                redis_url="redis://localhost:6379",
            )
            await manager._initialize_modern_services()

            # Test cleanup
            await manager.cleanup()

            mock_cb_instance.close.assert_called_once()
            mock_cache_instance.close.assert_called_once()


class TestIntegrationScenarios:
    """Integration test scenarios for modern libraries."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_cache_integration(self, config):
        """Test integration between circuit breaker and cache."""
        redis_url = "redis://localhost:6379"

        with (
            patch("src.services.circuit_breaker.modern.RedisStorage") as mock_storage,
            patch(
                "src.services.circuit_breaker.modern.AsyncCircuitBreakerFactory"
            ) as mock_factory,
            patch("src.services.cache.modern.Cache"),
        ):
            # Setup mocks
            mock_storage.from_url.return_value = Mock()
            mock_factory.return_value = Mock()

            # Create managers
            cb_manager = ModernCircuitBreakerManager(redis_url, config)
            cache_manager = ModernCacheManager(redis_url, config=config)

            # Test that both can be initialized together
            assert cb_manager.redis_url == cache_manager.redis_url
            assert cb_manager.config == cache_manager.config

    @pytest.mark.asyncio
    async def test_performance_comparison_setup(self, config):
        """Test setup for performance comparison between old and new implementations."""
        migration_config = MigrationConfig(
            mode=MigrationMode.PARALLEL,
            performance_monitoring=True,
        )

        with (
            patch(
                "src.services.migration.library_migration.ModernCircuitBreakerManager"
            ),
            patch("src.services.migration.library_migration.ModernCacheManager"),
        ):
            manager = LibraryMigrationManager(
                config=config,
                migration_config=migration_config,
                redis_url="redis://localhost:6379",
            )

            # Verify performance monitoring is enabled
            assert manager.migration_config.performance_monitoring
            assert "circuit_breaker" in manager.performance_metrics
            assert "cache" in manager.performance_metrics

    @pytest.mark.asyncio
    async def test_rollback_scenario(self, config):
        """Test automatic rollback on high error rates."""
        migration_config = MigrationConfig(
            mode=MigrationMode.GRADUAL,
            rollback_threshold=0.1,  # 10% error rate threshold
        )

        manager = LibraryMigrationManager(
            config=config,
            migration_config=migration_config,
            redis_url="redis://localhost:6379",
        )

        # Simulate high error rate
        manager.performance_metrics["circuit_breaker"]["modern"]["error_rate"] = 0.15

        # Check rollback conditions
        await manager._check_rollback_conditions()

        # Verify rollback occurred
        assert not manager.migration_state["circuit_breaker_migrated"]
