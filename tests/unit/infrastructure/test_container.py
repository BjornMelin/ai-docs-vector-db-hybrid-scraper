"""Tests for container factory functions."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.config import Settings
from src.config.models import CacheConfig, Environment
from src.infrastructure import container as container_module


class TestCreateQdrantClient:
    """Tests for _create_qdrant_client factory function."""

    def test_creates_client_with_valid_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create AsyncQdrantClient with proper configuration."""
        with patch.object(
            container_module, "AsyncQdrantClient", return_value=AsyncMock()
        ) as mock_cls:
            result = container_module._create_qdrant_client(minimal_config_namespace)

            mock_cls.assert_called_once()
            assert result is not None

    def test_extracts_url_from_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should extract URL from qdrant config."""
        with patch.object(
            container_module, "AsyncQdrantClient", return_value=AsyncMock()
        ) as mock_cls:
            container_module._create_qdrant_client(minimal_config_namespace)

            call_kwargs = mock_cls.call_args.kwargs
            assert "url" in call_kwargs or "location" in call_kwargs

    def test_handles_missing_qdrant_config(self) -> None:
        """Should handle missing qdrant configuration gracefully."""
        config = SimpleNamespace()

        with patch.object(
            container_module, "AsyncQdrantClient", return_value=AsyncMock()
        ):
            # Should not raise - uses defaults
            result = container_module._create_qdrant_client(config)
            assert result is not None


class TestCreateDragonflyClient:
    """Tests for _create_dragonfly_client factory function."""

    def test_creates_client_with_valid_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create Redis client with valid configuration."""
        mock_redis = MagicMock()

        with patch.object(
            container_module.redis, "from_url", return_value=mock_redis
        ) as mock_from_url:
            result = container_module._create_dragonfly_client(minimal_config_namespace)

            mock_from_url.assert_called()
            assert result is mock_redis

    def test_uses_default_url_on_config_error(self) -> None:
        """Should fall back to default URL when config extraction fails."""
        mock_redis = MagicMock()
        config = SimpleNamespace()  # Missing cache attribute

        with patch.object(
            container_module.redis, "from_url", return_value=mock_redis
        ) as mock_from_url:
            result = container_module._create_dragonfly_client(config)

            # Should have called from_url (possibly twice if fallback)
            assert mock_from_url.called
            assert result is mock_redis


class TestCreateFirecrawlClient:
    """Tests for _create_firecrawl_client factory function."""

    def test_creates_client_with_api_key(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create Firecrawl client with API key."""
        mock_client = MagicMock()
        mock_module = MagicMock()
        mock_client_cls = MagicMock(return_value=mock_client)
        mock_module.AsyncFirecrawlApp = mock_client_cls

        with patch("importlib.import_module", return_value=mock_module):
            result = container_module._create_firecrawl_client(minimal_config_namespace)

            mock_client_cls.assert_called_once()
            assert result is mock_client

    def test_returns_none_when_module_not_found(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should return None when Firecrawl module is not installed."""
        with patch(
            "importlib.import_module", side_effect=ModuleNotFoundError("not found")
        ):
            result = container_module._create_firecrawl_client(minimal_config_namespace)

            assert result is None

    def test_returns_none_when_async_client_missing(self) -> None:
        """Should return None when module lacks async client class."""
        mock_module = MagicMock(spec=[])  # Module without AsyncFirecrawlApp

        with patch("importlib.import_module", return_value=mock_module):
            result = container_module._create_firecrawl_client(
                SimpleNamespace(firecrawl=SimpleNamespace(api_key="test"))
            )

            assert result is None


class TestCreateCacheManager:
    """Tests for _create_cache_manager factory function."""

    def test_creates_manager_with_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create CacheManager from configuration."""
        with patch.object(
            container_module, "CacheManager", return_value=MagicMock()
        ) as mock_cls:
            result = container_module._create_cache_manager(minimal_config_namespace)

            mock_cls.assert_called_once()
            assert result is not None

    def test_uses_defaults_when_cache_config_missing(self) -> None:
        """Should use default values when cache config is missing."""
        config = SimpleNamespace()  # No cache attribute

        with patch.object(
            container_module, "CacheManager", return_value=MagicMock()
        ) as mock_cls:
            result = container_module._create_cache_manager(config)

            # Should still create manager with defaults
            mock_cls.assert_called_once()
            assert result is not None


@pytest.mark.service
class TestCreateCircuitBreakerManager:
    """Tests for _create_circuit_breaker_manager factory function."""

    def test_creates_manager_with_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should use in-memory state when Dragonfly is disabled."""
        with patch.object(
            container_module.CircuitBreakerManager,
            "in_memory",
            return_value=MagicMock(),
        ) as mock_cls:
            result = container_module._create_circuit_breaker_manager(
                minimal_config_namespace
            )

            mock_cls.assert_called_once_with(config=minimal_config_namespace)
            assert result is not None

    def test_creates_distributed_manager_when_dragonfly_enabled(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Dragonfly-enabled settings should retain distributed breaker state."""
        minimal_config_namespace.cache.enable_caching = True
        minimal_config_namespace.cache.enable_dragonfly_cache = True
        with patch.object(
            container_module, "CircuitBreakerManager", return_value=MagicMock()
        ) as mock_cls:
            result = container_module._create_circuit_breaker_manager(
                minimal_config_namespace
            )

            mock_cls.assert_called_once_with(
                redis_url=minimal_config_namespace.cache.dragonfly_url,
                config=minimal_config_namespace,
            )
            assert result is not None


class TestCreateProjectStorage:
    """Tests for _create_project_storage factory function."""

    def test_creates_storage_with_data_dir(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create ProjectStorage with data directory."""
        with patch.object(
            container_module, "ProjectStorage", return_value=MagicMock()
        ) as mock_cls:
            result = container_module._create_project_storage(minimal_config_namespace)

            mock_cls.assert_called_once()
            assert result is not None


@pytest.mark.rag
def test_create_rag_generator_skips_disabled_feature() -> None:
    """Disabled RAG should not import or construct provider dependencies."""
    config = SimpleNamespace(rag=SimpleNamespace(enable_rag=False))

    with patch.object(container_module.importlib, "import_module") as importer:
        result = container_module._create_rag_generator(config, MagicMock())

    assert result is None
    importer.assert_not_called()


class TestCreateBrowserManager:
    """Tests for _create_browser_manager factory function."""

    def test_creates_manager_with_valid_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create BrowserManager with valid configuration."""
        # Create a mock module that has a UnifiedBrowserManager class
        mock_manager = MagicMock()
        mock_module = MagicMock()
        mock_module.UnifiedBrowserManager = MagicMock(return_value=mock_manager)

        with patch("importlib.import_module", return_value=mock_module):
            result = container_module._create_browser_manager(minimal_config_namespace)

            mock_module.UnifiedBrowserManager.assert_called_once_with(
                minimal_config_namespace
            )
            assert result is mock_manager

    def test_returns_none_when_module_not_found(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should return None when browser module is not available."""
        with patch(
            "importlib.import_module", side_effect=ModuleNotFoundError("not found")
        ):
            result = container_module._create_browser_manager(minimal_config_namespace)

            assert result is None


@pytest.mark.service
class TestApplicationContainer:
    """Tests for ApplicationContainer class."""

    def test_container_preserves_settings_identity(self) -> None:
        """Container services should receive the canonical Settings instance."""
        settings = Settings(
            environment=Environment.TESTING,
            cache=CacheConfig(enable_caching=False, enable_dragonfly_cache=False),
        )
        container = container_module.ApplicationContainer(config=settings)

        embedding_manager = container.embedding_manager()

        assert container.config() is settings
        assert embedding_manager.config is settings
        assert embedding_manager.cache_manager is container.cache_manager()
        assert container.cache_manager().distributed_cache is None


@pytest.mark.service
class TestContainerManager:
    """Tests for ContainerManager singleton."""

    @pytest.mark.asyncio
    async def test_cleanup_graph_continues_after_os_error(self) -> None:
        """One cleanup failure should not skip later services."""
        service_names = (
            "rag_generator",
            "browser_manager",
            "content_intelligence_service",
            "vector_store_service",
            "embedding_manager",
            "cache_manager",
            "project_storage",
            "circuit_breaker_manager",
        )
        services = {name: MagicMock() for name in service_names}
        for service in services.values():
            service.cleanup = AsyncMock()
        services["rag_generator"].cleanup.side_effect = OSError("cleanup failed")

        resolved = [
            container_module._ResolvedService(name, services[name])
            for name in service_names
        ]

        await container_module._cleanup_service_graph(resolved)

        for service in services.values():
            service.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_graph_closes_services_without_cleanup(self) -> None:
        """Close-only services should participate in reverse graph teardown."""
        close = AsyncMock()

        await container_module._cleanup_service_graph(
            [
                container_module._ResolvedService(
                    "close_only", SimpleNamespace(close=close)
                )
            ]
        )

        close.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_cleanup_graph_awaits_custom_awaitable(self) -> None:
        """Cleanup should await every awaitable, not only coroutine objects."""

        class CleanupAwaitable:
            def __init__(self) -> None:
                self.awaited = False

            def __await__(self) -> Generator[None, None, None]:
                self.awaited = True
                yield from ()
                return None

        result = CleanupAwaitable()
        cleanup = MagicMock(return_value=result)

        await container_module._cleanup_service_graph(
            [
                container_module._ResolvedService(
                    "custom_awaitable", SimpleNamespace(cleanup=cleanup)
                )
            ]
        )

        cleanup.assert_called_once_with()
        assert result.awaited

    @pytest.mark.asyncio
    async def test_dependency_context_releases_its_own_lease(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlapping contexts should release only their own ownership token."""
        container = MagicMock()
        leases = [
            container_module.ContainerLease(container, generation=1, lease_id=1),
            container_module.ContainerLease(container, generation=1, lease_id=2),
        ]
        acquire = AsyncMock(side_effect=leases)
        release = AsyncMock()
        monkeypatch.setattr(container_module, "acquire_container", acquire)
        monkeypatch.setattr(container_module, "release_container", release)
        first = container_module.DependencyContext(Settings())
        second = container_module.DependencyContext(Settings())

        assert await first.__aenter__() is container
        assert await second.__aenter__() is container
        await first.__aexit__(None, None, None)

        release.assert_awaited_once_with(leases[0])
        assert second.container is container

        await second.__aexit__(None, None, None)
        assert release.await_args_list == [call(leases[0]), call(leases[1])]

    def test_get_container_is_callable(self) -> None:
        """get_container should be a callable function."""
        from src.infrastructure.container import get_container

        # Verify the function exists and is callable
        assert callable(get_container)
        # Call it - result depends on whether container was initialized
        result = get_container()
        # Result should be None or an ApplicationContainer
        assert result is None or hasattr(result, "qdrant_client")

    @pytest.mark.asyncio
    async def test_initialize_container_sets_up_services(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ContainerManager should preserve Settings through initialization."""
        settings = Settings(
            environment=Environment.TESTING,
            cache=CacheConfig(enable_caching=False, enable_dragonfly_cache=False),
        )
        manager = container_module.ContainerManager()
        graph_probe = AsyncMock()
        cleanup_probe = AsyncMock()
        monkeypatch.setattr(container_module, "_initialize_service_graph", graph_probe)
        monkeypatch.setattr(container_module, "_cleanup_service_graph", cleanup_probe)
        try:
            container = await manager.initialize(settings)

            assert container.config() is settings
            assert container.embedding_manager().config is settings
            assert container.cache_manager().distributed_cache is None
            graph_probe.assert_awaited_once()
            assert graph_probe.await_args is not None
            assert graph_probe.await_args.args[0] is container
            assert graph_probe.await_args.args[1] == []
        finally:
            await manager.shutdown()

        cleanup_probe.assert_awaited_once_with([])

    @pytest.mark.asyncio
    async def test_initialize_serializes_concurrent_callers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent startup requests should publish exactly one container."""
        settings = Settings(environment=Environment.TESTING)
        candidate = MagicMock()
        candidate.init_resources = AsyncMock()
        candidate.qdrant_client.return_value = AsyncMock()
        candidate.startup_tasks.return_value = []
        factory = MagicMock(return_value=candidate)

        async def yield_during_initialization(
            _container: object,
            _resolved_services: list[object],
        ) -> None:
            await asyncio.sleep(0)

        graph_probe = AsyncMock(side_effect=yield_during_initialization)
        monkeypatch.setattr(container_module, "ApplicationContainer", factory)
        monkeypatch.setattr(container_module, "_initialize_service_graph", graph_probe)
        manager = container_module.ContainerManager()

        first, second = await asyncio.gather(
            manager.initialize(settings),
            manager.initialize(settings),
        )

        assert first is candidate
        assert second is candidate
        factory.assert_called_once_with(config=settings)
        candidate.init_resources.assert_awaited_once()
        graph_probe.assert_awaited_once()
        assert graph_probe.await_args is not None
        assert graph_probe.await_args.args == (candidate, [])

    @pytest.mark.asyncio
    async def test_initialize_rolls_back_failed_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed startup should release resources without publishing the candidate."""
        settings = Settings(environment=Environment.TESTING)
        candidate = MagicMock()
        candidate.init_resources = AsyncMock()
        candidate.shutdown_resources = AsyncMock()
        qdrant_client = AsyncMock()
        candidate.qdrant_client.return_value = qdrant_client

        async def fail_startup() -> None:
            raise RuntimeError("startup failed")

        candidate.startup_tasks.return_value = [fail_startup]
        resolved_service = container_module._ResolvedService("stub", MagicMock())

        async def record_service(
            _container: object,
            resolved_services: list[object],
        ) -> None:
            resolved_services.append(resolved_service)

        graph_probe = AsyncMock(side_effect=record_service)
        cleanup_probe = AsyncMock()
        monkeypatch.setattr(
            container_module,
            "ApplicationContainer",
            MagicMock(return_value=candidate),
        )
        monkeypatch.setattr(container_module, "_initialize_service_graph", graph_probe)
        monkeypatch.setattr(container_module, "_cleanup_service_graph", cleanup_probe)
        manager = container_module.ContainerManager()

        with pytest.raises(RuntimeError, match="startup failed"):
            await manager.initialize(settings)

        assert manager.container is None
        assert manager._initialized is False  # pylint: disable=protected-access
        cleanup_probe.assert_awaited_once_with([resolved_service])
        qdrant_client.close.assert_awaited_once_with()
        candidate.shutdown_resources.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_graph_cleans_only_resolved_services_in_reverse(self) -> None:
        """Rollback must not resolve new providers after a partial startup failure."""
        cleanup_order: list[str] = []

        def service(name: str, *, fail: bool = False) -> MagicMock:
            instance = MagicMock()
            instance.initialize = AsyncMock(
                side_effect=RuntimeError("failed") if fail else None
            )

            async def cleanup() -> None:
                cleanup_order.append(name)

            instance.cleanup = AsyncMock(side_effect=cleanup)
            return instance

        cache = service("cache")
        embedding = service("embedding", fail=True)
        container = MagicMock()
        container.cache_manager.return_value = cache
        container.embedding_manager.return_value = embedding
        resolved: list[container_module._ResolvedService] = []

        with pytest.raises(RuntimeError, match="embedding_manager"):
            await container_module._initialize_service_graph(container, resolved)
        await container_module._cleanup_service_graph(resolved)

        assert [item.instance for item in resolved] == [cache, embedding]
        container.vector_store_service.assert_not_called()
        assert cleanup_order == ["embedding", "cache"]

    @pytest.mark.asyncio
    async def test_overlapping_leases_share_until_last_release(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One session ending must not destroy a container another session holds."""
        settings = Settings(environment=Environment.TESTING)
        candidate = MagicMock()
        candidate.init_resources = AsyncMock()
        candidate.shutdown_resources = AsyncMock()
        qdrant_client = AsyncMock()
        candidate.qdrant_client.return_value = qdrant_client
        candidate.startup_tasks.return_value = []
        candidate.shutdown_tasks.return_value = []
        monkeypatch.setattr(
            container_module,
            "ApplicationContainer",
            MagicMock(return_value=candidate),
        )
        monkeypatch.setattr(
            container_module,
            "_initialize_service_graph",
            AsyncMock(),
        )
        manager = container_module.ContainerManager()

        first = await manager.acquire(settings)
        second = await manager.acquire(settings)
        await manager.release(first)

        assert first.container is second.container is candidate
        assert manager.container is candidate
        candidate.shutdown_resources.assert_not_awaited()

        await manager.release(second)

        assert manager.container is None
        qdrant_client.close.assert_awaited_once_with()
        candidate.shutdown_resources.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_force_reload_and_shutdown_are_rejected_while_leased(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Destructive replacement cannot cross an active generation lease."""
        settings = Settings(environment=Environment.TESTING)
        candidate = MagicMock()
        candidate.init_resources = AsyncMock()
        candidate.shutdown_resources = AsyncMock()
        candidate.qdrant_client.return_value = AsyncMock()
        candidate.startup_tasks.return_value = []
        candidate.shutdown_tasks.return_value = []
        monkeypatch.setattr(
            container_module,
            "ApplicationContainer",
            MagicMock(return_value=candidate),
        )
        monkeypatch.setattr(
            container_module,
            "_initialize_service_graph",
            AsyncMock(),
        )
        manager = container_module.ContainerManager()
        lease = await manager.acquire(settings)

        with pytest.raises(RuntimeError, match="force-reload"):
            await manager.acquire(settings, force_reload=True)
        with pytest.raises(RuntimeError, match="sessions are active"):
            await manager.shutdown()

        assert manager.container is candidate
        candidate.shutdown_resources.assert_not_awaited()

        await manager.release(lease)
        with pytest.raises(RuntimeError, match="no longer active"):
            await manager.release(lease)

        assert manager.container is None
        candidate.shutdown_resources.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_shutdown_container_cleans_up(self) -> None:
        """shutdown_container should clean up resources."""
        from src.infrastructure.container import (
            _container_manager,
            shutdown_container,
        )

        # Create a mock container for shutdown
        mock_container = MagicMock()
        mock_container.shutdown_resources = AsyncMock()
        mock_container.shutdown_tasks = MagicMock(return_value=[])
        original_container = _container_manager.container
        original_initialized = _container_manager._initialized

        # Set up the manager state so shutdown will run
        _container_manager.container = mock_container
        _container_manager._initialized = True

        try:
            # Should not raise
            await shutdown_container()
            mock_container.shutdown_resources.assert_awaited_once()
        finally:
            _container_manager.container = original_container
            _container_manager._initialized = original_initialized
