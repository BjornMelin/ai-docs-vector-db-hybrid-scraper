"""Tests for container factory functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class TestCreateCircuitBreakerManager:
    """Tests for _create_circuit_breaker_manager factory function."""

    def test_creates_manager_with_config(
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """Should create CircuitBreakerManager with configuration."""
        with patch.object(
            container_module, "CircuitBreakerManager", return_value=MagicMock()
        ) as mock_cls:
            result = container_module._create_circuit_breaker_manager(
                minimal_config_namespace
            )

            mock_cls.assert_called_once()
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


class TestApplicationContainer:
    """Tests for ApplicationContainer class."""

    def test_container_provides_qdrant_client(self) -> None:
        """Container should provide qdrant_client provider."""
        from src.infrastructure.container import ApplicationContainer

        container = ApplicationContainer()

        assert hasattr(container, "qdrant_client")

    def test_container_provides_vector_store_service(self) -> None:
        """Container should provide vector_store_service provider."""
        from src.infrastructure.container import ApplicationContainer

        container = ApplicationContainer()

        assert hasattr(container, "vector_store_service")

    def test_container_provides_cache_manager(self) -> None:
        """Container should provide cache_manager provider."""
        from src.infrastructure.container import ApplicationContainer

        container = ApplicationContainer()

        assert hasattr(container, "cache_manager")

    def test_container_provides_embedding_manager(self) -> None:
        """Container should provide embedding_manager provider."""
        from src.infrastructure.container import ApplicationContainer

        container = ApplicationContainer()

        assert hasattr(container, "embedding_manager")


class TestContainerManager:
    """Tests for ContainerManager singleton."""

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
        self, minimal_config_namespace: SimpleNamespace
    ) -> None:
        """initialize_container should set up the container."""
        from src.infrastructure.container import (
            _container_manager,
            initialize_container,
        )

        # Save original state
        original_container = _container_manager.container

        try:
            with patch.object(
                container_module, "ApplicationContainer"
            ) as mock_container_cls:
                mock_container = MagicMock()
                mock_container.init_resources = AsyncMock()
                mock_container_cls.return_value = mock_container

                await initialize_container(minimal_config_namespace)

                # Should have created container
                mock_container_cls.assert_called_once()
        finally:
            # Restore original state to avoid affecting other tests
            _container_manager.container = original_container

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
