"""Comprehensive tests for monitoring system initialization."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client.registry import CollectorRegistry

from src.config import MonitoringConfig
from src.services.monitoring.initialization import (
    cleanup_monitoring,
    initialize_monitoring_system,
    run_periodic_health_checks,
    setup_fastmcp_monitoring,
    start_background_monitoring_tasks,
    stop_background_monitoring_tasks,
    update_cache_metrics_periodically,
    update_system_metrics_periodically,
)


class TestError(Exception):
    """Custom exception for this module."""


class TestMonitoringInitialization:
    """Test monitoring system initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create mock unified configuration."""
        config = MagicMock()
        config.monitoring = MonitoringConfig(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=60.0,
        )
        config.cache = MagicMock()
        config.cache.enable_dragonfly_cache = False
        return config

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        client.get_collections.return_value = MagicMock(collections=[])
        return client

    @pytest.fixture
    def isolated_registry(self):
        """Create isolated Prometheus registry."""
        return CollectorRegistry()

    @pytest.mark.asyncio
    async def test_initialize_monitoring_enabled(
        self, mock_config, mock_qdrant_client, _isolated_registry
    ):
        """Test initialization with monitoring enabled."""
        with (
            patch(
                "src.services.monitoring.initialization.initialize_metrics"
            ) as mock_init_metrics,
            patch(
                "src.services.monitoring.initialization.HealthCheckManager"
            ) as mock_health,
        ):
            mock_metrics_instance = MagicMock()
            mock_init_metrics.return_value = mock_metrics_instance

            mock_health_instance = MagicMock()
            mock_health_instance.config = MagicMock()
            mock_health.return_value = mock_health_instance

            metrics_registry, health_manager = initialize_monitoring_system(
                mock_config, mock_qdrant_client, "redis://localhost"
            )

            assert metrics_registry is not None
            assert health_manager is not None
            mock_init_metrics.assert_called_once()
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_monitoring_disabled(self, mock_qdrant_client):
        """Test initialization with monitoring disabled."""
        config = MagicMock()
        config.monitoring = MonitoringConfig(enabled=False)

        metrics_registry, health_manager = initialize_monitoring_system(
            config, mock_qdrant_client, None
        )

        assert metrics_registry is None
        assert health_manager is None

    @pytest.mark.asyncio
    async def test_initialize_monitoring_partial_disabled(self, mock_qdrant_client):
        """Test initialization with partial monitoring features disabled."""
        config = MagicMock()
        config.monitoring = MonitoringConfig(enabled=True)
        config.cache = MagicMock()
        config.cache.enable_dragonfly_cache = False

        with (
            patch(
                "src.services.monitoring.initialization.initialize_metrics"
            ) as mock_init_metrics,
            patch(
                "src.services.monitoring.initialization.HealthCheckManager"
            ) as mock_health,
        ):
            mock_health_instance = MagicMock()
            mock_health_instance.config = MagicMock()
            mock_health.return_value = mock_health_instance

            mock_metrics_instance = MagicMock()
            mock_init_metrics.return_value = mock_metrics_instance

            metrics_registry, health_manager = initialize_monitoring_system(
                config, mock_qdrant_client, None
            )

            assert metrics_registry is not None
            assert health_manager is not None

    @pytest.mark.asyncio
    async def test_cleanup_monitoring(self):
        """Test monitoring system cleanup."""
        mock_metrics = MagicMock()
        mock_health = MagicMock()

        # Should not raise any exceptions
        await cleanup_monitoring(mock_metrics, mock_health)

    @pytest.mark.asyncio
    async def test_cleanup_monitoring_with_none(self):
        """Test cleanup with None values."""
        # Should not raise exceptions
        await cleanup_monitoring(None, None)


class TestBackgroundMonitoringTasks:
    """Test background monitoring task management."""

    @pytest.fixture
    def mock_metrics_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock()
        registry.update_system_metrics = MagicMock()
        return registry

    @pytest.fixture
    def mock_health_manager(self):
        """Create mock health manager."""
        manager = MagicMock()
        manager.check_all = AsyncMock(return_value={})
        return manager

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        manager = MagicMock()
        manager.get_stats = AsyncMock(
            return_value={"local_cache": {"hits": 10, "misses": 2}}
        )
        return manager

    @pytest.mark.asyncio
    async def test_start_background_tasks_enabled(
        self, mock_metrics_registry, mock_health_manager, _mock_cache_manager
    ):
        """Test starting background tasks when enabled."""
        # Configure the mock registries to have the right config attributes
        mock_metrics_registry.config = MagicMock()
        mock_metrics_registry.config.enabled = True
        mock_metrics_registry.config.include_system_metrics = True
        mock_metrics_registry.config.collection_interval = 0.1

        mock_health_manager.config = MagicMock()
        mock_health_manager.config.enabled = True
        mock_health_manager.config.interval = 0.1

        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry, mock_health_manager
        )

        assert len(tasks) > 0

        # Let tasks run briefly
        await asyncio.sleep(0.2)

        # Cleanup
        await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_start_background_tasks_disabled(
        self, mock_metrics_registry, mock_health_manager
    ):
        """Test starting background tasks when disabled."""
        # Configure the mock registries to be disabled
        mock_metrics_registry.config = MagicMock()
        mock_metrics_registry.config.enabled = False

        mock_health_manager.config = MagicMock()
        mock_health_manager.config.enabled = False

        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry, mock_health_manager
        )

        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_start_background_tasks_with_none(self):
        """Test starting background tasks with None managers."""
        tasks = await start_background_monitoring_tasks(None, None)

        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_background_tasks(self):
        """Test stopping background tasks."""

        # Create real task objects that can be cancelled
        async def dummy_task():
            await asyncio.sleep(1)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        tasks = [task1, task2]

        await stop_background_monitoring_tasks(tasks)

        # Tasks should be cancelled
        assert task1.cancelled()
        assert task2.cancelled()

    @pytest.mark.asyncio
    async def test_stop_background_tasks_empty_list(self):
        """Test stopping empty task list."""
        # Should not raise exceptions
        await stop_background_monitoring_tasks([])

    @pytest.mark.asyncio
    async def test_system_metrics_task_execution(self, mock_metrics_registry):
        """Test system metrics task execution."""
        # Create a task that runs once
        task = asyncio.create_task(
            update_system_metrics_periodically(
                mock_metrics_registry, interval_seconds=0.01
            )
        )

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Cancel and cleanup
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Verify metrics were updated
        assert mock_metrics_registry.update_system_metrics.call_count > 0

    @pytest.mark.asyncio
    async def test_health_check_task_execution(self, mock_health_manager):
        """Test health check task execution."""
        task = asyncio.create_task(
            run_periodic_health_checks(mock_health_manager, interval_seconds=0.01)
        )

        await asyncio.sleep(0.05)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert mock_health_manager.check_all.call_count > 0

    @pytest.mark.asyncio
    async def test_task_error_handling(self, mock_metrics_registry):
        """Test task error handling and recovery."""
        # Mock registry to raise exception first time, then succeed
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "Test error"
                raise TestError(msg)

        mock_metrics_registry.update_system_metrics.side_effect = side_effect

        task = asyncio.create_task(
            update_system_metrics_periodically(
                mock_metrics_registry, interval_seconds=0.01
            )
        )

        await asyncio.sleep(0.05)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Task should continue running despite error
        assert mock_metrics_registry.update_system_metrics.call_count >= 2


class TestMonitoringIntegration:
    """Test monitoring system integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_monitoring_lifecycle(self):
        """Test complete monitoring lifecycle."""
        config = MagicMock()
        config.monitoring = MonitoringConfig(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=0.1,
        )
        config.cache = MagicMock()
        config.cache.enable_dragonfly_cache = False

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        with (
            patch(
                "src.services.monitoring.metrics.get_metrics_registry",
                return_value=None,
            ),
            patch("src.services.monitoring.metrics.MetricsRegistry") as mock_metrics,
            patch(
                "src.services.monitoring.initialization.HealthCheckManager"
            ) as mock_health,
        ):
            mock_metrics_instance = MagicMock()
            mock_metrics.return_value = mock_metrics_instance

            mock_health_instance = MagicMock()
            mock_health_instance.check_all = AsyncMock(return_value={})
            mock_health_instance.cleanup = AsyncMock()
            mock_health.return_value = mock_health_instance

            # Initialize
            metrics_registry, health_manager = initialize_monitoring_system(
                config, mock_qdrant, "redis://localhost"
            )

            # Start background tasks
            tasks = await start_background_monitoring_tasks(
                metrics_registry, health_manager
            )

            # Let tasks run
            await asyncio.sleep(0.2)

            # Cleanup
            await stop_background_monitoring_tasks(tasks)
            await cleanup_monitoring(metrics_registry, health_manager)

            assert len(tasks) > 0
            # Cleanup function doesn't actually call methods on the health manager
            # It just performs internal cleanup

    @pytest.mark.asyncio
    async def test_monitoring_with_custom_intervals(self):
        """Test monitoring with custom intervals."""
        MonitoringConfig(
            enabled=True,
            include_system_metrics=True,
            system_metrics_interval=0.05,  # 50ms
        )

        mock_metrics = MagicMock()
        mock_metrics.update_system_metrics = MagicMock()

        task = asyncio.create_task(
            update_system_metrics_periodically(mock_metrics, interval_seconds=0.05)
        )

        await asyncio.sleep(0.2)  # Let it run for 200ms

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should have called multiple times with 50ms interval
        assert mock_metrics.update_system_metrics.call_count >= 3

    @pytest.mark.asyncio
    async def test_partial_monitoring_configuration(self, mock_qdrant_client):
        """Test partial monitoring configuration."""
        config = MagicMock()
        config.monitoring = MonitoringConfig(
            enabled=True,
            include_system_metrics=False,
        )
        config.cache = MagicMock()
        config.cache.enable_dragonfly_cache = False

        with patch(
            "src.services.monitoring.initialization.HealthCheckManager"
        ) as mock_health:
            mock_health_instance = MagicMock()
            mock_health.return_value = mock_health_instance

            metrics_registry, health_manager = initialize_monitoring_system(
                config, mock_qdrant_client, None
            )

            tasks = await start_background_monitoring_tasks(
                metrics_registry, health_manager
            )

            # Monitoring should still be enabled, just without system metrics
            assert metrics_registry is not None
            assert health_manager is not None
            # Background tasks should be minimal since include_system_metrics=False

            await stop_background_monitoring_tasks(tasks)


class TestFastMCPIntegration:
    """Test FastMCP monitoring integration."""

    @pytest.mark.asyncio
    async def test_setup_fastmcp_monitoring(self):
        """Test FastMCP monitoring setup."""
        mock_mcp = MagicMock()
        config = MagicMock()
        config.monitoring = MonitoringConfig(enabled=True)
        config.monitoring.health_path = "/health"
        mock_metrics = MagicMock()
        mock_health = MagicMock()

        # Mock the FastMCP app to have an underlying FastAPI app
        mock_mcp.app = MagicMock()

        # This function mainly adds health endpoints to FastAPI app
        setup_fastmcp_monitoring(mock_mcp, config, mock_metrics, mock_health)

        # Verify that health endpoints were added to the FastAPI app
        assert mock_mcp.app.get.call_count >= 1  # Should add multiple health endpoints

    @pytest.mark.asyncio
    async def test_cache_metrics_periodic_update(self):
        """Test periodic cache metrics updates."""
        mock_metrics = MagicMock()
        # Ensure the mock is truthy
        mock_metrics.__bool__ = MagicMock(return_value=True)

        mock_cache_manager = MagicMock()
        mock_cache_manager.__bool__ = MagicMock(return_value=True)
        mock_cache_manager.get_stats = AsyncMock(
            return_value={"local_cache": {"hits": 10, "misses": 2, "size": 1024}}
        )

        task = asyncio.create_task(
            update_cache_metrics_periodically(
                mock_metrics, mock_cache_manager, interval_seconds=0.01
            )
        )

        # Let it run for a bit longer to ensure at least one execution
        await asyncio.sleep(0.1)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # The function should have called the metrics registry to update cache stats
        assert mock_metrics.update_cache_stats.call_count > 0
