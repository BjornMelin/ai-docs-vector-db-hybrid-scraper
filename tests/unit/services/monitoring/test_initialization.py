"""Tests for monitoring initialization functionality."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from src.config.models import MonitoringConfig
from src.services.monitoring.health import HealthCheckConfig
from src.services.monitoring.health import HealthCheckManager
from src.services.monitoring.initialization import cleanup_monitoring
from src.services.monitoring.initialization import initialize_monitoring
from src.services.monitoring.initialization import start_background_monitoring_tasks
from src.services.monitoring.initialization import stop_background_monitoring_tasks
from src.services.monitoring.metrics import MetricsConfig
from src.services.monitoring.metrics import MetricsRegistry


class TestMonitoringInitialization:
    """Test monitoring system initialization."""

    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration."""
        return MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(enabled=True),
            health_checks=HealthCheckConfig(enabled=True)
        )

    @pytest.mark.asyncio
    async def test_initialize_monitoring_enabled(self, monitoring_config):
        """Test monitoring initialization when enabled."""
        metrics_registry, health_manager = await initialize_monitoring(monitoring_config)

        assert metrics_registry is not None
        assert health_manager is not None
        assert isinstance(metrics_registry, MetricsRegistry)
        assert isinstance(health_manager, HealthCheckManager)
        assert metrics_registry.config.enabled is True
        assert health_manager.config.enabled is True

    @pytest.mark.asyncio
    async def test_initialize_monitoring_disabled(self):
        """Test monitoring initialization when disabled."""
        config = MonitoringConfig(enabled=False)
        metrics_registry, health_manager = await initialize_monitoring(config)

        assert metrics_registry is None
        assert health_manager is None

    @pytest.mark.asyncio
    async def test_initialize_monitoring_partial_disabled(self):
        """Test monitoring with some components disabled."""
        config = MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(enabled=False),
            health_checks=HealthCheckConfig(enabled=True)
        )

        metrics_registry, health_manager = await initialize_monitoring(config)

        assert metrics_registry is not None
        assert health_manager is not None
        assert metrics_registry.config.enabled is False
        assert health_manager.config.enabled is True

    @pytest.mark.asyncio
    async def test_cleanup_monitoring(self, monitoring_config):
        """Test monitoring cleanup."""
        metrics_registry, health_manager = await initialize_monitoring(monitoring_config)

        # Should not raise any errors
        await cleanup_monitoring(metrics_registry, health_manager)

    @pytest.mark.asyncio
    async def test_cleanup_monitoring_with_none(self):
        """Test cleanup with None values."""
        # Should handle None values gracefully
        await cleanup_monitoring(None, None)


class TestBackgroundMonitoringTasks:
    """Test background monitoring task management."""

    @pytest.fixture
    def mock_metrics_registry(self):
        """Create mock metrics registry."""
        registry = Mock(spec=MetricsRegistry)
        registry.config = MetricsConfig(
            enabled=True,
            collect_system_metrics=True,
            system_metrics_interval=0.1  # Short interval for testing
        )
        registry.update_system_metrics = Mock()
        return registry

    @pytest.fixture
    def mock_health_manager(self):
        """Create mock health manager."""
        manager = Mock(spec=HealthCheckManager)
        manager.config = HealthCheckConfig(enabled=True, interval=0.1)
        manager.get_overall_health = AsyncMock(return_value=("healthy", {}))
        manager.update_service_health = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_start_background_tasks_enabled(self, mock_metrics_registry, mock_health_manager):
        """Test starting background tasks when enabled."""
        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry,
            mock_health_manager
        )

        assert len(tasks) == 2  # System metrics + health checks
        assert all(isinstance(task, asyncio.Task) for task in tasks)

        # Let tasks run briefly
        await asyncio.sleep(0.2)

        # Clean up
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_background_tasks_disabled(self):
        """Test starting background tasks when disabled."""
        disabled_registry = Mock(spec=MetricsRegistry)
        disabled_registry.config = MetricsConfig(enabled=False)

        disabled_manager = Mock(spec=HealthCheckManager)
        disabled_manager.config = HealthCheckConfig(enabled=False)

        tasks = await start_background_monitoring_tasks(
            disabled_registry,
            disabled_manager
        )

        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_start_background_tasks_with_none(self):
        """Test starting background tasks with None values."""
        tasks = await start_background_monitoring_tasks(None, None)
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_background_tasks(self, mock_metrics_registry, mock_health_manager):
        """Test stopping background tasks."""
        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry,
            mock_health_manager
        )

        assert len(tasks) > 0
        assert all(not task.done() for task in tasks)

        await stop_background_monitoring_tasks(tasks)

        # Tasks should be cancelled
        assert all(task.done() for task in tasks)

    @pytest.mark.asyncio
    async def test_stop_background_tasks_empty_list(self):
        """Test stopping empty task list."""
        # Should not raise any errors
        await stop_background_monitoring_tasks([])

    @pytest.mark.asyncio
    async def test_system_metrics_task_execution(self, mock_metrics_registry):
        """Test that system metrics task executes correctly."""
        mock_health_manager = Mock(spec=HealthCheckManager)
        mock_health_manager.config = HealthCheckConfig(enabled=False)

        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry,
            mock_health_manager
        )

        # Should have one task for system metrics
        assert len(tasks) == 1

        # Let it run briefly
        await asyncio.sleep(0.15)

        # Should have called update_system_metrics at least once
        assert mock_metrics_registry.update_system_metrics.called

        # Clean up
        await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_health_check_task_execution(self, mock_health_manager):
        """Test that health check task executes correctly."""
        mock_metrics_registry = Mock(spec=MetricsRegistry)
        mock_metrics_registry.config = MetricsConfig(
            enabled=True,
            collect_system_metrics=False
        )

        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry,
            mock_health_manager
        )

        # Should have one task for health checks
        assert len(tasks) == 1

        # Let it run briefly
        await asyncio.sleep(0.15)

        # Should have called get_overall_health at least once
        assert mock_health_manager.get_overall_health.called

        # Clean up
        await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_task_error_handling(self, mock_metrics_registry, mock_health_manager):
        """Test that tasks handle errors gracefully."""
        # Make system metrics update fail
        mock_metrics_registry.update_system_metrics.side_effect = Exception("Test error")

        # Make health check fail
        mock_health_manager.get_overall_health.side_effect = Exception("Health check error")

        tasks = await start_background_monitoring_tasks(
            mock_metrics_registry,
            mock_health_manager
        )

        # Let tasks run and encounter errors
        await asyncio.sleep(0.15)

        # Tasks should still be running (error handling should prevent crashes)
        assert all(not task.done() for task in tasks)

        # Clean up
        await stop_background_monitoring_tasks(tasks)


class TestMonitoringIntegration:
    """Integration tests for complete monitoring initialization."""

    @pytest.mark.asyncio
    async def test_full_monitoring_lifecycle(self):
        """Test complete monitoring initialization and cleanup lifecycle."""
        config = MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(
                enabled=True,
                collect_system_metrics=True,
                system_metrics_interval=0.1
            ),
            health_checks=HealthCheckConfig(
                enabled=True,
                interval=0.1
            )
        )

        # Initialize
        metrics_registry, health_manager = await initialize_monitoring(config)

        assert metrics_registry is not None
        assert health_manager is not None

        # Start background tasks
        tasks = await start_background_monitoring_tasks(
            metrics_registry,
            health_manager
        )

        assert len(tasks) == 2

        # Let system run briefly
        await asyncio.sleep(0.2)

        # Stop background tasks
        await stop_background_monitoring_tasks(tasks)

        # Cleanup
        await cleanup_monitoring(metrics_registry, health_manager)

        # All tasks should be done
        assert all(task.done() for task in tasks)

    @pytest.mark.asyncio
    async def test_monitoring_with_custom_intervals(self):
        """Test monitoring with custom intervals."""
        config = MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(
                enabled=True,
                collect_system_metrics=True,
                system_metrics_interval=0.05  # Very short for testing
            ),
            health_checks=HealthCheckConfig(
                enabled=True,
                interval=0.05
            )
        )

        metrics_registry, health_manager = await initialize_monitoring(config)

        # Mock to count calls
        original_update = metrics_registry.update_system_metrics
        call_count = {"count": 0}

        def counting_update():
            call_count["count"] += 1
            original_update()

        metrics_registry.update_system_metrics = counting_update

        tasks = await start_background_monitoring_tasks(
            metrics_registry,
            health_manager
        )

        # Let run for longer than interval
        await asyncio.sleep(0.12)

        # Should have been called multiple times
        assert call_count["count"] >= 2

        await stop_background_monitoring_tasks(tasks)
        await cleanup_monitoring(metrics_registry, health_manager)

    @pytest.mark.asyncio
    async def test_partial_monitoring_configuration(self):
        """Test monitoring with only some components enabled."""
        # Only metrics enabled
        config1 = MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(enabled=True),
            health_checks=HealthCheckConfig(enabled=False)
        )

        registry1, manager1 = await initialize_monitoring(config1)
        tasks1 = await start_background_monitoring_tasks(registry1, manager1)

        # Should have metrics but no health checks
        assert registry1.config.enabled is True
        assert manager1.config.enabled is False
        # Exact task count depends on configuration

        await stop_background_monitoring_tasks(tasks1)
        await cleanup_monitoring(registry1, manager1)

        # Only health checks enabled
        config2 = MonitoringConfig(
            enabled=True,
            metrics=MetricsConfig(enabled=False),
            health_checks=HealthCheckConfig(enabled=True)
        )

        registry2, manager2 = await initialize_monitoring(config2)
        tasks2 = await start_background_monitoring_tasks(registry2, manager2)

        # Should have health checks but disabled metrics
        assert registry2.config.enabled is False
        assert manager2.config.enabled is True

        await stop_background_monitoring_tasks(tasks2)
        await cleanup_monitoring(registry2, manager2)
