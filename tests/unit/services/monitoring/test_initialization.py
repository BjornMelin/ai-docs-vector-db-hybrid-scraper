"""Tests for the simplified monitoring initialization module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.models import MonitoringConfig
from src.services.monitoring.initialization import (
    cleanup_monitoring,
    initialize_monitoring,
    initialize_monitoring_system,
    run_periodic_health_checks,
    setup_fastmcp_monitoring,
    start_background_monitoring_tasks,
    stop_background_monitoring_tasks,
)


class DummyMCP:
    """Small helper exposing a FastAPI app attribute to mimic FastMCP."""

    def __init__(self) -> None:
        self.app = FastAPI()


class TestInitializeMonitoring:
    """Unit tests for initialize_monitoring and initialize_monitoring_system."""

    @pytest.mark.asyncio
    async def test_initialize_monitoring_disabled(self) -> None:
        config = MonitoringConfig(enabled=False)
        manager = await initialize_monitoring(config)
        assert manager is None

    @pytest.mark.asyncio
    async def test_initialize_monitoring_enabled(self) -> None:
        config = MonitoringConfig(
            enabled=True,
            system_metrics_interval=12.0,
            health_check_timeout=4.0,
        )
        manager = await initialize_monitoring(config)
        assert manager is not None
        assert manager.config.interval == pytest.approx(12.0)
        assert manager.config.timeout == pytest.approx(4.0)

    def test_initialize_monitoring_system_disabled(self) -> None:
        settings = MagicMock()
        settings.monitoring.enabled = False
        manager = initialize_monitoring_system(settings)
        assert manager is None

    def test_initialize_monitoring_system_enabled(self) -> None:
        settings = MagicMock()
        settings.monitoring.enabled = True
        settings.monitoring.system_metrics_interval = 15.0
        settings.monitoring.health_check_timeout = 3.0
        settings.monitoring.health_path = "/health"
        settings.monitoring.include_system_metrics = True
        settings.monitoring.cpu_threshold = 90.0
        settings.monitoring.memory_threshold = 90.0
        settings.monitoring.disk_threshold = 90.0
        settings.cache.enable_dragonfly_cache = False
        settings.cache.enable_redis_cache = False

        with patch(
            "src.services.monitoring.initialization.build_health_manager"
        ) as mock_builder:
            mock_manager = MagicMock()
            mock_builder.return_value = mock_manager

            manager = initialize_monitoring_system(settings)

            assert manager is mock_manager
            mock_builder.assert_called_once()


class TestMonitoringTasks:
    """Tests covering background monitoring task management."""

    @pytest.mark.asyncio
    async def test_start_background_monitoring_tasks_enabled(self) -> None:
        manager = MagicMock()
        manager.config.enabled = True
        manager.config.interval = 0.01

        tasks = await start_background_monitoring_tasks(manager)
        assert len(tasks) == 1

        await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_start_background_monitoring_tasks_disabled(self) -> None:
        manager = MagicMock()
        manager.config.enabled = False

        tasks = await start_background_monitoring_tasks(manager)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_run_periodic_health_checks_runs_loop_once(self) -> None:
        manager = MagicMock()
        manager.check_all = AsyncMock()

        async def runner():
            task = asyncio.create_task(
                run_periodic_health_checks(manager, interval_seconds=0.01)
            )
            await asyncio.sleep(0.025)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        await runner()
        assert manager.check_all.await_count >= 1

    @pytest.mark.asyncio
    async def test_cleanup_monitoring(self) -> None:
        manager = MagicMock()
        await cleanup_monitoring(manager)
        await cleanup_monitoring(None)


class TestSetupFastMCPMonitoring:
    """Ensure FastMCP health endpoints are registered correctly."""

    def test_setup_fastmcp_monitoring_adds_routes(self) -> None:
        settings = MagicMock()
        settings.monitoring.enabled = True
        settings.monitoring.health_path = "/health"

        manager = MagicMock()
        manager.config.enabled = True
        manager.check_all = AsyncMock()
        manager.get_overall_status.return_value = MagicMock(value="healthy")
        manager.get_health_summary.return_value = {"status": "healthy"}

        mcp = DummyMCP()
        setup_fastmcp_monitoring(mcp, settings, manager)

        client = TestClient(mcp.app)
        response = client.get("/health")
        assert response.status_code in (200, 503)

        live_response = client.get("/health/live")
        assert live_response.status_code == 200

        ready_response = client.get("/health/ready")
        assert ready_response.status_code in (200, 503)

    def test_setup_fastmcp_monitoring_no_manager(self) -> None:
        settings = MagicMock()
        settings.monitoring.enabled = True
        settings.monitoring.health_path = "/health"
        mcp = DummyMCP()
        setup_fastmcp_monitoring(mcp, settings, None)
        client = TestClient(mcp.app)
        assert client.get("/health").status_code == 404
