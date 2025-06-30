"""Tests for zero-downtime configuration reloading.

This module provides comprehensive tests for the configuration reloading
system including validation, rollback, observability, and API integration.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import Config
from src.config.reload import (
    ConfigReloader,
    ReloadOperation,
    ReloadStatus,
    ReloadTrigger,
)
from src.services.observability.config_instrumentation import (
    ConfigOperationType,
    instrument_config_operation,
)


class TestConfigReloader:
    """Test configuration reloader functionality."""

    @pytest.fixture
    def sample_config(self) -> Config:
        """Sample configuration for testing."""
        return Config(
            app_name="Test App",
            version="1.0.0",
            debug=True,
        )

    @pytest.fixture
    def reloader(self, sample_config: Config) -> ConfigReloader:
        """Configuration reloader instance for testing."""
        reloader = ConfigReloader(enable_signal_handler=False)
        reloader.set_current_config(sample_config)
        return reloader

    @pytest.mark.asyncio
    async def test_manual_reload_success(self, reloader: ConfigReloader):
        """Test successful manual configuration reload."""
        # Track callback invocations
        callback_calls = []

        def mock_callback(old_config: Config, new_config: Config) -> bool:
            callback_calls.append((old_config, new_config))
            return True

        reloader.add_change_listener("test_service", mock_callback)

        # Perform reload
        operation = await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            force=True,
        )

        assert operation.success
        assert operation.status == ReloadStatus.SUCCESS
        assert operation.trigger == ReloadTrigger.MANUAL
        assert operation.total_duration_ms > 0
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_reload_with_validation_errors(self, reloader: ConfigReloader):
        """Test reload with configuration validation errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write invalid configuration
            json.dump({"invalid": "config"}, f)
            config_file = Path(f.name)

        try:
            operation = await reloader.reload_config(
                config_source=config_file,
                trigger=ReloadTrigger.MANUAL,
            )

            assert not operation.success
            assert operation.status == ReloadStatus.FAILED
            assert len(operation.validation_errors) > 0

        finally:
            config_file.unlink()

    @pytest.mark.asyncio
    async def test_reload_no_changes(self, reloader: ConfigReloader):
        """Test reload when no configuration changes detected."""
        operation = await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            force=False,
        )

        assert operation.success
        assert "No configuration changes detected" in operation.error_message

    @pytest.mark.asyncio
    async def test_callback_failure_handling(self, reloader: ConfigReloader):
        """Test handling of callback failures during reload."""

        def failing_callback(old_config: Config, new_config: Config) -> bool:
            return False

        def success_callback(old_config: Config, new_config: Config) -> bool:
            return True

        reloader.add_change_listener("failing_service", failing_callback)
        reloader.add_change_listener("success_service", success_callback)

        operation = await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            force=True,
        )

        # Should still succeed if majority of callbacks succeed
        assert operation.success
        assert "failing_service:failed" in operation.services_notified
        assert "success_service:success" in operation.services_notified

    @pytest.mark.asyncio
    async def test_callback_timeout_handling(self, reloader: ConfigReloader):
        """Test handling of callback timeouts."""

        async def slow_callback(old_config: Config, new_config: Config) -> bool:
            await asyncio.sleep(2.0)  # Longer than timeout
            return True

        reloader.add_change_listener(
            "slow_service",
            slow_callback,
            async_callback=True,
            timeout_seconds=0.5,
        )

        operation = await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            force=True,
        )

        assert "slow_service:timeout" in operation.services_notified

    @pytest.mark.asyncio
    async def test_configuration_rollback(self, reloader: ConfigReloader):
        """Test configuration rollback functionality."""
        # Create multiple configuration versions
        configs = [Config(app_name=f"App v{i}", version=f"1.0.{i}") for i in range(3)]

        for config in configs:
            reloader.set_current_config(config)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        # Perform rollback
        operation = await reloader.rollback_config()

        assert operation.success
        assert operation.status == ReloadStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_file_watching(self, reloader: ConfigReloader):
        """Test automatic file watching for configuration changes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AI_DOCS_APP_NAME=Original App\n")
            config_file = Path(f.name)

        try:
            # Create file-watching reloader
            file_reloader = ConfigReloader(
                config_source=config_file,
                enable_signal_handler=False,
            )
            file_reloader.set_current_config(Config(app_name="Original App"))

            # Enable file watching with short interval
            await file_reloader.enable_file_watching(poll_interval=0.1)

            # Modify file
            await asyncio.sleep(0.2)
            config_file.write_text("AI_DOCS_APP_NAME=Updated App\n")

            # Wait for change detection
            await asyncio.sleep(0.5)

            # Check if reload was triggered
            history = file_reloader.get_reload_history()
            file_watch_operations = [
                op for op in history if op.trigger == ReloadTrigger.FILE_WATCH
            ]

            assert len(file_watch_operations) > 0

            await file_reloader.disable_file_watching()
            await file_reloader.shutdown()

        finally:
            config_file.unlink()

    def test_listener_management(self, reloader: ConfigReloader):
        """Test configuration change listener management."""

        def test_callback(old_config: Config, new_config: Config) -> bool:
            return True

        # Add listener
        reloader.add_change_listener("test_listener", test_callback, priority=100)
        assert len(reloader._change_listeners) == 1  # noqa: SLF001
        assert reloader._change_listeners[0].name == "test_listener"  # noqa: SLF001
        assert reloader._change_listeners[0].priority == 100  # noqa: SLF001

        # Remove listener
        success = reloader.remove_change_listener("test_listener")
        assert success
        assert len(reloader._change_listeners) == 0  # noqa: SLF001

        # Try to remove non-existent listener
        success = reloader.remove_change_listener("non_existent")
        assert not success

    def test_reload_statistics(self, reloader: ConfigReloader):
        """Test reload statistics collection."""
        # Initially no operations
        stats = reloader.get_reload_stats()
        assert stats["total_operations"] == 0

        # Add some mock operations

        successful_op = ReloadOperation(trigger=ReloadTrigger.MANUAL)
        successful_op.complete(True)

        failed_op = ReloadOperation(trigger=ReloadTrigger.API)
        failed_op.complete(False, "Test error")

        reloader._reload_history.extend([successful_op, failed_op])  # noqa: SLF001

        # Check updated stats
        stats = reloader.get_reload_stats()
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_concurrent_reload_prevention(self, reloader: ConfigReloader):
        """Test prevention of concurrent reload operations."""
        # Start first reload (it will block on the lock)
        with patch("src.config.reload.time.sleep", return_value=None):
            task1 = asyncio.create_task(
                reloader.reload_config(trigger=ReloadTrigger.MANUAL, force=True)
            )

            # Give first reload time to acquire lock
            await asyncio.sleep(0.01)

            # Try second reload (should fail immediately)
            task2 = asyncio.create_task(
                reloader.reload_config(trigger=ReloadTrigger.API, force=True)
            )

            # Wait for both to complete
            op1, op2 = await asyncio.gather(task1, task2)

            # First should succeed, second should fail due to concurrent access
            assert op1.success or op2.success  # At least one succeeds
            assert not (op1.success and op2.success)  # Not both succeed

            # One should have concurrent access error
            concurrent_error = "Another reload operation is in progress"
            assert (concurrent_error in (op1.error_message or "")) or (
                concurrent_error in (op2.error_message or "")
            )


class TestConfigurationAPI:
    """Test configuration management API endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Test client for API testing."""
        return TestClient(app)

    @pytest.fixture
    def mock_reloader(self):
        """Mock configuration reloader for API testing."""
        with patch("src.config.reload.get_config_reloader") as mock:
            reloader = Mock(spec=ConfigReloader)
            mock.return_value = reloader
            yield reloader

    def test_reload_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test configuration reload API endpoint."""

        # Mock successful reload operation
        mock_operation = ReloadOperation(trigger=ReloadTrigger.API)
        mock_operation.status = ReloadStatus.SUCCESS
        mock_operation.complete(True)
        mock_reloader.reload_config.return_value = mock_operation

        response = client.post("/api/v1/config/reload", json={"force": True})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "success"
        assert "operation_id" in data

    def test_reload_endpoint_failure(self, client: TestClient, mock_reloader: Mock):
        """Test configuration reload API endpoint with failure."""

        # Mock failed reload operation
        mock_operation = ReloadOperation(trigger=ReloadTrigger.API)
        mock_operation.status = ReloadStatus.FAILED
        mock_operation.complete(False, "Validation failed")
        mock_reloader.reload_config.return_value = mock_operation

        response = client.post("/api/v1/config/reload", json={"force": False})

        assert response.status_code == 200  # API returns 200 with error details
        data = response.json()
        assert data["success"] is False
        assert "Validation failed" in data["message"]

    def test_rollback_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test configuration rollback API endpoint."""

        # Mock successful rollback operation
        mock_operation = ReloadOperation()
        mock_operation.status = ReloadStatus.ROLLED_BACK
        mock_operation.complete(True)
        mock_reloader.rollback_config.return_value = mock_operation

        response = client.post("/api/v1/config/rollback", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_history_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test configuration reload history API endpoint."""

        # Mock history data
        mock_operations = [
            ReloadOperation(trigger=ReloadTrigger.MANUAL),
            ReloadOperation(trigger=ReloadTrigger.API),
        ]
        for op in mock_operations:
            op.complete(True)

        mock_reloader.get_reload_history.return_value = mock_operations

        response = client.get("/api/v1/config/history?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["operations"]) == 2

    def test_stats_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test configuration reload statistics API endpoint."""
        # Mock statistics data
        mock_stats = {
            "total_operations": 10,
            "successful_operations": 8,
            "failed_operations": 2,
            "success_rate": 0.8,
            "average_duration_ms": 150.5,
            "listeners_registered": 5,
            "backups_available": 3,
            "current_config_hash": "abc123",
        }
        mock_reloader.get_reload_stats.return_value = mock_stats

        response = client.get("/api/v1/config/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 10
        assert data["success_rate"] == 0.8

    def test_status_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test configuration status API endpoint."""
        # Mock configuration status
        mock_stats = {
            "current_config_hash": "def456",
            "listeners_registered": 3,
            "backups_available": 2,
        }
        mock_reloader.get_reload_stats.return_value = mock_stats
        mock_reloader.enable_signal_handler = True
        mock_reloader._file_watch_enabled = False  # noqa: SLF001

        response = client.get("/api/v1/config/status")

        assert response.status_code == 200
        data = response.json()
        assert data["config_reloader_enabled"] is True
        assert data["current_config_hash"] == "def456"
        assert data["signal_handler_enabled"] is True

    def test_file_watch_enable_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test file watching enable API endpoint."""
        mock_reloader.enable_file_watching.return_value = None
        mock_reloader.config_source = Path("/test/config.env")

        response = client.post("/api/v1/config/file-watch/enable?poll_interval=2.0")

        assert response.status_code == 200
        data = response.json()
        assert data["file_watching_enabled"] is True
        assert data["poll_interval_seconds"] == 2.0

    def test_file_watch_disable_endpoint(self, client: TestClient, mock_reloader: Mock):
        """Test file watching disable API endpoint."""
        mock_reloader.disable_file_watching.return_value = None

        response = client.post("/api/v1/config/file-watch/disable")

        assert response.status_code == 200
        data = response.json()
        assert data["file_watching_enabled"] is False


class TestConfigurationIntegration:
    """Integration tests for configuration reloading system."""

    @pytest.mark.asyncio
    async def test_full_reload_cycle(self):
        """Test complete configuration reload cycle with observability."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write initial configuration
            initial_config = {
                "app_name": "Integration Test App",
                "version": "1.0.0",
                "debug": True,
            }
            json.dump(initial_config, f)
            config_file = Path(f.name)

        try:
            # Create reloader
            reloader = ConfigReloader(
                config_source=config_file,
                enable_signal_handler=False,
            )

            # Set initial config
            config = Config(**initial_config)
            reloader.set_current_config(config)

            # Add service callbacks
            service_updates = []

            def service_callback(old_config: Config, new_config: Config) -> bool:
                service_updates.append((old_config.version, new_config.version))
                return True

            reloader.add_change_listener("integration_service", service_callback)

            # Update configuration file
            updated_config = {
                "app_name": "Updated Integration Test App",
                "version": "1.0.1",
                "debug": False,
            }

            config_file.write_text(json.dumps(updated_config))

            # Perform reload
            operation = await reloader.reload_config(
                trigger=ReloadTrigger.MANUAL,
                config_source=config_file,
            )

            # Verify operation success
            assert operation.success
            assert operation.trigger == ReloadTrigger.MANUAL
            assert operation.validation_duration_ms > 0
            assert operation.apply_duration_ms > 0
            assert operation.total_duration_ms > 0

            # Verify service was updated
            assert len(service_updates) == 1
            assert service_updates[0] == ("1.0.0", "1.0.1")

            # Verify statistics
            stats = reloader.get_reload_stats()
            assert stats["total_operations"] == 1
            assert stats["successful_operations"] == 1
            assert stats["success_rate"] == 1.0

            await reloader.shutdown()

        finally:
            config_file.unlink()

    @pytest.mark.asyncio
    async def test_observability_integration(self):
        """Test configuration reloading with observability tracking."""
        # This test would require actual observability setup
        # For now, we'll test that the instrumentation decorators work

        call_count = 0

        @instrument_config_operation(
            operation_type=ConfigOperationType.UPDATE,
            operation_name="test_config_operation",
        )
        async def test_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_operation()

        assert result == "success"
        assert call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
