"""Integration tests for configuration error handling scenarios."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.config import (
    Config,
    ConfigManager,
    OpenAIConfig,
    QdrantConfig,
    get_degradation_handler,
)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


async def _create_manager_async(**kwargs: Any) -> ConfigManager:
    """Instantiate ``ConfigManager`` off the main event loop for async tests."""

    return await asyncio.to_thread(ConfigManager, **kwargs)


@pytest.fixture
def reset_degradation():
    """Reset degradation handler after each test."""
    yield
    get_degradation_handler().reset()


@pytest.fixture(autouse=True)
def _reset_degradation(reset_degradation):
    """Automatically reset graceful degradation between tests."""

    yield


class TestRealWorldErrorScenarios:
    """Test real-world error scenarios."""

    def test_corrupted_config_file_recovery(self, config_dir):
        """Test recovery from corrupted configuration file."""
        config_file = config_dir / "config.json"

        # Start with valid config
        valid_config = {
            "openai": {"api_key": "sk-test123"},
            "qdrant": {"url": "http://localhost:6333"},
        }
        config_file.write_text(json.dumps(valid_config))

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Verify initial config loads
        config = manager.get_config()
        assert config.openai.api_key == "sk-test123"

        # Corrupt the file
        config_file.write_text("{invalid json content")

        # Reload should fail but keep existing config
        result = manager.reload_config()
        assert not result

        # Should still have valid config
        current_config = manager.get_config()
        assert current_config.openai.api_key == "sk-test123"

        # Fix the config file
        fixed_config = {
            "openai": {"api_key": "sk-updated123"},
            "qdrant": {"url": "http://localhost:6333"},
        }
        config_file.write_text(json.dumps(fixed_config))

        # Now reload should succeed
        result = manager.reload_config()
        assert result

        # Should have new config
        new_config = manager.get_config()
        assert new_config.openai.api_key == "sk-updated123"

    def test_missing_required_fields_fallback(self, config_dir):
        """Test handling missing required fields with fallback."""
        config_file = config_dir / "config.yaml"

        # Create config missing required fields
        incomplete_config = {
            "qdrant": {"url": "http://localhost:6333"}
            # Missing openai config
        }
        config_file.write_text(yaml.dump(incomplete_config))

        # Create fallback config
        fallback = Config(
            openai=OpenAIConfig(api_key="sk-fallback"),
            qdrant=QdrantConfig(url="http://fallback:6333"),
        )

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
            fallback_config=fallback,
        )

        # Should use merged config (file values override fallback)
        config = manager.get_config()
        assert config.qdrant.url == "http://localhost:6333"
        assert config.openai.api_key == "sk-fallback"

    def test_file_permission_error_handling(self, config_dir):
        """Test handling file permission errors."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Load initial config
        config = manager.get_config()
        assert config.openai.api_key == "sk-test"

        # Simulate permission error by intercepting file reads
        original_read_text = Path.read_text

        def _raise_permission(self: Path, *args: Any, **kwargs: Any) -> str:
            if self == config_file:
                raise PermissionError("Access denied")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", side_effect=_raise_permission):
            result = manager.reload_config()
            assert not result

        # Should still have original config
        assert manager.get_config().openai.api_key == "sk-test"

    @pytest.mark.asyncio
    async def test_concurrent_reload_handling(self, config_dir):
        """Test handling concurrent reload attempts."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-original"}}')

        manager = await _create_manager_async(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Create multiple concurrent reload tasks
        reload_count = 10

        async def update_and_reload(index):
            # Update config file
            config_data = {"openai": {"api_key": f"sk-update{index}"}}
            config_file.write_text(json.dumps(config_data))

            # Small delay to ensure file is written
            await asyncio.sleep(0.01)

            # Reload config
            return await manager.reload_config_async()

        # Run concurrent reloads
        tasks = [update_and_reload(i) for i in range(reload_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some reloads may fail due to concurrent access, but shouldn't crash
        successful_reloads = sum(1 for r in results if r is True)
        assert successful_reloads > 0

        # Final config should be valid
        final_config = manager.get_config()
        final_key = final_config.openai.api_key
        assert final_key is not None
        assert final_key.startswith("sk-update")

    def test_graceful_degradation_activation(self, config_dir):
        """Test graceful degradation activating after repeated failures."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        _manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=True,  # Enable to test degradation
            enable_graceful_degradation=True,
        )

        # Simulate multiple file watch failures
        degradation = get_degradation_handler()

        for i in range(6):
            degradation.record_failure(
                "file_watch", RuntimeError(f"Watch error {i}"), {"attempt": i}
            )

        assert degradation.degradation_active

        # File watching should be skipped
        assert degradation.should_skip_operation("file_watch")

        # Critical operations should still work
        assert not degradation.should_skip_operation("reload_config")

    def test_validation_error_details(self, config_dir):
        """Test detailed validation error reporting."""
        config_file = config_dir / "config.json"

        # Create config with multiple validation errors
        invalid_config = {
            "openai": {
                "api_key": "invalid-key",  # Should start with sk-
                "max_retries": -1,  # Should be >= 0
            },
            "qdrant": {
                "url": "not-a-url",  # Invalid URL format
                "timeout": "not-a-number",  # Should be int
            },
        }
        config_file.write_text(json.dumps(invalid_config))

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Loading should handle validation errors gracefully
        # Due to our fallback mechanism, this won't raise
        _config = manager.get_config()

        # Check status for validation issues
        status = manager.get_status()
        assert "recent_failures" in status

    def test_backup_rotation(self, config_dir):
        """Test configuration backup rotation."""
        config_file = config_dir / "config.json"

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Create multiple config versions
        for i in range(15):
            config_data = {"openai": {"api_key": f"sk-version{i}"}}
            config_file.write_text(json.dumps(config_data))
            manager.reload_config()
            time.sleep(0.01)  # Ensure different timestamps

        # Should only keep last 10 backups
        assert len(manager._config_backups) == 10

        # Restore oldest available backup
        result = manager.restore_from_backup(0)
        assert result

        # Oldest retained backup becomes version 4 because the initial config
        # snapshot is preserved alongside the latest nine reloads.
        config = manager.get_config()
        assert config.openai.api_key == "sk-version4"

    def test_change_listener_failure_isolation(self, config_dir):
        """Test that listener failures don't affect config reload."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-initial"}}')

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        listener_calls = []

        def tracking_listener(old, new):
            listener_calls.append(
                (
                    old.openai.api_key or None,
                    new.openai.api_key or None,
                )
            )

        def failing_listener(_old, _new):
            msg = "Listener explosion!"
            raise RuntimeError(msg)

        # Add mixed listeners
        manager.add_change_listener(tracking_listener)
        manager.add_change_listener(failing_listener)
        manager.add_change_listener(tracking_listener)

        # Update config
        config_file.write_text('{"openai": {"api_key": "sk-updated"}}')
        result = manager.reload_config()

        # Reload should succeed despite listener failure
        assert result

        # Good listeners should have been called
        assert len(listener_calls) == 2
        assert all(call == ("sk-initial", "sk-updated") for call in listener_calls)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, config_dir):
        """Test handling of slow/hanging operations."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        manager = await _create_manager_async(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Simulate slow file read
        original_load = manager.safe_loader.load_from_file

        async def slow_load(path):
            await asyncio.sleep(5)  # Simulate slow operation
            return original_load(path)

        with (
            patch.object(manager.safe_loader, "load_from_file", side_effect=slow_load),
            pytest.raises(asyncio.TimeoutError),
        ):
            await asyncio.wait_for(manager.reload_config_async(), timeout=0.5)

        # Manager should still be functional
        config = manager.get_config()
        assert config.openai.api_key == "sk-test"


class TestEnvironmentSpecificErrors:
    """Test environment-specific error scenarios."""


class TestAsyncErrorPropagation:
    """Test async error propagation and handling."""

    @pytest.mark.asyncio
    async def test_async_validation_error_propagation(self, config_dir):
        """Test that validation errors propagate correctly in async context."""
        config_file = config_dir / "config.json"

        # Invalid config that will cause validation error
        config_file.write_text('{"qdrant": {"timeout": "not-a-number"}}')

        manager = await _create_manager_async(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Should handle validation error gracefully
        config = manager.get_config()
        assert config is not None  # Falls back to default

    @pytest.mark.asyncio
    async def test_async_error_context_preservation(self, config_dir):
        """Test that error context is preserved across async boundaries."""

        config_file = config_dir / "missing.json"

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Error context should be captured in the manager status
        result = await manager.reload_config_async()
        assert result is False

        failure_entries = manager.get_status()["recent_failures"]
        assert failure_entries, "Expected failure metadata to be recorded"
        latest_failure = failure_entries[0]
        assert latest_failure["operation"] == "reload_config"
        assert str(config_file) in latest_failure["error"]
