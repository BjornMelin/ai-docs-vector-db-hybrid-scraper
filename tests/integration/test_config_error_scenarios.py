"""Integration tests for configuration error handling scenarios."""

import asyncio
import json
import signal
import time
from unittest.mock import patch

import pytest
import yaml

from src.config.config_manager import ConfigManager
from src.config.core import Config
from src.config.error_handling import ConfigLoadError, get_degradation_handler


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def reset_degradation():
    """Reset degradation handler after each test."""
    yield
    get_degradation_handler().reset()


class TestRealWorldErrorScenarios:
    """Test real-world error scenarios."""

    def test_corrupted_config_file_recovery(self, config_dir, _reset_degradation):
        """Test recovery from corrupted configuration file."""
        config_file = config_dir / "config.json"

        # Start with valid config
        valid_config = {
            "openai": {"api_key": "sk-test123"},
            "vector_db": {"url": "http://localhost:6333"},
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
            "vector_db": {"url": "http://localhost:6333"},
        }
        config_file.write_text(json.dumps(fixed_config))

        # Now reload should succeed
        result = manager.reload_config()
        assert result

        # Should have new config
        new_config = manager.get_config()
        assert new_config.openai.api_key == "sk-updated123"

    def test_missing_required_fields_fallback(self, config_dir, _reset_degradation):
        """Test handling missing required fields with fallback."""
        config_file = config_dir / "config.yaml"

        # Create config missing required fields
        incomplete_config = {
            "vector_db": {"url": "http://localhost:6333"}
            # Missing openai config
        }
        config_file.write_text(yaml.dump(incomplete_config))

        # Create fallback config
        fallback = Config(
            openai=Config.OpenAIConfig(api_key="sk-fallback"),
            vector_db=Config.VectorDBConfig(url="http://fallback:6333"),
        )

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
            fallback_config=fallback,
        )

        # Should use merged config (file values override fallback)
        config = manager.get_config()
        assert config.vector_db.url == "http://localhost:6333"  # From file
        assert config.openai.api_key is None  # Default, not fallback

    def test_file_permission_error_handling(self, config_dir, _reset_degradation):
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

        # Simulate permission error
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = manager.reload_config()
            assert not result

        # Should still have original config
        assert manager.get_config().openai.api_key == "sk-test"

    @pytest.mark.asyncio
    async def test_concurrent_reload_handling(self, config_dir, _reset_degradation):
        """Test handling concurrent reload attempts."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-original"}}')

        manager = ConfigManager(
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
        assert final_config.openai.api_key.startswith("sk-update")

    def test_graceful_degradation_activation(self, config_dir, _reset_degradation):
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

    def test_validation_error_details(self, config_dir, _reset_degradation):
        """Test detailed validation error reporting."""
        config_file = config_dir / "config.json"

        # Create config with multiple validation errors
        invalid_config = {
            "openai": {
                "api_key": "invalid-key",  # Should start with sk-
                "max_retries": -1,  # Should be >= 0
            },
            "vector_db": {
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

    def test_backup_rotation(self, config_dir, _reset_degradation):
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

        # Should be version 5 (15 - 10)
        config = manager.get_config()
        assert config.openai.api_key == "sk-version5"

    def test_change_listener_failure_isolation(self, config_dir, _reset_degradation):
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
                    old.openai.api_key if old.openai.api_key else None,
                    new.openai.api_key if new.openai.api_key else None,
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
    async def test_timeout_handling(self, config_dir, _reset_degradation):
        """Test handling of slow/hanging operations."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        manager = ConfigManager(
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

    def test_signal_handler_error_recovery(self, config_dir, _reset_degradation):
        """Test signal handler error recovery."""
        if not hasattr(signal, "SIGHUP"):
            pytest.skip("SIGHUP not available on this platform")

        from src.config.reload import ConfigReloader

        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=True,
        )

        # Simulate signal with error
        with patch.object(
            reloader, "reload_config", side_effect=RuntimeError("Reload error")
        ):
            # Should not crash when receiving signal
            signal.raise_signal(signal.SIGHUP)
            time.sleep(0.1)  # Give signal handler time to run

        # Reloader should still be functional
        assert reloader._config is not None

    def test_file_watcher_recovery(self, config_dir, _reset_degradation, _caplog):
        """Test file watcher recovery after errors."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"openai": {"api_key": "sk-test"}}')

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=True,
        )

        # Wait for watcher to start
        time.sleep(0.1)

        # Simulate multiple file change events with failures
        if manager._observer and hasattr(manager._observer, "_emitter_for_watch"):
            # Force reload failures
            with patch.object(
                manager, "reload_config", side_effect=RuntimeError("Reload failed")
            ):
                # Trigger file changes
                for i in range(3):
                    config_file.write_text(
                        f'{{"openai": {{"api_key": "sk-change{i}"}}}}'
                    )
                    time.sleep(0.1)

            # Check that errors were logged
            assert "Failed to reload configuration on file change" in _caplog.text

        # Clean up
        manager.stop_file_watching()


class TestAsyncErrorPropagation:
    """Test async error propagation and handling."""

    @pytest.mark.asyncio
    async def test_async_validation_error_propagation(self, config_dir):
        """Test that validation errors propagate correctly in async context."""
        config_file = config_dir / "config.json"

        # Invalid config that will cause validation error
        config_file.write_text('{"vector_db": {"timeout": "not-a-number"}}')

        manager = ConfigManager(
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

        # Error context should be preserved
        with pytest.raises(ConfigLoadError) as exc_info:
            await manager.reload_config_async()

        error = exc_info.value
        assert error.context["config_file"] == str(config_file)
        assert "not found" in str(error)
