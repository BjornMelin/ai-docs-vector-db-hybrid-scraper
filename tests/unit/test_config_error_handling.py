"""Tests for enhanced configuration error handling."""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from src.config.config_manager import (
    ConfigManager,
    create_and_load_config_async,
)
from src.config.error_handling import (
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ErrorContext,
    GracefulDegradationHandler,
    RetryableConfigOperation,
    SafeConfigLoader,
    async_error_context,
    handle_validation_error,
)


class TestConfig(BaseSettings):
    """Test configuration model."""

    api_key: str = Field(default="test-key")
    port: int = Field(default=8080, ge=1, le=65535)
    debug: bool = Field(default=False)


class TestConfigError:
    """Tests for custom error classes."""

    def test_config_error_with_context(self):
        """Test ConfigError with context information."""
        error = ConfigError(
            "Test error",
            context={"file": "test.json", "line": 42},
            cause=ValueError("underlying error"),
        )

        assert "Test error" in str(error)
        assert "Context:" in str(error)
        assert '"file": "test.json"' in str(error)
        assert "Caused by: ValueError: underlying error" in str(error)

    def test_config_validation_error(self):
        """Test ConfigValidationError with validation details."""
        validation_errors = [
            {"loc": ("field1",), "msg": "required field", "type": "missing"},
            {
                "loc": ("field2", "nested"),
                "msg": "invalid value",
                "type": "value_error",
            },
        ]

        error = ConfigValidationError(
            "Validation failed",
            validation_errors=validation_errors,
            context={"source": "test.yaml"},
        )

        error_str = str(error)
        assert "Validation failed" in error_str
        assert "field1: required field" in error_str
        assert "field2.nested: invalid value" in error_str
        assert '"source": "test.yaml"' in error_str


class TestErrorContext:
    """Tests for error context managers."""

    def test_error_context_success(self):
        """Test ErrorContext when no error occurs."""
        with ErrorContext("test_operation", key="value") as ctx:
            assert ctx.operation == "test_operation"
            assert ctx.context["key"] == "value"
            assert "start_time" in ctx.context

    def test_error_context_with_exception(self, caplog):
        """Test ErrorContext when exception occurs."""
        with pytest.raises(ValueError), ErrorContext("failing_operation", test_id=123):
            raise ValueError("test error")

        # Check that error was logged with context
        assert "Error in failing_operation" in caplog.text
        assert "test_id" in caplog.text

    @pytest.mark.asyncio
    async def test_async_error_context(self, caplog):
        """Test async error context manager."""
        with pytest.raises(RuntimeError):
            async with async_error_context("async_operation", async_id=456):
                raise RuntimeError("async error")

        assert "Error in async_operation" in caplog.text
        assert "async_id" in caplog.text


class TestValidationErrorHandling:
    """Tests for validation error handling."""

    def test_handle_validation_error(self):
        """Test converting Pydantic ValidationError to ConfigValidationError."""
        # Create a validation error
        try:
            TestConfig(port=100000)  # Invalid port
        except ValidationError as e:
            config_error = handle_validation_error(e, "test_source")

            assert isinstance(config_error, ConfigValidationError)
            assert len(config_error.validation_errors) > 0
            assert config_error.context["config_source"] == "test_source"

            # Check error details
            error_detail = config_error.validation_errors[0]
            assert "port" in str(error_detail["loc"])
            assert "less than or equal to 65535" in error_detail["msg"]

    def test_handle_validation_error_masks_sensitive_data(self):
        """Test that sensitive data is masked in validation errors."""
        try:
            # Create error with sensitive field
            TestConfig(api_key="", port=8080)
        except ValidationError:
            pass  # This won't actually fail, so simulate

        # Simulate validation error with sensitive data
        mock_error = Mock(spec=ValidationError)
        mock_error.errors.return_value = [
            {
                "loc": ("api_key",),
                "msg": "invalid api key",
                "type": "value_error",
                "input": "sk-secret-key-12345",
            }
        ]

        config_error = handle_validation_error(mock_error, "test")
        error_detail = config_error.validation_errors[0]
        assert error_detail["input"] == "***MASKED***"


class TestRetryableConfigOperation:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async function retry on transient error."""
        call_count = 0

        @RetryableConfigOperation(max_attempts=3)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Transient error")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_exhausted(self):
        """Test async function retry exhaustion."""

        @RetryableConfigOperation(max_attempts=2)
        async def always_fails():
            raise OSError("Persistent error")

        with pytest.raises(OSError, match="Persistent error"):
            await always_fails()

    def test_sync_retry_success(self):
        """Test sync function retry."""
        call_count = 0

        @RetryableConfigOperation(max_attempts=3)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Transient error")
            return "success"

        result = flaky_operation()
        assert result == "success"
        assert call_count == 2


class TestSafeConfigLoader:
    """Tests for SafeConfigLoader."""

    def test_load_json_file(self, tmp_path):
        """Test loading JSON configuration."""
        config_file = tmp_path / "config.json"
        config_data = {"api_key": "test-key", "port": 8080, "debug": True}
        config_file.write_text(json.dumps(config_data))

        loader = SafeConfigLoader(TestConfig)
        data = loader.load_from_file(config_file)

        assert data == config_data

    def test_load_yaml_file(self, tmp_path):
        """Test loading YAML configuration."""
        config_file = tmp_path / "config.yaml"
        config_data = {"api_key": "test-key", "port": 8080, "debug": True}
        config_file.write_text(yaml.dump(config_data))

        loader = SafeConfigLoader(TestConfig)
        data = loader.load_from_file(config_file)

        assert data == config_data

    def test_load_missing_file(self):
        """Test loading non-existent file."""
        loader = SafeConfigLoader(TestConfig)

        with pytest.raises(ConfigLoadError) as exc_info:
            loader.load_from_file(Path("/non/existent/file.json"))

        assert "not found" in str(exc_info.value)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json}")

        loader = SafeConfigLoader(TestConfig)

        with pytest.raises(ConfigLoadError) as exc_info:
            loader.load_from_file(config_file)

        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.context["file_path"] == str(config_file)

    def test_create_config_with_validation_error(self):
        """Test creating config with validation error."""
        loader = SafeConfigLoader(TestConfig)

        with pytest.raises(ConfigValidationError) as exc_info:
            loader.create_config({"port": "not-a-number"})

        assert "validation failed" in str(exc_info.value).lower()

    def test_create_config_with_fallback(self):
        """Test fallback config on error."""
        fallback = TestConfig(api_key="fallback-key")
        loader = SafeConfigLoader(TestConfig, fallback_config=fallback)

        # Force an unexpected error
        with patch.object(
            TestConfig, "__init__", side_effect=RuntimeError("unexpected")
        ):
            config = loader.create_config()

        assert config == fallback

    @pytest.mark.asyncio
    async def test_load_config_async(self, tmp_path):
        """Test async config loading."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "async-key", "port": 9000}')

        loader = SafeConfigLoader(TestConfig)
        config = await loader.load_config_async(config_file)

        assert config.api_key == "async-key"
        assert config.port == 9000


class TestGracefulDegradationHandler:
    """Tests for graceful degradation."""

    def test_degradation_activation(self):
        """Test degradation activates after multiple failures."""
        handler = GracefulDegradationHandler()

        # Record multiple failures
        for i in range(5):
            handler.record_failure(
                "test_operation",
                RuntimeError(f"Error {i}"),
                {"attempt": i},
            )

        assert handler.degradation_active
        assert len(handler.failed_operations) == 5

    def test_should_skip_operation(self):
        """Test operation skipping during degradation."""
        handler = GracefulDegradationHandler()
        handler.degradation_active = True

        # Non-critical operations should be skipped
        assert handler.should_skip_operation("file_watch")
        assert handler.should_skip_operation("drift_detection")

        # Critical operations should not be skipped
        assert not handler.should_skip_operation("reload_config")

    def test_reset_degradation(self):
        """Test resetting degradation state."""
        handler = GracefulDegradationHandler()
        handler.degradation_active = True
        handler.failed_operations = [{"test": "data"}]

        handler.reset()

        assert not handler.degradation_active
        assert len(handler.failed_operations) == 0


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_initialization(self, tmp_path):
        """Test manager initialization."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "init-key"}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,  # Disable for testing
        )

        config = manager.get_config()
        assert config.api_key == "init-key"

    def test_reload_config_with_error_recovery(self, tmp_path):
        """Test config reload with error recovery."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "original"}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Get original config
        original_config = manager.get_config()
        assert original_config.api_key == "original"

        # Write invalid config
        config_file.write_text('{"port": "invalid"}')

        # Reload should fail but keep original config
        result = manager.reload_config()
        assert not result  # Reload failed

        # Should still have original config
        current_config = manager.get_config()
        assert current_config.api_key == "original"

    def test_config_backup_and_restore(self, tmp_path):
        """Test configuration backup and restore."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "v1", "port": 8080}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Update config
        config_file.write_text('{"api_key": "v2", "port": 9090}')
        manager.reload_config()

        # Restore from backup
        result = manager.restore_from_backup()
        assert result

        # Should have v1 config
        config = manager.get_config()
        assert config.api_key == "v1"
        assert config.port == 8080

    def test_change_listener_error_isolation(self, tmp_path):
        """Test that failing change listeners don't affect others."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "test"}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Add multiple listeners
        successful_calls = []

        def good_listener(old, new):
            successful_calls.append((old.api_key, new.api_key))

        def bad_listener(_old, _new):
            raise RuntimeError("Listener error")

        manager.add_change_listener(good_listener)
        manager.add_change_listener(bad_listener)
        manager.add_change_listener(good_listener)

        # Update config
        config_file.write_text('{"api_key": "updated"}')
        manager.reload_config()

        # Good listeners should have been called
        assert len(successful_calls) == 2
        assert all(call == ("test", "updated") for call in successful_calls)

    @pytest.mark.asyncio
    async def test_async_reload(self, tmp_path):
        """Test async configuration reload."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "async-test"}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Update config
        config_file.write_text('{"api_key": "async-updated"}')

        # Reload asynchronously
        result = await manager.reload_config_async()
        assert result

        config = manager.get_config()
        assert config.api_key == "async-updated"

    def test_status_information(self, tmp_path):
        """Test getting comprehensive status."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "status-test"}')

        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        status = manager.get_status()

        assert "config_file" in status
        assert "graceful_degradation_active" in status
        assert "recent_failures" in status
        assert "config_backups_count" in status
        assert status["config_exists"] is True

    @pytest.mark.asyncio
    async def test_create_and_load_async(self, tmp_path):
        """Test async creation and loading helper."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "helper-test", "port": 7777}')

        manager, config = await create_and_load_config_async(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=False,
        )

        assert isinstance(manager, ConfigManager)
        assert config.api_key == "helper-test"
        assert config.port == 7777


class TestFileWatchingErrorHandling:
    """Tests for file watching error handling."""

    @patch("watchdog.observers.Observer")
    def test_file_watch_start_failure(self, mock_observer_class, tmp_path, caplog):
        """Test handling file watch start failure."""
        # Make observer creation fail
        mock_observer_class.side_effect = RuntimeError("Observer error")

        config_file = tmp_path / "config.json"
        config_file.write_text('{"api_key": "test"}')

        # Should not raise, just log error
        manager = ConfigManager(
            config_class=TestConfig,
            config_file=config_file,
            enable_file_watching=True,
        )

        assert "File watching setup failed" in caplog.text
        assert manager._observer is None

    def test_file_watch_error_callback(self, tmp_path):
        """Test file watch error callback."""
        from src.config.config_manager import (
            ConfigFileWatcher as EnhancedConfigFileWatcher,
        )

        config_file = tmp_path / "config.json"
        error_calls = []

        def error_callback(error):
            error_calls.append(error)

        reload_calls = []

        def reload_callback():
            reload_calls.append(1)
            if len(reload_calls) > 2:
                raise RuntimeError("Reload error")

        watcher = EnhancedConfigFileWatcher(
            config_file, reload_callback, error_callback
        )

        # Simulate file change events
        event = Mock()
        event.is_directory = False
        event.src_path = str(config_file)
        event.event_type = "modified"

        # First few calls succeed
        watcher.on_modified(event)
        time.sleep(0.2)  # Wait for debounce
        watcher.on_modified(event)
        time.sleep(0.2)  # Wait for debounce

        # Next calls fail
        for _ in range(3):
            watcher.on_modified(event)
            time.sleep(0.2)  # Wait for debounce

        assert len(error_calls) == 3
        assert all(isinstance(e, RuntimeError) for e in error_calls)
