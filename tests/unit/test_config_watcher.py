"""Unit tests for configuration watcher functionality."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.config_watcher import ConfigFileHandler, ConfigWatcher


class TestConfigFileHandler:
    """Test ConfigFileHandler functionality."""

    def test_init(self):
        """Test handler initialization."""
        handler = ConfigFileHandler(debounce_seconds=2.0)
        assert handler.debounce_seconds == 2.0
        assert handler._timer is None
        assert handler._last_reload == 0.0

    def test_on_modified_directory(self):
        """Test that directory events are ignored."""
        handler = ConfigFileHandler()
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/test/dir"

        # Should not schedule reload for directories
        with patch.object(handler, "_schedule_reload") as mock_schedule:
            handler.on_modified(event)
            mock_schedule.assert_not_called()

    def test_on_modified_env_file(self):
        """Test that .env file changes trigger reload."""
        handler = ConfigFileHandler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/.env"

        # Should schedule reload for .env files
        with patch.object(handler, "_schedule_reload") as mock_schedule:
            handler.on_modified(event)
            mock_schedule.assert_called_once()

    def test_on_modified_non_env_file(self):
        """Test that non-.env files are ignored."""
        handler = ConfigFileHandler()
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/config.yaml"

        # Should not schedule reload for non-.env files
        with patch.object(handler, "_schedule_reload") as mock_schedule:
            handler.on_modified(event)
            mock_schedule.assert_not_called()

    @patch("src.services.config_watcher.reload_settings")
    def test_reload_config(self, mock_reload):
        """Test configuration reload with debouncing."""
        handler = ConfigFileHandler(debounce_seconds=0.1)

        # First reload should work
        handler._reload_config()
        mock_reload.assert_called_once()

        # Immediate second reload should be skipped due to debouncing
        mock_reload.reset_mock()
        handler._reload_config()
        mock_reload.assert_not_called()

        # After debounce period, reload should work again
        time.sleep(0.15)
        handler._reload_config()
        mock_reload.assert_called_once()

    @patch("src.services.config_watcher.reload_settings")
    def test_reload_config_error_handling(self, mock_reload):
        """Test error handling during reload."""
        handler = ConfigFileHandler()
        mock_reload.side_effect = Exception("Test error")

        # Should not raise exception
        handler._reload_config()


class TestConfigWatcher:
    """Test ConfigWatcher functionality."""

    def test_init_default(self):
        """Test watcher initialization with defaults."""
        watcher = ConfigWatcher()
        assert watcher.watch_dir == Path.cwd()
        assert not watcher._started
        assert watcher.handler.debounce_seconds == 1.0

    def test_init_custom(self):
        """Test watcher initialization with custom values."""
        watch_dir = Path("/test/dir")
        watcher = ConfigWatcher(watch_dir=watch_dir, debounce_seconds=2.0)
        assert watcher.watch_dir == watch_dir
        assert watcher.handler.debounce_seconds == 2.0

    @patch("src.services.config_watcher.Observer")
    def test_start_watching(self, mock_observer_class):
        """Test starting the watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        watcher = ConfigWatcher()
        watcher.start_watching()

        assert watcher._started
        mock_observer.schedule.assert_called_once()
        mock_observer.start.assert_called_once()

    @patch("src.services.config_watcher.Observer")
    def test_start_watching_already_started(self, mock_observer_class):
        """Test starting an already started watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        watcher = ConfigWatcher()
        watcher._started = True

        # Should not start again
        watcher.start_watching()
        mock_observer.start.assert_not_called()

    @patch("src.services.config_watcher.Observer")
    def test_stop_watching(self, mock_observer_class):
        """Test stopping the watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        watcher = ConfigWatcher()
        watcher._started = True
        watcher.stop_watching()

        assert not watcher._started
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once_with(timeout=5.0)

    @patch("src.services.config_watcher.Observer")
    def test_stop_watching_not_started(self, mock_observer_class):
        """Test stopping a non-started watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        watcher = ConfigWatcher()
        watcher._started = False

        # Should not stop if not started
        watcher.stop_watching()
        mock_observer.stop.assert_not_called()


@patch("src.services.config_watcher._watcher", None)
def test_get_config_watcher():
    """Test getting global config watcher instance."""
    from src.services.config_watcher import get_config_watcher

    # First call should create instance
    watcher1 = get_config_watcher()
    assert watcher1 is not None

    # Second call should return same instance
    watcher2 = get_config_watcher()
    assert watcher2 is watcher1
