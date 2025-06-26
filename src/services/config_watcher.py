"""Simple configuration file watcher for development environments.

This module provides automatic configuration reloading when .env files change,
using the watchdog library for file system monitoring.
"""

import logging
import time
from pathlib import Path
from threading import Timer

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config.settings import reload_settings


logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """Handle .env file changes with debouncing."""

    def __init__(self, debounce_seconds: float = 1.0):
        """Initialize handler with debounce timer.

        Args:
            debounce_seconds: Time to wait before reloading after changes
        """
        self.debounce_seconds = debounce_seconds
        self._timer: Timer | None = None
        self._last_reload = 0.0

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if it's a .env file
        if Path(event.src_path).name.startswith(".env"):
            logger.debug(f"Detected change in {event.src_path}")
            self._schedule_reload()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        self.on_modified(event)

    def _schedule_reload(self) -> None:
        """Schedule a configuration reload with debouncing."""
        # Cancel existing timer if any
        if self._timer and self._timer.is_alive():
            self._timer.cancel()

        # Schedule new reload
        self._timer = Timer(self.debounce_seconds, self._reload_config)
        self._timer.start()

    def _reload_config(self) -> None:
        """Reload configuration after debounce period."""
        try:
            # Prevent rapid reloads
            current_time = time.time()
            if current_time - self._last_reload < self.debounce_seconds:
                return

            self._last_reload = current_time
            logger.info("Reloading configuration...")
            reload_settings()
            logger.info("Configuration reloaded successfully")
        except Exception:
            logger.exception("Failed to reload configuration")


class ConfigWatcher:
    """Simple configuration file watcher."""

    def __init__(self, watch_dir: Path | None = None, debounce_seconds: float = 1.0):
        """Initialize config watcher.

        Args:
            watch_dir: Directory to watch (defaults to current directory)
            debounce_seconds: Time to wait before reloading after changes
        """
        self.watch_dir = watch_dir or Path.cwd()
        self.observer = Observer()
        self.handler = ConfigFileHandler(debounce_seconds)
        self._started = False

    def start_watching(self) -> None:
        """Start watching for configuration changes."""
        if self._started:
            logger.warning("Config watcher already started")
            return

        try:
            self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
            self.observer.start()
            self._started = True
            logger.info(f"Started watching for config changes in {self.watch_dir}")
        except Exception:
            logger.exception("Failed to start config watcher")
            raise

    def stop_watching(self) -> None:
        """Stop watching for configuration changes."""
        if not self._started:
            return

        try:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self._started = False
            logger.info("Stopped config watcher")
        except Exception:
            logger.exception("Error stopping config watcher")


# Global watcher instance
_watcher: ConfigWatcher | None = None


def get_config_watcher() -> ConfigWatcher:
    """Get or create global config watcher instance."""
    global _watcher
    if _watcher is None:
        _watcher = ConfigWatcher()
    return _watcher
