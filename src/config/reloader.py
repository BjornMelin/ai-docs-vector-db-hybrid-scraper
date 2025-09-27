from typing import Any


class ConfigReloader:
    """Configuration reloader for dynamic config updates.

    Provides functionality to reload configuration files at runtime and maintain
    backups of previous configurations for rollback purposes.
    """

    def __init__(self, enable_signal_handler: bool = True):
        self._file_watch_enabled = False
        self.enable_signal_handler = enable_signal_handler
        self._config_backups: list[Any] = []

    def is_file_watch_enabled(self) -> bool:
        """Check if file watching is enabled for configuration reloading.

        Returns:
            bool: True if file watching is enabled, False otherwise.
        """
        return self._file_watch_enabled

    def get_config_backups(self) -> list[Any]:
        """Retrieve the list of configuration backups.

        Returns:
            list[Any]: A list containing previous configuration states for rollback.
        """
        return self._config_backups
