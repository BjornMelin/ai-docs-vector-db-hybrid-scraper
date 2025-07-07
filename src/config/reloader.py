from typing import Any


class ConfigReloader:
    def __init__(self, enable_signal_handler: bool = True):
        self._file_watch_enabled = False
        self.enable_signal_handler = enable_signal_handler
        self._config_backups: list[Any] = []

    def is_file_watch_enabled(self) -> bool:
        return self._file_watch_enabled

    def get_config_backups(self) -> list[Any]:
        return self._config_backups
