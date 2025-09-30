"""High-level configuration manager and graceful degradation utilities."""

# pylint: disable=too-many-instance-attributes, too-many-arguments

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from .loader import Config, load_config
from .reloader import ConfigLoadError


@dataclass(slots=True)
class DegradationRecord:
    """Recorded failure information for diagnostics."""

    operation: str
    error: str
    context: dict[str, Any]


class GracefulDegradationHandler:
    """Track repeated failures and optionally pause sensitive operations."""

    def __init__(self, *, threshold: int = 5) -> None:
        self._threshold = threshold
        self._failure_counts: dict[str, int] = {}
        self._records: deque[DegradationRecord] = deque(maxlen=100)
        self._lock = Lock()

    def record_failure(
        self, operation: str, error: Exception, context: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._failure_counts[operation] = self._failure_counts.get(operation, 0) + 1
            self._records.append(
                DegradationRecord(
                    operation=operation,
                    error=str(error),
                    context=context or {},
                )
            )

    def should_skip_operation(self, operation: str) -> bool:
        with self._lock:
            return self._failure_counts.get(operation, 0) >= self._threshold

    @property
    def degradation_active(self) -> bool:
        with self._lock:
            return any(
                count >= self._threshold for count in self._failure_counts.values()
            )

    def reset(self) -> None:
        with self._lock:
            self._failure_counts.clear()
            self._records.clear()

    def iter_records(self) -> Iterable[DegradationRecord]:
        with self._lock:
            yield from list(self._records)


_global_degradation_handler = GracefulDegradationHandler()


class ConfigManager:
    """Manage configuration sourced from files with optional fallbacks."""

    def __init__(
        self,
        *,
        config_class: type[Config] = Config,
        config_file: Path | None = None,
        fallback_config: Config | None = None,
        enable_file_watching: bool = False,
        enable_graceful_degradation: bool = False,
        backup_limit: int = 10,
    ) -> None:
        self.config_class = config_class
        self.config_file = Path(config_file) if config_file else None
        self.fallback_config = fallback_config
        self.enable_file_watching = enable_file_watching
        self.enable_graceful_degradation = enable_graceful_degradation
        self._degradation = _global_degradation_handler
        self._config_backups: deque[Config] = deque(maxlen=backup_limit)
        self._recent_failures: deque[dict[str, Any]] = deque(maxlen=20)
        self._change_listeners: list[Callable[[Config, Config], None]] = []
        self.safe_loader = _SafeLoader(self)
        self._async_lock = asyncio.Lock()
        self._sync_lock = Lock()
        self._config = self._load_initial_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> Config:
        return self._config

    def get_reload_history(self, limit: int = 20) -> list[Config]:
        return list(self._config_backups)[:limit]

    def get_config_backups(self) -> list[Config]:
        return list(self._config_backups)

    def reload_config(self) -> bool:
        return asyncio.run(self.reload_config_async())

    async def reload_config_async(self) -> bool:
        async with self._async_lock:
            return await self._reload_internal_async()

    def rollback_config(self) -> bool:
        with self._sync_lock:
            if not self._config_backups:
                return False
            previous = list(self._config_backups)[-1]
            self._config = previous
            return True

    def add_change_listener(self, listener: Callable[[Config, Config], None]) -> None:
        self._change_listeners.append(listener)

    def remove_change_listener(
        self, listener: Callable[[Config, Config], None]
    ) -> None:
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def restore_from_backup(self, index: int) -> bool:
        with self._sync_lock:
            try:
                backup = list(self._config_backups)[index]
            except IndexError:
                return False
            self._config = backup
            return True

    def get_status(self) -> dict[str, Any]:
        return {
            "watching_enabled": self.enable_file_watching,
            "graceful_degradation": self.enable_graceful_degradation,
            "recent_failures": list(self._recent_failures),
            "backups_available": len(self._config_backups),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _reload_internal_async(self) -> bool:
        if self.enable_graceful_degradation and self._degradation.should_skip_operation(
            "reload_config"
        ):
            return False

        try:
            loaded = await self._load_from_source_async()
        except Exception as exc:  # pragma: no cover - defensive
            self._record_failure("reload_config", exc)
            return False

        with self._sync_lock:
            self._record_backup(self._config)
            old_config = self._config
            self._config = loaded
        self._notify_listeners(old_config, loaded)
        return True

    def _load_initial_config(self) -> Config:
        if self.config_file and self.config_file.exists():
            try:
                return asyncio.run(self._load_from_source_async())
            except Exception as exc:  # pragma: no cover - defensive
                self._record_failure("initial_load", exc)
        if self.fallback_config is not None:
            return self.fallback_config
        return load_config()

    async def _load_from_source_async(self) -> Config:
        if not self.config_file:
            return load_config()
        data_or_config = self.safe_loader.load_from_file(self.config_file)
        if asyncio.iscoroutine(data_or_config):
            data_or_config = await data_or_config
        if isinstance(data_or_config, self.config_class):
            return data_or_config
        payload = self.fallback_config.model_dump() if self.fallback_config else {}
        payload.update(data_or_config)
        return self.config_class(**payload)

    def _record_backup(self, config: Config) -> None:
        self._config_backups.append(config.model_copy(deep=True))

    def _notify_listeners(self, old_config: Config, new_config: Config) -> None:
        for listener in list(self._change_listeners):
            try:
                listener(old_config, new_config)
            except Exception as exc:  # pragma: no cover - defensive
                self._record_failure("listener", exc, {"listener": repr(listener)})

    def _record_failure(
        self, operation: str, error: Exception, context: dict[str, Any] | None = None
    ) -> None:
        entry = {
            "operation": operation,
            "error": str(error),
            "context": context or {},
        }
        self._recent_failures.appendleft(entry)
        if self.enable_graceful_degradation:
            self._degradation.record_failure(operation, error, context or {})

    def _parse_file(self, path: Path) -> dict[str, Any] | Config:
        if not path.exists():
            raise ConfigLoadError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text()) or {}
        if suffix == ".json":
            return json.loads(path.read_text())
        if suffix in {".env", ".ini"}:
            return load_config(_env_file=str(path))
        raise ConfigLoadError(f"Unsupported configuration file format: {path.suffix}")


class _SafeLoader:
    """Proxy object that can be monkey-patched in tests."""

    def __init__(self, manager: ConfigManager) -> None:
        self._manager = manager

    def load_from_file(self, path: Path) -> dict[str, Any] | Config:
        return self._manager._parse_file(path)


__all__ = [
    "ConfigManager",
    "ConfigLoadError",
    "GracefulDegradationHandler",
    "get_degradation_handler",
]


def get_degradation_handler() -> GracefulDegradationHandler:
    """Return the global degradation handler instance."""

    return _global_degradation_handler
