"""Runtime configuration reloader supporting hot updates and rollbacks."""

# pylint: disable=too-many-instance-attributes, import-outside-toplevel

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.services.monitoring.file_integrity import (
    FileChangeEvent,
    FileIntegrityProvider,
)

from .loader import Config, get_config, set_config


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Base exception for configuration reload failures."""


class ConfigLoadError(ConfigError):
    """Raised when a configuration source cannot be loaded."""


class ConfigReloadError(ConfigError):
    """Raised when a reload or rollback operation fails."""


class ReloadStatus(str, Enum):
    """Lifecycle states recorded for configuration operations."""

    COMPLETED = "completed"
    FAILED = "failed"


class ReloadTrigger(str, Enum):
    """Context describing what initiated a reload attempt."""

    API = "api"
    FILE_WATCH = "file_watch"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass(slots=True)
class ConfigBackup:
    """Materialised backup of a previous configuration instance."""

    config: Config
    created_at: datetime
    environment: str


@dataclass(slots=True)
class ReloadOperation:
    """Structured record describing the outcome of a reload or rollback."""

    operation_id: str
    trigger: ReloadTrigger
    status: ReloadStatus
    success: bool
    error_message: str | None
    total_duration_ms: float
    validation_duration_ms: float
    apply_duration_ms: float
    previous_config_hash: str | None
    new_config_hash: str | None
    changes_applied: list[str] = field(default_factory=list)
    services_notified: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConfigReloader:
    """Orchestrates hot reloads and rollbacks of the application configuration."""

    def __init__(
        self,
        *,
        history_limit: int = 50,
        config_source: Path | str | None = None,
    ) -> None:
        self._default_config_source = (
            Path(config_source).expanduser().resolve() if config_source else None
        )
        self._file_watch_enabled = False
        self._fim_provider: FileIntegrityProvider | None = None
        self._watch_target_is_dir = (
            self._default_config_source.is_dir()
            if self._default_config_source and self._default_config_source.exists()
            else False
        )
        self._history: deque[ReloadOperation] = deque(maxlen=history_limit)
        self._backups: deque[tuple[str, ConfigBackup]] = deque(maxlen=history_limit)
        self._lock = asyncio.Lock()
        self._total_operations = 0
        self._successful_operations = 0
        self._failed_operations = 0
        self._aggregate_duration_ms = 0.0

    async def reload_config(  # pylint: disable=too-many-locals
        self,
        *,
        trigger: ReloadTrigger = ReloadTrigger.MANUAL,
        config_source: Path | None = None,
        force: bool = False,
    ) -> ReloadOperation:
        """Reload configuration from environment variables or an override source.

        Args:
            trigger: Identifier describing what initiated the reload.
            config_source: Optional path to JSON/YAML/env file providing overrides.
            force: When ``True`` the reload is applied even if no effective change
                is detected in the configuration payload.

        Returns:
            ReloadOperation: Structured details describing the attempt.
        """

        async with self._lock:
            previous_config = get_config()
            previous_hash = self._hash_config(previous_config)
            if previous_config is not None:
                self._store_backup(previous_hash, previous_config)

            operation_id = uuid4().hex
            start = perf_counter()
            new_config_hash: str | None = None
            error_message: str | None = None
            success = False
            changes_applied: list[str] = []

            try:
                if config_source is not None:
                    self.set_default_config_source(config_source)
                replacement = self._load_config_source(
                    config_source or self._default_config_source
                )
                replacement_hash = self._hash_config(replacement)

                if not force and replacement_hash == previous_hash:
                    logger.debug(
                        "Reload skipped: configuration hash unchanged (%s)",
                        replacement_hash,
                    )
                    success = True
                    new_config_hash = previous_hash
                else:
                    set_config(replacement)
                    success = True
                    new_config_hash = replacement_hash
                    if replacement_hash != previous_hash:
                        changes_applied.append("reload")
            except Exception as exc:  # pragma: no cover - defensive path
                logger.exception("Configuration reload failed")
                error_message = str(exc)
                # Restore previous configuration snapshot on failure.
                if previous_config is not None:
                    set_config(previous_config)
            finally:
                duration_ms = (perf_counter() - start) * 1000.0
                operation = ReloadOperation(
                    operation_id=operation_id,
                    trigger=trigger,
                    status=ReloadStatus.COMPLETED if success else ReloadStatus.FAILED,
                    success=success,
                    error_message=error_message,
                    total_duration_ms=duration_ms,
                    validation_duration_ms=0.0,
                    apply_duration_ms=duration_ms,
                    previous_config_hash=previous_hash,
                    new_config_hash=new_config_hash if success else previous_hash,
                    changes_applied=changes_applied,
                )
                self._record_operation(operation)

            return operation

    async def rollback_config(
        self,
        *,
        target_hash: str | None = None,
    ) -> ReloadOperation:
        """Roll back to a previous configuration snapshot.

        Args:
            target_hash: Optional hash pointing to a specific backup.

        Returns:
            ReloadOperation: Outcome for the rollback attempt.
        """

        async with self._lock:
            operation_id = uuid4().hex
            start = perf_counter()
            previous_config = get_config()
            previous_hash = self._hash_config(previous_config)
            backup_hash, backup = self._select_backup(target_hash)
            success = backup is not None
            error_message: str | None = None

            if success and backup is not None:
                set_config(backup.config.model_copy(deep=True))
            else:
                error_message = "No matching configuration backup available"

            duration_ms = (perf_counter() - start) * 1000.0
            operation = ReloadOperation(
                operation_id=operation_id,
                trigger=ReloadTrigger.MANUAL,
                status=ReloadStatus.COMPLETED if success else ReloadStatus.FAILED,
                success=success,
                error_message=error_message,
                total_duration_ms=duration_ms,
                validation_duration_ms=0.0,
                apply_duration_ms=duration_ms,
                previous_config_hash=previous_hash,
                new_config_hash=backup_hash if success else previous_hash,
                changes_applied=["rollback"] if success else [],
            )
            self._record_operation(operation)

            return operation

    def get_reload_history(self, limit: int = 20) -> list[ReloadOperation]:
        """Return the most recent reload or rollback operations."""

        return list(self._history)[:limit]

    def get_default_config_source(self) -> Path | None:
        """Return the default configuration source path, if configured."""

        return self._default_config_source

    def set_default_config_source(self, config_source: Path | str | None) -> None:
        """Set the default configuration source used for reloads and watching."""

        if config_source is None:
            self._default_config_source = None
            self._watch_target_is_dir = False
            return

        path = Path(config_source).expanduser().resolve()
        if not path.exists():
            raise ConfigLoadError(f"Configuration source not found: {path}")
        if not (path.is_file() or path.is_dir()):
            raise ConfigLoadError(
                f"Configuration source must be a file or directory: {path}"
            )
        self._default_config_source = path
        self._watch_target_is_dir = path.is_dir()

    def get_reload_stats(self) -> dict[str, object]:
        """Return aggregate counters describing reload behaviour."""

        total = self._total_operations
        success_rate = self._successful_operations / total if total else 0.0
        average_duration = self._aggregate_duration_ms / total if total else 0.0
        return {
            "total_operations": total,
            "successful_operations": self._successful_operations,
            "failed_operations": self._failed_operations,
            "success_rate": success_rate,
            "average_duration_ms": average_duration,
            "backups_available": len(self._backups),
            "current_config_hash": self._hash_config(get_config()),
        }

    async def attach_file_integrity_provider(
        self, provider: FileIntegrityProvider | None
    ) -> None:
        """Attach a file integrity provider used for change notifications."""

        if provider is self._fim_provider:
            return

        if self._fim_provider is not None:
            with contextlib.suppress(Exception):
                await self._fim_provider.stop()

        self._fim_provider = provider

        if self._fim_provider is not None:
            self._fim_provider.subscribe(self._on_file_change)
            if self._file_watch_enabled:
                await self._fim_provider.start()

    async def enable_file_watching(self, poll_interval: float | None = None) -> None:
        """Enable file watching through the configured integrity provider."""

        target = self._resolve_watch_target()

        if self._fim_provider is None:
            msg = (
                "File integrity provider not configured; call "
                "attach_file_integrity_provider() first"
            )
            raise ConfigError(msg)

        if poll_interval is not None:
            logger.debug(
                "ConfigReloader.enable_file_watching received poll_interval=%s but "
                "the current provider manages scheduling internally",
                poll_interval,
            )

        if self._file_watch_enabled:
            logger.debug("File watching already active for %s", target)
            return

        await self._fim_provider.start()
        self._file_watch_enabled = True

    async def disable_file_watching(self) -> None:
        """Disable file watching hooks for configuration changes."""

        if not self._file_watch_enabled:
            return

        self._file_watch_enabled = False
        if self._fim_provider is not None:
            with contextlib.suppress(Exception):
                await self._fim_provider.stop()

    def is_file_watch_enabled(self) -> bool:
        """Return whether on-disk file watching is active."""

        return self._file_watch_enabled

    def get_file_integrity_provider(self) -> FileIntegrityProvider | None:
        """Return the currently configured file integrity provider."""

        return self._fim_provider

    def _resolve_watch_target(self) -> Path:
        """Validate and return the path being watched for changes."""

        source = self._default_config_source
        if source is None:
            msg = "Cannot enable file watching without a configured source path"
            raise ConfigError(msg)

        if not source.exists():
            msg = f"Configuration source not found for watching: {source}"
            raise ConfigLoadError(msg)

        self._watch_target_is_dir = source.is_dir()
        return source

    def get_config_backups(self) -> Iterator[tuple[str, ConfigBackup]]:
        """Yield configuration backups starting with the newest snapshot."""

        yield from self._backups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_operation(self, operation: ReloadOperation) -> None:
        """Record a completed reload or rollback operation in the history."""

        self._history.appendleft(operation)
        self._total_operations += 1
        if operation.success:
            self._successful_operations += 1
        else:
            self._failed_operations += 1
        self._aggregate_duration_ms += operation.total_duration_ms

    async def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle file integrity events emitted by the provider."""

        if self._default_config_source is None:
            return

        try:
            event_path = Path(event.path).resolve()
        except Exception:
            logger.debug("Ignoring malformed file integrity event: %s", event)
            return

        target = self._default_config_source
        target_path = target.resolve()

        relevant = False
        if self._watch_target_is_dir:
            with contextlib.suppress(ValueError):
                event_path.relative_to(target_path)
                relevant = True
        else:
            relevant = event_path == target_path

        if not relevant:
            return

        logger.debug(
            "Detected configuration change via file integrity event: %s", event
        )
        try:
            await self.reload_config(
                trigger=ReloadTrigger.FILE_WATCH,
                config_source=target,
                force=True,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("File integrity reload failed: %s", exc)

    def _store_backup(self, config_hash: str | None, config: Config) -> None:
        """Store a snapshot of the current configuration for potential rollback."""

        if config_hash is None:
            return
        if self._backups and self._backups[0][0] == config_hash:
            return
        snapshot = config.model_copy(deep=True)
        backup = ConfigBackup(
            config=snapshot,
            created_at=datetime.now(UTC),
            environment=str(snapshot.environment.value)
            if hasattr(snapshot, "environment")
            else "unknown",
        )
        self._backups.appendleft((config_hash, backup))

    def _select_backup(
        self, target_hash: str | None
    ) -> tuple[str, ConfigBackup] | tuple[None, None]:
        """Select a backup matching the specified hash, or the most recent one."""

        if not self._backups:
            return None, None
        if target_hash is None:
            return self._backups[0]
        for backup_hash, backup in self._backups:
            if backup_hash == target_hash:
                return backup_hash, backup
        return None, None

    def _load_config_source(self, config_source: Path | None) -> Config:
        """Load configuration from the specified source."""

        if config_source is None:
            return Config()
        path = config_source.expanduser()
        if not path.exists():
            raise ConfigLoadError(f"Configuration source not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return Config(**data)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ConfigLoadError(
                    "PyYAML is required to load YAML configuration"
                ) from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):  # pragma: no cover - defensive guard
                raise ConfigLoadError("YAML configuration must decode to an object")
            return Config(**data)
        return Config(_env_file=str(path))  # type: ignore[call-arg]

    @staticmethod
    def _hash_config(config: Config | None) -> str | None:
        """Return a short hash representing the current configuration state."""

        if config is None:
            return None
        payload = config.model_dump(mode="json")
        serialised = json.dumps(payload, sort_keys=True).encode("utf-8")
        return _sha256(serialised)


def _sha256(data: bytes) -> str:
    """Return a short hash for cached configuration snapshots."""

    from hashlib import sha256

    digest = sha256(data).hexdigest()
    return f"sha256:{digest}"
