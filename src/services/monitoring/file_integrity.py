"""File integrity provider abstractions and osquery-backed implementation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

CallbackType = Callable[["FileChangeEvent"], Awaitable[None] | None]
_CALLBACK_TIMEOUT_SECONDS = 2.0


class FileChangeAction(str, Enum):
    """Canonicalised file change actions emitted by providers."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ATTRIBUTES_MODIFIED = "attributes_modified"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class FileChangeEvent:  # pylint: disable=too-many-instance-attributes
    """Structured representation of a file system change notification."""

    path: Path
    action: FileChangeAction
    is_directory: bool | None = None
    size: int | None = None
    uid: int | None = None
    gid: int | None = None
    mode: int | None = None
    mtime: float | None = None
    hashes: dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    raw: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class FileIntegrityProvider(ABC):
    """Abstract base class for file integrity event providers."""

    def __init__(self) -> None:
        self._callbacks: list[CallbackType] = []
        self._running = False
        self._callback_lock = asyncio.Lock()
        self._ready = False
        self._ready_event: asyncio.Event = asyncio.Event()

    def subscribe(self, callback: CallbackType) -> None:
        """Register a callback invoked for each file change event."""

        self._callbacks.append(callback)

    @abstractmethod
    async def start(self) -> None:
        """Start streaming file system events."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming file system events and release resources."""

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait until the provider signals readiness."""

        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout)
        except TimeoutError:
            return False
        return self._ready

    def is_ready(self) -> bool:
        """Return whether the provider is currently ready to emit events."""

        return self._ready

    async def health_check(self) -> dict[str, Any]:
        """Return provider health details suitable for diagnostics."""

        return {"ready": self.is_ready()}

    def _mark_ready(self) -> None:
        self._ready = True
        self._ready_event.set()

    def _mark_not_ready(self) -> None:
        self._ready = False
        self._ready_event.clear()

    async def _notify(self, event: FileChangeEvent) -> None:
        """Notify registered callbacks of a file change event."""

        async with self._callback_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=_CALLBACK_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(
                    "File integrity callback exceeded %s seconds",
                    _CALLBACK_TIMEOUT_SECONDS,
                )
            except Exception:  # pragma: no cover - defensive path
                logger.exception("File integrity callback failed")


class StubFileIntegrityProvider(FileIntegrityProvider):
    """In-memory provider for deterministic testing."""

    async def start(self) -> None:
        """Start the stub provider."""

        self._running = True
        self._mark_ready()

    async def stop(self) -> None:
        """Stop the stub provider."""

        self._running = False
        self._mark_not_ready()

    async def emit(self, event: FileChangeEvent) -> None:
        """Emit a single event to registered callbacks."""

        if not self._running:
            return
        await self._notify(event)


class OsqueryFileIntegrityProvider(FileIntegrityProvider):  # pylint: disable=too-many-instance-attributes
    """Provider that tails ``osqueryd.results.log`` for file change events."""

    def __init__(
        self,
        *,
        results_log: Path,
        include_globs: Sequence[str] | None = None,
        exclude_globs: Sequence[str] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.results_log = Path(results_log).expanduser().resolve()
        self.include_globs = tuple(include_globs or ["*"])
        self.exclude_globs = tuple(exclude_globs or [])
        self.poll_interval = poll_interval
        self._watch_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start monitoring the osquery results log."""

        if self._watch_task is not None and not self._watch_task.done():
            return
        self._running = True
        self._mark_not_ready()
        self._watch_task = asyncio.create_task(self._watch_loop(), name="osquery-fim")

    async def stop(self) -> None:
        """Stop monitoring and cancel the watch task."""

        self._running = False
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        self._watch_task = None
        self._mark_not_ready()

    async def health_check(self) -> dict[str, Any]:
        """Return readiness information and the monitored log path."""

        return {
            "ready": self.is_ready(),
            "results_log": str(self.results_log),
        }

    async def _watch_loop(self) -> None:
        """Main loop for tailing the osquery results log."""

        file_handle: Any | None = None
        last_inode: int | None = None

        try:
            while self._running:
                if file_handle is None:
                    try:
                        file_handle = self.results_log.open(  # pylint: disable=consider-using-with
                            "r", encoding="utf-8"
                        )
                        file_handle.seek(0, os.SEEK_END)
                        last_inode = os.fstat(file_handle.fileno()).st_ino
                        self._mark_ready()
                        logger.debug(
                            "File integrity provider monitoring %s", self.results_log
                        )
                    except FileNotFoundError:
                        self._mark_not_ready()
                        await asyncio.sleep(self.poll_interval)
                        continue

                line = file_handle.readline()
                if not line:
                    await asyncio.sleep(self.poll_interval)
                    if not self.results_log.exists():
                        file_handle.close()
                        file_handle = None
                        self._mark_not_ready()
                    else:
                        current_inode = os.fstat(file_handle.fileno()).st_ino
                        if last_inode is not None and current_inode != last_inode:
                            file_handle.close()
                            file_handle = None
                            self._mark_not_ready()
                    continue

                event = self._parse_osquery_event(line)
                if event is None:
                    continue

                if self._filter_event(event.path):
                    await self._notify(event)
        except Exception:  # pragma: no cover - defensive path
            logger.exception("Error while tailing osquery results log")
        finally:
            if file_handle is not None:
                file_handle.close()
            self._mark_not_ready()

    def _filter_event(self, path: Path) -> bool:
        """Determine if a file event is included based on include/exclude globs."""

        path_str = str(path)
        included = any(fnmatch(path_str, pattern) for pattern in self.include_globs)
        excluded = any(fnmatch(path_str, pattern) for pattern in self.exclude_globs)
        return included and not excluded

    def _parse_osquery_event(self, line: str) -> FileChangeEvent | None:
        """Parse a JSON line from osquery into a FileChangeEvent."""

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON osquery line: %s", line.strip())
            return None

        if payload.get("name") != "file_events":
            return None

        columns = payload.get("columns", {})
        raw_path = columns.get("target_path") or columns.get("path")
        if not raw_path:
            return None

        action_raw = columns.get("action", "").upper()
        action = self._map_action(action_raw)

        hashes = {
            key: value
            for key, value in columns.items()
            if key in {"md5", "sha1", "sha256"} and value
        }

        def _coerce_int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        event = FileChangeEvent(
            path=Path(raw_path),
            action=action,
            is_directory=None,
            size=_coerce_int(columns.get("size")),
            uid=_coerce_int(columns.get("uid")),
            gid=_coerce_int(columns.get("gid")),
            mode=_coerce_int(columns.get("mode")),
            mtime=float(columns.get("time", 0.0)) if columns.get("time") else None,
            hashes=hashes,
            source="osquery",
            raw=payload,
        )
        return event

    @staticmethod
    def _map_action(action: str) -> FileChangeAction:
        """Map osquery action string to FileChangeAction enum."""

        if action in {"CREATED", "ADDED"}:
            return FileChangeAction.CREATED
        if action in {"UPDATED", "MODIFIED"}:
            return FileChangeAction.UPDATED
        if action in {"DELETED", "REMOVED"}:
            return FileChangeAction.DELETED
        if action in {"ATTRIBUTES_MODIFIED", "MOVED"}:
            return FileChangeAction.ATTRIBUTES_MODIFIED
        return FileChangeAction.UNKNOWN


__all__ = [
    "FileChangeAction",
    "FileChangeEvent",
    "FileIntegrityProvider",
    "OsqueryFileIntegrityProvider",
    "StubFileIntegrityProvider",
]
