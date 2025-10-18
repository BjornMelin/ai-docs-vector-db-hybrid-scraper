"""Filesystem-backed storage for project metadata."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.services.errors import BaseError


logger = logging.getLogger(__name__)


class ProjectStorageError(BaseError):
    """Raised when project persistence fails."""


class ProjectStorage:
    """Persist and retrieve project definitions from a JSON file."""

    def __init__(self, data_dir: str | Path, storage_path: str | Path | None = None):
        """Initialize project storage.

        Args:
            data_dir: The directory to store the project data.
            storage_path: Path to the project data file.
                - If not provided, data is stored in the data_dir/projects.json file.
                - When storage_path + data_dir provided, storage_path takes precedence.
        """
        base_dir = Path(data_dir)
        target_path = Path(storage_path) if storage_path else base_dir / "projects.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        self._path = target_path
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    async def load_projects(self) -> dict[str, dict[str, Any]]:
        """Load all projects from disk into the in-memory cache."""
        async with self._lock:
            self._cache = await self._read_projects()
            return self._cache.copy()

    async def save_project(self, project_id: str, project_data: dict[str, Any]) -> None:
        """Persist a single project definition."""
        async with self._lock:
            self._cache[project_id] = project_data
            await self._write_projects(self._cache)

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Return a previously cached project by identifier."""
        return self._cache.get(project_id)

    async def list_projects(self) -> list[dict[str, Any]]:
        """Return the cached project list."""
        return list(self._cache.values())

    async def update_project(self, project_id: str, updates: dict[str, Any]) -> None:
        """Apply partial updates to an existing project."""
        async with self._lock:
            if project_id not in self._cache:
                msg = f"Project {project_id} not found"
                raise ProjectStorageError(msg)

            self._cache[project_id].update(updates)
            self._cache[project_id]["updated_at"] = datetime.now(UTC).isoformat()
            await self._write_projects(self._cache)

    async def delete_project(self, project_id: str) -> None:
        """Remove a project from storage."""
        async with self._lock:
            if self._cache.pop(project_id, None) is not None:
                await self._write_projects(self._cache)

    async def _read_projects(self) -> dict[str, dict[str, Any]]:
        """Read projects from disk."""

        def _read() -> dict[str, dict[str, Any]]:
            if not self._path.exists():
                return {}

            text = self._path.read_text(encoding="utf-8")
            if not text:
                return {}

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                backup_path = self._path.with_suffix(".json.bak")
                self._path.replace(backup_path)
                logger.warning(
                    "Invalid JSON in %s; backed up to %s", self._path, backup_path
                )
                return {}

        try:
            return await asyncio.to_thread(_read)
        except Exception as exc:  # pragma: no cover - unexpected I/O failure
            msg = f"Failed to load projects from {self._path}"
            raise ProjectStorageError(msg) from exc

    async def _write_projects(self, projects: dict[str, dict[str, Any]]) -> None:
        """Write projects to disk."""

        def _write() -> None:
            """Write projects to disk."""
            temp_path = self._path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(projects, indent=2, default=str), encoding="utf-8"
            )
            temp_path.replace(self._path)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:  # pragma: no cover - unexpected I/O failure
            msg = f"Failed to save projects to {self._path}"
            raise ProjectStorageError(msg) from exc
