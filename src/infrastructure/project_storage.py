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
                - Relative storage_path values are resolved relative to data_dir.
        """
        base_dir = Path(data_dir)
        if storage_path:
            target_path = Path(storage_path)
            if not target_path.is_absolute():
                target_path = base_dir / target_path
        else:
            target_path = base_dir / "projects.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        self._path = target_path
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    async def load_projects(self) -> dict[str, dict[str, Any]]:
        """Load all projects from disk into the in-memory cache."""
        async with self._lock:
            self._cache = await self._read_projects()
            return {key: dict(value) for key, value in self._cache.items()}

    async def save_project(self, project_id: str, project_data: dict[str, Any]) -> None:
        """Persist a single project definition."""
        async with self._lock:
            now = datetime.now(UTC).isoformat()
            previous = self._cache.get(project_id, {})
            payload = dict(project_data)
            if "created_at" not in payload:
                payload["created_at"] = previous.get("created_at", now)
            payload["updated_at"] = now

            updated_cache = dict(self._cache)
            updated_cache[project_id] = payload
            await self._write_projects(updated_cache)
            self._cache = updated_cache

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Return a previously cached project by identifier."""
        async with self._lock:
            project = self._cache.get(project_id)
            return dict(project) if project else None

    async def list_projects(self) -> list[dict[str, Any]]:
        """Return the cached project list."""
        async with self._lock:
            return [dict(project) for project in self._cache.values()]

    async def update_project(self, project_id: str, updates: dict[str, Any]) -> None:
        """Apply partial updates to an existing project."""
        async with self._lock:
            if project_id not in self._cache:
                msg = f"Project {project_id} not found"
                raise ProjectStorageError(msg)

            updated_cache = dict(self._cache)
            updated_project = dict(updated_cache[project_id])
            updated_project.update(updates)
            updated_project["updated_at"] = datetime.now(UTC).isoformat()
            updated_cache[project_id] = updated_project
            await self._write_projects(updated_cache)
            self._cache = updated_cache

    async def delete_project(self, project_id: str) -> None:
        """Remove a project from storage.

        Missing projects are ignored (idempotent delete).
        """
        async with self._lock:
            updated_cache = dict(self._cache)
            if updated_cache.pop(project_id, None) is None:
                return

            await self._write_projects(updated_cache)
            self._cache = updated_cache

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
