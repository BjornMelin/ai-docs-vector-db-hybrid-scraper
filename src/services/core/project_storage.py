"""Project storage service for persistent project management."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


try:
    import aiofiles
except ImportError:
    # Fallback to synchronous file operations if aiofiles not available
    aiofiles = None

from src.services.errors import BaseError


logger = logging.getLogger(__name__)


class ProjectStorageError(BaseError):
    """Project storage specific errors."""

    pass


class ProjectStorage:
    """Manages persistent storage of project configurations."""

    def __init__(
        self,
        data_dir: str | Path,
        storage_path: str | Path | None = None,
    ):
        """Initialize project storage.

        Args:
            data_dir: Required base data directory from UnifiedConfig. Must be provided.
            storage_path: Optional custom path to storage file. If not provided, defaults to data_dir/projects.json.
                When both storage_path and data_dir are provided, storage_path takes precedence.
        """
        if storage_path is None:
            base_dir = Path(data_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            storage_path = base_dir / "projects.json"

        self.storage_path = Path(storage_path)
        self._lock = asyncio.Lock()
        self._projects_cache: dict[str, dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage and load existing projects."""
        if self._initialized:
            return

        try:
            # Create storage file if it doesn't exist
            if not self.storage_path.exists():
                await self._save_projects({})
                logger.info(f"Created new project storage at {self.storage_path}")

            # Load existing projects
            await self.load_projects()
            self._initialized = True
            logger.info(f"Loaded {len(self._projects_cache)} projects from storage")

        except Exception:
            raise ProjectStorageError("Failed to initialize project storage") from e

    async def load_projects(self) -> dict[str, dict[str, Any]]:
        """Load projects from storage file."""
        async with self._lock:
            try:
                if aiofiles:
                    async with aiofiles.open(self.storage_path, "r") as f:
                        content = await f.read()
                else:
                    # Fallback to synchronous read
                    with self.storage_path.open() as f:
                        content = f.read()

                self._projects_cache = json.loads(content) if content else {}
                return self._projects_cache.copy()
            except FileNotFoundError:
                self._projects_cache = {}
                return {}
            except json.JSONDecodeError as e:
                logger.exception("Invalid JSON in project storage")
                # Backup corrupted file
                backup_path = self.storage_path.with_suffix(".json.bak")
                if self.storage_path.exists():
                    self.storage_path.rename(backup_path)
                    logger.warning(f"Backed up corrupted file to {backup_path}")
                self._projects_cache = {}
                return {}

    async def save_project(self, project_id: str, project_data: dict[str, Any]) -> None:
        """Save a single project to storage."""
        async with self._lock:
            self._projects_cache[project_id] = project_data
            await self._save_projects(self._projects_cache)

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get a project by ID."""
        return self._projects_cache.get(project_id)

    async def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        return list(self._projects_cache.values())

    async def update_project(self, project_id: str, updates: dict[str, Any]) -> None:
        """Update a project's data."""
        async with self._lock:
            if project_id not in self._projects_cache:
                raise ProjectStorageError(f"Project {project_id} not found")

            self._projects_cache[project_id].update(updates)
            self._projects_cache[project_id]["updated_at"] = datetime.now(
                tz=UTC
            ).isoformat()
            await self._save_projects(self._projects_cache)

    async def delete_project(self, project_id: str) -> None:
        """Delete a project from storage."""
        async with self._lock:
            if project_id in self._projects_cache:
                del self._projects_cache[project_id]
                await self._save_projects(self._projects_cache)

    async def _save_projects(self, projects: dict[str, dict[str, Any]]) -> None:
        """Save projects to storage file."""
        try:
            # Write to temporary file first
            temp_path = self.storage_path.with_suffix(".tmp")

            if aiofiles:
                async with aiofiles.open(temp_path, "w") as f:
                    await f.write(json.dumps(projects, indent=2, default=str))
            else:
                # Fallback to synchronous write
                with temp_path.open("w") as f:
                    f.write(json.dumps(projects, indent=2, default=str))

            # Atomically replace the old file
            temp_path.replace(self.storage_path)

        except Exception:
            raise ProjectStorageError("Failed to save projects") from e

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._projects_cache.clear()
        self._initialized = False
