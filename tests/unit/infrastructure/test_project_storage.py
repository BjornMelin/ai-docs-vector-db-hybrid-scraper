"""Tests for infrastructure.project_storage."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.infrastructure.project_storage import ProjectStorage, ProjectStorageError


@pytest.fixture()
def storage(tmp_path: Path) -> ProjectStorage:
    """Provide a storage instance backed by a temporary directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return ProjectStorage(data_dir=data_dir)


async def test_load_projects_creates_empty_cache(storage: ProjectStorage) -> None:
    """Project store returns an empty mapping when no data exists."""
    projects = await storage.load_projects()
    assert projects == {}


async def test_save_and_load_roundtrip(storage: ProjectStorage) -> None:
    """Saved projects persist and can be reloaded."""
    payload: dict[str, Any] = {"id": "proj-1", "name": "Test"}
    await storage.save_project("proj-1", payload)

    cached = await storage.load_projects()
    assert cached["proj-1"]["id"] == "proj-1"
    assert cached["proj-1"]["name"] == "Test"
    assert "created_at" in cached["proj-1"]
    assert "updated_at" in cached["proj-1"]
    assert payload == {"id": "proj-1", "name": "Test"}


async def test_update_project_mutates_payload(storage: ProjectStorage) -> None:
    """Updating a project overwrites fields and stamps updated_at."""
    await storage.save_project("proj-1", {"id": "proj-1", "name": "Old"})
    await storage.update_project("proj-1", {"name": "New"})

    cached = await storage.load_projects()
    assert cached["proj-1"]["name"] == "New"
    assert "updated_at" in cached["proj-1"]
    parsed = datetime.fromisoformat(cached["proj-1"]["updated_at"]).astimezone(UTC)
    assert isinstance(parsed, datetime)
    assert "created_at" in cached["proj-1"]


async def test_update_missing_project_raises(storage: ProjectStorage) -> None:
    """Updating a missing project raises a storage error."""
    with pytest.raises(ProjectStorageError):
        await storage.update_project("missing", {"name": "noop"})


async def test_delete_project(storage: ProjectStorage) -> None:
    """Deleting a project removes it from subsequent loads."""
    await storage.save_project("proj-1", {"id": "proj-1"})
    await storage.delete_project("proj-1")
    cached = await storage.load_projects()
    assert "proj-1" not in cached


async def test_cache_not_mutated_on_write_failure(
    storage: ProjectStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache updates only after a successful disk write."""
    await storage.save_project("proj-1", {"id": "proj-1", "name": "Stable"})

    async def _fail_write(*_args: object, **_kwargs: object) -> None:
        raise ProjectStorageError("boom")

    monkeypatch.setattr(storage, "_write_projects", _fail_write)

    with pytest.raises(ProjectStorageError):
        await storage.save_project("proj-2", {"id": "proj-2"})

    assert await storage.get_project("proj-2") is None
    stable_project = await storage.get_project("proj-1")
    assert stable_project is not None
    assert stable_project["name"] == "Stable"

    with pytest.raises(ProjectStorageError):
        await storage.update_project("proj-1", {"name": "Mutated"})

    stable_project = await storage.get_project("proj-1")
    assert stable_project is not None
    assert stable_project["name"] == "Stable"

    with pytest.raises(ProjectStorageError):
        await storage.delete_project("proj-1")

    assert await storage.get_project("proj-1") is not None


async def test_load_projects_handles_corrupted_json(
    storage: ProjectStorage, tmp_path: Path
) -> None:
    """Corrupted JSON is backed up and replaced with an empty cache."""
    # Prime the file with invalid JSON
    path = tmp_path / "data" / "projects.json"
    path.write_text("{invalid", encoding="utf-8")

    projects = await storage.load_projects()
    assert projects == {}
    backup = path.with_suffix(".json.bak")
    assert backup.exists()
