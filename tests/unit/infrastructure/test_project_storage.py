"""Tests for project storage functionality."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from src.services.core.project_storage import ProjectStorage
from src.services.core.project_storage import ProjectStorageError


@pytest.fixture
async def temp_storage_path():
    """Create a temporary storage file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "projects.json"
        yield path


@pytest.fixture
async def storage(temp_storage_path):
    """Create a ProjectStorage instance with temporary file."""
    storage = ProjectStorage(
        data_dir=temp_storage_path.parent, storage_path=temp_storage_path
    )
    await storage.initialize()
    return storage


@pytest.mark.asyncio
async def test_initialize_creates_file(temp_storage_path):
    """Test that initialization creates storage file."""
    storage = ProjectStorage(
        data_dir=temp_storage_path.parent, storage_path=temp_storage_path
    )
    assert not temp_storage_path.exists()

    await storage.initialize()

    assert temp_storage_path.exists()
    with open(temp_storage_path) as f:
        data = json.load(f)
    assert data == {}


@pytest.mark.asyncio
async def test_save_and_load_project(storage):
    """Test saving and loading a project."""
    project_id = "test-123"
    project_data = {
        "id": project_id,
        "name": "Test Project",
        "description": "Test description",
        "collection": "test_collection",
    }

    # Save project
    await storage.save_project(project_id, project_data)

    # Load projects
    projects = await storage.load_projects()

    assert project_id in projects
    assert projects[project_id] == project_data


@pytest.mark.asyncio
async def test_update_project(storage):
    """Test updating project data."""
    project_id = "test-456"
    initial_data = {
        "id": project_id,
        "name": "Initial Name",
        "description": "Initial description",
    }

    await storage.save_project(project_id, initial_data)

    # Update project
    updates = {"name": "Updated Name", "new_field": "new_value"}
    await storage.update_project(project_id, updates)

    # Verify updates
    project = await storage.get_project(project_id)
    assert project["name"] == "Updated Name"
    assert project["description"] == "Initial description"
    assert project["new_field"] == "new_value"
    assert "updated_at" in project


@pytest.mark.asyncio
async def test_delete_project(storage):
    """Test deleting a project."""
    project_id = "test-789"
    project_data = {"id": project_id, "name": "To Delete"}

    await storage.save_project(project_id, project_data)
    assert await storage.get_project(project_id) is not None

    # Delete project
    await storage.delete_project(project_id)

    assert await storage.get_project(project_id) is None
    projects = await storage.list_projects()
    assert len(projects) == 0


@pytest.mark.asyncio
async def test_list_projects(storage):
    """Test listing all projects."""
    projects_data = {
        "proj1": {"id": "proj1", "name": "Project 1"},
        "proj2": {"id": "proj2", "name": "Project 2"},
        "proj3": {"id": "proj3", "name": "Project 3"},
    }

    for proj_id, data in projects_data.items():
        await storage.save_project(proj_id, data)

    projects = await storage.list_projects()

    assert len(projects) == 3
    names = {p["name"] for p in projects}
    assert names == {"Project 1", "Project 2", "Project 3"}


@pytest.mark.asyncio
async def test_persistence_across_instances(temp_storage_path):
    """Test that projects persist across storage instances."""
    # First instance
    storage1 = ProjectStorage(
        data_dir=temp_storage_path.parent, storage_path=temp_storage_path
    )
    await storage1.initialize()

    project_data = {"id": "persist-test", "name": "Persistent Project"}
    await storage1.save_project("persist-test", project_data)
    await storage1.cleanup()

    # Second instance
    storage2 = ProjectStorage(
        data_dir=temp_storage_path.parent, storage_path=temp_storage_path
    )
    await storage2.initialize()

    projects = await storage2.load_projects()
    assert "persist-test" in projects
    assert projects["persist-test"]["name"] == "Persistent Project"


@pytest.mark.asyncio
async def test_concurrent_access(storage):
    """Test concurrent project operations."""

    async def save_project(i):
        project_id = f"concurrent-{i}"
        await storage.save_project(project_id, {"id": project_id, "index": i})

    # Save multiple projects concurrently
    await asyncio.gather(*[save_project(i) for i in range(10)])

    projects = await storage.list_projects()
    assert len(projects) == 10


@pytest.mark.asyncio
async def test_corrupted_file_recovery(temp_storage_path):
    """Test recovery from corrupted storage file."""
    # Write corrupted JSON
    with open(temp_storage_path, "w") as f:
        f.write("{invalid json")

    storage = ProjectStorage(
        data_dir=temp_storage_path.parent, storage_path=temp_storage_path
    )
    await storage.initialize()

    # Should recover with empty projects
    projects = await storage.load_projects()
    assert projects == {}

    # Backup should be created
    backup_path = temp_storage_path.with_suffix(".json.bak")
    assert backup_path.exists()


@pytest.mark.asyncio
async def test_update_nonexistent_project(storage):
    """Test updating a project that doesn't exist."""
    with pytest.raises(ProjectStorageError, match="Project nonexistent not found"):
        await storage.update_project("nonexistent", {"name": "New Name"})


@pytest.mark.asyncio
async def test_data_dir_creates_default_path():
    """Test that providing only data_dir creates projects.json in that directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "custom_data"
        storage = ProjectStorage(data_dir=data_dir)

        # Verify the storage path is set correctly
        assert storage.storage_path == data_dir / "projects.json"

        await storage.initialize()

        # Verify the directory and file were created
        assert data_dir.exists()
        assert storage.storage_path.exists()

        # Test saving data
        await storage.save_project("test-1", {"name": "Test Project"})
        projects = await storage.load_projects()
        assert "test-1" in projects
