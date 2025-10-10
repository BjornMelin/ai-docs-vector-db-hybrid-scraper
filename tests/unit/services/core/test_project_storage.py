"""Tests for project storage service."""

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.services.core.project_storage import ProjectStorage, ProjectStorageError


class TestProjectStorageError:
    """Tests for ProjectStorageError exception."""

    def test_project_storage_error_inheritance(self):
        """Test that ProjectStorageError inherits from BaseError."""
        error = ProjectStorageError("Test error")

        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_project_storage_error_with_context(self):
        """Test ProjectStorageError with error context."""
        context = {"file": "projects.json", "operation": "save"}
        error = ProjectStorageError("Save failed", context=context)

        assert str(error) == "Save failed"
        assert hasattr(error, "context")


class TestProjectStorage:
    """Tests for ProjectStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def storage_path(self, temp_dir):
        """Create storage path."""
        return temp_dir / "projects.json"

    @pytest.fixture
    def project_storage(self, temp_dir):
        """Create ProjectStorage instance."""
        return ProjectStorage(data_dir=temp_dir)

    @pytest.fixture
    def project_storage_custom_path(self, storage_path):
        """Create ProjectStorage with custom storage path."""
        return ProjectStorage(
            data_dir="/tmp", storage_path=storage_path
        )  # test temp path

    def test_init_with_data_dir_only(self, temp_dir):
        """Test initialization with data_dir only."""
        storage = ProjectStorage(data_dir=temp_dir)

        assert storage.storage_path == temp_dir / "projects.json"
        assert hasattr(storage, "_lock")
        assert storage._projects_cache == {}
        assert storage._initialized is False

    def test_init_with_custom_storage_path(self, temp_dir, storage_path):
        """Test initialization with custom storage path."""
        storage = ProjectStorage(data_dir=temp_dir, storage_path=storage_path)

        assert storage.storage_path == storage_path
        assert hasattr(storage, "_lock")
        assert storage._projects_cache == {}
        assert storage._initialized is False

    def test_init_creates_data_directory(self):
        """Test that initialization creates data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "nonexistent" / "data"
            storage = ProjectStorage(data_dir=data_dir)

            assert data_dir.exists()
            assert storage.storage_path == data_dir / "projects.json"

    @pytest.mark.asyncio
    async def test_initialize_creates_storage_file(self, project_storage):
        """Test initialization creates storage file if it doesn't exist."""
        assert not project_storage.storage_path.exists()

        await project_storage.initialize()

        assert project_storage.storage_path.exists()
        assert project_storage._initialized is True

        # Verify file contains empty JSON object
        with project_storage.storage_path.open() as f:
            data = json.load(f)
            assert data == {}

    @pytest.mark.asyncio
    async def test_initialize_loads_existing_projects(self, project_storage):
        """Test initialization loads existing projects from file."""
        # Create file with existing data
        test_data = {
            "project1": {"name": "Test Project", "created_at": "2023-01-01"},
            "project2": {"name": "Another Project", "created_at": "2023-01-02"},
        }

        project_storage.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with project_storage.storage_path.open("w") as f:
            json.dump(test_data, f)

        await project_storage.initialize()

        assert project_storage._initialized is True
        assert project_storage._projects_cache == test_data

    @pytest.mark.asyncio
    async def test_initialize_handles_corrupted_json(self, project_storage):
        """Test initialization handles corrupted JSON file."""
        # Create corrupted JSON file
        project_storage.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with project_storage.storage_path.open("w") as f:
            f.write("{ invalid json }")

        await project_storage.initialize()

        assert project_storage._initialized is True
        assert project_storage._projects_cache == {}

        # Check backup file was created
        backup_path = project_storage.storage_path.with_suffix(".json.bak")
        assert backup_path.exists()

    @pytest.mark.asyncio
    async def test_initialize_failure_raises_error(self, project_storage):
        """Test initialization failure raises ProjectStorageError."""
        # Mock _save_projects to raise an exception during file creation
        with patch.object(
            project_storage,
            "_save_projects",
            side_effect=PermissionError("Access denied"),
        ):
            with pytest.raises(ProjectStorageError) as exc_info:
                await project_storage.initialize()

            assert "Failed to initialize project storage" in str(exc_info.value)
            assert project_storage._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, project_storage):
        """Test that multiple initialization calls are safe."""
        await project_storage.initialize()
        first_cache = project_storage._projects_cache.copy()

        # Second initialization should not change anything
        await project_storage.initialize()

        assert project_storage._projects_cache == first_cache
        assert project_storage._initialized is True

    @patch("src.services.core.project_storage.aiofiles", None)  # Test without aiofiles
    @pytest.mark.asyncio
    async def test_load_projects_without_aiofiles(self, project_storage):
        """Test loading projects without aiofiles (fallback mode)."""
        test_data = {"project1": {"name": "Test"}}

        project_storage.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with project_storage.storage_path.open("w") as f:
            json.dump(test_data, f)

        loaded_data = await project_storage.load_projects()

        assert loaded_data == test_data
        assert project_storage._projects_cache == test_data

    @pytest.mark.asyncio
    async def test_load_projects_with_aiofiles(self, project_storage):
        """Test loading projects with aiofiles."""
        test_data = {"project1": {"name": "Test"}}

        mock_file = AsyncMock()
        mock_file.read.return_value = json.dumps(test_data)

        with patch("src.services.core.project_storage.aiofiles") as mock_aiofiles:
            mock_aiofiles.open.return_value.__aenter__.return_value = mock_file

            project_storage.storage_path.parent.mkdir(parents=True, exist_ok=True)
            loaded_data = await project_storage.load_projects()

        assert loaded_data == test_data
        assert project_storage._projects_cache == test_data

    @pytest.mark.asyncio
    async def test_load_projects_file_not_found(self, project_storage):
        """Test loading projects when file doesn't exist."""
        loaded_data = await project_storage.load_projects()

        assert loaded_data == {}
        assert project_storage._projects_cache == {}

    @pytest.mark.asyncio
    async def test_load_projects_empty_file(self, project_storage):
        """Test loading projects from empty file."""
        project_storage.storage_path.parent.mkdir(parents=True, exist_ok=True)
        project_storage.storage_path.touch()  # Create empty file

        loaded_data = await project_storage.load_projects()

        assert loaded_data == {}
        assert project_storage._projects_cache == {}

    @pytest.mark.asyncio
    async def test_save_project(self, project_storage):
        """Test saving a single project."""
        await project_storage.initialize()

        project_data = {
            "name": "Test Project",
            "description": "A test project",
            "created_at": "2023-01-01T00:00:00",
        }

        await project_storage.save_project("test_project", project_data)

        # Verify project is in cache
        assert "test_project" in project_storage._projects_cache
        assert project_storage._projects_cache["test_project"] == project_data

        # Verify project is saved to file
        with project_storage.storage_path.open() as f:
            file_data = json.load(f)
            assert file_data["test_project"] == project_data

    @pytest.mark.asyncio
    async def test_get_project_exists(self, project_storage):
        """Test getting an existing project."""
        await project_storage.initialize()

        project_data = {"name": "Test Project"}
        await project_storage.save_project("test_project", project_data)

        retrieved_data = await project_storage.get_project("test_project")

        assert retrieved_data == project_data

    @pytest.mark.asyncio
    async def test_get_project_not_exists(self, project_storage):
        """Test getting a non-existent project."""
        await project_storage.initialize()

        retrieved_data = await project_storage.get_project("nonexistent")

        assert retrieved_data is None

    @pytest.mark.asyncio
    async def test_list_projects(self, project_storage):
        """Test listing all projects."""
        await project_storage.initialize()

        project1 = {"name": "Project 1"}
        project2 = {"name": "Project 2"}

        await project_storage.save_project("project1", project1)
        await project_storage.save_project("project2", project2)

        projects = await project_storage.list_projects()

        assert len(projects) == 2
        assert project1 in projects
        assert project2 in projects

    @pytest.mark.asyncio
    async def test_list_projects_empty(self, project_storage):
        """Test listing projects when none exist."""
        await project_storage.initialize()

        projects = await project_storage.list_projects()

        assert projects == []

    @pytest.mark.asyncio
    async def test_update_project_exists(self, project_storage):
        """Test updating an existing project."""
        await project_storage.initialize()

        original_data = {"name": "Original Name", "version": 1}
        await project_storage.save_project("test_project", original_data)

        updates = {"name": "Updated Name", "version": 2}
        await project_storage.update_project("test_project", updates)

        updated_data = await project_storage.get_project("test_project")
        assert updated_data["name"] == "Updated Name"
        assert updated_data["version"] == 2
        assert "updated_at" in updated_data

    @pytest.mark.asyncio
    async def test_update_project_not_exists(self, project_storage):
        """Test updating a non-existent project raises error."""
        await project_storage.initialize()

        updates = {"name": "Updated Name"}

        with pytest.raises(ProjectStorageError) as exc_info:
            await project_storage.update_project("nonexistent", updates)

        assert "Project nonexistent not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_project_adds_timestamp(self, project_storage):
        """Test that update_project adds updated_at timestamp."""
        await project_storage.initialize()

        original_data = {"name": "Test Project"}
        await project_storage.save_project("test_project", original_data)

        # Mock datetime to control timestamp
        with patch("src.services.core.project_storage.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            await project_storage.update_project("test_project", {"version": 2})

        updated_data = await project_storage.get_project("test_project")
        assert updated_data["updated_at"] == "2023-01-01T12:00:00"

    @pytest.mark.asyncio
    async def test_delete_project_exists(self, project_storage):
        """Test deleting an existing project."""
        await project_storage.initialize()

        project_data = {"name": "Test Project"}
        await project_storage.save_project("test_project", project_data)

        await project_storage.delete_project("test_project")

        # Verify project is removed from cache
        assert "test_project" not in project_storage._projects_cache

        # Verify project is removed from file
        with project_storage.storage_path.open() as f:
            file_data = json.load(f)
            assert "test_project" not in file_data

    @pytest.mark.asyncio
    async def test_delete_project_not_exists(self, project_storage):
        """Test deleting a non-existent project (should not raise error)."""
        await project_storage.initialize()

        # Should not raise an error
        await project_storage.delete_project("nonexistent")

        assert "nonexistent" not in project_storage._projects_cache

    @patch("src.services.core.project_storage.aiofiles", None)  # Test without aiofiles
    @pytest.mark.asyncio
    async def test_save_projects_without_aiofiles(self, project_storage):
        """Test saving projects without aiofiles (fallback mode)."""
        await project_storage.initialize()

        test_data = {"project1": {"name": "Test"}}

        await project_storage._save_projects(test_data)

        # Verify file was created
        assert project_storage.storage_path.exists()

        # Verify content
        with project_storage.storage_path.open() as f:
            file_data = json.load(f)
            assert file_data == test_data

    @pytest.mark.asyncio
    async def test_save_projects_with_aiofiles(self, project_storage):
        """Test saving projects with aiofiles."""
        await project_storage.initialize()

        test_data = {"project1": {"name": "Test"}}
        mock_file = AsyncMock()

        with patch("src.services.core.project_storage.aiofiles") as mock_aiofiles:
            mock_aiofiles.open.return_value.__aenter__.return_value = mock_file
            # Mock the Path.replace method to avoid file system issues
            with patch.object(Path, "replace") as mock_replace:
                await project_storage._save_projects(test_data)

        mock_file.write.assert_called_once()
        written_data = json.loads(mock_file.write.call_args[0][0])
        assert written_data == test_data
        mock_replace.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_projects_atomic_write(self, project_storage):
        """Test that _save_projects uses atomic write (temp file then rename)."""
        await project_storage.initialize()

        test_data = {"project1": {"name": "Test"}}

        with patch("pathlib.Path.replace") as mock_replace:
            await project_storage._save_projects(test_data)

        # Verify atomic operation
        mock_replace.assert_called_once_with(project_storage.storage_path)

    @pytest.mark.asyncio
    async def test_save_projects_failure_raises_error(self, project_storage):
        """Test that _save_projects raises error on failure."""
        await project_storage.initialize()

        test_data = {"project1": {"name": "Test"}}

        # Mock json.dumps to fail
        with patch("json.dumps", side_effect=TypeError("Cannot serialize")):
            with pytest.raises(ProjectStorageError) as exc_info:
                await project_storage._save_projects(test_data)

            assert "Failed to save projects" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self, project_storage):
        """Test cleanup method."""
        await project_storage.initialize()

        # Add some data
        await project_storage.save_project("test", {"name": "Test"})

        await project_storage.cleanup()

        assert project_storage._projects_cache == {}
        assert project_storage._initialized is False

    @pytest.mark.asyncio
    async def test_concurrent_access_protection(self, project_storage):
        """Test that concurrent operations are protected by lock."""
        await project_storage.initialize()

        async def save_project(project_id, data):
            await project_storage.save_project(project_id, data)

        # Start multiple concurrent save operations
        tasks = [
            save_project(f"project_{i}", {"name": f"Project {i}"}) for i in range(10)
        ]

        await asyncio.gather(*tasks)

        # Verify all projects were saved
        projects = await project_storage.list_projects()
        assert len(projects) == 10

        # Verify cache consistency
        with project_storage.storage_path.open() as f:
            file_data = json.load(f)
            assert len(file_data) == 10

    @pytest.mark.asyncio
    async def test_json_serialization_with_datetime(self, project_storage):
        """Test JSON serialization handles datetime objects."""
        await project_storage.initialize()

        # Project data with datetime-like string (common case)
        project_data = {
            "name": "Test Project",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "custom_field": "value",
        }

        await project_storage.save_project("test_project", project_data)

        # Verify data was saved and can be loaded
        loaded_data = await project_storage.get_project("test_project")
        assert loaded_data == project_data

    @pytest.mark.asyncio
    async def test_large_project_data(self, project_storage):
        """Test handling of large project data."""
        await project_storage.initialize()

        # Create large project data
        large_data = {
            "name": "Large Project",
            "data": ["item_" + str(i) for i in range(1000)],
            "metadata": {f"key_{i}": f"value_{i}" for i in range(100)},
        }

        await project_storage.save_project("large_project", large_data)

        # Verify data integrity
        loaded_data = await project_storage.get_project("large_project")
        assert loaded_data == large_data
        assert len(loaded_data["data"]) == 1000
        assert len(loaded_data["metadata"]) == 100

    @pytest.mark.asyncio
    async def test_special_characters_in_project_data(self, project_storage):
        """Test handling of special characters in project data."""
        await project_storage.initialize()

        project_data = {
            "name": "Special Characters Test",
            "description": (
                "Contains special chars: àáâãäåæçèéêë 中文 🚀 \"quotes\" 'apostrophes'"
            ),
            "unicode_field": "тест на кириллице",
            "emoji": "😀🎉🔥",
        }

        await project_storage.save_project("special_project", project_data)

        # Verify data integrity
        loaded_data = await project_storage.get_project("special_project")
        assert loaded_data == project_data

    @pytest.mark.asyncio
    async def test_error_handling_during_load(self, project_storage):
        """Test error handling during project loading."""
        # Test with permission denied
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            loaded_data = await project_storage.load_projects()
            assert loaded_data == {}

    @pytest.mark.asyncio
    async def test_file_locking_behavior(self, project_storage):
        """Test that file operations respect async locks."""
        await project_storage.initialize()

        # Track lock acquisition order
        lock_order = []
        original_acquire = project_storage._lock.acquire

        async def tracked_acquire():
            result = await original_acquire()
            lock_order.append(len(lock_order))
            return result

        project_storage._lock.acquire = tracked_acquire

        # Start concurrent operations
        tasks = [
            project_storage.save_project(f"project_{i}", {"name": f"Project {i}"})
            for i in range(3)
        ]

        await asyncio.gather(*tasks)

        # Verify operations were serialized (lock was acquired multiple times)
        assert len(lock_order) >= 3
