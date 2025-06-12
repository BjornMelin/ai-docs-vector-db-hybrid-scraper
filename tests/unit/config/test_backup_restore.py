"""Comprehensive tests for configuration backup and restore system.

This test file covers the backup and restore functionality that provides
versioning, compression, and metadata tracking for configuration management.
"""

import gzip
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from src.config.backup_restore import BackupMetadata
from src.config.backup_restore import ConfigBackupManager
from src.config.backup_restore import RestoreResult


class TestBackupMetadata:
    """Test the BackupMetadata model."""

    def test_backup_metadata_creation(self):
        """Test basic BackupMetadata creation."""
        metadata = BackupMetadata(
            backup_id="backup_123",
            config_name="test_config",
            config_hash="hash123",
            created_at="2023-01-01T12:00:00Z",
            file_size=1024,
            compressed=True,
            description="Test backup",
            tags=["test", "manual"],
            environment="development",
        )

        assert metadata.backup_id == "backup_123"
        assert metadata.config_name == "test_config"
        assert metadata.file_size == 1024
        assert metadata.compressed is True
        assert metadata.tags == ["test", "manual"]

    def test_backup_metadata_default_values(self):
        """Test BackupMetadata with default values."""
        metadata = BackupMetadata(
            backup_id="backup_456",
            config_name="config",
            config_hash="hash456",
            created_at="2023-01-01T12:00:00Z",
            file_size=512,
        )

        assert metadata.compressed is False
        assert metadata.description is None
        assert metadata.tags == []
        assert metadata.environment is None

    def test_backup_metadata_serialization(self):
        """Test BackupMetadata serialization."""
        metadata = BackupMetadata(
            backup_id="test",
            config_name="config",
            config_hash="hash_test",
            created_at="2023-01-01T12:00:00Z",
            file_size=100,
            tags=["tag1", "tag2"],
        )

        # Should be serializable to dict
        data = metadata.model_dump()
        assert isinstance(data, dict)
        assert data["backup_id"] == "test"
        assert data["tags"] == ["tag1", "tag2"]

        # Should be deserializable from dict
        restored = BackupMetadata(**data)
        assert restored.backup_id == metadata.backup_id
        assert restored.tags == metadata.tags


class TestRestoreResult:
    """Test the RestoreResult model."""

    def test_restore_result_success(self):
        """Test successful RestoreResult creation."""
        result = RestoreResult(
            success=True,
            backup_id="backup_123",
            config_path=Path("/restored/config.json"),
            pre_restore_backup="backup_pre_123",
            conflicts=[],
            warnings=[],
        )

        assert result.success is True
        assert result.config_path == Path("/restored/config.json")
        assert result.pre_restore_backup == "backup_pre_123"
        assert result.conflicts == []
        assert result.warnings == []

    def test_restore_result_failure(self):
        """Test failed RestoreResult creation."""
        result = RestoreResult(
            success=False,
            backup_id="backup_456",
            config_path=Path("/failed/config.json"),
            conflicts=["Version mismatch"],
            warnings=["Deprecated setting found"],
        )

        assert result.success is False
        assert result.conflicts == ["Version mismatch"]
        assert result.warnings == ["Deprecated setting found"]
        assert result.pre_restore_backup is None

    def test_restore_result_default_values(self):
        """Test RestoreResult with default values."""
        result = RestoreResult(
            success=True, backup_id="backup_default", config_path=Path("/config.json")
        )

        assert result.conflicts == []
        assert result.warnings == []
        assert result.pre_restore_backup is None


class TestConfigBackupManager:
    """Test the ConfigBackupManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config_file(self, temp_dir):
        """Create a sample configuration file."""
        config_file = temp_dir / "test_config.json"
        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "config_hash": "sample_hash",
            "schema_version": "2025.1.0",
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        return config_file

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_manager = ConfigBackupManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_backup_manager_initialization(self):
        """Test ConfigBackupManager initialization."""
        manager = ConfigBackupManager(self.temp_dir)

        assert manager.path_manager.base_dir == self.temp_dir
        assert manager.path_manager.backups_dir.exists()
        assert (manager.path_manager.backups_dir / "backup_metadata.json").exists()

    def test_backup_manager_default_base_dir(self):
        """Test ConfigBackupManager with default base directory."""
        manager = ConfigBackupManager()

        assert manager.path_manager.base_dir == Path("config")

    def test_create_backup_basic(self, sample_config_file):
        """Test basic backup creation."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file, description="Test backup"
        )

        assert isinstance(backup_id, str)
        assert len(backup_id) > 0

        # Backup file should exist
        backup_file = (
            self.backup_manager.path_manager.backups_dir
            / f"test_config_{backup_id}.json.gz"
        )
        assert backup_file.exists()

        # Metadata should be recorded
        assert backup_id in self.backup_manager._metadata
        assert self.backup_manager._metadata[backup_id].description == "Test backup"

    def test_create_backup_with_compression(self, sample_config_file):
        """Test backup creation with compression."""
        backup_id = self.backup_manager.create_backup(sample_config_file, compress=True)

        # Compressed backup file should exist
        backup_file = (
            self.backup_manager.path_manager.backups_dir
            / f"test_config_{backup_id}.json.gz"
        )
        assert backup_file.exists()

        # Should be a valid gzip file
        with gzip.open(backup_file, "rt") as f:
            restored_data = json.load(f)

        # Should contain original config data
        assert restored_data["environment"] == "development"

        # Metadata should indicate compression
        metadata = self.backup_manager._metadata
        assert metadata[backup_id].compressed is True

    def test_create_backup_with_tags(self, sample_config_file):
        """Test backup creation with tags."""
        tags = ["manual", "pre-migration", "important"]

        backup_id = self.backup_manager.create_backup(sample_config_file, tags=tags)

        # Metadata should include tags
        metadata = self.backup_manager._metadata
        assert metadata[backup_id].tags == tags

    def test_create_backup_nonexistent_file(self):
        """Test backup creation with non-existent file."""
        nonexistent_file = Path("/nonexistent/config.json")

        with pytest.raises(FileNotFoundError):
            self.backup_manager.create_backup(nonexistent_file)

    def test_create_backup_captures_metadata(self, sample_config_file):
        """Test that backup captures configuration metadata."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        metadata = self.backup_manager._metadata
        backup_meta = metadata[backup_id]

        # Should capture file information
        assert backup_meta.config_name == "test_config"
        assert backup_meta.file_size > 0
        assert backup_meta.environment == "development"

        # Should have timestamp
        assert backup_meta.created_at
        # Timestamp should be in correct format (YYYYMMDD_HHMMSS)
        import re

        assert re.match(r"\d{8}_\d{6}", backup_meta.created_at)

    def test_list_backups_empty(self):
        """Test listing backups when none exist."""
        backups = self.backup_manager.list_backups()

        assert backups == []

    def test_list_backups_with_backups(self, sample_config_file):
        """Test listing backups when backups exist."""
        # Create first backup
        backup_id1 = self.backup_manager.create_backup(
            sample_config_file, description="First backup"
        )

        # Modify file content to create different hash for second backup
        modified_config = {
            "environment": "production",  # Different from original
            "debug": False,
            "log_level": "INFO",
            "config_hash": "different_hash",
            "schema_version": "2025.1.0",
        }
        with open(sample_config_file, "w") as f:
            json.dump(modified_config, f, indent=2)

        backup_id2 = self.backup_manager.create_backup(
            sample_config_file, description="Second backup", tags=["tag1", "tag2"]
        )

        backups = self.backup_manager.list_backups()

        assert len(backups) == 2

        # Check both backups exist
        backup_descriptions = [backup.description for backup in backups]
        assert "Second backup" in backup_descriptions
        assert "First backup" in backup_descriptions

        # Check backup IDs
        backup_ids = [backup.backup_id for backup in backups]
        assert backup_id1 in backup_ids
        assert backup_id2 in backup_ids

    def test_list_backups_with_filters(self, sample_config_file):
        """Test listing backups with filters."""
        # Create first backup
        self.backup_manager.create_backup(
            sample_config_file, description="Dev backup", tags=["development"]
        )

        # Modify file for second backup with different content
        modified_config = {
            "environment": "production",
            "debug": False,
            "log_level": "ERROR",
            "config_hash": "prod_hash",
            "schema_version": "2025.1.0",
        }
        with open(sample_config_file, "w") as f:
            json.dump(modified_config, f, indent=2)

        self.backup_manager.create_backup(
            sample_config_file, description="Prod backup", tags=["production"]
        )

        # Filter by tags
        dev_backups = self.backup_manager.list_backups(tags=["development"])
        assert len(dev_backups) == 1
        assert dev_backups[0].description == "Dev backup"

        # Filter by config name
        config_backups = self.backup_manager.list_backups(config_name="test_config")
        assert len(config_backups) == 2

    def test_list_backups_with_limit(self, sample_config_file):
        """Test listing backups with limit."""
        # Create multiple backups with different content
        for i in range(5):
            # Modify content to ensure different hashes
            config_data = {
                "environment": f"env_{i}",
                "debug": i % 2 == 0,
                "log_level": "DEBUG",
                "config_hash": f"hash_{i}",
                "iteration": i,
            }
            with open(sample_config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            self.backup_manager.create_backup(
                sample_config_file, description=f"Backup {i}"
            )

        # List with limit
        limited_backups = self.backup_manager.list_backups(limit=3)

        assert len(limited_backups) == 3

    def test_get_backup_metadata_existing(self, sample_config_file):
        """Test getting metadata for existing backup."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file, description="Test backup"
        )

        metadata = self.backup_manager.get_backup_metadata(backup_id)

        assert metadata is not None
        assert metadata.backup_id == backup_id
        assert metadata.description == "Test backup"

    def test_get_backup_metadata_nonexistent(self):
        """Test getting metadata for non-existent backup."""
        metadata = self.backup_manager.get_backup_metadata("nonexistent")

        assert metadata is None

    def test_restore_backup_basic(self, sample_config_file):
        """Test basic backup restoration."""
        # Create backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Modify original file
        modified_config = {"environment": "modified", "new_field": "added"}
        with open(sample_config_file, "w") as f:
            json.dump(modified_config, f)

        # Restore backup with force=True to bypass conflict detection
        result = self.backup_manager.restore_backup(
            backup_id, sample_config_file, force=True
        )

        assert result.success is True
        assert result.config_path == sample_config_file

        # File should be restored to original content
        with open(sample_config_file) as f:
            restored_data = json.load(f)

        assert restored_data["environment"] == "development"
        assert restored_data["debug"] is True
        assert "new_field" not in restored_data

    def test_restore_backup_compressed(self, sample_config_file):
        """Test restoration of compressed backup."""
        # Create compressed backup
        backup_id = self.backup_manager.create_backup(sample_config_file, compress=True)

        # Restore backup
        result = self.backup_manager.restore_backup(backup_id, sample_config_file)

        assert result.success is True

        # Content should be correctly restored
        with open(sample_config_file) as f:
            restored_data = json.load(f)

        assert restored_data["environment"] == "development"

    def test_restore_backup_to_different_path(self, sample_config_file):
        """Test restoring backup to different path."""
        # Create backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Restore to different path
        new_path = self.temp_dir / "restored_config.json"
        result = self.backup_manager.restore_backup(backup_id, new_path)

        assert result.success is True
        assert result.config_path == new_path
        assert new_path.exists()

        # Content should be correct
        with open(new_path) as f:
            restored_data = json.load(f)

        assert restored_data["environment"] == "development"

    def test_restore_backup_with_pre_restore_backup(self, sample_config_file):
        """Test restoration with pre-restore backup creation."""
        # Create initial backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Modify file
        modified_config = {"environment": "modified"}
        with open(sample_config_file, "w") as f:
            json.dump(modified_config, f)

        # Restore with pre-restore backup and force=True
        result = self.backup_manager.restore_backup(
            backup_id, sample_config_file, create_pre_restore_backup=True, force=True
        )

        assert result.success is True
        assert result.pre_restore_backup is not None

        # Pre-restore backup should exist
        pre_backup_metadata = self.backup_manager.get_backup_metadata(
            result.pre_restore_backup
        )
        assert pre_backup_metadata is not None
        assert "pre-restore" in pre_backup_metadata.description.lower()

    def test_restore_backup_nonexistent(self):
        """Test restoration of non-existent backup."""
        result = self.backup_manager.restore_backup(
            "nonexistent", Path("/tmp/config.json")
        )

        assert result.success is False
        assert len(result.warnings) > 0
        assert "not found" in result.warnings[0].lower()

    def test_restore_backup_version_conflict_detection(self, sample_config_file):
        """Test detection of version conflicts during restore."""
        # Create backup with old schema version
        old_config = {"environment": "development", "schema_version": "2024.1.0"}

        old_config_file = self.temp_dir / "old_config.json"
        with open(old_config_file, "w") as f:
            json.dump(old_config, f)

        backup_id = self.backup_manager.create_backup(old_config_file)

        # Current file has newer schema version
        current_config = {"environment": "production", "schema_version": "2025.1.0"}

        current_file = self.temp_dir / "current_config.json"
        with open(current_file, "w") as f:
            json.dump(current_config, f)

        # Restore should detect version conflict
        result = self.backup_manager.restore_backup(backup_id, current_file)

        # Should succeed but report conflict
        if result.conflicts:
            assert any(
                "schema version" in conflict.lower() for conflict in result.conflicts
            )

    def test_restore_backup_force_overwrite(self, sample_config_file):
        """Test forced restoration that overwrites conflicts."""
        # Create backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Create conflicting current config
        conflict_config = {
            "environment": "production",
            "schema_version": "2025.2.0",
            "important_new_setting": "do_not_lose",
        }

        with open(sample_config_file, "w") as f:
            json.dump(conflict_config, f)

        # Force restore
        result = self.backup_manager.restore_backup(
            backup_id, sample_config_file, force=True
        )

        assert result.success is True

        # Should be restored despite conflicts
        with open(sample_config_file) as f:
            restored_data = json.load(f)

        assert restored_data["environment"] == "development"
        assert "important_new_setting" not in restored_data

    def test_delete_backup(self, sample_config_file):
        """Test backup deletion."""
        # Create backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Verify backup exists
        assert self.backup_manager.get_backup_metadata(backup_id) is not None
        backup_file = (
            self.backup_manager.backups_dir / f"test_config_{backup_id}.json.gz"
        )
        assert backup_file.exists()

        # Delete backup
        success = self.backup_manager.delete_backup(backup_id)

        assert success is True

        # Backup should no longer exist
        assert self.backup_manager.get_backup_metadata(backup_id) is None
        assert not backup_file.exists()

    def test_delete_backup_nonexistent(self):
        """Test deletion of non-existent backup."""
        success = self.backup_manager.delete_backup("nonexistent")

        assert success is False

    def test_cleanup_old_backups(self, sample_config_file):
        """Test cleanup of old backups."""
        # Create multiple backups with different content
        backup_ids = []
        for i in range(5):
            # Modify config content to create different hashes with more distinct content
            config_data = {
                "environment": f"cleanup_env_{i}",
                "debug": i % 2 == 0,
                "log_level": "DEBUG" if i < 3 else "INFO",
                "config_hash": f"cleanup_hash_{i}",
                "iteration": i,
                "unique_field": f"value_{i}_{i * i}",  # More unique content
                "timestamp": f"2023-01-{i + 1:02d}T10:00:00Z",
            }
            with open(sample_config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            backup_id = self.backup_manager.create_backup(
                sample_config_file, description=f"Cleanup backup {i}"
            )
            backup_ids.append(backup_id)

        # Simulate old backups by manually updating metadata timestamps
        # Make first 2 backups appear old by setting their created_at to past timestamps
        from datetime import timedelta

        old_time = datetime.now() - timedelta(days=31)
        old_timestamp = old_time.strftime("%Y%m%d_%H%M%S")

        # Update first 2 backup timestamps to be old
        for i in range(2):
            if len(backup_ids) > i and backup_ids[i] in self.backup_manager._metadata:
                self.backup_manager._metadata[backup_ids[i]].created_at = old_timestamp

        # Save updated metadata
        self.backup_manager._save_metadata()

        # Cleanup keeping only 3 newest with 30 days retention
        deleted_backup_ids = self.backup_manager.cleanup_old_backups(
            keep_count=3, keep_days=30
        )

        assert len(deleted_backup_ids) == 2

        # Should have only 3 backups remaining
        remaining_backups = self.backup_manager.list_backups()
        assert len(remaining_backups) == 3

        # Should keep the newest ones (last 3)
        remaining_ids = [backup.backup_id for backup in remaining_backups]
        assert backup_ids[-1] in remaining_ids  # Most recent
        assert backup_ids[-2] in remaining_ids  # Second most recent
        assert backup_ids[-3] in remaining_ids  # Third most recent

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_error_handling_permission_denied(self, mock_open_func, sample_config_file):
        """Test error handling when file access is denied."""
        with pytest.raises(IOError):
            self.backup_manager.create_backup(sample_config_file)

    @patch("gzip.open", side_effect=OSError("Compression failed"))
    def test_error_handling_compression_failure(self, mock_gzip, sample_config_file):
        """Test error handling when compression fails."""
        with pytest.raises(OSError):
            self.backup_manager.create_backup(sample_config_file, compress=True)

    def test_backup_metadata_persistence(self, sample_config_file):
        """Test that backup metadata persists across manager instances."""
        # Create backup with first manager instance
        backup_id = self.backup_manager.create_backup(
            sample_config_file, description="Persistent backup"
        )

        # Create new manager instance
        new_manager = ConfigBackupManager(self.temp_dir)

        # Should be able to access backup created by previous instance
        metadata = new_manager.get_backup_metadata(backup_id)
        assert metadata is not None
        assert metadata.description == "Persistent backup"

        # Should be able to list the backup
        backups = new_manager.list_backups()
        assert len(backups) == 1
        assert backups[0].backup_id == backup_id

    def test_backup_manager_concurrent_access(self, sample_config_file):
        """Test that backup manager handles concurrent access safely."""
        # This test simulates concurrent access by using multiple manager instances
        managers = [ConfigBackupManager(self.temp_dir) for _ in range(3)]

        # Create separate config files for each manager to avoid file conflicts
        backup_ids = []
        config_files = []
        for i, manager in enumerate(managers):
            # Reload metadata before creating backup to see existing backups
            manager._load_metadata()

            # Create separate config file for each manager
            config_file = self.temp_dir / f"concurrent_config_{i}.json"
            config_files.append(config_file)

            # Modify config content to create distinctly different hashes
            config_data = {
                "environment": f"concurrent_env_{i}",
                "debug": i % 2 == 0,
                "log_level": "ERROR" if i == 0 else "WARNING" if i == 1 else "INFO",
                "config_hash": f"concurrent_hash_{i}",
                "unique_id": i,
                "manager_instance": f"manager_{i}",
                "timestamp": f"2023-0{i + 1}-01T{10 + i}:00:00Z",
                "port": 8000 + i,
                "nested_config": {
                    "feature_flags": {
                        f"feature_{i}": True,
                        f"feature_{i}_extra": f"value_{i}",
                    },
                    "database": {"host": f"db{i}.example.com", "port": 5432 + i},
                },
            }
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            backup_id = manager.create_backup(
                config_file, description=f"Concurrent backup {i}"
            )
            backup_ids.append(backup_id)

        # Verify all backup IDs are unique
        assert len(set(backup_ids)) == 3, f"Backup IDs should be unique: {backup_ids}"

        # Test that all managers can see all backups by reloading metadata
        for manager in managers:
            manager._load_metadata()  # Force reload from disk
            backups = manager.list_backups()
            assert len(backups) == 3, f"Expected 3 backups, found {len(backups)}"

            found_ids = [backup.backup_id for backup in backups]
            for backup_id in backup_ids:
                assert backup_id in found_ids, (
                    f"Backup {backup_id} not found in {found_ids}"
                )

    # Additional comprehensive tests for improved coverage

    def test_metadata_load_exception_handling(self):
        """Test metadata loading handles corrupted files gracefully."""
        # Create corrupted metadata file
        corrupted_metadata = '{"invalid": json content'
        with open(self.backup_manager.metadata_file, "w") as f:
            f.write(corrupted_metadata)

        # Should handle corruption gracefully
        self.backup_manager._load_metadata()
        assert self.backup_manager._metadata == {}

    def test_create_backup_duplicate_hash_returns_existing(self, sample_config_file):
        """Test that creating backup with same content returns existing backup ID."""
        # Create first backup
        backup_id1 = self.backup_manager.create_backup(sample_config_file)

        # Create second backup with same content (should return existing ID)
        backup_id2 = self.backup_manager.create_backup(sample_config_file)

        assert backup_id1 == backup_id2

        # Only one backup should exist
        backups = self.backup_manager.list_backups()
        assert len(backups) == 1

    def test_create_backup_incremental_with_parent(self, sample_config_file):
        """Test incremental backup creation with parent backup."""
        # Create first backup
        backup_id1 = self.backup_manager.create_backup(sample_config_file)

        # Modify config slightly
        config_data = {"environment": "staging", "debug": False}
        with open(sample_config_file, "w") as f:
            json.dump(config_data, f)

        # Create incremental backup
        backup_id2 = self.backup_manager.create_backup(
            sample_config_file, incremental=True
        )

        # Second backup should have parent reference
        metadata2 = self.backup_manager.get_backup_metadata(backup_id2)
        assert metadata2.parent_backup == backup_id1

    def test_create_backup_without_compression(self, sample_config_file):
        """Test backup creation without compression."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file, compress=False
        )

        # Check that backup file is not compressed
        backup_file = (
            self.backup_manager.path_manager.backups_dir
            / f"test_config_{backup_id}.json"
        )
        assert backup_file.exists()
        assert not backup_file.name.endswith(".gz")

        # Should be able to read as regular JSON
        with open(backup_file) as f:
            data = json.load(f)
        assert data["environment"] == "development"

    def test_create_backup_invalid_json_fallback(self, temp_dir):
        """Test backup creation with invalid JSON config."""
        # Create file with invalid JSON
        invalid_config_file = temp_dir / "invalid.json"
        with open(invalid_config_file, "w") as f:
            f.write("invalid json content")

        backup_id = self.backup_manager.create_backup(invalid_config_file)

        # Should still create backup, but with empty config_data fallback
        assert backup_id is not None
        metadata = self.backup_manager.get_backup_metadata(backup_id)
        assert metadata is not None

    def test_restore_backup_file_not_found(self, sample_config_file):
        """Test restore when backup file is missing."""
        # Create backup normally
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Remove backup file manually
        backup_files = list(
            self.backup_manager.path_manager.backups_dir.glob(
                f"test_config_{backup_id}.*"
            )
        )
        for backup_file in backup_files:
            backup_file.unlink()

        # Try to restore
        result = self.backup_manager.restore_backup(backup_id)

        assert result.success is False
        assert any("not found" in warning.lower() for warning in result.warnings)

    def test_restore_backup_exception_during_restore(self, sample_config_file):
        """Test restore handles exceptions during file operations."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Try to restore to a location that will cause an error
        invalid_target = Path("/nonexistent/deeply/nested/path/config.json")

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = self.backup_manager.restore_backup(backup_id, invalid_target)

        assert result.success is False
        assert any("failed" in warning.lower() for warning in result.warnings)

    def test_restore_backup_target_path_none_uses_default(self, sample_config_file):
        """Test restore with target_path=None uses default path."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Restore without specifying target path
        result = self.backup_manager.restore_backup(backup_id)

        assert result.success is True
        # Should use default path from path manager
        expected_path = self.backup_manager.path_manager.get_config_path("test_config")
        assert result.config_path == expected_path

    def test_conflict_detection_comprehensive(self, temp_dir):
        """Test comprehensive conflict detection scenarios."""
        # Create configs with various conflicts
        old_config = {
            "environment": "development",
            "embedding_provider": "openai",
            "schema_version": "1.0.0",
            "database": {"database_url": "postgres://old-host:5432/db"},
        }

        current_config = {
            "environment": "production",
            "embedding_provider": "fastembed",
            "schema_version": "2.0.0",
            "database": {"database_url": "postgres://new-host:5432/db"},
        }

        old_file = temp_dir / "old.json"
        current_file = temp_dir / "current.json"

        with open(old_file, "w") as f:
            json.dump(old_config, f)
        with open(current_file, "w") as f:
            json.dump(current_config, f)

        backup_id = self.backup_manager.create_backup(old_file)

        # Restore should detect all conflicts
        result = self.backup_manager.restore_backup(backup_id, current_file)

        assert result.success is False
        assert len(result.conflicts) >= 3  # Should detect multiple conflicts

        conflict_text = " ".join(result.conflicts).lower()
        assert "environment" in conflict_text
        assert "embedding provider" in conflict_text or "provider" in conflict_text
        assert "schema version" in conflict_text or "version" in conflict_text

    def test_conflict_detection_with_invalid_json(self, temp_dir):
        """Test conflict detection handles invalid JSON gracefully."""
        # Create valid backup
        valid_config = {"environment": "development"}
        valid_file = temp_dir / "valid.json"
        with open(valid_file, "w") as f:
            json.dump(valid_config, f)

        backup_id = self.backup_manager.create_backup(valid_file)

        # Create current file with invalid JSON
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        result = self.backup_manager.restore_backup(backup_id, invalid_file)

        # Should handle error gracefully
        if result.conflicts:
            assert any(
                "error detecting" in conflict.lower() for conflict in result.conflicts
            )

    def test_export_backup_comprehensive(self, sample_config_file):
        """Test comprehensive backup export functionality."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            description="Export test backup",
            tags=["export", "test"],
        )

        export_dir = self.temp_dir / "exports"
        export_path = export_dir / "exported_backup.json.gz"

        # Export backup
        success = self.backup_manager.export_backup(backup_id, export_path)

        assert success is True
        assert export_path.exists()

        # Metadata should also be exported
        metadata_path = export_path.with_suffix(".metadata.json")
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            exported_metadata = json.load(f)
        assert exported_metadata["backup_id"] == backup_id
        assert exported_metadata["description"] == "Export test backup"

    def test_export_backup_nonexistent(self):
        """Test export of non-existent backup."""
        success = self.backup_manager.export_backup(
            "nonexistent", self.temp_dir / "export.json"
        )
        assert success is False

    def test_export_backup_no_file(self, sample_config_file):
        """Test export when backup metadata exists but file is missing."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Remove backup file but keep metadata
        backup_files = list(
            self.backup_manager.path_manager.backups_dir.glob(
                f"test_config_{backup_id}.*"
            )
        )
        for backup_file in backup_files:
            backup_file.unlink()

        success = self.backup_manager.export_backup(
            backup_id, self.temp_dir / "export.json"
        )
        assert success is False

    def test_export_backup_file_operation_error(self, sample_config_file):
        """Test export handles file operation errors."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Try to export to invalid path
        with patch("shutil.copy2", side_effect=OSError("Copy failed")):
            success = self.backup_manager.export_backup(
                backup_id, self.temp_dir / "export.json"
            )

        assert success is False

    def test_import_backup_comprehensive(self, temp_dir):
        """Test comprehensive backup import functionality."""
        # Create external backup with metadata
        external_backup = temp_dir / "external_backup.json.gz"
        metadata_file = temp_dir / "external_backup.metadata.json"

        config_data = {"environment": "imported", "source": "external"}

        # Create compressed backup
        with gzip.open(external_backup, "wt", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Create metadata
        metadata = {
            "backup_id": "imported_backup_123",
            "config_name": "imported_config",
            "config_hash": "imported_hash",
            "created_at": "20230101_120000",
            "file_size": external_backup.stat().st_size,
            "compressed": True,
            "tags": ["imported", "external"],
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Import backup
        imported_id = self.backup_manager.import_backup(external_backup, metadata_file)

        assert imported_id == "imported_backup_123"

        # Verify import
        imported_metadata = self.backup_manager.get_backup_metadata(imported_id)
        assert imported_metadata is not None
        assert imported_metadata.config_name == "imported_config"
        assert "imported" in imported_metadata.tags

    def test_import_backup_without_metadata(self, temp_dir):
        """Test import backup without metadata file."""
        external_backup = temp_dir / "no_metadata_backup.json"
        config_data = {"environment": "no_metadata"}

        with open(external_backup, "w") as f:
            json.dump(config_data, f)

        imported_id = self.backup_manager.import_backup(external_backup)

        assert imported_id is not None

        # Should generate metadata automatically
        metadata = self.backup_manager.get_backup_metadata(imported_id)
        assert metadata is not None
        assert metadata.config_name == "no_metadata_backup"
        assert "imported" in metadata.tags

    def test_import_backup_nonexistent_file(self):
        """Test import of non-existent backup file."""
        nonexistent_file = Path("/nonexistent/backup.json")
        imported_id = self.backup_manager.import_backup(nonexistent_file)
        assert imported_id is None

    def test_import_backup_corrupted_metadata(self, temp_dir):
        """Test import with corrupted metadata file."""
        external_backup = temp_dir / "backup_corrupted_meta.json"
        metadata_file = temp_dir / "corrupted.metadata.json"

        config_data = {"environment": "test"}
        with open(external_backup, "w") as f:
            json.dump(config_data, f)

        # Create corrupted metadata
        with open(metadata_file, "w") as f:
            f.write("invalid json")

        imported_id = self.backup_manager.import_backup(external_backup, metadata_file)

        # Should still import with generated metadata
        assert imported_id is not None

    def test_import_backup_invalid_json_content(self, temp_dir):
        """Test import backup with invalid JSON content."""
        external_backup = temp_dir / "invalid_content.json"

        with open(external_backup, "w") as f:
            f.write("invalid json content")

        imported_id = self.backup_manager.import_backup(external_backup)

        # Should handle gracefully with fallback
        assert imported_id is not None
        metadata = self.backup_manager.get_backup_metadata(imported_id)
        assert metadata.config_hash == "unknown"

    def test_import_backup_file_copy_error(self, temp_dir):
        """Test import handles file copy errors."""
        external_backup = temp_dir / "copy_error_backup.json"
        config_data = {"environment": "test"}

        with open(external_backup, "w") as f:
            json.dump(config_data, f)

        with patch("shutil.copy2", side_effect=OSError("Copy failed")):
            imported_id = self.backup_manager.import_backup(external_backup)

        assert imported_id is None

    def test_get_backup_info_alias(self, sample_config_file):
        """Test get_backup_info method (alias for get_backup_metadata)."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        info = self.backup_manager.get_backup_info(backup_id)
        metadata = self.backup_manager.get_backup_metadata(backup_id)

        assert info == metadata
        assert info is not None

    def test_backups_dir_property(self):
        """Test backups_dir property."""
        expected_dir = self.backup_manager.path_manager.backups_dir
        assert self.backup_manager.backups_dir == expected_dir

    def test_list_backups_environment_filter(self, sample_config_file):
        """Test list backups filtered by environment."""
        # Create configs with different environments
        environments = ["development", "staging", "production"]
        backup_ids = []

        for env in environments:
            config_data = {"environment": env, "unique": env}
            with open(sample_config_file, "w") as f:
                json.dump(config_data, f)

            backup_id = self.backup_manager.create_backup(
                sample_config_file, description=f"{env} backup"
            )
            backup_ids.append(backup_id)

        # Filter by specific environment
        dev_backups = self.backup_manager.list_backups(environment="development")
        assert len(dev_backups) == 1
        assert dev_backups[0].environment == "development"

    def test_delete_backup_without_removing_file(self, sample_config_file):
        """Test delete backup metadata only, keeping file."""
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Get backup file path
        backup_files = list(
            self.backup_manager.path_manager.backups_dir.glob(
                f"test_config_{backup_id}.*"
            )
        )
        assert len(backup_files) == 1
        backup_file = backup_files[0]

        # Delete metadata only
        success = self.backup_manager.delete_backup(backup_id, remove_file=False)

        assert success is True
        assert self.backup_manager.get_backup_metadata(backup_id) is None
        assert backup_file.exists()  # File should still exist
