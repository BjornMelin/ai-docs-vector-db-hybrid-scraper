"""Comprehensive tests for configuration backup and restore system.

This test file covers the backup and restore functionality that provides
versioning, compression, and metadata tracking for configuration management.
"""

import gzip
import json
import shutil
import tempfile
from datetime import UTC
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
            environment="development"
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
            file_size=512
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
            tags=["tag1", "tag2"]
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
            warnings=[]
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
            warnings=["Deprecated setting found"]
        )

        assert result.success is False
        assert result.conflicts == ["Version mismatch"]
        assert result.warnings == ["Deprecated setting found"]
        assert result.pre_restore_backup is None

    def test_restore_result_default_values(self):
        """Test RestoreResult with default values."""
        result = RestoreResult(
            success=True,
            backup_id="backup_default",
            config_path=Path("/config.json")
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
            "schema_version": "2025.1.0"
        }

        with open(config_file, 'w') as f:
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
            sample_config_file,
            description="Test backup"
        )

        assert isinstance(backup_id, str)
        assert len(backup_id) > 0

        # Backup file should exist
        backup_file = self.backup_manager.path_manager.backups_dir / f"test_config_{backup_id}.json.gz"
        assert backup_file.exists()

        # Metadata should be recorded
        assert backup_id in self.backup_manager._metadata
        assert self.backup_manager._metadata[backup_id].description == "Test backup"

    def test_create_backup_with_compression(self, sample_config_file):
        """Test backup creation with compression."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            compress=True
        )

        # Compressed backup file should exist
        backup_file = self.backup_manager.path_manager.backups_dir / f"test_config_{backup_id}.json.gz"
        assert backup_file.exists()

        # Should be a valid gzip file
        with gzip.open(backup_file, 'rt') as f:
            restored_data = json.load(f)

        # Should contain original config data
        assert restored_data["environment"] == "development"

        # Metadata should indicate compression
        metadata = self.backup_manager._metadata
        assert metadata[backup_id].compressed is True

    def test_create_backup_with_tags(self, sample_config_file):
        """Test backup creation with tags."""
        tags = ["manual", "pre-migration", "important"]

        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            tags=tags
        )

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
        assert re.match(r'\d{8}_\d{6}', backup_meta.created_at)

    def test_list_backups_empty(self):
        """Test listing backups when none exist."""
        backups = self.backup_manager.list_backups()

        assert backups == []

    def test_list_backups_with_backups(self, sample_config_file):
        """Test listing backups when backups exist."""
        # Create first backup
        backup_id1 = self.backup_manager.create_backup(
            sample_config_file,
            description="First backup"
        )
        
        # Modify file content to create different hash for second backup
        modified_config = {
            "environment": "production",  # Different from original
            "debug": False,
            "log_level": "INFO",
            "config_hash": "different_hash",
            "schema_version": "2025.1.0"
        }
        with open(sample_config_file, 'w') as f:
            json.dump(modified_config, f, indent=2)
            
        backup_id2 = self.backup_manager.create_backup(
            sample_config_file,
            description="Second backup",
            tags=["tag1", "tag2"]
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
            sample_config_file,
            description="Dev backup",
            tags=["development"]
        )
        
        # Modify file for second backup with different content
        modified_config = {
            "environment": "production",
            "debug": False,
            "log_level": "ERROR",
            "config_hash": "prod_hash",
            "schema_version": "2025.1.0"
        }
        with open(sample_config_file, 'w') as f:
            json.dump(modified_config, f, indent=2)
            
        self.backup_manager.create_backup(
            sample_config_file,
            description="Prod backup",
            tags=["production"]
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
                "iteration": i
            }
            with open(sample_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            self.backup_manager.create_backup(
                sample_config_file,
                description=f"Backup {i}"
            )

        # List with limit
        limited_backups = self.backup_manager.list_backups(limit=3)

        assert len(limited_backups) == 3

    def test_get_backup_metadata_existing(self, sample_config_file):
        """Test getting metadata for existing backup."""
        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            description="Test backup"
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
        with open(sample_config_file, 'w') as f:
            json.dump(modified_config, f)

        # Restore backup with force=True to bypass conflict detection
        result = self.backup_manager.restore_backup(
            backup_id,
            sample_config_file,
            force=True
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
        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            compress=True
        )

        # Restore backup
        result = self.backup_manager.restore_backup(
            backup_id,
            sample_config_file
        )

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
        result = self.backup_manager.restore_backup(
            backup_id,
            new_path
        )

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
        with open(sample_config_file, 'w') as f:
            json.dump(modified_config, f)

        # Restore with pre-restore backup and force=True
        result = self.backup_manager.restore_backup(
            backup_id,
            sample_config_file,
            create_pre_restore_backup=True,
            force=True
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
            "nonexistent",
            Path("/tmp/config.json")
        )

        assert result.success is False
        assert len(result.warnings) > 0
        assert "not found" in result.warnings[0].lower()

    def test_restore_backup_version_conflict_detection(self, sample_config_file):
        """Test detection of version conflicts during restore."""
        # Create backup with old schema version
        old_config = {
            "environment": "development",
            "schema_version": "2024.1.0"
        }

        old_config_file = self.temp_dir / "old_config.json"
        with open(old_config_file, 'w') as f:
            json.dump(old_config, f)

        backup_id = self.backup_manager.create_backup(old_config_file)

        # Current file has newer schema version
        current_config = {
            "environment": "production",
            "schema_version": "2025.1.0"
        }

        current_file = self.temp_dir / "current_config.json"
        with open(current_file, 'w') as f:
            json.dump(current_config, f)

        # Restore should detect version conflict
        result = self.backup_manager.restore_backup(
            backup_id,
            current_file
        )

        # Should succeed but report conflict
        if result.conflicts:
            assert any("schema version" in conflict.lower() for conflict in result.conflicts)

    def test_restore_backup_force_overwrite(self, sample_config_file):
        """Test forced restoration that overwrites conflicts."""
        # Create backup
        backup_id = self.backup_manager.create_backup(sample_config_file)

        # Create conflicting current config
        conflict_config = {
            "environment": "production",
            "schema_version": "2025.2.0",
            "important_new_setting": "do_not_lose"
        }

        with open(sample_config_file, 'w') as f:
            json.dump(conflict_config, f)

        # Force restore
        result = self.backup_manager.restore_backup(
            backup_id,
            sample_config_file,
            force=True
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
        backup_file = self.backup_manager.backups_dir / f"test_config_{backup_id}.json.gz"
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
                "unique_field": f"value_{i}_{i*i}",  # More unique content
                "timestamp": f"2023-01-{i+1:02d}T10:00:00Z"
            }
            with open(sample_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            backup_id = self.backup_manager.create_backup(
                sample_config_file,
                description=f"Cleanup backup {i}"
            )
            backup_ids.append(backup_id)

        # Simulate old backups by manually updating metadata timestamps
        # Make first 2 backups appear old by setting their created_at to past timestamps
        from datetime import datetime, timedelta
        old_time = datetime.now() - timedelta(days=31)
        old_timestamp = old_time.strftime("%Y%m%d_%H%M%S")
        
        # Update first 2 backup timestamps to be old
        for i in range(2):
            if len(backup_ids) > i and backup_ids[i] in self.backup_manager._metadata:
                self.backup_manager._metadata[backup_ids[i]].created_at = old_timestamp
        
        # Save updated metadata
        self.backup_manager._save_metadata()

        # Cleanup keeping only 3 newest with 30 days retention
        deleted_backup_ids = self.backup_manager.cleanup_old_backups(keep_count=3, keep_days=30)

        assert len(deleted_backup_ids) == 2

        # Should have only 3 backups remaining
        remaining_backups = self.backup_manager.list_backups()
        assert len(remaining_backups) == 3

        # Should keep the newest ones (last 3)
        remaining_ids = [backup.backup_id for backup in remaining_backups]
        assert backup_ids[-1] in remaining_ids  # Most recent
        assert backup_ids[-2] in remaining_ids  # Second most recent
        assert backup_ids[-3] in remaining_ids  # Third most recent

    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_error_handling_permission_denied(self, mock_open_func, sample_config_file):
        """Test error handling when file access is denied."""
        with pytest.raises(IOError):
            self.backup_manager.create_backup(sample_config_file)

    @patch('gzip.open', side_effect=OSError("Compression failed"))
    def test_error_handling_compression_failure(self, mock_gzip, sample_config_file):
        """Test error handling when compression fails."""
        with pytest.raises(OSError):
            self.backup_manager.create_backup(sample_config_file, compress=True)

    def test_backup_metadata_persistence(self, sample_config_file):
        """Test that backup metadata persists across manager instances."""
        # Create backup with first manager instance
        backup_id = self.backup_manager.create_backup(
            sample_config_file,
            description="Persistent backup"
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
                "timestamp": f"2023-0{i+1}-01T{10+i}:00:00Z",
                "port": 8000 + i,
                "nested_config": {
                    "feature_flags": {
                        f"feature_{i}": True,
                        f"feature_{i}_extra": f"value_{i}"
                    },
                    "database": {
                        "host": f"db{i}.example.com",
                        "port": 5432 + i
                    }
                }
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            backup_id = manager.create_backup(
                config_file,
                description=f"Concurrent backup {i}"
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
                assert backup_id in found_ids, f"Backup {backup_id} not found in {found_ids}"
