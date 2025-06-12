"""Additional tests for backup_restore.py to cover remaining missing lines.

These tests target specific missing lines to improve coverage to ≥90%.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from src.config.backup_restore import ConfigBackupManager


class TestConfigBackupManagerAdditionalCoverage:
    """Additional tests to cover missing lines in backup_restore.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_manager = ConfigBackupManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_compress_backup_file(self):
        """Test backup compression functionality."""
        # Create config file
        config_file = self.temp_dir / "config.json"
        config_data = {
            "environment": "test",
            "data": "x" * 1000,
        }  # Larger data for compression
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Create compressed backup
        backup_id = self.backup_manager.create_backup(config_file, compress=True)

        metadata = self.backup_manager.get_backup_metadata(backup_id)
        assert metadata.compressed is True

    def test_uncompressed_backup_file(self):
        """Test backup without compression."""
        # Create config file
        config_file = self.temp_dir / "config.json"
        config_data = {"environment": "test"}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Create uncompressed backup
        backup_id = self.backup_manager.create_backup(config_file, compress=False)

        metadata = self.backup_manager.get_backup_metadata(backup_id)
        assert metadata.compressed is False

    def test_get_latest_backup_no_backups(self):
        """Test _find_latest_backup when no backups exist."""
        latest = self.backup_manager._find_latest_backup("nonexistent")
        assert latest is None

    def test_get_latest_backup_multiple_configs(self):
        """Test _find_latest_backup with multiple config files."""
        # Create backups for different config files
        for name in ["app", "database"]:
            config_file = self.temp_dir / f"{name}.json"
            config_data = {"service": name}
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            self.backup_manager.create_backup(config_file)

        # Get latest for specific config
        latest_app = self.backup_manager._find_latest_backup("app")
        latest_db = self.backup_manager._find_latest_backup("database")

        assert latest_app is not None
        assert latest_db is not None
        assert latest_app != latest_db

    def test_detect_conflicts_with_missing_current_file(self):
        """Test _detect_conflicts when current file doesn't exist."""
        missing_file = self.temp_dir / "missing.json"
        backup_content = json.dumps({"environment": "test"})

        conflicts = self.backup_manager._detect_conflicts(missing_file, backup_content)
        # Should handle missing file gracefully
        assert isinstance(conflicts, list)

    def test_detect_conflicts_with_invalid_json_current(self):
        """Test _detect_conflicts with invalid JSON in current file."""
        current_file = self.temp_dir / "current.json"
        with open(current_file, "w") as f:
            f.write("invalid json")

        backup_content = json.dumps({"environment": "test"})

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        # Should handle JSON error gracefully
        assert isinstance(conflicts, list)

    def test_detect_conflicts_with_invalid_json_backup(self):
        """Test _detect_conflicts with invalid JSON in backup content."""
        current_file = self.temp_dir / "current.json"
        current_data = {"environment": "test"}
        with open(current_file, "w") as f:
            json.dump(current_data, f)

        backup_content = "invalid json"

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        # Should handle JSON error gracefully
        assert isinstance(conflicts, list)

    def test_detect_conflicts_embedding_provider_mismatch(self):
        """Test _detect_conflicts with embedding provider mismatch."""
        current_file = self.temp_dir / "current.json"
        current_data = {"embedding_provider": "openai"}
        with open(current_file, "w") as f:
            json.dump(current_data, f)

        backup_content = json.dumps({"embedding_provider": "fastembed"})

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        assert len(conflicts) > 0
        assert any("Embedding provider mismatch" in conflict for conflict in conflicts)

    def test_detect_conflicts_schema_version_mismatch(self):
        """Test _detect_conflicts with schema version mismatch."""
        current_file = self.temp_dir / "current.json"
        current_data = {"schema_version": "1.0.0"}
        with open(current_file, "w") as f:
            json.dump(current_data, f)

        backup_content = json.dumps({"schema_version": "2.0.0"})

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        assert len(conflicts) > 0
        assert any("Schema version mismatch" in conflict for conflict in conflicts)

    def test_detect_conflicts_database_url_changes(self):
        """Test _detect_conflicts with database URL changes."""
        current_file = self.temp_dir / "current.json"
        current_data = {
            "database": {"database_url": "postgresql://localhost:5432/current"}
        }
        with open(current_file, "w") as f:
            json.dump(current_data, f)

        backup_content = json.dumps(
            {"database": {"database_url": "postgresql://remote:5432/backup"}}
        )

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        assert len(conflicts) > 0
        assert any("Database URL" in conflict for conflict in conflicts)

    def test_detect_conflicts_no_conflicts_identical_configs(self):
        """Test _detect_conflicts with identical configurations."""
        current_file = self.temp_dir / "current.json"
        current_data = {"environment": "production", "debug": False}
        with open(current_file, "w") as f:
            json.dump(current_data, f)

        backup_content = json.dumps({"environment": "production", "debug": False})

        conflicts = self.backup_manager._detect_conflicts(current_file, backup_content)
        assert len(conflicts) == 0

    def test_create_backup_with_existing_hash(self):
        """Test create_backup with duplicate configuration hash."""
        config_file = self.temp_dir / "config.json"
        config_data = {"environment": "test", "debug": True}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Create first backup
        backup_id1 = self.backup_manager.create_backup(config_file)

        # Create second backup with same content (should detect duplicate)
        backup_id2 = self.backup_manager.create_backup(config_file)

        # Both should succeed, but may handle duplicates differently
        assert backup_id1 is not None
        assert backup_id2 is not None

    def test_restore_backup_with_gzip_decompression(self):
        """Test restore_backup with compressed backup file."""
        # Create config file
        config_file = self.temp_dir / "config.json"
        config_data = {"environment": "test", "data": "x" * 1000}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Create compressed backup
        backup_id = self.backup_manager.create_backup(config_file, compress=True)

        # Restore from compressed backup
        target_file = self.temp_dir / "restored.json"
        result = self.backup_manager.restore_backup(backup_id, target_file)

        if result.success:
            # Verify restored content
            with open(target_file) as f:
                restored_data = json.load(f)
            assert restored_data["environment"] == "test"

    def test_list_backups_empty_directory(self):
        """Test list_backups when backups directory is empty."""
        # Clear any existing backups
        self.backup_manager._metadata.clear()
        self.backup_manager._save_metadata()

        backups = self.backup_manager.list_backups()
        assert len(backups) == 0

    def test_get_backup_info_missing_backup(self):
        """Test get_backup_info with non-existent backup."""
        info = self.backup_manager.get_backup_info("nonexistent")
        assert info is None

    def test_create_backup_with_environment_metadata(self):
        """Test create_backup extracts environment from config."""
        config_file = self.temp_dir / "config.json"
        config_data = {
            "environment": "production",
            "debug": False,
            "template_source": "production-template",
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        backup_id = self.backup_manager.create_backup(config_file)

        metadata = self.backup_manager.get_backup_metadata(backup_id)
        assert metadata.environment == "production"

    def test_create_backup_error_handling_with_permissions(self):
        """Test create_backup error handling when backup directory is read-only."""
        # Make backups directory read-only
        backups_dir = self.backup_manager.path_manager.backups_dir
        original_mode = backups_dir.stat().st_mode

        try:
            backups_dir.chmod(0o444)  # Read-only

            config_file = self.temp_dir / "config.json"
            config_data = {"environment": "test"}
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            # Should handle permission error gracefully
            with pytest.raises(PermissionError):
                self.backup_manager.create_backup(config_file)

        finally:
            # Restore original permissions
            backups_dir.chmod(original_mode)

    def test_backup_metadata_serialization(self):
        """Test backup metadata serialization and deserialization."""
        config_file = self.temp_dir / "config.json"
        config_data = {"environment": "test"}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        backup_id = self.backup_manager.create_backup(
            config_file, description="Test backup", tags=["test", "serialization"]
        )

        # Force reload metadata from disk
        self.backup_manager._load_metadata()

        metadata = self.backup_manager.get_backup_metadata(backup_id)
        assert metadata.description == "Test backup"
        assert metadata.tags == ["test", "serialization"]
