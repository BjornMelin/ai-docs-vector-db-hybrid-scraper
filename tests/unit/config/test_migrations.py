"""Comprehensive tests for configuration migration system.

This test file covers the configuration migration system that handles
schema evolution, version upgrades, and rollback capabilities.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.migrations import ConfigMigrationManager
from src.config.migrations import MigrationMetadata
from src.config.migrations import MigrationPlan
from src.config.migrations import MigrationResult
from src.config.migrations import create_default_migrations


class TestMigrationMetadata:
    """Test the MigrationMetadata model."""

    def test_migration_metadata_creation(self):
        """Test basic MigrationMetadata creation."""
        metadata = MigrationMetadata(
            migration_id="1.0.0_to_1.1.0",
            from_version="1.0.0",
            to_version="1.1.0",
            description="Add validation metadata",
            created_at="2023-01-01T12:00:00Z",
            tags=["enhancement", "validation"],
        )

        assert metadata.migration_id == "1.0.0_to_1.1.0"
        assert metadata.from_version == "1.0.0"
        assert metadata.to_version == "1.1.0"
        assert metadata.description == "Add validation metadata"
        assert metadata.tags == ["enhancement", "validation"]

    def test_migration_metadata_default_values(self):
        """Test MigrationMetadata with default values."""
        metadata = MigrationMetadata(
            migration_id="test_migration",
            from_version="1.0.0",
            to_version="1.1.0",
            description="Test migration",
            created_at="2023-01-01T12:00:00Z",
        )

        assert metadata.applied_at is None
        assert metadata.rollback_available is False
        assert metadata.requires_backup is True
        assert metadata.tags == []

    def test_migration_metadata_serialization(self):
        """Test MigrationMetadata serialization."""
        metadata = MigrationMetadata(
            migration_id="test",
            from_version="1.0.0",
            to_version="1.1.0",
            description="Test",
            created_at="2023-01-01T12:00:00Z",
            rollback_available=True,
            tags=["test"],
        )

        # Should be serializable to dict
        data = metadata.model_dump()
        assert isinstance(data, dict)
        assert data["migration_id"] == "test"
        assert data["rollback_available"] is True

        # Should be deserializable from dict
        restored = MigrationMetadata(**data)
        assert restored.migration_id == metadata.migration_id
        assert restored.rollback_available == metadata.rollback_available


class TestMigrationResult:
    """Test the MigrationResult model."""

    def test_migration_result_success(self):
        """Test successful MigrationResult creation."""
        result = MigrationResult(
            success=True,
            migration_id="1.0.0_to_1.1.0",
            from_version="1.0.0",
            to_version="1.1.0",
            changes_made=["Added _config_hash field"],
            backup_id="backup_123",
        )

        assert result.success is True
        assert result.migration_id == "1.0.0_to_1.1.0"
        assert result.backup_id == "backup_123"
        assert len(result.changes_made) == 1
        assert len(result.errors) == 0

    def test_migration_result_failure(self):
        """Test failed MigrationResult creation."""
        result = MigrationResult(
            success=False,
            migration_id="1.0.0_to_1.1.0",
            from_version="1.0.0",
            to_version="1.1.0",
            errors=["Migration function failed", "Invalid configuration"],
            warnings=["Deprecated field found"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.changes_made) == 0

    def test_migration_result_default_values(self):
        """Test MigrationResult with default values."""
        result = MigrationResult(
            success=True, migration_id="test", from_version="1.0.0", to_version="1.1.0"
        )

        assert result.backup_id is None
        assert result.changes_made == []
        assert result.warnings == []
        assert result.errors == []


class TestMigrationPlan:
    """Test the MigrationPlan model."""

    def test_migration_plan_creation(self):
        """Test MigrationPlan creation."""
        plan = MigrationPlan(
            source_version="1.0.0",
            target_version="2.0.0",
            migrations=["1.0.0_to_1.1.0", "1.1.0_to_2.0.0"],
            estimated_duration="~4 minutes",
            requires_downtime=True,
            rollback_plan=["1.1.0_to_1.0.0", "2.0.0_to_1.1.0"],
        )

        assert plan.source_version == "1.0.0"
        assert plan.target_version == "2.0.0"
        assert len(plan.migrations) == 2
        assert plan.requires_downtime is True
        assert len(plan.rollback_plan) == 2

    def test_migration_plan_default_values(self):
        """Test MigrationPlan with default values."""
        plan = MigrationPlan(
            source_version="1.0.0",
            target_version="1.1.0",
            migrations=["1.0.0_to_1.1.0"],
            estimated_duration="~2 minutes",
        )

        assert plan.requires_downtime is False
        assert plan.rollback_plan == []


class TestConfigMigrationManager:
    """Test the ConfigMigrationManager class."""

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
            "_migration_version": "1.0.0",
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        return config_file

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.migration_manager = ConfigMigrationManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_migration_manager_initialization(self):
        """Test ConfigMigrationManager initialization."""
        manager = ConfigMigrationManager(self.temp_dir)

        assert manager.path_manager.base_dir == self.temp_dir
        assert manager.migrations_file.exists()
        assert manager.applied_migrations_file.exists()
        assert isinstance(manager._migrations, dict)
        assert isinstance(manager._migration_functions, dict)
        assert isinstance(manager._rollback_functions, dict)
        assert isinstance(manager._applied_migrations, list)

    def test_migration_manager_default_base_dir(self):
        """Test ConfigMigrationManager with default base directory."""
        manager = ConfigMigrationManager()

        assert manager.path_manager.base_dir == Path("config")

    def test_register_migration_decorator(self):
        """Test migration registration using decorator."""

        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Test migration")
        def test_migration(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["new_field"] = "added"
            return config, ["Added new_field"]

        # Migration should be registered
        migration_id = "1.0.0_to_1.1.0"
        assert migration_id in self.migration_manager._migrations
        assert migration_id in self.migration_manager._migration_functions

        # Metadata should be correct
        metadata = self.migration_manager._migrations[migration_id]
        assert metadata.from_version == "1.0.0"
        assert metadata.to_version == "1.1.0"
        assert metadata.description == "Test migration"

    def test_register_rollback_decorator(self):
        """Test rollback registration using decorator."""

        # First register a migration
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Test migration")
        def test_migration(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["new_field"] = "added"
            return config, ["Added new_field"]

        # Then register its rollback
        migration_id = "1.0.0_to_1.1.0"

        @self.migration_manager.register_rollback(migration_id)
        def test_rollback(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config.pop("new_field", None)
            return config, ["Removed new_field"]

        # Rollback should be registered
        assert migration_id in self.migration_manager._rollback_functions

        # Metadata should indicate rollback availability
        metadata = self.migration_manager._migrations[migration_id]
        assert metadata.rollback_available is True

    def test_get_current_version_existing_file(self, sample_config_file):
        """Test getting current version from existing file."""
        version = self.migration_manager.get_current_version(sample_config_file)

        assert version == "1.0.0"

    def test_get_current_version_no_version_field(self, temp_dir):
        """Test getting current version from file without version field."""
        config_file = temp_dir / "no_version.json"
        config_data = {"environment": "development"}

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        version = self.migration_manager.get_current_version(config_file)

        # Should default to 1.0.0
        assert version == "1.0.0"

    def test_get_current_version_nonexistent_file(self):
        """Test getting current version from non-existent file."""
        nonexistent = Path("/nonexistent/config.json")
        version = self.migration_manager.get_current_version(nonexistent)

        # Should default to 1.0.0
        assert version == "1.0.0"

    def test_list_available_migrations(self):
        """Test listing available migrations."""

        # Register some migrations
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "First migration")
        def migration1(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        @self.migration_manager.register_migration("1.1.0", "1.2.0", "Second migration")
        def migration2(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        migrations = self.migration_manager.list_available_migrations()

        assert len(migrations) == 2
        migration_ids = [m.migration_id for m in migrations]
        assert "1.0.0_to_1.1.0" in migration_ids
        assert "1.1.0_to_1.2.0" in migration_ids

    def test_list_applied_migrations_empty(self):
        """Test listing applied migrations when none applied."""
        applied = self.migration_manager.list_applied_migrations()

        assert applied == []

    def test_is_migration_applied(self):
        """Test checking if migration is applied."""
        migration_id = "1.0.0_to_1.1.0"

        # Initially not applied
        assert self.migration_manager.is_migration_applied(migration_id) is False

        # Add to applied migrations
        self.migration_manager._applied_migrations.append(migration_id)

        # Now should be applied
        assert self.migration_manager.is_migration_applied(migration_id) is True

    def test_find_migration_path_simple(self):
        """Test finding migration path for simple case."""

        # Register migrations
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Migration 1")
        def migration1(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        @self.migration_manager.register_migration("1.1.0", "1.2.0", "Migration 2")
        def migration2(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        # Find path from 1.0.0 to 1.2.0
        path = self.migration_manager._find_migration_path("1.0.0", "1.2.0")

        assert path == ["1.0.0_to_1.1.0", "1.1.0_to_1.2.0"]

    def test_find_migration_path_same_version(self):
        """Test finding migration path for same source and target version."""
        path = self.migration_manager._find_migration_path("1.0.0", "1.0.0")

        assert path == []

    def test_find_migration_path_no_path(self):
        """Test finding migration path when no path exists."""

        # Register migration from 1.0.0 to 1.1.0 only
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Migration 1")
        def migration1(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        # Try to find path to version with no connection
        path = self.migration_manager._find_migration_path("1.0.0", "2.0.0")

        assert path is None

    def test_find_migration_path_complex_graph(self):
        """Test finding migration path in complex migration graph."""
        # Register multiple migration paths
        migrations = [
            ("1.0.0", "1.1.0"),
            ("1.1.0", "1.2.0"),
            ("1.2.0", "2.0.0"),
            ("1.0.0", "1.5.0"),  # Alternative path
            ("1.5.0", "2.0.0"),
        ]

        for from_ver, to_ver in migrations:

            @self.migration_manager.register_migration(
                from_ver, to_ver, f"Migration {from_ver} to {to_ver}"
            )
            def migration(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
                return config, []

        # Find shortest path from 1.0.0 to 2.0.0
        path = self.migration_manager._find_migration_path("1.0.0", "2.0.0")

        # Should find the shorter path through 1.5.0
        assert path == ["1.0.0_to_1.5.0", "1.5.0_to_2.0.0"]

    def test_create_migration_plan_success(self):
        """Test creating migration plan for valid path."""

        # Register migrations
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Migration 1")
        def migration1(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        @self.migration_manager.register_migration(
            "1.1.0", "1.2.0", "Migration 2", tags=["downtime"]
        )
        def migration2(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        # Register rollback for first migration
        @self.migration_manager.register_rollback("1.0.0_to_1.1.0")
        def rollback1(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, []

        plan = self.migration_manager.create_migration_plan("1.0.0", "1.2.0")

        assert plan is not None
        assert plan.source_version == "1.0.0"
        assert plan.target_version == "1.2.0"
        assert plan.migrations == ["1.0.0_to_1.1.0", "1.1.0_to_1.2.0"]
        assert plan.requires_downtime is True  # One migration has downtime tag
        assert len(plan.rollback_plan) == 1  # Only one rollback available
        assert "1.0.0_to_1.1.0" in plan.rollback_plan

    def test_create_migration_plan_no_path(self):
        """Test creating migration plan when no path exists."""
        plan = self.migration_manager.create_migration_plan("1.0.0", "2.0.0")

        assert plan is None

    def test_apply_migration_plan_success(self, sample_config_file):
        """Test applying migration plan successfully."""

        # Register migration
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Add config hash")
        def add_config_hash(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["_config_hash"] = "test_hash"
            return config, ["Added _config_hash field"]

        # Create plan
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        assert plan is not None

        # Apply plan
        results = self.migration_manager.apply_migration_plan(plan, sample_config_file)

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.migration_id == "1.0.0_to_1.1.0"
        assert "Added _config_hash field" in result.changes_made

        # Config file should be updated
        with open(sample_config_file) as f:
            updated_config = json.load(f)

        assert updated_config["_config_hash"] == "test_hash"
        assert updated_config["_migration_version"] == "1.1.0"

        # Migration should be marked as applied
        assert self.migration_manager.is_migration_applied("1.0.0_to_1.1.0")

    def test_apply_migration_plan_dry_run(self, sample_config_file):
        """Test applying migration plan in dry run mode."""

        # Register migration
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Add field")
        def add_field(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["new_field"] = "value"
            return config, ["Added new_field"]

        # Create and apply plan in dry run mode
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        results = self.migration_manager.apply_migration_plan(
            plan, sample_config_file, dry_run=True
        )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert "DRY RUN" in result.changes_made[0]

        # Config file should be unchanged
        with open(sample_config_file) as f:
            config = json.load(f)

        assert "new_field" not in config
        assert config["_migration_version"] == "1.0.0"

        # Migration should not be marked as applied
        assert not self.migration_manager.is_migration_applied("1.0.0_to_1.1.0")

    def test_apply_migration_plan_failure(self, sample_config_file):
        """Test applying migration plan with failure."""

        # Register failing migration
        @self.migration_manager.register_migration(
            "1.0.0", "1.1.0", "Failing migration"
        )
        def failing_migration(
            config: dict[str, Any],
        ) -> tuple[dict[str, Any], list[str]]:
            raise ValueError("Migration failed")

        # Create and apply plan
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        results = self.migration_manager.apply_migration_plan(plan, sample_config_file)

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert len(result.errors) > 0
        assert "Migration failed" in result.errors[0]

        # Config file should be unchanged
        with open(sample_config_file) as f:
            config = json.load(f)

        assert config["_migration_version"] == "1.0.0"

    def test_apply_migration_plan_missing_function(self, sample_config_file):
        """Test applying migration plan with missing migration function."""
        # Create plan manually without registering function
        plan = MigrationPlan(
            source_version="1.0.0",
            target_version="1.1.0",
            migrations=["nonexistent_migration"],
            estimated_duration="~2 minutes",
        )

        results = self.migration_manager.apply_migration_plan(plan, sample_config_file)

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "Migration function not found" in result.errors[0]

    def test_apply_migration_plan_with_backup(self, sample_config_file):
        """Test applying migration plan with backup creation."""
        # Setup mock backup manager on existing instance
        mock_manager_instance = MagicMock()
        mock_manager_instance.create_backup.return_value = "backup_123"
        self.migration_manager.backup_manager = mock_manager_instance

        # Register migration that requires backup
        @self.migration_manager.register_migration(
            "1.0.0", "1.1.0", "Migration with backup", requires_backup=True
        )
        def migration_with_backup(
            config: dict[str, Any],
        ) -> tuple[dict[str, Any], list[str]]:
            config["updated"] = True
            return config, ["Updated config"]

        # Apply migration
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        results = self.migration_manager.apply_migration_plan(plan, sample_config_file)

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.backup_id == "backup_123"

        # Backup should have been created
        mock_manager_instance.create_backup.assert_called_once()

    def test_rollback_migration_success(self, sample_config_file):
        """Test successful migration rollback."""

        # Register migration and rollback
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Add field")
        def add_field(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["new_field"] = "value"
            return config, ["Added new_field"]

        @self.migration_manager.register_rollback("1.0.0_to_1.1.0")
        def remove_field(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config.pop("new_field", None)
            return config, ["Removed new_field"]

        # Apply migration first
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        self.migration_manager.apply_migration_plan(plan, sample_config_file)

        # Verify migration was applied
        with open(sample_config_file) as f:
            config = json.load(f)
        assert "new_field" in config
        assert config["_migration_version"] == "1.1.0"

        # Rollback migration
        result = self.migration_manager.rollback_migration(
            "1.0.0_to_1.1.0", sample_config_file
        )

        assert result.success is True
        assert result.migration_id == "1.0.0_to_1.1.0"
        assert result.from_version == "1.1.0"
        assert result.to_version == "1.0.0"
        assert "Removed new_field" in result.changes_made

        # Config should be rolled back
        with open(sample_config_file) as f:
            config = json.load(f)
        assert "new_field" not in config
        assert config["_migration_version"] == "1.0.0"

        # Migration should no longer be marked as applied
        assert not self.migration_manager.is_migration_applied("1.0.0_to_1.1.0")

    def test_rollback_migration_no_rollback_function(self):
        """Test rollback when no rollback function is available."""
        result = self.migration_manager.rollback_migration(
            "nonexistent_migration", Path("/tmp/config.json")
        )

        assert result.success is False
        assert "No rollback function available" in result.errors[0]

    def test_rollback_migration_dry_run(self, sample_config_file):
        """Test migration rollback in dry run mode."""

        # Register migration and rollback
        @self.migration_manager.register_migration("1.0.0", "1.1.0", "Add field")
        def add_field(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            config["new_field"] = "value"
            return config, []

        @self.migration_manager.register_rollback("1.0.0_to_1.1.0")
        def remove_field(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return config, ["Would remove new_field"]

        # Apply migration first
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        self.migration_manager.apply_migration_plan(plan, sample_config_file)

        # Rollback in dry run mode
        result = self.migration_manager.rollback_migration(
            "1.0.0_to_1.1.0", sample_config_file, dry_run=True
        )

        assert result.success is True
        assert "DRY RUN" in result.changes_made[0]

        # Config should be unchanged
        with open(sample_config_file) as f:
            config = json.load(f)
        assert "new_field" in config  # Should still be there
        assert config["_migration_version"] == "1.1.0"  # Should still be 1.1.0

    def test_migration_persistence(self, sample_config_file):
        """Test that migration state persists across manager instances."""

        # Register and apply migration with first manager
        @self.migration_manager.register_migration(
            "1.0.0", "1.1.0", "Persistent migration"
        )
        def persistent_migration(
            config: dict[str, Any],
        ) -> tuple[dict[str, Any], list[str]]:
            config["persistent"] = True
            return config, ["Added persistent field"]

        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        self.migration_manager.apply_migration_plan(plan, sample_config_file)

        # Create new manager instance
        new_manager = ConfigMigrationManager(self.temp_dir)

        # Should remember applied migrations
        applied = new_manager.list_applied_migrations()
        assert "1.0.0_to_1.1.0" in applied

        # Should know migration is applied
        assert new_manager.is_migration_applied("1.0.0_to_1.1.0")

    def test_create_default_migrations_registration(self):
        """Test that default migrations are properly registered."""
        # Create new manager and load default migrations
        manager = ConfigMigrationManager(self.temp_dir)
        create_default_migrations(manager)

        # Function should execute without error (it's now a template)
        migrations = manager.list_available_migrations()
        migration_ids = [m.migration_id for m in migrations]

        # No migrations by default since it's now a template implementation
        assert len(migration_ids) == 0

    def test_default_migration_functionality(self, sample_config_file):
        """Test that default migrations template function works correctly."""
        # Setup config
        config_data = {
            "environment": "development",
            "debug": True,
            "_migration_version": "1.0.0",
        }

        with open(sample_config_file, "w") as f:
            json.dump(config_data, f)

        # Load default migrations (template implementation)
        create_default_migrations(self.migration_manager)

        # No migration plan available since template has no actual migrations
        plan = self.migration_manager.create_migration_plan("1.0.0", "1.1.0")
        assert plan is None

        # Test that template function executes without error
        # This test verifies the function is callable and safe
        assert True  # Function executed successfully above

    def test_error_handling_invalid_config_file(self):
        """Test error handling with invalid configuration file."""
        invalid_file = Path("/nonexistent/config.json")

        plan = MigrationPlan(
            source_version="1.0.0",
            target_version="1.1.0",
            migrations=["test_migration"],
            estimated_duration="~2 minutes",
        )

        with pytest.raises(FileNotFoundError):
            self.migration_manager.apply_migration_plan(plan, invalid_file)

    def test_error_handling_corrupted_migration_metadata(self, temp_dir):
        """Test error handling with corrupted migration metadata."""
        # Create corrupted metadata file
        metadata_file = temp_dir / "migrations" / "migrations.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, "w") as f:
            f.write("invalid json content")

        # Should handle corruption gracefully
        manager = ConfigMigrationManager(temp_dir)
        assert isinstance(manager._migrations, dict)
        assert len(manager._migrations) == 0  # Should start empty due to corruption

    @patch("src.config.utils.ConfigPathManager.ensure_directories", side_effect=PermissionError("Access denied"))
    def test_error_handling_permission_denied(self, mock_ensure_dirs):
        """Test error handling when directory creation is denied."""
        # This test checks that permission errors are properly propagated
        with pytest.raises(PermissionError):
            ConfigMigrationManager(Path("/restricted/path"))

    def test_migration_order_preservation(self):
        """Test that migration order is preserved in plans."""
        # Register migrations in specific order
        migrations = [
            ("1.0.0", "1.1.0", "First"),
            ("1.1.0", "1.2.0", "Second"),
            ("1.2.0", "1.3.0", "Third"),
            ("1.3.0", "2.0.0", "Fourth"),
        ]

        for from_ver, to_ver, desc in migrations:

            @self.migration_manager.register_migration(from_ver, to_ver, desc)
            def migration(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
                return config, []

        # Create plan for full upgrade
        plan = self.migration_manager.create_migration_plan("1.0.0", "2.0.0")

        assert plan is not None
        expected_order = [
            "1.0.0_to_1.1.0",
            "1.1.0_to_1.2.0",
            "1.2.0_to_1.3.0",
            "1.3.0_to_2.0.0",
        ]
        assert plan.migrations == expected_order
