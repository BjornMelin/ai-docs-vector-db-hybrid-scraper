"""Additional tests for wizard.py edge cases to improve coverage."""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.wizard import ConfigurationWizard

# Mock questionary before importing wizard
sys.modules["questionary"] = MagicMock()


class TestWizardEdgeCases:
    """Test edge cases and error conditions for wizard to improve coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary.select")
    def test_restore_wizard_no_backup_selected(self, mock_select):
        """Test restore wizard when no backup is selected."""
        # Mock backup manager with available backups
        from src.config.backup_restore import BackupMetadata

        mock_backup = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2024-01-01T10:00:00Z",
            file_size=1024,
            environment="test",
        )

        self.wizard.backup_manager.list_backups = MagicMock(return_value=[mock_backup])

        # Mock user selecting nothing (None)
        mock_select.return_value.ask.return_value = None

        result = self.wizard.run_restore_wizard()

        # Should return False since no backup was selected
        assert result is False

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.confirm")
    def test_restore_wizard_backup_id_not_found(
        self, mock_confirm, mock_path, mock_select
    ):
        """Test restore wizard when backup ID is not found in choices."""
        # Mock backup manager with available backups
        from src.config.backup_restore import BackupMetadata

        mock_backup = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2024-01-01T10:00:00Z",
            file_size=1024,
            environment="test",
        )

        self.wizard.backup_manager.list_backups = MagicMock(return_value=[mock_backup])

        # Mock user selecting a backup that doesn't exist in choices
        mock_select.return_value.ask.return_value = "nonexistent_backup"
        mock_path.return_value.ask.return_value = "restored_config.json"
        mock_confirm.return_value.ask.return_value = True

        result = self.wizard.run_restore_wizard()

        # Should return False since backup ID was not found
        assert result is False

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.confirm")
    def test_restore_wizard_restore_failure(self, mock_confirm, mock_path, mock_select):
        """Test restore wizard when restore operation fails."""
        # Mock backup manager with available backups
        from src.config.backup_restore import BackupMetadata

        mock_backup = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2024-01-01T10:00:00Z",
            file_size=1024,
            environment="test",
        )

        self.wizard.backup_manager.list_backups = MagicMock(return_value=[mock_backup])

        # Mock user selections
        mock_select.return_value.ask.return_value = "backup_12345... - test_config"
        mock_path.return_value.ask.return_value = "restored_config.json"
        mock_confirm.return_value.ask.return_value = True

        # Mock failed restore operation
        from src.config.backup_restore import RestoreResult

        mock_result = RestoreResult(
            success=False,
            backup_id="backup_123456789012",
            config_path=Path("restored_config.json"),
            warnings=["Restore failed"],
        )
        self.wizard.backup_manager.restore_backup = MagicMock(return_value=mock_result)

        result = self.wizard.run_restore_wizard()

        # Should return False since restore failed
        assert result is False

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.confirm")
    def test_restore_wizard_exception_handling(
        self, mock_confirm, mock_path, mock_select
    ):
        """Test restore wizard when an exception occurs during restore."""
        # Mock backup manager with available backups
        from src.config.backup_restore import BackupMetadata

        mock_backup = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2024-01-01T10:00:00Z",
            file_size=1024,
            environment="test",
        )

        self.wizard.backup_manager.list_backups = MagicMock(return_value=[mock_backup])

        # Mock user selections
        mock_select.return_value.ask.return_value = "backup_12345... - test_config"
        mock_path.return_value.ask.return_value = "restored_config.json"
        mock_confirm.return_value.ask.return_value = True

        # Mock exception during restore
        self.wizard.backup_manager.restore_backup = MagicMock(
            side_effect=Exception("Restore error")
        )

        result = self.wizard.run_restore_wizard()

        # Should return False due to exception
        assert result is False

    def test_migration_setup_current_version_check_failure(self):
        """Test migration setup when current version check fails."""
        # Create a test config file
        config_path = self.temp_dir / "test_config.json"
        config_data = {"environment": "development"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock migration manager to raise exception on version check
        self.wizard.migration_manager.get_current_version = MagicMock(
            side_effect=Exception("Version check failed")
        )

        with pytest.raises(Exception, match="Version check failed"):
            self.wizard._migration_setup(config_path)

    @patch("src.config.wizard.questionary.confirm")
    def test_import_setup_file_load_exception(self, mock_confirm):
        """Test import setup when file loading fails."""
        # Create an invalid config file
        invalid_config_path = self.temp_dir / "invalid_config.json"
        with open(invalid_config_path, "w") as f:
            f.write("invalid json content")

        target_path = self.temp_dir / "target_config.json"

        with patch("src.config.wizard.questionary.path") as mock_path:
            mock_path.return_value.ask.side_effect = [
                str(invalid_config_path),
                str(target_path),
            ]

            result = self.wizard._import_setup(None)

            # Should return target_path even when import fails
            assert result == target_path

    def test_display_validation_report_with_issues(self):
        """Test validation report display with various issue types."""
        from src.config.enhanced_validators import ValidationIssue
        from src.config.enhanced_validators import ValidationReport
        from src.config.enhanced_validators import ValidationSeverity

        # Create report with different issue severities
        issues = [
            ValidationIssue(
                field_path="test_error",
                message="Test error message",
                severity=ValidationSeverity.ERROR,
                category="test",
            ),
            ValidationIssue(
                field_path="test_warning",
                message="Test warning message",
                severity=ValidationSeverity.WARNING,
                category="test",
            ),
            ValidationIssue(
                field_path="test_info",
                message="Test info message",
                severity=ValidationSeverity.INFO,
                category="test",
            ),
        ]

        report = ValidationReport(
            issues=issues, is_valid=False, config_hash="test_hash"
        )

        # Mock console to verify output
        with patch.object(self.wizard.console, "print") as mock_print:
            self.wizard._display_validation_report(report)

            # Should print at least headers for errors, warnings, and info
            assert mock_print.call_count >= 6
