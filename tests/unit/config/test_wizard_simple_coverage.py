"""Simple tests to achieve 90% coverage target for wizard module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.enhanced_validators import ValidationIssue
from src.config.enhanced_validators import ValidationReport
from src.config.enhanced_validators import ValidationSeverity
from src.config.wizard import ConfigurationWizard


class TestConfigurationWizardSimpleCoverage:
    """Simple tests to cover missing lines in wizard module."""

    @pytest.fixture
    def wizard(self):
        """Create wizard instance for testing."""
        return ConfigurationWizard()

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(
                '{"environment": "development", "debug": true, "log_level": "DEBUG"}'
            )
            f.flush()
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)

    def test_run_validation_wizard_with_errors_no_fix(self, wizard, sample_config_file):
        """Test validation wizard when user doesn't want to fix errors."""
        with (
            patch("src.config.wizard.UnifiedConfig.load_from_file"),
            patch("src.config.wizard.ConfigurationValidator") as mock_validator_class,
            patch("src.config.wizard.questionary.confirm") as mock_confirm,
        ):
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator

            # Create error issues
            error_issue = ValidationIssue(
                field_path="api_key",
                message="API key required",
                severity=ValidationSeverity.ERROR,
                category="authentication",
            )

            mock_report = ValidationReport(
                issues=[error_issue], is_valid=False, config_hash="test_hash"
            )
            mock_validator.validate_configuration.return_value = mock_report

            # User says no to fixing issues
            mock_confirm.return_value.ask.return_value = False

            result = wizard.run_validation_wizard(sample_config_file)

            # Should return False (line 116)
            assert result is False

    def test_backup_wizard_error_handling(self, wizard, sample_config_file):
        """Test backup wizard with error condition to cover error handling lines."""
        with (
            patch("src.config.wizard.questionary.text") as mock_text,
            patch("src.config.wizard.questionary.confirm") as mock_confirm,
            patch.object(wizard.backup_manager, "create_backup") as mock_create,
        ):
            # Mock backup creation failure
            mock_create.side_effect = Exception("Backup failed")

            mock_text.return_value.ask.return_value = "test backup"
            mock_confirm.return_value.ask.return_value = True

            # Should handle exception and return None
            result = wizard.run_backup_wizard(sample_config_file)
            assert result is None

    def test_display_validation_report_coverage(self, wizard):
        """Test validation report display to cover display methods."""
        # Create validation report with mixed issues
        issues = [
            ValidationIssue(
                field_path="api_key",
                message="API key required",
                severity=ValidationSeverity.ERROR,
                category="authentication",
            ),
            ValidationIssue(
                field_path="debug",
                message="Debug enabled in production",
                severity=ValidationSeverity.WARNING,
                category="environment",
            ),
            ValidationIssue(
                field_path="cache",
                message="Using default cache settings",
                severity=ValidationSeverity.INFO,
                category="configuration",
            ),
        ]

        report = ValidationReport(
            issues=issues, is_valid=False, config_hash="test_hash"
        )

        # This should execute without error and cover display lines
        wizard._display_validation_report(report)

    def test_backup_wizard_with_tags_and_description(self, wizard, sample_config_file):
        """Test backup wizard with tags and description to cover those code paths."""
        with (
            patch("src.config.wizard.questionary.text") as mock_text,
            patch("src.config.wizard.questionary.select") as mock_select,
            patch("src.config.wizard.questionary.confirm") as mock_confirm,
            patch.object(wizard.backup_manager, "create_backup") as mock_create,
        ):
            # Mock responses for description and tags
            mock_text.return_value.ask.side_effect = [
                "Test backup description",  # description
                "tag1,tag2,tag3",  # tags
            ]
            mock_select.return_value.ask.return_value = "gzip"  # compression
            mock_confirm.return_value.ask.return_value = True  # compress

            mock_create.return_value = "backup_123456789012"

            result = wizard.run_backup_wizard(sample_config_file)

            assert result == "backup_123456789012"

            # Verify backup was created with correct parameters
            mock_create.assert_called_once_with(
                sample_config_file,
                description="Test backup description",
                tags=["tag1", "tag2", "tag3"],
                compress=True,
            )

    def test_validation_wizard_basic_coverage(self, wizard, sample_config_file):
        """Test basic validation wizard functionality to cover simple paths."""
        with (
            patch("src.config.wizard.UnifiedConfig.load_from_file"),
            patch("src.config.wizard.ConfigurationValidator") as mock_validator_class,
        ):
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator

            # Create valid report
            mock_report = ValidationReport(
                issues=[], is_valid=True, config_hash="test_hash"
            )
            mock_validator.validate_configuration.return_value = mock_report

            result = wizard.run_validation_wizard(sample_config_file)

            # Should return True for valid config
            assert result is True
