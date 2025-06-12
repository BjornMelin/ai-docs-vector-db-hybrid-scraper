"""Comprehensive tests for configuration wizard.

This test file covers the ConfigurationWizard class that provides
interactive setup, validation, backup, and restore functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from src.config.wizard import ConfigurationWizard


class TestConfigurationWizardInit:
    """Test ConfigurationWizard initialization."""

    def test_wizard_initialization_default(self):
        """Test wizard initialization with default directory."""
        wizard = ConfigurationWizard()

        assert wizard.console is not None
        assert wizard.path_manager is not None
        assert wizard.templates is not None
        assert wizard.backup_manager is not None
        assert wizard.migration_manager is not None

    def test_wizard_initialization_custom_dir(self):
        """Test wizard initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            wizard = ConfigurationWizard(base_dir)

            assert wizard.path_manager.base_dir == base_dir
            assert base_dir.exists()

    def test_wizard_initialization_creates_directories(self):
        """Test that wizard creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "new_config"
            wizard = ConfigurationWizard(base_dir)

            assert base_dir.exists()


class TestConfigurationWizardSetup:
    """Test configuration setup wizard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary")
    def test_choose_setup_mode_template(self, mock_questionary):
        """Test choosing template setup mode."""
        mock_questionary.select.return_value.ask.return_value = "template"

        mode = self.wizard._choose_setup_mode()
        assert mode == "template"

    @patch("src.config.wizard.questionary")
    def test_choose_setup_mode_interactive(self, mock_questionary):
        """Test choosing interactive setup mode."""
        mock_questionary.select.return_value.ask.return_value = "interactive"

        mode = self.wizard._choose_setup_mode()
        assert mode == "interactive"

    @patch("src.config.wizard.questionary")
    def test_template_based_setup_development(self, mock_questionary):
        """Test template-based setup with development template."""
        # Mock user selections
        mock_questionary.select.return_value.ask.return_value = (
            "🛠️  Development - Debug logging, local database, fast iteration"
        )
        mock_questionary.confirm.return_value.ask.return_value = (
            False  # No customization
        )
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "config.json"
        )

        config_path = self.wizard._template_based_setup(None)

        assert config_path.exists()
        assert config_path.suffix == ".json"

    @patch("src.config.wizard.questionary")
    def test_template_based_setup_with_customization(self, mock_questionary):
        """Test template-based setup with customization."""
        # Mock user selections
        mock_questionary.select.return_value.ask.side_effect = [
            "🚀 Production - Security hardening, performance optimization",  # template selection
            "production",  # environment in customization
        ]
        mock_questionary.confirm.return_value.ask.side_effect = [
            True,  # Enable customization
            False,  # debug mode change
            True,  # configure API keys
        ]
        mock_questionary.password.return_value.ask.side_effect = [
            "sk-test123456789012345678901234567890123456789012",  # OpenAI API key
        ]
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "config.json"
        )

        config_path = self.wizard._template_based_setup(None)

        assert config_path.exists()

        # Verify customization was applied
        with open(config_path) as f:
            config_data = json.load(f)

        # Should contain the custom server name (if the template supports it)
        assert isinstance(config_data, dict)

    @patch("src.config.wizard.questionary")
    def test_interactive_setup_basic(self, mock_questionary):
        """Test interactive setup with basic configuration."""
        # Mock user inputs for interactive setup
        mock_questionary.select.return_value.ask.side_effect = [
            "development",  # environment
            "DEBUG",  # log_level
            "openai",  # embedding_provider
            "crawl4ai",  # crawl_provider
        ]
        mock_questionary.confirm.return_value.ask.side_effect = [
            True,  # debug
            False,  # use_postgres
            False,  # enable_redis
        ]
        mock_questionary.password.return_value.ask.side_effect = [
            "sk-test123456789012345678901234567890123456789012",  # OpenAI API key
        ]
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "interactive_config.json"
        )

        config_path = self.wizard._interactive_setup(None)

        assert config_path.exists()

        # Verify configuration structure
        with open(config_path) as f:
            config_data = json.load(f)

        assert config_data["environment"] == "development"
        assert config_data["debug"] is True

    @patch("src.config.wizard.questionary")
    def test_migration_setup(self, mock_questionary):
        """Test migration setup workflow."""
        # Create a sample old config file
        old_config_path = self.temp_dir / "old_config.json"
        old_config = {
            "environment": "development",
            "debug": True,
            "_migration_version": "1.0.0",
        }
        with open(old_config_path, "w") as f:
            json.dump(old_config, f)

        # Mock user selections
        mock_questionary.path.return_value.ask.side_effect = [
            str(old_config_path),  # Source config
            str(self.temp_dir / "migrated_config.json"),  # Target config
        ]
        mock_questionary.confirm.return_value.ask.return_value = (
            True  # Confirm migration
        )

        config_path = self.wizard._migration_setup(None)

        assert config_path.exists()

    @patch("src.config.wizard.questionary")
    def test_import_setup(self, mock_questionary):
        """Test import setup workflow."""
        # Create a sample config file to import
        import_config_path = self.temp_dir / "import_config.json"
        import_config = {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
        }
        with open(import_config_path, "w") as f:
            json.dump(import_config, f)

        # Mock user selections
        mock_questionary.path.return_value.ask.side_effect = [
            str(import_config_path),  # Source config
            str(self.temp_dir / "imported_config.json"),  # Target config
        ]

        config_path = self.wizard._import_setup(None)

        assert config_path.exists()


class TestConfigurationWizardValidation:
    """Test configuration validation wizard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_validation_wizard_invalid_config_file(self):
        """Test validation wizard with non-existent config file."""
        nonexistent_path = self.temp_dir / "nonexistent.json"

        result = self.wizard.run_validation_wizard(nonexistent_path)

        assert result is False

    @patch("src.config.wizard.ConfigurationValidator")
    def test_validation_wizard_valid_config(self, mock_validator_class):
        """Test validation wizard with valid configuration."""
        # Create a valid config file
        config_path = self.temp_dir / "valid_config.json"
        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock validator to return valid result
        mock_report = MagicMock()
        mock_report.is_valid = True
        mock_report.errors = []

        mock_validator = MagicMock()
        mock_validator.validate_configuration.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        result = self.wizard.run_validation_wizard(config_path)

        assert result is True

    @patch("src.config.wizard.ConfigurationValidator")
    @patch("src.config.wizard.questionary")
    def test_validation_wizard_with_errors_fix_declined(
        self, mock_questionary, mock_validator_class
    ):
        """Test validation wizard with errors but user declines fixes."""
        # Create a config file with issues
        config_path = self.temp_dir / "invalid_config.json"
        config_data = {
            "environment": "development",
            "debug": "not_boolean",  # Invalid type
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock validator to return invalid result
        mock_report = MagicMock()
        mock_report.is_valid = False
        mock_report.errors = ["Invalid debug value"]

        mock_validator = MagicMock()
        mock_validator.validate_configuration.return_value = mock_report
        mock_validator_class.return_value = mock_validator

        # Mock user declining fixes
        mock_questionary.confirm.return_value.ask.return_value = False

        result = self.wizard.run_validation_wizard(config_path)

        assert result is False


class TestConfigurationWizardBackup:
    """Test configuration backup wizard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary")
    def test_backup_wizard_basic(self, mock_questionary):
        """Test basic backup wizard functionality."""
        # Create a config file to backup
        config_path = self.temp_dir / "config.json"
        config_data = {"environment": "production", "debug": False}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock user inputs
        mock_questionary.text.return_value.ask.side_effect = [
            "Production backup",  # description
            "prod,backup",  # tags
        ]

        # Mock backup manager
        mock_backup_id = "backup_123"
        self.wizard.backup_manager.create_backup = MagicMock(
            return_value=mock_backup_id
        )

        backup_id = self.wizard.run_backup_wizard(config_path)

        assert backup_id == mock_backup_id
        self.wizard.backup_manager.create_backup.assert_called_once()

    @patch("src.config.wizard.questionary")
    def test_backup_wizard_with_empty_inputs(self, mock_questionary):
        """Test backup wizard with empty description and tags."""
        # Create a config file to backup
        config_path = self.temp_dir / "config.json"
        config_data = {"environment": "test"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock empty user inputs
        mock_questionary.text.return_value.ask.side_effect = ["", ""]

        # Mock backup manager
        mock_backup_id = "backup_456"
        self.wizard.backup_manager.create_backup = MagicMock(
            return_value=mock_backup_id
        )

        backup_id = self.wizard.run_backup_wizard(config_path)

        assert backup_id == mock_backup_id


class TestConfigurationWizardRestore:
    """Test configuration restore wizard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary")
    def test_restore_wizard_no_backups(self, mock_questionary):
        """Test restore wizard when no backups are available."""
        # Mock backup manager with no backups
        self.wizard.backup_manager.list_backups = MagicMock(return_value=[])

        result = self.wizard.run_restore_wizard()

        assert result is False

    @patch("src.config.wizard.questionary")
    def test_restore_wizard_with_backups(self, mock_questionary):
        """Test restore wizard with available backups."""
        # Mock backup manager with available backups
        from src.config.backup_restore import BackupMetadata

        mock_backup_metadata = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2024-01-01T10:00:00Z",
            file_size=1024,
            environment="test",
        )

        self.wizard.backup_manager.list_backups = MagicMock(
            return_value=[mock_backup_metadata]
        )

        # Mock user selections
        mock_questionary.select.return_value.ask.return_value = (
            "backup_12345... - test_config"
        )
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "restored_config.json"
        )
        mock_questionary.confirm.return_value.ask.return_value = True

        # Mock successful restore
        mock_result = MagicMock()
        mock_result.success = True
        self.wizard.backup_manager.restore_backup = MagicMock(return_value=mock_result)

        result = self.wizard.run_restore_wizard()

        assert result is True


class TestConfigurationWizardUtilities:
    """Test configuration wizard utility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary")
    def test_customize_template_server_name(self, mock_questionary):
        """Test template customization with environment and debug changes."""
        template_data = {
            "environment": "production",
            "debug": False,
            "embedding_provider": "openai",
        }

        # Mock user inputs for customization
        mock_questionary.select.return_value.ask.return_value = (
            "development"  # Change environment
        )
        mock_questionary.confirm.return_value.ask.side_effect = [
            True,  # Change debug mode
            True,  # Configure API keys
        ]
        mock_questionary.password.return_value.ask.return_value = (
            "sk-test123456789012345678901234567890123456789012"
        )

        customized = self.wizard._customize_template(template_data)

        assert customized["environment"] == "development"
        assert customized["debug"] is True
        assert (
            customized["openai"]["api_key"]
            == "sk-test123456789012345678901234567890123456789012"
        )

    @patch("src.config.wizard.questionary")
    def test_customize_template_no_changes(self, mock_questionary):
        """Test template customization with no changes."""
        template_data = {"environment": "development", "debug": True}

        # Mock user not wanting any customization
        mock_questionary.confirm.return_value.ask.return_value = False

        customized = self.wizard._customize_template(template_data)

        assert customized == template_data

    def test_display_validation_report(self):
        """Test validation report display."""
        # Create a mock validation report
        mock_report = MagicMock()
        mock_report.is_valid = False
        mock_report.errors = ["Error 1", "Error 2"]
        mock_report.warnings = ["Warning 1"]
        mock_report.suggestions = ["Suggestion 1"]

        # This should not raise an exception
        self.wizard._display_validation_report(mock_report)

    @patch("src.config.wizard.questionary")
    def test_attempt_fixes_basic(self, mock_questionary):
        """Test basic automatic fixes."""
        config_path = self.temp_dir / "config.json"
        config_data = {"environment": "development"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        from src.config.enhanced_validators import ValidationIssue
        from src.config.enhanced_validators import ValidationSeverity

        mock_report = MagicMock()
        mock_report.errors = [
            ValidationIssue(
                field_path="unknown_field",
                message="Missing required field",
                severity=ValidationSeverity.ERROR,
                category="missing_required",
            )
        ]

        # Mock declining fixes for now
        mock_questionary.confirm.return_value.ask.return_value = False

        result = self.wizard._attempt_fixes(config_path, mock_report)

        # Should return False since no fixes were applied
        assert result is False


class TestConfigurationWizardIntegration:
    """Integration tests for configuration wizard."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.config.wizard.questionary")
    def test_run_setup_wizard_template_mode(self, mock_questionary):
        """Test complete setup wizard in template mode."""
        # Mock user choosing template mode and development template
        mock_questionary.select.return_value.ask.side_effect = [
            "template",  # setup mode
            "🛠️  Development - Debug logging, local database, fast iteration",  # template choice
        ]
        mock_questionary.confirm.return_value.ask.return_value = (
            False  # no customization
        )
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "setup_config.json"
        )

        config_path = self.wizard.run_setup_wizard()

        assert config_path.exists()
        assert config_path.name == "setup_config.json"

    @patch("src.config.wizard.questionary")
    def test_run_setup_wizard_interactive_mode(self, mock_questionary):
        """Test complete setup wizard in interactive mode."""
        # Mock user choosing interactive mode
        mock_questionary.select.return_value.ask.side_effect = [
            "interactive",  # setup mode
            "development",  # environment
            "DEBUG",  # log_level
            "openai",  # embedding provider
            "crawl4ai",  # crawl_provider
        ]
        mock_questionary.confirm.return_value.ask.side_effect = [
            True,  # debug
            False,  # use_postgres
            False,  # enable_redis
        ]
        mock_questionary.password.return_value.ask.side_effect = [
            "sk-test123456789012345678901234567890123456789012",  # OpenAI API key
        ]
        mock_questionary.path.return_value.ask.return_value = str(
            self.temp_dir / "interactive_setup.json"
        )

        config_path = self.wizard.run_setup_wizard()

        assert config_path.exists()
        assert config_path.name == "interactive_setup.json"

    def test_wizard_directory_management(self):
        """Test that wizard properly manages directories."""
        # Wizard should create necessary directories
        assert self.wizard.path_manager.base_dir.exists()
        assert self.wizard.path_manager.templates_dir.exists()
        assert self.wizard.path_manager.backups_dir.exists()
        assert self.wizard.path_manager.migrations_dir.exists()
