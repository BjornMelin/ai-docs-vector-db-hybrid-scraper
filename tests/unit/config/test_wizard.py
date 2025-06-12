"""Comprehensive tests for configuration wizard.

This test file covers the interactive configuration wizard that provides
guided setup, validation, backup, and restore functionality.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import pytest
from src.config.enhanced_validators import ValidationIssue
from src.config.enhanced_validators import ValidationReport
from src.config.enhanced_validators import ValidationSeverity
from src.config.models import UnifiedConfig
from src.config.wizard import ConfigurationWizard

# Mock questionary before importing wizard
sys.modules["questionary"] = MagicMock()


class TestConfigurationWizard:
    """Test the ConfigurationWizard class."""

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
            "embedding_provider": "fastembed",
            "crawl_provider": "crawl4ai",
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        return config_file

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wizard = ConfigurationWizard(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_wizard_initialization(self):
        """Test ConfigurationWizard initialization."""
        wizard = ConfigurationWizard(self.temp_dir)

        assert wizard.path_manager.base_dir == self.temp_dir
        assert hasattr(wizard, "console")
        assert hasattr(wizard, "templates")
        assert hasattr(wizard, "backup_manager")
        assert hasattr(wizard, "migration_manager")

        # Directories should be created
        assert wizard.path_manager.base_dir.exists()

    def test_wizard_default_initialization(self):
        """Test ConfigurationWizard with default base directory."""
        with patch("src.config.wizard.ConfigPathManager") as mock_path_manager:
            with patch("src.config.wizard.ConfigBackupManager"):
                with patch("src.config.wizard.ConfigMigrationManager"):
                    with patch("src.config.wizard.create_default_migrations"):
                        ConfigurationWizard()

                        # Should use default config directory
                        mock_path_manager.assert_called_with(Path("config"))

    @patch("src.config.wizard.questionary.select")
    def test_choose_setup_mode(self, mock_select):
        """Test setup mode selection."""
        mock_select.return_value.ask.return_value = "template"

        result = self.wizard._choose_setup_mode()

        assert result == "template"
        mock_select.assert_called_once()

        # Check the choices include all expected modes
        call_args = mock_select.call_args
        assert "template" in str(call_args)
        assert "interactive" in str(call_args)
        assert "migrate" in str(call_args)
        assert "import" in str(call_args)

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.confirm")
    @patch("src.config.wizard.questionary.path")
    def test_template_based_setup_basic(self, mock_path, mock_confirm, mock_select):
        """Test basic template-based setup."""
        # Mock user inputs
        mock_select.return_value.ask.return_value = (
            "🛠️  Development - Debug logging, local database, fast iteration"
        )
        mock_confirm.return_value.ask.return_value = False  # No customization
        mock_path.return_value.ask.return_value = "test_config.json"

        # Mock template application
        with patch.object(
            self.wizard.templates, "apply_template_to_config"
        ) as mock_apply:
            mock_apply.return_value = {
                "environment": "development",
                "debug": True,
                "log_level": "DEBUG",
            }

            with patch.object(UnifiedConfig, "save_to_file") as mock_save:
                result_path = self.wizard._template_based_setup(None)

                assert result_path == Path("test_config.json")
                mock_apply.assert_called_once_with("development")
                mock_save.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.confirm")
    @patch("src.config.wizard.questionary.path")
    def test_template_based_setup_with_customization(
        self, mock_path, mock_confirm, mock_select
    ):
        """Test template-based setup with customization."""
        # Mock user inputs for template selection
        mock_select.return_value.ask.return_value = (
            "🚀 Production - Security hardening, performance optimization"
        )
        mock_confirm.return_value.ask.return_value = True  # Enable customization
        mock_path.return_value.ask.return_value = "prod_config.json"

        # Mock template application
        with patch.object(
            self.wizard.templates, "apply_template_to_config"
        ) as mock_apply:
            mock_apply.return_value = {
                "environment": "production",
                "debug": False,
                "log_level": "WARNING",
            }

            with patch.object(self.wizard, "_customize_template") as mock_customize:
                mock_customize.return_value = {
                    "environment": "production",
                    "debug": False,
                    "log_level": "ERROR",  # Customized
                }

                with patch.object(UnifiedConfig, "save_to_file") as mock_save:
                    result_path = self.wizard._template_based_setup(None)

                    assert result_path == Path("prod_config.json")
                    mock_customize.assert_called_once()
                    mock_save.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.confirm")
    @patch("src.config.wizard.questionary.password")
    @patch("src.config.wizard.questionary.text")
    @patch("src.config.wizard.questionary.path")
    def test_interactive_setup_complete(
        self, mock_path, mock_text, mock_password, mock_confirm, mock_select
    ):
        """Test complete interactive setup."""
        # Mock all user inputs with proper chaining
        mock_select.return_value.ask.side_effect = [
            "production",  # environment
            "WARNING",  # log_level
            "openai",  # embedding_provider
            "firecrawl",  # crawl_provider
        ]
        mock_confirm.return_value.ask.side_effect = [
            False,  # debug
            True,  # use_postgres
            True,  # enable_redis
        ]
        mock_password.return_value.ask.side_effect = [
            "sk-test123456789012345678901234567890123456789012",  # OpenAI API key
            "fc-test123456789012345678901234567890123456789012345678",  # Firecrawl API key
        ]
        mock_text.return_value.ask.side_effect = [
            "postgresql+asyncpg://user:pass@localhost:5432/db",  # database URL
            "redis://localhost:6379",  # Redis URL
        ]
        mock_path.return_value.ask.return_value = "interactive_config.json"

        with patch.object(UnifiedConfig, "save_to_file") as mock_save:
            result_path = self.wizard._interactive_setup(None)

            assert result_path == Path("interactive_config.json")
            mock_save.assert_called_once()

            # Verify save was called with the correct path
            call_args = mock_save.call_args[0]
            assert call_args[0] == Path("interactive_config.json")

    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.text")
    @patch("src.config.wizard.questionary.confirm")
    def test_migration_setup_success(
        self, mock_confirm, mock_text, mock_path, sample_config_file
    ):
        """Test successful migration setup."""
        mock_path.return_value.ask.return_value = str(sample_config_file)
        mock_text.return_value.ask.return_value = "1.1.0"  # target version
        mock_confirm.return_value.ask.return_value = True  # proceed with migration

        # Mock migration manager
        with patch.object(
            self.wizard.migration_manager, "get_current_version"
        ) as mock_get_version:
            mock_get_version.return_value = "1.0.0"

            with patch.object(
                self.wizard.migration_manager, "create_migration_plan"
            ) as mock_create_plan:
                from src.config.migrations import MigrationPlan

                mock_plan = MigrationPlan(
                    source_version="1.0.0",
                    target_version="1.1.0",
                    migrations=["1.0.0_to_1.1.0"],
                    estimated_duration="~2 minutes",
                )
                mock_create_plan.return_value = mock_plan

                with patch.object(
                    self.wizard.migration_manager, "apply_migration_plan"
                ) as mock_apply:
                    from src.config.migrations import MigrationResult

                    mock_result = MigrationResult(
                        success=True,
                        migration_id="1.0.0_to_1.1.0",
                        from_version="1.0.0",
                        to_version="1.1.0",
                        changes_made=["Added config hash"],
                    )
                    mock_apply.return_value = [mock_result]

                    result_path = self.wizard._migration_setup(sample_config_file)

                    assert result_path == sample_config_file
                    mock_apply.assert_called_once()

    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.confirm")
    def test_import_setup_success(self, mock_confirm, mock_path, sample_config_file):
        """Test successful configuration import."""
        target_path = self.temp_dir / "imported_config.json"

        mock_path.return_value.ask.side_effect = [
            str(sample_config_file),  # import path
            str(target_path),  # output path
        ]
        mock_confirm.return_value.ask.return_value = True  # proceed despite issues

        # Mock validation
        with patch("src.config.wizard.ConfigurationValidator") as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator

            # Create a validation report with some warnings
            mock_report = ValidationReport(
                issues=[
                    ValidationIssue(
                        field_path="debug",
                        message="Debug enabled in production",
                        severity=ValidationSeverity.WARNING,
                        category="environment",
                    )
                ],
                is_valid=False,
                config_hash="test_hash_123",
            )
            mock_validator.validate_configuration.return_value = mock_report

            with patch.object(UnifiedConfig, "load_from_file") as mock_load:
                mock_config = UnifiedConfig(
                    environment="development", debug=True, log_level="DEBUG"
                )
                mock_load.return_value = mock_config

                with patch.object(UnifiedConfig, "save_to_file") as mock_save:
                    result_path = self.wizard._import_setup(None)

                    assert result_path == target_path
                    mock_save.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.confirm")
    @patch("src.config.wizard.questionary.password")
    def test_customize_template(self, mock_password, mock_confirm, mock_select):
        """Test template customization."""
        config_data = {
            "environment": "development",
            "debug": True,
            "embedding_provider": "openai",
            "crawl_provider": "firecrawl",
        }

        # Mock user inputs
        mock_select.return_value.ask.return_value = "production"  # change environment
        mock_confirm.return_value.ask.side_effect = [
            False,  # change debug (will be False)
            True,  # configure API keys
        ]
        mock_password.return_value.ask.side_effect = [
            "sk-test123456789012345678901234567890123456789012",  # OpenAI API key
            "fc-test123456789012345678901234567890123456789012345678",  # Firecrawl API key
        ]

        result = self.wizard._customize_template(config_data)

        assert result["environment"] == "production"
        assert result["debug"] is False
        assert (
            result["openai"]["api_key"]
            == "sk-test123456789012345678901234567890123456789012"
        )
        assert (
            result["firecrawl"]["api_key"]
            == "fc-test123456789012345678901234567890123456789012345678"
        )

    def test_display_validation_report(self):
        """Test validation report display."""
        report = ValidationReport(
            issues=[
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
            ],
            is_valid=False,
            config_hash="test_hash_123",
        )

        # Mock console to capture output
        with patch.object(self.wizard.console, "print") as mock_print:
            self.wizard._display_validation_report(report)

            # Should print errors, warnings, and info
            assert mock_print.call_count >= 6  # Headers + messages

            # Check that different severities are handled
            call_args_list = [call[0][0] for call in mock_print.call_args_list]
            assert any("❌ Errors:" in str(arg) for arg in call_args_list)
            assert any("⚠️  Warnings:" in str(arg) for arg in call_args_list)
            assert any("i  Information:" in str(arg) for arg in call_args_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.config.wizard.json.load")
    @patch("src.config.wizard.json.dump")
    @patch("src.config.wizard.questionary.password")
    def test_attempt_fixes_success(
        self,
        mock_password,
        mock_json_dump,
        mock_json_load,
        mock_file,
        sample_config_file,
    ):
        """Test successful automatic fixes."""
        # Mock configuration data
        config_data = {
            "environment": "production",
            "debug": True,  # This should be fixed
            "openai": {},  # Missing API key
        }
        mock_json_load.return_value = config_data
        mock_password.return_value.ask.return_value = (
            "sk-test123456789012345678901234567890123456789012"
        )

        # Create validation report with fixable issues
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    field_path="debug",
                    message="Debug should be disabled in production",
                    severity=ValidationSeverity.ERROR,
                    category="environment",
                ),
                ValidationIssue(
                    field_path="openai.api_key",
                    message="API key is required",
                    severity=ValidationSeverity.ERROR,
                    category="authentication",
                ),
            ],
            is_valid=False,
            config_hash="test_hash_123",
        )

        # Mock backup creation
        with patch.object(self.wizard.backup_manager, "create_backup") as mock_backup:
            mock_backup.return_value = "backup_123"

            # Mock validation after fixes
            with patch("src.config.wizard.UnifiedConfig.load_from_file"):
                with patch(
                    "src.config.wizard.ConfigurationValidator"
                ) as mock_validator_class:
                    mock_validator = MagicMock()
                    mock_validator_class.return_value = mock_validator
                    mock_validator.validate_configuration.return_value = (
                        ValidationReport(
                            issues=[], is_valid=True, config_hash="test_hash"
                        )
                    )

                    result = self.wizard._attempt_fixes(sample_config_file, report)

                    assert result is True
                    mock_backup.assert_called_once()
                    mock_json_dump.assert_called_once()

                    # Check that fixes were applied
                    saved_config = mock_json_dump.call_args[0][0]
                    assert saved_config["debug"] is False
                    assert (
                        saved_config["openai"]["api_key"]
                        == "sk-test123456789012345678901234567890123456789012"
                    )

    @patch("src.config.wizard.questionary.text")
    @patch("src.config.wizard.questionary.confirm")
    def test_run_backup_wizard_success(
        self, mock_confirm, mock_text, sample_config_file
    ):
        """Test successful backup wizard."""
        mock_text.return_value.ask.side_effect = [
            "Test backup description",  # description
            "manual,important",  # tags
        ]
        mock_confirm.return_value.ask.return_value = True  # compress

        with patch.object(self.wizard.backup_manager, "create_backup") as mock_backup:
            mock_backup.return_value = "backup_test_123"

            result = self.wizard.run_backup_wizard(sample_config_file)

            assert result == "backup_test_123"
            mock_backup.assert_called_once_with(
                sample_config_file,
                description="Test backup description",
                tags=["manual", "important"],
                compress=True,
            )

    @patch("src.config.wizard.questionary.select")
    @patch("src.config.wizard.questionary.path")
    @patch("src.config.wizard.questionary.confirm")
    def test_run_restore_wizard_success(self, mock_confirm, mock_path, mock_select):
        """Test successful restore wizard."""
        # Mock backup list
        from src.config.backup_restore import BackupMetadata

        mock_backup = BackupMetadata(
            backup_id="backup_123456789012",
            config_name="test_config",
            config_hash="test_hash_123",
            created_at="2023-01-01T12:00:00Z",
            file_size=1024,
            environment="development",
        )

        with patch.object(self.wizard.backup_manager, "list_backups") as mock_list:
            mock_list.return_value = [mock_backup]

            # Mock user selections
            mock_select.return_value.ask.return_value = "backup_12345... - test_config"
            mock_path.return_value.ask.return_value = "restored_config.json"
            mock_confirm.return_value.ask.return_value = True  # create backup

            # Mock restore operation
            with patch.object(
                self.wizard.backup_manager, "restore_backup"
            ) as mock_restore:
                from src.config.backup_restore import RestoreResult

                mock_result = RestoreResult(
                    success=True,
                    backup_id="backup_123456789012",
                    config_path=Path("restored_config.json"),
                    pre_restore_backup="pre_backup_123",
                )
                mock_restore.return_value = mock_result

                result = self.wizard.run_restore_wizard()

                assert result is True
                mock_restore.assert_called_once()

    @patch("src.config.wizard.questionary.confirm")
    def test_run_validation_wizard_valid_config(self, mock_confirm, sample_config_file):
        """Test validation wizard with valid configuration."""
        with patch.object(UnifiedConfig, "load_from_file") as mock_load:
            mock_config = UnifiedConfig(
                environment="development", debug=True, log_level="DEBUG"
            )
            mock_load.return_value = mock_config

            with patch(
                "src.config.wizard.ConfigurationValidator"
            ) as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator_class.return_value = mock_validator
                mock_validator.validate_configuration.return_value = ValidationReport(
                    issues=[], is_valid=True, config_hash="test_hash"
                )

                result = self.wizard.run_validation_wizard(sample_config_file)

                assert result is True

    @patch("src.config.wizard.questionary.confirm")
    def test_run_validation_wizard_with_fixes(self, mock_confirm, sample_config_file):
        """Test validation wizard that applies fixes."""
        mock_confirm.return_value.ask.return_value = True  # apply fixes

        with patch.object(UnifiedConfig, "load_from_file") as mock_load:
            mock_config = UnifiedConfig(
                environment="development", debug=True, log_level="DEBUG"
            )
            mock_load.return_value = mock_config

            # Mock validation with errors
            with patch(
                "src.config.wizard.ConfigurationValidator"
            ) as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator_class.return_value = mock_validator

                error_report = ValidationReport(
                    issues=[
                        ValidationIssue(
                            field_path="debug",
                            message="Debug should be disabled in production",
                            severity=ValidationSeverity.ERROR,
                            category="environment",
                        )
                    ],
                    is_valid=False,
                    config_hash="test_hash_123",
                )
                mock_validator.validate_configuration.return_value = error_report

                with patch.object(self.wizard, "_attempt_fixes") as mock_attempt_fixes:
                    mock_attempt_fixes.return_value = True

                    result = self.wizard.run_validation_wizard(sample_config_file)

                    assert result is True
                    mock_attempt_fixes.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    def test_run_setup_wizard_template_mode(self, mock_select):
        """Test setup wizard in template mode."""
        mock_select.return_value.ask.return_value = "template"

        with patch.object(self.wizard, "_template_based_setup") as mock_template_setup:
            mock_template_setup.return_value = Path("template_config.json")

            result = self.wizard.run_setup_wizard()

            assert result == Path("template_config.json")
            mock_template_setup.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    def test_run_setup_wizard_interactive_mode(self, mock_select):
        """Test setup wizard in interactive mode."""
        mock_select.return_value.ask.return_value = "interactive"

        with patch.object(self.wizard, "_interactive_setup") as mock_interactive_setup:
            mock_interactive_setup.return_value = Path("interactive_config.json")

            result = self.wizard.run_setup_wizard()

            assert result == Path("interactive_config.json")
            mock_interactive_setup.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    def test_run_setup_wizard_migrate_mode(self, mock_select):
        """Test setup wizard in migrate mode."""
        mock_select.return_value.ask.return_value = "migrate"

        with patch.object(self.wizard, "_migration_setup") as mock_migration_setup:
            mock_migration_setup.return_value = Path("migrated_config.json")

            result = self.wizard.run_setup_wizard()

            assert result == Path("migrated_config.json")
            mock_migration_setup.assert_called_once()

    @patch("src.config.wizard.questionary.select")
    def test_run_setup_wizard_import_mode(self, mock_select):
        """Test setup wizard in import mode."""
        mock_select.return_value.ask.return_value = "import"

        with patch.object(self.wizard, "_import_setup") as mock_import_setup:
            mock_import_setup.return_value = Path("imported_config.json")

            result = self.wizard.run_setup_wizard()

            assert result == Path("imported_config.json")
            mock_import_setup.assert_called_once()

    def test_run_backup_wizard_failure(self, sample_config_file):
        """Test backup wizard when backup fails."""
        with patch.object(self.wizard.backup_manager, "create_backup") as mock_backup:
            mock_backup.side_effect = Exception("Backup failed")

            with patch("src.config.wizard.questionary.text") as mock_text:
                with patch("src.config.wizard.questionary.confirm") as mock_confirm:
                    mock_text.return_value.ask.return_value = ""
                    mock_confirm.return_value.ask.return_value = False

                    result = self.wizard.run_backup_wizard(sample_config_file)

                    assert result is None

    def test_run_restore_wizard_no_backups(self):
        """Test restore wizard when no backups exist."""
        with patch.object(self.wizard.backup_manager, "list_backups") as mock_list:
            mock_list.return_value = []

            result = self.wizard.run_restore_wizard()

            assert result is False

    def test_run_validation_wizard_load_failure(self):
        """Test validation wizard when config loading fails."""
        with patch.object(UnifiedConfig, "load_from_file") as mock_load:
            mock_load.side_effect = Exception("Failed to load config")

            result = self.wizard.run_validation_wizard(Path("nonexistent.json"))

            assert result is False

    def test_migration_setup_no_migration_path(self, sample_config_file):
        """Test migration setup when no migration path exists."""
        with patch("src.config.wizard.questionary.path") as mock_path:
            with patch("src.config.wizard.questionary.text") as mock_text:
                mock_path.return_value.ask.return_value = str(sample_config_file)
                mock_text.return_value.ask.return_value = "3.0.0"  # No path to 3.0.0

                with patch.object(
                    self.wizard.migration_manager, "get_current_version"
                ) as mock_get_version:
                    mock_get_version.return_value = "1.0.0"

                    with patch.object(
                        self.wizard.migration_manager, "create_migration_plan"
                    ) as mock_create_plan:
                        mock_create_plan.return_value = None  # No migration path

                        result = self.wizard._migration_setup(sample_config_file)

                        assert result == sample_config_file  # Returns original path

    def test_import_setup_validation_failure(self, sample_config_file):
        """Test import setup when validation fails and user cancels."""
        target_path = self.temp_dir / "imported_config.json"

        with patch("src.config.wizard.questionary.path") as mock_path:
            with patch("src.config.wizard.questionary.confirm") as mock_confirm:
                mock_path.return_value.ask.side_effect = [
                    str(sample_config_file),
                    str(target_path),
                ]
                mock_confirm.return_value.ask.return_value = False  # Don't proceed

                with patch(
                    "src.config.wizard.ConfigurationValidator"
                ) as mock_validator_class:
                    mock_validator = MagicMock()
                    mock_validator_class.return_value = mock_validator

                    # Validation fails
                    error_report = ValidationReport(
                        issues=[
                            ValidationIssue(
                                field_path="test",
                                message="Test error",
                                severity=ValidationSeverity.ERROR,
                                category="validation",
                            )
                        ],
                        is_valid=False,
                        config_hash="test_hash_123",
                    )
                    mock_validator.validate_configuration.return_value = error_report

                    with patch.object(UnifiedConfig, "load_from_file") as mock_load:
                        mock_load.return_value = UnifiedConfig(
                            environment="development", debug=True, log_level="DEBUG"
                        )

                        result = self.wizard._import_setup(None)

                        assert (
                            result == target_path
                        )  # Returns target path even on cancel

    def test_attempt_fixes_no_fixes_available(self, sample_config_file):
        """Test attempt fixes when no automatic fixes are available."""
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    field_path="unknown_field",
                    message="Unknown validation error",
                    severity=ValidationSeverity.ERROR,
                    category="validation",
                )
            ],
            is_valid=False,
            config_hash="test_hash_123",
        )

        with patch.object(self.wizard.backup_manager, "create_backup") as mock_backup:
            mock_backup.return_value = "backup_123"

            with patch("builtins.open", mock_open(read_data='{"test": "config"}')):
                with patch("src.config.wizard.json.load") as mock_json_load:
                    mock_json_load.return_value = {"test": "config"}

                    result = self.wizard._attempt_fixes(sample_config_file, report)

                    assert result is False

    def test_error_handling_template_application_failure(self):
        """Test error handling when template application fails."""
        with patch.object(
            self.wizard.templates, "apply_template_to_config"
        ) as mock_apply:
            mock_apply.return_value = None  # Template loading failed

            with patch("src.config.wizard.questionary.select") as mock_select:
                with patch("src.config.wizard.questionary.confirm") as mock_confirm:
                    mock_select.return_value.ask.return_value = (
                        "🛠️  Development - Debug logging, local database, fast iteration"
                    )
                    mock_confirm.return_value.ask.return_value = False

                    with pytest.raises(ValueError, match="Failed to load template"):
                        self.wizard._template_based_setup(None)

    @patch("src.config.wizard.questionary.select")
    def test_template_based_setup_no_template_selected(self, mock_select):
        """Test template setup when no template is selected."""
        mock_select.return_value.ask.return_value = None  # No selection

        with pytest.raises(ValueError, match="No template selected"):
            self.wizard._template_based_setup(None)

    def test_migration_setup_file_not_found(self):
        """Test migration setup with non-existent file."""
        nonexistent_file = Path("/nonexistent/config.json")

        with pytest.raises(FileNotFoundError):
            self.wizard._migration_setup(nonexistent_file)
