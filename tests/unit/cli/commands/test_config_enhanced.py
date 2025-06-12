"""Comprehensive tests for enhanced configuration CLI commands.

This test file covers all the new configuration management CLI commands
including templates, wizard, backup, and migration functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from src.cli.commands.config import _mask_sensitive_data
from src.cli.commands.config import apply_migration
from src.cli.commands.config import apply_template
from src.cli.commands.config import backup
from src.cli.commands.config import config
from src.cli.commands.config import convert_config
from src.cli.commands.config import create_backup
from src.cli.commands.config import create_example
from src.cli.commands.config import list_backups
from src.cli.commands.config import list_templates
from src.cli.commands.config import migrate
from src.cli.commands.config import migration_plan
from src.cli.commands.config import migration_status
from src.cli.commands.config import restore_backup
from src.cli.commands.config import rollback_migration
from src.cli.commands.config import show_config
from src.cli.commands.config import template
from src.cli.commands.config import validate_config
from src.cli.commands.config import wizard


class TestConfigCommand:
    """Test the main config command group."""

    def test_config_group_exists(self):
        """Test that config command group exists."""
        assert isinstance(config, click.Group)
        assert config.name == "config"


class TestCreateExampleCommand:
    """Test the create-example command."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        mock_cli.console.print = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    def test_create_example_success(self, mock_context):
        """Test successful example configuration creation."""
        runner = CliRunner()

        with patch("src.config.cli.create_example") as mock_create:
            result = runner.invoke(
                create_example,
                ["--format", "json", "--template", "development"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_create.assert_called_once_with("json", "config.json")

    def test_create_example_with_output_path(self, mock_context):
        """Test create example with custom output path."""
        runner = CliRunner()

        with patch("src.config.cli.create_example") as mock_create:
            result = runner.invoke(
                create_example,
                ["--output", "/custom/config.json", "--template", "production"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_create.assert_called_once_with("json", "/custom/config.json")

    def test_create_example_failure(self, mock_context):
        """Test create example command failure."""
        runner = CliRunner()

        with patch("src.config.cli.create_example") as mock_create:
            mock_create.side_effect = Exception("Creation failed")

            result = runner.invoke(
                create_example, ["--template", "development"], obj=mock_context.obj
            )

            assert result.exit_code == 1  # click.Abort()


class TestValidateConfigCommand:
    """Test the validate command."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        temp_dir = Path(tempfile.mkdtemp())
        config_file = temp_dir / "test_config.json"

        config_data = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"api_key": "test_key"},
            "fastembed": {"enabled": True, "model": "test_model"},
            "cache": {"redis": {"enabled": True, "host": "localhost", "port": 6379}},
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        yield config_file
        shutil.rmtree(temp_dir)

    def test_validate_config_with_file(self, mock_context, temp_config_file):
        """Test validate command with specific config file."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigLoader") as mock_loader:
            mock_config = MagicMock()
            mock_config.qdrant.host = "localhost"
            mock_config.qdrant.port = 6333
            mock_config.openai.api_key = "test_key"
            mock_config.fastembed.enabled = True
            mock_config.fastembed.model = "test_model"
            mock_config.cache.redis.enabled = True
            mock_config.cache.redis.host = "localhost"
            mock_config.cache.redis.port = 6379

            mock_loader.from_file.return_value = mock_config

            result = runner.invoke(
                validate_config, [str(temp_config_file)], obj=mock_context.obj
            )

            assert result.exit_code == 0
            mock_loader.from_file.assert_called_once_with(temp_config_file)

    def test_validate_config_with_health_check(self, mock_context, temp_config_file):
        """Test validate command with health checks."""
        runner = CliRunner()

        with (
            patch("src.cli.commands.config.ConfigLoader") as mock_loader,
            patch("src.cli.commands.config.ServiceHealthChecker") as mock_health,
        ):
            mock_config = MagicMock()
            mock_config.qdrant.host = "localhost"
            mock_config.qdrant.port = 6333
            mock_config.openai.api_key = "test_key"
            mock_config.fastembed.enabled = True
            mock_config.fastembed.model = "test_model"
            mock_config.cache.redis.enabled = False

            mock_loader.from_file.return_value = mock_config

            health_results = {
                "qdrant": {"connected": True},
                "openai": {"connected": False, "error": "Invalid API key"},
            }
            mock_health.perform_all_health_checks.return_value = health_results

            result = runner.invoke(
                validate_config,
                [str(temp_config_file), "--health-check"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_health.perform_all_health_checks.assert_called_once()

    def test_validate_config_default_loading(self, mock_context):
        """Test validate command with default config loading."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigLoader") as mock_loader:
            mock_config = MagicMock()
            mock_config.qdrant.host = "localhost"
            mock_config.qdrant.port = 6333
            mock_config.openai.api_key = None
            mock_config.fastembed.enabled = False
            mock_config.cache.redis.enabled = False

            mock_loader.load_config.return_value = mock_config

            result = runner.invoke(validate_config, obj=mock_context.obj)

            assert result.exit_code == 0
            mock_loader.load_config.assert_called_once()

    def test_validate_config_failure(self, mock_context):
        """Test validate command failure."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigLoader") as mock_loader:
            mock_loader.load_config.side_effect = Exception("Config loading failed")

            result = runner.invoke(validate_config, obj=mock_context.obj)

            assert result.exit_code == 1


class TestShowConfigCommand:
    """Test the show command."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        mock_config = MagicMock()
        mock_config.qdrant.host = "localhost"
        mock_config.qdrant.port = 6333
        mock_config.openai.api_key = "test_key"
        mock_config.openai.model = "gpt-3.5-turbo"
        mock_config.fastembed.enabled = True
        mock_config.fastembed.model = "test_model"
        mock_config.cache.redis.enabled = True
        mock_config.cache.redis.host = "localhost"
        mock_config.cache.redis.port = 6379

        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "rich_cli": mock_rich_cli}
        return ctx

    def test_show_config_table_format(self, mock_context):
        """Test show config in table format."""
        runner = CliRunner()

        result = runner.invoke(show_config, ["--format", "table"], obj=mock_context.obj)

        assert result.exit_code == 0
        # Should call console.print for table
        mock_context.obj["rich_cli"].console.print.assert_called()

    def test_show_config_specific_section(self, mock_context):
        """Test show config for specific section."""
        runner = CliRunner()

        # Mock the specific section
        mock_qdrant = MagicMock()
        mock_qdrant.model_dump.return_value = {"host": "localhost", "port": 6333}
        mock_context.obj["config"].qdrant = mock_qdrant

        result = runner.invoke(
            show_config,
            ["--format", "table", "--section", "qdrant"],
            obj=mock_context.obj,
        )

        assert result.exit_code == 0
        mock_qdrant.model_dump.assert_called_once()

    def test_show_config_json_format(self, mock_context):
        """Test show config in JSON format."""
        runner = CliRunner()

        mock_context.obj["config"].model_dump.return_value = {"test": "data"}

        result = runner.invoke(show_config, ["--format", "json"], obj=mock_context.obj)

        assert result.exit_code == 0
        mock_context.obj["config"].model_dump.assert_called_once()

    def test_show_config_yaml_format(self, mock_context):
        """Test show config in YAML format."""
        runner = CliRunner()

        mock_context.obj["config"].model_dump.return_value = {"test": "data"}

        result = runner.invoke(show_config, ["--format", "yaml"], obj=mock_context.obj)

        assert result.exit_code == 0


class TestConvertConfigCommand:
    """Test the convert command."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    @pytest.fixture
    def temp_files(self):
        """Create temporary input and output files."""
        temp_dir = Path(tempfile.mkdtemp())
        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.yaml"

        # Create input file
        with open(input_file, "w") as f:
            json.dump({"test": "data"}, f)

        yield input_file, output_file
        shutil.rmtree(temp_dir)

    def test_convert_config_success(self, mock_context, temp_files):
        """Test successful config conversion."""
        input_file, output_file = temp_files
        runner = CliRunner()

        with patch("src.config.cli.convert") as mock_convert:
            result = runner.invoke(
                convert_config,
                [str(input_file), str(output_file), "--format", "yaml"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_convert.assert_called_once_with(
                str(input_file), str(output_file), None, "yaml"
            )

    def test_convert_config_failure(self, mock_context, temp_files):
        """Test config conversion failure."""
        input_file, output_file = temp_files
        runner = CliRunner()

        with patch("src.config.cli.convert") as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")

            result = runner.invoke(
                convert_config,
                [str(input_file), str(output_file)],
                obj=mock_context.obj,
            )

            assert result.exit_code == 1


class TestTemplateCommands:
    """Test template management commands."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    def test_template_group_exists(self):
        """Test that template command group exists."""
        assert isinstance(template, click.Group)

    def test_list_templates_success(self, mock_context):
        """Test successful template listing."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigurationTemplates"
        ) as mock_templates_class:
            mock_templates = MagicMock()
            mock_templates.list_available_templates.return_value = [
                "development",
                "production",
                "high_performance",
            ]
            mock_templates_class.return_value = mock_templates

            result = runner.invoke(list_templates, obj=mock_context.obj)

            assert result.exit_code == 0
            mock_templates.list_available_templates.assert_called_once()

    def test_list_templates_empty(self, mock_context):
        """Test template listing when no templates available."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigurationTemplates"
        ) as mock_templates_class:
            mock_templates = MagicMock()
            mock_templates.list_available_templates.return_value = []
            mock_templates_class.return_value = mock_templates

            result = runner.invoke(list_templates, obj=mock_context.obj)

            assert result.exit_code == 0

    def test_apply_template_success(self, mock_context):
        """Test successful template application."""
        runner = CliRunner()

        with (
            patch(
                "src.cli.commands.config.ConfigurationTemplates"
            ) as mock_templates_class,
            patch("src.cli.commands.config.UnifiedConfig") as mock_config_class,
        ):
            mock_templates = MagicMock()
            mock_templates.apply_template_to_config.return_value = {
                "environment": "development",
                "debug": True,
            }
            mock_templates_class.return_value = mock_templates

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            result = runner.invoke(
                apply_template,
                ["development", "--output", "dev_config.json"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_templates.apply_template_to_config.assert_called_once_with(
                "development", environment_overrides=None
            )
            mock_config.save_to_file.assert_called_once()

    def test_apply_template_not_found(self, mock_context):
        """Test template application with non-existent template."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigurationTemplates"
        ) as mock_templates_class:
            mock_templates = MagicMock()
            mock_templates.apply_template_to_config.return_value = None
            mock_templates_class.return_value = mock_templates

            result = runner.invoke(
                apply_template, ["nonexistent"], obj=mock_context.obj
            )

            assert result.exit_code == 1

    def test_apply_template_with_environment_override(self, mock_context):
        """Test template application with environment override."""
        runner = CliRunner()

        with (
            patch(
                "src.cli.commands.config.ConfigurationTemplates"
            ) as mock_templates_class,
            patch("src.cli.commands.config.UnifiedConfig") as mock_config_class,
        ):
            mock_templates = MagicMock()
            mock_templates.apply_template_to_config.return_value = {"test": "config"}
            mock_templates_class.return_value = mock_templates

            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            result = runner.invoke(
                apply_template,
                ["development", "--environment-override", "staging"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_templates.apply_template_to_config.assert_called_once_with(
                "development", environment_overrides="staging"
            )


class TestWizardCommand:
    """Test the wizard command."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    def test_wizard_success(self, mock_context):
        """Test successful wizard execution."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup_wizard.return_value = Path("wizard_config.json")
            mock_wizard_class.return_value = mock_wizard

            result = runner.invoke(wizard, obj=mock_context.obj)

            assert result.exit_code == 0
            mock_wizard.run_setup_wizard.assert_called_once_with(None)

    def test_wizard_with_config_path(self, mock_context):
        """Test wizard with specific config path."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup_wizard.return_value = Path("custom_config.json")
            mock_wizard_class.return_value = mock_wizard

            result = runner.invoke(
                wizard, ["--config-path", "custom_config.json"], obj=mock_context.obj
            )

            assert result.exit_code == 0
            mock_wizard.run_setup_wizard.assert_called_once_with(
                Path("custom_config.json")
            )

    def test_wizard_failure(self, mock_context):
        """Test wizard execution failure."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup_wizard.side_effect = Exception("Wizard failed")
            mock_wizard_class.return_value = mock_wizard

            result = runner.invoke(wizard, obj=mock_context.obj)

            assert result.exit_code == 1


class TestBackupCommands:
    """Test backup management commands."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        temp_dir = Path(tempfile.mkdtemp())
        config_file = temp_dir / "test_config.json"

        with open(config_file, "w") as f:
            json.dump({"test": "config"}, f)

        yield config_file
        shutil.rmtree(temp_dir)

    def test_backup_group_exists(self):
        """Test that backup command group exists."""
        assert isinstance(backup, click.Group)

    def test_create_backup_success(self, mock_context, temp_config_file):
        """Test successful backup creation."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.create_backup.return_value = "backup_123"
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                create_backup,
                [
                    str(temp_config_file),
                    "--description",
                    "Test backup",
                    "--tags",
                    "test,manual",
                ],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_manager.create_backup.assert_called_once_with(
                temp_config_file,
                description="Test backup",
                tags=["test", "manual"],
                compress=True,
            )

    def test_create_backup_no_compression(self, mock_context, temp_config_file):
        """Test backup creation without compression."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.create_backup.return_value = "backup_456"
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                create_backup,
                [str(temp_config_file), "--no-compress"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_manager.create_backup.assert_called_once_with(
                temp_config_file, description=None, tags=[], compress=False
            )

    def test_list_backups_success(self, mock_context):
        """Test successful backup listing."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()

            # Mock backup metadata
            from src.config.backup_restore import BackupMetadata

            mock_backup = BackupMetadata(
                backup_id="backup_123456789012",
                config_name="test_config",
                config_hash="abc123",
                created_at="2023-01-01T12:00:00Z",
                file_size=1024000,
                environment="development",
                description="Test backup",
            )

            mock_manager.list_backups.return_value = [mock_backup]
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                list_backups,
                ["--config-name", "test_config", "--limit", "10"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_manager.list_backups.assert_called_once_with(
                config_name="test_config", environment=None, tags=None, limit=10
            )

    def test_list_backups_empty(self, mock_context):
        """Test backup listing when no backups exist."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.list_backups.return_value = []
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(list_backups, obj=mock_context.obj)

            assert result.exit_code == 0

    def test_restore_backup_success(self, mock_context):
        """Test successful backup restoration."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()

            from src.config.backup_restore import RestoreResult

            mock_result = RestoreResult(
                success=True,
                backup_id="backup_123",
                config_path=Path("restored_config.json"),
                pre_restore_backup="pre_backup_123",
            )
            mock_manager.restore_backup.return_value = mock_result
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                restore_backup,
                ["backup_123", "--target", "restored_config.json"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_manager.restore_backup.assert_called_once()

    def test_restore_backup_failure(self, mock_context):
        """Test backup restoration failure."""
        runner = CliRunner()

        with patch("src.cli.commands.config.ConfigBackupManager") as mock_manager_class:
            mock_manager = MagicMock()

            from src.config.backup_restore import RestoreResult

            mock_result = RestoreResult(
                success=False,
                backup_id="backup_123",
                config_path=Path("failed_config.json"),
                conflicts=["Version mismatch"],
                warnings=["Deprecated settings found"],
            )
            mock_manager.restore_backup.return_value = mock_result
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(restore_backup, ["backup_123"], obj=mock_context.obj)

            assert result.exit_code == 1


class TestMigrationCommands:
    """Test migration management commands."""

    @pytest.fixture
    def mock_rich_cli(self):
        """Mock rich CLI object."""
        mock_cli = MagicMock()
        return mock_cli

    @pytest.fixture
    def mock_context(self, mock_rich_cli):
        """Mock click context."""
        ctx = MagicMock()
        ctx.obj = {"rich_cli": mock_rich_cli}
        return ctx

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        temp_dir = Path(tempfile.mkdtemp())
        config_file = temp_dir / "test_config.json"

        with open(config_file, "w") as f:
            json.dump({"_migration_version": "1.0.0"}, f)

        yield config_file
        shutil.rmtree(temp_dir)

    def test_migrate_group_exists(self):
        """Test that migrate command group exists."""
        assert isinstance(migrate, click.Group)

    def test_migration_plan_success(self, mock_context, temp_config_file):
        """Test successful migration plan creation."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_current_version.return_value = "1.0.0"

            from src.config.migrations import MigrationPlan

            mock_plan = MigrationPlan(
                source_version="1.0.0",
                target_version="1.1.0",
                migrations=["1.0.0_to_1.1.0"],
                estimated_duration="~2 minutes",
                requires_downtime=False,
                rollback_plan=["1.1.0_to_1.0.0"],
            )
            mock_manager.create_migration_plan.return_value = mock_plan
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                migration_plan, [str(temp_config_file), "1.1.0"], obj=mock_context.obj
            )

            assert result.exit_code == 0
            mock_manager.create_migration_plan.assert_called_once_with("1.0.0", "1.1.0")

    def test_migration_plan_no_path(self, mock_context, temp_config_file):
        """Test migration plan when no path exists."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_current_version.return_value = "1.0.0"
            mock_manager.create_migration_plan.return_value = None
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                migration_plan, [str(temp_config_file), "3.0.0"], obj=mock_context.obj
            )

            assert result.exit_code == 1

    def test_apply_migration_success(self, mock_context, temp_config_file):
        """Test successful migration application."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_current_version.return_value = "1.0.0"

            from src.config.migrations import MigrationPlan
            from src.config.migrations import MigrationResult

            mock_plan = MigrationPlan(
                source_version="1.0.0",
                target_version="1.1.0",
                migrations=["1.0.0_to_1.1.0"],
                estimated_duration="~2 minutes",
            )
            mock_manager.create_migration_plan.return_value = mock_plan

            mock_result = MigrationResult(
                success=True,
                migration_id="1.0.0_to_1.1.0",
                from_version="1.0.0",
                to_version="1.1.0",
                changes_made=["Added config hash"],
            )
            mock_manager.apply_migration_plan.return_value = [mock_result]
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                apply_migration, [str(temp_config_file), "1.1.0"], obj=mock_context.obj
            )

            assert result.exit_code == 0
            mock_manager.apply_migration_plan.assert_called_once()

    def test_apply_migration_dry_run(self, mock_context, temp_config_file):
        """Test migration application in dry run mode."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_current_version.return_value = "1.0.0"

            from src.config.migrations import MigrationPlan
            from src.config.migrations import MigrationResult

            mock_plan = MigrationPlan(
                source_version="1.0.0",
                target_version="1.1.0",
                migrations=["1.0.0_to_1.1.0"],
                estimated_duration="~2 minutes",
            )
            mock_manager.create_migration_plan.return_value = mock_plan

            mock_result = MigrationResult(
                success=True,
                migration_id="1.0.0_to_1.1.0",
                from_version="1.0.0",
                to_version="1.1.0",
                changes_made=["DRY RUN: Would add config hash"],
            )
            mock_manager.apply_migration_plan.return_value = [mock_result]
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                apply_migration,
                [str(temp_config_file), "1.1.0", "--dry-run"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            # Check that dry_run=True was passed
            call_args = mock_manager.apply_migration_plan.call_args
            assert call_args[1]["dry_run"] is True

    def test_rollback_migration_success(self, mock_context, temp_config_file):
        """Test successful migration rollback."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()

            from src.config.migrations import MigrationResult

            mock_result = MigrationResult(
                success=True,
                migration_id="1.0.0_to_1.1.0",
                from_version="1.1.0",
                to_version="1.0.0",
                changes_made=["Removed config hash"],
            )
            mock_manager.rollback_migration.return_value = mock_result
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                rollback_migration,
                [str(temp_config_file), "1.0.0_to_1.1.0"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 0
            mock_manager.rollback_migration.assert_called_once()

    def test_rollback_migration_failure(self, mock_context, temp_config_file):
        """Test migration rollback failure."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()

            from src.config.migrations import MigrationResult

            mock_result = MigrationResult(
                success=False,
                migration_id="1.0.0_to_1.1.0",
                from_version="1.1.0",
                to_version="1.0.0",
                errors=["No rollback function available"],
            )
            mock_manager.rollback_migration.return_value = mock_result
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                rollback_migration,
                [str(temp_config_file), "1.0.0_to_1.1.0"],
                obj=mock_context.obj,
            )

            assert result.exit_code == 1

    def test_migration_status(self, mock_context, temp_config_file):
        """Test migration status command."""
        runner = CliRunner()

        with patch(
            "src.cli.commands.config.ConfigMigrationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_current_version.return_value = "1.0.0"

            from src.config.migrations import MigrationMetadata

            mock_migration = MigrationMetadata(
                migration_id="1.0.0_to_1.1.0",
                from_version="1.0.0",
                to_version="1.1.0",
                description="Add config hash",
                created_at="2023-01-01T12:00:00Z",
            )

            mock_manager.list_available_migrations.return_value = [mock_migration]
            mock_manager.list_applied_migrations.return_value = []
            mock_manager_class.return_value = mock_manager

            result = runner.invoke(
                migration_status, [str(temp_config_file)], obj=mock_context.obj
            )

            assert result.exit_code == 0
            mock_manager.get_current_version.assert_called_once()
            mock_manager.list_available_migrations.assert_called_once()
            mock_manager.list_applied_migrations.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_mask_sensitive_data_basic(self):
        """Test basic sensitive data masking."""
        data = {
            "api_key": "secret123",
            "password": "mypassword",
            "secret": "topsecret",
            "normal_field": "visible",
        }

        result = _mask_sensitive_data(data)

        assert result["api_key"] == "***"
        assert result["password"] == "***"
        assert result["secret"] == "***"
        assert result["normal_field"] == "visible"

    def test_mask_sensitive_data_nested(self):
        """Test sensitive data masking with nested dictionaries."""
        data = {
            "database": {"host": "localhost", "password": "dbpass"},
            "api": {"key": "apikey123", "url": "https://api.example.com"},
        }

        result = _mask_sensitive_data(data)

        assert result["database"]["host"] == "localhost"
        assert result["database"]["password"] == "***"
        assert result["api"]["key"] == "***"
        assert result["api"]["url"] == "https://api.example.com"

    def test_mask_sensitive_data_empty_values(self):
        """Test sensitive data masking with empty values."""
        data = {"api_key": "", "password": None, "normal_field": "value"}

        result = _mask_sensitive_data(data)

        assert result["api_key"] is None  # Empty string becomes None
        assert result["password"] is None
        assert result["normal_field"] == "value"

    def test_mask_sensitive_data_non_dict(self):
        """Test sensitive data masking with non-dictionary input."""
        data = "not a dictionary"

        result = _mask_sensitive_data(data)

        assert result == "not a dictionary"
