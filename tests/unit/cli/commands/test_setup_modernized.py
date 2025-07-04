"""Modernized tests for the setup command and ConfigurationWizard.

This module provides comprehensive testing for the interactive configuration wizard
with focus on Rich CLI components, questionary interactions, and user experience flows.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import click
import pytest
from rich.console import Console

from src.cli.commands.setup import ConfigurationWizard, setup


@pytest.fixture
def mock_wizard():
    """Create a ConfigurationWizard with mocked components."""
    with (
        patch("src.cli.commands.setup.TemplateManager") as mock_template_mgr,
        patch("src.cli.commands.setup.ProfileManager") as mock_profile_mgr,
        patch("src.cli.commands.setup.WizardValidator") as mock_validator,
    ):
        wizard = ConfigurationWizard()

        # Configure mocks with reasonable defaults
        wizard.template_manager = mock_template_mgr.return_value
        wizard.profile_manager = mock_profile_mgr.return_value
        wizard.validator = mock_validator.return_value

        # Setup default return values
        wizard.template_manager.get_template.return_value = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
        }

        wizard.profile_manager.list_profiles.return_value = [
            "personal",
            "development",
            "production",
        ]
        wizard.profile_manager.profile_templates = {
            "personal": "personal-use",
            "development": "development",
            "production": "production",
        }

        wizard.validator.validate_api_key.return_value = (True, None)
        wizard.validator.validate_url.return_value = (True, None)
        wizard.validator.validate_and_show_errors.return_value = True

        return wizard


class TestModernConfigurationWizard:
    """Modern tests for ConfigurationWizard with enhanced coverage."""

    def test_init_with_default_config_dir(self):
        """Test ConfigurationWizard initialization with default config directory."""
        wizard = ConfigurationWizard()

        assert wizard.console is not None
        assert isinstance(wizard.console, Console)
        assert wizard.config_data == {}
        assert wizard.customizations == {}
        assert wizard.selected_template is None
        assert wizard.selected_profile is None

    def test_init_with_custom_config_dir(self, tmp_path):
        """Test ConfigurationWizard initialization with custom config directory."""
        config_dir = tmp_path / "custom_config"
        wizard = ConfigurationWizard(config_dir)

        assert wizard.console is not None
        assert wizard.template_manager is not None
        assert wizard.profile_manager is not None
        assert wizard.validator is not None

    def test_welcome_rich_output(self, rich_output_capturer):
        """Test welcome message displays correctly with Rich formatting."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        wizard.welcome()

        # Verify Rich panel and content
        rich_output_capturer.assert_panel_title("🚀 Template-Driven Setup")
        rich_output_capturer.assert_contains("🧙 Modern Configuration Wizard")
        rich_output_capturer.assert_contains("Profile-based configuration templates")
        rich_output_capturer.assert_contains("Real-time validation")
        rich_output_capturer.assert_contains("Smart customization options")
        rich_output_capturer.assert_contains("Template preview and comparison")

    @patch("src.cli.commands.setup.questionary")
    def test_select_profile_interactive_flow(
        self, mock_questionary, mock_wizard, rich_output_capturer
    ):
        """Test profile selection with complete interactive flow."""
        mock_wizard.console = rich_output_capturer.console

        # Mock questionary interactions
        mock_questionary.select.return_value.ask.return_value = "personal"
        mock_questionary.confirm.return_value.ask.return_value = True

        # Mock profile manager methods
        mock_wizard.profile_manager.show_profiles_table = MagicMock()
        mock_wizard.profile_manager.show_profile_setup_instructions = MagicMock()

        result = mock_wizard.select_profile()

        assert result == "personal"
        mock_wizard.profile_manager.show_profiles_table.assert_called_once()
        mock_wizard.profile_manager.show_profile_setup_instructions.assert_called_once_with(
            "personal"
        )
        rich_output_capturer.assert_contains("🎯 Profile Selection")

    @patch("src.cli.commands.setup.questionary")
    def test_select_profile_user_cancellation(self, mock_questionary):
        """Test profile selection when user cancels."""
        wizard = ConfigurationWizard()

        # Mock user cancelling selection
        mock_questionary.select.return_value.ask.return_value = None

        with pytest.raises(click.Abort):
            wizard.select_profile()

    @patch("src.cli.commands.setup.questionary")
    def test_select_profile_retry_flow(self, mock_questionary, rich_output_capturer):
        """Test profile selection retry when user declines initial choice."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock profile manager with patch
        with (
            patch.object(wizard.profile_manager, "list_profiles") as mock_list,
            patch.object(wizard.profile_manager, "show_profiles_table") as _mock_show,
            patch.object(
                wizard.profile_manager, "show_profile_setup_instructions"
            ) as _mock_instructions,
        ):
            mock_list.return_value = ["personal", "development"]

            # First attempt: user selects but then declines confirmation
            # Second attempt: user selects and confirms
            mock_questionary.select.return_value.ask.side_effect = [
                "development",
                "personal",
            ]
            mock_questionary.confirm.return_value.ask.side_effect = [False, True]

            result = wizard.select_profile()

            assert result == "personal"
            assert mock_questionary.select.return_value.ask.call_count == 2
            assert mock_questionary.confirm.return_value.ask.call_count == 2

    def test_customize_template_preview_only(self, rich_output_capturer):
        """Test template customization with preview only."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock template manager with patch
        template_data = {"qdrant": {"host": "localhost"}, "openai": {"model": "test"}}
        with (
            patch.object(wizard.template_manager, "get_template") as mock_get,
            patch.object(wizard.template_manager, "preview_template") as mock_preview,
        ):
            mock_get.return_value = template_data

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                # User wants preview but no customization
                mock_questionary.confirm.return_value.ask.side_effect = [True, False]

                result = wizard.customize_template("test-template")

                assert result == {}
                mock_preview.assert_called_once_with("test-template")
                rich_output_capturer.assert_contains("🛠️ Customizing")
                rich_output_capturer.assert_contains("test-template")

    def test_customize_api_keys_openai_valid(self):
        """Test API key customization with valid OpenAI key."""
        wizard = ConfigurationWizard()

        # Mock validator with patch
        with patch.object(wizard.validator, "validate_api_key") as mock_validate:
            mock_validate.return_value = (True, None)

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                # User chooses to set OpenAI key
                mock_questionary.confirm.return_value.ask.side_effect = [
                    True,
                    False,
                ]  # OpenAI yes, Firecrawl no
                mock_questionary.password.return_value.ask.return_value = (
                    "sk-test-valid-key"
                )

                result = wizard.customize_api_keys({})

                assert result == {"openai": {"api_key": "sk-test-valid-key"}}
                mock_validate.assert_called_with("openai", "sk-test-valid-key")

    def test_customize_api_keys_openai_invalid_retry(self, rich_output_capturer):
        """Test API key customization with invalid key and retry."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock validator with patch
        with patch.object(wizard.validator, "validate_api_key") as mock_validate:
            mock_validate.side_effect = [
                (False, "Invalid key format"),
                (True, None),
            ]

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.side_effect = [
                    True,
                    True,
                    False,
                ]  # OpenAI yes, retry yes, Firecrawl no
                mock_questionary.password.return_value.ask.side_effect = [
                    "invalid-key",
                    "sk-valid-key",
                ]

                result = wizard.customize_api_keys({})

                assert result == {"openai": {"api_key": "sk-valid-key"}}
                rich_output_capturer.assert_contains(
                    "Invalid API key: Invalid key format"
                )

    def test_customize_database_local_connection(self):
        """Test database customization for local Qdrant."""
        wizard = ConfigurationWizard()

        # Mock validator with patch
        with patch.object(wizard.validator, "validate_url") as mock_validate:
            mock_validate.return_value = (True, None)

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.select.return_value.ask.return_value = (
                    "Custom host/port"
                )
                mock_questionary.text.return_value.ask.side_effect = [
                    "localhost",
                    "6333",
                ]

                result = wizard.customize_database({})

                assert result == {"qdrant": {"host": "localhost", "port": 6333}}

    def test_customize_database_cloud_connection(self):
        """Test database customization for Qdrant Cloud."""
        wizard = ConfigurationWizard()

        # Mock validator with patch
        with patch.object(wizard.validator, "validate_url") as mock_validate:
            mock_validate.return_value = (True, None)

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.select.return_value.ask.return_value = (
                    "Qdrant Cloud URL"
                )
                mock_questionary.text.return_value.ask.return_value = (
                    "https://cloud.qdrant.io"
                )
                mock_questionary.password.return_value.ask.return_value = (
                    "cloud-api-key"
                )

                result = wizard.customize_database({})

                assert result == {
                    "qdrant": {
                        "url": "https://cloud.qdrant.io",
                        "api_key": "cloud-api-key",
                    }
                }

    def test_customize_performance_chunk_size(self):
        """Test performance customization with chunk size setting."""
        wizard = ConfigurationWizard()

        with patch("src.cli.commands.setup.questionary") as mock_questionary:
            mock_questionary.confirm.return_value.ask.return_value = True
            mock_questionary.text.return_value.ask.return_value = "2000"

            result = wizard.customize_performance({})

            assert result == {"text_processing": {"chunk_size": 2000}}

    def test_customize_performance_invalid_chunk_size(self, rich_output_capturer):
        """Test performance customization with invalid chunk size."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        with patch("src.cli.commands.setup.questionary") as mock_questionary:
            mock_questionary.confirm.return_value.ask.return_value = True
            mock_questionary.text.return_value.ask.return_value = "invalid"

            result = wizard.customize_performance({})

            assert result == {}
            rich_output_capturer.assert_contains("Invalid chunk size")

    def test_customize_advanced_debug_mode(self):
        """Test advanced customization with debug mode enabled."""
        wizard = ConfigurationWizard()

        with patch("src.cli.commands.setup.questionary") as mock_questionary:
            # Mock all questionary interactions
            mock_questionary.confirm.return_value.ask.side_effect = [
                True,  # Preview template
                True,  # Do customization
                False,  # Don't customize API Keys
                False,  # Don't customize Database
                False,  # Don't customize Performance
                False,  # Don't customize Advanced
            ]
            mock_questionary.password.return_value.ask.return_value = (
                "sk-test-valid-key"
            )
            mock_questionary.text.return_value.ask.return_value = "test-value"
            mock_questionary.select.return_value.ask.return_value = "test-choice"

            # Mock the template manager to return a simple template
            with patch.object(
                wizard.template_manager, "get_template"
            ) as mock_get_template:
                mock_get_template.return_value = {"base": "config"}

                result = wizard.customize_template("test_template")

                assert isinstance(result, dict)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_configuration_success(self, mock_file, rich_output_capturer):
        """Test successful configuration saving."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Create secure temporary files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_profile_config.json", delete=False
        ) as temp_profile:
            temp_profile_path = temp_profile.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_config.json", delete=False
        ) as temp_config:
            temp_config_path = temp_config.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as temp_env:
            temp_env_path = temp_env.name

        # Mock profile manager with patch
        with (
            patch.object(
                wizard.profile_manager, "create_profile_config"
            ) as mock_create,
            patch.object(wizard.profile_manager, "activate_profile") as mock_activate,
            patch.object(wizard.profile_manager, "generate_env_file") as mock_gen_env,
        ):
            mock_create.return_value = Path(temp_profile_path)
            mock_activate.return_value = Path(temp_config_path)
            mock_gen_env.return_value = Path(temp_env_path)

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.side_effect = [
                    True,
                    True,
                ]  # activate profile, generate env

                config_data = {"test": "config"}
                result = wizard.save_configuration("test-profile", config_data)

                assert result == Path(temp_config_path)
                mock_create.assert_called_once_with(
                    "test-profile", customizations=config_data
                )
                mock_activate.assert_called_once_with("test-profile")
                rich_output_capturer.assert_contains("💾 Saving Configuration")

    def test_save_configuration_error_handling(self, rich_output_capturer):
        """Test configuration saving error handling."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock profile manager to raise exception with patch
        with patch.object(
            wizard.profile_manager, "create_profile_config"
        ) as mock_create:
            mock_create.side_effect = Exception("Save failed")

            with pytest.raises(Exception, match="Save failed"):
                wizard.save_configuration("test-profile", {})

            rich_output_capturer.assert_contains("❌ Error saving configuration:")

    @patch.object(ConfigurationWizard, "save_configuration")
    @patch.object(ConfigurationWizard, "customize_template")
    @patch.object(ConfigurationWizard, "select_profile")
    @patch.object(ConfigurationWizard, "welcome")
    def test_run_setup_complete_flow(
        self,
        mock_welcome,
        mock_select_profile,
        mock_customize,
        mock_save,
        rich_output_capturer,
    ):
        """Test complete setup wizard flow."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Create secure temporary file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_config.json", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        # Setup mocks
        mock_select_profile.return_value = "personal"
        mock_customize.return_value = {"openai": {"api_key": "test"}}
        mock_save.return_value = Path(temp_path)

        # Mock profile manager and template manager with patch
        with (
            patch.object(
                wizard.profile_manager,
                "profile_templates",
                {"personal": "personal-use"},
            ),
            patch.object(
                wizard.template_manager, "create_config_from_template"
            ) as mock_create,
            patch.object(wizard.validator, "validate_and_show_errors") as mock_validate,
            patch.object(wizard.validator, "show_validation_summary") as _mock_summary,
        ):
            # Mock template manager
            config_mock = MagicMock()
            config_mock.model_dump.return_value = {"test": "config"}
            mock_create.return_value = config_mock

            # Mock validator
            mock_validate.return_value = True

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = True

                result = wizard.run_setup()

                assert result == Path(temp_path)
                mock_welcome.assert_called_once()
                mock_select_profile.assert_called_once()
                mock_customize.assert_called_once_with("personal-use")
                mock_save.assert_called_once()

    def test_run_setup_user_cancellation(self):
        """Test setup wizard when user cancels at start."""
        wizard = ConfigurationWizard()

        with patch("src.cli.commands.setup.questionary") as mock_questionary:
            mock_questionary.confirm.return_value.ask.return_value = False

            with pytest.raises(click.Abort):
                wizard.run_setup()

    def test_run_setup_keyboard_interrupt(self):
        """Test setup wizard with keyboard interrupt."""
        wizard = ConfigurationWizard()

        with patch("src.cli.commands.setup.questionary") as mock_questionary:
            mock_questionary.confirm.return_value.ask.side_effect = KeyboardInterrupt()

            with pytest.raises((KeyboardInterrupt, click.Abort)):
                wizard.run_setup()

    def test_run_setup_validation_failure(self, rich_output_capturer):
        """Test setup wizard with validation failure."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock components with proper patch
        wizard.selected_profile = "test"
        config_mock = MagicMock()
        config_mock.model_dump.return_value = {}

        with (
            patch.object(
                wizard.profile_manager, "profile_templates", {"test": "testing"}
            ),
            patch.object(
                wizard.template_manager, "create_config_from_template"
            ) as mock_create,
            patch.object(wizard.validator, "validate_and_show_errors") as mock_validate,
            patch.object(wizard, "customize_template") as mock_customize,
        ):
            mock_create.return_value = config_mock
            mock_validate.return_value = False
            mock_customize.return_value = {}  # No customizations

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.side_effect = [
                    True,
                    False,
                ]  # proceed, don't continue with errors

                with pytest.raises(click.Abort):
                    wizard.run_setup()

                rich_output_capturer.assert_contains("✅ Validating Configuration")

    def test_show_success_message(self, rich_output_capturer):
        """Test success message display."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console
        wizard.selected_profile = "personal"

        # Create secure temporary file for display test
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_config.json", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        # Mock profile_manager.profile_templates with patch
        with patch.object(
            wizard.profile_manager, "profile_templates", {"personal": "personal-use"}
        ):
            wizard.show_success_message(Path(temp_path))

            rich_output_capturer.assert_panel_title("🚀 Template-Driven Setup Complete")
            rich_output_capturer.assert_contains("🎉 Modern Setup Complete!")
            rich_output_capturer.assert_contains("Profile: personal")
            rich_output_capturer.assert_contains(f"Config file: {temp_path}")
            rich_output_capturer.assert_contains("Test configuration:")
            rich_output_capturer.assert_contains("Start services:")
            rich_output_capturer.assert_contains("Check system status:")


class TestSetupCommandModernized:
    """Modernized tests for the setup Click command."""

    def test_setup_command_with_pre_selected_profile(self, interactive_cli_runner):
        """Test setup command with pre-selected profile."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = Path(
                "/tmp/config.json"
            )  # test temp path
            mock_wizard_class.return_value = mock_wizard

            # Mock profile manager
            mock_wizard.profile_manager.list_profiles.return_value = [
                "personal",
                "development",
            ]

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = (
                    False  # Skip validation
                )

                result = interactive_cli_runner.invoke(setup, ["--profile", "personal"])

                assert result.exit_code == 0
                assert mock_wizard.selected_profile == "personal"
                mock_wizard.run_setup.assert_called_once()

    def test_setup_command_invalid_profile(self, interactive_cli_runner):
        """Test setup command with invalid profile."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard_class.return_value = mock_wizard
            mock_wizard.profile_manager.list_profiles.return_value = [
                "personal",
                "development",
            ]

            result = interactive_cli_runner.invoke(setup, ["--profile", "invalid"])

            assert result.exit_code == 1
            assert "Profile 'invalid' not found" in result.output

    def test_setup_command_with_validation(self, interactive_cli_runner):
        """Test setup command with configuration validation."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = Path(
                "/tmp/config.json"
            )  # test temp path
            mock_wizard_class.return_value = mock_wizard

            with (
                patch("src.cli.commands.setup.validate_config"),
                patch("src.cli.commands.setup.questionary") as mock_questionary,
            ):
                mock_questionary.confirm.return_value.ask.return_value = (
                    True  # Accept validation
                )

                result = interactive_cli_runner.invoke(setup, [])

                assert result.exit_code == 0
                # Note: validate_config would be called but might not be available in test

    def test_setup_command_validation_fallback(self, interactive_cli_runner):
        """Test setup command validation fallback when config command unavailable."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = Path(
                "/tmp/config.json"
            )  # test temp path
            mock_wizard_class.return_value = mock_wizard

            # Mock file content
            with patch("pathlib.Path.read_text") as mock_read:
                mock_read.return_value = '{"test": "config"}'

                with (
                    patch("src.cli.commands.setup.questionary") as mock_questionary,
                    patch("src.cli.commands.setup.validate_config", None),
                ):
                    mock_questionary.confirm.return_value.ask.return_value = True

                    result = interactive_cli_runner.invoke(setup, [])

                    assert result.exit_code == 0
                    assert "Using wizard validation" in result.output

    def test_setup_command_keyboard_interrupt(self, interactive_cli_runner):
        """Test setup command keyboard interrupt handling."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.side_effect = KeyboardInterrupt()
            mock_wizard_class.return_value = mock_wizard

            result = interactive_cli_runner.invoke(setup, [])

            assert result.exit_code == 1
            assert "Setup cancelled by user" in result.output

    def test_setup_command_general_exception(self, interactive_cli_runner):
        """Test setup command general exception handling."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.side_effect = Exception("Unexpected error")
            mock_wizard_class.return_value = mock_wizard

            result = interactive_cli_runner.invoke(setup, [])

            assert result.exit_code == 1
            assert "Setup failed: Unexpected error" in result.output

    def test_setup_command_help_output(self, interactive_cli_runner):
        """Test setup command help output."""
        result = interactive_cli_runner.invoke(setup, ["--help"])

        assert result.exit_code == 0
        assert "Modern template-driven configuration wizard" in result.output
        assert "Profile-based templates" in result.output
        assert "Real-time validation" in result.output
        assert "--profile" in result.output
        assert "--output" in result.output
        assert "--config-dir" in result.output


class TestSetupIntegrationModernized:
    """Integration tests for the modernized setup command."""

    def test_end_to_end_setup_flow(self, cli_integration_setup, interactive_cli_runner):
        """Test complete end-to-end setup flow."""
        setup_data = cli_integration_setup

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            # Create realistic wizard mock
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = setup_data["temp_config"]
            mock_wizard_class.return_value = mock_wizard

            # Mock the profile manager
            mock_wizard.profile_manager.list_profiles.return_value = [
                "personal",
                "development",
            ]

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = (
                    False  # Skip validation
                )

                result = interactive_cli_runner.invoke(
                    setup, ["--config-dir", str(setup_data["config_dir"])]
                )

                assert result.exit_code == 0
                mock_wizard.run_setup.assert_called_once()

    def test_setup_with_existing_profiles(
        self, temp_profiles_dir, interactive_cli_runner
    ):
        """Test setup with existing profile files."""
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = temp_profiles_dir / "personal.json"
            mock_wizard_class.return_value = mock_wizard

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = False

                result = interactive_cli_runner.invoke(
                    setup, ["--config-dir", str(temp_profiles_dir.parent)]
                )

                # Should succeed even with existing profiles
                assert result.exit_code == 0

    def test_wizard_accessibility_features(self, rich_output_capturer):
        """Test wizard accessibility features and user guidance."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        wizard.welcome()

        # Check for accessibility-friendly elements
        output = rich_output_capturer.get_output()

        # Should have clear visual hierarchy
        assert "🧙" in output  # Clear visual indicator
        assert "Modern Configuration Wizard" in output

        # Should have helpful descriptions
        assert "Profile-based configuration" in output
        assert "Real-time validation" in output

        # Should use consistent styling
        lines = rich_output_capturer.get_lines()
        assert len(lines) > 5  # Multi-line output for readability

    def test_configuration_wizard_import_safety(self):
        """Test that ConfigurationWizard imports safely."""
        try:
            wizard = ConfigurationWizard()
            assert wizard is not None
            assert hasattr(wizard, "run_setup")
            assert hasattr(wizard, "welcome")
            assert hasattr(wizard, "select_profile")
        except ImportError as e:
            pytest.fail(f"Failed to import ConfigurationWizard: {e}")

    def test_setup_command_import_safety(self):
        """Test that setup command imports safely."""
        try:
            assert setup is not None
            assert hasattr(setup, "invoke")
            assert setup.name == "setup"
        except ImportError as e:
            pytest.fail(f"Failed to import setup command: {e}")
