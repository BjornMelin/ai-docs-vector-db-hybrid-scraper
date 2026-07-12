"""Tests for the setup command module.

This module tests the  template-driven configuration wizard including
profile selection, template customization, and Rich console output.
"""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from src.cli.commands.setup import ConfigurationWizard, setup
from src.cli.wizard import profile_manager as profile_manager_module
from src.cli.wizard.profile_manager import ProfileManager


def test_profile_manager_uses_canonical_local_only_name(tmp_path: Path) -> None:
    """The documented local-only profile should map directly to its template."""
    manager = ProfileManager(tmp_path)

    assert manager.profile_templates["local-only"] == "local-only"
    assert manager.get_profile_info("local-only") is not None
    assert manager.get_environment_overrides("local-only")["AI_DOCS_ENVIRONMENT"] == (
        "development"
    )


@pytest.mark.parametrize(
    "profile_name",
    ["personal", "development", "production", "testing", "local-only", "minimal"],
)
def test_profile_instructions_omit_unneeded_openai_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
) -> None:
    """Built-in FastEmbed profiles should not request an OpenAI credential."""
    output = StringIO()
    monkeypatch.setattr(
        profile_manager_module,
        "console",
        Console(file=output, width=120, no_color=True),
    )
    manager = ProfileManager(tmp_path)

    manager.show_profile_setup_instructions(profile_name)

    instructions = output.getvalue()
    assert "No provider credentials are required by this profile." in instructions
    assert "AI_DOCS_OPENAI__API_KEY" not in instructions


def test_profile_manager_writes_json_serializable_configuration(
    tmp_path: Path,
) -> None:
    """Profile files should serialize Pydantic paths and sets as JSON values."""
    manager = ProfileManager(tmp_path)

    profile_path = manager.create_profile_config("development")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert isinstance(profile["data_dir"], str)
    assert isinstance(profile["deployment"]["enabled_strategies"], list)


class TestConfigurationWizard:
    """Test the ConfigurationWizard class."""

    def test_initialization(self, rich_output_capturer):
        """Test wizard initialization."""
        wizard = ConfigurationWizard()

        assert wizard.console is not None
        assert wizard.template_manager is not None
        assert wizard.profile_manager is not None
        assert wizard.validator is not None

    def test_welcome(self, rich_output_capturer):
        """Test welcome message display."""
        wizard = ConfigurationWizard()

        # This should not raise an exception
        wizard.welcome()

        # Since the output might be captured differently by Rich,
        # just verify the method executes without error

    def test_select_profile_personal(self, rich_output_capturer):
        """Test profile selection for personal profile."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.profile_manager, "show_profiles_table") as mock_show,
            patch.object(wizard.profile_manager, "list_profiles") as mock_list,
            patch.object(
                wizard.profile_manager, "show_profile_setup_instructions"
            ) as mock_instructions,
            patch("questionary.select") as mock_select,
            patch("questionary.confirm") as mock_confirm,
        ):
            mock_list.return_value = ["personal", "development", "production"]
            mock_select.return_value.ask.return_value = "personal"
            mock_confirm.return_value.ask.return_value = True

            result = wizard.select_profile()

            assert result == "personal"
            mock_show.assert_called_once()
            mock_instructions.assert_called_once_with("personal")

    def test_select_profile_with_reselection(self, rich_output_capturer):
        """Test profile selection with user wanting to reselect."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.profile_manager, "show_profiles_table"),
            patch.object(wizard.profile_manager, "list_profiles") as mock_list,
            patch.object(wizard.profile_manager, "show_profile_setup_instructions"),
            patch("questionary.select") as mock_select,
            patch("questionary.confirm") as mock_confirm,
        ):
            mock_list.return_value = ["personal", "development", "production"]
            mock_select.return_value.ask.side_effect = ["development", "personal"]
            mock_confirm.return_value.ask.side_effect = [
                False,
                True,
            ]  # First no, then yes

            result = wizard.select_profile()

            assert result == "personal"

    def test_customize_template_no_customization(self, rich_output_capturer):
        """Test template customization with no changes."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.template_manager, "get_template") as mock_get,
            patch.object(wizard.template_manager, "preview_template") as mock_preview,
            patch("questionary.confirm") as mock_confirm,
        ):
            mock_get.return_value = {"template": "data"}
            mock_confirm.return_value.ask.side_effect = [
                True,
                False,
            ]  # Preview yes, customize no

            result = wizard.customize_template("personal")

            assert result == {}
            mock_preview.assert_called_once_with("personal")

    def test_customize_template_with_api_keys(self, rich_output_capturer):
        """Test template customization with API key customization."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.template_manager, "get_template") as mock_get,
            patch.object(wizard.template_manager, "preview_template"),
            patch.object(wizard.validator, "validate_api_key") as mock_validate,
            patch("questionary.confirm") as mock_confirm,
            patch("questionary.password") as mock_password,
        ):
            mock_get.return_value = {"template": "data"}
            mock_validate.return_value = (True, None)
            mock_confirm.return_value.ask.side_effect = [
                False,  # No preview
                True,  # Do customize
                True,  # Customize API Keys
                True,  # Set OpenAI key
                False,  # Don't set Firecrawl key
                False,  # Don't customize other sections
                False,  # Don't customize other sections
                False,  # Don't customize other sections
            ]
            mock_password.return_value.ask.return_value = "sk-test123"

            result = wizard.customize_template("personal")

            assert "openai" in result
            assert result["openai"]["api_key"] == "sk-test123"

    def test_customize_firecrawl_key_uses_browser_section(self, rich_output_capturer):
        """Firecrawl credentials should map to the canonical browser model."""
        wizard = ConfigurationWizard()

        with (
            patch.object(
                wizard.validator, "validate_api_key", return_value=(True, None)
            ),
            patch("questionary.confirm") as mock_confirm,
            patch("questionary.password") as mock_password,
        ):
            mock_confirm.return_value.ask.side_effect = [False, True]
            mock_password.return_value.ask.return_value = "fc-test123"

            result = wizard.customize_api_keys({})

        assert result == {"browser": {"firecrawl": {"api_key": "fc-test123"}}}

    def test_customize_database_local(self, rich_output_capturer):
        """Test database customization for local setup."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.validator, "validate_url") as mock_validate,
            patch("questionary.select") as mock_select,
            patch("questionary.text") as mock_text,
        ):
            mock_validate.return_value = (True, None)
            mock_select.return_value.ask.return_value = "Custom host/port"
            mock_text.return_value.ask.side_effect = ["localhost", "6333"]

            result = wizard.customize_database({})

            assert result["qdrant"] == {"url": "http://localhost:6333"}

    def test_customize_database_cloud(self, rich_output_capturer):
        """Test database customization for cloud setup."""
        wizard = ConfigurationWizard()

        with (
            patch.object(wizard.validator, "validate_url") as mock_validate,
            patch("questionary.select") as mock_select,
            patch("questionary.text") as mock_text,
            patch("questionary.password") as mock_password,
        ):
            mock_validate.return_value = (True, None)
            mock_select.return_value.ask.return_value = "Qdrant Cloud URL"
            mock_text.return_value.ask.return_value = "https://xyz.cloud.qdrant.io"
            mock_password.return_value.ask.return_value = "api-key-123"

            result = wizard.customize_database({})

            assert result["qdrant"]["url"] == "https://xyz.cloud.qdrant.io"
            assert result["qdrant"]["api_key"] == "api-key-123"

    def test_customize_performance_uses_chunking_section(self, rich_output_capturer):
        """Chunk settings should map to the canonical chunking model."""
        wizard = ConfigurationWizard()

        with (
            patch("questionary.confirm") as mock_confirm,
            patch("questionary.text") as mock_text,
        ):
            mock_confirm.return_value.ask.side_effect = [True, True, True]
            mock_text.return_value.ask.side_effect = [
                "1200",
                "200",
                "600",
                "120",
                "cl100k_base",
                "20000",
            ]

            result = wizard.customize_performance({})

        assert result["chunking"]["chunk_size"] == 1200
        assert result["chunking"]["chunk_overlap"] == 200
        assert "text_processing" not in result

    def test_save_configuration(self, rich_output_capturer):
        """Test configuration saving."""
        wizard = ConfigurationWizard()

        with (
            patch.object(
                wizard.profile_manager, "create_profile_config"
            ) as mock_create,
            patch.object(wizard.profile_manager, "activate_profile") as mock_activate,
            patch.object(wizard.profile_manager, "generate_env_file") as mock_env,
            patch("questionary.confirm") as mock_confirm,
        ):
            config_path = Path("/test/config.json")
            mock_create.return_value = config_path
            mock_activate.return_value = config_path
            mock_env.return_value = Path("/test/.env")
            mock_confirm.return_value.ask.side_effect = [
                True,
                True,
            ]  # Activate and generate env

            result = wizard.save_configuration("personal", {"test": "data"})

            assert result == config_path
            mock_create.assert_called_once_with(
                "personal", customizations={"test": "data"}
            )
            mock_activate.assert_called_once_with("personal")
            mock_env.assert_called_once_with("personal")


class TestSetupCommand:
    """Test the setup command."""

    def test_setup_command_help(self, cli_runner):
        """Test setup command help output."""
        result = cli_runner.invoke(setup, ["--help"])

        assert result.exit_code == 0
        assert "Template-driven configuration wizard." in result.output

    def test_setup_command_with_profile(self, cli_runner):
        """Test setup command with profile option."""
        result = cli_runner.invoke(setup, ["--help"])

        assert result.exit_code == 0
        assert "--profile" in result.output

    def test_setup_command_parameters(self, cli_runner):
        """Test setup command parameters."""
        result = cli_runner.invoke(setup, ["--help"])

        assert result.exit_code == 0
        assert "--profile" in result.output
        assert "--config-dir" in result.output

    def test_setup_validates_created_profile(self, cli_runner, tmp_path: Path):
        """Default post-setup validation should invoke the config command correctly."""
        config_path = tmp_path / "profile.json"
        config_path.write_text("{}\n", encoding="utf-8")

        with (
            patch(
                "src.cli.commands.setup.ConfigurationWizard.run_setup",
                return_value=config_path,
            ),
            patch("src.cli.commands.setup.questionary.confirm") as mock_confirm,
        ):
            mock_confirm.return_value.ask.return_value = True
            result = cli_runner.invoke(setup, [], obj={"config": None})

        assert result.exit_code == 0
        assert "Configuration is valid" in result.output
