"""Tests for the setup command and ConfigurationWizard.

This module tests the interactive configuration wizard including all setup flows,
user interactions, file operations, and Rich console output.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import click
import pytest
from rich.console import Console

from src.cli.commands.setup import ConfigurationWizard, setup


class TestConfigurationWizard:
    """Test the ConfigurationWizard class for interactive setup."""

    def test_init(self):
        """Test ConfigurationWizard initialization."""
        wizard = ConfigurationWizard()

        assert wizard.console is not None
        assert isinstance(wizard.console, Console)
        assert wizard.config_data == {}

    def test_welcome(self, rich_output_capturer):
        """Test welcome message display."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        wizard.welcome()

        # Verify welcome content
        rich_output_capturer.assert_contains("🧙 Configuration Wizard")
        rich_output_capturer.assert_contains(
            "Let's set up your AI Documentation Scraper!"
        )
        rich_output_capturer.assert_contains("Vector database connection")
        rich_output_capturer.assert_contains("API keys for embedding providers")
        rich_output_capturer.assert_contains("Caching and performance settings")
        rich_output_capturer.assert_contains("Browser automation preferences")
        rich_output_capturer.assert_contains("Welcome to Setup")

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_configure_database_local(
        self, mock_prompt, mock_confirm, rich_output_capturer
    ):
        """Test database configuration with local Qdrant."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses for local setup
        mock_confirm.return_value = True  # Use local instance
        mock_prompt.side_effect = ["localhost", "6333"]  # host, port

        result = wizard.configure_database()

        # Verify configuration
        assert result == {
            "qdrant": {"host": "localhost", "port": 6333, "use_memory": False}
        }

        # Verify Rich output
        rich_output_capturer.assert_contains("📊 Vector Database Configuration")

        # Verify prompts were called correctly
        mock_confirm.assert_called_once_with("Use local Qdrant instance?", default=True)
        assert mock_prompt.call_count == 2

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_configure_database_cloud(self, mock_prompt, mock_confirm):
        """Test database configuration with Qdrant Cloud."""
        wizard = ConfigurationWizard()

        # Mock user responses for cloud setup
        mock_confirm.return_value = False  # Use cloud instance
        mock_prompt.side_effect = [
            "https://xyz.qdrant.tech:6333",  # URL
            "test-api-key",  # API key
        ]

        result = wizard.configure_database()

        # Verify configuration
        assert result == {
            "qdrant": {"url": "https://xyz.qdrant.tech:6333", "api_key": "test-api-key"}
        }

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_configure_database_cloud_no_api_key(self, mock_prompt, mock_confirm):
        """Test database configuration with cloud but no API key."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_confirm.return_value = False
        mock_prompt.side_effect = [
            "https://xyz.qdrant.tech:6333",  # URL
            "",  # Empty API key
        ]

        result = wizard.configure_database()

        # Verify API key is set to None when empty
        assert result["qdrant"]["api_key"] is None

    @patch("rich.prompt.Prompt.ask")
    def test_configure_embeddings_openai_only(self, mock_prompt, rich_output_capturer):
        """Test embedding configuration with OpenAI only."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses
        mock_prompt.side_effect = [
            "1",  # OpenAI only
            "sk-test-api-key",  # OpenAI API key
        ]

        result = wizard.configure_embeddings()

        # Verify configuration
        assert result == {
            "openai": {"api_key": "sk-test-api-key", "model": "text-embedding-3-small"}
        }

        # Verify Rich output contains provider table
        rich_output_capturer.assert_contains("🔑 Embedding Provider Configuration")
        rich_output_capturer.assert_contains("Available Embedding Providers")
        rich_output_capturer.assert_contains("OpenAI")
        rich_output_capturer.assert_contains("FastEmbed")

    @patch("rich.prompt.Prompt.ask")
    def test_configure_embeddings_fastembed_only(self, mock_prompt):
        """Test embedding configuration with FastEmbed only."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_prompt.return_value = "2"  # FastEmbed only

        result = wizard.configure_embeddings()

        # Verify configuration
        assert result == {
            "fastembed": {
                "model": "BAAI/bge-small-en-v1.5",
                "cache_dir": "~/.cache/fastembed",
            }
        }

    @patch("rich.prompt.Prompt.ask")
    def test_configure_embeddings_both_providers(self, mock_prompt):
        """Test embedding configuration with both providers."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_prompt.side_effect = [
            "3",  # Both providers
            "sk-test-api-key",  # OpenAI API key
        ]

        result = wizard.configure_embeddings()

        # Verify both providers are configured
        assert "openai" in result
        assert "fastembed" in result
        assert result["openai"]["api_key"] == "sk-test-api-key"
        assert result["fastembed"]["model"] == "BAAI/bge-small-en-v1.5"

    @patch("rich.prompt.Prompt.ask")
    def test_configure_embeddings_empty_openai_key(self, mock_prompt):
        """Test embedding configuration with empty OpenAI key."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_prompt.side_effect = [
            "1",  # OpenAI only
            "",  # Empty API key
        ]

        result = wizard.configure_embeddings()

        # Should not include OpenAI config if no key provided
        assert "openai" not in result

    @patch("rich.prompt.Confirm.ask")
    def test_configure_browser(self, mock_confirm, rich_output_capturer):
        """Test browser configuration."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses
        mock_confirm.side_effect = [
            True,  # headless mode
            True,  # anti-detection
        ]

        result = wizard.configure_browser()

        # Verify configuration
        assert result == {
            "browser": {
                "headless": True,
                "anti_detection": True,
                "timeout": 30000,
                "max_concurrent": 3,
            }
        }

        # Verify Rich output
        rich_output_capturer.assert_contains("🌐 Browser Automation Configuration")

    @patch("rich.prompt.Confirm.ask")
    def test_configure_browser_no_headless_no_stealth(self, mock_confirm):
        """Test browser configuration with headless and stealth disabled."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_confirm.side_effect = [
            False,  # no headless mode
            False,  # no anti-detection
        ]

        result = wizard.configure_browser()

        # Verify configuration
        assert result["browser"]["headless"] is False
        assert result["browser"]["anti_detection"] is False

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_configure_performance_with_cache_and_queue(
        self, mock_prompt, mock_confirm, rich_output_capturer
    ):
        """Test performance configuration with cache and queue enabled."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses
        mock_confirm.side_effect = [
            True,  # enable cache
            True,  # enable queue
        ]
        mock_prompt.side_effect = [
            "localhost",  # redis host
            "6379",  # redis port
        ]

        result = wizard.configure_performance()

        # Verify configuration
        assert result == {
            "cache": {"redis": {"host": "localhost", "port": 6379, "db": 0}},
            "task_queue": {"redis_url": "redis://localhost:6379/1"},
        }

        # Verify Rich output
        rich_output_capturer.assert_contains("⚡ Performance Configuration")

    @patch("rich.prompt.Confirm.ask")
    def test_configure_performance_no_cache_no_queue(self, mock_confirm):
        """Test performance configuration with cache and queue disabled."""
        wizard = ConfigurationWizard()

        # Mock user responses - disable both
        mock_confirm.side_effect = [
            False,  # no cache
            False,  # no queue
        ]

        result = wizard.configure_performance()

        # Should return empty config
        assert result == {}

    @patch("builtins.open", new_callable=mock_open)
    @patch("rich.prompt.Prompt.ask")
    def test_save_configuration_json(
        self, mock_prompt, mock_file, rich_output_capturer
    ):
        """Test saving configuration in JSON format."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses
        mock_prompt.side_effect = [
            "json",  # format
            "/tmp/test_config.json",  # path
        ]

        config_data = {"test": "data"}

        result = wizard.save_configuration(config_data)

        # Verify return path
        assert result == Path("/tmp/test_config.json")

        # Verify file was opened for writing
        mock_file.assert_called_once_with(Path("/tmp/test_config.json"), "w")

        # Verify Rich output
        rich_output_capturer.assert_contains("💾 Saving Configuration")
        rich_output_capturer.assert_contains("✅ Configuration saved to:")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.dump")
    @patch("rich.prompt.Prompt.ask")
    def test_save_configuration_yaml(self, mock_prompt, mock_yaml_dump, mock_file):
        """Test saving configuration in YAML format."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_prompt.side_effect = [
            "yaml",  # format
            "/tmp/test_config.yaml",  # path
        ]

        config_data = {"test": "data"}

        result = wizard.save_configuration(config_data)

        # Verify YAML dump was called
        mock_yaml_dump.assert_called_once()
        assert result == Path("/tmp/test_config.yaml")

    @patch("builtins.open", new_callable=mock_open)
    @patch("tomli_w.dump")
    @patch("rich.prompt.Prompt.ask")
    def test_save_configuration_toml(self, mock_prompt, mock_toml_dump, mock_file):
        """Test saving configuration in TOML format."""
        wizard = ConfigurationWizard()

        # Mock user responses
        mock_prompt.side_effect = [
            "toml",  # format
            "/tmp/test_config.toml",  # path
        ]

        config_data = {"test": "data"}

        result = wizard.save_configuration(config_data)

        # Verify TOML dump was called
        mock_toml_dump.assert_called_once()
        assert result == Path("/tmp/test_config.toml")

    @patch("builtins.open", new_callable=mock_open)
    @patch("rich.prompt.Prompt.ask")
    def test_save_configuration_error_handling(
        self, mock_prompt, mock_file, rich_output_capturer
    ):
        """Test error handling during configuration save."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user responses
        mock_prompt.side_effect = [
            "json",  # format
            "/tmp/test_config.json",  # path
        ]

        # Mock file operation error
        mock_file.side_effect = PermissionError("Permission denied")

        config_data = {"test": "data"}

        with pytest.raises(PermissionError):
            wizard.save_configuration(config_data)

        # Verify error message was displayed
        rich_output_capturer.assert_contains("❌ Error saving configuration:")

    @patch.object(ConfigurationWizard, "save_configuration")
    @patch.object(ConfigurationWizard, "configure_performance")
    @patch.object(ConfigurationWizard, "configure_browser")
    @patch.object(ConfigurationWizard, "configure_embeddings")
    @patch.object(ConfigurationWizard, "configure_database")
    @patch.object(ConfigurationWizard, "welcome")
    @patch("rich.prompt.Confirm.ask")
    def test_run_setup_complete_flow(
        self,
        mock_confirm,
        mock_welcome,
        mock_db,
        mock_embed,
        mock_browser,
        mock_perf,
        mock_save,
        rich_output_capturer,
    ):
        """Test complete setup wizard flow."""
        wizard = ConfigurationWizard()
        wizard.console = rich_output_capturer.console

        # Mock user confirmation to proceed
        mock_confirm.return_value = True

        # Mock configuration methods
        mock_db.return_value = {"qdrant": {"host": "localhost"}}
        mock_embed.return_value = {"openai": {"api_key": "test"}}
        mock_browser.return_value = {"browser": {"headless": True}}
        mock_perf.return_value = {"cache": {"enabled": True}}
        mock_save.return_value = Path("/tmp/config.json")

        result = wizard.run_setup()

        # Verify all methods were called
        mock_welcome.assert_called_once()
        mock_confirm.assert_called_once_with(
            "\nReady to start configuration?", default=True
        )
        mock_db.assert_called_once()
        mock_embed.assert_called_once()
        mock_browser.assert_called_once()
        mock_perf.assert_called_once()
        mock_save.assert_called_once()

        # Verify return value
        assert result == Path("/tmp/config.json")

        # Verify success message
        rich_output_capturer.assert_contains("🎉 Setup Complete!")
        rich_output_capturer.assert_contains(
            "Your AI Documentation Scraper is now configured"
        )
        rich_output_capturer.assert_contains("Test your configuration:")
        rich_output_capturer.assert_contains("Check system status:")
        rich_output_capturer.assert_contains("Create your first collection:")

    @patch("rich.prompt.Confirm.ask")
    def test_run_setup_user_cancellation(self, mock_confirm):
        """Test setup wizard when user cancels."""
        wizard = ConfigurationWizard()

        # Mock user declining to proceed
        mock_confirm.return_value = False

        with pytest.raises(click.Abort):
            wizard.run_setup()


class TestSetupCommand:
    """Test the setup Click command integration."""

    @patch("rich.prompt.Confirm.ask")
    @patch("src.cli.commands.setup.ConfigurationWizard")
    def test_setup_command_basic(self, mock_wizard_class, mock_confirm, cli_runner):
        """Test basic setup command execution."""
        # Mock wizard instance
        mock_wizard = MagicMock()
        mock_wizard.run_setup.return_value = Path("/tmp/config.json")
        mock_wizard_class.return_value = mock_wizard

        # Mock user declining validation
        mock_confirm.return_value = False

        result = cli_runner.invoke(setup, [])

        assert result.exit_code == 0
        mock_wizard_class.assert_called_once()
        mock_wizard.run_setup.assert_called_once()

    @patch("rich.prompt.Confirm.ask")
    @patch("src.cli.commands.setup.ConfigurationWizard")
    def test_setup_command_with_options(
        self, mock_wizard_class, mock_confirm, cli_runner
    ):
        """Test setup command with output and format options."""
        # Mock wizard instance
        mock_wizard = MagicMock()
        mock_wizard.run_setup.return_value = Path("/tmp/custom_config.yaml")
        mock_wizard_class.return_value = mock_wizard

        # Mock user declining validation
        mock_confirm.return_value = False

        result = cli_runner.invoke(
            setup, ["--output", "/tmp/custom_config.yaml", "--format", "yaml"]
        )

        assert result.exit_code == 0
        mock_wizard_class.assert_called_once()
        mock_wizard.run_setup.assert_called_once()

    @patch("src.cli.commands.config.validate_config")
    @patch("src.cli.commands.setup.ConfigurationWizard")
    @patch("src.cli.main.ConfigLoader")
    @patch("rich.prompt.Confirm.ask")
    def test_setup_command_with_validation(
        self,
        mock_confirm,
        mock_config_loader,
        mock_wizard_class,
        mock_validate,
        cli_runner,
        mock_config,
    ):
        """Test setup command with configuration validation."""
        mock_config_loader.load_config.return_value = mock_config

        # Mock wizard instance
        mock_wizard = MagicMock()
        mock_wizard.run_setup.return_value = Path("/tmp/config.json")
        mock_wizard_class.return_value = mock_wizard

        # Mock user responses: first for wizard, then for validation
        mock_confirm.side_effect = [True, True]  # Accept setup, then validate

        result = cli_runner.invoke(setup, [])

        assert result.exit_code == 0

    @patch("rich.prompt.Confirm.ask")
    @patch("src.cli.commands.setup.ConfigurationWizard")
    def test_setup_command_no_validation(
        self,
        mock_wizard_class,
        mock_confirm,
        cli_runner,
    ):
        """Test setup command without validation."""
        # Mock wizard instance
        mock_wizard = MagicMock()
        mock_wizard.run_setup.return_value = Path("/tmp/config.json")
        mock_wizard_class.return_value = mock_wizard

        # Mock user declining validation
        mock_confirm.return_value = False

        result = cli_runner.invoke(setup, [])

        assert result.exit_code == 0
        # Validation should not be imported/called

    @patch("src.cli.commands.setup.ConfigurationWizard")
    def test_setup_command_keyboard_interrupt(self, mock_wizard_class, cli_runner):
        """Test setup command with keyboard interrupt."""
        # Mock wizard raising KeyboardInterrupt
        mock_wizard = MagicMock()
        mock_wizard.run_setup.side_effect = KeyboardInterrupt()
        mock_wizard_class.return_value = mock_wizard

        result = cli_runner.invoke(setup, [])

        assert result.exit_code == 1  # click.Abort() exit code
        assert "Setup cancelled by user" in result.output

    @patch("src.cli.commands.setup.ConfigurationWizard")
    def test_setup_command_general_exception(self, mock_wizard_class, cli_runner):
        """Test setup command with general exception."""
        # Mock wizard raising general exception
        mock_wizard = MagicMock()
        mock_wizard.run_setup.side_effect = Exception("Setup failed")
        mock_wizard_class.return_value = mock_wizard

        result = cli_runner.invoke(setup, [])

        assert result.exit_code == 1  # click.Abort() exit code
        assert "Setup failed: Setup failed" in result.output


class TestSetupIntegration:
    """Integration tests for the setup command."""

    def test_setup_command_help(self, cli_runner):
        """Test setup command help output."""
        result = cli_runner.invoke(setup, ["--help"])

        assert result.exit_code == 0
        assert "Interactive configuration wizard" in result.output
        assert "Vector database connection (Qdrant)" in result.output
        assert "Embedding providers (OpenAI, FastEmbed)" in result.output
        assert "Browser automation settings" in result.output
        assert "Caching and performance options" in result.output
        assert "--output" in result.output
        assert "--format" in result.output

    def test_configuration_wizard_import(self):
        """Test that ConfigurationWizard can be imported."""
        from src.cli.commands.setup import ConfigurationWizard

        wizard = ConfigurationWizard()
        assert wizard is not None
        assert hasattr(wizard, "run_setup")
        assert hasattr(wizard, "configure_database")
        assert hasattr(wizard, "configure_embeddings")
        assert hasattr(wizard, "configure_browser")
        assert hasattr(wizard, "configure_performance")

    def test_setup_command_import(self):
        """Test that setup command can be imported."""
        from src.cli.commands.setup import setup

        assert setup is not None
        assert hasattr(setup, "invoke")
        assert setup.name == "setup"
