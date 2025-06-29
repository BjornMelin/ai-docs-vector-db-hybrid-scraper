"""Tests for the config command module.

This module tests configuration management commands including validation,
display, export, and load functionality with Rich console output.
"""

from unittest.mock import MagicMock, patch

from src.cli.commands.config import (
    _show_config_json,
    _show_config_table,
    _show_config_yaml,
    config,
)


# Test constants to avoid hardcoded sensitive values
TEST_SECRET_VALUE = "test_secret"
TEST_PASS_VALUE = "test_password"
TEST_TOKEN_VALUE = "test_secret_token"


class TestConfigCommandGroup:
    """Test the config command group."""

    def test_config_group_help(self, cli_runner):
        """Test config command group help output."""
        result = cli_runner.invoke(config, ["--help"])

        assert result.exit_code == 0
        assert "Configuration management commands" in result.output
        assert "export" in result.output
        assert "validate" in result.output
        assert "show" in result.output
        assert "load" in result.output


class TestShowCommand:
    """Test the show configuration command."""

    def test_show_help(self, cli_runner):
        """Test show command help."""
        result = cli_runner.invoke(config, ["show", "--help"])

        assert result.exit_code == 0
        assert "Show current configuration" in result.output

    def test_show_command_with_mock_config(self, cli_runner):
        """Test show command with mock configuration."""
        # Create a minimal mock config object
        mock_config = MagicMock()
        mock_config.environment = "test"
        mock_config.debug = True
        mock_config.log_level = "INFO"
        mock_config.app_name = "Test App"
        mock_config.version = "1.0.0"
        mock_config.embedding_provider = "openai"
        mock_config.crawl_provider = "firecrawl"

        result = cli_runner.invoke(config, ["show", "--format", "table"], obj={"config": mock_config})

        assert result.exit_code == 0

    def test_show_formats(self, cli_runner):
        """Test show command format options."""
        result = cli_runner.invoke(config, ["show", "--help"])

        assert result.exit_code == 0
        assert "--format" in result.output
        assert "table" in result.output
        assert "json" in result.output
        assert "yaml" in result.output


class TestValidateCommand:
    """Test the validate configuration command."""

    def test_validate_help(self, cli_runner):
        """Test validate command help."""
        result = cli_runner.invoke(config, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate current configuration" in result.output

    def test_validate_command(self, cli_runner):
        """Test validate command execution."""
        mock_config = MagicMock()
        mock_config.environment = "test"
        mock_config.debug = True
        mock_config.log_level = "INFO"
        mock_config.embedding_provider = "openai"
        mock_config.crawl_provider = "firecrawl"

        result = cli_runner.invoke(config, ["validate"], obj={"config": mock_config})

        assert result.exit_code == 0
        assert "Configuration is valid" in result.output


class TestExportCommand:
    """Test the export configuration command."""

    def test_export_help(self, cli_runner):
        """Test export command help."""
        result = cli_runner.invoke(config, ["export", "--help"])

        assert result.exit_code == 0
        assert "Export configuration to file" in result.output


class TestLoadCommand:
    """Test the load configuration command."""

    def test_load_help(self, cli_runner):
        """Test load command help."""
        result = cli_runner.invoke(config, ["load", "--help"])

        assert result.exit_code == 0
        assert "Load configuration from file" in result.output


class TestConfigDisplayHelpers:
    """Test configuration display helper functions."""

    def test_show_config_table_function(self):
        """Test _show_config_table function."""
        mock_config = MagicMock()
        mock_config.environment = "test"
        mock_config.debug = True
        mock_config.log_level = "INFO"
        mock_config.app_name = "Test App"
        mock_config.version = "1.0.0"
        mock_config.embedding_provider = "openai"
        mock_config.crawl_provider = "firecrawl"

        # This should not raise an exception
        _show_config_table(mock_config)

    def test_show_config_json_function(self):
        """Test _show_config_json function."""
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"test": "data"}

        # This should not raise an exception
        _show_config_json(mock_config)

    def test_show_config_yaml_function(self):
        """Test _show_config_yaml function."""
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"test": "data"}

        # This should not raise an exception
        _show_config_yaml(mock_config)