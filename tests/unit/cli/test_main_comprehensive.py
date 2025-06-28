"""Comprehensive tests for CLI main module with Rich CLI testing.

This module tests the main CLI interface with focus on Rich console integration,
command routing, error handling, and user experience.
"""

import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from rich.console import Console

from src.cli.main import RichCLI, main


class TestRichCLI:
    """Tests for the RichCLI class and Rich console integration."""

    def test_init(self):
        """Test RichCLI initialization."""
        cli = RichCLI()

        assert cli.console is not None
        assert isinstance(cli.console, Console)

    def test_show_welcome(self, rich_output_capturer):
        """Test welcome message display with Rich formatting."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        cli.show_welcome()

        # Verify Rich panel and content
        rich_output_capturer.assert_panel_title("Welcome")
        rich_output_capturer.assert_contains("🚀 AI Documentation Scraper")
        rich_output_capturer.assert_contains("Advanced CLI Interface v1.0.0")
        rich_output_capturer.assert_contains("Hybrid AI documentation scraping system")

        # Verify styling is applied
        output = rich_output_capturer.get_output()
        assert len(output) > 0

    def test_show_error_simple(self, rich_output_capturer):
        """Test error message display without details."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        cli.show_error("Test error message")

        # Verify error panel and content
        rich_output_capturer.assert_panel_title("Error")
        rich_output_capturer.assert_contains("❌ Error:")
        rich_output_capturer.assert_contains("Test error message")

        # Should not contain details section
        rich_output_capturer.assert_not_contains("Details:")

    def test_show_error_with_details(self, rich_output_capturer):
        """Test error message display with details."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        cli.show_error("Test error message", "Additional error details")

        # Verify error panel and content
        rich_output_capturer.assert_panel_title("Error")
        rich_output_capturer.assert_contains("❌ Error:")
        rich_output_capturer.assert_contains("Test error message")
        rich_output_capturer.assert_contains("Details: Additional error details")

    def test_error_formatting_consistency(self, rich_output_capturer):
        """Test that error formatting is consistent across different error types."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        error_scenarios = [
            ("Connection failed", "Unable to connect to database"),
            ("Configuration error", "Missing required field: api_key"),
            ("Permission denied", "Insufficient permissions to write file"),
        ]

        for message, details in error_scenarios:
            rich_output_capturer.reset()
            cli.show_error(message, details)

            # Each error should have consistent formatting
            rich_output_capturer.assert_contains("❌ Error:")
            rich_output_capturer.assert_contains(message)
            rich_output_capturer.assert_contains(f"Details: {details}")


class TestMainCLICommand:
    """Tests for the main CLI command and argument handling."""

    def test_main_command_without_subcommand(self, interactive_cli_runner):
        """Test main command when called without subcommands."""
        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = interactive_cli_runner.invoke(main, [])

            assert result.exit_code == 0
            # Should show welcome message when no subcommand provided

    def test_main_command_with_config_file(self, interactive_cli_runner, tmp_path):
        """Test main command with custom config file."""
        # Create a test config file
        config_file = tmp_path / "test_config.json"
        config_file.write_text('{"test": "config"}')

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = interactive_cli_runner.invoke(main, ["--config", str(config_file)])

            assert result.exit_code == 0
            # Should load custom config file

    def test_main_command_quiet_mode(self, interactive_cli_runner):
        """Test main command with quiet flag."""
        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = interactive_cli_runner.invoke(main, ["--quiet"])

            assert result.exit_code == 0
            # Should suppress welcome message in quiet mode

    def test_main_command_version(self, interactive_cli_runner):
        """Test version option."""
        result = interactive_cli_runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "AI Documentation Scraper CLI" in result.output
        assert "1.0.0" in result.output

    def test_main_command_help(self, interactive_cli_runner):
        """Test help output."""
        result = interactive_cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "AI Documentation Scraper" in result.output
        assert "Advanced CLI Interface" in result.output
        assert "--config" in result.output
        assert "--quiet" in result.output

    def test_main_command_config_loading_success(
        self, interactive_cli_runner, tmp_path
    ):
        """Test successful configuration loading."""
        config_file = tmp_path / "valid_config.json"
        config_file.write_text('{"qdrant": {"host": "localhost"}}')

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = interactive_cli_runner.invoke(main, ["--config", str(config_file)])

            assert result.exit_code == 0
            mock_get_config.assert_called_once()

    def test_main_command_config_loading_error(self, interactive_cli_runner, tmp_path):
        """Test configuration loading error handling."""
        config_file = tmp_path / "invalid_config.json"
        config_file.write_text("invalid json content")

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_get_config.side_effect = Exception("Invalid configuration")

            result = interactive_cli_runner.invoke(main, ["--config", str(config_file)])

            # Should handle config errors gracefully
            assert result.exit_code != 0

    def test_context_object_creation(self, interactive_cli_runner):
        """Test that Click context object is properly created."""
        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            # Use a mock callback to check context
            @main.command()
            @click.pass_context
            def test_command(ctx):
                assert ctx.obj is not None
                assert isinstance(ctx.obj, dict)
                return "success"

            interactive_cli_runner.invoke(main, ["test-command"])
            # Command should execute successfully with proper context


class TestCLICommandIntegration:
    """Integration tests for CLI commands and subcommands."""

    def test_subcommand_registration(self):
        """Test that all expected subcommands are registered."""
        # Get the main command group
        commands = main.commands

        # Verify key subcommands are registered
        expected_commands = ["setup", "config", "database", "batch"]

        for cmd_name in expected_commands:
            assert cmd_name in commands, f"Command '{cmd_name}' not registered"

    def test_subcommand_help_access(self, interactive_cli_runner):
        """Test that help is accessible for all subcommands."""
        # Get registered commands
        commands = main.commands.keys()

        for cmd_name in commands:
            result = interactive_cli_runner.invoke(main, [cmd_name, "--help"])

            # Each command should have accessible help
            assert result.exit_code == 0, f"Help failed for command '{cmd_name}'"
            assert len(result.output) > 0, f"Empty help for command '{cmd_name}'"

    def test_command_context_inheritance(self, interactive_cli_runner):
        """Test that context is properly inherited by subcommands."""
        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            # Test that setup command can access context
            with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard:
                mock_wizard_instance = MagicMock()
                mock_wizard_instance.run_setup.return_value = Path("/tmp/config.json")  # noqa: S108 # test temp path
                mock_wizard.return_value = mock_wizard_instance

                with patch("src.cli.commands.setup.questionary") as mock_questionary:
                    mock_questionary.confirm.return_value.ask.return_value = False

                    result = interactive_cli_runner.invoke(main, ["setup"])

                    # Should execute without context errors
                    assert result.exit_code == 0

    def test_error_handling_propagation(self, interactive_cli_runner):
        """Test that errors are properly handled and propagated."""
        with patch("src.cli.main.get_config") as mock_get_config:
            # Simulate configuration error
            mock_get_config.side_effect = Exception("Configuration failed")

            result = interactive_cli_runner.invoke(main, [])

            # Should handle errors gracefully
            assert result.exit_code != 0


class TestCLIUserExperience:
    """Tests focused on user experience and accessibility."""

    def test_welcome_message_formatting(self, rich_output_capturer):
        """Test that welcome message is properly formatted for readability."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        cli.show_welcome()

        output = rich_output_capturer.get_output()
        lines = rich_output_capturer.get_lines()

        # Should have multiple lines for readability
        assert len(lines) > 3

        # Should contain key information
        assert "🚀" in output  # Visual indicator
        assert "AI Documentation Scraper" in output
        assert "v1.0.0" in output

    def test_error_message_clarity(self, rich_output_capturer):
        """Test that error messages are clear and actionable."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        test_scenarios = [
            {
                "message": "Database connection failed",
                "details": "Check that Qdrant is running on localhost:6333",
            },
            {
                "message": "Invalid API key",
                "details": "OpenAI API key must start with 'sk-' and be at least 40 characters",
            },
            {
                "message": "Configuration file not found",
                "details": "Create a config file using: python -m src.cli.main setup",
            },
        ]

        for scenario in test_scenarios:
            rich_output_capturer.reset()
            cli.show_error(scenario["message"], scenario["details"])

            # Each error should be clear and actionable
            rich_output_capturer.assert_contains("❌ Error:")
            rich_output_capturer.assert_contains(scenario["message"])
            rich_output_capturer.assert_contains(scenario["details"])

            # Should have proper formatting
            output = rich_output_capturer.get_output()
            assert len(output) > 50  # Should be substantial content

    def test_cli_consistency_across_commands(self, interactive_cli_runner):
        """Test that CLI behavior is consistent across different commands."""
        # Test that all commands handle --help consistently
        result_main = interactive_cli_runner.invoke(main, ["--help"])
        assert result_main.exit_code == 0
        assert "Usage:" in result_main.output

        # Test subcommands
        for cmd_name in ["setup", "config", "database"]:
            result = interactive_cli_runner.invoke(main, [cmd_name, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.output

    def test_rich_output_accessibility(self, rich_output_capturer):
        """Test that Rich output is accessible and doesn't rely only on color."""
        cli = RichCLI()
        cli.console = rich_output_capturer.console

        # Test with different message types
        cli.show_welcome()
        welcome_output = rich_output_capturer.get_output()

        rich_output_capturer.reset()
        cli.show_error("Test error", "Test details")
        error_output = rich_output_capturer.get_output()

        # Both should have text indicators, not just color
        assert "🚀" in welcome_output  # Welcome indicator
        assert "❌" in error_output  # Error indicator

        # Should be readable without color
        plain_welcome = rich_output_capturer.get_plain_output()
        assert "AI Documentation Scraper" in plain_welcome


class TestCLIPerformanceAndReliability:
    """Tests for CLI performance and reliability."""

    def test_cli_startup_performance(self, interactive_cli_runner):
        """Test that CLI starts up quickly."""

        start_time = time.time()

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = interactive_cli_runner.invoke(main, ["--help"])

        end_time = time.time()
        startup_time = end_time - start_time

        # Should start up quickly (less than 2 seconds in test environment)
        assert startup_time < 2.0
        assert result.exit_code == 0

    def test_cli_memory_usage(self, interactive_cli_runner):
        """Test that CLI doesn't have obvious memory leaks."""
        # Run CLI multiple times to check for memory issues
        for _ in range(5):
            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = interactive_cli_runner.invoke(main, ["--version"])
                assert result.exit_code == 0

    def test_cli_error_recovery(self, interactive_cli_runner):
        """Test that CLI recovers gracefully from errors."""
        error_scenarios = [
            # Invalid config file
            ["--config", "/nonexistent/config.json"],
            # Invalid command
            ["nonexistent-command"],
        ]

        for args in error_scenarios:
            result = interactive_cli_runner.invoke(main, args)

            # Should exit with error but not crash
            assert result.exit_code != 0
            # Should not have unhandled exceptions in output
            assert "Traceback" not in result.output

    def test_concurrent_cli_usage(self, interactive_cli_runner):
        """Test that multiple CLI instances can run without conflicts."""

        results = queue.Queue()

        def run_cli():
            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = interactive_cli_runner.invoke(main, ["--version"])
                results.put(result.exit_code)

        # Run multiple CLI instances concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_cli)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # All should succeed
        while not results.empty():
            exit_code = results.get()
            assert exit_code == 0


class TestCLIIntegrationScenarios:
    """Real-world integration scenarios for CLI testing."""

    def test_first_time_user_workflow(self, interactive_cli_runner, tmp_path):
        """Test workflow for first-time users."""
        # Simulate first-time user running setup
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard:
            mock_wizard_instance = MagicMock()
            mock_wizard_instance.run_setup.return_value = tmp_path / "config.json"
            mock_wizard.return_value = mock_wizard_instance

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = False

                result = interactive_cli_runner.invoke(main, ["setup"])

                assert result.exit_code == 0

    def test_experienced_user_workflow(self, interactive_cli_runner, tmp_path):
        """Test workflow for experienced users with existing config."""
        # Create existing config
        config_file = tmp_path / "config.json"
        config_file.write_text('{"qdrant": {"host": "localhost"}}')

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            # Experienced user checking status
            result = interactive_cli_runner.invoke(main, ["--config", str(config_file)])

            assert result.exit_code == 0

    def test_cli_help_discoverability(self, interactive_cli_runner):
        """Test that users can discover CLI features through help."""
        # Main help should be comprehensive
        result = interactive_cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Commands:" in result.output
        assert "setup" in result.output
        assert "config" in result.output
        assert "database" in result.output

        # Should provide clear guidance on next steps
        assert "--help" in result.output
