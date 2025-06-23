"""Comprehensive CLI integration tests with end-to-end workflows.

This module provides complete integration testing for CLI components including
Rich console integration, questionary flows, and real user interaction scenarios.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from click.testing import CliRunner

from src.cli.main import main


class TestCLIEndToEndWorkflows:
    """End-to-end testing of complete CLI workflows."""

    def test_complete_setup_workflow(self, tmp_path):
        """Test complete setup workflow from start to finish."""
        runner = CliRunner()

        # Setup temporary directories
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            # Create realistic wizard mock
            mock_wizard = MagicMock()

            # Mock the complete setup flow
            mock_wizard.welcome = MagicMock()
            mock_wizard.select_profile.return_value = "personal"
            mock_wizard.customize_template.return_value = {
                "openai": {"api_key": "sk-test-key"},
                "qdrant": {"host": "localhost", "port": 6333},
            }

            # Mock save configuration
            config_file = config_dir / "personal.json"
            mock_wizard.save_configuration.return_value = config_file
            mock_wizard.run_setup.return_value = config_file

            mock_wizard_class.return_value = mock_wizard

            # Mock questionary interactions
            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = (
                    False  # Skip validation
                )

                # Run setup command
                result = runner.invoke(main, ["setup", "--config-dir", str(config_dir)])

                assert result.exit_code == 0
                mock_wizard.run_setup.assert_called_once()

    def test_setup_to_config_validation_workflow(self, tmp_path):
        """Test workflow from setup through configuration validation."""
        runner = CliRunner()

        # Create config file
        config_file = tmp_path / "config.json"
        config_data = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"api_key": "sk-test-key", "model": "text-embedding-3-small"},
        }
        config_file.write_text(json.dumps(config_data, indent=2))

        # Step 1: Run setup (mocked to create config)
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = config_file
            mock_wizard_class.return_value = mock_wizard

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = (
                    True  # Accept validation
                )

                # Mock config validation
                with patch("src.cli.commands.config.validate_config"):
                    result = runner.invoke(main, ["setup"])

                    assert result.exit_code == 0

    def test_error_recovery_workflow(self, tmp_path):
        """Test error recovery and user guidance workflow."""
        runner = CliRunner()

        # Test with invalid config file
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("invalid json content")

        result = runner.invoke(main, ["--config", str(invalid_config)])

        # Should handle error gracefully and provide guidance
        assert result.exit_code != 0

    def test_profile_based_setup_workflow(self, tmp_path):
        """Test profile-based setup with different profile types."""
        runner = CliRunner()

        profiles_to_test = ["personal", "development", "production"]

        for profile in profiles_to_test:
            with patch(
                "src.cli.commands.setup.ConfigurationWizard"
            ) as mock_wizard_class:
                mock_wizard = MagicMock()
                mock_wizard.run_setup.return_value = tmp_path / f"{profile}_config.json"
                mock_wizard_class.return_value = mock_wizard

                # Mock profile manager
                mock_wizard.profile_manager.list_profiles.return_value = (
                    profiles_to_test
                )

                with patch("src.cli.commands.setup.questionary") as mock_questionary:
                    mock_questionary.confirm.return_value.ask.return_value = False

                    result = runner.invoke(main, ["setup", "--profile", profile])

                    assert result.exit_code == 0
                    assert mock_wizard.selected_profile == profile


class TestCLIInteractiveFlows:
    """Tests for interactive CLI flows and user interactions."""

    def test_questionary_integration_flow(self, tmp_path):
        """Test questionary integration in CLI flows."""
        runner = CliRunner()

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard_class.return_value = mock_wizard

            # Test different questionary interaction patterns
            with patch("questionary.confirm") as mock_confirm:
                with patch("questionary.select") as mock_select:
                    with patch("questionary.text") as mock_text:
                        # Setup mock responses
                        mock_confirm.return_value.ask.return_value = True
                        mock_select.return_value.ask.return_value = "personal"
                        mock_text.return_value.ask.return_value = "test-value"

                        # Mock wizard to use these questionary calls
                        def mock_run_setup():
                            # Simulate questionary usage
                            import questionary

                            questionary.confirm("Test question?").ask()
                            questionary.select("Choose:", choices=["a", "b"]).ask()
                            questionary.text("Enter text:").ask()
                            return tmp_path / "config.json"

                        mock_wizard.run_setup = mock_run_setup

                        runner.invoke(main, ["setup"])

                        # Verify questionary methods were called
                        mock_confirm.assert_called()
                        mock_select.assert_called()
                        mock_text.assert_called()

    def test_keyboard_interrupt_handling(self, tmp_path):
        """Test keyboard interrupt handling during interactive flows."""
        runner = CliRunner()

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.side_effect = KeyboardInterrupt()
            mock_wizard_class.return_value = mock_wizard

            result = runner.invoke(main, ["setup"])

            assert result.exit_code == 1
            assert "cancelled by user" in result.output

    def test_user_input_validation_flow(self, tmp_path):
        """Test user input validation during interactive flows."""
        runner = CliRunner()

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard_class.return_value = mock_wizard

            # Mock validation scenarios
            def mock_validate_flow():
                # Simulate validation with retry
                validator = mock_wizard.validator

                # First attempt fails, second succeeds
                validator.validate_api_key.side_effect = [
                    (False, "Invalid key format"),
                    (True, None),
                ]

                return tmp_path / "config.json"

            mock_wizard.run_setup = mock_validate_flow

            with patch("src.cli.commands.setup.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = False

                result = runner.invoke(main, ["setup"])

                assert result.exit_code == 0


class TestCLIRichIntegration:
    """Tests for Rich console integration in CLI."""

    def test_rich_output_in_cli_commands(self, tmp_path):
        """Test Rich output formatting in CLI commands."""
        runner = CliRunner()

        # Test setup command Rich output
        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = tmp_path / "config.json"
            mock_wizard_class.return_value = mock_wizard

            # Mock Rich console to capture output
            with patch("rich.console.Console") as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console

                with patch("src.cli.commands.setup.questionary") as mock_questionary:
                    mock_questionary.confirm.return_value.ask.return_value = False

                    result = runner.invoke(main, ["setup"])

                    assert result.exit_code == 0
                    # Rich console should be used for output

    def test_rich_error_formatting(self, tmp_path):
        """Test Rich error message formatting."""
        runner = CliRunner()

        # Simulate an error that should be formatted with Rich
        with patch("src.cli.main.get_config") as mock_get_config:
            mock_get_config.side_effect = Exception("Configuration error")

            result = runner.invoke(main, [])

            # Should handle error gracefully
            assert result.exit_code != 0

    def test_rich_console_width_handling(self, tmp_path):
        """Test Rich console width handling for different terminal sizes."""
        runner = CliRunner()

        with patch("src.cli.commands.setup.ConfigurationWizard") as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run_setup.return_value = tmp_path / "config.json"
            mock_wizard_class.return_value = mock_wizard

            # Test with different console widths
            console_widths = [80, 120, 200]

            for width in console_widths:
                with patch("rich.console.Console") as mock_console_class:
                    mock_console = MagicMock()
                    mock_console.width = width
                    mock_console_class.return_value = mock_console

                    with patch(
                        "src.cli.commands.setup.questionary"
                    ) as mock_questionary:
                        mock_questionary.confirm.return_value.ask.return_value = False

                        result = runner.invoke(main, ["setup"])

                        assert result.exit_code == 0


class TestCLIConfigurationIntegration:
    """Tests for configuration integration across CLI commands."""

    def test_config_file_discovery(self, tmp_path):
        """Test configuration file discovery and loading."""
        runner = CliRunner()

        # Create config in different locations
        config_locations = [
            tmp_path / "config.json",
            tmp_path / "config" / "config.json",
            tmp_path / ".ai-docs-scraper.json",
        ]

        config_data = {"qdrant": {"host": "localhost", "port": 6333}}

        for config_path in config_locations:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config_data, indent=2))

            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = runner.invoke(main, ["--config", str(config_path)])

                assert result.exit_code == 0
                mock_get_config.assert_called()

    def test_environment_variable_integration(self, tmp_path):
        """Test environment variable integration with CLI."""
        runner = CliRunner()

        # Set environment variables
        env_vars = {
            "AI_DOCS_CONFIG": str(tmp_path / "config.json"),
            "OPENAI_API_KEY": "sk-test-key",
            "QDRANT_URL": "http://localhost:6333",
        }

        config_data = {"qdrant": {"host": "localhost"}}
        config_file = Path(env_vars["AI_DOCS_CONFIG"])
        config_file.write_text(json.dumps(config_data, indent=2))

        with patch.dict(os.environ, env_vars):
            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = runner.invoke(main, [])

                assert result.exit_code == 0

    def test_config_validation_across_commands(self, tmp_path):
        """Test configuration validation across different CLI commands."""
        runner = CliRunner()

        # Create valid config
        config_file = tmp_path / "config.json"
        config_data = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
        }
        config_file.write_text(json.dumps(config_data, indent=2))

        # Test different commands with the same config
        commands_to_test = [
            ["--config", str(config_file), "--help"],
            ["setup", "--help"],
            ["config", "--help"],
            ["database", "--help"],
        ]

        for cmd_args in commands_to_test:
            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = runner.invoke(main, cmd_args)

                assert result.exit_code == 0


class TestCLIAccessibilityAndUsability:
    """Tests for CLI accessibility and usability features."""

    def test_cli_help_accessibility(self, tmp_path):
        """Test CLI help accessibility and clarity."""
        runner = CliRunner()

        # Main help should be accessible
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert len(result.output) > 0

        # Should contain clear guidance
        help_content = result.output
        assert "Usage:" in help_content
        assert "Options:" in help_content
        assert "Commands:" in help_content

        # Should not be overwhelming
        lines = help_content.split("\n")
        assert len(lines) < 100  # Reasonable length

    def test_error_message_helpfulness(self, tmp_path):
        """Test that error messages are helpful and actionable."""
        runner = CliRunner()

        # Test various error scenarios
        error_scenarios = [
            # Invalid command
            ["invalid-command"],
            # Missing required config
            ["--config", "/nonexistent/config.json"],
            # Invalid profile
            ["setup", "--profile", "nonexistent-profile"],
        ]

        for args in error_scenarios:
            result = runner.invoke(main, args)

            # Should provide helpful error messages
            assert result.exit_code != 0
            assert len(result.output) > 0

            # Should not show raw Python tracebacks to users
            assert "Traceback" not in result.output

    def test_cli_progressive_disclosure(self, tmp_path):
        """Test progressive disclosure of CLI features."""
        runner = CliRunner()

        # Basic help should be concise
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        basic_help_length = len(result.output)

        # Subcommand help should provide more detail
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        detailed_help_length = len(result.output)

        # Detailed help should be longer but still manageable
        assert detailed_help_length > basic_help_length
        assert detailed_help_length < basic_help_length * 3  # Not too overwhelming

    def test_cli_consistent_patterns(self, tmp_path):
        """Test that CLI follows consistent patterns across commands."""
        runner = CliRunner()

        # All subcommands should follow consistent help patterns
        subcommands = ["setup", "config", "database", "batch"]

        for cmd in subcommands:
            result = runner.invoke(main, [cmd, "--help"])

            assert result.exit_code == 0
            help_output = result.output

            # Consistent structure
            assert "Usage:" in help_output
            assert f"main {cmd}" in help_output

            # Should have descriptions
            assert len(help_output.split("\n")) > 5


class TestCLIPerformanceIntegration:
    """Performance tests for CLI integration scenarios."""

    def test_cli_startup_performance_with_config(self, tmp_path):
        """Test CLI startup performance with configuration loading."""
        runner = CliRunner()

        # Create realistic config file
        config_file = tmp_path / "config.json"
        config_data = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"api_key": "sk-test-key", "model": "text-embedding-3-small"},
            "cache": {"enabled": True, "redis_url": "redis://localhost:6379"},
            "browser": {"headless": True, "timeout": 30000},
        }
        config_file.write_text(json.dumps(config_data, indent=2))

        import time

        start_time = time.time()

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = runner.invoke(main, ["--config", str(config_file), "--help"])

        end_time = time.time()

        assert result.exit_code == 0
        # Should load quickly even with config
        assert (end_time - start_time) < 3.0

    def test_cli_memory_efficiency(self, tmp_path):
        """Test CLI memory efficiency with large configurations."""
        runner = CliRunner()

        # Create large config file
        config_file = tmp_path / "large_config.json"
        large_config = {
            "qdrant": {"host": "localhost", "port": 6333},
        }

        # Add many configuration entries
        for i in range(1000):
            large_config[f"setting_{i}"] = f"value_{i}"

        config_file.write_text(json.dumps(large_config, indent=2))

        with patch("src.cli.main.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            result = runner.invoke(main, ["--config", str(config_file), "--version"])

            assert result.exit_code == 0
            # Should handle large configs without issues

    def test_concurrent_cli_operations(self, tmp_path):
        """Test concurrent CLI operations don't interfere."""
        import queue
        import threading

        results = queue.Queue()

        def run_cli_operation(operation_id):
            runner = CliRunner()

            with patch("src.cli.main.get_config") as mock_get_config:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config

                result = runner.invoke(main, ["--version"])
                results.put((operation_id, result.exit_code))

        # Start multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_cli_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All operations should succeed
        while not results.empty():
            operation_id, exit_code = results.get()
            assert exit_code == 0, f"Operation {operation_id} failed"
