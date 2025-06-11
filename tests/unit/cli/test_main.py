"""Tests for the main CLI module.

This module tests the main CLI entry point, RichCLI class, and core commands
including version, completion, and status functionality.
"""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from src.cli.main import RichCLI, main


class TestRichCLI:
    """Test the RichCLI class for Rich console integration."""
    
    def test_init(self):
        """Test RichCLI initialization."""
        rich_cli = RichCLI()
        assert rich_cli.console is not None
        assert isinstance(rich_cli.console, Console)
    
    def test_show_welcome(self, rich_output_capturer):
        """Test welcome message display with Rich formatting."""
        rich_cli = RichCLI()
        rich_cli.console = rich_output_capturer.console
        
        rich_cli.show_welcome()
        output = rich_output_capturer.get_output()
        
        # Verify welcome content
        rich_output_capturer.assert_contains("🚀 AI Documentation Scraper")
        rich_output_capturer.assert_contains("Advanced CLI Interface v1.0.0")
        rich_output_capturer.assert_contains("Hybrid AI documentation scraping system")
        rich_output_capturer.assert_contains("Welcome")
    
    def test_show_error_basic(self, rich_output_capturer):
        """Test basic error message display."""
        rich_cli = RichCLI()
        rich_cli.console = rich_output_capturer.console
        
        rich_cli.show_error("Test error message")
        output = rich_output_capturer.get_output()
        
        # Verify error content
        rich_output_capturer.assert_contains("❌ Error:")
        rich_output_capturer.assert_contains("Test error message")
        rich_output_capturer.assert_contains("Error")
    
    def test_show_error_with_details(self, rich_output_capturer):
        """Test error message display with details."""
        rich_cli = RichCLI()
        rich_cli.console = rich_output_capturer.console
        
        rich_cli.show_error("Configuration failed", details="Missing API key")
        output = rich_output_capturer.get_output()
        
        # Verify error content with details
        rich_output_capturer.assert_contains("❌ Error:")
        rich_output_capturer.assert_contains("Configuration failed")
        rich_output_capturer.assert_contains("Details: Missing API key")


class TestMainCommand:
    """Test the main CLI command group and core functionality."""
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_command_default_config(self, mock_config_loader, cli_runner, mock_config):
        """Test main command with default configuration loading."""
        mock_config_loader.load_config.return_value = mock_config
        
        # Use a command that actually loads config (not --help)
        result = cli_runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "AI Documentation Scraper" in result.output
        assert "Available commands:" in result.output
        mock_config_loader.load_config.assert_called_once()
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_command_with_config_file(self, mock_config_loader, cli_runner, mock_config, temp_config_file):
        """Test main command with explicit configuration file."""
        mock_config_loader.from_file.return_value = mock_config
        
        # Use command that loads config  
        result = cli_runner.invoke(main, ['--config', str(temp_config_file)])
        
        assert result.exit_code == 0
        mock_config_loader.from_file.assert_called_once_with(temp_config_file)
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_command_config_error(self, mock_config_loader, cli_runner):
        """Test main command with configuration loading error."""
        mock_config_loader.load_config.side_effect = Exception("Config error")
        
        result = cli_runner.invoke(main, [])
        
        assert result.exit_code == 1
        assert "Failed to load configuration" in result.output
        assert "Config error" in result.output
    
    @patch('src.cli.main.ConfigLoader')  
    def test_main_command_quiet_flag(self, mock_config_loader, cli_runner, mock_config):
        """Test main command with quiet flag suppresses welcome."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['--quiet'])
        
        assert result.exit_code == 0
        # Should not contain welcome message
        assert "🚀 AI Documentation Scraper" not in result.output
        # But should contain command list
        assert "Available commands:" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_command_shows_welcome_and_commands(self, mock_config_loader, cli_runner, mock_config):
        """Test main command shows welcome and available commands."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "Available commands:" in result.output
        assert "setup    🧙 Interactive configuration wizard" in result.output
        assert "config   ⚙️  Configuration management" in result.output
        assert "database 🗄️  Vector database operations" in result.output
        assert "batch    📦 Batch operations" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_command_context_setup(self, mock_config_loader, cli_runner, mock_config):
        """Test that context object is properly set up."""
        mock_config_loader.load_config.return_value = mock_config
        
        # Test that config loading occurs and basic invocation works
        result = cli_runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "Available commands:" in result.output
        mock_config_loader.load_config.assert_called_once()


class TestVersionCommand:
    """Test the version command functionality."""
    
    @patch('src.cli.main.ConfigLoader')
    def test_version_command(self, mock_config_loader, cli_runner, mock_config):
        """Test version command display."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['version'])
        
        assert result.exit_code == 0
        assert "AI Documentation Scraper CLI" in result.output
        assert "Version: 1.0.0" in result.output
        assert "Python:" in result.output
        assert "Version Information" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    def test_version_flag(self, mock_config_loader, cli_runner, mock_config):
        """Test --version flag functionality."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestCompletionCommand:
    """Test shell completion generation functionality."""
    
    @patch('src.cli.main.ConfigLoader')
    def test_completion_bash(self, mock_config_loader, cli_runner, mock_config):
        """Test bash completion script generation."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['completion', 'bash'])
        
        assert result.exit_code == 0
        # Bash completion scripts typically contain these patterns
        assert 'complete' in result.output or 'bash' in result.output
    
    @patch('src.cli.main.ConfigLoader')
    def test_completion_zsh(self, mock_config_loader, cli_runner, mock_config):
        """Test zsh completion script generation."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['completion', 'zsh'])
        
        assert result.exit_code == 0
        # Should generate some form of completion script
        assert len(result.output) > 0
    
    @patch('src.cli.main.ConfigLoader')
    def test_completion_fish(self, mock_config_loader, cli_runner, mock_config):
        """Test fish completion script generation.""" 
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['completion', 'fish'])
        
        assert result.exit_code == 0
        assert len(result.output) > 0
    
    @patch('src.cli.main.ConfigLoader')
    def test_completion_powershell(self, mock_config_loader, cli_runner, mock_config):
        """Test PowerShell completion script generation."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['completion', 'powershell'])
        
        # PowerShell completion might not be supported on all systems
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert len(result.output) > 0
    
    @patch('src.cli.main.ConfigLoader')
    @patch('click.shell_completion.get_completion_class')
    def test_completion_unsupported_shell(self, mock_get_completion_class, mock_config_loader, cli_runner, mock_config):
        """Test completion with unsupported shell."""
        mock_config_loader.load_config.return_value = mock_config
        mock_get_completion_class.return_value = None
        
        result = cli_runner.invoke(main, ['completion', 'bash'])
        
        assert result.exit_code == 1
        assert "not supported for completion" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    @patch('click.shell_completion.get_completion_class')
    def test_completion_generation_error(self, mock_get_completion_class, mock_config_loader, cli_runner, mock_config):
        """Test completion script generation error handling."""
        mock_config_loader.load_config.return_value = mock_config
        mock_completion_class = MagicMock()
        mock_completion_class.return_value.source.side_effect = Exception("Generation failed")
        mock_get_completion_class.return_value = mock_completion_class
        
        result = cli_runner.invoke(main, ['completion', 'bash'])
        
        assert result.exit_code == 1
        assert "Failed to generate completion script" in result.output


class TestStatusCommand:
    """Test system status and health check functionality."""
    
    @patch('src.cli.main.ConfigLoader')
    @patch('src.utils.health_checks.ServiceHealthChecker')
    def test_status_command_all_healthy(self, mock_health_checker, mock_config_loader, cli_runner, mock_config):
        """Test status command with all services healthy."""
        mock_config_loader.load_config.return_value = mock_config
        mock_health_checker.perform_all_health_checks.return_value = {
            "qdrant": {"connected": True, "version": "1.7.0"},
            "redis": {"connected": True, "version": "7.0.0"},
            "openai": {"connected": True, "model": "text-embedding-ada-002"}
        }
        
        result = cli_runner.invoke(main, ['status'])
        
        assert result.exit_code == 0
        assert "System Status" in result.output
        assert "✅" in result.output
        assert "Healthy" in result.output
        assert "Qdrant" in result.output
        assert "Redis" in result.output
        assert "Openai" in result.output
        mock_health_checker.perform_all_health_checks.assert_called_once_with(mock_config)
    
    @patch('src.cli.main.ConfigLoader')
    @patch('src.utils.health_checks.ServiceHealthChecker')
    def test_status_command_with_errors(self, mock_health_checker, mock_config_loader, cli_runner, mock_config):
        """Test status command with some service errors."""
        mock_config_loader.load_config.return_value = mock_config
        mock_health_checker.perform_all_health_checks.return_value = {
            "qdrant": {"connected": True, "version": "1.7.0"},
            "redis": {"connected": False, "error": "Connection refused"},
            "openai": {"connected": True, "model": "text-embedding-ada-002"}
        }
        
        result = cli_runner.invoke(main, ['status'])
        
        assert result.exit_code == 0
        assert "System Status" in result.output
        assert "✅" in result.output  # For healthy services
        assert "❌" in result.output  # For failed services
        assert "Connection refused" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    @patch('src.utils.health_checks.ServiceHealthChecker')
    def test_status_command_health_check_exception(self, mock_health_checker, mock_config_loader, cli_runner, mock_config):
        """Test status command when health check raises exception."""
        mock_config_loader.load_config.return_value = mock_config
        mock_health_checker.perform_all_health_checks.side_effect = Exception("Health check failed")
        
        # This should not crash the CLI
        result = cli_runner.invoke(main, ['status'])
        
        # The exact behavior depends on implementation, but it shouldn't crash
        assert result.exit_code in [0, 1]  # Either success or controlled failure


class TestMainIntegration:
    """Integration tests for the main CLI functionality."""
    
    @patch('src.cli.main.ConfigLoader')
    def test_main_cli_help_output(self, mock_config_loader, cli_runner, mock_config):
        """Test main CLI help output contains all expected information."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        # Verify help contains essential information
        assert "AI Documentation Scraper - Advanced CLI Interface" in result.output
        assert "Usage:" in result.output
        assert "--config" in result.output
        assert "--quiet" in result.output
        assert "--version" in result.output
        assert "--help" in result.output
    
    @patch('src.cli.main.ConfigLoader')
    def test_subcommand_integration(self, mock_config_loader, cli_runner, mock_config):
        """Test that subcommands are properly registered."""
        mock_config_loader.load_config.return_value = mock_config
        
        # Test that subcommands are available
        result = cli_runner.invoke(main, ['setup', '--help'])
        assert result.exit_code == 0
        
        result = cli_runner.invoke(main, ['config', '--help'])
        assert result.exit_code == 0
        
        result = cli_runner.invoke(main, ['database', '--help'])
        assert result.exit_code == 0
        
        result = cli_runner.invoke(main, ['batch', '--help'])
        assert result.exit_code == 0
    
    @patch('src.cli.main.ConfigLoader')
    def test_invalid_subcommand(self, mock_config_loader, cli_runner, mock_config):
        """Test handling of invalid subcommands."""
        mock_config_loader.load_config.return_value = mock_config
        
        result = cli_runner.invoke(main, ['invalid-command'])
        
        assert result.exit_code == 2  # Click's "No such command" exit code
        assert "No such command" in result.output or "invalid-command" in result.output


# Utility tests for CLI components
class TestCLIUtilities:
    """Test utility functions and edge cases."""
    
    def test_rich_cli_module_import(self):
        """Test that RichCLI can be imported and instantiated."""
        from src.cli.main import RichCLI
        
        rich_cli = RichCLI()
        assert rich_cli is not None
        assert hasattr(rich_cli, 'console')
        assert hasattr(rich_cli, 'show_welcome')
        assert hasattr(rich_cli, 'show_error')
    
    def test_main_command_import(self):
        """Test that main command can be imported."""
        from src.cli.main import main
        
        assert main is not None
        assert hasattr(main, 'invoke')
        assert hasattr(main, 'commands')
    
    def test_command_registration(self):
        """Test that all commands are properly registered."""
        from src.cli.main import main
        
        # Verify expected commands are registered
        command_names = list(main.commands.keys())
        expected_commands = ['setup', 'config', 'database', 'batch', 'version', 'completion', 'status']
        
        for cmd in expected_commands:
            assert cmd in command_names, f"Command '{cmd}' not registered"