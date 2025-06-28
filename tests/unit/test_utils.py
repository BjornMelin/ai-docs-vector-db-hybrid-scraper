"""Unit tests for core utils module."""

import asyncio
from unittest.mock import patch

import click
import pytest

from src import utils
from src.utils import async_command, async_to_sync_click


class TestAsyncToSyncClick:
    """Test cases for async_to_sync_click function."""

    def test_async_to_sync_click_basic(self):
        """Test basic conversion of async commands to sync."""
        # Create a mock CLI group
        cli_group = click.Group()

        # Create an async command
        @cli_group.command()
        async def test_command():
            return "async result"

        # Verify it's async before conversion
        assert asyncio.iscoroutinefunction(test_command.callback)

        # Convert to sync
        async_to_sync_click(cli_group)

        # Verify it's no longer async
        assert not asyncio.iscoroutinefunction(test_command.callback)

        # Verify it works
        result = test_command.callback()
        assert result == "async result"

    def test_async_to_sync_click_with_parameters(self):
        """Test conversion with command parameters."""
        cli_group = click.Group()

        @cli_group.command()
        @click.option("--name", default="World")
        async def greet(name):
            return f"Hello, {name}!"

        # Convert to sync
        async_to_sync_click(cli_group)

        # Test with parameters
        result = greet.callback(name="Alice")
        assert result == "Hello, Alice!"

    def test_async_to_sync_click_preserves_sync_commands(self):
        """Test that sync commands are not affected."""
        cli_group = click.Group()

        @cli_group.command()
        def sync_command():
            return "sync result"

        original_callback = sync_command.callback

        # Convert (should not affect sync commands)
        async_to_sync_click(cli_group)

        # Should be the same function
        assert sync_command.callback is original_callback
        assert sync_command.callback() == "sync result"

    def test_async_to_sync_click_prevents_double_wrapping(self):
        """Test that double wrapping is prevented."""
        cli_group = click.Group()

        @cli_group.command()
        async def test_command():
            return "result"

        # First conversion
        async_to_sync_click(cli_group)
        first_callback = test_command.callback

        # Second conversion (should be no-op)
        async_to_sync_click(cli_group)
        second_callback = test_command.callback

        # Should be the same function after second call
        assert first_callback is second_callback
        assert hasattr(cli_group, "_commands_wrapped")

    def test_async_to_sync_click_with_exception(self):
        """Test handling of exceptions in async commands."""
        cli_group = click.Group()

        @cli_group.command()
        async def failing_command():
            raise ValueError("Test error")

        async_to_sync_click(cli_group)

        # Should raise the exception synchronously
        with pytest.raises(ValueError, match="Test error"):
            failing_command.callback()

    def test_async_to_sync_click_empty_group(self):
        """Test with empty command group."""
        cli_group = click.Group()

        # Should not raise any errors
        async_to_sync_click(cli_group)

        # Should be marked as wrapped
        assert hasattr(cli_group, "_commands_wrapped")

    def test_async_to_sync_click_mixed_commands(self):
        """Test with mixed async and sync commands."""
        cli_group = click.Group()

        @cli_group.command()
        async def async_cmd():
            return "async"

        @cli_group.command()
        def sync_cmd():
            return "sync"

        # Track original sync command
        original_sync_callback = sync_cmd.callback

        async_to_sync_click(cli_group)

        # Async command should be converted
        assert not asyncio.iscoroutinefunction(async_cmd.callback)
        assert async_cmd.callback() == "async"

        # Sync command should be unchanged
        assert sync_cmd.callback is original_sync_callback
        assert sync_cmd.callback() == "sync"

    @patch("asyncio.run")
    def test_async_to_sync_click_uses_asyncio_run(self, mock_run):
        """Test that converted commands use asyncio.run."""
        mock_run.return_value = "mocked result"

        cli_group = click.Group()

        @cli_group.command()
        async def test_command():
            return "original result"

        async_to_sync_click(cli_group)

        # Call the converted command
        result = test_command.callback()

        # Should have called asyncio.run
        assert mock_run.called
        assert result == "mocked result"

    def test_async_to_sync_click_preserves_function_metadata(self):
        """Test that function metadata is preserved."""
        cli_group = click.Group()

        @cli_group.command()
        async def documented_command():
            """This is a test command."""
            return "result"

        original_name = documented_command.callback.__name__
        original_doc = documented_command.callback.__doc__

        async_to_sync_click(cli_group)

        # Metadata should be preserved
        assert documented_command.callback.__name__ == original_name
        assert documented_command.callback.__doc__ == original_doc


class TestAsyncCommand:
    """Test cases for async_command decorator."""

    def test_async_command_basic(self):
        """Test basic async_command decorator."""

        @async_command
        async def test_func():
            return "async result"

        # Should no longer be a coroutine function
        assert not asyncio.iscoroutinefunction(test_func)

        # Should return the result directly
        result = test_func()
        assert result == "async result"

    def test_async_command_with_parameters(self):
        """Test async_command with parameters."""

        @async_command
        async def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    def test_async_command_with_exception(self):
        """Test async_command with exceptions."""

        @async_command
        async def failing_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_func()

    @patch("asyncio.run")
    def test_async_command_uses_asyncio_run(self, mock_run):
        """Test that async_command uses asyncio.run."""
        mock_run.return_value = "mocked result"

        @async_command
        async def test_func():
            return "original result"

        result = test_func()

        assert mock_run.called
        assert result == "mocked result"

    def test_async_command_preserves_metadata(self):
        """Test that async_command preserves function metadata."""

        @async_command
        async def documented_func():
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    def test_async_command_with_args_and_kwargs(self):
        """Test async_command with mixed arguments."""

        @async_command
        async def complex_func(pos1, pos2, kw1=None, kw2="default"):
            return f"{pos1}-{pos2}-{kw1}-{kw2}"

        result = complex_func("a", "b", kw1="c", kw2="d")
        assert result == "a-b-c-d"

    def test_async_command_return_type_preservation(self):
        """Test that return types are preserved."""

        @async_command
        async def return_dict():
            return {"key": "value", "number": 42}

        @async_command
        async def return_list():
            return [1, 2, 3]

        @async_command
        async def return_none():
            return None

        assert return_dict() == {"key": "value", "number": 42}
        assert return_list() == [1, 2, 3]
        assert return_none() is None

    def test_async_command_with_awaitable_operations(self):
        """Test async_command with actual async operations."""

        @async_command
        async def async_sleep_func():
            await asyncio.sleep(0.01)  # Very short sleep for testing
            return "slept"

        result = async_sleep_func()
        assert result == "slept"


class TestModuleExports:
    """Test cases for module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""

        assert hasattr(utils, "__all__")
        assert "async_command" in utils.__all__
        assert "async_to_sync_click" in utils.__all__

    def test_exported_functions_exist(self):
        """Test that all exported functions exist."""

        for export in utils.__all__:
            assert hasattr(utils, export)
            assert callable(getattr(utils, export))


class TestIntegration:
    """Integration tests combining different utilities."""

    def test_async_command_with_click_integration(self):
        """Test async_command decorator with Click commands."""

        @click.command()
        @async_command
        async def cli_command():
            await asyncio.sleep(0.01)
            return "CLI result"

        # Should work as a regular sync function for Click
        result = cli_command.callback()
        assert result == "CLI result"

    def test_mixed_conversion_approaches(self):
        """Test using both conversion approaches on the same functions."""
        # Create a command that will be converted via async_to_sync_click
        cli_group = click.Group()

        @cli_group.command()
        async def group_command():
            return "group result"

        # Create a standalone command with async_command decorator
        @async_command
        async def standalone_command():
            return "standalone result"

        # Convert the group
        async_to_sync_click(cli_group)

        # Both should work as sync functions
        assert group_command.callback() == "group result"
        assert standalone_command() == "standalone result"

    def test_error_handling_consistency(self):
        """Test that error handling is consistent between approaches."""
        cli_group = click.Group()

        @cli_group.command()
        async def group_error():
            raise ValueError("Group error")

        @async_command
        async def standalone_error():
            raise ValueError("Standalone error")

        async_to_sync_click(cli_group)

        # Both should raise the same type of exception
        with pytest.raises(ValueError, match="Group error"):
            group_error.callback()

        with pytest.raises(ValueError, match="Standalone error"):
            standalone_error()
