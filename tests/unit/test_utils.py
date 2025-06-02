"""Unit tests for src/utils.py module."""

import asyncio

# Import directly from the utils.py module at src level
import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest

# Add src to path if needed
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from utils.py file directly
import importlib.util

utils_spec = importlib.util.spec_from_file_location("utils", src_path + "/utils.py")
utils_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_module)

async_command = utils_module.async_command
async_to_sync_click = utils_module.async_to_sync_click


class TestAsyncToSyncClick:
    """Test cases for async_to_sync_click function."""

    def test_converts_async_commands_to_sync(self):
        """Test that async command callbacks are converted to sync."""

        @click.group()
        def cli():
            pass

        async def async_callback():
            return "async result"

        # Create a mock command with async callback
        command = click.Command("test", callback=async_callback)
        cli.add_command(command)

        # Verify the callback is initially async
        assert asyncio.iscoroutinefunction(command.callback)

        # Convert to sync
        async_to_sync_click(cli)

        # Verify callback is no longer async
        assert not asyncio.iscoroutinefunction(command.callback)

        # Verify the callback still works and returns the correct result
        result = command.callback()
        assert result == "async result"

    def test_preserves_sync_commands(self):
        """Test that sync commands are left unchanged."""

        @click.group()
        def cli():
            pass

        def sync_callback():
            return "sync result"

        command = click.Command("test", callback=sync_callback)
        cli.add_command(command)
        original_callback = command.callback

        async_to_sync_click(cli)

        # Callback should remain unchanged
        assert command.callback is original_callback
        assert command.callback() == "sync result"

    def test_prevents_double_wrapping(self):
        """Test that commands are not wrapped multiple times."""

        @click.group()
        def cli():
            pass

        async def async_callback():
            return "async result"

        command = click.Command("test", callback=async_callback)
        cli.add_command(command)

        # First conversion
        async_to_sync_click(cli)
        first_callback = command.callback

        # Second conversion should be no-op
        async_to_sync_click(cli)
        second_callback = command.callback

        # Callbacks should be identical
        assert first_callback is second_callback

    def test_marks_group_as_wrapped(self):
        """Test that the group is marked as wrapped to prevent double-wrapping."""

        @click.group()
        def cli():
            pass

        async def async_callback():
            return "result"

        command = click.Command("test", callback=async_callback)
        cli.add_command(command)

        # Initially not marked as wrapped
        assert not hasattr(cli, "_commands_wrapped")

        async_to_sync_click(cli)

        # Should be marked as wrapped
        assert hasattr(cli, "_commands_wrapped")
        assert cli._commands_wrapped is True

    def test_handles_empty_group(self):
        """Test that empty command groups are handled gracefully."""

        @click.group()
        def cli():
            pass

        # Should not raise any errors
        async_to_sync_click(cli)
        assert hasattr(cli, "_commands_wrapped")
        assert cli._commands_wrapped is True

    def test_handles_multiple_async_commands(self):
        """Test conversion of multiple async commands."""

        @click.group()
        def cli():
            pass

        async def async_callback1():
            return "result1"

        async def async_callback2():
            return "result2"

        command1 = click.Command("test1", callback=async_callback1)
        command2 = click.Command("test2", callback=async_callback2)
        cli.add_command(command1)
        cli.add_command(command2)

        async_to_sync_click(cli)

        # Both commands should be converted
        assert not asyncio.iscoroutinefunction(command1.callback)
        assert not asyncio.iscoroutinefunction(command2.callback)

        # Both should work correctly
        assert command1.callback() == "result1"
        assert command2.callback() == "result2"

    def test_callback_args_and_kwargs_preserved(self):
        """Test that callback arguments and keyword arguments are preserved."""

        @click.group()
        def cli():
            pass

        async def async_callback(arg1, arg2, kwarg1=None, kwarg2=None):
            return f"{arg1}-{arg2}-{kwarg1}-{kwarg2}"

        command = click.Command("test", callback=async_callback)
        cli.add_command(command)

        async_to_sync_click(cli)

        result = command.callback("a", "b", kwarg1="c", kwarg2="d")
        assert result == "a-b-c-d"

    def test_callback_metadata_preserved(self):
        """Test that function metadata is preserved after wrapping."""

        @click.group()
        def cli():
            pass

        async def async_callback():
            """Test callback function."""
            return "result"

        async_callback.custom_attr = "custom_value"
        command = click.Command("test", callback=async_callback)
        cli.add_command(command)

        async_to_sync_click(cli)

        # Metadata should be preserved
        assert command.callback.__name__ == "async_callback"
        assert command.callback.__doc__ == "Test callback function."
        assert hasattr(command.callback, "custom_attr")
        assert command.callback.custom_attr == "custom_value"

    def test_exception_handling_in_async_callback(self):
        """Test that exceptions in async callbacks are properly propagated."""

        @click.group()
        def cli():
            pass

        async def failing_callback():
            raise ValueError("Test error")

        command = click.Command("test", callback=failing_callback)
        cli.add_command(command)

        async_to_sync_click(cli)

        # Exception should be propagated
        with pytest.raises(ValueError, match="Test error"):
            command.callback()


class TestAsyncCommand:
    """Test cases for async_command decorator."""

    def test_converts_async_function_to_sync(self):
        """Test that async function is converted to sync."""

        @async_command
        async def test_func():
            return "async result"

        # Function should no longer be async
        assert not asyncio.iscoroutinefunction(test_func)

        # Should return the correct result
        result = test_func()
        assert result == "async result"

    def test_preserves_function_metadata(self):
        """Test that function metadata is preserved."""

        @async_command
        async def test_func():
            """Test function docstring."""
            return "result"

        test_func.custom_attr = "custom_value"

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."
        assert hasattr(test_func, "custom_attr")
        assert test_func.custom_attr == "custom_value"

    def test_handles_function_arguments(self):
        """Test that function arguments are handled correctly."""

        @async_command
        async def test_func(arg1, arg2, kwarg1=None, kwarg2=None):
            return f"{arg1}-{arg2}-{kwarg1}-{kwarg2}"

        result = test_func("a", "b", kwarg1="c", kwarg2="d")
        assert result == "a-b-c-d"

    def test_handles_positional_arguments(self):
        """Test that positional arguments are handled correctly."""

        @async_command
        async def test_func(*args):
            return "-".join(args)

        result = test_func("a", "b", "c")
        assert result == "a-b-c"

    def test_handles_keyword_arguments(self):
        """Test that keyword arguments are handled correctly."""

        @async_command
        async def test_func(**kwargs):
            return "-".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))

        result = test_func(x="1", y="2", z="3")
        assert result == "x:1-y:2-z:3"

    def test_exception_propagation(self):
        """Test that exceptions are properly propagated."""

        @async_command
        async def failing_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_func()

    def test_return_value_handling(self):
        """Test that various return values are handled correctly."""

        @async_command
        async def return_none():
            return None

        @async_command
        async def return_dict():
            return {"key": "value"}

        @async_command
        async def return_list():
            return [1, 2, 3]

        assert return_none() is None
        assert return_dict() == {"key": "value"}
        assert return_list() == [1, 2, 3]

    def test_async_operations_in_decorated_function(self):
        """Test that async operations work correctly in decorated function."""

        @async_command
        async def async_operations():
            # Simulate async operation
            await asyncio.sleep(0.001)
            return "completed"

        result = async_operations()
        assert result == "completed"

    def test_multiple_decorations(self):
        """Test that the decorator works with multiple function decorations."""

        @async_command
        async def multi_decorated():
            """Multi-decorated function."""
            return "decorated"

        # Add another attribute after decoration
        multi_decorated.test_attr = "test_value"

        result = multi_decorated()
        assert result == "decorated"
        assert multi_decorated.__doc__ == "Multi-decorated function."
        assert multi_decorated.test_attr == "test_value"

    @patch("asyncio.run")
    def test_uses_asyncio_run(self, mock_run):
        """Test that the decorator uses asyncio.run internally."""
        mock_run.return_value = "mocked result"

        @async_command
        async def test_func():
            return "original result"

        result = test_func()

        assert result == "mocked result"
        mock_run.assert_called_once()

    def test_with_click_command(self):
        """Test integration with Click commands."""

        @click.command()
        @async_command
        async def cli_command():
            return "cli result"

        # Should work as a regular Click command
        result = cli_command.callback()
        assert result == "cli result"


class TestTypeHinting:
    """Test cases for type hinting and generic functionality."""

    def test_type_preservation(self):
        """Test that type annotations are preserved where possible."""

        @async_command
        async def typed_func(x: int, y: str) -> str:
            return f"{x}-{y}"

        # Function should still work correctly
        result = typed_func(42, "test")
        assert result == "42-test"

        # Original function metadata should be preserved
        assert typed_func.__name__ == "typed_func"


class TestAsyncToSyncClickIntegration:
    """Integration tests for async_to_sync_click with real Click functionality."""

    def test_click_group_integration(self):
        """Test integration with actual Click group functionality."""

        @click.group()
        def cli():
            """Main CLI group."""
            pass

        @cli.command()
        async def async_subcommand():
            """Async subcommand."""
            return "subcommand result"

        @cli.command()
        def sync_subcommand():
            """Sync subcommand."""
            return "sync result"

        # Convert async commands
        async_to_sync_click(cli)

        # Both commands should work
        async_cmd = cli.commands["async-subcommand"]
        sync_cmd = cli.commands["sync-subcommand"]

        assert async_cmd.callback() == "subcommand result"
        assert sync_cmd.callback() == "sync result"

        # Group should be marked as wrapped
        assert hasattr(cli, "_commands_wrapped")

    def test_nested_groups(self):
        """Test handling of nested command groups."""

        @click.group()
        def main():
            pass

        @main.group()
        def sub():
            pass

        async def async_callback():
            return "nested result"

        nested_command = click.Command("nested", callback=async_callback)
        sub.add_command(nested_command)

        # Convert only the sub-group
        async_to_sync_click(sub)

        # Nested command should be converted
        assert not asyncio.iscoroutinefunction(nested_command.callback)
        assert nested_command.callback() == "nested result"

        # Main group should not be marked as wrapped
        assert not hasattr(main, "_commands_wrapped")
        # Sub group should be marked as wrapped
        assert hasattr(sub, "_commands_wrapped")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_async_to_sync_with_malformed_command(self):
        """Test handling of commands with None callback."""

        @click.group()
        def cli():
            pass

        # Create command with None callback
        command = click.Command("test", callback=None)
        cli.add_command(command)

        # Should not raise error
        async_to_sync_click(cli)

        assert hasattr(cli, "_commands_wrapped")

    def test_async_command_with_coroutine_not_awaited_warning(self):
        """Test that proper async handling prevents coroutine warnings."""

        @async_command
        async def test_func():
            # This would normally create a coroutine that needs to be awaited
            await asyncio.sleep(0.001)
            return "success"

        # Should not generate any warnings and should work correctly
        result = test_func()
        assert result == "success"
