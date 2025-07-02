"""Shared utility functions for the AI documentation vector database system."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar
from weakref import WeakSet

import click


F = TypeVar("F", bound=Callable[..., Any])

# Keep track of wrapped CLI groups to prevent double-wrapping
_wrapped_cli_groups: WeakSet[click.Group] = WeakSet()


def async_to_sync_click(cli_group: click.Group) -> None:
    """Convert async Click commands to sync commands for CLI compatibility.

    This utility wraps async command callbacks to run them synchronously,
    preserving all function metadata and avoiding double-wrapping.

    Args:
        cli_group: The Click group containing commands to convert

    """
    # Avoid double-wrapping if already processed
    if cli_group in _wrapped_cli_groups:
        return

    # Convert each async command to sync
    for command in cli_group.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback

            def make_sync_callback(func: Callable[..., Any]) -> Callable[..., Any]:
                @wraps(func)
                def sync_callback(*args: Any, **kwargs: Any) -> Any:
                    return asyncio.run(func(*args, **kwargs))

                return sync_callback

            command.callback = make_sync_callback(original_callback)

    # Mark as wrapped to prevent double-wrapping
    _wrapped_cli_groups.add(cli_group)


def async_command(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to automatically convert async Click commands to sync.

    This can be used as an alternative to the async_to_sync_click function
    for individual commands.

    Args:
        func: The async function to wrap

    Returns:
        A sync version of the function that runs the async code

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper  # type: ignore[return-value]
