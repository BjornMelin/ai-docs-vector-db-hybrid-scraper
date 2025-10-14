"""Asynchronous helper utilities for CLI bindings.

This module provides helpers to adapt asynchronous Click command handlers to
synchronous callbacks without relying on runtime import tricks. All helpers are
pure functions and do not mutate global import state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any
from weakref import WeakSet

import click


__all__ = ["async_command", "async_to_sync_click"]

_wrapped_cli_groups: WeakSet[click.Group] = WeakSet()


def async_to_sync_click(cli_group: click.Group) -> None:
    """Convert async Click command callbacks to synchronous functions.

    The conversion happens in-place and is idempotent for a given command group.

    Args:
        cli_group: Click group whose commands should be wrapped.
    """
    if cli_group in _wrapped_cli_groups:
        return

    for command in cli_group.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback

            def _make_sync(
                func: Callable[..., Coroutine[Any, Any, Any]],
            ) -> Callable[..., Any]:
                @wraps(func)
                def sync_callback(*args: Any, **kwargs: Any) -> Any:
                    return asyncio.run(func(*args, **kwargs))

                return sync_callback

            command.callback = _make_sync(original_callback)

    _wrapped_cli_groups.add(cli_group)


def async_command(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
    """Wrap an async Click command so it runs via ``asyncio.run``.

    Args:
        func: Asynchronous callback.

    Returns:
        Callable that executes ``func`` inside a new asyncio event loop.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper
