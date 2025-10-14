"""Tests for asynchronous Click helper utilities."""

from __future__ import annotations

import inspect

import click
import pytest
from click.testing import CliRunner

from src.utils import async_utils


@pytest.fixture
def cli_group() -> click.Group:
    """Provide a Click CLI containing async and sync commands."""

    @click.group()
    def cli() -> None:
        """Entry point for the temporary CLI."""

    @cli.command()
    async def greet() -> None:
        """Emit a greeting asynchronously."""
        click.echo("hello")

    @cli.command()
    def farewell() -> None:
        """Emit a farewell synchronously."""
        click.echo("goodbye")

    return cli


def test_async_to_sync_click_executes_async_command(cli_group: click.Group) -> None:
    """Ensure async Click callbacks run successfully after patching."""
    async_utils.async_to_sync_click(cli_group)

    runner = CliRunner()
    result = runner.invoke(cli_group, ["greet"])

    assert result.exit_code == 0
    assert "hello" in result.output


def test_async_to_sync_click_is_idempotent(cli_group: click.Group) -> None:
    """Verify multiple invocations keep callbacks stable."""
    async_utils.async_to_sync_click(cli_group)
    first_callback = cli_group.commands["greet"].callback

    async_utils.async_to_sync_click(cli_group)
    second_callback = cli_group.commands["greet"].callback

    assert first_callback is second_callback
    assert not inspect.iscoroutinefunction(second_callback)


def test_async_to_sync_click_preserves_sync_commands(cli_group: click.Group) -> None:
    """Confirm synchronous callbacks remain untouched."""
    original_callback = cli_group.commands["farewell"].callback

    async_utils.async_to_sync_click(cli_group)

    assert cli_group.commands["farewell"].callback is original_callback


def test_async_command_runs_async_function() -> None:
    """Validate the decorator drives coroutine execution."""

    async def compute(value: int) -> int:
        return value * 2

    wrapped = async_utils.async_command(compute)

    assert wrapped(21) == 42


def test_async_command_propagates_errors() -> None:
    """Ensure exceptions raised inside coroutines bubble up."""

    async def explode() -> None:
        raise ValueError("boom")

    wrapped = async_utils.async_command(explode)

    with pytest.raises(ValueError, match="boom"):
        wrapped()
