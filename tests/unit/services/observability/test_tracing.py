"""Tests for tracing helpers."""

import asyncio

import pytest

from src.services.observability.tracing import (
    ConfigOperationType,
    instrument_config_operation,
    trace_function,
)


def test_trace_function_sync() -> None:
    """Test tracing a synchronous function."""
    calls: list[int] = []

    @trace_function("example.span")
    def wrapped(value: int) -> int:
        calls.append(value)
        return value * 2

    assert wrapped(3) == 6
    assert calls == [3]


@pytest.mark.asyncio
async def test_trace_function_async() -> None:
    """Test tracing an asynchronous function."""
    calls: list[int] = []

    @trace_function("example.async")
    async def wrapped(value: int) -> int:
        calls.append(value)
        await asyncio.sleep(0)
        return value * 2

    assert await wrapped(2) == 4
    assert calls == [2]


def test_instrument_config_operation() -> None:
    """Test instrumenting a config operation."""
    calls: list[str] = []

    @instrument_config_operation(operation_type=ConfigOperationType.UPDATE)
    def reload_config() -> None:
        calls.append("called")

    reload_config()
    assert calls == ["called"]
