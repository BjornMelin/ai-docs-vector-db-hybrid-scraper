"""Unit tests for retry, validation, and MCP error decorators."""

from __future__ import annotations

import asyncio

import pytest

from src.services.errors import (
    ConfigurationError,
    NetworkError,
    ToolError,
    ValidationError,
    handle_mcp_errors,
    retry_async,
    validate_input,
)


@pytest.mark.asyncio
async def test_retry_async_retries_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise NetworkError("temporary outage")
        return "ok"

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @retry_async(max_attempts=3, base_delay=0.1, backoff_factor=2.0)
    async def decorated() -> str:
        return await flaky()

    result = await decorated()

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_retry_async_stops_on_non_retryable() -> None:
    @retry_async(max_attempts=3, exceptions=(NetworkError,))
    async def boom() -> None:
        raise ConfigurationError("bad config")

    with pytest.raises(ConfigurationError):
        await boom()


@pytest.mark.asyncio
async def test_handle_mcp_errors_passthrough() -> None:
    @handle_mcp_errors
    async def success() -> dict[str, object]:
        return {"success": True, "data": 42}

    assert await success() == {"success": True, "data": 42}


@pytest.mark.asyncio
async def test_handle_mcp_errors_masks_internal() -> None:
    @handle_mcp_errors
    async def failure() -> None:
        raise RuntimeError("secret info")

    result = await failure()
    assert result["success"] is False
    assert result["error"] == "Internal server error"
    assert result["error_type"] == "internal"


@pytest.mark.asyncio
async def test_handle_mcp_errors_preserves_tool_error() -> None:
    @handle_mcp_errors
    async def failure() -> None:
        raise ToolError("invalid request", error_code="tool")

    result = await failure()
    assert result["error"] == "invalid request"
    assert result["error_type"] == "tool"


@pytest.mark.asyncio
async def test_validate_input_applies_validators() -> None:
    calls: list[int] = []

    def ensure_positive(value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        calls.append(value)
        return value

    @validate_input(count=ensure_positive)
    async def double(count: int) -> int:
        return count * 2

    assert await double(3) == 6
    assert calls == [3]


@pytest.mark.asyncio
async def test_validate_input_raises_validation_error() -> None:
    def ensure_positive(value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @validate_input(count=ensure_positive)
    async def identity(count: int) -> int:
        return count

    with pytest.raises(ValidationError):
        await identity(0)
