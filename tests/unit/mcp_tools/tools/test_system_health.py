"""Tests for system health MCP tools."""

# ruff: noqa: E402
# pylint: disable=wrong-import-position

from __future__ import annotations

import sys
import types
from collections.abc import Callable


def _stub_fastmcp() -> None:
    """Provide a minimal fastmcp module stub for tests."""

    if "fastmcp" in sys.modules:  # pragma: no cover - already available
        return

    module = types.ModuleType("fastmcp")

    class _DummyContext:  # pragma: no cover - used only for typing
        async def info(self, *_args, **_kwargs) -> None:
            return None

        async def error(self, *_args, **_kwargs) -> None:
            return None

    def tool():  # type: ignore
        def decorator(func):
            return func

        return decorator

    module.Context = _DummyContext  # type: ignore[attr-defined]
    module.tool = tool  # type: ignore[attr-defined]
    sys.modules["fastmcp"] = module


_stub_fastmcp()

import pytest

from src.mcp_tools.tools.system_health import register_tools
from src.services.health.manager import (
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
)


@pytest.fixture()
def registered_tools(mocker: pytest.MockFixture) -> dict[str, Callable]:
    """Register the system health tools with mocked dependencies."""

    mock_mcp = mocker.MagicMock()
    tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture

    health_manager = mocker.MagicMock(spec=HealthCheckManager)
    client_manager = mocker.MagicMock()
    client_manager.get_health_manager.return_value = health_manager

    register_tools(mock_mcp, client_manager)
    tools["_health_manager"] = health_manager
    return tools


@pytest.fixture()
def mock_context(mocker: pytest.MockFixture):
    """Provide a stub MCP context."""

    ctx = mocker.MagicMock()
    ctx.info = mocker.AsyncMock()
    ctx.error = mocker.AsyncMock()
    return ctx


@pytest.mark.asyncio()
async def test_get_system_health_uses_manager(
    registered_tools: dict[str, Callable],
    mock_context,
    mocker: pytest.MockFixture,
) -> None:
    """Overall health should be retrieved from the health manager."""

    health_manager: HealthCheckManager = registered_tools.pop("_health_manager")
    health_manager.get_overall_health = mocker.AsyncMock(
        return_value={"overall_status": "healthy"}
    )

    result = await registered_tools["get_system_health"](mock_context)

    assert result == {"overall_status": "healthy"}
    health_manager.get_overall_health.assert_awaited_once()
    mock_context.info.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_process_info_returns_check_metadata(
    registered_tools: dict[str, Callable],
    mock_context,
    mocker: pytest.MockFixture,
) -> None:
    """Process info should surface system resource metadata."""

    health_manager: HealthCheckManager = registered_tools.pop("_health_manager")
    health_manager.check_single = mocker.AsyncMock(
        return_value=HealthCheckResult(
            name="system_resources",
            status=HealthStatus.DEGRADED,
            message="High CPU",
            duration_ms=10.0,
            metadata={"cpu_percent": 92.0, "memory_percent": 81.0},
        )
    )

    response = await registered_tools["get_process_info"](mock_context)

    assert response["status"] == HealthStatus.DEGRADED.value
    assert response["metrics"]["cpu_percent"] == 92.0
    health_manager.check_single.assert_awaited_once_with("system_resources")
    mock_context.info.assert_awaited()
