"""Tests for system health MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from src.mcp_tools.tools.system_health import register_tools
from src.services.health.manager import (
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
)


@pytest.fixture()
def registered_tools(
    mocker: MockerFixture,
) -> tuple[dict[str, Callable[..., Any]], HealthCheckManager]:
    """Register the system health tools with mocked dependencies."""

    mock_mcp = mocker.MagicMock()
    tools: dict[str, Callable[..., Any]] = {}

    def capture(func: Callable[..., Any]) -> Callable[..., Any]:
        tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture

    health_manager = mocker.MagicMock(spec=HealthCheckManager)
    client_manager = mocker.MagicMock()
    client_manager.get_health_manager.return_value = health_manager

    register_tools(mock_mcp, client_manager)
    return tools, cast(HealthCheckManager, health_manager)


@pytest.fixture()
def mock_context(mocker: MockerFixture):
    """Provide a stub MCP context."""

    ctx = mocker.MagicMock()
    ctx.info = mocker.AsyncMock()
    ctx.error = mocker.AsyncMock()
    return ctx


@pytest.mark.asyncio()
async def test_get_system_health_uses_manager(
    registered_tools: tuple[dict[str, Callable[..., Any]], HealthCheckManager],
    mock_context,
    mocker: MockerFixture,
) -> None:
    """Overall health should be retrieved from the health manager."""

    tools, health_manager = registered_tools
    health_manager.get_overall_health = mocker.AsyncMock(
        return_value={"overall_status": "healthy"}
    )

    result = await tools["get_system_health"](mock_context)

    assert result == {"overall_status": "healthy"}
    health_manager.get_overall_health.assert_awaited_once()
    mock_context.info.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_process_info_returns_check_metadata(
    registered_tools: tuple[dict[str, Callable[..., Any]], HealthCheckManager],
    mock_context,
    mocker: MockerFixture,
) -> None:
    """Process info should surface system resource metadata."""

    tools, health_manager = registered_tools
    resource_snapshot = {
        "psutil_available": True,
        "process": {
            "cpu_percent": 88.0,
            "memory_percent": 79.0,
            "rss_memory_mb": 512.0,
        },
        "system": {"cpu_count_logical": 8},
    }
    mocker.patch(
        "src.mcp_tools.tools.system_health._collect_resource_snapshot",
        return_value=resource_snapshot,
    )
    health_manager.check_single = mocker.AsyncMock(
        return_value=HealthCheckResult(
            name="system_resources",
            status=HealthStatus.DEGRADED,
            message="High CPU",
            duration_ms=10.0,
            metadata={"cpu_percent": 92.0, "memory_percent": 81.0},
        )
    )

    response = await tools["get_process_info"](mock_context)

    assert response["status"] == HealthStatus.DEGRADED.value
    assert response["metrics"]["cpu_percent"] == 92.0
    assert response["metrics"]["rss_memory_mb"] == 512.0
    assert response["resource_snapshot"] == resource_snapshot
    health_manager.check_single.assert_awaited_once_with("system_resources")
    mock_context.info.assert_awaited()


@pytest.mark.asyncio()
async def test_get_system_health_reports_manager_failure(
    mocker: MockerFixture,
    mock_context,
) -> None:
    """Errors retrieving the health manager should surface through the context."""

    mock_mcp = mocker.MagicMock()
    tools: dict[str, Callable[..., Any]] = {}

    def capture(func: Callable[..., Any]) -> Callable[..., Any]:
        tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture

    client_manager = mocker.MagicMock()
    client_manager.get_health_manager.side_effect = RuntimeError("not configured")

    register_tools(mock_mcp, client_manager)

    result = await tools["get_system_health"](mock_context)

    assert result["status"] == HealthStatus.UNKNOWN.value
    mock_context.error.assert_awaited_once_with(
        "Health manager unavailable: not configured"
    )
