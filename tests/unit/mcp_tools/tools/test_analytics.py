"""Tests for analytics MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.mcp_tools.models.requests import AnalyticsRequest
from src.mcp_tools.tools.analytics import register_tools


@pytest.fixture
def mock_vector_service() -> Mock:
    service = Mock()
    service.is_initialized.return_value = True
    service.list_collections = AsyncMock(return_value=["c1", "c2"])
    service.collection_stats = AsyncMock(
        side_effect=[{"points_count": 10}, {"points_count": 5}]
    )
    return service


@pytest.fixture
def mock_context() -> Mock:
    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
async def registered_tools(mock_vector_service: Mock) -> dict[str, Callable]:
    mock_mcp = MagicMock()
    tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    register_tools(mock_mcp, vector_service=mock_vector_service)
    return tools


@pytest.mark.asyncio
async def test_get_analytics_includes_performance_and_costs(
    registered_tools: dict[str, Callable],
    mock_context: Mock,
) -> None:
    request = AnalyticsRequest(include_performance=True, include_costs=True)

    response = await registered_tools["get_analytics"](request, mock_context)

    assert response.collections["c1"]["points_count"] == 10
    assert response.performance == {"total_vectors": 15, "collection_count": 2}
    assert "estimated_storage_gb" in response.costs
    mock_context.info.assert_awaited()


@pytest.mark.asyncio
async def test_get_analytics_handles_collection_failure(
    registered_tools: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.list_collections.return_value = ["c1"]
    mock_vector_service.collection_stats.side_effect = RuntimeError("boom")

    request = AnalyticsRequest()

    response = await registered_tools["get_analytics"](request, mock_context)

    assert response.collections == {}
    mock_context.warning.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_system_health_reports_unhealthy(
    registered_tools: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.list_collections.side_effect = RuntimeError("offline")

    response = await registered_tools["get_system_health"](mock_context)

    assert response.status == "unhealthy"
    assert response.services["vector_store"].status == "unhealthy"
    mock_context.info.assert_awaited_once()
