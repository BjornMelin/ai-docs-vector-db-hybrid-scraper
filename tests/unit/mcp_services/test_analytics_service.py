"""Tests for ``AnalyticsService`` using container-free dependencies."""

from unittest.mock import Mock, patch

import pytest

from src.mcp_services.analytics_service import AnalyticsService


def test_initializes_with_vector_service() -> None:
    """Constructor should wire FastMCP tools with the provided vector service."""

    vector_service = Mock(name="VectorStoreService")

    with patch(
        "src.mcp_services.analytics_service.analytics.register_tools"
    ) as register_tools:
        service = AnalyticsService(
            "analytics-custom",
            vector_service=vector_service,
        )

    assert service.get_mcp_server().name == "analytics-custom"
    register_tools.assert_called_once_with(service.mcp, vector_service=vector_service)


@pytest.mark.asyncio()
async def test_get_service_info_returns_expected_payload() -> None:
    """Service info should surface declared capabilities and status."""

    with patch("src.mcp_services.analytics_service.analytics.register_tools"):
        service = AnalyticsService()

    info = await service.get_service_info()

    assert info == {
        "service": "analytics",
        "version": "3.0",
        "capabilities": [
            "collection_analytics",
            "performance_estimates",
            "cost_estimates",
        ],
        "status": "active",
    }


def test_get_mcp_server_exposes_fastmcp_instance() -> None:
    """The getter should return the configured FastMCP server."""

    with patch("src.mcp_services.analytics_service.analytics.register_tools"):
        service = AnalyticsService("analytics-test")

    server = service.get_mcp_server()
    assert server.name == "analytics-test"
    assert hasattr(server, "tool")
