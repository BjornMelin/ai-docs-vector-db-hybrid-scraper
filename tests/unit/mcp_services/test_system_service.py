"""Tests for ``SystemService`` using explicit dependency injection."""

from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from src.mcp_services.system_service import SystemService


def test_initializes_with_dependency_managers() -> None:
    """Constructor should register FastMCP tools with supplied managers."""

    embedding_manager = Mock(name="EmbeddingManager")
    health_manager = Mock(name="HealthCheckManager")
    vector_service = Mock(name="VectorStoreService")

    with ExitStack() as stack:
        system_health_register = stack.enter_context(
            patch("src.mcp_services.system_service.system_health.register_tools")
        )
        configuration_register = stack.enter_context(
            patch("src.mcp_services.system_service.configuration.register_tools")
        )
        cost_register = stack.enter_context(
            patch("src.mcp_services.system_service.cost_estimation.register_tools")
        )
        embeddings_register = stack.enter_context(
            patch("src.mcp_services.system_service.embeddings.register_tools")
        )

        service = SystemService(
            "system-custom",
            vector_service=vector_service,
            embedding_manager=embedding_manager,
            health_manager=health_manager,
        )

    assert service.get_mcp_server().name == "system-custom"
    system_health_register.assert_called_once_with(
        service.mcp, health_manager=health_manager
    )
    configuration_register.assert_called_once_with(
        service.mcp, vector_service=vector_service
    )
    cost_register.assert_called_once_with(service.mcp)
    embeddings_register.assert_called_once_with(
        service.mcp, embedding_manager=embedding_manager
    )


@pytest.mark.asyncio()
async def test_get_service_info_returns_expected_payload() -> None:
    """Service info should enumerate available capabilities."""

    vector_service = Mock()
    embedding_manager = Mock()
    health_manager = Mock()
    with ExitStack() as stack:
        stack.enter_context(
            patch("src.mcp_services.system_service.system_health.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.configuration.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.cost_estimation.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.embeddings.register_tools")
        )
        service = SystemService(
            vector_service=vector_service,
            embedding_manager=embedding_manager,
            health_manager=health_manager,
        )

    info = await service.get_service_info()

    assert info == {
        "service": "system",
        "version": "2.0",
        "capabilities": [
            "health_monitoring",
            "resource_management",
            "configuration_management",
            "cost_estimation",
            "embedding_management",
        ],
        "status": "active",
    }


def test_get_mcp_server_returns_fastmcp_instance() -> None:
    """The getter should return the configured FastMCP server."""

    vector_service = Mock()
    embedding_manager = Mock()
    health_manager = Mock()
    with ExitStack() as stack:
        stack.enter_context(
            patch("src.mcp_services.system_service.system_health.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.configuration.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.cost_estimation.register_tools")
        )
        stack.enter_context(
            patch("src.mcp_services.system_service.embeddings.register_tools")
        )
        service = SystemService(
            "system-test",
            vector_service=vector_service,
            embedding_manager=embedding_manager,
            health_manager=health_manager,
        )

    server = service.get_mcp_server()
    assert server.name == "system-test"
    assert hasattr(server, "tool")
