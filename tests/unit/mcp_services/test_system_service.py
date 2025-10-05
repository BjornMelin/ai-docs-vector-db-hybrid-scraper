"""Unit tests for SystemService.

Tests cover:
- Service initialization and configuration
- Tool registration behavior with actual modules
- Public API contracts (get_service_info, get_mcp_server)
- Error handling and edge cases
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.system_service import SystemService


class TestSystemServiceInitialization:
    """Test service initialization and configuration."""

    def test_creates_service_with_custom_name(self):
        """Test SystemService initializes with custom name."""
        service = SystemService("test-system-service")

        assert service.mcp.name == "test-system-service"
        assert service.client_manager is None

    def test_creates_service_with_default_name(self):
        """Test SystemService initialization with default name."""
        service = SystemService()

        assert service.mcp.name == "system-service"
        assert service.client_manager is None

    def test_creates_service_with_instructions(self):
        """Test that service has proper instructions."""
        service = SystemService()

        instructions = service.mcp.instructions
        assert instructions is not None
        assert "System service" in instructions
        assert "monitoring" in instructions.lower()
        assert "configuration" in instructions.lower()

    @pytest.mark.asyncio
    async def test_initialize_sets_client_manager(self, mock_client_manager):
        """Test initialize sets client manager correctly."""
        service = SystemService()

        with patch.object(
            service, "_register_system_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_logs_success(self, mock_client_manager, caplog):
        """Test initialization logs success message."""
        service = SystemService()

        with patch.object(service, "_register_system_tools", new_callable=AsyncMock):
            with caplog.at_level(logging.INFO):
                await service.initialize(mock_client_manager)

            assert "SystemService initialized" in caplog.text


class TestSystemServiceToolRegistration:
    """Test tool registration behavior."""

    @pytest.mark.asyncio
    async def test_register_tools_raises_error_when_not_initialized(self):
        """Test tool registration fails without initialization."""
        service = SystemService()

        with pytest.raises(RuntimeError, match="SystemService not initialized"):
            await service._register_system_tools()

    @pytest.mark.asyncio
    async def test_register_tools_calls_all_four_modules(self, mock_client_manager):
        """Test all four tool modules are registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        mock_system_health = Mock()
        mock_configuration = Mock()
        mock_cost_estimation = Mock()
        mock_embeddings = Mock()

        with patch.multiple(
            "src.mcp_services.system_service",
            system_health=mock_system_health,
            configuration=mock_configuration,
            cost_estimation=mock_cost_estimation,
            embeddings=mock_embeddings,
        ):
            await service._register_system_tools()

            mock_system_health.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_configuration.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_cost_estimation.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_embeddings.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    @pytest.mark.asyncio
    async def test_register_tools_logs_completion(self, mock_client_manager, caplog):
        """Test tool registration logs completion."""
        service = SystemService()
        service.client_manager = mock_client_manager

        with patch.multiple(
            "src.mcp_services.system_service",
            system_health=Mock(),
            configuration=Mock(),
            cost_estimation=Mock(),
            embeddings=Mock(),
        ):
            with caplog.at_level(logging.INFO):
                await service._register_system_tools()

            assert "Registered system tools" in caplog.text

    @pytest.mark.asyncio
    async def test_full_initialization_workflow(self, mock_client_manager):
        """Test complete initialization workflow."""
        service = SystemService("test-service")

        assert service.client_manager is None

        with patch.multiple(
            "src.mcp_services.system_service",
            system_health=Mock(),
            configuration=Mock(),
            cost_estimation=Mock(),
            embeddings=Mock(),
        ):
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            assert service.mcp.name == "test-service"


class TestSystemServiceAPI:
    """Test public API contracts."""

    @pytest.mark.asyncio
    async def test_get_service_info_returns_correct_structure(self):
        """Test get_service_info returns correct metadata."""
        service = SystemService()

        info = await service.get_service_info()

        assert info["service"] == "system"
        assert info["version"] == "2.0"
        assert info["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_service_info_includes_all_capabilities(self):
        """Test service info includes all five capabilities."""
        service = SystemService()

        info = await service.get_service_info()

        expected_capabilities = {
            "health_monitoring",
            "resource_management",
            "configuration_management",
            "cost_estimation",
            "embedding_management",
        }
        assert set(info["capabilities"]) == expected_capabilities

    def test_get_mcp_server_returns_fastmcp_instance(self):
        """Test get_mcp_server returns the FastMCP instance."""
        service = SystemService("test-service")

        mcp = service.get_mcp_server()

        assert mcp.name == "test-service"
        assert hasattr(mcp, "tool")  # FastMCP decorator
        assert hasattr(mcp, "instructions")


class TestSystemServiceErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_initialize_handles_client_manager_gracefully(self):
        """Test service initialization with client manager works correctly."""
        service = SystemService()

        # This should work - initialization sets the manager
        with patch.object(service, "_register_system_tools", new_callable=AsyncMock):
            await service.initialize(AsyncMock())

        assert service.client_manager is not None

    @pytest.mark.asyncio
    async def test_service_info_works_before_initialization(self):
        """Test get_service_info works before initialize is called."""
        service = SystemService()

        # Should not raise - service info is static
        info = await service.get_service_info()

        assert info["service"] == "system"
        assert len(info["capabilities"]) == 5

    def test_get_mcp_server_works_before_initialization(self):
        """Test get_mcp_server works before initialize is called."""
        service = SystemService()

        # Should not raise - MCP server created in __init__
        mcp = service.get_mcp_server()

        assert mcp is not None
        assert mcp.name == "system-service"
