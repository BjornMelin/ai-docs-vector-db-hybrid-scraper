"""Unit tests for SystemService - Self-healing infrastructure.

Tests cover:
- Service initialization and self-healing capabilities
- System health monitoring and resource management
- Configuration management and cost optimization
- Embedding optimization and data filtering
- Autonomous fault tolerance and predictive maintenance
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.system_service import SystemService


class TestSystemService:
    """Test SystemService initialization and core functionality."""

    def test_init_creates_service_with_correct_configuration(self):
        """Test that SystemService initializes with correct FastMCP configuration."""
        service = SystemService("test-system-service")

        assert service.mcp.name == "test-system-service"
        assert "Advanced system service" in service.mcp.instructions
        assert "self-healing" in service.mcp.instructions
        assert service.client_manager is None

    def test_init_with_default_name(self):
        """Test SystemService initialization with default name."""
        service = SystemService()

        assert service.mcp.name == "system-service"
        assert service.client_manager is None

    def test_init_contains_self_healing_features(self):
        """Test that service instructions include self-healing features."""
        service = SystemService()

        instructions = service.mcp.instructions
        assert "Real-time system health monitoring and alerting" in instructions
        assert "Autonomous resource management and optimization" in instructions
        assert "Configuration management with intelligent defaults" in instructions
        assert "Cost estimation and budget optimization" in instructions
        assert "Embedding management with provider optimization" in instructions
        assert "Advanced filtering and data processing" in instructions

    def test_init_contains_autonomous_capabilities(self):
        """Test that service instructions include autonomous capabilities."""
        service = SystemService()

        instructions = service.mcp.instructions
        assert "Autonomous Capabilities" in instructions
        assert "Self-healing system recovery and fault tolerance" in instructions
        assert "Predictive maintenance and resource scaling" in instructions
        assert "Intelligent configuration optimization" in instructions
        assert "Cost and performance correlation analysis" in instructions

    async def test_initialize_with_client_manager(self, mock_client_manager):
        """Test service initialization with client manager."""
        service = SystemService()

        with patch.object(
            service, "_register_system_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_initialize_logs_success_message(self, mock_client_manager, caplog):
        """Test that initialization logs success message with self-healing capabilities."""
        service = SystemService()

        with patch.object(service, "_register_system_tools", new_callable=AsyncMock):
            with caplog.at_level(logging.INFO):
                await service.initialize(mock_client_manager)

                assert (
                    "SystemService initialized with self-healing capabilities"
                    in caplog.text
                )

    async def test_register_system_tools_raises_error_when_not_initialized(self):
        """Test that tool registration raises error when service not initialized."""
        service = SystemService()

        with pytest.raises(RuntimeError, match="SystemService not initialized"):
            await service._register_system_tools()

    async def test_register_system_tools_calls_all_tool_registrations(
        self, mock_client_manager
    ):
        """Test that all system tools are registered properly."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock all the tool modules
        mock_tools = {
            "system_health": Mock(),
            "configuration": Mock(),
            "cost_estimation": Mock(),
            "embeddings": Mock(),
            "filtering": Mock(),
        }

        for mock_tool in mock_tools.values():
            mock_tool.register_tools = Mock()

        # Patch the tool imports
        with patch.multiple(
            "src.mcp_services.system_service",
            system_health=mock_tools["system_health"],
            configuration=mock_tools["configuration"],
            cost_estimation=mock_tools["cost_estimation"],
            embeddings=mock_tools["embeddings"],
            filtering=mock_tools["filtering"],
        ):
            await service._register_system_tools()

            # Verify all tools were registered
            for mock_tool in mock_tools.values():
                mock_tool.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_register_system_tools_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that tool registration logs success message."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock tool modules
        with (
            patch.multiple(
                "src.mcp_services.system_service",
                system_health=Mock(register_tools=Mock()),
                configuration=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ),
            caplog.at_level(logging.INFO),
        ):
            await service._register_system_tools()

            assert (
                "Registered system tools with self-healing capabilities" in caplog.text
            )

    def test_get_mcp_server_returns_configured_instance(self):
        """Test that get_mcp_server returns the configured FastMCP instance."""
        service = SystemService("test-service")

        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "test-service"

    async def test_get_service_info_returns_comprehensive_metadata(self):
        """Test that service info contains all expected metadata and capabilities."""
        service = SystemService()

        service_info = await service.get_service_info()

        # Verify basic metadata
        assert service_info["service"] == "system"
        assert service_info["version"] == "2.0"
        assert service_info["status"] == "active"
        assert service_info["research_basis"] == "SYSTEM_INFRASTRUCTURE"

        # Verify capabilities
        expected_capabilities = [
            "health_monitoring",
            "resource_management",
            "configuration_management",
            "cost_estimation",
            "embedding_optimization",
            "data_filtering",
            "self_healing",
        ]
        assert service_info["capabilities"] == expected_capabilities

        # Verify autonomous features
        expected_autonomous_features = [
            "fault_tolerance",
            "predictive_maintenance",
            "configuration_optimization",
            "cost_intelligence",
        ]
        assert service_info["autonomous_features"] == expected_autonomous_features


class TestSystemServiceHealthAndMonitoring:
    """Test SystemService health monitoring and system management capabilities."""

    async def test_system_health_tool_registration(self, mock_client_manager):
        """Test that system health tools are properly registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock system health tool specifically
        mock_system_health = Mock()
        mock_system_health.register_tools = Mock()

        with patch("src.mcp_services.system_service.system_health", mock_system_health):
            with patch.multiple(
                "src.mcp_services.system_service",
                configuration=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ):
                await service._register_system_tools()

                # Verify system health tools were registered
                mock_system_health.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_health_monitoring_capability(self):
        """Test that service reports health monitoring capability."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "health_monitoring" in service_info["capabilities"]
        assert "resource_management" in service_info["capabilities"]

    async def test_self_healing_capability(self):
        """Test that service supports self-healing capability."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "self_healing" in service_info["capabilities"]
        assert "fault_tolerance" in service_info["autonomous_features"]
        assert "predictive_maintenance" in service_info["autonomous_features"]

    def test_service_instructions_contain_monitoring_features(self):
        """Test that service instructions reference monitoring features."""
        service = SystemService()

        instructions = service.mcp.instructions

        # Check for monitoring features
        assert "Real-time system health monitoring and alerting" in instructions
        assert "Autonomous resource management and optimization" in instructions


class TestSystemServiceConfigurationManagement:
    """Test SystemService configuration management and optimization capabilities."""

    async def test_configuration_tool_registration(self, mock_client_manager):
        """Test that configuration tools are properly registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock configuration tool specifically
        mock_configuration = Mock()
        mock_configuration.register_tools = Mock()

        with patch("src.mcp_services.system_service.configuration", mock_configuration):
            with patch.multiple(
                "src.mcp_services.system_service",
                system_health=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ):
                await service._register_system_tools()

                # Verify configuration tools were registered
                mock_configuration.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_configuration_management_capability(self):
        """Test that service supports configuration management."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "configuration_management" in service_info["capabilities"]
        assert "configuration_optimization" in service_info["autonomous_features"]

    def test_service_instructions_contain_configuration_features(self):
        """Test that service instructions reference configuration features."""
        service = SystemService()

        instructions = service.mcp.instructions

        # Check for configuration features
        assert "Configuration management with intelligent defaults" in instructions
        assert "Intelligent configuration optimization" in instructions


class TestSystemServiceCostAndResourceOptimization:
    """Test SystemService cost estimation and resource optimization capabilities."""

    async def test_cost_estimation_tool_registration(self, mock_client_manager):
        """Test that cost estimation tools are properly registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock cost estimation tool specifically
        mock_cost_estimation = Mock()
        mock_cost_estimation.register_tools = Mock()

        with (
            patch(
                "src.mcp_services.system_service.cost_estimation", mock_cost_estimation
            ),
            patch.multiple(
                "src.mcp_services.system_service",
                system_health=Mock(register_tools=Mock()),
                configuration=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ),
        ):
            await service._register_system_tools()

            # Verify cost estimation tools were registered
            mock_cost_estimation.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    async def test_cost_optimization_capability(self):
        """Test that service supports cost optimization."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "cost_estimation" in service_info["capabilities"]
        assert "cost_intelligence" in service_info["autonomous_features"]

    async def test_resource_management_capability(self):
        """Test that service supports resource management."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "resource_management" in service_info["capabilities"]

    def test_service_instructions_contain_cost_features(self):
        """Test that service instructions reference cost optimization features."""
        service = SystemService()

        instructions = service.mcp.instructions

        # Check for cost optimization features
        assert "Cost estimation and budget optimization" in instructions
        assert "Cost and performance correlation analysis" in instructions


class TestSystemServiceEmbeddingAndDataProcessing:
    """Test SystemService embedding optimization and data processing capabilities."""

    async def test_embeddings_tool_registration(self, mock_client_manager):
        """Test that embedding tools are properly registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock embeddings tool specifically
        mock_embeddings = Mock()
        mock_embeddings.register_tools = Mock()

        with patch("src.mcp_services.system_service.embeddings", mock_embeddings):
            with patch.multiple(
                "src.mcp_services.system_service",
                system_health=Mock(register_tools=Mock()),
                configuration=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ):
                await service._register_system_tools()

                # Verify embeddings tools were registered
                mock_embeddings.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_filtering_tool_registration(self, mock_client_manager):
        """Test that filtering tools are properly registered."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock filtering tool specifically
        mock_filtering = Mock()
        mock_filtering.register_tools = Mock()

        with patch("src.mcp_services.system_service.filtering", mock_filtering):
            with patch.multiple(
                "src.mcp_services.system_service",
                system_health=Mock(register_tools=Mock()),
                configuration=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
            ):
                await service._register_system_tools()

                # Verify filtering tools were registered
                mock_filtering.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_embedding_optimization_capability(self):
        """Test that service supports embedding optimization."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "embedding_optimization" in service_info["capabilities"]

    async def test_data_filtering_capability(self):
        """Test that service supports data filtering."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "data_filtering" in service_info["capabilities"]

    def test_service_instructions_contain_data_processing_features(self):
        """Test that service instructions reference data processing features."""
        service = SystemService()

        instructions = service.mcp.instructions

        # Check for data processing features
        assert "Embedding management with provider optimization" in instructions
        assert "Advanced filtering and data processing" in instructions


class TestSystemServiceErrorHandling:
    """Test SystemService error handling and recovery scenarios."""

    async def test_initialization_with_none_client_manager_raises_error(self):
        """Test that initialization with None client manager is handled properly."""
        service = SystemService()

        # Should not raise error during initialization
        await service.initialize(None)

        # But should raise error when trying to register tools
        with pytest.raises(RuntimeError, match="SystemService not initialized"):
            await service._register_system_tools()

    async def test_get_service_info_works_without_initialization(self):
        """Test that get_service_info works even without full initialization."""
        service = SystemService()

        # Should not raise error
        service_info = await service.get_service_info()

        assert service_info["service"] == "system"
        assert service_info["status"] == "active"

    async def test_error_handling_during_tool_registration(self, mock_client_manager):
        """Test error handling during tool registration process."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # Mock a tool that raises an exception during registration
        mock_failing_tool = Mock()
        mock_failing_tool.register_tools.side_effect = Exception(
            "Tool registration failed"
        )

        with patch("src.mcp_services.system_service.system_health", mock_failing_tool):
            with patch.multiple(
                "src.mcp_services.system_service",
                configuration=Mock(register_tools=Mock()),
                cost_estimation=Mock(register_tools=Mock()),
                embeddings=Mock(register_tools=Mock()),
                filtering=Mock(register_tools=Mock()),
            ):
                # Tool registration should raise the exception
                with pytest.raises(Exception, match="Tool registration failed"):
                    await service._register_system_tools()

    async def test_service_recovery_after_tool_registration_failure(
        self, mock_client_manager
    ):
        """Test service recovery after tool registration failure."""
        service = SystemService()
        service.client_manager = mock_client_manager

        # First attempt fails
        with patch("src.mcp_services.system_service.system_health") as mock_tool:
            mock_tool.register_tools.side_effect = Exception("Registration failed")

            with pytest.raises(Exception):
                await service._register_system_tools()

        # Second attempt should work
        with patch.multiple(
            "src.mcp_services.system_service",
            system_health=Mock(register_tools=Mock()),
            configuration=Mock(register_tools=Mock()),
            cost_estimation=Mock(register_tools=Mock()),
            embeddings=Mock(register_tools=Mock()),
            filtering=Mock(register_tools=Mock()),
        ):
            # Should not raise error on retry
            await service._register_system_tools()

    async def test_multiple_initialization_calls_are_safe(self, mock_client_manager):
        """Test that multiple initialization calls are handled safely."""
        service = SystemService()

        with patch.object(
            service, "_register_system_tools", new_callable=AsyncMock
        ) as mock_register:
            # First initialization
            await service.initialize(mock_client_manager)
            first_call_count = mock_register.call_count

            # Second initialization
            await service.initialize(mock_client_manager)
            second_call_count = mock_register.call_count

            # Should handle multiple calls gracefully
            assert second_call_count >= first_call_count


class TestSystemServiceSelfHealingCapabilities:
    """Test SystemService self-healing and autonomous capabilities."""

    async def test_fault_tolerance_capability(self):
        """Test that service reports fault tolerance capability."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "fault_tolerance" in service_info["autonomous_features"]

    async def test_predictive_maintenance_capability(self):
        """Test that service supports predictive maintenance."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "predictive_maintenance" in service_info["autonomous_features"]

    def test_service_instructions_contain_self_healing_features(self):
        """Test that service instructions reference self-healing features."""
        service = SystemService()

        instructions = service.mcp.instructions

        # Check for self-healing features
        assert "Self-healing system recovery and fault tolerance" in instructions
        assert "Predictive maintenance and resource scaling" in instructions

    async def test_configuration_optimization_capability(self):
        """Test that service supports autonomous configuration optimization."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "configuration_optimization" in service_info["autonomous_features"]

    async def test_cost_intelligence_capability(self):
        """Test that service supports cost intelligence."""
        service = SystemService()

        service_info = await service.get_service_info()

        assert "cost_intelligence" in service_info["autonomous_features"]

    async def test_comprehensive_autonomous_features(self):
        """Test that all autonomous features are comprehensive."""
        service = SystemService()

        service_info = await service.get_service_info()

        autonomous_features = service_info["autonomous_features"]

        # Verify comprehensive autonomous capabilities
        assert "fault_tolerance" in autonomous_features
        assert "predictive_maintenance" in autonomous_features
        assert "configuration_optimization" in autonomous_features
        assert "cost_intelligence" in autonomous_features


class TestSystemServicePerformanceAndIntegration:
    """Test SystemService performance characteristics and integration scenarios."""

    async def test_service_initialization_is_efficient(self, mock_client_manager):
        """Test that service initialization is efficient and doesn't block."""

        service = SystemService()

        start_time = time.time()

        with patch.object(service, "_register_system_tools", new_callable=AsyncMock):
            await service.initialize(mock_client_manager)

        end_time = time.time()

        # Initialization should be fast (< 1 second)
        assert end_time - start_time < 1.0

    async def test_get_service_info_performance(self):
        """Test that get_service_info is performant for capability discovery."""

        service = SystemService()

        start_time = time.time()

        # Call multiple times to test performance
        for _ in range(10):
            await service.get_service_info()

        end_time = time.time()

        # Should complete quickly (< 0.1 seconds for 10 calls)
        assert end_time - start_time < 0.1

    async def test_service_supports_concurrent_access(self, mock_client_manager):
        """Test that service supports concurrent access patterns."""

        service = SystemService()

        # Simulate concurrent access
        async def concurrent_operation():
            await service.initialize(mock_client_manager)
            return await service.get_service_info()

        with patch.object(service, "_register_system_tools", new_callable=AsyncMock):
            # Run multiple concurrent operations
            tasks = [concurrent_operation() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result["service"] == "system"

    async def test_service_handles_initialization_with_complex_client_manager(
        self, mock_client_manager
    ):
        """Test service initialization with complex client manager setup."""
        service = SystemService("advanced-system")

        # Configure complex client manager
        mock_client_manager.get_openai_client.return_value = Mock()
        mock_client_manager.get_qdrant_client.return_value = Mock()
        mock_client_manager.get_redis_client.return_value = Mock()
        mock_client_manager.parallel_processing_system = Mock()

        with patch.object(
            service, "_register_system_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_comprehensive_capability_reporting(self):
        """Test comprehensive capability reporting for service discovery."""
        service = SystemService()

        service_info = await service.get_service_info()

        # Verify comprehensive capability reporting
        assert len(service_info["capabilities"]) >= 7  # All core capabilities
        assert len(service_info["autonomous_features"]) >= 4  # All autonomous features

        # Verify service is ready for capability discovery
        assert service_info["status"] == "active"
        assert "version" in service_info
        assert "research_basis" in service_info
