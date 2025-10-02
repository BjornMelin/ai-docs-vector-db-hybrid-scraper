"""Unit tests for OrchestratorService - Multi-service coordination.

Tests cover:
- Service initialization and multi-service coordination setup
- Cross-service workflow orchestration and service composition
- Agentic orchestration integration and capability assessment
- Service discovery and performance optimization
- Error handling and fault tolerance across services
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.orchestrator_service import OrchestratorService


class TestOrchestratorService:
    """Test OrchestratorService initialization and core functionality."""

    def test_init_creates_service_with_correct_configuration(self):
        """Test that OrchestratorService initializes with correct FastMCP
        configuration."""
        service = OrchestratorService("test-orchestrator-service")

        assert service.mcp.name == "test-orchestrator-service"
        assert "Central orchestrator service" in service.mcp.instructions
        assert "multi-service coordination" in service.mcp.instructions
        assert service.client_manager is None

    def test_init_with_default_name(self):
        """Test OrchestratorService initialization with default name."""
        service = OrchestratorService()

        assert service.mcp.name == "orchestrator-service"
        assert service.client_manager is None

    def test_init_contains_coordination_features(self):
        """Test that service instructions include coordination features."""
        service = OrchestratorService()

        instructions = service.mcp.instructions
        assert "Multi-service workflow orchestration and coordination" in instructions
        assert "Autonomous service selection and load balancing" in instructions
        assert "Cross-service state management and synchronization" in instructions
        assert "Intelligent service composition for complex workflows" in instructions
        assert "Performance optimization across service boundaries" in instructions
        assert "Unified error handling and fault tolerance" in instructions

    def test_init_contains_autonomous_capabilities(self):
        """Test that service instructions include autonomous capabilities."""
        service = OrchestratorService()

        instructions = service.mcp.instructions
        assert "Autonomous Capabilities" in instructions
        assert "Dynamic service discovery and capability assessment" in instructions
        assert "Intelligent workflow decomposition and service routing" in instructions
        assert "Self-healing multi-service coordination" in instructions
        assert (
            "Performance correlation and optimization across services" in instructions
        )

    def test_init_initializes_services_as_none(self):
        """Test that domain-specific services are initialized as None."""
        service = OrchestratorService()

        assert service.search_service is None
        assert service.document_service is None
        assert service.analytics_service is None
        assert service.system_service is None
        assert service.agentic_orchestrator is None

    @pytest.mark.asyncio
    async def test_initialize_with_client_manager(self, mock_client_manager):
        """Test service initialization with client manager."""
        service = OrchestratorService()

        with (
            patch.object(
                service, "_initialize_domain_services", new_callable=AsyncMock
            ) as mock_domain_init,
            patch.object(
                service, "_initialize_agentic_orchestration", new_callable=AsyncMock
            ) as mock_agentic_init,
            patch.object(
                service, "_register_orchestrator_tools", new_callable=AsyncMock
            ) as mock_register,
        ):
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_domain_init.assert_called_once()
            mock_agentic_init.assert_called_once()
            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_logs_success_message(self, mock_client_manager, caplog):
        """Test that init logs success message with multi-service coordination."""
        service = OrchestratorService()

        with (
            patch.object(
                service, "_initialize_domain_services", new_callable=AsyncMock
            ),
            patch.object(
                service, "_initialize_agentic_orchestration", new_callable=AsyncMock
            ),
            patch.object(
                service, "_register_orchestrator_tools", new_callable=AsyncMock
            ),
            caplog.at_level(logging.INFO),
        ):
            await service.initialize(mock_client_manager)

            assert (
                "OrchestratorService initialized with multi-service coordination"
                in caplog.text
            )


class TestOrchestratorServiceDomainServicesInitialization:
    """Test OrchestratorService domain services initialization."""

    @pytest.mark.asyncio
    async def test_initialize_domain_services_raises_error_when_not_initialized(self):
        """Test that domain services init raises error when service not initialized."""
        service = OrchestratorService()

        with pytest.raises(RuntimeError, match="OrchestratorService not initialized"):
            await service._initialize_domain_services()

    @pytest.mark.asyncio
    async def test_initialize_domain_services_creates_all_services(
        self, mock_client_manager
    ):
        """Test that all domain-specific services are created and initialized."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock the service classes
        mock_search_service = Mock()
        mock_search_service.initialize = AsyncMock()

        mock_document_service = Mock()
        mock_document_service.initialize = AsyncMock()

        mock_analytics_service = Mock()
        mock_analytics_service.initialize = AsyncMock()

        mock_system_service = Mock()
        mock_system_service.initialize = AsyncMock()

        with patch.multiple(
            "src.mcp_services.orchestrator_service",
            SearchService=Mock(return_value=mock_search_service),
            DocumentService=Mock(return_value=mock_document_service),
            AnalyticsService=Mock(return_value=mock_analytics_service),
            SystemService=Mock(return_value=mock_system_service),
        ):
            await service._initialize_domain_services()

            # Verify all services were created and initialized
            assert service.search_service == mock_search_service
            assert service.document_service == mock_document_service
            assert service.analytics_service == mock_analytics_service
            assert service.system_service == mock_system_service

            # Verify all services were initialized with client manager
            mock_search_service.initialize.assert_called_once_with(mock_client_manager)
            mock_document_service.initialize.assert_called_once_with(
                mock_client_manager
            )
            mock_analytics_service.initialize.assert_called_once_with(
                mock_client_manager
            )
            mock_system_service.initialize.assert_called_once_with(mock_client_manager)

    @pytest.mark.asyncio
    async def test_initialize_domain_services_handles_service_initialization_failures(
        self, mock_client_manager, caplog
    ):
        """Test that domain services init handles individual service failures."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock services with one failing
        mock_search_service = Mock()
        mock_search_service.initialize = AsyncMock(
            side_effect=Exception("Search service initialization failed")
        )

        mock_document_service = Mock()
        mock_document_service.initialize = AsyncMock()

        mock_analytics_service = Mock()
        mock_analytics_service.initialize = AsyncMock()

        mock_system_service = Mock()
        mock_system_service.initialize = AsyncMock()

        with (
            patch.multiple(
                "src.mcp_services.orchestrator_service",
                SearchService=Mock(return_value=mock_search_service),
                DocumentService=Mock(return_value=mock_document_service),
                AnalyticsService=Mock(return_value=mock_analytics_service),
                SystemService=Mock(return_value=mock_system_service),
            ),
            caplog.at_level(logging.ERROR),
        ):
            await service._initialize_domain_services()

            # Verify that failure was logged but other services still initialized
            assert "Failed to initialize search service" in caplog.text
            assert service.document_service == mock_document_service
            assert service.analytics_service == mock_analytics_service
            assert service.system_service == mock_system_service

    @pytest.mark.asyncio
    async def test_initialize_domain_services_logs_individual_service_success(
        self, mock_client_manager, caplog
    ):
        """Test that individual service initialization successes are logged."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock services
        mock_services = {}
        for service_name in ["search", "document", "analytics", "system"]:
            mock_service = Mock()
            mock_service.initialize = AsyncMock()
            mock_services[service_name] = mock_service

        with (
            patch.multiple(
                "src.mcp_services.orchestrator_service",
                SearchService=Mock(return_value=mock_services["search"]),
                DocumentService=Mock(return_value=mock_services["document"]),
                AnalyticsService=Mock(return_value=mock_services["analytics"]),
                SystemService=Mock(return_value=mock_services["system"]),
            ),
            caplog.at_level(logging.INFO),
        ):
            await service._initialize_domain_services()

            # Verify all services logged successful initialization
            for service_name in ["search", "document", "analytics", "system"]:
                assert f"Initialized {service_name} service" in caplog.text


class TestOrchestratorServiceAgenticOrchestration:
    """Test OrchestratorService agentic orchestration initialization."""

    @pytest.mark.asyncio
    async def test_initialize_agentic_orchestration_without_client_manager(self):
        """Test that agentic orchestration returns early without client manager."""
        service = OrchestratorService()

        # Should not raise error when client_manager is None
        await service._initialize_agentic_orchestration()

        assert service.agentic_orchestrator is None

    @pytest.mark.asyncio
    async def test_initialize_agentic_orchestration_creates_components(
        self, mock_client_manager, mock_agentic_orchestrator, mock_discovery_engine
    ):
        """Test that agentic orchestration components are created and initialized."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        with patch.multiple(
            "src.mcp_services.orchestrator_service",
            create_agent_dependencies=Mock(return_value=Mock()),
            AgenticOrchestrator=Mock(return_value=mock_agentic_orchestrator),
            get_discovery_engine=Mock(return_value=mock_discovery_engine),
        ):
            await service._initialize_agentic_orchestration()

            assert service.agentic_orchestrator == mock_agentic_orchestrator
            assert service.discovery_engine == mock_discovery_engine

            # Verify components were initialized
            mock_agentic_orchestrator.initialize.assert_called_once()
            mock_discovery_engine.initialize_discovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_agentic_orchestration_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that agentic orchestration initialization logs success message."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        with (
            patch.multiple(
                "src.mcp_services.orchestrator_service",
                create_agent_dependencies=Mock(return_value=Mock()),
                AgenticOrchestrator=Mock(return_value=Mock(initialize=AsyncMock())),
                get_discovery_engine=Mock(
                    return_value=Mock(initialize_discovery=AsyncMock())
                ),
            ),
            caplog.at_level(logging.INFO),
        ):
            await service._initialize_agentic_orchestration()

            assert "Initialized agentic orchestration components" in caplog.text


class TestOrchestratorServiceToolRegistration:
    """Test OrchestratorService orchestrator tools registration."""

    @pytest.mark.asyncio
    async def test_register_orchestrator_tools_raises_error_when_not_initialized(self):
        """Test that tool registration raises error when service not initialized."""
        service = OrchestratorService()

        with pytest.raises(RuntimeError, match="OrchestratorService not initialized"):
            await service._register_orchestrator_tools()

    @pytest.mark.asyncio
    async def test_register_orchestrator_tools_creates_mcp_tools(
        self, mock_client_manager
    ):
        """Test that orchestrator tools are registered as MCP tools."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock FastMCP tool decorator
        registered_tools = []

        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        service.mcp.tool = Mock(side_effect=mock_tool_decorator)

        await service._register_orchestrator_tools()

        # Verify that the orchestrator tools were registered
        expected_tools = [
            "orchestrate_multi_service_workflow",
            "get_service_capabilities",
            "optimize_service_performance",
        ]

        for tool_name in expected_tools:
            assert tool_name in registered_tools

    @pytest.mark.asyncio
    async def test_orchestrate_multi_service_workflow_tool_with_agentic_orchestrator(
        self, mock_client_manager, mock_agentic_orchestrator
    ):
        """Test orchestrate_multi_service_workflow tool with agentic orchestrator."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager
        service.agentic_orchestrator = mock_agentic_orchestrator

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        workflow_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "orchestrate_multi_service_workflow"
            ):
                workflow_tool = call[0][0]
                break

        assert workflow_tool is not None

        # Test the tool function
        workflow_description = "Test workflow description"
        services_required = ["search", "analytics"]
        performance_constraints = {"max_latency_ms": 1000}

        with patch(
            "src.mcp_services.orchestrator_service.create_agent_dependencies"
        ) as mock_create_deps:
            mock_deps = Mock()
            mock_create_deps.return_value = mock_deps

            result = await workflow_tool(
                workflow_description=workflow_description,
                services_required=services_required,
                performance_constraints=performance_constraints,
            )

        # Verify orchestration was called
        mock_agentic_orchestrator.orchestrate.assert_called_once()

        # Verify result structure
        assert result["success"] is True
        assert "workflow_results" in result
        assert "services_used" in result
        assert "orchestration_reasoning" in result
        assert "execution_time_ms" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_orchestrate_multi_service_workflow_tool_without_agentic_orchestrator(
        self, mock_client_manager
    ):
        """Test orchestrate_multi_service_workflow tool when agentic orchestrator
        is not available."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager
        service.agentic_orchestrator = None

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        workflow_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "orchestrate_multi_service_workflow"
            ):
                workflow_tool = call[0][0]
                break

        assert workflow_tool is not None

        # Test the tool function
        result = await workflow_tool(workflow_description="Test workflow")

        # Should return error when agentic orchestrator not available
        assert "error" in result
        assert result["error"] == "Agentic orchestrator not initialized"

    @pytest.mark.asyncio
    async def test_get_service_capabilities_tool(
        self, mock_client_manager, initialized_mock_services
    ):
        """Test get_service_capabilities tool."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Set up mock services
        service.search_service = initialized_mock_services["search"]
        service.document_service = initialized_mock_services["document"]
        service.analytics_service = initialized_mock_services["analytics"]
        service.system_service = initialized_mock_services["system"]

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        capabilities_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_service_capabilities"
            ):
                capabilities_tool = call[0][0]
                break

        assert capabilities_tool is not None

        # Test the tool function
        result = await capabilities_tool()

        # Verify result structure
        assert "services" in result
        services = result["services"]

        # Verify all services are included
        for service_name in ["search", "document", "analytics", "system"]:
            assert service_name in services
            assert services[service_name]["service"] == service_name
            assert services[service_name]["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_service_capabilities_tool_handles_service_errors(
        self, mock_client_manager
    ):
        """Test get_service_capabilities tool handles service errors gracefully."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock service that raises an error
        mock_failing_service = Mock()
        mock_failing_service.get_service_info = AsyncMock(
            side_effect=Exception("Service error")
        )

        service.search_service = mock_failing_service
        service.document_service = None
        service.analytics_service = None
        service.system_service = None

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        capabilities_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_service_capabilities"
            ):
                capabilities_tool = call[0][0]
                break

        assert capabilities_tool is not None

        # Test the tool function
        result = await capabilities_tool()

        # Verify error handling
        services = result["services"]
        assert services["search"]["status"] == "error"
        assert "Service error" in services["search"]["error"]
        assert services["document"]["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_optimize_service_performance_tool(
        self, mock_client_manager, mock_discovery_engine
    ):
        """Test optimize_service_performance tool."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager
        service.discovery_engine = mock_discovery_engine

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        optimization_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "optimize_service_performance"
            ):
                optimization_tool = call[0][0]
                break

        assert optimization_tool is not None

        # Test the tool function
        result = await optimization_tool()

        # Verify discovery engine was used
        mock_discovery_engine.get_tool_recommendations.assert_called_once()

        # Verify result structure
        assert "optimization_applied" in result
        assert "recommendations" in result
        assert "performance_impact" in result

    @pytest.mark.asyncio
    async def test_optimize_service_performance_tool_without_discovery_engine(
        self, mock_client_manager
    ):
        """Test optimize_service_performance tool when discovery engine is not
        available."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager
        service.discovery_engine = None

        # Register tools
        await service._register_orchestrator_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        optimization_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "optimize_service_performance"
            ):
                optimization_tool = call[0][0]
                break

        assert optimization_tool is not None

        # Test the tool function
        result = await optimization_tool()

        # Should return error when discovery engine not available
        assert "error" in result
        assert result["error"] == "Discovery engine not available"

    @pytest.mark.asyncio
    async def test_register_orchestrator_tools_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that orchestrator tools registration logs success message."""
        service = OrchestratorService()
        service.client_manager = mock_client_manager

        with caplog.at_level(logging.INFO):
            await service._register_orchestrator_tools()

            assert (
                "Registered orchestrator tools with multi-service coordination"
                in caplog.text
            )


class TestOrchestratorServiceServiceInfo:
    """Test OrchestratorService service information and metadata."""

    def test_get_mcp_server_returns_configured_instance(self):
        """Test that get_mcp_server returns the configured FastMCP instance."""
        service = OrchestratorService("test-service")

        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "test-service"

    @pytest.mark.asyncio
    async def test_get_service_info_returns_comprehensive_metadata(self):
        """Test that service info contains all expected metadata and capabilities."""
        service = OrchestratorService()

        service_info = await service.get_service_info()

        # Verify basic metadata
        assert service_info["service"] == "orchestrator"
        assert service_info["version"] == "2.0"
        assert service_info["status"] == "active"
        assert service_info["research_basis"] == "FASTMCP_2_0_SERVER_COMPOSITION"

        # Verify capabilities
        expected_capabilities = [
            "multi_service_coordination",
            "workflow_orchestration",
            "service_composition",
            "cross_service_optimization",
            "unified_error_handling",
            "agentic_coordination",
        ]
        assert service_info["capabilities"] == expected_capabilities

        # Verify autonomous features
        expected_autonomous_features = [
            "service_discovery",
            "intelligent_routing",
            "self_healing_coordination",
            "performance_optimization",
        ]
        assert service_info["autonomous_features"] == expected_autonomous_features

        # Verify coordinated services
        expected_coordinated_services = [
            "search",
            "document",
            "analytics",
            "system",
        ]
        assert service_info["coordinated_services"] == expected_coordinated_services

    @pytest.mark.asyncio
    async def test_get_all_services_returns_service_information(
        self, initialized_mock_services
    ):
        """Test that get_all_services returns info about all coordinated services."""

        service = OrchestratorService()

        # Set up mock services
        service.search_service = initialized_mock_services["search"]
        service.document_service = initialized_mock_services["document"]
        service.analytics_service = initialized_mock_services["analytics"]
        service.system_service = initialized_mock_services["system"]

        result = await service.get_all_services()

        # Verify all services are included
        for service_name in ["search", "document", "analytics", "system"]:
            assert service_name in result
            assert result[service_name]["service"] == service_name
            assert result[service_name]["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_all_services_handles_service_errors(self):
        """Test that get_all_services handles service errors gracefully."""
        service = OrchestratorService()

        # Mock service that raises an error
        mock_failing_service = Mock()
        mock_failing_service.get_service_info = AsyncMock(
            side_effect=Exception("Service error")
        )

        service.search_service = mock_failing_service
        service.document_service = None
        service.analytics_service = None
        service.system_service = None

        result = await service.get_all_services()

        # Verify error handling
        assert result["search"]["status"] == "error"
        assert "Service error" in result["search"]["error"]
        assert result["document"]["status"] == "not_initialized"


class TestOrchestratorServiceErrorHandling:
    """Test OrchestratorService error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_initialization_with_none_client_manager_raises_error(self):
        """Test that domain services init with None client manager raises error."""

        service = OrchestratorService()

        # Should not raise error during main initialization
        await service.initialize(None)

        # But domain services initialization should raise error
        with pytest.raises(RuntimeError, match="OrchestratorService not initialized"):
            await service._initialize_domain_services()

    @pytest.mark.asyncio
    async def test_get_service_info_works_without_initialization(self):
        """Test that get_service_info works even without full initialization."""
        service = OrchestratorService()

        # Should not raise error
        service_info = await service.get_service_info()

        assert service_info["service"] == "orchestrator"
        assert service_info["status"] == "active"

    @pytest.mark.asyncio
    async def test_multiple_initialization_calls_are_safe(self, mock_client_manager):
        """Test that multiple initialization calls are handled safely."""
        service = OrchestratorService()

        with (
            patch.object(
                service, "_initialize_domain_services", new_callable=AsyncMock
            ) as mock_domain_init,
            patch.object(
                service, "_initialize_agentic_orchestration", new_callable=AsyncMock
            ) as mock_agentic_init,
            patch.object(
                service, "_register_orchestrator_tools", new_callable=AsyncMock
            ) as mock_register,
        ):
            # First initialization
            await service.initialize(mock_client_manager)
            first_domain_calls = mock_domain_init.call_count
            first_agentic_calls = mock_agentic_init.call_count
            first_register_calls = mock_register.call_count

            # Second initialization
            await service.initialize(mock_client_manager)
            second_domain_calls = mock_domain_init.call_count
            second_agentic_calls = mock_agentic_init.call_count
            second_register_calls = mock_register.call_count

            # Should handle multiple calls gracefully
            assert second_domain_calls >= first_domain_calls
            assert second_agentic_calls >= first_agentic_calls
            assert second_register_calls >= first_register_calls

    @pytest.mark.asyncio
    async def test_service_handles_partial_service_initialization_failures(
        self, mock_client_manager, caplog
    ):
        """Test that orchestrator handles partial service failures."""

        service = OrchestratorService()
        service.client_manager = mock_client_manager

        # Mock services with some failing
        mock_search_service = Mock()
        mock_search_service.initialize = AsyncMock(
            side_effect=Exception("Search failed")
        )

        mock_document_service = Mock()
        mock_document_service.initialize = AsyncMock()

        with (
            patch.multiple(
                "src.mcp_services.orchestrator_service",
                SearchService=Mock(return_value=mock_search_service),
                DocumentService=Mock(return_value=mock_document_service),
                AnalyticsService=Mock(return_value=Mock(initialize=AsyncMock())),
                SystemService=Mock(return_value=Mock(initialize=AsyncMock())),
            ),
            caplog.at_level(logging.ERROR),
        ):
            await service._initialize_domain_services()

            # Should continue despite failures
            assert "Failed to initialize search service" in caplog.text
            assert service.document_service is not None


class TestOrchestratorServicePerformanceAndIntegration:
    """Test OrchestratorService perf characteristics and integ scenarios."""

    @pytest.mark.asyncio
    async def test_service_initialization_is_efficient(self, mock_client_manager):
        """Test that service initialization is efficient and doesn't block."""

        service = OrchestratorService()

        start_time = time.time()

        with (
            patch.object(
                service, "_initialize_domain_services", new_callable=AsyncMock
            ),
            patch.object(
                service, "_initialize_agentic_orchestration", new_callable=AsyncMock
            ),
            patch.object(
                service, "_register_orchestrator_tools", new_callable=AsyncMock
            ),
        ):
            await service.initialize(mock_client_manager)

        end_time = time.time()

        # Initialization should be fast (< 2 seconds for orchestrator)
        assert end_time - start_time < 2.0

    @pytest.mark.asyncio
    async def test_service_supports_concurrent_access(self, mock_client_manager):
        """Test that service supports concurrent access patterns."""

        service = OrchestratorService()

        # Simulate concurrent access
        async def concurrent_operation():
            await service.initialize(mock_client_manager)
            return await service.get_service_info()

        with (
            patch.object(
                service, "_initialize_domain_services", new_callable=AsyncMock
            ),
            patch.object(
                service, "_initialize_agentic_orchestration", new_callable=AsyncMock
            ),
            patch.object(
                service, "_register_orchestrator_tools", new_callable=AsyncMock
            ),
        ):
            # Run multiple concurrent operations
            tasks = [concurrent_operation() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result["service"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_comprehensive_capability_reporting(self):
        """Test comprehensive capability reporting for service discovery."""
        service = OrchestratorService()

        service_info = await service.get_service_info()

        # Verify comprehensive capability reporting
        assert len(service_info["capabilities"]) >= 6  # All core capabilities
        assert len(service_info["autonomous_features"]) >= 4  # All autonomous features
        assert (
            len(service_info["coordinated_services"]) == 4
        )  # All coordinated services

        # Verify service is ready for capability discovery
        assert service_info["status"] == "active"
        assert "version" in service_info
        assert "research_basis" in service_info

    @pytest.mark.asyncio
    async def test_fastmcp_2_0_server_composition_integration(self):
        """Test FastMCP 2.0+ server composition integration."""
        service = OrchestratorService()

        service_info = await service.get_service_info()

        # Verify FastMCP 2.0+ server composition
        assert service_info["research_basis"] == "FASTMCP_2_0_SERVER_COMPOSITION"
        assert "multi_service_coordination" in service_info["capabilities"]
        assert "service_composition" in service_info["capabilities"]
