"""Integration tests for MCP services - Cross-service coordination and workflows.

Tests cover:
- Complete service initialization and coordination
- Cross-service workflow orchestration scenarios
- Real-world multi-service integration patterns
- Enterprise observability integration validation
- Service discovery and autonomous capability assessment
- Error handling and recovery across service boundaries
"""

import asyncio
import gc
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services import (
    AnalyticsService,
    DocumentService,
    OrchestratorService,
    SearchService,
    SystemService,
)


class TestMCPServicesCompleteIntegration:
    """Test complete MCP services integration with real-world scenarios."""

    @pytest.fixture
    async def complete_mcp_services_setup(
        self, mock_client_manager, mock_observability_components
    ):
        """Set up complete MCP services ecosystem for integration testing."""
        services = {}

        # Initialize all services
        service_classes = {
            "search": SearchService,
            "document": DocumentService,
            "analytics": AnalyticsService,
            "system": SystemService,
            "orchestrator": OrchestratorService,
        }

        # Mock tool registration for all services
        async def mock_tool_registration(*args, **kwargs):
            pass

        # Mock observability integration for analytics service
        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(
                return_value=mock_observability_components["ai_tracker"]
            ),
            get_correlation_manager=Mock(
                return_value=mock_observability_components["correlation_manager"]
            ),
            get_performance_monitor=Mock(
                return_value=mock_observability_components["performance_monitor"]
            ),
        ):
            # Initialize each service
            for service_name, service_class in service_classes.items():
                service = service_class(f"test-{service_name}-service")

                # Mock tool registration methods
                if hasattr(service, "_register_search_tools"):
                    service._register_search_tools = AsyncMock()
                if hasattr(service, "_register_document_tools"):
                    service._register_document_tools = AsyncMock()
                if hasattr(service, "_register_analytics_tools"):
                    service._register_analytics_tools = AsyncMock()
                if hasattr(service, "_register_system_tools"):
                    service._register_system_tools = AsyncMock()
                if hasattr(service, "_register_orchestrator_tools"):
                    service._register_orchestrator_tools = AsyncMock()
                if hasattr(service, "_register_enhanced_observability_tools"):
                    service._register_enhanced_observability_tools = AsyncMock()
                if hasattr(service, "_initialize_domain_services"):
                    service._initialize_domain_services = AsyncMock()
                if hasattr(service, "_initialize_agentic_orchestration"):
                    service._initialize_agentic_orchestration = AsyncMock()
                if hasattr(service, "_initialize_observability_integration"):
                    service._initialize_observability_integration = AsyncMock()

                await service.initialize(mock_client_manager)
                services[service_name] = service

        return services

    @pytest.mark.asyncio
    async def test_all_services_initialize_successfully(
        self, complete_mcp_services_setup
    ):
        """Test that all MCP services initialize successfully in integration."""
        services = complete_mcp_services_setup

        # Verify all services are initialized
        assert len(services) == 5
        for service_name in [
            "search",
            "document",
            "analytics",
            "system",
            "orchestrator",
        ]:
            assert service_name in services
            assert services[service_name].client_manager is not None

    @pytest.mark.asyncio
    async def test_all_services_report_capabilities_correctly(
        self, complete_mcp_services_setup
    ):
        """Test that all services report correct features for service discovery."""
        services = complete_mcp_services_setup

        # Test each service's capabilities
        for service_name, service in services.items():
            service_info = await service.get_service_info()

            # Verify basic service info structure
            assert service_info["service"] == service_name
            assert service_info["version"] == "2.0"
            assert service_info["status"] == "active"
            assert "capabilities" in service_info
            assert "autonomous_features" in service_info
            assert len(service_info["capabilities"]) > 0
            assert len(service_info["autonomous_features"]) > 0

    @pytest.mark.asyncio
    async def test_service_capability_discovery_integration(
        self, complete_mcp_services_setup
    ):
        """Test service capability discovery across all services."""
        services = complete_mcp_services_setup

        # Collect all capabilities from all services
        all_capabilities = {}
        for service_name, service in services.items():
            service_info = await service.get_service_info()
            all_capabilities[service_name] = service_info["capabilities"]

        # Verify expected capabilities are present
        expected_capabilities = {
            "search": [
                "hybrid_search",
                "autonomous_web_search",
                "multi_provider_orchestration",
            ],
            "document": [
                "5_tier_crawling",
                "intelligent_crawling",
                "content_intelligence",
            ],
            "analytics": [
                "agent_decision_metrics",
                "workflow_visualization",
                "agentic_observability",
            ],
            "system": ["health_monitoring", "self_healing", "cost_estimation"],
            "orchestrator": [
                "multi_service_coordination",
                "workflow_orchestration",
                "agentic_coordination",
            ],
        }

        for service_name, expected_caps in expected_capabilities.items():
            service_caps = all_capabilities[service_name]
            for cap in expected_caps:
                assert cap in service_caps, (
                    f"Service {service_name} missing capability {cap}"
                )

    @pytest.mark.asyncio
    async def test_research_basis_integration_validation(
        self, complete_mcp_services_setup
    ):
        """Test that all services properly reference their research basis."""
        services = complete_mcp_services_setup

        expected_research_basis = {
            "search": "I5_WEB_SEARCH_TOOL_ORCHESTRATION",
            "document": "I3_5_TIER_CRAWLING_ENHANCEMENT",
            "analytics": (
                "J1_ENTERPRISE_AGENTIC_OBSERVABILITY_WITH_EXISTING_INTEGRATION"
            ),
            "system": "SYSTEM_INFRASTRUCTURE",
            "orchestrator": "FASTMCP_2_0_SERVER_COMPOSITION",
        }

        for service_name, service in services.items():
            service_info = await service.get_service_info()
            expected_basis = expected_research_basis[service_name]
            assert expected_basis in service_info["research_basis"]

    @pytest.mark.asyncio
    async def test_enterprise_observability_integration_validation(
        self, complete_mcp_services_setup
    ):
        """Test enterprise observability integration across services."""
        services = complete_mcp_services_setup

        # Analytics service should have enterprise integration details
        analytics_service = services["analytics"]
        analytics_info = await analytics_service.get_service_info()

        assert "enterprise_integration" in analytics_info
        enterprise_integration = analytics_info["enterprise_integration"]
        assert enterprise_integration["no_duplicate_infrastructure"] is True
        assert enterprise_integration["existing_ai_tracker_extended"] is True
        assert enterprise_integration["correlation_manager_leveraged"] is True
        assert enterprise_integration["performance_monitor_integrated"] is True

    @pytest.mark.asyncio
    async def test_concurrent_service_access_performance(
        self, complete_mcp_services_setup
    ):
        """Test concurrent access to multiple services for performance validation."""
        services = complete_mcp_services_setup

        # Define concurrent operations
        async def get_service_info_concurrent(service):
            return await service.get_service_info()

        # Run concurrent operations on all services
        tasks = []
        for service in services.values():
            task = asyncio.create_task(get_service_info_concurrent(service))
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all operations succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert "service" in result
            assert result["status"] == "active"

        # Run concurrent operations on all services
        start_time = time.time()
        tasks = []
        for service in services.values():
            task = asyncio.create_task(get_service_info_concurrent(service))
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Verify all operations succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert "service" in result
            assert result["status"] == "active"

        # Verify performance (should complete quickly)
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second


class TestMCPServicesOrchestratorIntegration:
    """Test orchestrator service integration with domain-specific services."""

    @pytest.fixture
    async def orchestrator_with_real_services(
        self, mock_client_manager, mock_observability_components
    ):
        """Set up orchestrator with real domain-specific services for testing."""
        # Create real domain services
        search_service = SearchService("integration-search")
        document_service = DocumentService("integration-document")
        analytics_service = AnalyticsService("integration-analytics")
        system_service = SystemService("integration-system")

        # Mock tool registration for all services
        search_service._register_search_tools = AsyncMock()
        document_service._register_document_tools = AsyncMock()
        analytics_service._register_analytics_tools = AsyncMock()
        analytics_service._register_enhanced_observability_tools = AsyncMock()
        analytics_service._initialize_observability_integration = AsyncMock()
        system_service._register_system_tools = AsyncMock()

        # Mock observability integration for analytics service
        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(
                return_value=mock_observability_components["ai_tracker"]
            ),
            get_correlation_manager=Mock(
                return_value=mock_observability_components["correlation_manager"]
            ),
            get_performance_monitor=Mock(
                return_value=mock_observability_components["performance_monitor"]
            ),
        ):
            # Initialize services
            await search_service.initialize(mock_client_manager)
            await document_service.initialize(mock_client_manager)
            await analytics_service.initialize(mock_client_manager)
            await system_service.initialize(mock_client_manager)

        # Create orchestrator and set up domain services
        orchestrator = OrchestratorService("integration-orchestrator")
        orchestrator.client_manager = mock_client_manager
        orchestrator.search_service = search_service
        orchestrator.document_service = document_service
        orchestrator.analytics_service = analytics_service
        orchestrator.system_service = system_service

        # Mock orchestrator-specific initialization
        orchestrator._register_orchestrator_tools = AsyncMock()
        orchestrator._initialize_agentic_orchestration = AsyncMock()

        await orchestrator.initialize(mock_client_manager)

        return orchestrator

    @pytest.mark.asyncio
    async def test_orchestrator_coordinates_all_domain_services(
        self, orchestrator_with_real_services
    ):
        """Test that orchestrator successfully coordinates all domain services."""
        orchestrator = orchestrator_with_real_services

        # Get all services information through orchestrator
        all_services = await orchestrator.get_all_services()

        # Verify all domain services are coordinated
        expected_services = ["search", "document", "analytics", "system"]
        for service_name in expected_services:
            assert service_name in all_services
            assert all_services[service_name]["service"] == service_name
            assert all_services[service_name]["status"] == "active"

    @pytest.mark.asyncio
    async def test_orchestrator_service_capability_aggregation(
        self, orchestrator_with_real_services
    ):
        """Test orchestrator's ability to aggregate service capabilities."""
        orchestrator = orchestrator_with_real_services

        # Mock the orchestrator tools to test capability aggregation
        registered_tools = []

        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)

        # Register orchestrator tools (simulate real registration)
        await orchestrator._register_orchestrator_tools()

        # Find and test get_service_capabilities tool
        capabilities_tool = None
        for call in orchestrator.mcp.tool.call_args_list:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_service_capabilities"
            ):
                capabilities_tool = call[0][0]
                break

        assert capabilities_tool is not None

        # Test the capabilities aggregation
        result = await capabilities_tool()

        # Verify aggregated capabilities
        assert "services" in result
        services = result["services"]

        # Each service should report its capabilities
        for service_name in ["search", "document", "analytics", "system"]:
            assert service_name in services
            service_info = services[service_name]
            assert "capabilities" in service_info
            assert len(service_info["capabilities"]) > 0

    @pytest.mark.asyncio
    async def test_orchestrator_handles_individual_service_failures(
        self, orchestrator_with_real_services
    ):
        """Test orchestrator's resilience to individual service failures."""
        orchestrator = orchestrator_with_real_services

        # Make one service fail
        orchestrator.search_service.get_service_info = AsyncMock(
            side_effect=Exception("Search service failed")
        )

        # Get all services should handle the failure gracefully
        all_services = await orchestrator.get_all_services()

        # Verify graceful degradation
        assert all_services["search"]["status"] == "error"
        assert "Search service failed" in all_services["search"]["error"]

        # Other services should still work
        assert all_services["document"]["status"] == "active"
        assert all_services["analytics"]["status"] == "active"
        assert all_services["system"]["status"] == "active"


class TestMCPServicesWorkflowOrchestration:
    """Test real-world workflow orchestration scenarios across services."""

    @pytest.fixture
    async def workflow_test_environment(
        self, mock_client_manager, mock_agentic_orchestrator
    ):
        """Set up environment for workflow orchestration testing."""
        # Create services for workflow testing
        search_service = SearchService("workflow-search")
        document_service = DocumentService("workflow-document")
        analytics_service = AnalyticsService("workflow-analytics")
        orchestrator = OrchestratorService("workflow-orchestrator")

        # Mock all tool registrations
        search_service._register_search_tools = AsyncMock()
        document_service._register_document_tools = AsyncMock()
        analytics_service._register_analytics_tools = AsyncMock()
        analytics_service._register_enhanced_observability_tools = AsyncMock()
        analytics_service._initialize_observability_integration = AsyncMock()

        # Initialize services
        await search_service.initialize(mock_client_manager)
        await document_service.initialize(mock_client_manager)
        await analytics_service.initialize(mock_client_manager)

        # Set up orchestrator with agentic orchestrator
        orchestrator.client_manager = mock_client_manager
        orchestrator.search_service = search_service
        orchestrator.document_service = document_service
        orchestrator.analytics_service = analytics_service
        orchestrator.agentic_orchestrator = mock_agentic_orchestrator

        # Mock orchestrator tool registration
        orchestrator._register_orchestrator_tools = AsyncMock()
        orchestrator._initialize_agentic_orchestration = AsyncMock()
        orchestrator._initialize_domain_services = AsyncMock()

        await orchestrator.initialize(mock_client_manager)

        return {
            "orchestrator": orchestrator,
            "search": search_service,
            "document": document_service,
            "analytics": analytics_service,
        }

    @pytest.mark.asyncio
    async def test_complex_research_workflow_orchestration(
        self, workflow_test_environment, sample_workflow_description
    ):
        """Test complex workflow orchestration across multiple services."""
        orchestrator = workflow_test_environment["orchestrator"]

        # Register orchestrator tools to test workflow orchestration
        registered_tools = []

        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)
        await orchestrator._register_orchestrator_tools()

        # Find workflow orchestration tool
        workflow_tool = None
        for call in orchestrator.mcp.tool.call_args_list:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "orchestrate_multi_service_workflow"
            ):
                workflow_tool = call[0][0]
                break

        assert workflow_tool is not None

        # Test complex workflow orchestration
        with patch(
            "src.mcp_services.orchestrator_service.create_agent_dependencies"
        ) as mock_create_deps:
            mock_deps = Mock()
            mock_create_deps.return_value = mock_deps

            result = await workflow_tool(
                workflow_description=sample_workflow_description,
                services_required=["search", "document", "analytics"],
                performance_constraints={"max_latency_ms": 5000},
            )

        # Verify workflow orchestration result
        assert result["success"] is True
        assert "workflow_results" in result
        assert "orchestration_reasoning" in result
        assert "services_used" in result
        assert "execution_time_ms" in result
        assert "confidence" in result

        # Verify agentic orchestrator was used
        orchestrator.agentic_orchestrator.orchestrate.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_performance_optimization_workflow(
        self, workflow_test_environment, mock_discovery_engine
    ):
        """Test service performance optimization workflow."""
        orchestrator = workflow_test_environment["orchestrator"]
        orchestrator.discovery_engine = mock_discovery_engine

        # Register orchestrator tools
        registered_tools = []

        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)
        await orchestrator._register_orchestrator_tools()

        # Find performance optimization tool
        optimization_tool = None
        for call in orchestrator.mcp.tool.call_args_list:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "optimize_service_performance"
            ):
                optimization_tool = call[0][0]
                break

        assert optimization_tool is not None

        # Test performance optimization workflow
        result = await optimization_tool()

        # Verify optimization workflow
        assert "optimization_applied" in result
        assert "recommendations" in result
        assert "performance_impact" in result

        # Verify discovery engine was consulted
        mock_discovery_engine.get_tool_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_service_error_propagation_and_recovery(
        self, workflow_test_environment
    ):
        """Test error propagation and recovery across services."""
        services = workflow_test_environment
        orchestrator = services["orchestrator"]

        # Make analytics service fail
        services["analytics"].get_service_info = AsyncMock(
            side_effect=Exception("Analytics service error")
        )

        # Test service capabilities aggregation with failure
        all_services = await orchestrator.get_all_services()

        # Verify error handling and recovery
        assert all_services["analytics"]["status"] == "error"
        assert "Analytics service error" in all_services["analytics"]["error"]

        # Other services should continue working
        assert all_services["search"]["status"] == "active"
        assert all_services["document"]["status"] == "active"


class TestMCPServicesEnterpriseIntegration:
    """Test enterprise integration scenarios and observability."""

    @pytest.mark.asyncio
    async def test_analytics_service_enterprise_observability_integration(
        self, mock_client_manager, mock_observability_components
    ):
        """Test analytics service integration with existing enterprise observability."""
        analytics_service = AnalyticsService("enterprise-analytics")

        # Mock observability integration
        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(
                return_value=mock_observability_components["ai_tracker"]
            ),
            get_correlation_manager=Mock(
                return_value=mock_observability_components["correlation_manager"]
            ),
            get_performance_monitor=Mock(
                return_value=mock_observability_components["performance_monitor"]
            ),
        ):
            # Mock tool registrations
            analytics_service._register_analytics_tools = AsyncMock()
            analytics_service._register_enhanced_observability_tools = AsyncMock()

            await analytics_service.initialize(mock_client_manager)

        # Verify observability components are integrated
        assert (
            analytics_service.ai_tracker == mock_observability_components["ai_tracker"]
        )
        assert (
            analytics_service.correlation_manager
            == mock_observability_components["correlation_manager"]
        )
        assert (
            analytics_service.performance_monitor
            == mock_observability_components["performance_monitor"]
        )

        # Verify enterprise integration details
        service_info = await analytics_service.get_service_info()
        enterprise_integration = service_info["enterprise_integration"]
        assert enterprise_integration["no_duplicate_infrastructure"] is True
        assert enterprise_integration["opentelemetry_integration"] is True

    @pytest.mark.asyncio
    async def test_no_duplicate_infrastructure_validation(
        self, mock_client_manager, mock_observability_components
    ):
        """Test that no duplicate observability infrastructure is created."""
        analytics_service = AnalyticsService("no-duplicate-analytics")

        # Track infrastructure creation calls
        ai_tracker_calls = []
        correlation_manager_calls = []
        performance_monitor_calls = []

        def track_ai_tracker():
            ai_tracker_calls.append("called")
            return mock_observability_components["ai_tracker"]

        def track_correlation_manager():
            correlation_manager_calls.append("called")
            return mock_observability_components["correlation_manager"]

        def track_performance_monitor():
            performance_monitor_calls.append("called")
            return mock_observability_components["performance_monitor"]

        # Mock observability integration with tracking
        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(side_effect=track_ai_tracker),
            get_correlation_manager=Mock(side_effect=track_correlation_manager),
            get_performance_monitor=Mock(side_effect=track_performance_monitor),
        ):
            # Mock tool registrations
            analytics_service._register_analytics_tools = AsyncMock()
            analytics_service._register_enhanced_observability_tools = AsyncMock()

            await analytics_service.initialize(mock_client_manager)

        # Verify existing infrastructure is reused, not duplicated
        assert len(ai_tracker_calls) == 1
        assert len(correlation_manager_calls) == 1
        assert len(performance_monitor_calls) == 1

        # Verify service uses existing infrastructure
        assert analytics_service.ai_tracker is not None
        assert analytics_service.correlation_manager is not None
        assert analytics_service.performance_monitor is not None


class TestMCPServicesAutonomousCapabilities:
    """Test autonomous capabilities across all services."""

    @pytest.mark.asyncio
    async def test_all_services_report_autonomous_features(
        self, complete_mcp_services_setup
    ):
        """Test that all services report their autonomous features correctly."""
        services = complete_mcp_services_setup

        expected_autonomous_features = {
            "search": [
                "provider_optimization",
                "strategy_adaptation",
                "quality_assessment",
            ],
            "document": [
                "tier_selection_optimization",
                "content_quality_assessment",
                "processing_pattern_learning",
            ],
            "analytics": [
                "failure_prediction",
                "decision_quality_tracking",
                "cost_intelligence",
            ],
            "system": [
                "fault_tolerance",
                "predictive_maintenance",
                "configuration_optimization",
            ],
            "orchestrator": [
                "service_discovery",
                "intelligent_routing",
                "self_healing_coordination",
            ],
        }

        for service_name, service in services.items():
            service_info = await service.get_service_info()
            autonomous_features = service_info["autonomous_features"]
            expected_features = expected_autonomous_features[service_name]

            for feature in expected_features:
                assert feature in autonomous_features, (
                    f"Service {service_name} missing autonomous feature {feature}"
                )

    @pytest.mark.asyncio
    async def test_autonomous_capability_assessment_integration(
        self, complete_mcp_services_setup
    ):
        """Test autonomous capability assessment across services."""
        services = complete_mcp_services_setup

        # Collect autonomous capabilities from all services
        all_autonomous_capabilities = {}
        for service_name, service in services.items():
            service_info = await service.get_service_info()
            all_autonomous_capabilities[service_name] = service_info[
                "autonomous_features"
            ]

        # Verify autonomous capability diversity
        all_features = set()
        for features in all_autonomous_capabilities.values():
            all_features.update(features)

        # Should have diverse autonomous capabilities across services
        assert (
            len(all_features) >= 15
        )  # Expect at least 15 different autonomous features

        # Verify no single service has all capabilities (proper separation of concerns)
        for features in all_autonomous_capabilities.values():
            assert len(features) < len(all_features)

    @pytest.mark.asyncio
    async def test_service_coordination_autonomous_features(
        self, complete_mcp_services_setup
    ):
        """Test that services can coordinate their autonomous features."""
        services = complete_mcp_services_setup
        orchestrator = services["orchestrator"]

        # Test orchestrator's autonomous coordination capabilities
        orchestrator_info = await orchestrator.get_service_info()
        autonomous_features = orchestrator_info["autonomous_features"]

        # Orchestrator should have coordination-specific autonomous features
        coordination_features = [
            "service_discovery",
            "intelligent_routing",
            "self_healing_coordination",
        ]
        for feature in coordination_features:
            assert feature in autonomous_features

    @pytest.mark.asyncio
    async def test_research_implementation_autonomous_validation(
        self, complete_mcp_services_setup
    ):
        """Test that implementations provide expected autonomous capabilities."""
        services = complete_mcp_services_setup

        # Validate Search service autonomous capabilities
        search_info = await services["search"].get_service_info()
        assert "strategy_adaptation" in search_info["autonomous_features"]
        assert "provider_optimization" in search_info["autonomous_features"]

        # Validate Document service autonomous capabilities
        document_info = await services["document"].get_service_info()
        assert "tier_selection_optimization" in document_info["autonomous_features"]
        assert "content_quality_assessment" in document_info["autonomous_features"]

        # Validate J1 research (Analytics service) autonomous capabilities
        analytics_info = await services["analytics"].get_service_info()
        assert "failure_prediction" in analytics_info["autonomous_features"]
        assert "decision_quality_tracking" in analytics_info["autonomous_features"]


class TestMCPServicesPerformanceAndScalability:
    """Test performance and scalability characteristics of MCP services."""

    @pytest.mark.asyncio
    async def test_services_initialization_performance(self, mock_client_manager):
        """Test that services initialize within acceptable time limits."""
        # Test each service initialization performance
        service_classes = [
            SearchService,
            DocumentService,
            AnalyticsService,
            SystemService,
        ]

        for service_class in service_classes:
            service = service_class(f"perf-test-{service_class.__name__.lower()}")

            # Mock tool registration for performance testing
            if hasattr(service, "_register_search_tools"):
                service._register_search_tools = AsyncMock()
            if hasattr(service, "_register_document_tools"):
                service._register_document_tools = AsyncMock()
            if hasattr(service, "_register_analytics_tools"):
                service._register_analytics_tools = AsyncMock()
            if hasattr(service, "_register_system_tools"):
                service._register_system_tools = AsyncMock()
            if hasattr(service, "_register_enhanced_observability_tools"):
                service._register_enhanced_observability_tools = AsyncMock()
            if hasattr(service, "_initialize_observability_integration"):
                service._initialize_observability_integration = AsyncMock()

            start_time = time.time()
            await service.initialize(mock_client_manager)
            end_time = time.time()

            # Each service should initialize quickly
            initialization_time = end_time - start_time
            assert initialization_time < 0.5, (
                f"{service_class.__name__} took {initialization_time}s to initialize"
            )

    @pytest.mark.asyncio
    async def test_concurrent_service_operations_scalability(
        self, complete_mcp_services_setup
    ):
        """Test scalability of concurrent operations across services."""
        services = complete_mcp_services_setup

        # Define concurrent operations
        async def concurrent_service_info(service):
            return await service.get_service_info()

        # Run many concurrent operations
        tasks = []
        for _ in range(20):  # 20 concurrent operations per service
            for service in services.values():
                task = asyncio.create_task(concurrent_service_info(service))
                tasks.append(task)

        # Execute all concurrent operations
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Verify all operations succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert "service" in result

        # Verify scalability (100 operations should complete quickly)
        total_time = end_time - start_time
        assert total_time < 2.0  # Should complete within 2 seconds

    @pytest.mark.asyncio
    async def test_memory_efficiency_during_operations(
        self, complete_mcp_services_setup
    ):
        """Test memory efficiency during service operations."""
        services = complete_mcp_services_setup

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform multiple operations
        for _ in range(10):
            for service in services.values():
                await service.get_service_info()

        # Check memory usage after operations
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Should not create too many objects
