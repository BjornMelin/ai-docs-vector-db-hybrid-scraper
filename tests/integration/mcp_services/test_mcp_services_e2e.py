"""End-to-end tests for MCP services - Real-world workflow validation.

Tests cover:
- Complete end-to-end workflows spanning multiple services
- Real-world scenarios with autonomous decision making
- Performance validation under realistic conditions
- Error handling and recovery in complex scenarios
- Service composition patterns and orchestration validation
"""

import asyncio
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


class TestMCPServicesEndToEndWorkflows:
    """Test complete end-to-end workflows across MCP services."""

    @pytest.fixture
    async def e2e_services_environment(
        self,
        mock_client_manager,
        mock_observability_components,
        mock_agentic_orchestrator,
        mock_discovery_engine,
    ):
        """Set up complete end-to-end testing environment with all services."""
        # Create all services
        search_service = SearchService("e2e-search")
        document_service = DocumentService("e2e-document")
        analytics_service = AnalyticsService("e2e-analytics")
        system_service = SystemService("e2e-system")
        orchestrator_service = OrchestratorService("e2e-orchestrator")

        # Mock all internal components
        async def mock_tool_registration(*args, **kwargs):
            pass

        # Configure services with mocks
        search_service._register_search_tools = AsyncMock()  # noqa: SLF001
        document_service._register_document_tools = AsyncMock()  # noqa: SLF001
        system_service._register_system_tools = AsyncMock()  # noqa: SLF001

        # Configure analytics service with observability mocks
        analytics_service._register_analytics_tools = AsyncMock()  # noqa: SLF001
        analytics_service._register_enhanced_observability_tools = AsyncMock()  # noqa: SLF001

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
            analytics_service._initialize_observability_integration = AsyncMock()  # noqa: SLF001
            await analytics_service.initialize(mock_client_manager)

        # Initialize other services
        await search_service.initialize(mock_client_manager)
        await document_service.initialize(mock_client_manager)
        await system_service.initialize(mock_client_manager)

        # Configure orchestrator with real services and mocks
        orchestrator_service.client_manager = mock_client_manager
        orchestrator_service.search_service = search_service
        orchestrator_service.document_service = document_service
        orchestrator_service.analytics_service = analytics_service
        orchestrator_service.system_service = system_service
        orchestrator_service.agentic_orchestrator = mock_agentic_orchestrator
        orchestrator_service.discovery_engine = mock_discovery_engine

        orchestrator_service._initialize_domain_services = AsyncMock()  # noqa: SLF001
        orchestrator_service._initialize_agentic_orchestration = AsyncMock()  # noqa: SLF001
        orchestrator_service._register_orchestrator_tools = AsyncMock()  # noqa: SLF001

        await orchestrator_service.initialize(mock_client_manager)

        return {
            "search": search_service,
            "document": document_service,
            "analytics": analytics_service,
            "system": system_service,
            "orchestrator": orchestrator_service,
        }

    async def test_complete_research_workflow_e2e(self, e2e_services_environment):
        """Test complete research workflow from search to analytics end-to-end."""
        services = e2e_services_environment
        orchestrator = services["orchestrator"]

        # Register orchestrator tools for workflow execution
        workflow_tools = {}

        def mock_tool_decorator(func):
            workflow_tools[func.__name__] = func
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)
        await orchestrator._register_orchestrator_tools()  # noqa: SLF001

        # Execute complete research workflow
        workflow_description = """
        Comprehensive research workflow:
        1. Search for relevant documents using hybrid search
        2. Process documents with 5-tier crawling
        3. Analyze results with agent decision metrics
        4. Optimize system performance based on findings
        """

        # Test workflow orchestration
        orchestrate_workflow = workflow_tools["orchestrate_multi_service_workflow"]

        with patch(
            "src.mcp_services.orchestrator_service.create_agent_dependencies"
        ) as mock_create_deps:
            mock_deps = Mock()
            mock_create_deps.return_value = mock_deps

            result = await orchestrate_workflow(
                workflow_description=workflow_description,
                services_required=["search", "document", "analytics", "system"],
                performance_constraints={
                    "max_latency_ms": 10000,
                    "min_confidence": 0.8,
                },
            )

        # Verify complete workflow execution
        assert result["success"] is True
        assert "workflow_results" in result
        assert "services_used" in result
        assert "orchestration_reasoning" in result
        assert "execution_time_ms" in result
        assert result["confidence"] >= 0.8

        # Verify agentic orchestrator was invoked
        orchestrator.agentic_orchestrator.orchestrate.assert_called_once()

        # Verify workflow constraints were applied
        call_args = orchestrator.agentic_orchestrator.orchestrate.call_args
        assert "preferred_services" in call_args[1]["constraints"]

    async def test_service_discovery_and_capability_assessment_e2e(
        self, e2e_services_environment
    ):
        """Test end-to-end service discovery and capability assessment."""
        services = e2e_services_environment
        orchestrator = services["orchestrator"]

        # Register orchestrator tools
        workflow_tools = {}

        def mock_tool_decorator(func):
            workflow_tools[func.__name__] = func
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)
        await orchestrator._register_orchestrator_tools()  # noqa: SLF001

        # Test service capability discovery
        get_capabilities = workflow_tools["get_service_capabilities"]
        capabilities_result = await get_capabilities()

        # Verify comprehensive capability discovery
        assert "services" in capabilities_result
        discovered_services = capabilities_result["services"]

        expected_services = ["search", "document", "analytics", "system"]
        for service_name in expected_services:
            assert service_name in discovered_services
            service_info = discovered_services[service_name]
            assert "capabilities" in service_info
            assert "autonomous_features" in service_info
            assert len(service_info["capabilities"]) > 0

        # Test performance optimization discovery
        optimize_performance = workflow_tools["optimize_service_performance"]
        optimization_result = await optimize_performance()

        # Verify performance optimization workflow
        assert "optimization_applied" in optimization_result
        assert "recommendations" in optimization_result
        assert "performance_impact" in optimization_result

        # Verify discovery engine was consulted
        orchestrator.discovery_engine.get_tool_recommendations.assert_called_once()

    async def test_autonomous_decision_making_e2e(self, e2e_services_environment):
        """Test autonomous decision making across services end-to-end."""
        services = e2e_services_environment

        # Test each service's autonomous capabilities
        autonomous_decisions = {}

        for service_name, service in services.items():
            if service_name == "orchestrator":
                continue  # Skip orchestrator for individual service testing

            service_info = await service.get_service_info()
            autonomous_features = service_info["autonomous_features"]

            # Verify each service has autonomous capabilities
            assert len(autonomous_features) > 0
            autonomous_decisions[service_name] = autonomous_features

        # Test analytics service enhanced autonomous capabilities
        analytics = services["analytics"]

        # Register analytics enhanced tools
        analytics_tools = {}

        def mock_analytics_tool_decorator(func):
            analytics_tools[func.__name__] = func
            return func

        analytics.mcp.tool = Mock(side_effect=mock_analytics_tool_decorator)
        await analytics._register_enhanced_observability_tools()  # noqa: SLF001

        # Test agent decision metrics (autonomous capability)
        if "get_agentic_decision_metrics" in analytics_tools:
            get_metrics = analytics_tools["get_agentic_decision_metrics"]
            metrics_result = await get_metrics(
                agent_id="test_agent", time_range_minutes=30
            )

            # Verify autonomous decision tracking
            assert "decision_metrics" in metrics_result
            assert "performance_correlation" in metrics_result
            decision_metrics = metrics_result["decision_metrics"]
            assert "avg_confidence" in decision_metrics
            assert "success_rate" in decision_metrics

    async def test_error_handling_and_recovery_e2e(self, e2e_services_environment):
        """Test error handling and recovery across services end-to-end."""
        services = e2e_services_environment
        orchestrator = services["orchestrator"]

        # Inject failures in different services
        services["search"].get_service_info = AsyncMock(
            side_effect=Exception("Search service failure")
        )
        services["document"].get_service_info = AsyncMock(
            side_effect=Exception("Document service failure")
        )

        # Test orchestrator's resilience to service failures
        all_services_result = await orchestrator.get_all_services()

        # Verify graceful degradation
        assert all_services_result["search"]["status"] == "error"
        assert all_services_result["document"]["status"] == "error"
        assert "Search service failure" in all_services_result["search"]["error"]
        assert "Document service failure" in all_services_result["document"]["error"]

        # Analytics and system services should still work
        assert all_services_result["analytics"]["status"] == "active"
        assert all_services_result["system"]["status"] == "active"

        # Test recovery - fix one service
        services["search"].get_service_info = AsyncMock(
            return_value={
                "service": "search",
                "status": "active",
                "capabilities": ["hybrid_search"],
            }
        )

        # Verify recovery
        recovered_services = await orchestrator.get_all_services()
        assert recovered_services["search"]["status"] == "active"

    async def test_performance_under_load_e2e(self, e2e_services_environment):
        """Test performance under load end-to-end."""
        services = e2e_services_environment

        # Define load test operations
        async def high_frequency_operation(service):
            return await service.get_service_info()

        # Run high-frequency operations on all services
        load_tasks = []
        operations_per_service = 50

        for service in services.values():
            for _ in range(operations_per_service):
                task = asyncio.create_task(high_frequency_operation(service))
                load_tasks.append(task)

        # Execute load test
        start_time = time.time()
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        end_time = time.time()

        # Verify all operations completed successfully
        successful_operations = 0
        for result in results:
            if not isinstance(result, Exception) and "service" in result:
                successful_operations += 1

        total_operations = len(services) * operations_per_service
        success_rate = successful_operations / total_operations

        # Verify high success rate under load
        assert success_rate >= 0.95  # At least 95% success rate

        # Verify acceptable performance under load
        total_time = end_time - start_time
        operations_per_second = total_operations / total_time
        assert operations_per_second >= 50  # At least 50 operations per second

    async def test_enterprise_integration_e2e(self, e2e_services_environment):
        """Test enterprise integration features end-to-end."""
        services = e2e_services_environment
        analytics = services["analytics"]

        # Verify enterprise observability integration
        service_info = await analytics.get_service_info()
        assert "enterprise_integration" in service_info

        enterprise_integration = service_info["enterprise_integration"]
        assert enterprise_integration["opentelemetry_integration"] is True
        assert enterprise_integration["no_duplicate_infrastructure"] is True
        assert enterprise_integration["existing_ai_tracker_extended"] is True

        # Test enhanced observability tools
        analytics_tools = {}

        def mock_analytics_tool_decorator(func):
            analytics_tools[func.__name__] = func
            return func

        analytics.mcp.tool = Mock(side_effect=mock_analytics_tool_decorator)
        await analytics._register_enhanced_observability_tools()  # noqa: SLF001

        # Test enterprise observability tools
        expected_enterprise_tools = [
            "get_agentic_decision_metrics",
            "get_multi_agent_workflow_visualization",
            "get_auto_rag_performance_monitoring",
        ]

        for tool_name in expected_enterprise_tools:
            assert tool_name in analytics_tools

            # Test each enterprise tool
            tool_func = analytics_tools[tool_name]
            result = await tool_func()

            # Verify enterprise integration in tool results
            assert "integration_status" in result
            assert "enterprise_integration" in result
            enterprise_tool_integration = result["enterprise_integration"]
            assert len(enterprise_tool_integration) > 0

    async def test_research_implementation_validation_e2e(
        self, e2e_services_environment
    ):
        """Test research implementation validation end-to-end."""
        services = e2e_services_environment

        # Validate I5 research implementation (Search service)
        search_info = await services["search"].get_service_info()
        assert "I5_WEB_SEARCH_TOOL_ORCHESTRATION" in search_info["research_basis"]
        assert "autonomous_web_search" in search_info["capabilities"]
        assert "multi_provider_orchestration" in search_info["capabilities"]

        # Validate I3 research implementation (Document service)
        document_info = await services["document"].get_service_info()
        assert "I3_5_TIER_CRAWLING_ENHANCEMENT" in document_info["research_basis"]
        assert "5_tier_crawling" in document_info["capabilities"]
        assert "intelligent_crawling" in document_info["capabilities"]

        # Validate J1 research implementation (Analytics service)
        analytics_info = await services["analytics"].get_service_info()
        assert (
            "J1_ENTERPRISE_AGENTIC_OBSERVABILITY_WITH_EXISTING_INTEGRATION"
            in analytics_info["research_basis"]
        )
        assert "agentic_observability" in analytics_info["capabilities"]
        assert "agent_decision_metrics" in analytics_info["capabilities"]

        # Validate FastMCP 2.0+ implementation (Orchestrator service)
        orchestrator_info = await services["orchestrator"].get_service_info()
        assert "FASTMCP_2_0_SERVER_COMPOSITION" in orchestrator_info["research_basis"]
        assert "multi_service_coordination" in orchestrator_info["capabilities"]
        assert "agentic_coordination" in orchestrator_info["capabilities"]


class TestMCPServicesRealWorldScenarios:
    """Test real-world scenarios and use cases."""

    async def test_intelligent_document_processing_workflow(
        self, e2e_services_environment
    ):
        """Test intelligent document processing workflow scenario."""
        services = e2e_services_environment

        # Simulate real-world document processing workflow
        search_service = services["search"]
        document_service = services["document"]
        analytics_service = services["analytics"]
        orchestrator = services["orchestrator"]

        # 1. Search for documents
        search_info = await search_service.get_service_info()
        assert "hybrid_search" in search_info["capabilities"]
        assert "provider_optimization" in search_info["autonomous_features"]

        # 2. Process documents with intelligent crawling
        document_info = await document_service.get_service_info()
        assert "5_tier_crawling" in document_info["capabilities"]
        assert "tier_selection_optimization" in document_info["autonomous_features"]

        # 3. Analyze processing results
        analytics_info = await analytics_service.get_service_info()
        assert "workflow_visualization" in analytics_info["capabilities"]
        assert "decision_quality_tracking" in analytics_info["autonomous_features"]

        # 4. Orchestrate the complete workflow
        orchestrator_info = await orchestrator.get_service_info()
        assert "workflow_orchestration" in orchestrator_info["capabilities"]
        assert "intelligent_routing" in orchestrator_info["autonomous_features"]

    async def test_autonomous_system_optimization_scenario(
        self, e2e_services_environment
    ):
        """Test autonomous system optimization scenario."""
        services = e2e_services_environment

        system_service = services["system"]
        analytics_service = services["analytics"]
        orchestrator = services["orchestrator"]

        # 1. System monitors health and performance
        system_info = await system_service.get_service_info()
        assert "health_monitoring" in system_info["capabilities"]
        assert "self_healing" in system_info["capabilities"]
        assert "predictive_maintenance" in system_info["autonomous_features"]

        # 2. Analytics provides intelligent insights
        analytics_info = await analytics_service.get_service_info()
        assert "performance_analytics" in analytics_info["capabilities"]
        assert "failure_prediction" in analytics_info["autonomous_features"]

        # 3. Orchestrator coordinates optimization
        orchestrator_info = await orchestrator.get_service_info()
        assert "cross_service_optimization" in orchestrator_info["capabilities"]
        assert "performance_optimization" in orchestrator_info["autonomous_features"]

    async def test_enterprise_observability_scenario(self, e2e_services_environment):
        """Test enterprise observability integration scenario."""
        services = e2e_services_environment
        analytics = services["analytics"]

        # Verify enterprise observability integration
        analytics_info = await analytics.get_service_info()
        enterprise_integration = analytics_info["enterprise_integration"]

        # Validate no duplicate infrastructure
        assert enterprise_integration["no_duplicate_infrastructure"] is True
        assert enterprise_integration["existing_ai_tracker_extended"] is True
        assert enterprise_integration["correlation_manager_leveraged"] is True
        assert enterprise_integration["performance_monitor_integrated"] is True

        # Verify OpenTelemetry integration
        assert enterprise_integration["opentelemetry_integration"] is True

    async def test_multi_agent_coordination_scenario(self, e2e_services_environment):
        """Test multi-agent coordination scenario."""
        services = e2e_services_environment
        orchestrator = services["orchestrator"]

        # Register orchestrator tools for coordination testing
        coordination_tools = {}

        def mock_tool_decorator(func):
            coordination_tools[func.__name__] = func
            return func

        orchestrator.mcp.tool = Mock(side_effect=mock_tool_decorator)
        await orchestrator._register_orchestrator_tools()  # noqa: SLF001

        # Test multi-service coordination
        if "orchestrate_multi_service_workflow" in coordination_tools:
            orchestrate = coordination_tools["orchestrate_multi_service_workflow"]

            workflow_description = "Multi-agent coordination test workflow"
            services_required = ["search", "document", "analytics"]

            with patch(
                "src.mcp_services.orchestrator_service.create_agent_dependencies"
            ) as mock_create_deps:
                mock_deps = Mock()
                mock_create_deps.return_value = mock_deps

                result = await orchestrate(
                    workflow_description=workflow_description,
                    services_required=services_required,
                )

            # Verify coordination was successful
            assert result["success"] is True
            assert "services_used" in result
            assert "orchestration_reasoning" in result

    async def test_scalability_and_performance_scenario(self, e2e_services_environment):
        """Test scalability and performance under realistic load."""
        services = e2e_services_environment

        # Simulate realistic concurrent operations
        concurrent_operations = []

        # Create varied operations across services
        for i in range(100):  # 100 concurrent operations
            service = services[list(services.keys())[i % len(services)]]
            operation = asyncio.create_task(service.get_service_info())
            concurrent_operations.append(operation)

        # Execute concurrent operations
        start_time = time.time()
        results = await asyncio.gather(*concurrent_operations, return_exceptions=True)
        end_time = time.time()

        # Verify performance characteristics
        successful_operations = sum(
            1 for result in results if not isinstance(result, Exception)
        )
        success_rate = successful_operations / len(concurrent_operations)

        assert success_rate >= 0.98  # Very high success rate expected

        total_time = end_time - start_time
        throughput = len(concurrent_operations) / total_time
        assert throughput >= 100  # At least 100 operations per second


class TestMCPServicesQualityAndReliability:
    """Test quality and reliability characteristics of MCP services."""

    async def test_service_consistency_and_reliability(self, e2e_services_environment):
        """Test service consistency and reliability over multiple operations."""
        services = e2e_services_environment

        # Test each service multiple times to verify consistency
        for service in services.values():
            results = []

            # Perform multiple operations
            for _ in range(10):
                result = await service.get_service_info()
                results.append(result)

            # Verify consistency across operations
            first_result = results[0]
            for result in results[1:]:
                assert result["service"] == first_result["service"]
                assert result["version"] == first_result["version"]
                assert result["capabilities"] == first_result["capabilities"]
                assert (
                    result["autonomous_features"] == first_result["autonomous_features"]
                )

    async def test_service_isolation_and_fault_tolerance(
        self, e2e_services_environment
    ):
        """Test service isolation and fault tolerance."""
        services = e2e_services_environment

        # Test that failure in one service doesn't affect others
        original_search_method = services["search"].get_service_info

        # Inject failure in search service
        services["search"].get_service_info = AsyncMock(
            side_effect=Exception("Injected failure")
        )

        # Other services should continue working normally
        for service_name, service in services.items():
            if service_name == "search":
                continue

            result = await service.get_service_info()
            assert result["status"] == "active"
            assert "capabilities" in result

        # Restore search service
        services["search"].get_service_info = original_search_method

        # Verify search service works after restoration
        search_result = await services["search"].get_service_info()
        assert search_result["status"] == "active"

    async def test_comprehensive_coverage_validation(self, e2e_services_environment):
        """Test comprehensive coverage of all expected features."""
        services = e2e_services_environment

        # Verify all expected research implementations are covered
        research_coverage = {
            "I5_WEB_SEARCH_TOOL_ORCHESTRATION": False,
            "I3_5_TIER_CRAWLING_ENHANCEMENT": False,
            "J1_ENTERPRISE_AGENTIC_OBSERVABILITY": False,
            "FASTMCP_2_0_SERVER_COMPOSITION": False,
            "SYSTEM_INFRASTRUCTURE": False,
        }

        for service in services.values():
            service_info = await service.get_service_info()
            research_basis = service_info.get("research_basis", "")

            for research_key in research_coverage:
                if research_key in research_basis:
                    research_coverage[research_key] = True

        # Verify all research areas are covered
        uncovered_research = [
            key for key, covered in research_coverage.items() if not covered
        ]
        assert len(uncovered_research) == 0, (
            f"Uncovered research areas: {uncovered_research}"
        )

        # Verify comprehensive autonomous features coverage
        all_autonomous_features = set()
        for service in services.values():
            service_info = await service.get_service_info()
            all_autonomous_features.update(service_info.get("autonomous_features", []))

        # Should have diverse autonomous capabilities
        assert len(all_autonomous_features) >= 15

        # Verify comprehensive capabilities coverage
        all_capabilities = set()
        for service in services.values():
            service_info = await service.get_service_info()
            all_capabilities.update(service_info.get("capabilities", []))

        # Should have diverse service capabilities
        assert len(all_capabilities) >= 25
