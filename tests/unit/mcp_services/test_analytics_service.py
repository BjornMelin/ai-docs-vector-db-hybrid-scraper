"""Unit tests for AnalyticsService - J1 enterprise observability integration.

Tests cover:
- Service initialization and enterprise observability integration
- Agent decision metrics and multi-agent workflow visualization
- Auto-RAG performance monitoring with existing infrastructure
- Enhanced observability tools with no duplicate infrastructure
- Enterprise integration with existing AI tracker and correlation manager
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.analytics_service import AnalyticsService


class TestAnalyticsService:
    """Test AnalyticsService initialization and core functionality."""

    def test_init_creates_service_with_correct_configuration(self):
        """Test that AnalyticsService initializes with correct FastMCP configuration."""
        service = AnalyticsService("test-analytics-service")

        assert service.mcp.name == "test-analytics-service"
        assert "Advanced analytics service" in service.mcp.instructions
        assert "agentic observability" in service.mcp.instructions
        assert service.client_manager is None

    def test_init_with_default_name(self):
        """Test AnalyticsService initialization with default name."""
        service = AnalyticsService()

        assert service.mcp.name == "analytics-service"
        assert service.client_manager is None

    def test_init_contains_enterprise_integration_features(self):
        """Test that service instructions include enterprise integration features."""
        service = AnalyticsService()

        instructions = service.mcp.instructions
        assert (
            "Extends existing enterprise observability infrastructure" in instructions
        )
        assert "Integration Features" in instructions
        assert "Extends existing OpenTelemetry infrastructure" in instructions
        assert "Integrates with AIOperationTracker" in instructions
        assert (
            "Leverages existing correlation and performance monitoring" in instructions
        )

    def test_init_contains_j1_research_capabilities(self):
        """Test that service instructions include J1 research capabilities."""
        service = AnalyticsService()

        instructions = service.mcp.instructions
        assert (
            "Agent decision metrics with confidence and quality tracking"
            in instructions
        )
        assert (
            "Multi-agent workflow visualization and dependency mapping" in instructions
        )
        assert (
            "Auto-RAG performance monitoring with convergence analysis" in instructions
        )
        assert (
            "Self-healing capabilities with predictive failure detection"
            in instructions
        )

    def test_init_initializes_observability_components_as_none(self):
        """Test that observability components are initialized as None."""
        service = AnalyticsService()

        assert service.ai_tracker is None
        assert service.correlation_manager is None
        assert service.performance_monitor is None

    async def test_initialize_with_client_manager(
        self, mock_client_manager, mock_observability_components
    ):
        """Test service initialization with client manager."""
        service = AnalyticsService()

        with patch.object(
            service, "_initialize_observability_integration", new_callable=AsyncMock
        ) as mock_obs_init:
            with patch.object(
                service, "_register_analytics_tools", new_callable=AsyncMock
            ) as mock_register:
                await service.initialize(mock_client_manager)

                assert service.client_manager == mock_client_manager
                mock_obs_init.assert_called_once()
                mock_register.assert_called_once()

    async def test_initialize_logs_success_message(self, mock_client_manager, caplog):
        """Test that initialization logs success message with agentic observability."""
        service = AnalyticsService()

        with patch.object(
            service, "_initialize_observability_integration", new_callable=AsyncMock
        ):
            with patch.object(
                service, "_register_analytics_tools", new_callable=AsyncMock
            ):
                with caplog.at_level(logging.INFO):
                    await service.initialize(mock_client_manager)

                    assert (
                        "AnalyticsService initialized with agentic observability integration"
                        in caplog.text
                    )

    async def test_initialize_observability_integration(self, caplog):
        """Test observability integration initialization."""
        service = AnalyticsService()

        # Mock the observability functions
        mock_ai_tracker = Mock()
        mock_correlation_manager = Mock()
        mock_performance_monitor = Mock()

        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(return_value=mock_ai_tracker),
            get_correlation_manager=Mock(return_value=mock_correlation_manager),
            get_performance_monitor=Mock(return_value=mock_performance_monitor),
        ):
            with caplog.at_level(logging.INFO):
                await service._initialize_observability_integration()

                assert service.ai_tracker == mock_ai_tracker
                assert service.correlation_manager == mock_correlation_manager
                assert service.performance_monitor == mock_performance_monitor
                assert (
                    "Integrated with existing enterprise observability infrastructure"
                    in caplog.text
                )

    async def test_register_analytics_tools_raises_error_when_not_initialized(self):
        """Test that tool registration raises error when service not initialized."""
        service = AnalyticsService()

        with pytest.raises(RuntimeError, match="AnalyticsService not initialized"):
            await service._register_analytics_tools()

    async def test_register_analytics_tools_calls_all_tool_registrations(
        self, mock_client_manager
    ):
        """Test that all analytics tools are registered properly."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Mock all the tool modules
        mock_tools = {
            "analytics": Mock(),
            "query_processing": Mock(),
            "agentic_rag": Mock(),
        }

        for tool_name, mock_tool in mock_tools.items():
            mock_tool.register_tools = Mock()

        # Patch the tool imports and enhanced tools registration
        with patch.multiple(
            "src.mcp_services.analytics_service",
            analytics=mock_tools["analytics"],
            query_processing=mock_tools["query_processing"],
            agentic_rag=mock_tools["agentic_rag"],
        ):
            with patch.object(
                service,
                "_register_enhanced_observability_tools",
                new_callable=AsyncMock,
            ) as mock_enhanced:
                await service._register_analytics_tools()

                # Verify all core tools were registered
                mock_tools["analytics"].register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )
                mock_tools["query_processing"].register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )
                mock_tools["agentic_rag"].register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

                # Verify enhanced tools were registered
                mock_enhanced.assert_called_once()

    async def test_register_analytics_tools_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that tool registration logs success message."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Mock tool modules
        with patch.multiple(
            "src.mcp_services.analytics_service",
            analytics=Mock(register_tools=Mock()),
            query_processing=Mock(register_tools=Mock()),
            agentic_rag=Mock(register_tools=Mock()),
        ):
            with patch.object(
                service,
                "_register_enhanced_observability_tools",
                new_callable=AsyncMock,
            ):
                with caplog.at_level(logging.INFO):
                    await service._register_analytics_tools()

                    assert (
                        "Registered analytics tools with agentic observability capabilities and enterprise integration"
                        in caplog.text
                    )

    def test_get_mcp_server_returns_configured_instance(self):
        """Test that get_mcp_server returns the configured FastMCP instance."""
        service = AnalyticsService("test-service")

        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "test-service"

    async def test_get_service_info_returns_comprehensive_metadata(self):
        """Test that service info contains all expected metadata and capabilities."""
        service = AnalyticsService()

        service_info = await service.get_service_info()

        # Verify basic metadata
        assert service_info["service"] == "analytics"
        assert service_info["version"] == "2.0"
        assert service_info["status"] == "active"
        assert (
            service_info["research_basis"]
            == "J1_ENTERPRISE_AGENTIC_OBSERVABILITY_WITH_EXISTING_INTEGRATION"
        )

        # Verify capabilities
        expected_capabilities = [
            "agent_decision_metrics",
            "workflow_visualization",
            "auto_rag_monitoring",
            "performance_analytics",
            "cost_optimization",
            "predictive_monitoring",
            "agentic_observability",
        ]
        assert service_info["capabilities"] == expected_capabilities

        # Verify autonomous features
        expected_autonomous_features = [
            "failure_prediction",
            "performance_optimization",
            "cost_intelligence",
            "decision_quality_tracking",
        ]
        assert service_info["autonomous_features"] == expected_autonomous_features

    async def test_get_service_info_includes_enterprise_integration_details(self):
        """Test that service info includes enterprise integration details."""
        service = AnalyticsService()

        service_info = await service.get_service_info()

        # Verify enterprise integration section
        enterprise_integration = service_info["enterprise_integration"]
        assert enterprise_integration["opentelemetry_integration"] is True
        assert enterprise_integration["existing_ai_tracker_extended"] is True
        assert enterprise_integration["correlation_manager_leveraged"] is True
        assert enterprise_integration["performance_monitor_integrated"] is True
        assert enterprise_integration["no_duplicate_infrastructure"] is True


class TestAnalyticsServiceEnhancedObservabilityTools:
    """Test AnalyticsService enhanced observability tools with enterprise integration."""

    async def test_register_enhanced_observability_tools_creates_mcp_tools(
        self, mock_client_manager
    ):
        """Test that enhanced observability tools are registered as MCP tools."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Mock FastMCP tool decorator
        registered_tools = []

        def mock_tool_decorator(func):
            registered_tools.append(func.__name__)
            return func

        service.mcp.tool = Mock(side_effect=mock_tool_decorator)

        await service._register_enhanced_observability_tools()

        # Verify that the enhanced tools were registered
        expected_tools = [
            "get_agentic_decision_metrics",
            "get_multi_agent_workflow_visualization",
            "get_auto_rag_performance_monitoring",
        ]

        for tool_name in expected_tools:
            assert tool_name in registered_tools

    async def test_get_agentic_decision_metrics_tool_with_ai_tracker(
        self, mock_client_manager
    ):
        """Test get_agentic_decision_metrics tool with AI tracker integration."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager
        service.ai_tracker = Mock()

        # Register tools to create the actual tool functions
        await service._register_enhanced_observability_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        get_metrics_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_agentic_decision_metrics"
            ):
                get_metrics_tool = call[0][0]
                break

        assert get_metrics_tool is not None

        # Test the tool function
        result = await get_metrics_tool(agent_id="test_agent", time_range_minutes=30)

        # Verify integration with existing AI tracker
        assert result["integration_status"] == "using_existing_ai_tracker"
        assert result["time_range_minutes"] == 30
        assert result["agent_id"] == "test_agent"
        assert "decision_metrics" in result
        assert "performance_correlation" in result
        assert "enterprise_integration" in result

        # Verify enterprise integration details
        enterprise_integration = result["enterprise_integration"]
        assert enterprise_integration["opentelemetry_traces"] is True
        assert enterprise_integration["existing_metrics_extended"] is True
        assert enterprise_integration["correlation_tracking"] is True

    async def test_get_agentic_decision_metrics_tool_without_ai_tracker(
        self, mock_client_manager
    ):
        """Test get_agentic_decision_metrics tool when AI tracker is not available."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager
        service.ai_tracker = None

        # Register tools
        await service._register_enhanced_observability_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        get_metrics_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_agentic_decision_metrics"
            ):
                get_metrics_tool = call[0][0]
                break

        assert get_metrics_tool is not None

        # Test the tool function
        result = await get_metrics_tool()

        # Should return error when AI tracker not available
        assert "error" in result
        assert result["error"] == "AI tracker not available"

    async def test_get_multi_agent_workflow_visualization_tool(
        self, mock_client_manager
    ):
        """Test get_multi_agent_workflow_visualization tool with correlation manager integration."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager
        service.correlation_manager = Mock()

        # Register tools
        await service._register_enhanced_observability_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        visualization_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_multi_agent_workflow_visualization"
            ):
                visualization_tool = call[0][0]
                break

        assert visualization_tool is not None

        # Test the tool function
        result = await visualization_tool()

        # Verify integration with existing correlation manager
        assert result["integration_status"] == "using_existing_correlation_manager"
        assert "workflow_data" in result
        assert "enterprise_integration" in result

        # Verify workflow data structure
        workflow_data = result["workflow_data"]
        assert "total_workflows" in workflow_data
        assert "active_agents" in workflow_data
        assert "coordination_efficiency" in workflow_data
        assert "workflow_nodes" in workflow_data

        # Verify enterprise integration
        enterprise_integration = result["enterprise_integration"]
        assert enterprise_integration["distributed_tracing"] is True
        assert enterprise_integration["existing_correlation_extended"] is True
        assert enterprise_integration["workflow_context_propagation"] is True

    async def test_get_auto_rag_performance_monitoring_tool(self, mock_client_manager):
        """Test get_auto_rag_performance_monitoring tool with performance monitor integration."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager
        service.performance_monitor = Mock()

        # Register tools
        await service._register_enhanced_observability_tools()

        # Find the registered tool function
        tool_calls = service.mcp.tool.call_args_list
        monitoring_tool = None

        for call in tool_calls:
            if (
                hasattr(call[0][0], "__name__")
                and call[0][0].__name__ == "get_auto_rag_performance_monitoring"
            ):
                monitoring_tool = call[0][0]
                break

        assert monitoring_tool is not None

        # Test the tool function
        result = await monitoring_tool()

        # Verify integration with existing performance monitor
        assert result["integration_status"] == "using_existing_performance_monitor"
        assert "auto_rag_metrics" in result
        assert "performance_correlation" in result
        assert "enterprise_integration" in result

        # Verify Auto-RAG metrics
        auto_rag_metrics = result["auto_rag_metrics"]
        assert "iteration_count" in auto_rag_metrics
        assert "convergence_rate" in auto_rag_metrics
        assert "retrieval_effectiveness" in auto_rag_metrics
        assert "answer_quality_improvement" in auto_rag_metrics

        # Verify enterprise integration
        enterprise_integration = result["enterprise_integration"]
        assert enterprise_integration["existing_performance_metrics_extended"] is True
        assert enterprise_integration["ai_operation_tracking_integrated"] is True
        assert enterprise_integration["cost_attribution_enabled"] is True

    async def test_enhanced_tools_log_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that enhanced observability tools registration logs success message."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        with caplog.at_level(logging.INFO):
            await service._register_enhanced_observability_tools()

            assert (
                "Registered enhanced observability tools with enterprise integration"
                in caplog.text
            )


class TestAnalyticsServiceJ1ResearchIntegration:
    """Test AnalyticsService J1 research integration and enterprise observability."""

    async def test_j1_research_tool_registration(self, mock_client_manager):
        """Test that J1 research tools (agentic RAG) are properly registered."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Mock agentic_rag tool specifically (J1 research implementation)
        mock_agentic_rag = Mock()
        mock_agentic_rag.register_tools = Mock()

        with patch("src.mcp_services.analytics_service.agentic_rag", mock_agentic_rag):
            with patch.multiple(
                "src.mcp_services.analytics_service",
                analytics=Mock(register_tools=Mock()),
                query_processing=Mock(register_tools=Mock()),
            ):
                with patch.object(
                    service,
                    "_register_enhanced_observability_tools",
                    new_callable=AsyncMock,
                ):
                    await service._register_analytics_tools()

                    # Verify J1 research tools were registered
                    mock_agentic_rag.register_tools.assert_called_once_with(
                        service.mcp, mock_client_manager
                    )

    def test_service_instructions_contain_j1_research_features(self):
        """Test that service instructions reference J1 research features."""
        service = AnalyticsService()

        instructions = service.mcp.instructions

        # Check for J1 research specific features
        assert (
            "Agent decision metrics with confidence and quality tracking"
            in instructions
        )
        assert (
            "Multi-agent workflow visualization and dependency mapping" in instructions
        )
        assert (
            "Auto-RAG performance monitoring with convergence analysis" in instructions
        )
        assert (
            "Self-healing capabilities with predictive failure detection"
            in instructions
        )

    async def test_enterprise_observability_integration_no_duplication(self):
        """Test that enterprise integration doesn't create duplicate infrastructure."""
        service = AnalyticsService()

        service_info = await service.get_service_info()

        # Verify no duplicate infrastructure is created
        enterprise_integration = service_info["enterprise_integration"]
        assert enterprise_integration["no_duplicate_infrastructure"] is True
        assert enterprise_integration["existing_ai_tracker_extended"] is True
        assert enterprise_integration["correlation_manager_leveraged"] is True
        assert enterprise_integration["performance_monitor_integrated"] is True

    async def test_agentic_observability_capabilities(self):
        """Test comprehensive agentic observability capabilities."""
        service = AnalyticsService()

        service_info = await service.get_service_info()

        # Verify agentic observability capabilities
        assert "agent_decision_metrics" in service_info["capabilities"]
        assert "workflow_visualization" in service_info["capabilities"]
        assert "auto_rag_monitoring" in service_info["capabilities"]
        assert "agentic_observability" in service_info["capabilities"]

        # Verify autonomous features
        assert "failure_prediction" in service_info["autonomous_features"]
        assert "decision_quality_tracking" in service_info["autonomous_features"]

    async def test_opentelemetry_integration_capability(self):
        """Test OpenTelemetry integration capability."""
        service = AnalyticsService()

        service_info = await service.get_service_info()

        enterprise_integration = service_info["enterprise_integration"]
        assert enterprise_integration["opentelemetry_integration"] is True

    def test_service_instructions_contain_enterprise_integration_features(self):
        """Test that service instructions reference enterprise integration features."""
        service = AnalyticsService()

        instructions = service.mcp.instructions

        # Check for enterprise integration features
        assert "Extends existing OpenTelemetry infrastructure" in instructions
        assert (
            "Integrates with AIOperationTracker for agent-specific metrics"
            in instructions
        )
        assert (
            "Leverages existing correlation and performance monitoring" in instructions
        )


class TestAnalyticsServiceErrorHandling:
    """Test AnalyticsService error handling and recovery scenarios."""

    async def test_initialization_with_none_client_manager_raises_error(self):
        """Test that initialization with None client manager is handled properly."""
        service = AnalyticsService()

        # Should not raise error during initialization
        await service.initialize(None)

        # But should raise error when trying to register tools
        with pytest.raises(RuntimeError, match="AnalyticsService not initialized"):
            await service._register_analytics_tools()

    async def test_get_service_info_works_without_initialization(self):
        """Test that get_service_info works even without full initialization."""
        service = AnalyticsService()

        # Should not raise error
        service_info = await service.get_service_info()

        assert service_info["service"] == "analytics"
        assert service_info["status"] == "active"

    async def test_enhanced_tools_handle_missing_observability_components(
        self, mock_client_manager
    ):
        """Test that enhanced tools handle missing observability components gracefully."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Set all observability components to None
        service.ai_tracker = None
        service.correlation_manager = None
        service.performance_monitor = None

        # Register tools should not raise error
        await service._register_enhanced_observability_tools()

        # Tools should be registered even without observability components
        assert service.mcp.tool.called

    async def test_error_handling_during_tool_registration(self, mock_client_manager):
        """Test error handling during tool registration process."""
        service = AnalyticsService()
        service.client_manager = mock_client_manager

        # Mock a tool that raises an exception during registration
        mock_failing_tool = Mock()
        mock_failing_tool.register_tools.side_effect = Exception(
            "Tool registration failed"
        )

        with patch("src.mcp_services.analytics_service.analytics", mock_failing_tool):
            with patch.multiple(
                "src.mcp_services.analytics_service",
                query_processing=Mock(register_tools=Mock()),
                agentic_rag=Mock(register_tools=Mock()),
            ):
                with patch.object(
                    service,
                    "_register_enhanced_observability_tools",
                    new_callable=AsyncMock,
                ):
                    # Tool registration should raise the exception
                    with pytest.raises(Exception, match="Tool registration failed"):
                        await service._register_analytics_tools()

    async def test_multiple_initialization_calls_are_safe(self, mock_client_manager):
        """Test that multiple initialization calls are handled safely."""
        service = AnalyticsService()

        with patch.object(
            service, "_initialize_observability_integration", new_callable=AsyncMock
        ) as mock_obs_init:
            with patch.object(
                service, "_register_analytics_tools", new_callable=AsyncMock
            ) as mock_register:
                # First initialization
                await service.initialize(mock_client_manager)
                first_obs_calls = mock_obs_init.call_count
                first_register_calls = mock_register.call_count

                # Second initialization
                await service.initialize(mock_client_manager)
                second_obs_calls = mock_obs_init.call_count
                second_register_calls = mock_register.call_count

                # Should handle multiple calls gracefully
                assert second_obs_calls >= first_obs_calls
                assert second_register_calls >= first_register_calls


class TestAnalyticsServicePerformanceAndIntegration:
    """Test AnalyticsService performance characteristics and integration scenarios."""

    async def test_service_initialization_is_efficient(self, mock_client_manager):
        """Test that service initialization is efficient and doesn't block."""
        import time

        service = AnalyticsService()

        start_time = time.time()

        with patch.object(
            service, "_initialize_observability_integration", new_callable=AsyncMock
        ):
            with patch.object(
                service, "_register_analytics_tools", new_callable=AsyncMock
            ):
                await service.initialize(mock_client_manager)

        end_time = time.time()

        # Initialization should be fast (< 1 second)
        assert end_time - start_time < 1.0

    async def test_observability_integration_performance(self):
        """Test that observability integration is performant."""
        import time

        service = AnalyticsService()

        start_time = time.time()

        with patch.multiple(
            "src.mcp_services.analytics_service",
            get_ai_tracker=Mock(return_value=Mock()),
            get_correlation_manager=Mock(return_value=Mock()),
            get_performance_monitor=Mock(return_value=Mock()),
        ):
            await service._initialize_observability_integration()

        end_time = time.time()

        # Observability integration should be fast (< 0.1 seconds)
        assert end_time - start_time < 0.1

    async def test_service_supports_concurrent_access(self, mock_client_manager):
        """Test that service supports concurrent access patterns."""
        import asyncio

        service = AnalyticsService()

        # Simulate concurrent access
        async def concurrent_operation():
            await service.initialize(mock_client_manager)
            return await service.get_service_info()

        with patch.object(
            service, "_initialize_observability_integration", new_callable=AsyncMock
        ):
            with patch.object(
                service, "_register_analytics_tools", new_callable=AsyncMock
            ):
                # Run multiple concurrent operations
                tasks = [concurrent_operation() for _ in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # All operations should succeed
                for result in results:
                    assert not isinstance(result, Exception)
                    assert result["service"] == "analytics"
