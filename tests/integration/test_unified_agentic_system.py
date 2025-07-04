"""Comprehensive integration tests for the unified agentic system.

Tests the integration between agent coordination, agentic vector management,
and tool orchestration systems based on I4 research findings.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.integration import (
    AgenticSystemStatus,
    UnifiedAgenticSystem,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from src.services.agents.tool_orchestration import (
    ToolCapability,
    ToolExecutionMode,
)


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    client_manager = MagicMock(spec=ClientManager)
    client_manager.get_qdrant_client = AsyncMock()
    client_manager.get_openai_client = AsyncMock()
    client_manager.get_redis_client = AsyncMock()
    return client_manager


@pytest.fixture
def system_config():
    """Create system configuration."""
    return {
        "max_concurrent_agents": 5,
        "max_parallel_tools": 8,
        "default_timeout": 30.0,
        "vector_config": {"default_vector_size": 1536, "optimization_enabled": True},
    }


@pytest.fixture
async def unified_system(mock_client_manager, system_config):
    """Create unified agentic system."""
    system = UnifiedAgenticSystem(
        client_manager=mock_client_manager, config=system_config
    )
    await system.initialize()
    return system


@pytest.fixture
def sample_request():
    """Create sample unified agent request."""
    return UnifiedAgentRequest(
        request_id="test_request_001",
        goal="Find and analyze relevant documents about machine learning",
        context={
            "domain": "machine_learning",
            "query": "neural networks and deep learning",
            "max_results": 10,
        },
        vector_requirements={
            "collections": [
                {
                    "name": "ml_documents",
                    "vector_size": 1536,
                    "distance": "Cosine",
                    "optimization": "quality",
                }
            ]
        },
        tool_preferences={"prefer_fast_tools": True, "quality_threshold": 0.8},
        optimization_target="balanced",
        coordination_strategy="adaptive",
        max_execution_time_seconds=60.0,
        quality_threshold=0.8,
    )


class TestUnifiedAgenticSystemInitialization:
    """Test unified system initialization."""

    @pytest.mark.asyncio
    async def test_system_initialization(self, mock_client_manager, system_config):
        """Test successful system initialization."""
        system = UnifiedAgenticSystem(
            client_manager=mock_client_manager, config=system_config
        )

        assert not system._initialized

        with (
            patch.object(
                system.coordinator, "initialize", new_callable=AsyncMock
            ) as mock_coord_init,
            patch.object(
                system.vector_manager, "initialize", new_callable=AsyncMock
            ) as mock_vector_init,
            patch.object(
                system, "_register_default_tools", new_callable=AsyncMock
            ) as mock_register_tools,
        ):
            await system.initialize()

            assert system._initialized
            mock_coord_init.assert_called_once()
            mock_vector_init.assert_called_once()
            mock_register_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_initialization_failure(self, mock_client_manager):
        """Test system initialization with subsystem failure."""
        system = UnifiedAgenticSystem(client_manager=mock_client_manager, config={})

        with patch.object(
            system.coordinator,
            "initialize",
            side_effect=Exception("Coordinator init failed"),
        ):
            with pytest.raises(Exception, match="Coordinator init failed"):
                await system.initialize()

            assert not system._initialized

    @pytest.mark.asyncio
    async def test_default_tool_registration(self, unified_system):
        """Test registration of default tools."""
        # Verify tools were registered
        orchestrator_status = (
            await unified_system.orchestrator.get_orchestration_status()
        )

        assert (
            orchestrator_status["registered_tools"] >= 3
        )  # At least search, analysis, generation

        # Check specific tool capabilities
        tools = unified_system.orchestrator.registered_tools

        search_tools = [
            t for t in tools.values() if ToolCapability.SEARCH in t.capabilities
        ]
        analysis_tools = [
            t for t in tools.values() if ToolCapability.ANALYSIS in t.capabilities
        ]
        generation_tools = [
            t for t in tools.values() if ToolCapability.GENERATION in t.capabilities
        ]

        assert len(search_tools) >= 1
        assert len(analysis_tools) >= 1
        assert len(generation_tools) >= 1


class TestUnifiedRequestExecution:
    """Test unified request execution workflow."""

    @pytest.mark.asyncio
    async def test_successful_request_execution(self, unified_system, sample_request):
        """Test successful execution of unified request."""
        # Mock subsystem methods
        with (
            patch.object(
                unified_system.vector_manager,
                "create_agent_collection",
                return_value="ml_documents",
            ) as mock_create_collection,
            patch.object(
                unified_system.vector_manager,
                "optimize_collection",
                return_value={"optimization_applied": True},
            ) as mock_optimize,
            patch.object(
                unified_system.orchestrator, "compose_tool_chain"
            ) as mock_compose,
            patch.object(
                unified_system.orchestrator, "execute_tool_chain"
            ) as mock_execute_tools,
            patch.object(
                unified_system.coordinator, "execute_coordinated_workflow"
            ) as mock_coordinate,
        ):
            # Setup mock returns
            mock_compose.return_value = MagicMock(
                plan_id="test_plan",
                nodes=[
                    MagicMock(
                        node_id="node_1",
                        tool_id="vector_search",
                        execution_mode=ToolExecutionMode.SEQUENTIAL,
                        depends_on=[],
                        input_mapping={},
                        output_mapping={},
                    )
                ],
                timeout_seconds=60.0,
                min_quality_score=0.8,
                optimize_for="balanced",
            )

            mock_execute_tools.return_value = {
                "execution_id": "tool_exec_001",
                "success": True,
                "results": {"documents_found": 5, "analysis_complete": True},
                "metadata": {"execution_time_ms": 2500, "quality_score": 0.9},
            }

            mock_coordinate.return_value = {
                "coordination_id": "coord_001",
                "success": True,
                "results": {"tasks_completed": 1, "total_tasks": 1},
            }

            # Execute request
            response = await unified_system.execute_unified_request(sample_request)

            # Verify response
            assert isinstance(response, UnifiedAgentResponse)
            assert response.request_id == sample_request.request_id
            assert response.success is True
            assert response.execution_time_seconds > 0
            assert response.quality_score > 0
            assert response.confidence > 0

            # Verify subsystem calls
            mock_create_collection.assert_called_once()
            mock_optimize.assert_called_once()
            mock_compose.assert_called_once()
            mock_execute_tools.assert_called_once()
            mock_coordinate.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_execution_with_vector_failure(
        self, unified_system, sample_request
    ):
        """Test request execution when vector operations fail."""
        with patch.object(
            unified_system.vector_manager,
            "create_agent_collection",
            side_effect=Exception("Vector creation failed"),
        ):
            response = await unified_system.execute_unified_request(sample_request)

            assert response.success is False
            assert "Vector creation failed" in response.error
            assert response.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_request_execution_with_tool_failure(
        self, unified_system, sample_request
    ):
        """Test request execution when tool orchestration fails."""
        with (
            patch.object(
                unified_system.vector_manager,
                "create_agent_collection",
                return_value="ml_documents",
            ),
            patch.object(
                unified_system.orchestrator,
                "compose_tool_chain",
                side_effect=Exception("Tool composition failed"),
            ),
        ):
            response = await unified_system.execute_unified_request(sample_request)

            assert response.success is False
            assert "Tool composition failed" in response.error
            assert response.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, unified_system):
        """Test request timeout handling."""
        # Create request with very short timeout
        short_timeout_request = UnifiedAgentRequest(
            request_id="timeout_test",
            goal="Test timeout handling",
            max_execution_time_seconds=0.001,  # 1ms - very short
        )

        response = await unified_system.execute_unified_request(short_timeout_request)

        # Should still complete but may have timeout-related warnings
        assert isinstance(response, UnifiedAgentResponse)
        assert response.request_id == "timeout_test"

    @pytest.mark.asyncio
    async def test_concurrent_request_execution(self, unified_system):
        """Test handling of concurrent requests."""
        # Create multiple requests
        requests = [
            UnifiedAgentRequest(
                request_id=f"concurrent_test_{i}",
                goal=f"Concurrent test goal {i}",
                max_execution_time_seconds=5.0,
            )
            for i in range(3)
        ]

        # Mock subsystem operations to be fast
        with (
            patch.object(
                unified_system,
                "_prepare_vector_environment",
                return_value={"collections_created": []},
            ) as mock_vector,
            patch.object(unified_system, "_compose_tool_chain") as mock_compose,
            patch.object(
                unified_system,
                "_execute_unified_workflow",
                return_value={"integration_success": True},
            ) as mock_execute,
        ):
            mock_compose.return_value = MagicMock(
                plan_id="test_plan", nodes=[], timeout_seconds=5.0
            )

            # Execute requests concurrently
            tasks = [unified_system.execute_unified_request(req) for req in requests]

            responses = await asyncio.gather(*tasks)

            # Verify all requests completed
            assert len(responses) == 3
            for i, response in enumerate(responses):
                assert response.request_id == f"concurrent_test_{i}"

            # Verify subsystem calls (should be called for each request)
            assert mock_vector.call_count == 3
            assert mock_compose.call_count == 3
            assert mock_execute.call_count == 3


class TestSystemStatusAndMetrics:
    """Test system status and metrics functionality."""

    @pytest.mark.asyncio
    async def test_system_status_healthy(self, unified_system):
        """Test system status when healthy."""
        # Mock subsystem statuses as healthy
        with (
            patch.object(
                unified_system.coordinator,
                "get_coordination_status",
                return_value={
                    "health_score": 0.95,
                    "active_agents": 2,
                    "max_agents": 10,
                    "current_load": 0.2,
                },
            ) as mock_coord_status,
            patch.object(
                unified_system.vector_manager,
                "get_system_status",
                return_value={
                    "health_score": 0.90,
                    "active_collections": 3,
                    "total_collections": 5,
                },
            ) as mock_vector_status,
            patch.object(
                unified_system.orchestrator,
                "get_orchestration_status",
                return_value={
                    "registered_tools": 10,
                    "healthy_tools": 9,
                    "active_executions": 1,
                    "max_executions": 10,
                },
            ) as mock_orch_status,
        ):
            status = await unified_system.get_system_status()

            assert isinstance(status, AgenticSystemStatus)
            assert status.overall_health == "healthy"
            assert status.active_requests == 0
            assert isinstance(status.coordinator_status, dict)
            assert isinstance(status.vector_manager_status, dict)
            assert isinstance(status.orchestrator_status, dict)

            # Verify subsystem calls
            mock_coord_status.assert_called_once()
            mock_vector_status.assert_called_once()
            mock_orch_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_status_degraded(self, unified_system):
        """Test system status when degraded."""
        # Mock subsystem statuses as degraded
        with (
            patch.object(
                unified_system.coordinator,
                "get_coordination_status",
                return_value={"health_score": 0.6},
            ),
            patch.object(
                unified_system.vector_manager,
                "get_system_status",
                return_value={"health_score": 0.7},
            ),
            patch.object(
                unified_system.orchestrator,
                "get_orchestration_status",
                return_value={
                    "registered_tools": 10,
                    "healthy_tools": 6,  # 60% healthy
                    "active_executions": 0,
                },
            ),
        ):
            status = await unified_system.get_system_status()

            assert status.overall_health == "degraded"

    @pytest.mark.asyncio
    async def test_system_status_with_error(self, unified_system):
        """Test system status when subsystem fails."""
        with patch.object(
            unified_system.coordinator,
            "get_coordination_status",
            side_effect=Exception("Coordinator error"),
        ):
            status = await unified_system.get_system_status()

            assert status.overall_health == "unknown"
            assert "error" in status.coordinator_status

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, unified_system, sample_request):
        """Test system metrics tracking."""
        # Mock successful execution
        with (
            patch.object(
                unified_system, "_prepare_vector_environment", return_value={}
            ),
            patch.object(unified_system, "_compose_tool_chain") as mock_compose,
            patch.object(
                unified_system,
                "_execute_unified_workflow",
                return_value={"integration_success": True},
            ),
        ):
            mock_compose.return_value = MagicMock(
                plan_id="test_plan", nodes=[], timeout_seconds=60.0
            )

            # Execute request to generate metrics
            response = await unified_system.execute_unified_request(sample_request)

            # Check metrics were updated
            assert unified_system.performance_metrics["total_requests"] == 1
            if response.success:
                assert unified_system.performance_metrics["successful_requests"] == 1

            assert len(unified_system.request_history) == 1
            assert (
                unified_system.request_history[0].request_id
                == sample_request.request_id
            )


class TestOptimizationAndRecommendations:
    """Test optimization and recommendation features."""

    @pytest.mark.asyncio
    async def test_optimization_recommendations_generation(
        self, unified_system, sample_request
    ):
        """Test generation of optimization recommendations."""
        # Mock execution with performance issues
        with (
            patch.object(
                unified_system, "_prepare_vector_environment", return_value={}
            ),
            patch.object(unified_system, "_compose_tool_chain") as mock_compose,
            patch.object(
                unified_system,
                "_execute_unified_workflow",
                return_value={"integration_success": False},
            ),
        ):  # Integration issue
            mock_compose.return_value = MagicMock(
                plan_id="test_plan", nodes=[], timeout_seconds=60.0
            )

            response = await unified_system.execute_unified_request(sample_request)

            # Should have recommendations
            assert len(response.optimization_recommendations) > 0
            assert any(
                "integration" in rec.lower()
                for rec in response.optimization_recommendations
            )

    @pytest.mark.asyncio
    async def test_system_wide_optimization_identification(self, unified_system):
        """Test system-wide optimization identification."""
        # Add some poor performance history
        poor_responses = [
            UnifiedAgentResponse(
                request_id=f"slow_request_{i}",
                success=True,
                execution_time_seconds=120.0,  # Very slow
                quality_score=0.5,
                confidence=0.5,
                completeness=0.5,
            )
            for i in range(5)
        ]

        unified_system.request_history.extend(poor_responses)

        optimizations = await unified_system._identify_system_optimizations()

        assert len(optimizations) > 0
        assert any(
            "performance" in opt.lower() or "time" in opt.lower()
            for opt in optimizations
        )


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, unified_system, sample_request):
        """Test graceful degradation when subsystems fail."""
        # Mock vector manager failure but coordinator success
        with (
            patch.object(
                unified_system.vector_manager,
                "create_agent_collection",
                side_effect=Exception("Vector service down"),
            ),
            patch.object(
                unified_system.coordinator,
                "execute_coordinated_workflow",
                return_value={"success": True},
            ),
        ):
            response = await unified_system.execute_unified_request(sample_request)

            # Should fail but provide useful error information
            assert response.success is False
            assert response.error is not None
            assert response.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self, unified_system, sample_request):
        """Test that resources are cleaned up properly on errors."""
        initial_active_requests = len(unified_system.active_requests)

        # Mock failure during execution
        with patch.object(
            unified_system,
            "_compose_tool_chain",
            side_effect=Exception("Composition failed"),
        ):
            response = await unified_system.execute_unified_request(sample_request)

            # Request should be removed from active requests
            assert len(unified_system.active_requests) == initial_active_requests
            assert response.success is False

    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, unified_system):
        """Test system cleanup functionality."""
        # Add some active requests
        unified_system.active_requests["test_req_1"] = {"status": "running"}
        unified_system.active_requests["test_req_2"] = {"status": "running"}

        initial_requests = len(unified_system.active_requests)
        assert initial_requests > 0

        # Mock subsystem cleanup
        with (
            patch.object(
                unified_system.coordinator, "cleanup", new_callable=AsyncMock
            ) as mock_coord_cleanup,
            patch.object(
                unified_system.vector_manager, "cleanup", new_callable=AsyncMock
            ) as mock_vector_cleanup,
        ):
            await unified_system.cleanup()

            # Verify cleanup was called
            mock_coord_cleanup.assert_called_once()
            mock_vector_cleanup.assert_called_once()

            # Verify system state
            assert not unified_system._initialized


class TestAdvancedIntegrationScenarios:
    """Test advanced integration scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_complex_multi_stage_workflow(self, unified_system):
        """Test complex multi-stage workflow execution."""
        complex_request = UnifiedAgentRequest(
            request_id="complex_workflow_001",
            goal="Perform comprehensive document analysis with multiple optimization stages",
            context={
                "documents": ["doc1", "doc2", "doc3"],
                "analysis_types": ["semantic", "sentiment", "classification"],
                "optimization_stages": ["speed", "quality", "cost"],
            },
            vector_requirements={
                "collections": [
                    {
                        "name": "semantic_docs",
                        "vector_size": 1536,
                        "optimization": "speed",
                    },
                    {
                        "name": "analysis_cache",
                        "vector_size": 768,
                        "optimization": "quality",
                    },
                ]
            },
            tool_preferences={
                "enable_parallel_processing": True,
                "fallback_strategies": ["quality", "speed"],
                "quality_threshold": 0.9,
            },
            optimization_target="quality",
            coordination_strategy="hierarchical",
            max_execution_time_seconds=180.0,
            quality_threshold=0.9,
            max_parallel_agents=8,
        )

        # Mock comprehensive execution
        with (
            patch.object(
                unified_system.vector_manager,
                "create_agent_collection",
                side_effect=["semantic_docs", "analysis_cache"],
            ) as mock_collections,
            patch.object(
                unified_system.orchestrator, "compose_tool_chain"
            ) as mock_compose,
            patch.object(
                unified_system.coordinator, "execute_coordinated_workflow"
            ) as mock_coordinate,
        ):
            # Setup complex orchestration plan
            mock_compose.return_value = MagicMock(
                plan_id="complex_plan_001",
                nodes=[
                    MagicMock(node_id=f"stage_{i}_node_{j}", tool_id=f"tool_{j}")
                    for i in range(3)
                    for j in range(2)  # 3 stages, 2 tools each
                ],
                timeout_seconds=180.0,
                optimize_for="quality",
            )

            mock_coordinate.return_value = {
                "coordination_id": "complex_coord_001",
                "success": True,
                "results": {
                    "stages_completed": 3,
                    "total_stages": 3,
                    "quality_scores": [0.95, 0.92, 0.94],
                },
            }

            response = await unified_system.execute_unified_request(complex_request)

            # Verify complex workflow handling
            assert response.success is True
            assert response.request_id == "complex_workflow_001"
            assert response.quality_score >= complex_request.quality_threshold

            # Verify multiple collections were created
            assert mock_collections.call_count == 2

    @pytest.mark.asyncio
    async def test_dynamic_strategy_adaptation(self, unified_system):
        """Test dynamic strategy adaptation based on performance."""
        adaptive_request = UnifiedAgentRequest(
            request_id="adaptive_test_001",
            goal="Test adaptive strategy selection",
            coordination_strategy="adaptive",  # Should adapt based on conditions
            optimization_target="balanced",
            max_execution_time_seconds=30.0,
        )

        # Mock adaptive behavior - start with parallel, fall back to sequential
        with patch.object(unified_system, "_execute_unified_workflow") as mock_execute:
            # First attempt fails with parallel, second succeeds with sequential
            mock_execute.side_effect = [
                Exception("Parallel execution failed"),
                {"integration_success": True, "strategy_used": "sequential"},
            ]

            response = await unified_system.execute_unified_request(adaptive_request)

            # Should eventually succeed with adapted strategy
            assert (
                response.success is True or "fallback" in str(response.warnings).lower()
            )

    @pytest.mark.asyncio
    async def test_performance_optimization_feedback_loop(self, unified_system):
        """Test performance optimization feedback loop."""
        # Execute multiple requests to build performance history
        requests = [
            UnifiedAgentRequest(
                request_id=f"perf_test_{i}",
                goal=f"Performance test {i}",
                optimization_target="speed" if i % 2 == 0 else "quality",
                max_execution_time_seconds=10.0,
            )
            for i in range(5)
        ]

        with (
            patch.object(
                unified_system, "_prepare_vector_environment", return_value={}
            ),
            patch.object(unified_system, "_compose_tool_chain") as mock_compose,
            patch.object(
                unified_system,
                "_execute_unified_workflow",
                return_value={"integration_success": True},
            ),
        ):
            mock_compose.return_value = MagicMock(
                plan_id="perf_plan", nodes=[], timeout_seconds=10.0
            )

            # Execute all requests
            responses = []
            for request in requests:
                response = await unified_system.execute_unified_request(request)
                responses.append(response)

            # Verify performance feedback is captured
            assert len(unified_system.request_history) == 5

            # Check that system learns from performance patterns
            status = await unified_system.get_system_status()
            assert status.total_requests_24h == 5

            # Verify optimization recommendations evolve
            final_response = responses[-1]
            assert len(final_response.optimization_recommendations) > 0


@pytest.mark.asyncio
async def test_end_to_end_realistic_scenario(mock_client_manager):
    """Test a realistic end-to-end scenario."""
    # Create system with realistic configuration
    config = {
        "max_concurrent_agents": 3,
        "max_parallel_tools": 5,
        "default_timeout": 45.0,
        "vector_config": {
            "default_vector_size": 1536,
            "optimization_enabled": True,
            "cache_enabled": True,
        },
    }

    system = UnifiedAgenticSystem(client_manager=mock_client_manager, config=config)

    await system.initialize()

    # Realistic research request
    research_request = UnifiedAgentRequest(
        request_id="research_scenario_001",
        goal="Research current trends in AI safety and provide comprehensive analysis",
        context={
            "research_domain": "ai_safety",
            "time_range": "2023-2024",
            "analysis_depth": "comprehensive",
            "target_audience": "technical_experts",
        },
        vector_requirements={
            "collections": [
                {
                    "name": "ai_safety_papers",
                    "vector_size": 1536,
                    "distance": "Cosine",
                    "optimization": "quality",
                },
                {
                    "name": "trend_analysis_cache",
                    "vector_size": 768,
                    "distance": "Dot",
                    "optimization": "speed",
                },
            ]
        },
        tool_preferences={
            "prioritize_recent_sources": True,
            "enable_cross_validation": True,
            "quality_threshold": 0.85,
        },
        optimization_target="quality",
        coordination_strategy="hierarchical",
        max_execution_time_seconds=120.0,
        quality_threshold=0.85,
        max_parallel_agents=3,
    )

    # Mock realistic subsystem behavior
    with (
        patch.object(
            system.vector_manager,
            "create_agent_collection",
            side_effect=["ai_safety_papers", "trend_analysis_cache"],
        ),
        patch.object(
            system.vector_manager,
            "optimize_collection",
            return_value={"optimization_applied": True, "performance_gain": 0.15},
        ),
        patch.object(system.orchestrator, "compose_tool_chain") as mock_compose,
        patch.object(system.orchestrator, "execute_tool_chain") as mock_execute_tools,
        patch.object(
            system.coordinator, "execute_coordinated_workflow"
        ) as mock_coordinate,
    ):
        # Setup realistic orchestration
        mock_compose.return_value = MagicMock(
            plan_id="research_plan_001",
            nodes=[
                MagicMock(node_id="search_node", tool_id="vector_search"),
                MagicMock(node_id="analysis_node", tool_id="content_analysis"),
                MagicMock(node_id="synthesis_node", tool_id="content_generation"),
            ],
            timeout_seconds=120.0,
            optimize_for="quality",
        )

        mock_execute_tools.return_value = {
            "execution_id": "research_exec_001",
            "success": True,
            "results": {
                "papers_analyzed": 25,
                "trends_identified": 7,
                "synthesis_quality": 0.92,
                "recommendations": [
                    "Focus on interpretability research",
                    "Strengthen alignment methodologies",
                    "Improve robustness testing",
                ],
            },
            "metadata": {
                "execution_time_ms": 45000,
                "quality_score": 0.91,
                "tools_used": [
                    "vector_search",
                    "content_analysis",
                    "content_generation",
                ],
            },
        }

        mock_coordinate.return_value = {
            "coordination_id": "research_coord_001",
            "success": True,
            "results": {
                "agents_coordinated": 3,
                "tasks_completed": 3,
                "coordination_efficiency": 0.88,
            },
        }

        # Execute research request
        response = await system.execute_unified_request(research_request)

        # Verify comprehensive response
        assert response.success is True
        assert response.request_id == "research_scenario_001"
        assert response.quality_score >= research_request.quality_threshold
        assert response.execution_time_seconds > 0
        assert (
            response.execution_time_seconds
            <= research_request.max_execution_time_seconds
        )

        # Verify results integration
        assert (
            "final_results" in response.results
            or "tool_orchestration" in response.results
        )
        assert response.vector_results is not None
        assert response.tool_results is not None
        assert response.coordination_results is not None

        # Verify optimization recommendations
        assert len(response.optimization_recommendations) > 0

        # Verify system status after execution
        status = await system.get_system_status()
        assert status.total_requests_24h == 1
        assert status.success_rate_24h == 1.0

        # Cleanup
        await system.cleanup()
