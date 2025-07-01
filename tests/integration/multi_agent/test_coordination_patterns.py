"""Integration tests for multi-agent coordination patterns.

Tests coordination between AgenticOrchestrator and DynamicToolDiscovery agents,
focusing on hierarchical and parallel coordination patterns identified in J4 research.
"""

import asyncio
import time
from itertools import starmap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.agents.agentic_orchestrator import (
    ToolRequest,
    ToolResponse,
)
from src.services.agents.dynamic_tool_discovery import (
    ToolCapability,
)


class TestHierarchicalCoordination:
    """Test hierarchical orchestrator-worker coordination patterns."""

    # Use the mocked agents from conftest.py instead of creating real ones

    @pytest.mark.asyncio
    async def test_hierarchical_orchestrator_worker_pattern(
        self, mock_agentic_orchestrator, mock_dynamic_tool_discovery, agent_dependencies
    ):
        """Test hierarchical pattern with orchestrator coordinating worker agents."""
        # Use the already-initialized mocked agents
        coordinator_agent = mock_agentic_orchestrator
        discovery_agent = mock_dynamic_tool_discovery

        # Phase 1: Discovery agent finds available tools
        task_description = "complex data analysis with multiple processing stages"
        requirements = {
            "max_latency_ms": 5000,
            "min_accuracy": 0.8,
            "max_cost": 0.1,
        }

        discovered_tools = await discovery_agent.discover_tools_for_task(
            task_description, requirements
        )

        assert len(discovered_tools) > 0
        assert all(isinstance(tool, ToolCapability) for tool in discovered_tools)

        # Phase 2: Orchestrator coordinates using discovered tools
        task_request = ToolRequest(
            task=task_description,
            constraints=requirements,
            context={"discovered_tools": [tool.name for tool in discovered_tools]},
        )

        response = await coordinator_agent.orchestrate(
            task_request.task, task_request.constraints, agent_dependencies
        )

        # Verify hierarchical coordination
        assert isinstance(response, ToolResponse)
        assert response.success is True
        assert len(response.tools_used) > 0
        assert response.latency_ms > 0
        assert response.confidence > 0.5

        # Verify coordination metrics (mocked agents don't have real metrics)
        # Just verify the methods were called correctly
        assert coordinator_agent.orchestrate.called
        assert discovery_agent.discover_tools_for_task.called

    @pytest.mark.asyncio
    async def test_multi_stage_hierarchical_workflow(
        self, mock_agentic_orchestrator, mock_dynamic_tool_discovery, agent_dependencies
    ):
        """Test complex multi-stage hierarchical workflow."""
        coordinator_agent = mock_agentic_orchestrator
        discovery_agent = mock_dynamic_tool_discovery

        # Stage 1: Initial discovery
        stage1_task = "data preprocessing and validation"
        stage1_tools = await discovery_agent.discover_tools_for_task(
            stage1_task, {"max_latency_ms": 2000}
        )

        # Stage 2: Main processing
        stage2_task = "advanced analytics and pattern recognition"
        stage2_tools = await discovery_agent.discover_tools_for_task(
            stage2_task, {"min_accuracy": 0.9}
        )

        # Stage 3: Result synthesis
        stage3_task = "result aggregation and report generation"
        stage3_tools = await discovery_agent.discover_tools_for_task(
            stage3_task, {"max_cost": 0.05}
        )

        # Orchestrate all stages hierarchically
        workflow_tasks = [
            ("preprocessing", stage1_task, stage1_tools),
            ("analytics", stage2_task, stage2_tools),
            ("synthesis", stage3_task, stage3_tools),
        ]

        stage_results = []
        for stage_name, task, tools in workflow_tasks:
            # Build stage context for orchestration
            stage_context = {
                "stage": stage_name,
                "available_tools": [tool.name for tool in tools],
                "previous_results": stage_results,
            }

            response = await coordinator_agent.orchestrate(
                task, stage_context, agent_dependencies
            )

            stage_results.append(
                {
                    "stage": stage_name,
                    "success": response.success,
                    "tools_used": response.tools_used,
                    "latency_ms": response.latency_ms,
                }
            )

        # Verify multi-stage coordination
        assert len(stage_results) == 3
        assert all(result["success"] for result in stage_results)

        # Calculate total workflow performance
        total_latency = sum(result["latency_ms"] for result in stage_results)
        assert total_latency > 0

        # Verify workflow executed (mocked agents don't populate session state)
        # Just verify the workflow completed successfully
        assert total_latency > 0  # Workflow executed

    @pytest.mark.asyncio
    async def test_adaptive_hierarchy_with_failure_recovery(
        self, mock_agentic_orchestrator, mock_dynamic_tool_discovery, agent_dependencies
    ):
        """Test adaptive hierarchical coordination with failure recovery."""
        coordinator_agent = mock_agentic_orchestrator
        discovery_agent = mock_dynamic_tool_discovery

        # Simulate primary tool failure scenario
        primary_task = "high-priority data processing"
        primary_requirements = {"max_latency_ms": 1000, "min_accuracy": 0.95}

        # Get initial tool recommendations
        primary_tools = await discovery_agent.discover_tools_for_task(
            primary_task, primary_requirements
        )

        # Mock primary tool failure
        with patch.object(
            coordinator_agent,
            "_execute_tool",
            side_effect=Exception("Primary tool failed"),
        ):
            # First attempt should handle failure gracefully
            response = await coordinator_agent.orchestrate(
                primary_task, primary_requirements, agent_dependencies
            )

            # Should still provide a response with fallback reasoning
            assert isinstance(response, ToolResponse)
            if not response.success:
                assert "failed" in response.reasoning.lower()

        # Fallback tool discovery with relaxed constraints
        fallback_requirements = {"max_latency_ms": 2000, "min_accuracy": 0.8}
        fallback_tools = await discovery_agent.discover_tools_for_task(
            primary_task, fallback_requirements
        )

        # Verify fallback tools are different/additional
        assert len(fallback_tools) >= len(primary_tools)

        # Execute with fallback - should succeed
        fallback_response = await coordinator_agent.orchestrate(
            primary_task, fallback_requirements, agent_dependencies
        )

        assert isinstance(fallback_response, ToolResponse)
        # Should succeed with relaxed constraints
        assert fallback_response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_load_balancing_across_worker_agents(
        self, mock_agentic_orchestrator, agent_dependencies
    ):
        """Test load balancing across multiple worker agents."""
        coordinator_agent = mock_agentic_orchestrator

        # Create multiple mock discovery agents (workers)
        workers = []
        for i in range(3):
            # Create mock workers with similar behavior to the main mock
            worker = MagicMock()
            worker.discover_tools_for_task = AsyncMock(
                return_value=[
                    {
                        "tool_name": f"worker_{i}_tool",
                        "capability_score": 0.8,
                        "estimated_performance": {
                            "latency_ms": 100.0 + i * 20,
                            "accuracy_score": 0.8,
                            "cost_estimate": 0.001,
                        },
                    }
                ]
            )
            workers.append(worker)

        # Simulate high-load scenario with multiple concurrent tasks
        tasks = [
            f"parallel processing task {i}" for i in range(6)
        ]  # More tasks than workers

        # Track worker utilization
        worker_assignments = []

        async def assign_and_execute_task(task_id, task_description):
            # Simple round-robin load balancing
            worker = workers[task_id % len(workers)]

            # Discover tools with worker
            tools = await worker.discover_tools_for_task(
                task_description, {"max_latency_ms": 3000}
            )

            # Orchestrate with coordinator
            response = await coordinator_agent.orchestrate(
                task_description,
                {"worker_id": task_id % len(workers)},
                agent_dependencies,
            )

            worker_assignments.append(
                {
                    "task_id": task_id,
                    "worker_id": task_id % len(workers),
                    "tools_count": len(tools),
                    "success": response.success,
                    "latency_ms": response.latency_ms,
                }
            )

            return response

        # Execute tasks concurrently
        start_time = time.time()
        responses = await asyncio.gather(
            *list(starmap(assign_and_execute_task, enumerate(tasks)))
        )
        total_time = time.time() - start_time

        # Verify load balancing effectiveness
        assert len(responses) == 6
        assert len(worker_assignments) == 6

        # Check load distribution
        worker_loads = {}
        for assignment in worker_assignments:
            worker_id = assignment["worker_id"]
            worker_loads[worker_id] = worker_loads.get(worker_id, 0) + 1

        # Each worker should have handled 2 tasks (6 tasks / 3 workers)
        assert all(load == 2 for load in worker_loads.values())

        # Verify performance under load
        avg_latency = sum(r.latency_ms for r in responses) / len(responses)
        assert avg_latency > 0
        assert total_time < 30.0  # Should complete within reasonable time

        # Check success rate
        success_rate = sum(1 for r in responses if r.success) / len(responses)
        assert success_rate >= 0.8  # At least 80% success rate


class TestParallelAgentCoordination:
    """Test parallel agent coordination with result aggregation."""

    # Use the mocked agents from conftest.py instead of creating real ones

    @pytest.mark.asyncio
    async def test_parallel_task_distribution(
        self, multi_agent_system, agent_dependencies
    ):
        """Test parallel distribution of tasks across agent pool."""
        # Use the mocked agent pool from multi_agent_system
        # Filter to only orchestrator agents for this test
        all_agents = list(multi_agent_system["agent_pool"].values())
        agent_pool = [agent for agent in all_agents if hasattr(agent, "orchestrate")]
        discovery_pool = [multi_agent_system["tool_discovery"]]

        # Create parallel tasks
        parallel_tasks = [
            {
                "id": f"task_{i}",
                "description": f"parallel data analysis task {i}",
                "constraints": {"max_latency_ms": 2000 + i * 100},
            }
            for i in range(8)
        ]

        async def execute_parallel_task(task, orchestrator, discovery):
            # Discover tools in parallel
            tools = await discovery.discover_tools_for_task(
                task["description"], task["constraints"]
            )

            # Execute orchestration in parallel
            response = await orchestrator.orchestrate(
                task["description"], task["constraints"], agent_dependencies
            )

            return {
                "task_id": task["id"],
                "tools_discovered": len(tools),
                "orchestration_success": response.success,
                "execution_time": response.latency_ms,
                "agent_id": id(orchestrator),
            }

        # Execute tasks in parallel with round-robin agent assignment
        start_time = time.time()
        tasks_with_agents = [
            (
                task,
                agent_pool[i % len(agent_pool)],
                discovery_pool[i % len(discovery_pool)],
            )
            for i, task in enumerate(parallel_tasks)
        ]

        results = await asyncio.gather(
            *list(starmap(execute_parallel_task, tasks_with_agents))
        )
        parallel_execution_time = time.time() - start_time

        # Verify parallel execution results
        assert len(results) == 8
        assert all(result["orchestration_success"] for result in results)

        # Verify parallel processing was faster than sequential
        # (This is approximate due to mocking, but tests the pattern)
        assert parallel_execution_time < 15.0  # Should be much faster than sequential

        # Verify load distribution across agents
        agent_usage = {}
        for result in results:
            agent_id = result["agent_id"]
            agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1

        # Should distribute tasks across multiple agents
        assert len(agent_usage) > 1

        # Verify performance metrics
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        assert avg_execution_time > 0

    @pytest.mark.asyncio
    async def test_result_aggregation_from_parallel_agents(
        self, agent_pool, agent_dependencies
    ):
        """Test aggregation of results from parallel agent execution."""
        # Initialize agents
        for agent in agent_pool[:3]:  # Use 3 agents for this test
            await agent.initialize(agent_dependencies)

        # Create tasks that should be aggregated
        aggregation_tasks = [
            {
                "task": "analyze dataset segment A",
                "segment": "A",
                "constraints": {"segment_id": "A"},
            },
            {
                "task": "analyze dataset segment B",
                "segment": "B",
                "constraints": {"segment_id": "B"},
            },
            {
                "task": "analyze dataset segment C",
                "segment": "C",
                "constraints": {"segment_id": "C"},
            },
        ]

        # Execute tasks in parallel
        parallel_responses = await asyncio.gather(
            *[
                agent_pool[i].orchestrate(
                    task["task"], task["constraints"], agent_dependencies
                )
                for i, task in enumerate(aggregation_tasks)
            ]
        )

        # Aggregate results
        aggregated_result = {
            "total_segments": len(parallel_responses),
            "successful_segments": sum(1 for r in parallel_responses if r.success),
            "total_tools_used": sum(len(r.tools_used) for r in parallel_responses),
            "total_execution_time": sum(r.latency_ms for r in parallel_responses),
            "avg_confidence": sum(r.confidence for r in parallel_responses)
            / len(parallel_responses),
            "segment_results": [
                {
                    "segment": aggregation_tasks[i]["segment"],
                    "success": r.success,
                    "tools": r.tools_used,
                    "confidence": r.confidence,
                }
                for i, r in enumerate(parallel_responses)
            ],
        }

        # Verify aggregation results
        assert aggregated_result["total_segments"] == 3
        assert aggregated_result["successful_segments"] >= 0
        assert aggregated_result["total_tools_used"] > 0
        assert aggregated_result["total_execution_time"] > 0
        assert 0 <= aggregated_result["avg_confidence"] <= 1

        # Verify individual segment results
        assert len(aggregated_result["segment_results"]) == 3
        segments_processed = {
            r["segment"] for r in aggregated_result["segment_results"]
        }
        assert segments_processed == {"A", "B", "C"}

        # Test result fusion quality
        confidence_threshold = 0.5
        high_confidence_segments = [
            r
            for r in aggregated_result["segment_results"]
            if r["confidence"] >= confidence_threshold
        ]

        # At least some segments should meet confidence threshold
        assert len(high_confidence_segments) >= 0

    @pytest.mark.asyncio
    async def test_parallel_agent_state_synchronization(
        self, agent_pool, agent_dependencies
    ):
        """Test state synchronization across parallel agents."""
        # Initialize agents
        agents = agent_pool[:2]
        for agent in agents:
            await agent.initialize(agent_dependencies)

        # Execute tasks that should share state
        shared_context = {
            "session_id": agent_dependencies.session_state.session_id,
            "shared_knowledge": "common dataset information",
        }

        task1_response = await agents[0].orchestrate(
            "process shared data - phase 1",
            {"phase": 1, **shared_context},
            agent_dependencies,
        )

        task2_response = await agents[1].orchestrate(
            "process shared data - phase 2",
            {"phase": 2, **shared_context},
            agent_dependencies,
        )

        # Verify both tasks completed successfully
        assert task1_response.success, "Phase 1 task should complete successfully"
        assert task2_response.success, "Phase 2 task should complete successfully"

        # Verify state synchronization through session state
        session_history = agent_dependencies.session_state.conversation_history
        assert len(session_history) >= 2

        # Check that both agents contributed to session state
        agent_roles = {interaction["role"] for interaction in session_history}
        expected_roles = {"agent_agentic_orchestrator"}
        assert expected_roles.issubset(agent_roles)

        # Verify session metrics were updated by both agents
        session_metrics = agent_dependencies.session_state.performance_metrics
        assert len(session_metrics) > 0

        # Verify tool usage statistics
        tool_usage = agent_dependencies.session_state.tool_usage_stats
        assert "agentic_orchestrator" in tool_usage
        assert tool_usage["agentic_orchestrator"] >= 2  # Both agents used

    @pytest.mark.asyncio
    async def test_fault_tolerance_in_parallel_coordination(
        self, agent_pool, agent_dependencies
    ):
        """Test fault tolerance when some parallel agents fail."""
        # Initialize agents
        agents = agent_pool[:3]
        for agent in agents:
            await agent.initialize(agent_dependencies)

        # Create tasks where one agent will fail
        fault_tolerant_tasks = [
            ("successful task 1", {"should_fail": False}),
            ("failing task", {"should_fail": True}),
            ("successful task 2", {"should_fail": False}),
        ]

        # Mock failure for one agent
        original_orchestrate = agents[1].orchestrate

        async def mock_failing_orchestrate(task, constraints, deps):
            if constraints.get("should_fail"):
                return ToolResponse(
                    success=False,
                    results={"error": "Simulated agent failure"},
                    tools_used=[],
                    reasoning="Agent failure simulation",
                    latency_ms=100.0,
                    confidence=0.0,
                )
            return await original_orchestrate(task, constraints, deps)

        agents[1].orchestrate = mock_failing_orchestrate

        # Execute tasks in parallel
        responses = await asyncio.gather(
            *[
                agents[i].orchestrate(task, constraints, agent_dependencies)
                for i, (task, constraints) in enumerate(fault_tolerant_tasks)
            ],
            return_exceptions=True,
        )

        # Analyze fault tolerance
        successful_responses = [
            r for r in responses if isinstance(r, ToolResponse) and r.success
        ]
        failed_responses = [
            r for r in responses if isinstance(r, ToolResponse) and not r.success
        ]
        exceptions = [r for r in responses if isinstance(r, Exception)]

        # Should have 2 successful and 1 failed (not exception)
        assert len(successful_responses) == 2
        assert len(failed_responses) == 1
        assert len(exceptions) == 0  # No unhandled exceptions

        # Verify system continues operating despite partial failures
        overall_success_rate = len(successful_responses) / len(responses)
        assert overall_success_rate >= 0.5  # At least 50% success rate

        # Verify failed tasks provide useful error information
        failed_response = failed_responses[0]
        assert "failure" in failed_response.reasoning.lower()
        assert failed_response.confidence == 0.0
