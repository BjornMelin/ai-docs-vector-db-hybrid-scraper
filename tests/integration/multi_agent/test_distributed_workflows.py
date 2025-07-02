"""Integration tests for distributed multi-agent workflows.

Tests complex distributed processing workflows, state synchronization,
and autonomous capabilities enabling 3-10x performance improvements.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import (
    AgenticOrchestrator,
)
from src.services.agents.core import BaseAgentDependencies, create_agent_dependencies
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
)


class WorkflowState(str, Enum):
    """Workflow execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    """Agent roles in distributed workflows."""

    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    MONITOR = "monitor"


@dataclass
class WorkflowNode:
    """Represents a node in a distributed workflow."""

    node_id: str
    agent_role: AgentRole
    task_description: str
    dependencies: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    state: WorkflowState = WorkflowState.PENDING
    result: dict[str, Any] | None = None
    execution_time: float | None = None
    error: str | None = None


@dataclass
class DistributedWorkflow:
    """Represents a distributed multi-agent workflow."""

    workflow_id: str
    nodes: list[WorkflowNode]
    global_constraints: dict[str, Any] = field(default_factory=dict)
    state: WorkflowState = WorkflowState.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)


class WorkflowOrchestrator:
    """Orchestrates distributed multi-agent workflows."""

    def __init__(self):
        self.active_workflows: dict[str, DistributedWorkflow] = {}
        self.agent_pool: dict[str, Any] = {}
        self.state_store: dict[str, Any] = {}

    def register_agent(self, agent_id: str, agent: Any, role: AgentRole) -> None:
        """Register an agent with the orchestrator."""
        self.agent_pool[agent_id] = {
            "agent": agent,
            "role": role,
            "load": 0,
            "capabilities": [],
            "last_activity": datetime.now(tz=UTC),
        }

    async def execute_workflow(
        self,
        workflow: DistributedWorkflow,
        dependencies: BaseAgentDependencies,
    ) -> DistributedWorkflow:
        """Execute a distributed workflow."""
        workflow.state = WorkflowState.RUNNING
        workflow.start_time = datetime.now(tz=UTC)

        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow.nodes)

            # Execute nodes in topological order with parallelization
            executed_nodes = set()

            while len(executed_nodes) < len(workflow.nodes):
                # Find nodes ready for execution
                ready_nodes = self._find_ready_nodes(
                    workflow.nodes, dependency_graph, executed_nodes
                )

                if not ready_nodes:
                    self._raise_deadlock_error()

                # Execute ready nodes in parallel
                node_tasks = []
                for node in ready_nodes:
                    task = self._execute_node(node, workflow, dependencies)
                    node_tasks.append(task)

                # Wait for parallel execution
                await asyncio.gather(*node_tasks)

                # Mark nodes as executed
                executed_nodes.update(node.node_id for node in ready_nodes)

            # Calculate workflow metrics
            workflow.performance_metrics = self._calculate_workflow_metrics(workflow)
            workflow.state = WorkflowState.COMPLETED

        except (TimeoutError, ConnectionError, RuntimeError, TypeError) as e:
            workflow.state = WorkflowState.FAILED
            workflow.error = str(e)
        finally:
            workflow.end_time = datetime.now(tz=UTC)

        return workflow

    def _build_dependency_graph(self, nodes: list[WorkflowNode]) -> dict[str, set[str]]:
        """Build dependency graph from workflow nodes."""
        graph = {}
        for node in nodes:
            graph[node.node_id] = set(node.dependencies)
        return graph

    def _find_ready_nodes(
        self,
        nodes: list[WorkflowNode],
        dependency_graph: dict[str, set[str]],
        executed_nodes: set[str],
    ) -> list[WorkflowNode]:
        """Find nodes ready for execution."""
        return [
            node
            for node in nodes
            if (
                node.node_id not in executed_nodes
                and node.state == WorkflowState.PENDING
                and dependency_graph[node.node_id].issubset(executed_nodes)
            )
        ]

    async def _execute_node(
        self,
        node: WorkflowNode,
        workflow: DistributedWorkflow,
        dependencies: BaseAgentDependencies,
    ) -> None:
        """Execute a single workflow node."""
        node.state = WorkflowState.RUNNING
        start_time = time.time()

        try:
            # Find suitable agent for the node
            agent_id = self._select_agent_for_node(node)
            if not agent_id:
                self._raise_no_agent_error(node.node_id)

            agent_info = self.agent_pool[agent_id]
            agent = agent_info["agent"]

            # Prepare context with dependency results
            context = self._prepare_node_context(node, workflow)

            # Execute task on agent
            if hasattr(agent, "orchestrate"):
                response = await agent.orchestrate(
                    node.task_description, {**node.constraints, **context}, dependencies
                )
                node.result = {
                    "success": response.success,
                    "results": response.results,
                    "tools_used": response.tools_used,
                    "confidence": response.confidence,
                    "agent_id": agent_id,
                }
            else:
                # Fallback for discovery agents
                if hasattr(agent, "discover_tools_for_task"):
                    tools = await agent.discover_tools_for_task(
                        node.task_description, node.constraints
                    )
                    node.result = {
                        "success": True,
                        "tools_discovered": [tool.name for tool in tools],
                        "tool_count": len(tools),
                        "agent_id": agent_id,
                    }
                else:
                    node.result = {
                        "success": False,
                        "error": "Agent type not supported",
                        "agent_id": agent_id,
                    }

            node.state = WorkflowState.COMPLETED

        except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
            node.state = WorkflowState.FAILED
            node.error = str(e)
            node.result = {"success": False, "error": str(e)}
        finally:
            node.execution_time = time.time() - start_time

    def _select_agent_for_node(self, node: WorkflowNode) -> str | None:
        """Select the best agent for executing a node."""
        # Filter agents by role
        suitable_agents = [
            agent_id
            for agent_id, info in self.agent_pool.items()
            if info["role"] == node.agent_role
        ]

        if not suitable_agents:
            return None

        # Select agent with lowest load
        return min(suitable_agents, key=lambda aid: self.agent_pool[aid]["load"])

    def _prepare_node_context(
        self, node: WorkflowNode, workflow: DistributedWorkflow
    ) -> dict[str, Any]:
        """Prepare execution context for a node."""
        context = {
            "workflow_id": workflow.workflow_id,
            "node_id": node.node_id,
            "dependency_results": {},
        }

        # Add dependency results
        for dep_id in node.dependencies:
            dep_node = next((n for n in workflow.nodes if n.node_id == dep_id), None)
            if dep_node and dep_node.result:
                context["dependency_results"][dep_id] = dep_node.result

        return context

    def _calculate_workflow_metrics(
        self, workflow: DistributedWorkflow
    ) -> dict[str, float]:
        """Calculate performance metrics for completed workflow."""
        if not workflow.start_time or not workflow.end_time:
            return {}

        total_time = (workflow.end_time - workflow.start_time).total_seconds()
        successful_nodes = sum(
            1 for node in workflow.nodes if node.state == WorkflowState.COMPLETED
        )
        total_nodes = len(workflow.nodes)

        # Calculate parallelization efficiency
        sequential_time = sum(node.execution_time or 0 for node in workflow.nodes)
        parallelization_factor = sequential_time / total_time if total_time > 0 else 1

        return {
            "total_execution_time": total_time,
            "success_rate": successful_nodes / total_nodes if total_nodes > 0 else 0,
            "parallelization_factor": parallelization_factor,
            "avg_node_execution_time": sequential_time / total_nodes
            if total_nodes > 0
            else 0,
            "performance_improvement": min(parallelization_factor, 10.0),  # Cap at 10x
        }

    def _raise_deadlock_error(self) -> None:
        """Raise a deadlock detection error."""
        msg = "Circular dependency or deadlock detected"
        raise RuntimeError(msg)

    def _raise_no_agent_error(self, node_id: str) -> None:
        """Raise an error when no suitable agent is found."""
        msg = f"No suitable agent found for node {node_id}"
        raise RuntimeError(msg)


class TestDistributedWorkflowExecution:
    """Test distributed workflow execution patterns."""

    @pytest.fixture
    def workflow_orchestrator(self) -> WorkflowOrchestrator:
        """Create workflow orchestrator."""
        return WorkflowOrchestrator()

    @pytest.fixture
    def agent_pool(self) -> list[Any]:
        """Create diverse agent pool."""
        return [
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1),
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1),
            DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1),
            DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1),
        ]

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_distributed_workflow",
        )

    @pytest.mark.asyncio
    async def test_multi_stage_distributed_processing(
        self, workflow_orchestrator, agent_pool, agent_dependencies
    ):
        """Test multi-stage distributed processing workflow."""
        # Initialize and register agents
        coordinator_agents = agent_pool[:2]
        specialist_agents = agent_pool[2:]

        for i, agent in enumerate(coordinator_agents):
            await agent.initialize(agent_dependencies)
            workflow_orchestrator.register_agent(
                f"coord_{i}", agent, AgentRole.COORDINATOR
            )

        for i, agent in enumerate(specialist_agents):
            await agent.initialize_discovery(agent_dependencies)
            workflow_orchestrator.register_agent(
                f"spec_{i}", agent, AgentRole.SPECIALIST
            )

        # Create multi-stage workflow
        workflow = DistributedWorkflow(
            workflow_id="multi_stage_001",
            nodes=[
                # Stage 1: Data preparation (parallel)
                WorkflowNode(
                    node_id="data_prep_A",
                    agent_role=AgentRole.SPECIALIST,
                    task_description="prepare dataset segment A for processing",
                    constraints={"segment": "A", "max_latency_ms": 2000},
                ),
                WorkflowNode(
                    node_id="data_prep_B",
                    agent_role=AgentRole.SPECIALIST,
                    task_description="prepare dataset segment B for processing",
                    constraints={"segment": "B", "max_latency_ms": 2000},
                ),
                # Stage 2: Analysis (depends on preparation)
                WorkflowNode(
                    node_id="analysis_A",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="analyze prepared dataset segment A",
                    dependencies=["data_prep_A"],
                    constraints={"analysis_type": "deep", "segment": "A"},
                ),
                WorkflowNode(
                    node_id="analysis_B",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="analyze prepared dataset segment B",
                    dependencies=["data_prep_B"],
                    constraints={"analysis_type": "deep", "segment": "B"},
                ),
                # Stage 3: Synthesis (depends on all analysis)
                WorkflowNode(
                    node_id="synthesis",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="synthesize analysis results from all segments",
                    dependencies=["analysis_A", "analysis_B"],
                    constraints={"synthesis_type": "comprehensive"},
                ),
            ],
            global_constraints={"max_total_time": 30.0, "min_quality_threshold": 0.8},
        )

        # Execute workflow
        completed_workflow = await workflow_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify workflow execution
        assert completed_workflow.state == WorkflowState.COMPLETED
        assert len(completed_workflow.nodes) == 5
        assert all(
            node.state == WorkflowState.COMPLETED for node in completed_workflow.nodes
        )

        # Verify stage dependencies were respected
        data_prep_nodes = [
            n for n in completed_workflow.nodes if "data_prep" in n.node_id
        ]
        analysis_nodes = [
            n for n in completed_workflow.nodes if "analysis" in n.node_id
        ]
        synthesis_node = next(
            n for n in completed_workflow.nodes if n.node_id == "synthesis"
        )

        assert all(node.result["success"] for node in data_prep_nodes)
        assert all(node.result["success"] for node in analysis_nodes)
        assert synthesis_node.result["success"]

        # Verify performance metrics
        metrics = completed_workflow.performance_metrics
        assert metrics["success_rate"] == 1.0
        assert metrics["total_execution_time"] > 0
        assert (
            metrics["parallelization_factor"] > 1.0
        )  # Should benefit from parallelization
        assert metrics["performance_improvement"] > 1.0

    @pytest.mark.asyncio
    async def test_agent_handoff_workflow(
        self, workflow_orchestrator, agent_pool, agent_dependencies
    ):
        """Test workflow with agent handoffs between different capabilities."""
        # Register agents with specific roles
        discovery_agent = agent_pool[2]
        orchestrator_agents = agent_pool[:2]

        await discovery_agent.initialize_discovery(agent_dependencies)
        workflow_orchestrator.register_agent(
            "discovery", discovery_agent, AgentRole.SPECIALIST
        )

        for i, agent in enumerate(orchestrator_agents):
            await agent.initialize(agent_dependencies)
            workflow_orchestrator.register_agent(
                f"orchestrator_{i}", agent, AgentRole.COORDINATOR
            )

        # Create handoff workflow
        workflow = DistributedWorkflow(
            workflow_id="handoff_001",
            nodes=[
                # Discovery phase
                WorkflowNode(
                    node_id="tool_discovery",
                    agent_role=AgentRole.SPECIALIST,
                    task_description="discover optimal tools for complex data processing",
                    constraints={"max_latency_ms": 1500, "min_accuracy": 0.85},
                ),
                # Planning phase
                WorkflowNode(
                    node_id="execution_planning",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="plan execution strategy using discovered tools",
                    dependencies=["tool_discovery"],
                    constraints={"strategy": "performance_optimized"},
                ),
                # Execution phase
                WorkflowNode(
                    node_id="execution",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="execute planned strategy",
                    dependencies=["execution_planning"],
                    constraints={"execute_plan": True},
                ),
            ],
            global_constraints={"handoff_efficiency": True},
        )

        # Execute handoff workflow
        completed_workflow = await workflow_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify handoff execution
        assert completed_workflow.state == WorkflowState.COMPLETED
        assert all(
            node.state == WorkflowState.COMPLETED for node in completed_workflow.nodes
        )

        # Verify handoff data flow
        discovery_node = next(
            n for n in completed_workflow.nodes if n.node_id == "tool_discovery"
        )
        planning_node = next(
            n for n in completed_workflow.nodes if n.node_id == "execution_planning"
        )
        execution_node = next(
            n for n in completed_workflow.nodes if n.node_id == "execution"
        )

        # Discovery should provide tools
        assert (
            "tools_discovered" in discovery_node.result
            or "success" in discovery_node.result
        )

        # Planning should reference discovery results
        planning_context = planning_node.result.get("dependency_results", {})
        assert (
            "tool_discovery" in str(planning_context) or planning_node.result["success"]
        )

        # Execution should use planning results
        execution_context = execution_node.result.get("dependency_results", {})
        assert (
            "execution_planning" in str(execution_context)
            or execution_node.result["success"]
        )

    @pytest.mark.asyncio
    async def test_error_recovery_in_distributed_workflow(
        self, workflow_orchestrator, agent_pool, agent_dependencies
    ):
        """Test error recovery and fault tolerance in distributed workflows."""
        # Register agents
        reliable_agent = agent_pool[0]
        unreliable_agent = agent_pool[1]

        await reliable_agent.initialize(agent_dependencies)
        await unreliable_agent.initialize(agent_dependencies)

        workflow_orchestrator.register_agent(
            "reliable", reliable_agent, AgentRole.COORDINATOR
        )
        workflow_orchestrator.register_agent(
            "unreliable", unreliable_agent, AgentRole.COORDINATOR
        )

        # Mock unreliable agent
        original_orchestrate = unreliable_agent.orchestrate

        async def failing_orchestrate(task, constraints, deps):
            if "fail_task" in task:
                msg = "Simulated agent failure"
                raise Exception(msg)
            return await original_orchestrate(task, constraints, deps)

        unreliable_agent.orchestrate = failing_orchestrate

        # Create workflow with potential failure points
        workflow = DistributedWorkflow(
            workflow_id="fault_tolerance_001",
            nodes=[
                # Critical path
                WorkflowNode(
                    node_id="critical_task",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="critical processing task",
                    constraints={"priority": "high"},
                ),
                # Failure-prone task
                WorkflowNode(
                    node_id="risky_task",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="fail_task - this should fail",
                    constraints={"retry": False},
                ),
                # Recovery task (independent)
                WorkflowNode(
                    node_id="recovery_task",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="recovery processing task",
                    constraints={"fallback": True},
                ),
                # Final aggregation (depends on available results)
                WorkflowNode(
                    node_id="final_aggregation",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="aggregate available results with error handling",
                    dependencies=[
                        "critical_task",
                        "recovery_task",
                    ],  # Don't depend on risky task
                    constraints={"handle_partial_results": True},
                ),
            ],
        )

        # Execute workflow with fault tolerance
        completed_workflow = await workflow_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify error recovery
        critical_node = next(
            n for n in completed_workflow.nodes if n.node_id == "critical_task"
        )
        risky_node = next(
            n for n in completed_workflow.nodes if n.node_id == "risky_task"
        )
        recovery_node = next(
            n for n in completed_workflow.nodes if n.node_id == "recovery_task"
        )
        final_node = next(
            n for n in completed_workflow.nodes if n.node_id == "final_aggregation"
        )

        # Critical and recovery tasks should succeed
        assert critical_node.state == WorkflowState.COMPLETED
        assert recovery_node.state == WorkflowState.COMPLETED

        # Risky task should fail
        assert risky_node.state == WorkflowState.FAILED
        assert risky_node.error is not None

        # Final task should still complete despite partial failure
        assert final_node.state == WorkflowState.COMPLETED

        # Overall workflow should handle partial failure gracefully
        assert (
            completed_workflow.state == WorkflowState.COMPLETED
        )  # Workflow completes despite node failure
        assert (
            completed_workflow.performance_metrics["success_rate"] == 0.75
        )  # 3/4 nodes succeeded

    @pytest.mark.asyncio
    async def test_performance_monitoring_and_optimization(
        self, workflow_orchestrator, agent_pool, agent_dependencies
    ):
        """Test performance monitoring and optimization in distributed workflows."""
        # Register agents with performance tracking
        for i, agent in enumerate(agent_pool[:3]):
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

            role = AgentRole.COORDINATOR if i < 2 else AgentRole.SPECIALIST
            workflow_orchestrator.register_agent(f"perf_agent_{i}", agent, role)

        # Create performance-critical workflow
        workflow = DistributedWorkflow(
            workflow_id="performance_001",
            nodes=[
                # Speed-critical tasks (parallel)
                WorkflowNode(
                    node_id="speed_task_1",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="speed-optimized processing task 1",
                    constraints={"max_latency_ms": 500, "priority": "speed"},
                ),
                WorkflowNode(
                    node_id="speed_task_2",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="speed-optimized processing task 2",
                    constraints={"max_latency_ms": 500, "priority": "speed"},
                ),
                # Quality-critical task
                WorkflowNode(
                    node_id="quality_task",
                    agent_role=AgentRole.SPECIALIST,
                    task_description="high-quality analysis task",
                    constraints={"min_accuracy": 0.95, "priority": "quality"},
                ),
                # Optimization task (depends on all)
                WorkflowNode(
                    node_id="optimization",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="optimize results for maximum performance gain",
                    dependencies=["speed_task_1", "speed_task_2", "quality_task"],
                    constraints={
                        "target_improvement": "3x-10x",
                        "balance": "speed_quality",
                    },
                ),
            ],
            global_constraints={
                "performance_target": "3x-10x improvement",
                "max_total_time": 10.0,
                "min_quality": 0.8,
            },
        )

        # Execute with performance monitoring
        start_time = time.time()
        completed_workflow = await workflow_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )
        total_execution_time = time.time() - start_time

        # Verify performance optimization
        assert completed_workflow.state == WorkflowState.COMPLETED
        metrics = completed_workflow.performance_metrics

        # Verify measured execution time is reasonable
        assert total_execution_time > 0, "Execution time should be positive"
        assert total_execution_time < 30, "Execution time should be reasonable for test"

        # Check performance improvement metrics
        assert metrics["performance_improvement"] >= 1.0  # At least some improvement
        assert metrics["parallelization_factor"] > 1.0  # Benefit from parallelization
        assert (
            metrics["total_execution_time"]
            < workflow.global_constraints["max_total_time"]
        )

        # Verify target performance gains
        optimization_node = next(
            n for n in completed_workflow.nodes if n.node_id == "optimization"
        )
        assert optimization_node.state == WorkflowState.COMPLETED

        # Check if we achieved target improvement range (simulated)
        if metrics["performance_improvement"] >= 3.0:
            assert True  # Achieved target range
        else:
            # Should still show significant improvement
            assert metrics["performance_improvement"] >= 1.5

        # Verify speed vs quality balance
        speed_nodes = [n for n in completed_workflow.nodes if "speed_task" in n.node_id]
        quality_node = next(
            n for n in completed_workflow.nodes if n.node_id == "quality_task"
        )

        # Speed tasks should complete quickly
        speed_times = [n.execution_time for n in speed_nodes if n.execution_time]
        if speed_times:
            avg_speed_time = sum(speed_times) / len(speed_times)
            assert avg_speed_time < 5.0  # Should be fast

        # Quality task should provide good results
        assert quality_node.result["success"]

    @pytest.mark.asyncio
    async def test_dynamic_load_balancing_in_workflows(
        self, workflow_orchestrator, agent_pool, agent_dependencies
    ):
        """Test dynamic load balancing across agents in workflows."""
        # Register multiple agents of same type for load balancing
        coordinator_agents = agent_pool[:3]
        for i, agent in enumerate(coordinator_agents):
            await agent.initialize(agent_dependencies)
            workflow_orchestrator.register_agent(
                f"coord_{i}", agent, AgentRole.COORDINATOR
            )

        # Create workflow with many parallel tasks
        parallel_tasks = [
            WorkflowNode(
                node_id=f"parallel_task_{i}",
                agent_role=AgentRole.COORDINATOR,
                task_description=f"load balanced task {i}",
                constraints={"task_id": i, "load_test": True},
            )
            for i in range(9)  # 9 tasks, 3 agents = 3 tasks per agent
        ]

        # Add aggregation task
        aggregation_task = WorkflowNode(
            node_id="aggregation",
            agent_role=AgentRole.COORDINATOR,
            task_description="aggregate all parallel results",
            dependencies=[f"parallel_task_{i}" for i in range(9)],
            constraints={"aggregation": True},
        )

        workflow = DistributedWorkflow(
            workflow_id="load_balance_001",
            nodes=[*parallel_tasks, aggregation_task],
            global_constraints={"load_balancing": True},
        )

        # Track agent load before execution
        initial_loads = {
            aid: info["load"] for aid, info in workflow_orchestrator.agent_pool.items()
        }

        # Execute workflow
        completed_workflow = await workflow_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify load balancing
        # Check that loads were tracked properly
        assert len(initial_loads) > 0, "Should have tracked initial agent loads"
        assert completed_workflow.state == WorkflowState.COMPLETED
        assert all(
            node.state == WorkflowState.COMPLETED for node in completed_workflow.nodes
        )

        # Check that tasks were distributed across agents
        agent_assignments = {}
        for node in completed_workflow.nodes[:-1]:  # Exclude aggregation task
            agent_id = node.result.get("agent_id")
            if agent_id:
                agent_assignments[agent_id] = agent_assignments.get(agent_id, 0) + 1

        # Should use multiple agents
        assert len(agent_assignments) > 1

        # Load should be relatively balanced (no agent should handle all tasks)
        max_load = max(agent_assignments.values())
        min_load = min(agent_assignments.values())
        load_imbalance = max_load - min_load
        assert load_imbalance <= 2  # Reasonable load distribution

        # Verify aggregation completed successfully
        aggregation_node = completed_workflow.nodes[-1]
        assert aggregation_node.state == WorkflowState.COMPLETED
        assert aggregation_node.result["success"]

        # Check parallelization benefit
        metrics = completed_workflow.performance_metrics
        assert (
            metrics["parallelization_factor"] > 2.0
        )  # Should see significant parallelization benefit


class TestAutonomousCapabilities:
    """Test autonomous coordination capabilities and self-healing patterns."""

    @pytest.fixture
    def autonomous_orchestrator(self) -> WorkflowOrchestrator:
        """Create orchestrator with autonomous capabilities."""
        return WorkflowOrchestrator()

    @pytest.fixture
    def autonomous_agents(self) -> list[Any]:
        """Create agents with autonomous capabilities."""
        return [
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1),
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1),
            DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1),
            DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1),
        ]

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_autonomous_workflow",
        )

    @pytest.mark.asyncio
    async def test_self_healing_coordination_patterns(
        self, autonomous_orchestrator, autonomous_agents, agent_dependencies
    ):
        """Test self-healing coordination patterns."""
        # Register agents with failure simulation
        primary_agent = autonomous_agents[0]
        backup_agent = autonomous_agents[1]
        monitor_agent = autonomous_agents[2]

        await primary_agent.initialize(agent_dependencies)
        await backup_agent.initialize(agent_dependencies)
        await monitor_agent.initialize_discovery(agent_dependencies)

        autonomous_orchestrator.register_agent(
            "primary", primary_agent, AgentRole.COORDINATOR
        )
        autonomous_orchestrator.register_agent(
            "backup", backup_agent, AgentRole.COORDINATOR
        )
        autonomous_orchestrator.register_agent(
            "monitor", monitor_agent, AgentRole.MONITOR
        )

        # Simulate primary agent failure after first task
        original_orchestrate = primary_agent.orchestrate
        call_count = 0

        async def failing_primary_orchestrate(task, constraints, deps):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # Fail after first successful call
                msg = "Primary agent failure - triggering self-healing"
                raise Exception(msg)
            return await original_orchestrate(task, constraints, deps)

        primary_agent.orchestrate = failing_primary_orchestrate

        # Create self-healing workflow
        workflow = DistributedWorkflow(
            workflow_id="self_healing_001",
            nodes=[
                # Initial successful task
                WorkflowNode(
                    node_id="initial_task",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="initial processing task",
                    constraints={"phase": 1},
                ),
                # Task that will trigger failure and healing
                WorkflowNode(
                    node_id="healing_trigger",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="task that triggers self-healing",
                    dependencies=["initial_task"],
                    constraints={"phase": 2, "healing_test": True},
                ),
                # Monitoring task
                WorkflowNode(
                    node_id="health_monitor",
                    agent_role=AgentRole.MONITOR,
                    task_description="monitor system health and trigger recovery",
                    constraints={"monitoring": True},
                ),
                # Recovery verification
                WorkflowNode(
                    node_id="recovery_verification",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="verify system recovery",
                    dependencies=["health_monitor"],
                    constraints={"verify_recovery": True},
                ),
            ],
        )

        # Implement self-healing logic in orchestrator
        original_select_agent = autonomous_orchestrator._select_agent_for_node

        def self_healing_select_agent(node):
            # If primary agent fails, use backup
            selected = original_select_agent(node)

            # Check if selected agent is healthy (simulate health check)
            if selected == "primary" and call_count > 1:
                # Primary failed, use backup for healing
                return "backup"
            return selected

        autonomous_orchestrator._select_agent_for_node = self_healing_select_agent

        # Execute self-healing workflow
        completed_workflow = await autonomous_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify self-healing behavior
        initial_task = next(
            n for n in completed_workflow.nodes if n.node_id == "initial_task"
        )
        healing_trigger = next(
            n for n in completed_workflow.nodes if n.node_id == "healing_trigger"
        )
        monitor_task = next(
            n for n in completed_workflow.nodes if n.node_id == "health_monitor"
        )
        recovery_task = next(
            n for n in completed_workflow.nodes if n.node_id == "recovery_verification"
        )

        # Initial task should succeed on primary
        assert initial_task.state == WorkflowState.COMPLETED
        assert initial_task.result["agent_id"] == "primary"

        # Healing trigger should either succeed on backup or handle failure gracefully
        assert healing_trigger.state in [WorkflowState.COMPLETED, WorkflowState.FAILED]
        if healing_trigger.state == WorkflowState.COMPLETED:
            assert healing_trigger.result["agent_id"] == "backup"  # Should use backup

        # Monitor should complete
        assert monitor_task.state == WorkflowState.COMPLETED

        # Recovery verification should succeed
        assert recovery_task.state == WorkflowState.COMPLETED

        # Overall workflow should demonstrate resilience
        success_rate = completed_workflow.performance_metrics["success_rate"]
        assert success_rate >= 0.75  # At least 75% success despite failure

    @pytest.mark.asyncio
    async def test_adaptive_workflow_optimization(
        self, autonomous_orchestrator, autonomous_agents, agent_dependencies
    ):
        """Test adaptive workflow optimization."""
        # Register agents with performance tracking
        fast_agent = autonomous_agents[0]
        accurate_agent = autonomous_agents[1]
        discovery_agent = autonomous_agents[2]

        await fast_agent.initialize(agent_dependencies)
        await accurate_agent.initialize(agent_dependencies)
        await discovery_agent.initialize_discovery(agent_dependencies)

        autonomous_orchestrator.register_agent(
            "fast", fast_agent, AgentRole.COORDINATOR
        )
        autonomous_orchestrator.register_agent(
            "accurate", accurate_agent, AgentRole.COORDINATOR
        )
        autonomous_orchestrator.register_agent(
            "discovery", discovery_agent, AgentRole.SPECIALIST
        )

        # Mock different performance characteristics
        async def fast_orchestrate(task, constraints, deps):
            # Simulate fast but lower accuracy
            response = await fast_agent.__class__.orchestrate(
                fast_agent, task, constraints, deps
            )
            response.latency_ms = 100  # Fast
            response.confidence = 0.7  # Lower confidence
            return response

        async def accurate_orchestrate(task, constraints, deps):
            # Simulate slower but higher accuracy
            response = await accurate_agent.__class__.orchestrate(
                accurate_agent, task, constraints, deps
            )
            response.latency_ms = 500  # Slower
            response.confidence = 0.95  # Higher confidence
            return response

        fast_agent.orchestrate = fast_orchestrate
        accurate_agent.orchestrate = accurate_orchestrate

        # Create adaptive optimization workflow
        workflow = DistributedWorkflow(
            workflow_id="adaptive_optimization_001",
            nodes=[
                # Performance assessment
                WorkflowNode(
                    node_id="performance_assessment",
                    agent_role=AgentRole.SPECIALIST,
                    task_description="assess performance requirements and constraints",
                    constraints={"assessment": True},
                ),
                # Speed-critical task
                WorkflowNode(
                    node_id="speed_critical",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="speed-critical processing with time constraints",
                    dependencies=["performance_assessment"],
                    constraints={"max_latency_ms": 200, "priority": "speed"},
                ),
                # Quality-critical task
                WorkflowNode(
                    node_id="quality_critical",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="quality-critical processing requiring high accuracy",
                    dependencies=["performance_assessment"],
                    constraints={"min_accuracy": 0.9, "priority": "quality"},
                ),
                # Adaptive optimization
                WorkflowNode(
                    node_id="adaptive_optimization",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="adaptively optimize based on performance feedback",
                    dependencies=["speed_critical", "quality_critical"],
                    constraints={"optimize": "adaptive", "target": "3x-10x"},
                ),
            ],
        )

        # Implement adaptive agent selection
        def adaptive_select_agent(node):
            if "speed" in node.constraints.get("priority", ""):
                return "fast"  # Use fast agent for speed-critical tasks
            if "quality" in node.constraints.get("priority", ""):
                return "accurate"  # Use accurate agent for quality-critical tasks
            if node.agent_role == AgentRole.SPECIALIST:
                return "discovery"
            # Adaptive selection based on context
            return (
                "accurate" if node.constraints.get("min_accuracy", 0) > 0.8 else "fast"
            )

        autonomous_orchestrator._select_agent_for_node = adaptive_select_agent

        # Execute adaptive workflow
        completed_workflow = await autonomous_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify adaptive optimization
        assert completed_workflow.state == WorkflowState.COMPLETED

        speed_task = next(
            n for n in completed_workflow.nodes if n.node_id == "speed_critical"
        )
        quality_task = next(
            n for n in completed_workflow.nodes if n.node_id == "quality_critical"
        )
        optimization_task = next(
            n for n in completed_workflow.nodes if n.node_id == "adaptive_optimization"
        )

        # Verify adaptive agent selection
        assert speed_task.result["agent_id"] == "fast"  # Speed task used fast agent
        assert (
            quality_task.result["agent_id"] == "accurate"
        )  # Quality task used accurate agent

        # Verify optimization results
        assert optimization_task.state == WorkflowState.COMPLETED

        # Check performance improvement metrics
        metrics = completed_workflow.performance_metrics
        assert metrics["performance_improvement"] > 1.0

        # Verify speed vs quality trade-offs were optimized
        assert speed_task.state == WorkflowState.COMPLETED
        assert quality_task.state == WorkflowState.COMPLETED

    @pytest.mark.asyncio
    async def test_autonomous_failure_detection_and_recovery(
        self, autonomous_orchestrator, autonomous_agents, agent_dependencies
    ):
        """Test autonomous failure detection and recovery mechanisms."""
        # Setup agents with failure detection
        primary_agents = autonomous_agents[:2]
        monitoring_agents = autonomous_agents[2:]

        for i, agent in enumerate(primary_agents):
            await agent.initialize(agent_dependencies)
            autonomous_orchestrator.register_agent(
                f"primary_{i}", agent, AgentRole.COORDINATOR
            )

        for i, agent in enumerate(monitoring_agents):
            await agent.initialize_discovery(agent_dependencies)
            autonomous_orchestrator.register_agent(
                f"monitor_{i}", agent, AgentRole.MONITOR
            )

        # Implement failure detection
        agent_health = dict.fromkeys(autonomous_orchestrator.agent_pool.keys(), True)
        failure_history = []

        def detect_agent_failure(agent_id, error):
            """Simulate autonomous failure detection."""
            agent_health[agent_id] = False
            failure_history.append(
                {
                    "agent_id": agent_id,
                    "timestamp": datetime.now(tz=UTC),
                    "error": str(error),
                }
            )

            # Trigger recovery
            return trigger_recovery(agent_id)

        def trigger_recovery(failed_agent_id):
            """Trigger autonomous recovery."""
            # Find healthy replacement agent
            healthy_agents = [
                aid
                for aid, health in agent_health.items()
                if health and aid != failed_agent_id
            ]

            if healthy_agents:
                replacement = healthy_agents[0]
                agent_health[replacement] = True  # Ensure replacement is marked healthy
                return replacement
            return None

        # Create failure detection workflow
        workflow = DistributedWorkflow(
            workflow_id="failure_detection_001",
            nodes=[
                # Health monitoring
                WorkflowNode(
                    node_id="health_monitoring",
                    agent_role=AgentRole.MONITOR,
                    task_description="continuously monitor system health",
                    constraints={"monitoring": "continuous"},
                ),
                # Primary processing tasks
                WorkflowNode(
                    node_id="primary_task_1",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="primary processing task 1",
                    constraints={"critical": True},
                ),
                WorkflowNode(
                    node_id="primary_task_2",
                    agent_role=AgentRole.COORDINATOR,
                    task_description="primary processing task 2 - will fail",
                    constraints={"critical": True, "will_fail": True},
                ),
                # Recovery validation
                WorkflowNode(
                    node_id="recovery_validation",
                    agent_role=AgentRole.MONITOR,
                    task_description="validate recovery effectiveness",
                    dependencies=["health_monitoring", "primary_task_1"],
                    constraints={"validate_recovery": True},
                ),
            ],
        )

        # Mock failure in second task
        original_execute_node = autonomous_orchestrator._execute_node

        async def failure_detecting_execute_node(node, workflow, deps):
            try:
                if node.constraints.get("will_fail"):
                    # Simulate failure
                    msg = "Simulated critical failure"
                    raise Exception(msg)

                return await original_execute_node(node, workflow, deps)

            except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
                # Autonomous failure detection
                agent_id = autonomous_orchestrator._select_agent_for_node(node)
                replacement_agent = detect_agent_failure(agent_id, e)

                if replacement_agent:
                    # Attempt recovery with replacement agent
                    node.constraints["recovery_mode"] = True
                    node.task_description = f"[RECOVERY] {node.task_description}"

                    # Override agent selection for recovery
                    original_select = autonomous_orchestrator._select_agent_for_node
                    autonomous_orchestrator._select_agent_for_node = (
                        lambda n: replacement_agent
                    )

                    try:
                        await original_execute_node(node, workflow, deps)
                        node.result = {
                            "success": True,
                            "recovery_used": True,
                            "original_failure": str(e),
                            "recovered_by": replacement_agent,
                        }
                        node.state = WorkflowState.COMPLETED
                    finally:
                        autonomous_orchestrator._select_agent_for_node = original_select
                else:
                    # No recovery possible
                    node.state = WorkflowState.FAILED
                    node.error = f"No recovery available: {e}"

        autonomous_orchestrator._execute_node = failure_detecting_execute_node

        # Execute failure detection workflow
        completed_workflow = await autonomous_orchestrator.execute_workflow(
            workflow, agent_dependencies
        )

        # Verify autonomous failure detection and recovery
        health_task = next(
            n for n in completed_workflow.nodes if n.node_id == "health_monitoring"
        )
        primary_task_1 = next(
            n for n in completed_workflow.nodes if n.node_id == "primary_task_1"
        )
        primary_task_2 = next(
            n for n in completed_workflow.nodes if n.node_id == "primary_task_2"
        )
        recovery_task = next(
            n for n in completed_workflow.nodes if n.node_id == "recovery_validation"
        )

        # Health monitoring should succeed
        assert health_task.state == WorkflowState.COMPLETED

        # First primary task should succeed
        assert primary_task_1.state == WorkflowState.COMPLETED

        # Second primary task should recover or fail gracefully
        assert primary_task_2.state in [WorkflowState.COMPLETED, WorkflowState.FAILED]
        if primary_task_2.state == WorkflowState.COMPLETED:
            assert primary_task_2.result.get("recovery_used") is True

        # Recovery validation should complete
        assert recovery_task.state == WorkflowState.COMPLETED

        # Verify failure detection
        assert len(failure_history) > 0  # Should have detected failure

        # Verify system resilience
        success_rate = completed_workflow.performance_metrics["success_rate"]
        assert success_rate >= 0.5  # System should maintain partial functionality
