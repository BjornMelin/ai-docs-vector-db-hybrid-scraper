"""Advanced tool orchestration system for agentic workflows.

This module implements sophisticated tool composition and orchestration patterns
based on research findings, enabling agents to
compose and coordinate complex tool workflows.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager

# BaseAgent and BaseAgentDependencies imports removed (unused)
from src.services.cache.patterns import CircuitBreakerPattern


logger = logging.getLogger(__name__)


def _raise_circuit_breaker_open(tool_id: str) -> None:
    """Raise RuntimeError for circuit breaker open."""
    msg = f"Circuit breaker open for tool {tool_id}"
    raise RuntimeError(msg)


class ToolExecutionMode(str, Enum):
    """Tool execution modes for different coordination patterns."""

    SEQUENTIAL = "sequential"  # Execute tools one after another
    PARALLEL = "parallel"  # Execute tools simultaneously
    CONDITIONAL = "conditional"  # Execute based on conditions
    PIPELINE = "pipeline"  # Pipeline execution with data flow
    ADAPTIVE = "adaptive"  # Adaptive execution based on results


class ToolPriority(str, Enum):
    """Tool execution priority levels."""

    CRITICAL = "critical"  # Must execute immediately
    HIGH = "high"  # High priority
    NORMAL = "normal"  # Standard priority
    LOW = "low"  # Background execution
    OPTIONAL = "optional"  # Execute if resources available


class ToolCapability(str, Enum):
    """Tool capability categories for selection."""

    SEARCH = "search"  # Search and retrieval
    ANALYSIS = "analysis"  # Data analysis and processing
    GENERATION = "generation"  # Content generation
    VALIDATION = "validation"  # Quality and validation checks
    OPTIMIZATION = "optimization"  # Performance optimization
    MONITORING = "monitoring"  # System monitoring
    COORDINATION = "coordination"  # Multi-agent coordination


@dataclass
class ToolDefinition:
    """Definition of a composable tool."""

    tool_id: str
    name: str
    description: str
    capabilities: set[ToolCapability]
    priority: ToolPriority

    # Execution characteristics
    estimated_duration_ms: float
    resource_requirements: dict[str, float]  # CPU, memory, etc.
    dependencies: list[str]  # Tool IDs this depends on

    # Quality and reliability
    success_rate: float = 0.95
    fallback_tools: list[str] = None
    timeout_ms: float | None = None

    # Function reference
    executor: Callable | None = None

    def __post_init__(self):
        if self.fallback_tools is None:
            self.fallback_tools = []


class ToolChainNode(BaseModel):
    """Node in a tool execution chain."""

    node_id: str = Field(..., description="Unique node identifier")
    tool_id: str = Field(..., description="Tool to execute")
    execution_mode: ToolExecutionMode = Field(..., description="Execution mode")

    # Dependencies and conditions
    depends_on: list[str] = Field(default_factory=list, description="Node dependencies")
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Execution conditions"
    )

    # Data flow
    input_mapping: dict[str, str] = Field(
        default_factory=dict, description="Input data mapping"
    )
    output_mapping: dict[str, str] = Field(
        default_factory=dict, description="Output data mapping"
    )

    # Execution state
    status: str = Field("pending", description="Execution status")
    start_time: datetime | None = Field(None, description="Execution start time")
    end_time: datetime | None = Field(None, description="Execution end time")
    result: dict[str, Any] | None = Field(None, description="Execution result")
    error: str | None = Field(None, description="Error message if failed")


class ToolOrchestrationPlan(BaseModel):
    """Plan for orchestrating multiple tools."""

    plan_id: str = Field(..., description="Unique plan identifier")
    goal: str = Field(..., description="High-level goal of the orchestration")
    nodes: list[ToolChainNode] = Field(..., description="Tool chain nodes")

    # Execution constraints
    max_parallel_tools: int = Field(5, description="Maximum parallel tool execution")
    timeout_seconds: float = Field(60.0, description="Overall timeout")
    failure_threshold: float = Field(0.2, description="Acceptable failure rate")

    # Quality requirements
    min_quality_score: float = Field(0.8, description="Minimum quality threshold")
    max_cost: float | None = Field(None, description="Maximum cost constraint")

    # Optimization preferences
    optimize_for: str = Field(
        "balanced", description="Optimization target: speed, quality, cost, balanced"
    )


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""

    execution_id: str = Field(..., description="Unique execution identifier")
    tool_id: str = Field(..., description="Tool that was executed")
    success: bool = Field(..., description="Whether execution succeeded")

    # Timing information
    start_time: datetime = Field(..., description="Execution start time")
    end_time: datetime = Field(..., description="Execution end time")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")

    # Results and quality
    result: dict[str, Any] | None = Field(None, description="Execution result")
    quality_score: float | None = Field(None, description="Quality score")
    confidence: float | None = Field(None, description="Confidence in result")

    # Error handling
    error: str | None = Field(None, description="Error message if failed")
    fallback_used: bool = Field(False, description="Whether fallback was used")
    retry_count: int = Field(0, description="Number of retries")

    # Resource usage
    resource_usage: dict[str, float] = Field(
        default_factory=dict, description="Resource consumption"
    )


class AdvancedToolOrchestrator:
    """Orchestrator for tool composition and execution.

    Implements orchestration patterns including:
    - Tool selection based on capabilities and performance
    - Dynamic workflow composition based on goals and constraints
    - Parallel and pipeline execution with dependency management
    - Execution with fallback and recovery mechanisms
    - Performance optimization and resource management
    """

    def __init__(
        self,
        client_manager: ClientManager,
        max_parallel_executions: int = 10,
        default_timeout_seconds: float = 30.0,
        enable_circuit_breakers: bool = True,
    ):
        """Initialize the tool orchestrator.

        Args:
            client_manager: Client manager for resource access
            max_parallel_executions: Maximum parallel tool executions
            default_timeout_seconds: Default timeout for tool execution
            enable_circuit_breakers: Enable circuit breakers for fault tolerance

        """
        self.client_manager = client_manager
        self.max_parallel_executions = max_parallel_executions
        self.default_timeout_seconds = default_timeout_seconds
        self.enable_circuit_breakers = enable_circuit_breakers

        # Tool registry and management
        self.registered_tools: dict[str, ToolDefinition] = {}
        self.tool_performance_history: dict[str, list[ToolExecutionResult]] = {}
        self.circuit_breakers: dict[str, CircuitBreakerPattern] = {}

        # Execution state
        self.active_executions: dict[str, asyncio.Task] = {}
        self.execution_history: list[ToolExecutionResult] = []
        self.orchestration_plans: dict[str, ToolOrchestrationPlan] = {}

        # Performance optimization
        self.tool_success_rates: dict[str, float] = {}
        self.tool_avg_durations: dict[str, float] = {}
        self.capability_performance: dict[ToolCapability, list[str]] = {}

        logger.info("ToolOrchestrator initialized")

    async def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool for orchestrated execution.

        Args:
            tool_def: Tool definition to register

        """
        self.registered_tools[tool_def.tool_id] = tool_def
        self.tool_performance_history[tool_def.tool_id] = []
        self.tool_success_rates[tool_def.tool_id] = tool_def.success_rate
        self.tool_avg_durations[tool_def.tool_id] = tool_def.estimated_duration_ms

        # Initialize circuit breaker
        if self.enable_circuit_breakers:
            self.circuit_breakers[tool_def.tool_id] = CircuitBreakerPattern(
                failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
            )

        # Update capability mapping
        for capability in tool_def.capabilities:
            if capability not in self.capability_performance:
                self.capability_performance[capability] = []
            self.capability_performance[capability].append(tool_def.tool_id)

        logger.info(
            f"Registered tool {tool_def.tool_id} with capabilities: "
            f"{tool_def.capabilities}"
        )

    async def compose_tool_chain(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> ToolOrchestrationPlan:
        """Compose an optimal tool chain for achieving a goal.

        Args:
            goal: High-level goal description
            constraints: Execution constraints (time, cost, quality)
            preferences: Optimization preferences

        Returns:
            Orchestration plan for achieving the goal

        """
        plan_id = str(uuid4())
        constraints = constraints or {}
        preferences = preferences or {}

        logger.info(f"Composing tool chain {plan_id} for goal: {goal}")

        try:
            # Analyze goal to determine required capabilities
            required_capabilities = await self._analyze_goal_capabilities(
                goal, constraints
            )

            # Select optimal tools for each capability
            selected_tools = await self._select_optimal_tools(
                required_capabilities, constraints, preferences
            )

            # Create execution nodes
            nodes = await self._create_execution_nodes(selected_tools, constraints)

            # Optimize execution order and dependencies
            optimized_nodes = await self._optimize_execution_plan(nodes, preferences)

            # Create orchestration plan
            plan = ToolOrchestrationPlan(
                plan_id=plan_id,
                goal=goal,
                nodes=optimized_nodes,
                max_parallel_tools=constraints.get("max_parallel_tools", 5),
                timeout_seconds=constraints.get("timeout_seconds", 60.0),
                failure_threshold=constraints.get("failure_threshold", 0.2),
                min_quality_score=constraints.get("min_quality_score", 0.8),
                max_cost=constraints.get("max_cost"),
                optimize_for=preferences.get("optimize_for", "balanced"),
            )

            self.orchestration_plans[plan_id] = plan

            logger.info(f"Composed tool chain {plan_id} with {len(nodes)} tools")

        except Exception:  # noqa: BLE001
            logger.exception("Failed to compose tool chain")
            raise

        return plan

    async def execute_tool_chain(
        self,
        plan: ToolOrchestrationPlan,
        input_data: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute a tool orchestration plan.

        Args:
            plan: Orchestration plan to execute
            input_data: Input data for the tool chain
            timeout_seconds: Execution timeout

        Returns:
            Execution results with metadata

        """
        execution_id = str(uuid4())
        timeout_seconds = timeout_seconds or plan.timeout_seconds
        start_time = time.time()

        logger.info(
            f"Executing tool chain {plan.plan_id} with execution ID {execution_id}"
        )

        try:
            # Initialize execution state
            execution_state = {
                "input_data": input_data,
                "intermediate_results": {},
                "completed_nodes": set(),
                "failed_nodes": set(),
                "execution_metadata": {
                    "execution_id": execution_id,
                    "plan_id": plan.plan_id,
                    "start_time": datetime.now(tz=datetime.timezone.utc),
                    "goal": plan.goal,
                },
            }

            # Execute nodes based on orchestration plan
            if plan.optimize_for == "speed":
                results = await self._execute_speed_optimized(
                    plan, execution_state, timeout_seconds
                )
            elif plan.optimize_for == "quality":
                results = await self._execute_quality_optimized(
                    plan, execution_state, timeout_seconds
                )
            elif plan.optimize_for == "cost":
                results = await self._execute_cost_optimized(
                    plan, execution_state, timeout_seconds
                )
            else:  # balanced
                results = await self._execute_balanced(
                    plan, execution_state, timeout_seconds
                )

            execution_time = time.time() - start_time

            # Calculate execution metrics
            successful_nodes = len(execution_state["completed_nodes"])
            total_nodes = len(plan.nodes)
            success_rate = successful_nodes / total_nodes if total_nodes > 0 else 0.0

            # Update tool performance history
            await self._update_performance_history(plan, execution_state)

            final_results = {
                "execution_id": execution_id,
                "plan_id": plan.plan_id,
                "success": success_rate >= (1.0 - plan.failure_threshold),
                "execution_time_seconds": execution_time,
                "results": results,
                "metadata": {
                    "total_nodes": total_nodes,
                    "successful_nodes": successful_nodes,
                    "success_rate": success_rate,
                    "total_execution_time_ms": execution_time * 1000,
                    "goal_achieved": success_rate >= plan.min_quality_score,
                    "optimization_target": plan.optimize_for,
                },
            }

            logger.info(
                f"Tool chain execution {execution_id} completed with "
                f"{success_rate * 100:.2f}%% success rate"
            )

        except Exception as e:  # noqa: BLE001
            execution_time = time.time() - start_time

            logger.exception(f"Tool chain execution {execution_id} failed")

            return {
                "execution_id": execution_id,
                "plan_id": plan.plan_id,
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "metadata": {"goal": plan.goal, "failure_point": "orchestration"},
            }

        return final_results

    async def execute_single_tool(
        self,
        tool_id: str,
        input_data: dict[str, Any],
        timeout_seconds: float | None = None,
        fallback_enabled: bool = True,
    ) -> ToolExecutionResult:
        """Execute a single tool with error handling.

        Args:
            tool_id: Tool identifier
            input_data: Input data for tool execution
            timeout_seconds: Execution timeout
            fallback_enabled: Enable fallback tools on failure

        Returns:
            Tool execution result

        """
        execution_id = str(uuid4())
        start_time = datetime.now(tz=datetime.timezone.utc)

        if tool_id not in self.registered_tools:
            msg = f"Tool {tool_id} not registered"
            raise ValueError(msg)

        tool_def = self.registered_tools[tool_id]
        timeout_seconds = timeout_seconds or (tool_def.timeout_ms or 30000) / 1000.0

        logger.debug(f"Executing tool {tool_id} with execution ID {execution_id}")

        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(tool_id)
            if circuit_breaker and circuit_breaker.is_open():
                if fallback_enabled and tool_def.fallback_tools:
                    return await self._execute_fallback_tool(
                        tool_def.fallback_tools[0],
                        input_data,
                        timeout_seconds,
                        execution_id,
                    )
                _raise_circuit_breaker_open(tool_id)

            # Execute tool with timeout
            if tool_def.executor:
                result = await asyncio.wait_for(
                    tool_def.executor(input_data), timeout=timeout_seconds
                )
            else:
                result = await self._mock_tool_execution(tool_def, input_data)

            end_time = datetime.now(tz=datetime.timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Track success in circuit breaker
            if circuit_breaker:
                await circuit_breaker.call(lambda: True)

            execution_result = ToolExecutionResult(
                execution_id=execution_id,
                tool_id=tool_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                result=result,
                quality_score=result.get("quality_score", 0.9),
                confidence=result.get("confidence", 0.85),
            )

            # Update performance metrics
            await self._update_tool_metrics(tool_id, execution_result)
        except TimeoutError:
            end_time = datetime.now(tz=datetime.timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.warning(f"Tool {tool_id} execution timed out")

            # Try fallback if available
            if fallback_enabled and tool_def.fallback_tools:
                return await self._execute_fallback_tool(
                    tool_def.fallback_tools[0],
                    input_data,
                    timeout_seconds,
                    execution_id,
                )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_id=tool_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error="Execution timeout",
            )

        except Exception as e:
            end_time = datetime.now(tz=datetime.timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Track failure in circuit breaker
            if circuit_breaker:
                try:
                    # Create a lambda that captures the exception to test
                    # circuit breaker
                    exception_to_throw = e
                    await circuit_breaker.call(
                        lambda: (_ for _ in ()).throw(exception_to_throw)
                    )
                except (asyncio.CancelledError, TimeoutError, RuntimeError):  # nosec # Expected to fail for circuit breaker testing
                    pass

            logger.exception(f"Tool {tool_id} execution failed")

            # Try fallback if available
            if fallback_enabled and tool_def.fallback_tools:
                return await self._execute_fallback_tool(
                    tool_def.fallback_tools[0],
                    input_data,
                    timeout_seconds,
                    execution_id,
                )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_id=tool_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def get_orchestration_status(self) -> dict[str, Any]:
        """Get current orchestration system status.

        Returns:
            Comprehensive orchestration status

        """
        total_tools = len(self.registered_tools)
        healthy_tools = sum(
            1
            for tool_id in self.registered_tools
            if not (
                self.circuit_breakers.get(tool_id, CircuitBreakerPattern()).is_open()
            )
        )

        # Calculate aggregate metrics
        recent_executions = [
            result
            for result in self.execution_history
            if result.start_time > datetime.now(tz=UTC) - timedelta(hours=1)
        ]

        avg_success_rate = (
            sum(1 for result in recent_executions if result.success)
            / len(recent_executions)
            if recent_executions
            else 0.0
        )

        avg_duration = (
            sum(result.duration_ms for result in recent_executions)
            / len(recent_executions)
            if recent_executions
            else 0.0
        )

        return {
            "registered_tools": total_tools,
            "healthy_tools": healthy_tools,
            "active_executions": len(self.active_executions),
            "orchestration_plans": len(self.orchestration_plans),
            "recent_executions_1h": len(recent_executions),
            "avg_success_rate_1h": avg_success_rate,
            "avg_duration_ms_1h": avg_duration,
            "tool_capabilities": {
                capability.value: len(tools)
                for capability, tools in self.capability_performance.items()
            },
            "circuit_breaker_status": {
                tool_id: "open" if cb.is_open() else "closed"
                for tool_id, cb in self.circuit_breakers.items()
            },
        }

    # Private helper methods

    async def _analyze_goal_capabilities(
        self, goal: str, _constraints: dict[str, Any]
    ) -> list[ToolCapability]:
        """Analyze goal to determine required tool capabilities."""
        # This would implement NLP analysis of the goal
        # For now, return a heuristic-based analysis

        goal_lower = goal.lower()
        required_capabilities = []

        if any(
            keyword in goal_lower
            for keyword in ["search", "find", "retrieve", "lookup"]
        ):
            required_capabilities.append(ToolCapability.SEARCH)

        if any(
            keyword in goal_lower
            for keyword in ["analyze", "process", "examine", "evaluate"]
        ):
            required_capabilities.append(ToolCapability.ANALYSIS)

        if any(
            keyword in goal_lower
            for keyword in ["generate", "create", "build", "compose"]
        ):
            required_capabilities.append(ToolCapability.GENERATION)

        if any(
            keyword in goal_lower
            for keyword in ["validate", "verify", "check", "confirm"]
        ):
            required_capabilities.append(ToolCapability.VALIDATION)

        if any(
            keyword in goal_lower
            for keyword in ["optimize", "improve", "enhance", "tune"]
        ):
            required_capabilities.append(ToolCapability.OPTIMIZATION)

        # Default to search if no specific capabilities detected
        if not required_capabilities:
            required_capabilities.append(ToolCapability.SEARCH)

        return required_capabilities

    async def _select_optimal_tools(
        self,
        required_capabilities: list[ToolCapability],
        _constraints: dict[str, Any],
        preferences: dict[str, Any],
    ) -> list[str]:
        """Select optimal tools for required capabilities."""
        selected_tools = []

        for capability in required_capabilities:
            if capability in self.capability_performance:
                # Get tools with this capability
                candidate_tools = self.capability_performance[capability]

                # Sort by performance metrics
                if preferences.get("optimize_for") == "speed":
                    # Prefer faster tools
                    candidate_tools = sorted(
                        candidate_tools,
                        key=lambda t: self.tool_avg_durations.get(t, float("inf")),
                    )
                elif preferences.get("optimize_for") == "quality":
                    # Prefer higher success rate tools
                    candidate_tools = sorted(
                        candidate_tools,
                        key=lambda t: self.tool_success_rates.get(t, 0.0),
                        reverse=True,
                    )
                else:  # balanced
                    # Balance speed and success rate
                    candidate_tools = sorted(
                        candidate_tools,
                        key=lambda t: (
                            self.tool_success_rates.get(t, 0.0) * 0.7
                            + (
                                1.0
                                / max(
                                    1.0, self.tool_avg_durations.get(t, 1000.0) / 1000.0
                                )
                            )
                            * 0.3
                        ),
                        reverse=True,
                    )

                # Select the best tool for this capability
                if candidate_tools:
                    selected_tools.append(candidate_tools[0])

        return selected_tools

    async def _create_execution_nodes(
        self, selected_tools: list[str], _constraints: dict[str, Any]
    ) -> list[ToolChainNode]:
        """Create execution nodes from selected tools."""
        nodes = []

        for i, tool_id in enumerate(selected_tools):
            tool_def = self.registered_tools[tool_id]

            node = ToolChainNode(
                node_id=f"node_{i}_{tool_id}",
                tool_id=tool_id,
                execution_mode=ToolExecutionMode.SEQUENTIAL,  # Default mode
                depends_on=tool_def.dependencies,
                input_mapping={},
                output_mapping={},
            )

            nodes.append(node)

        return nodes

    async def _optimize_execution_plan(
        self, nodes: list[ToolChainNode], preferences: dict[str, Any]
    ) -> list[ToolChainNode]:
        """Optimize the execution plan based on preferences."""
        # Simple optimization: enable parallel execution where possible
        if preferences.get("optimize_for") == "speed":
            for node in nodes:
                if not node.depends_on:  # No dependencies
                    node.execution_mode = ToolExecutionMode.PARALLEL

        return nodes

    async def _execute_speed_optimized(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute plan optimized for speed."""
        return await self._execute_parallel_nodes(
            plan, execution_state, timeout_seconds
        )

    async def _execute_quality_optimized(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute plan optimized for quality."""
        return await self._execute_sequential_nodes(
            plan, execution_state, timeout_seconds
        )

    async def _execute_cost_optimized(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute plan optimized for cost."""
        return await self._execute_selective_nodes(
            plan, execution_state, timeout_seconds
        )

    async def _execute_balanced(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute plan with balanced optimization."""
        return await self._execute_adaptive_nodes(
            plan, execution_state, timeout_seconds
        )

    async def _execute_parallel_nodes(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute nodes in parallel where possible."""
        results = {}
        tasks = []

        # Create tasks for all nodes that can run in parallel
        for node in plan.nodes:
            task = asyncio.create_task(self._execute_node(node, execution_state))
            tasks.append((node.node_id, task))

        # Wait for completion with timeout
        try:
            done_tasks = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout_seconds,
            )

            # Collect results
            for i, result in enumerate(done_tasks):
                node_id = tasks[i][0]
                if isinstance(result, Exception):
                    results[node_id] = {"error": str(result)}
                else:
                    results[node_id] = result

        except TimeoutError:
            # Cancel remaining tasks
            for _, task in tasks:
                if not task.done():
                    task.cancel()

            results["timeout"] = "Execution timed out"

        return results

    async def _execute_sequential_nodes(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute nodes sequentially."""
        results = {}
        start_time = time.time()

        for node in plan.nodes:
            if time.time() - start_time > timeout_seconds:
                results["timeout"] = "Execution timed out"
                break

            try:
                result = await self._execute_node(node, execution_state)
                results[node.node_id] = result
            except (
                asyncio.CancelledError,
                TimeoutError,
                RuntimeError,
            ) as e:  # noqa: BLE001
                results[node.node_id] = {"error": str(e)}

        return results

    async def _execute_selective_nodes(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        _timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute only essential nodes to minimize cost."""
        # Select high-priority nodes only
        essential_nodes = [
            node
            for node in plan.nodes
            if self.registered_tools[node.tool_id].priority
            in [ToolPriority.CRITICAL, ToolPriority.HIGH]
        ]

        results = {}
        for node in essential_nodes:
            try:
                result = await self._execute_node(node, execution_state)
                results[node.node_id] = result
            except (
                asyncio.CancelledError,
                TimeoutError,
                RuntimeError,
            ) as e:  # noqa: BLE001
                results[node.node_id] = {"error": str(e)}

        return results

    async def _execute_adaptive_nodes(
        self,
        plan: ToolOrchestrationPlan,
        execution_state: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute nodes with adaptive strategy."""
        # Start with parallel execution, fall back to sequential if needed
        try:
            return await self._execute_parallel_nodes(
                plan, execution_state, timeout_seconds / 2
            )
        except (
            OSError,
            PermissionError,
            RuntimeError,
        ):  # noqa: BLE001
            return await self._execute_sequential_nodes(
                plan, execution_state, timeout_seconds / 2
            )

    async def _execute_node(
        self, node: ToolChainNode, execution_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single node in the tool chain."""
        node.start_time = datetime.now(tz=UTC)
        node.status = "running"

        try:
            # Prepare input data
            input_data = execution_state["input_data"].copy()

            # Apply input mapping
            if node.input_mapping:
                for target, source in node.input_mapping.items():
                    if source in execution_state["intermediate_results"]:
                        input_data[target] = execution_state["intermediate_results"][
                            source
                        ]

            # Execute tool
            result = await self.execute_single_tool(
                node.tool_id, input_data, fallback_enabled=True
            )

            node.end_time = datetime.now(tz=UTC)
            node.result = result.result

            if result.success:
                node.status = "completed"
                execution_state["completed_nodes"].add(node.node_id)

                # Apply output mapping
                if node.output_mapping and result.result:
                    for source, target in node.output_mapping.items():
                        if source in result.result:
                            execution_state["intermediate_results"][target] = (
                                result.result[source]
                            )
            else:
                node.status = "failed"
                node.error = result.error
                execution_state["failed_nodes"].add(node.node_id)

        except Exception as e:  # noqa: BLE001
            node.end_time = datetime.now(tz=UTC)
            node.status = "failed"
            node.error = str(e)
            execution_state["failed_nodes"].add(node.node_id)

            logger.exception(f"Node {node.node_id} execution failed")

            return {"error": str(e)}

        return result.result or {"error": result.error}

    async def _execute_fallback_tool(
        self,
        fallback_tool_id: str,
        input_data: dict[str, Any],
        timeout_seconds: float,
        original_execution_id: str,
    ) -> ToolExecutionResult:
        """Execute a fallback tool."""
        logger.info(
            f"Executing fallback tool {fallback_tool_id} for execution "
            f"{original_execution_id}"
        )

        result = await self.execute_single_tool(
            fallback_tool_id,
            input_data,
            timeout_seconds,
            fallback_enabled=False,  # Prevent recursive fallbacks
        )

        result.fallback_used = True
        return result

    async def _mock_tool_execution(
        self, tool_def: ToolDefinition, _input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock tool execution for tools without executors."""
        # Simulate execution time
        await asyncio.sleep(tool_def.estimated_duration_ms / 1000.0)

        return {
            "mock_result": f"Executed {tool_def.name}",
            "quality_score": 0.85,
            "confidence": 0.80,
            "tool_capabilities": list(tool_def.capabilities),
        }

    async def _update_tool_metrics(
        self, tool_id: str, result: ToolExecutionResult
    ) -> None:
        """Update tool performance metrics."""
        # Add to execution history
        self.execution_history.append(result)
        if len(self.execution_history) > 1000:  # Keep last 1000 executions
            self.execution_history = self.execution_history[-1000:]

        # Update tool-specific history
        if tool_id not in self.tool_performance_history:
            self.tool_performance_history[tool_id] = []
        self.tool_performance_history[tool_id].append(result)

        # Keep last 100 executions per tool
        if len(self.tool_performance_history[tool_id]) > 100:
            self.tool_performance_history[tool_id] = self.tool_performance_history[
                tool_id
            ][-100:]

        # Update aggregated metrics
        recent_results = self.tool_performance_history[tool_id][
            -20:
        ]  # Last 20 executions

        self.tool_success_rates[tool_id] = sum(
            1 for r in recent_results if r.success
        ) / len(recent_results)

        self.tool_avg_durations[tool_id] = sum(
            r.duration_ms for r in recent_results
        ) / len(recent_results)

    async def _update_performance_history(
        self, plan: ToolOrchestrationPlan, _execution_state: dict[str, Any]
    ) -> None:
        """Update performance history for the orchestration plan."""
        # This would implement performance tracking
        # For now, just log the completion
        logger.info(f"Updated performance history for plan {plan.plan_id}")
