"""Advanced tool orchestration system for agentic workflows."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager
from src.services.cache.patterns import CircuitBreakerPattern


logger = logging.getLogger(__name__)


class ToolExecutionMode(str, Enum):
    """Execution modes for tools."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class ToolPriority(str, Enum):
    """Tool execution priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    OPTIONAL = "optional"


class ToolCapability(str, Enum):
    """Tool capability categories for selection."""

    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    COORDINATION = "coordination"


@dataclass(slots=True)
class ToolDefinition:  # pylint: disable=too-many-instance-attributes
    """Definition of a composable tool."""

    tool_id: str
    name: str
    description: str
    capabilities: set[ToolCapability]
    priority: ToolPriority
    estimated_duration_ms: float
    resource_requirements: dict[str, float]
    dependencies: list[str]
    success_rate: float = 0.95
    fallback_tools: list[str] = field(default_factory=list)
    timeout_ms: float | None = None
    executor: Callable[[dict[str, Any]], asyncio.Future[Any] | Any] | None = None


class ToolChainNode(BaseModel):
    """Node in a tool execution chain."""

    node_id: str = Field(...)
    tool_id: str = Field(...)
    execution_mode: ToolExecutionMode = Field(...)
    depends_on: list[str] = Field(default_factory=list)
    conditions: dict[str, Any] = Field(default_factory=dict)
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    status: str = Field("pending")
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ToolOrchestrationPlan(BaseModel):
    """Plan for orchestrating multiple tools."""

    plan_id: str = Field(...)
    goal: str = Field(...)
    nodes: list[ToolChainNode] = Field(...)
    max_parallel_tools: int = Field(5)
    timeout_seconds: float = Field(60.0)
    failure_threshold: float = Field(0.2)
    min_quality_score: float = Field(0.8)
    max_cost: float | None = None
    optimize_for: str = Field("balanced")


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""

    execution_id: str = Field(...)
    tool_id: str = Field(...)
    success: bool = Field(...)
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    duration_ms: float = Field(...)
    result: dict[str, Any] | None = Field(None)
    quality_score: float | None = Field(None)
    confidence: float | None = Field(None)
    error: str | None = Field(None)
    fallback_used: bool = Field(False)
    retry_count: int = Field(0)
    resource_usage: dict[str, float] = Field(default_factory=dict)


# ---- Config & State to avoid R0902 -----------------------------------------


@dataclass(slots=True)
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    max_parallel_executions: int
    default_timeout_seconds: float
    enable_circuit_breakers: bool


@dataclass(slots=True)
class OrchestratorState:  # pylint: disable=too-many-instance-attributes
    """Mutable runtime state for orchestration."""

    registered_tools: dict[str, ToolDefinition]
    tool_performance_history: dict[str, list[ToolExecutionResult]]
    circuit_breakers: dict[str, CircuitBreakerPattern]

    active_executions: dict[str, asyncio.Task]
    execution_history: list[ToolExecutionResult]
    orchestration_plans: dict[str, ToolOrchestrationPlan]

    tool_success_rates: dict[str, float]
    tool_avg_durations: dict[str, float]
    capability_performance: dict[ToolCapability, list[str]]


def _raise_exception(exception: Exception) -> None:
    """Helper that raises the provided exception when invoked."""

    raise exception


class AdvancedToolOrchestrator:
    """Tool orchestrator with parallel execution and circuit breakers."""

    def __init__(
        self,
        client_manager: ClientManager,
        max_parallel_executions: int = 10,
        default_timeout_seconds: float = 30.0,
        enable_circuit_breakers: bool = True,
    ) -> None:
        self.client_manager = client_manager
        self.config = OrchestratorConfig(
            max_parallel_executions=max_parallel_executions,
            default_timeout_seconds=default_timeout_seconds,
            enable_circuit_breakers=enable_circuit_breakers,
        )
        self.state = OrchestratorState(
            registered_tools={},
            tool_performance_history={},
            circuit_breakers={},
            active_executions={},
            execution_history=[],
            orchestration_plans={},
            tool_success_rates={},
            tool_avg_durations={},
            capability_performance={},
        )
        logger.info("ToolOrchestrator initialized")

    # ---- registration -------------------------------------------------------

    async def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool for orchestration."""
        st = self.state
        st.registered_tools[tool_def.tool_id] = tool_def
        st.tool_performance_history[tool_def.tool_id] = []
        st.tool_success_rates[tool_def.tool_id] = tool_def.success_rate
        st.tool_avg_durations[tool_def.tool_id] = tool_def.estimated_duration_ms

        if self.config.enable_circuit_breakers:
            st.circuit_breakers[tool_def.tool_id] = CircuitBreakerPattern(
                failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
            )

        for capability in tool_def.capabilities:
            st.capability_performance.setdefault(capability, []).append(
                tool_def.tool_id
            )

        logger.info(
            "Registered tool %s with capabilities=%s",
            tool_def.tool_id,
            tool_def.capabilities,
        )

    # ---- composition --------------------------------------------------------

    async def compose_tool_chain(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> ToolOrchestrationPlan:
        """Compose an orchestration plan from a goal."""
        st = self.state
        plan_id = str(uuid4())
        constraints = constraints or {}
        preferences = preferences or {}
        logger.info("Composing tool chain %s for goal: %s", plan_id, goal)

        required = await self._analyze_goal_capabilities(goal, constraints)
        selected = await self._select_optimal_tools(required, constraints, preferences)
        nodes = await self._create_execution_nodes(selected, constraints)
        nodes = await self._optimize_execution_plan(nodes, preferences)

        plan = ToolOrchestrationPlan(
            plan_id=plan_id,
            goal=goal,
            nodes=nodes,
            max_parallel_tools=constraints.get("max_parallel_tools", 5),
            timeout_seconds=constraints.get("timeout_seconds", 60.0),
            failure_threshold=constraints.get("failure_threshold", 0.2),
            min_quality_score=constraints.get("min_quality_score", 0.8),
            max_cost=constraints.get("max_cost"),
            optimize_for=preferences.get("optimize_for", "balanced"),
        )
        st.orchestration_plans[plan_id] = plan
        logger.info("Composed tool chain %s with %d nodes", plan_id, len(nodes))
        return plan

    # ---- execution ----------------------------------------------------------

    async def execute_tool_chain(
        self,
        plan: ToolOrchestrationPlan,
        input_data: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute an orchestration plan."""
        execution_id = str(uuid4())
        timeout_seconds = timeout_seconds or plan.timeout_seconds
        start = time.time()
        logger.info("Executing tool chain %s (%s)", plan.plan_id, execution_id)

        try:
            state: dict[str, Any] = {
                "input_data": input_data,
                "intermediate_results": {},
                "completed_nodes": set(),
                "failed_nodes": set(),
                "execution_metadata": {
                    "execution_id": execution_id,
                    "plan_id": plan.plan_id,
                    "start_time": datetime.now(tz=UTC),
                    "goal": plan.goal,
                },
            }

            if plan.optimize_for == "speed":
                results = await self._execute_parallel_nodes(
                    plan, state, timeout_seconds
                )
            elif plan.optimize_for == "quality":
                results = await self._execute_sequential_nodes(
                    plan, state, timeout_seconds
                )
            elif plan.optimize_for == "cost":
                results = await self._execute_selective_nodes(
                    plan, state, timeout_seconds
                )
            else:
                results = await self._execute_adaptive_nodes(
                    plan, state, timeout_seconds
                )

            elapsed = time.time() - start
            success_nodes = len(state["completed_nodes"])
            total = len(plan.nodes)
            success_rate = success_nodes / total if total else 0.0

            await self._update_performance_history(plan, state)
            logger.info(
                "Tool chain %s completed with %.2f%% success",
                execution_id,
                success_rate * 100.0,
            )

            return {
                "execution_id": execution_id,
                "plan_id": plan.plan_id,
                "success": success_rate >= (1.0 - plan.failure_threshold),
                "execution_time_seconds": elapsed,
                "results": results,
                "metadata": {
                    "total_nodes": total,
                    "successful_nodes": success_nodes,
                    "success_rate": success_rate,
                    "total_execution_time_ms": elapsed * 1000,
                    "goal_achieved": success_rate >= plan.min_quality_score,
                    "optimization_target": plan.optimize_for,
                },
            }

        except Exception as exc:
            logger.exception("Tool chain execution %s failed", execution_id)
            return {
                "execution_id": execution_id,
                "plan_id": plan.plan_id,
                "success": False,
                "error": str(exc),
                "execution_time_seconds": time.time() - start,
                "metadata": {"goal": plan.goal, "failure_point": "orchestration"},
            }

    async def execute_single_tool(
        self,
        tool_id: str,
        input_data: dict[str, Any],
        timeout_seconds: float | None = None,
        fallback_enabled: bool = True,
    ) -> ToolExecutionResult:
        """Execute a single tool with circuit breaker & fallback handling."""
        st = self.state
        execution_id = str(uuid4())
        start_time = datetime.now(tz=UTC)

        if tool_id not in st.registered_tools:
            raise ValueError(f"Tool {tool_id} not registered")

        tool_def = st.registered_tools[tool_id]
        timeout_seconds = timeout_seconds or (tool_def.timeout_ms or 30000) / 1000.0
        logger.debug("Executing tool %s (%s)", tool_id, execution_id)

        cb = st.circuit_breakers.get(tool_id)
        try:
            if cb and cb.is_open():
                if fallback_enabled and tool_def.fallback_tools:
                    return await self._execute_fallback_tool(
                        tool_def.fallback_tools[0],
                        input_data,
                        timeout_seconds,
                        execution_id,
                    )
                raise RuntimeError(f"Circuit breaker open for tool {tool_id}")

            if tool_def.executor:
                result = await asyncio.wait_for(
                    tool_def.executor(input_data), timeout=timeout_seconds
                )
            else:
                result = await self._mock_tool_execution(tool_def, input_data)

            end_time = datetime.now(tz=UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000.0

            if cb:
                await cb.call(lambda: True)

            exec_result = ToolExecutionResult(
                execution_id=execution_id,
                tool_id=tool_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                result=result,
                quality_score=result.get("quality_score", 0.9),
                confidence=result.get("confidence", 0.85),
                error=None,
                fallback_used=False,
                retry_count=0,
            )
            await self._update_tool_metrics(tool_id, exec_result)
            return exec_result

        except TimeoutError:
            end_time = datetime.now(tz=UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000.0
            logger.warning("Tool %s execution timed out", tool_id)

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
                result=None,
                quality_score=None,
                confidence=None,
                error="Execution timeout",
                fallback_used=False,
                retry_count=0,
            )

        except Exception as exc:
            end_time = datetime.now(tz=UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000.0

            if cb:
                with contextlib.suppress(Exception):
                    await cb.call(_raise_exception, exc)

            logger.exception("Tool %s execution failed", tool_id)

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
                result=None,
                quality_score=None,
                confidence=None,
                error=str(exc),
                fallback_used=False,
                retry_count=0,
            )

    async def get_orchestration_status(self) -> dict[str, Any]:
        """Return a status snapshot for monitoring."""
        st = self.state
        total = len(st.registered_tools)
        healthy = sum(
            1
            for tool_id in st.registered_tools
            if not (st.circuit_breakers.get(tool_id, CircuitBreakerPattern()).is_open())
        )

        recent = [
            r
            for r in st.execution_history
            if r.start_time > datetime.now(tz=UTC) - timedelta(hours=1)
        ]
        avg_success = (
            (sum(1 for r in recent if r.success) / len(recent)) if recent else 0.0
        )
        avg_duration = (
            (sum(r.duration_ms for r in recent) / len(recent)) if recent else 0.0
        )

        return {
            "registered_tools": total,
            "healthy_tools": healthy,
            "active_executions": len(st.active_executions),
            "orchestration_plans": len(st.orchestration_plans),
            "recent_executions_1h": len(recent),
            "avg_success_rate_1h": avg_success,
            "avg_duration_ms_1h": avg_duration,
            "tool_capabilities": {
                cap.value: len(ids) for cap, ids in st.capability_performance.items()
            },
            "circuit_breaker_status": {
                tool_id: ("open" if cb.is_open() else "closed")
                for tool_id, cb in st.circuit_breakers.items()
            },
        }

    # ---- helpers ------------------------------------------------------------

    async def _analyze_goal_capabilities(
        self, goal: str, _constraints: dict[str, Any]
    ) -> list[ToolCapability]:
        """Analyze goal requirements to determine needed tool capabilities."""
        s = goal.lower()
        required: list[ToolCapability] = []
        if any(k in s for k in ["search", "find", "retrieve", "lookup"]):
            required.append(ToolCapability.SEARCH)
        if any(k in s for k in ["analyze", "process", "examine", "evaluate"]):
            required.append(ToolCapability.ANALYSIS)
        if any(k in s for k in ["generate", "create", "build", "compose"]):
            required.append(ToolCapability.GENERATION)
        if any(k in s for k in ["validate", "verify", "check", "confirm"]):
            required.append(ToolCapability.VALIDATION)
        if any(k in s for k in ["optimize", "improve", "enhance", "tune"]):
            required.append(ToolCapability.OPTIMIZATION)
        if not required:
            required.append(ToolCapability.SEARCH)
        return required

    async def _select_optimal_tools(
        self,
        required_capabilities: list[ToolCapability],
        _constraints: dict[str, Any],
        preferences: dict[str, Any],
    ) -> list[str]:
        """Select optimal tools based on capabilities and performance preferences."""
        st = self.state
        selected: list[str] = []
        for cap in required_capabilities:
            candidates = list(st.capability_performance.get(cap, []))
            if not candidates:
                continue
            if preferences.get("optimize_for") == "speed":
                candidates.sort(
                    key=lambda t: st.tool_avg_durations.get(t, float("inf"))
                )
            elif preferences.get("optimize_for") == "quality":
                candidates.sort(
                    key=lambda t: st.tool_success_rates.get(t, 0.0), reverse=True
                )
            else:
                candidates.sort(
                    key=lambda t: (
                        st.tool_success_rates.get(t, 0.0) * 0.7
                        + (
                            1.0
                            / max(1.0, st.tool_avg_durations.get(t, 1000.0) / 1000.0)
                        )
                        * 0.3
                    ),
                    reverse=True,
                )
            selected.append(candidates[0])
        return selected

    async def _create_execution_nodes(
        self, selected_tools: list[str], _constraints: dict[str, Any]
    ) -> list[ToolChainNode]:
        """Create execution nodes for selected tools."""
        st = self.state
        nodes: list[ToolChainNode] = []
        for i, tool_id in enumerate(selected_tools):
            tool_def = st.registered_tools[tool_id]
            nodes.append(
                ToolChainNode(
                    node_id=f"node_{i}_{tool_id}",
                    tool_id=tool_id,
                    execution_mode=ToolExecutionMode.SEQUENTIAL,
                    depends_on=tool_def.dependencies,
                    input_mapping={},
                    output_mapping={},
                    status="pending",
                )
            )
        return nodes

    async def _optimize_execution_plan(
        self, nodes: list[ToolChainNode], preferences: dict[str, Any]
    ) -> list[ToolChainNode]:
        """Optimize execution plan based on performance preferences."""
        if preferences.get("optimize_for") == "speed":
            for n in nodes:
                if not n.depends_on:
                    n.execution_mode = ToolExecutionMode.PARALLEL
        return nodes

    async def _execute_parallel_nodes(
        self, plan: ToolOrchestrationPlan, state: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Execute tool chain nodes in parallel."""
        results: dict[str, Any] = {}
        tasks: list[tuple[str, asyncio.Task]] = [
            (node.node_id, asyncio.create_task(self._execute_node(node, state)))
            for node in plan.nodes
        ]
        try:
            done_results = await asyncio.wait_for(
                asyncio.gather(*(t for _, t in tasks), return_exceptions=True),
                timeout=timeout_seconds,
            )
            for idx, res in enumerate(done_results):
                node_id = tasks[idx][0]
                results[node_id] = (
                    {"error": str(res)} if isinstance(res, Exception) else res
                )
        except TimeoutError:
            for _, task in tasks:
                if not task.done():
                    task.cancel()
            results["timeout"] = "Execution timed out"
        return results

    async def _execute_sequential_nodes(
        self, plan: ToolOrchestrationPlan, state: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Execute tool chain nodes sequentially."""
        results: dict[str, Any] = {}
        start = time.time()
        for node in plan.nodes:
            if (time.time() - start) > timeout_seconds:
                results["timeout"] = "Execution timed out"
                break
            try:
                results[node.node_id] = await self._execute_node(node, state)
            except Exception as exc:
                results[node.node_id] = {"error": str(exc)}
        return results

    async def _execute_selective_nodes(
        self,
        plan: ToolOrchestrationPlan,
        state: dict[str, Any],
        _timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute tool chain nodes selectively based on conditions."""
        st = self.state
        essential = [
            n
            for n in plan.nodes
            if st.registered_tools[n.tool_id].priority
            in (ToolPriority.CRITICAL, ToolPriority.HIGH)
        ]
        results: dict[str, Any] = {}
        for node in essential:
            try:
                results[node.node_id] = await self._execute_node(node, state)
            except Exception as exc:
                results[node.node_id] = {"error": str(exc)}
        return results

    async def _execute_adaptive_nodes(
        self, plan: ToolOrchestrationPlan, state: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Execute tool chain nodes with adaptive behavior."""
        try:
            return await self._execute_parallel_nodes(
                plan, state, timeout_seconds / 2.0
            )
        except Exception:
            return await self._execute_sequential_nodes(
                plan, state, timeout_seconds / 2.0
            )

    async def _execute_node(
        self, node: ToolChainNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single tool chain node."""
        node.start_time = datetime.now(tz=UTC)
        node.status = "running"
        try:
            inputs = dict(state["input_data"])
            for tgt, src in node.input_mapping.items():
                if src in state["intermediate_results"]:
                    inputs[tgt] = state["intermediate_results"][src]

            result = await self.execute_single_tool(
                node.tool_id, inputs, fallback_enabled=True
            )
            node.end_time = datetime.now(tz=UTC)
            node.result = result.result

            if result.success:
                node.status = "completed"
                state["completed_nodes"].add(node.node_id)
                if node.output_mapping and result.result:
                    for src, tgt in node.output_mapping.items():
                        if src in result.result:
                            state["intermediate_results"][tgt] = result.result[src]
            else:
                node.status = "failed"
                node.error = result.error
                state["failed_nodes"].add(node.node_id)
        except Exception as exc:
            node.end_time = datetime.now(tz=UTC)
            node.status = "failed"
            node.error = str(exc)
            state["failed_nodes"].add(node.node_id)
            logger.exception("Node %s execution failed", node.node_id)
            return {"error": str(exc)}
        return node.result or {"error": "Unknown error"}

    async def _execute_fallback_tool(
        self,
        fallback_tool_id: str,
        input_data: dict[str, Any],
        timeout_seconds: float,
        original_execution_id: str,
    ) -> ToolExecutionResult:
        """Execute fallback tool when primary execution fails."""
        logger.info(
            "Executing fallback tool %s for %s", fallback_tool_id, original_execution_id
        )
        result = await self.execute_single_tool(
            fallback_tool_id, input_data, timeout_seconds, fallback_enabled=False
        )
        result.fallback_used = True
        return result

    async def _mock_tool_execution(
        self, tool_def: ToolDefinition, _input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute mock tool for testing and fallback scenarios."""
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
        """Update performance metrics for a tool after execution."""
        st = self.state
        st.execution_history.append(result)
        if len(st.execution_history) > 1000:
            st.execution_history = st.execution_history[-1000:]

        hist = st.tool_performance_history.setdefault(tool_id, [])
        hist.append(result)
        if len(hist) > 100:
            st.tool_performance_history[tool_id] = hist[-100:]

        recent = st.tool_performance_history[tool_id][-20:]
        st.tool_success_rates[tool_id] = (
            (sum(1 for r in recent if r.success) / len(recent)) if recent else 0.0
        )
        st.tool_avg_durations[tool_id] = (
            (sum(r.duration_ms for r in recent) / len(recent)) if recent else 0.0
        )

    async def _update_performance_history(
        self, plan: ToolOrchestrationPlan, _state: dict[str, Any]
    ) -> None:
        """Update performance history after plan execution."""
        logger.info("Updated performance history for plan %s", plan.plan_id)
