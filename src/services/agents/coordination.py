"""Parallel agent coordination system for distributed task execution.

This module provides a coordinator for managing parallel execution of tasks across
multiple agents with health monitoring, circuit breaker patterns, and various
coordination strategies (sequential, parallel, pipeline, hierarchical, adaptive).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.services.agents.core import AgentState, BaseAgent, BaseAgentDependencies
from src.services.cache.patterns import CircuitBreakerPattern


logger = logging.getLogger(__name__)


class CoordinationStrategy(str, Enum):
    """Enumeration of available agent coordination strategies.

    Attributes:
        SEQUENTIAL: Execute tasks one after another.
        PARALLEL: Execute tasks simultaneously.
        PIPELINE: Execute tasks in dependency order with overlap.
        HIERARCHICAL: Use coordinator agents for task delegation.
        ADAPTIVE: Automatically choose optimal strategy based on task characteristics.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class TaskPriority(str, Enum):
    """Enumeration of task priority levels for scheduling.

    Attributes:
        CRITICAL: Highest priority, execute immediately.
        HIGH: High priority tasks.
        NORMAL: Standard priority.
        LOW: Low priority tasks.
        BATCH: Lowest priority, execute during idle periods.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BATCH = "batch"


class AgentRole(str, Enum):
    """Enumeration of agent roles in the coordination system.

    Attributes:
        COORDINATOR: High-level coordination and delegation.
        SPECIALIST: Domain-specific task execution.
        WORKER: General-purpose task execution.
        MONITOR: Health monitoring and reporting.
        BACKUP: Standby agents for failover.
    """

    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    WORKER = "worker"
    MONITOR = "monitor"
    BACKUP = "backup"


@dataclass(slots=True)
class TaskDefinition:  # pylint: disable=too-many-instance-attributes
    """Definition of a coordinated task with execution requirements.

    Attributes:
        task_id: Unique identifier for the task.
        description: Human-readable description of the task.
        priority: Priority level for task scheduling.
        estimated_duration_ms: Expected execution time in milliseconds.
        dependencies: List of task IDs that must complete before this task.
        required_capabilities: List of agent capabilities required for execution.
        input_data: Dictionary of input data for task execution.
        timeout_ms: Optional timeout in milliseconds.
        retry_count: Number of retry attempts made so far.
        max_retries: Maximum number of retry attempts allowed.
    """

    task_id: str
    description: str
    priority: TaskPriority
    estimated_duration_ms: float
    dependencies: list[str]
    required_capabilities: list[str]
    input_data: dict[str, Any]
    timeout_ms: float | None = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass(slots=True)
class AgentAssignment:
    """Assignment of an agent to a specific task with timing information.

    Attributes:
        agent_name: Name of the assigned agent.
        task_id: ID of the assigned task.
        assigned_at: Timestamp when the assignment was made.
        estimated_completion: Expected completion timestamp.
        actual_start: Actual start timestamp, None if not started.
        actual_completion: Actual completion timestamp, None if not completed.
        status: Current assignment status
            (assigned, running, completed, failed, cancelled).
    """

    agent_name: str
    task_id: str
    assigned_at: datetime
    estimated_completion: datetime
    actual_start: datetime | None = None
    actual_completion: datetime | None = None
    status: str = "assigned"  # assigned, running, completed, failed, cancelled


class CoordinationMetrics(BaseModel):
    """Metrics for coordination performance analysis and monitoring.

    Attributes:
        total_tasks: Total number of tasks submitted.
        completed_tasks: Number of successfully completed tasks.
        failed_tasks: Number of failed tasks.
        avg_completion_time_ms: Average task completion time in milliseconds.
        max_completion_time_ms: Maximum task completion time in milliseconds.
        parallelism_achieved: Actual parallelism achieved (0.0-1.0).
        resource_utilization: Agent resource utilization (0.0-1.0).
        coordination_overhead_ms: Time spent on coordination activities.
        task_success_rate: Ratio of successful tasks (0.0-1.0).
        strategy_effectiveness: Effectiveness of chosen strategies (0.0-1.0).
        adaptation_success_rate: Success rate of adaptive strategy selection.
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_completion_time_ms: float = 0.0
    max_completion_time_ms: float = 0.0
    parallelism_achieved: float = 0.0
    resource_utilization: float = 0.0
    coordination_overhead_ms: float = 0.0
    task_success_rate: float = 0.0
    strategy_effectiveness: float = 0.0
    adaptation_success_rate: float = 0.0


class AgentCoordinationResult(BaseModel):
    """Result of agent coordination execution."""

    coordination_id: str = Field(...)
    success: bool = Field(...)
    strategy_used: CoordinationStrategy = Field(...)
    execution_time_seconds: float = Field(...)
    task_results: dict[str, Any] = Field(default_factory=dict)
    metrics: CoordinationMetrics = Field(default_factory=CoordinationMetrics)
    error_message: str | None = None
    agent_assignments: list[AgentAssignment] = Field(default_factory=list)


# ---- Strategy selection -----------------------------------------------------


def _priority_score(priority: TaskPriority) -> int:
    order = {
        TaskPriority.CRITICAL: 4,
        TaskPriority.HIGH: 3,
        TaskPriority.NORMAL: 2,
        TaskPriority.LOW: 1,
        TaskPriority.BATCH: 0,
    }
    return order.get(priority, 2)


def choose_coordination_strategy(tasks: list[TaskDefinition]) -> CoordinationStrategy:
    """Choose optimal coordination strategy using simple heuristics.

    Uses task characteristics like dependencies, priority, and estimated duration
    to select the most appropriate coordination strategy.

    Args:
        tasks: List of task definitions to analyze.

    Returns:
        The recommended coordination strategy.
    """
    if not tasks or len(tasks) == 1:
        return CoordinationStrategy.SEQUENTIAL

    has_deps = any(t.dependencies for t in tasks)
    avg_ms = sum(t.estimated_duration_ms for t in tasks) / len(tasks)
    hi_or_crit = sum(1 for t in tasks if _priority_score(t.priority) >= 3)

    if has_deps and len(tasks) > 3:
        return CoordinationStrategy.PIPELINE
    if hi_or_crit > len(tasks) / 2:
        return CoordinationStrategy.HIERARCHICAL
    if avg_ms < 1000.0:
        return CoordinationStrategy.PARALLEL
    return CoordinationStrategy.PARALLEL


# ---- Config & State to avoid R0902 -----------------------------------------


@dataclass(slots=True)
class CoordinatorConfig:
    """Coordinator configuration values."""

    max_parallel_agents: int
    default_strategy: CoordinationStrategy
    enable_circuit_breaker: bool
    health_check_interval_ms: float


@dataclass(slots=True)
class AgentRegistry:
    """Agent registration and health state management.

    Attributes:
        available_agents: Dictionary mapping agent names to agent instances.
        agent_capabilities: Dictionary mapping agent names to capability sets.
        agent_roles: Dictionary mapping agent names to their roles.
        agent_health_status: Dictionary mapping agent names to health status.
        circuit_breakers: Dictionary mapping agent names to circuit breaker instances.
    """

    available_agents: dict[str, BaseAgent]
    agent_capabilities: dict[str, set[str]]
    agent_roles: dict[str, AgentRole]
    agent_health_status: dict[str, bool]
    circuit_breakers: dict[str, CircuitBreakerPattern]  # pylint: disable=too-many-instance-attributes


@dataclass(slots=True)
class TaskQueues:
    """Task management queues for different execution states.

    Attributes:
        pending_tasks: Queue of tasks waiting for execution.
        running_tasks: Dictionary mapping task IDs to current assignments.
        completed_tasks: List of successfully completed task assignments.
        failed_tasks: List of failed task assignments.
    """

    pending_tasks: list[TaskDefinition]
    running_tasks: dict[str, AgentAssignment]
    completed_tasks: list[AgentAssignment]
    failed_tasks: list[AgentAssignment]


@dataclass(slots=True)
class CoordinatorState:
    """Mutable runtime state for the coordinator.

    Attributes:
        agents: Registry of available agents and their state.
        tasks: Task queues for different execution phases.
        metrics: Performance and coordination metrics.
    """

    agents: AgentRegistry
    tasks: TaskQueues
    metrics: CoordinationMetrics


class ParallelAgentCoordinator:
    """Coordinator for parallel agent exec with health monitoring and fault tolerance.

    Manages coordination of tasks across multiple agents using various strategies.
    Provides health monitoring, circuit breaker patterns, and automatic task scheduling
    based on agent capabilities and availability.

    Attributes:
        config: Immutable configuration settings.
        state: Mutable runtime state.
        _coordination_active: Whether coordination loops are running.
        _health_monitor_task: Background health monitoring task.
        _coordination_task: Background coordination task.
    """

    def __init__(
        self,
        max_parallel_agents: int = 5,
        default_strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE,
        enable_circuit_breaker: bool = True,
        health_check_interval_ms: float = 5000.0,
    ) -> None:
        """Initialize the parallel agent coordinator.

        Args:
            max_parallel_agents: Maximum number of agents that can execute in parallel.
            default_strategy: Default coordination strategy to use.
            enable_circuit_breaker: Whether to enable circuit breaker pattern for
                fault tolerance.
            health_check_interval_ms: Interval between agent health checks in
                milliseconds.
        """
        self.config = CoordinatorConfig(
            max_parallel_agents=max_parallel_agents,
            default_strategy=default_strategy,
            enable_circuit_breaker=enable_circuit_breaker,
            health_check_interval_ms=health_check_interval_ms,
        )
        self.state = CoordinatorState(
            agents=AgentRegistry(
                available_agents={},
                agent_capabilities={},
                agent_roles={},
                agent_health_status={},
                circuit_breakers={},
            ),
            tasks=TaskQueues(
                pending_tasks=[],
                running_tasks={},
                completed_tasks=[],
                failed_tasks=[],
            ),
            metrics=CoordinationMetrics(),
        )

        self._coordination_active = False
        self._health_monitor_task: asyncio.Task | None = None
        self._coordination_task: asyncio.Task | None = None

        logger.info(
            "ParallelAgentCoordinator initialized with max_parallel=%d",
            self.config.max_parallel_agents,
        )

    async def register_agent(
        self,
        agent: BaseAgent,
        capabilities: list[str],
        role: AgentRole = AgentRole.WORKER,
    ) -> None:
        """Register an agent for coordinated execution.

        Args:
            agent: The agent instance to register.
            capabilities: List of capabilities this agent provides.
            role: The role this agent should play in coordination.
        """
        name = agent.name
        st = self.state
        st.agents.available_agents[name] = agent
        st.agents.agent_capabilities[name] = set(capabilities)
        st.agents.agent_roles[name] = role
        st.agents.agent_health_status[name] = True

        if self.config.enable_circuit_breaker:
            st.agents.circuit_breakers[name] = CircuitBreakerPattern(
                failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
            )

        logger.info(
            "Agent %s registered with capabilities=%s role=%s", name, capabilities, role
        )

    async def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent and cancel its running tasks.

        Args:
            agent_name: Name of the agent to unregister.
        """
        st = self.state
        to_cancel = [
            task_id
            for task_id, assignment in st.tasks.running_tasks.items()
            if assignment.agent_name == agent_name
        ]
        for task_id in to_cancel:
            await self._cancel_task(task_id, reason="Agent unregistered")

        st.agents.available_agents.pop(agent_name, None)
        st.agents.agent_capabilities.pop(agent_name, None)
        st.agents.agent_roles.pop(agent_name, None)
        st.agents.agent_health_status.pop(agent_name, None)
        st.agents.circuit_breakers.pop(agent_name, None)

        logger.info("Agent %s unregistered", agent_name)

    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for coordination.

        Args:
            task: The task definition to submit.

        Returns:
            The task ID for tracking.
        """
        st = self.state
        st.tasks.pending_tasks.append(task)
        st.metrics.total_tasks += 1
        logger.debug("Task %s submitted with priority %s", task.task_id, task.priority)

        if not self._coordination_active:
            await self.start_coordination()
        return task.task_id

    async def start_coordination(self) -> None:
        """Start coordination and health monitoring loops."""
        if self._coordination_active:
            return
        self._coordination_active = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        logger.info("Parallel agent coordination started")

    async def stop_coordination(self) -> None:
        """Stop all background coordination loops."""
        self._coordination_active = False
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
        if self._coordination_task:
            self._coordination_task.cancel()
        await asyncio.sleep(0.1)
        logger.info("Parallel agent coordination stopped")

    async def execute_coordinated_workflow(
        self,
        tasks: list[TaskDefinition],
        strategy: CoordinationStrategy | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute a set of tasks using the chosen coordination strategy.

        Args:
            tasks: List of task definitions to execute.
            strategy: Coordination strategy to use, defaults to configured strategy.
            timeout_seconds: Optional timeout for the entire workflow.

        Returns:
            Dictionary containing workflow results with success status, execution time,
            and task results.
        """
        workflow_id = str(uuid4())
        chosen = strategy or self.config.default_strategy
        start = time.time()

        logger.info(
            "Starting coordinated workflow %s with %d tasks using %s",
            workflow_id,
            len(tasks),
            chosen.value,
        )

        try:
            ids: list[str] = []
            for task in tasks:
                ids.append(await self.submit_task(task))

            if chosen == CoordinationStrategy.SEQUENTIAL:
                results = await self._execute_sequential(ids, timeout_seconds)
            elif chosen == CoordinationStrategy.PARALLEL:
                results = await self._execute_parallel(ids, timeout_seconds)
            elif chosen == CoordinationStrategy.PIPELINE:
                results = await self._execute_pipeline(ids, timeout_seconds)
            elif chosen == CoordinationStrategy.HIERARCHICAL:
                results = await self._execute_hierarchical(ids, timeout_seconds)
            elif chosen == CoordinationStrategy.ADAPTIVE:
                optimal = await self._select_optimal_strategy(tasks)
                results = await self.execute_coordinated_workflow(
                    tasks, optimal, timeout_seconds
                )
            else:
                raise ValueError(f"Unknown strategy: {chosen}")

            elapsed = time.time() - start
            successful = sum(1 for r in results.values() if r.get("success"))
            success_rate = successful / len(tasks) if tasks else 0.0

            return {
                "workflow_id": workflow_id,
                "success": success_rate >= 0.8,
                "strategy_used": chosen,
                "execution_time_seconds": elapsed,
                "task_results": results,
                "metrics": {
                    "total_tasks": len(tasks),
                    "successful_tasks": successful,
                    "success_rate": success_rate,
                    "avg_task_time": (elapsed / len(tasks)) if tasks else 0.0,
                },
            }

        except (ValueError, RuntimeError) as exc:
            logger.exception("Coordinated workflow %s failed", workflow_id)
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(exc),
                "strategy_used": chosen,
                "execution_time_seconds": time.time() - start,
            }

    async def get_coordination_status(self) -> dict[str, Any]:
        """Return a snapshot of coordinator state for monitoring.

        Returns:
            Dictionary containing current coordination status including agent counts,
            task queue lengths, and performance metrics.
        """
        st = self.state
        return {
            "coordination_active": self._coordination_active,
            "registered_agents": len(st.agents.available_agents),
            "healthy_agents": sum(st.agents.agent_health_status.values()),
            "pending_tasks": len(st.tasks.pending_tasks),
            "running_tasks": len(st.tasks.running_tasks),
            "completed_tasks": len(st.tasks.completed_tasks),
            "failed_tasks": len(st.tasks.failed_tasks),
            "metrics": st.metrics.model_dump(),
        }

    # ---- internal loops -----------------------------------------------------

    async def _coordination_loop(self) -> None:
        """Main coordination loop processing tasks and monitoring."""
        while self._coordination_active:
            try:
                await self._process_pending_tasks()
                await self._monitor_running_tasks()
                await self._update_metrics()
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception:  # pragma: no cover - safety
                logger.exception("Error in coordination loop")
                await asyncio.sleep(0.5)

    async def _health_monitor_loop(self) -> None:
        """Background loop for monitoring agent health status."""
        while self._coordination_active:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.config.health_check_interval_ms / 1000.0)
            except asyncio.CancelledError:
                break
            except Exception:  # pragma: no cover - safety
                logger.exception("Error in health monitor loop")
                await asyncio.sleep(1.0)

    # ---- scheduling ---------------------------------------------------------

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks and assign them to available agents."""
        st = self.state
        if not st.tasks.pending_tasks:
            return

        def _key(t: TaskDefinition) -> tuple[str, int]:
            return (t.priority.value, len(t.dependencies))

        sorted_tasks = sorted(st.tasks.pending_tasks, key=_key)
        to_remove: list[TaskDefinition] = []

        for task in sorted_tasks:
            if not await self._dependencies_satisfied(task):
                continue

            agent_name = await self._find_suitable_agent(task)
            if agent_name is None:
                continue

            assignment = AgentAssignment(
                agent_name=agent_name,
                task_id=task.task_id,
                assigned_at=datetime.now(tz=UTC),
                estimated_completion=(
                    datetime.now(tz=UTC)
                    + timedelta(milliseconds=task.estimated_duration_ms)
                ),
            )

            st.tasks.running_tasks[task.task_id] = assignment
            to_remove.append(task)

            exec_task = asyncio.create_task(self._execute_task(task, assignment))
            exec_task.add_done_callback(
                lambda _t, tid=task.task_id: logger.debug(
                    "Task execution completed for %s", tid
                )
            )

            if len(st.tasks.running_tasks) >= self.config.max_parallel_agents:
                break

        for task in to_remove:
            st.tasks.pending_tasks.remove(task)

    async def _execute_task(
        self, task: TaskDefinition, assignment: AgentAssignment
    ) -> None:
        """Execute a task using the assigned agent with circuit breaker protection."""
        st = self.state
        name = assignment.agent_name
        agent = st.agents.available_agents[name]
        assignment.actual_start = datetime.now(tz=UTC)
        assignment.status = "running"

        try:
            if (
                self.config.enable_circuit_breaker
                and st.agents.circuit_breakers[name].is_open()
            ):
                raise RuntimeError(f"Circuit breaker open for agent {name}")

            deps = BaseAgentDependencies(
                client_manager=None,
                config=None,
                session_state=AgentState(session_id=task.task_id, user_id=None),
            )

            await agent.execute(task.description, deps, task.input_data)

            assignment.actual_completion = datetime.now(tz=UTC)
            assignment.status = "completed"

            if self.config.enable_circuit_breaker:
                await st.agents.circuit_breakers[name].call(lambda: True)

            st.tasks.completed_tasks.append(assignment)
            st.metrics.completed_tasks += 1
            logger.debug(
                "Task %s completed successfully by agent %s", task.task_id, name
            )

        except TimeoutError:
            assignment.status = "failed"
            assignment.actual_completion = datetime.now(tz=UTC)
            st.tasks.failed_tasks.append(assignment)
            st.metrics.failed_tasks += 1
            logger.warning("Task %s timed out for agent %s", task.task_id, name)

            if task.retry_count < task.max_retries:
                task.retry_count += 1
                st.tasks.pending_tasks.append(task)
                logger.info(
                    "Retrying task %s (attempt %d)", task.task_id, task.retry_count + 1
                )

        except Exception:  # pragma: no cover - safety
            assignment.status = "failed"
            assignment.actual_completion = datetime.now(tz=UTC)
            st.tasks.failed_tasks.append(assignment)
            st.metrics.failed_tasks += 1
            logger.exception("Task %s failed for agent %s", task.task_id, name)
        finally:
            st.tasks.running_tasks.pop(task.task_id, None)

    async def _dependencies_satisfied(self, task: TaskDefinition) -> bool:
        """Check if all task dependencies have been completed."""
        st = self.state
        if not task.dependencies:
            return True
        completed_ids = {a.task_id for a in st.tasks.completed_tasks}
        return all(dep_id in completed_ids for dep_id in task.dependencies)

    async def _find_suitable_agent(self, task: TaskDefinition) -> str | None:
        """Find an available agent that can handle the task requirements."""
        st = self.state
        busy = {a.agent_name for a in st.tasks.running_tasks.values()}
        healthy_free = [
            n for n, ok in st.agents.agent_health_status.items() if ok and n not in busy
        ]
        if not healthy_free:
            return None

        capable = [
            n
            for n in healthy_free
            if all(
                c in st.agents.agent_capabilities.get(n, set())
                for c in task.required_capabilities
            )
        ]
        if not capable:
            return None

        if task.priority in (TaskPriority.CRITICAL, TaskPriority.HIGH):
            coords = [
                n
                for n in capable
                if st.agents.agent_roles.get(n) == AgentRole.COORDINATOR
            ]
            if coords:
                return coords[0]
        return capable[0]

    async def _monitor_running_tasks(self) -> None:
        """Monitor running tasks for estimated completion time violations."""
        st = self.state
        now = datetime.now(tz=UTC)
        for task_id, assignment in list(st.tasks.running_tasks.items()):
            if assignment.estimated_completion < now:
                logger.warning("Task %s estimated completion exceeded", task_id)

    async def _check_agent_health(self) -> None:
        """Perform health checks on all registered agents."""
        st = self.state
        for name, agent in st.agents.available_agents.items():
            try:
                st.agents.agent_health_status[name] = bool(
                    getattr(agent, "is_initialized", False)
                )
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Health check failed for agent %s: %s", name, exc)
                st.agents.agent_health_status[name] = False

    async def _update_metrics(self) -> None:
        """Update coordination system performance metrics."""
        st = self.state
        if st.metrics.total_tasks == 0:
            return

        st.metrics.task_success_rate = (
            st.metrics.completed_tasks / st.metrics.total_tasks
        )

        durations: list[float] = []
        for a in st.tasks.completed_tasks:
            if a.actual_start and a.actual_completion:
                durations.append(
                    (a.actual_completion - a.actual_start).total_seconds() * 1000.0
                )

        if durations:
            st.metrics.avg_completion_time_ms = sum(durations) / len(durations)
            st.metrics.max_completion_time_ms = max(durations)

        if st.agents.available_agents:
            st.metrics.resource_utilization = len(st.tasks.running_tasks) / len(
                st.agents.available_agents
            )

    async def _cancel_task(self, task_id: str, reason: str = "Cancelled") -> None:
        """Cancel a running task with the specified reason."""
        st = self.state
        if task_id in st.tasks.running_tasks:
            assignment = st.tasks.running_tasks[task_id]
            assignment.status = "cancelled"
            assignment.actual_completion = datetime.now(tz=UTC)
            st.tasks.failed_tasks.append(assignment)
            st.tasks.running_tasks.pop(task_id, None)
            logger.info("Task %s cancelled: %s", task_id, reason)

    # ---- strategy exec ------------------------------------------------------

    async def _execute_sequential(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in sequential order with optional timeout."""
        st = self.state
        results: dict[str, Any] = {}
        start = time.time()
        for task_id in task_ids:
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                results["timeout"] = True
                break

            while task_id in st.tasks.running_tasks or task_id in [
                t.task_id for t in st.tasks.pending_tasks
            ]:
                await asyncio.sleep(0.05)
                if timeout_seconds and (time.time() - start) > timeout_seconds:
                    results["timeout"] = True
                    break

            completed = next(
                (a for a in st.tasks.completed_tasks if a.task_id == task_id), None
            )
            results[task_id] = {"success": bool(completed), "assignment": completed}
        return results

    async def _execute_parallel(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in parallel with optional timeout."""
        st = self.state
        results: dict[str, Any] = {}
        start = time.time()

        remaining = list(task_ids)
        while remaining and (
            not timeout_seconds or (time.time() - start) < timeout_seconds
        ):
            done: list[str] = []
            for tid in remaining:
                if tid not in st.tasks.running_tasks and tid not in [
                    t.task_id for t in st.tasks.pending_tasks
                ]:
                    completed = next(
                        (a for a in st.tasks.completed_tasks if a.task_id == tid), None
                    )
                    results[tid] = {"success": bool(completed), "assignment": completed}
                    done.append(tid)
            for tid in done:
                remaining.remove(tid)
            if remaining:
                await asyncio.sleep(0.05)

        for tid in remaining:
            results[tid] = {"success": False, "error": "Timed out"}
        return results

    async def _execute_pipeline(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in pipeline mode (sequential execution)."""
        return await self._execute_sequential(task_ids, timeout_seconds)

    async def _execute_hierarchical(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in hierarchical mode (parallel execution)."""
        return await self._execute_parallel(task_ids, timeout_seconds)

    async def _select_optimal_strategy(
        self, tasks: list[TaskDefinition]
    ) -> CoordinationStrategy:
        """Select the optimal coordination strategy for the given tasks."""
        return choose_coordination_strategy(tasks)
