"""Advanced parallel agent coordination system.

This module implements sophisticated coordination patterns for parallel agent execution,
building on the I4 Vector Database Modernization research to create autonomous,
self-optimizing agent orchestration with production-grade performance and reliability.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager
from src.services.agents.core import AgentState, BaseAgent, BaseAgentDependencies
from src.services.cache.patterns import CircuitBreakerPattern
from src.services.observability.tracking import PerformanceTracker


logger = logging.getLogger(__name__)


class CoordinationStrategy(str, Enum):
    """Agent coordination strategies for different workload types."""

    SEQUENTIAL = "sequential"  # One agent at a time
    PARALLEL = "parallel"  # All agents simultaneously
    PIPELINE = "pipeline"  # Sequential with overlap
    HIERARCHICAL = "hierarchical"  # Coordinator-worker pattern
    ADAPTIVE = "adaptive"  # Dynamic strategy selection


class TaskPriority(str, Enum):
    """Task priority levels for resource allocation."""

    CRITICAL = "critical"  # Immediate execution required
    HIGH = "high"  # High priority, minimal delay
    NORMAL = "normal"  # Standard priority
    LOW = "low"  # Background processing
    BATCH = "batch"  # Batch processing when resources available


class AgentRole(str, Enum):
    """Agent roles in coordinated execution."""

    COORDINATOR = "coordinator"  # Orchestrates other agents
    SPECIALIST = "specialist"  # Specialized domain agent
    WORKER = "worker"  # Generic processing agent
    MONITOR = "monitor"  # Health and performance monitoring
    BACKUP = "backup"  # Failover agent


@dataclass
class TaskDefinition:
    """Definition of a coordinated task."""

    task_id: str
    description: str
    priority: TaskPriority
    estimated_duration_ms: float
    dependencies: list[str]  # Task IDs this depends on
    required_capabilities: list[str]  # Required agent capabilities
    input_data: dict[str, Any]
    timeout_ms: float | None = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentAssignment:
    """Assignment of agent to specific task."""

    agent_name: str
    task_id: str
    assigned_at: datetime
    estimated_completion: datetime
    actual_start: datetime | None = None
    actual_completion: datetime | None = None
    status: str = "assigned"  # assigned, running, completed, failed, cancelled


class CoordinationMetrics(BaseModel):
    """Metrics for coordination performance analysis."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_completion_time_ms: float = 0.0
    max_completion_time_ms: float = 0.0

    # Parallel execution metrics
    parallelism_achieved: float = 0.0  # Actual vs. theoretical maximum
    resource_utilization: float = 0.0  # Average agent utilization
    coordination_overhead_ms: float = 0.0

    # Quality metrics
    task_success_rate: float = 0.0
    strategy_effectiveness: float = 0.0
    adaptation_success_rate: float = 0.0


class AgentCoordinationResult(BaseModel):
    """Result of agent coordination execution."""

    coordination_id: str = Field(..., description="Unique coordination identifier")
    success: bool = Field(..., description="Whether coordination succeeded")
    strategy_used: CoordinationStrategy = Field(
        ..., description="Coordination strategy used"
    )
    execution_time_seconds: float = Field(..., description="Total execution time")
    task_results: dict[str, Any] = Field(
        default_factory=dict, description="Individual task results"
    )
    metrics: CoordinationMetrics = Field(
        default_factory=CoordinationMetrics, description="Performance metrics"
    )
    error_message: str | None = Field(None, description="Error message if failed")
    agent_assignments: list[AgentAssignment] = Field(
        default_factory=list, description="Agent task assignments"
    )


class ParallelAgentCoordinator:
    """Advanced coordinator for parallel agent execution.

    Implements sophisticated coordination patterns including:
    - Dynamic strategy selection based on workload characteristics
    - Intelligent load balancing across available agents
    - Fault tolerance with automatic failover and recovery
    - Performance optimization through adaptive scheduling
    - Real-time monitoring and health management
    """

    def __init__(
        self,
        max_parallel_agents: int = 5,
        default_strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE,
        enable_circuit_breaker: bool = True,
        health_check_interval_ms: float = 5000.0,
    ):
        """Initialize the parallel agent coordinator.

        Args:
            max_parallel_agents: Maximum number of agents to run in parallel
            default_strategy: Default coordination strategy
            enable_circuit_breaker: Enable circuit breaker for fault tolerance
            health_check_interval_ms: Interval for agent health checks
        """
        self.max_parallel_agents = max_parallel_agents
        self.default_strategy = default_strategy
        self.enable_circuit_breaker = enable_circuit_breaker
        self.health_check_interval_ms = health_check_interval_ms

        # Agent management
        self.available_agents: dict[str, BaseAgent] = {}
        self.agent_capabilities: dict[str, set[str]] = {}
        self.agent_roles: dict[str, AgentRole] = {}
        self.agent_health_status: dict[str, bool] = {}

        # Task and execution management
        self.pending_tasks: list[TaskDefinition] = []
        self.running_tasks: dict[str, AgentAssignment] = {}
        self.completed_tasks: list[AgentAssignment] = []
        self.failed_tasks: list[AgentAssignment] = []

        # Performance and monitoring
        self.metrics = CoordinationMetrics()
        self.performance_tracker = PerformanceTracker()
        self.circuit_breakers: dict[str, CircuitBreakerPattern] = {}

        # State management
        self._coordination_active = False
        self._health_monitor_task: asyncio.Task | None = None
        self._coordination_task: asyncio.Task | None = None

        logger.info(
            f"ParallelAgentCoordinator initialized with max_parallel={max_parallel_agents}"
        )

    async def register_agent(
        self,
        agent: BaseAgent,
        capabilities: list[str],
        role: AgentRole = AgentRole.WORKER,
    ) -> None:
        """Register an agent for coordinated execution.

        Args:
            agent: Agent instance to register
            capabilities: List of agent capabilities
            role: Agent role in coordination
        """
        agent_name = agent.name

        self.available_agents[agent_name] = agent
        self.agent_capabilities[agent_name] = set(capabilities)
        self.agent_roles[agent_name] = role
        self.agent_health_status[agent_name] = True

        # Initialize circuit breaker for agent
        if self.enable_circuit_breaker:
            self.circuit_breakers[agent_name] = CircuitBreakerPattern(
                failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
            )

        logger.info(
            f"Agent {agent_name} registered with capabilities: {capabilities}, role: {role}"
        )

    async def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from coordination.

        Args:
            agent_name: Name of agent to unregister
        """
        # Cancel any running tasks for this agent
        tasks_to_cancel = [
            task_id
            for task_id, assignment in self.running_tasks.items()
            if assignment.agent_name == agent_name
        ]

        for task_id in tasks_to_cancel:
            await self._cancel_task(task_id, reason="Agent unregistered")

        # Remove agent from all registries
        self.available_agents.pop(agent_name, None)
        self.agent_capabilities.pop(agent_name, None)
        self.agent_roles.pop(agent_name, None)
        self.agent_health_status.pop(agent_name, None)
        self.circuit_breakers.pop(agent_name, None)

        logger.info(f"Agent {agent_name} unregistered")

    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for coordinated execution.

        Args:
            task: Task definition to execute

        Returns:
            Task ID for tracking
        """
        self.pending_tasks.append(task)
        self.metrics.total_tasks += 1

        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")

        # Trigger coordination if not already active
        if not self._coordination_active:
            await self.start_coordination()

        return task.task_id

    async def start_coordination(self) -> None:
        """Start the coordination system."""
        if self._coordination_active:
            return

        self._coordination_active = True

        # Start health monitoring
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        # Start coordination loop
        self._coordination_task = asyncio.create_task(self._coordination_loop())

        logger.info("Parallel agent coordination started")

    async def stop_coordination(self) -> None:
        """Stop the coordination system."""
        self._coordination_active = False

        # Cancel running tasks
        if self._health_monitor_task:
            self._health_monitor_task.cancel()

        if self._coordination_task:
            self._coordination_task.cancel()

        # Wait for tasks to complete or timeout
        await asyncio.sleep(1.0)

        logger.info("Parallel agent coordination stopped")

    async def execute_coordinated_workflow(
        self,
        tasks: list[TaskDefinition],
        strategy: CoordinationStrategy | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute a coordinated workflow of multiple tasks.

        Args:
            tasks: List of tasks to execute
            strategy: Coordination strategy to use
            timeout_seconds: Maximum time to wait for completion

        Returns:
            Workflow execution results
        """
        workflow_id = str(uuid4())
        strategy = strategy or self.default_strategy
        start_time = time.time()

        logger.info(
            f"Starting coordinated workflow {workflow_id} with {len(tasks)} tasks using {strategy} strategy"
        )

        try:
            # Submit all tasks
            task_ids = []
            for task in tasks:
                task_id = await self.submit_task(task)
                task_ids.append(task_id)

            # Wait for completion based on strategy
            if strategy == CoordinationStrategy.SEQUENTIAL:
                results = await self._execute_sequential(task_ids, timeout_seconds)
            elif strategy == CoordinationStrategy.PARALLEL:
                results = await self._execute_parallel(task_ids, timeout_seconds)
            elif strategy == CoordinationStrategy.PIPELINE:
                results = await self._execute_pipeline(task_ids, timeout_seconds)
            elif strategy == CoordinationStrategy.HIERARCHICAL:
                results = await self._execute_hierarchical(task_ids, timeout_seconds)
            elif strategy == CoordinationStrategy.ADAPTIVE:
                # Select best strategy based on task characteristics
                optimal_strategy = await self._select_optimal_strategy(tasks)
                results = await self.execute_coordinated_workflow(
                    tasks, optimal_strategy, timeout_seconds
                )
            else:
                msg = f"Unknown coordination strategy: {strategy}"
                raise ValueError(msg)

            execution_time = time.time() - start_time

            # Calculate workflow-level metrics
            successful_tasks = sum(
                1 for r in results.values() if r.get("success", False)
            )
            success_rate = successful_tasks / len(tasks) if tasks else 0.0

            return {
                "workflow_id": workflow_id,
                "success": success_rate > 0.8,  # 80% success threshold
                "strategy_used": strategy,
                "execution_time_seconds": execution_time,
                "task_results": results,
                "metrics": {
                    "total_tasks": len(tasks),
                    "successful_tasks": successful_tasks,
                    "success_rate": success_rate,
                    "avg_task_time": execution_time / len(tasks) if tasks else 0.0,
                },
            }

        except Exception as e:
            logger.error(
                f"Coordinated workflow {workflow_id} failed: {e}", exc_info=True
            )
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "strategy_used": strategy,
                "execution_time_seconds": time.time() - start_time,
            }

    async def get_coordination_status(self) -> dict[str, Any]:
        """Get current coordination system status.

        Returns:
            Comprehensive status information
        """
        return {
            "coordination_active": self._coordination_active,
            "registered_agents": len(self.available_agents),
            "healthy_agents": sum(
                1 for status in self.agent_health_status.values() if status
            ),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "metrics": self.metrics.model_dump(),
            "agent_status": {
                name: {
                    "healthy": self.agent_health_status.get(name, False),
                    "role": self.agent_roles.get(name, "unknown"),
                    "capabilities": list(self.agent_capabilities.get(name, set())),
                    "circuit_breaker_open": (
                        self.circuit_breakers.get(
                            name, CircuitBreakerPattern()
                        ).is_open()
                        if self.enable_circuit_breaker
                        else False
                    ),
                }
                for name in self.available_agents
            },
        }

    async def _coordination_loop(self) -> None:
        """Main coordination loop for task scheduling and execution."""
        while self._coordination_active:
            try:
                await self._process_pending_tasks()
                await self._monitor_running_tasks()
                await self._update_metrics()

                # Short delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Longer delay on error

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop for agent status tracking."""
        while self._coordination_active:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Longer delay on error

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks for scheduling."""
        if not self.pending_tasks:
            return

        # Sort by priority and dependencies
        sorted_tasks = sorted(
            self.pending_tasks, key=lambda t: (t.priority.value, len(t.dependencies))
        )

        tasks_to_remove = []

        for task in sorted_tasks:
            # Check if dependencies are satisfied
            if not await self._dependencies_satisfied(task):
                continue

            # Find available agent for task
            agent_name = await self._find_suitable_agent(task)
            if agent_name is None:
                continue  # No suitable agent available

            # Assign task to agent
            assignment = AgentAssignment(
                agent_name=agent_name,
                task_id=task.task_id,
                assigned_at=datetime.now(tz=datetime.timezone.utc),
                estimated_completion=datetime.now(tz=datetime.timezone.utc)
                + timedelta(milliseconds=task.estimated_duration_ms),
            )

            self.running_tasks[task.task_id] = assignment
            tasks_to_remove.append(task)

            # Start task execution
            asyncio.create_task(self._execute_task(task, assignment))

            # Respect parallel execution limit
            if len(self.running_tasks) >= self.max_parallel_agents:
                break

        # Remove scheduled tasks from pending
        for task in tasks_to_remove:
            self.pending_tasks.remove(task)

    async def _execute_task(
        self, task: TaskDefinition, assignment: AgentAssignment
    ) -> None:
        """Execute a specific task with an assigned agent.

        Args:
            task: Task to execute
            assignment: Agent assignment details
        """
        agent_name = assignment.agent_name
        agent = self.available_agents[agent_name]

        assignment.actual_start = datetime.now(tz=datetime.timezone.utc)
        assignment.status = "running"

        try:
            # Check circuit breaker
            if (
                self.enable_circuit_breaker
                and self.circuit_breakers[agent_name].is_open()
            ):
                msg = f"Circuit breaker open for agent {agent_name}"
                raise RuntimeError(msg)

            # Create agent dependencies
            deps = BaseAgentDependencies(
                client_manager=None,  # Will be provided by context
                config=None,  # Will be provided by context
                session_state=AgentState(session_id=task.task_id),
            )

            # Execute task with timeout
            timeout_seconds = (task.timeout_ms or 30000) / 1000.0

            result = await asyncio.wait_for(
                agent.execute(task.description, deps, task.input_data),
                timeout=timeout_seconds,
            )

            assignment.actual_completion = datetime.now(tz=datetime.timezone.utc)
            assignment.status = "completed"

            # Track success in circuit breaker
            if self.enable_circuit_breaker:
                await self.circuit_breakers[agent_name].call(lambda: True)

            self.completed_tasks.append(assignment)
            self.metrics.completed_tasks += 1

            logger.debug(
                f"Task {task.task_id} completed successfully by agent {agent_name}"
            )

        except TimeoutError:
            assignment.status = "failed"
            assignment.actual_completion = datetime.now(tz=datetime.timezone.utc)

            self.failed_tasks.append(assignment)
            self.metrics.failed_tasks += 1

            logger.warning(f"Task {task.task_id} timed out for agent {agent_name}")

            # Consider retry if within limits
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.pending_tasks.append(task)
                logger.info(
                    f"Retrying task {task.task_id} (attempt {task.retry_count + 1})"
                )

        except Exception as e:
            assignment.status = "failed"
            assignment.actual_completion = datetime.now(tz=datetime.timezone.utc)

            self.failed_tasks.append(assignment)
            self.metrics.failed_tasks += 1

            logger.exception(f"Task {task.task_id} failed for agent {agent_name}: {e}")

            # Track failure in circuit breaker
            if self.enable_circuit_breaker:
                try:
                    await self.circuit_breakers[agent_name].call(
                        lambda: (_ for _ in ()).throw(e)
                    )
                except:
                    pass  # Expected to fail

        finally:
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)

    async def _dependencies_satisfied(self, task: TaskDefinition) -> bool:
        """Check if task dependencies are satisfied.

        Args:
            task: Task to check

        Returns:
            True if all dependencies are completed
        """
        if not task.dependencies:
            return True

        completed_task_ids = {assignment.task_id for assignment in self.completed_tasks}

        return all(dep_id in completed_task_ids for dep_id in task.dependencies)

    async def _find_suitable_agent(self, task: TaskDefinition) -> str | None:
        """Find a suitable agent for the given task.

        Args:
            task: Task requiring execution

        Returns:
            Agent name if suitable agent found, None otherwise
        """
        # Filter by health status and availability
        healthy_agents = [
            name
            for name, status in self.agent_health_status.items()
            if status
            and name not in [a.agent_name for a in self.running_tasks.values()]
        ]

        if not healthy_agents:
            return None

        # Filter by capabilities
        capable_agents = [
            name
            for name in healthy_agents
            if all(
                capability in self.agent_capabilities.get(name, set())
                for capability in task.required_capabilities
            )
        ]

        if not capable_agents:
            return None

        # Prefer coordinator agents for complex tasks
        if task.priority == TaskPriority.CRITICAL:
            coordinator_agents = [
                name
                for name in capable_agents
                if self.agent_roles.get(name) == AgentRole.COORDINATOR
            ]
            if coordinator_agents:
                return coordinator_agents[0]

        # Select agent with best performance history
        # For now, return first available
        return capable_agents[0]

    async def _monitor_running_tasks(self) -> None:
        """Monitor running tasks for completion and timeouts."""
        current_time = datetime.now(tz=datetime.timezone.utc)

        for task_id, assignment in list(self.running_tasks.items()):
            # Check for timeouts
            if assignment.estimated_completion < current_time:
                logger.warning(f"Task {task_id} estimated completion exceeded")

    async def _check_agent_health(self) -> None:
        """Check health status of all registered agents."""
        for agent_name, agent in self.available_agents.items():
            try:
                # Simple health check - could be enhanced with agent-specific checks
                is_healthy = hasattr(agent, "_initialized") and agent._initialized
                self.agent_health_status[agent_name] = is_healthy

            except Exception as e:
                logger.warning(f"Health check failed for agent {agent_name}: {e}")
                self.agent_health_status[agent_name] = False

    async def _update_metrics(self) -> None:
        """Update coordination performance metrics."""
        if self.metrics.total_tasks == 0:
            return

        # Calculate success rate
        self.metrics.task_success_rate = (
            self.metrics.completed_tasks / self.metrics.total_tasks
        )

        # Calculate average completion time
        if self.completed_tasks:
            completion_times = []
            for assignment in self.completed_tasks:
                if assignment.actual_start and assignment.actual_completion:
                    duration = (
                        assignment.actual_completion - assignment.actual_start
                    ).total_seconds() * 1000
                    completion_times.append(duration)

            if completion_times:
                self.metrics.avg_completion_time_ms = sum(completion_times) / len(
                    completion_times
                )
                self.metrics.max_completion_time_ms = max(completion_times)

        # Calculate resource utilization
        if self.available_agents:
            active_agents = len(self.running_tasks)
            self.metrics.resource_utilization = active_agents / len(
                self.available_agents
            )

    async def _cancel_task(self, task_id: str, reason: str = "Cancelled") -> None:
        """Cancel a running task.

        Args:
            task_id: ID of task to cancel
            reason: Cancellation reason
        """
        if task_id in self.running_tasks:
            assignment = self.running_tasks[task_id]
            assignment.status = "cancelled"
            assignment.actual_completion = datetime.now(tz=datetime.timezone.utc)

            self.failed_tasks.append(assignment)
            self.running_tasks.pop(task_id)

            logger.info(f"Task {task_id} cancelled: {reason}")

    async def _execute_sequential(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks sequentially."""
        results = {}
        start_time = time.time()

        for task_id in task_ids:
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                break

            # Wait for this specific task to complete
            while task_id in self.running_tasks or task_id in [
                t.task_id for t in self.pending_tasks
            ]:
                await asyncio.sleep(0.1)
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    break

            # Collect result
            completed_assignment = next(
                (a for a in self.completed_tasks if a.task_id == task_id), None
            )
            if completed_assignment:
                results[task_id] = {"success": True, "assignment": completed_assignment}
            else:
                results[task_id] = {"success": False, "error": "Task not completed"}

        return results

    async def _execute_parallel(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in parallel."""
        results = {}
        start_time = time.time()

        # Wait for all tasks to complete or timeout
        while task_ids and (
            not timeout_seconds or (time.time() - start_time) < timeout_seconds
        ):
            # Check for completed tasks
            completed_ids = []
            for task_id in task_ids:
                if task_id not in self.running_tasks and task_id not in [
                    t.task_id for t in self.pending_tasks
                ]:
                    completed_assignment = next(
                        (a for a in self.completed_tasks if a.task_id == task_id), None
                    )
                    if completed_assignment:
                        results[task_id] = {
                            "success": True,
                            "assignment": completed_assignment,
                        }
                    else:
                        results[task_id] = {
                            "success": False,
                            "error": "Task not completed",
                        }
                    completed_ids.append(task_id)

            # Remove completed tasks
            for task_id in completed_ids:
                task_ids.remove(task_id)

            if task_ids:
                await asyncio.sleep(0.1)

        # Handle remaining tasks as timed out
        for task_id in task_ids:
            results[task_id] = {"success": False, "error": "Timed out"}

        return results

    async def _execute_pipeline(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks in pipeline mode with overlap."""
        # Pipeline execution allows next task to start when previous is 50% complete
        # For now, implement as sequential - can be enhanced
        return await self._execute_sequential(task_ids, timeout_seconds)

    async def _execute_hierarchical(
        self, task_ids: list[str], timeout_seconds: float | None
    ) -> dict[str, Any]:
        """Execute tasks using hierarchical coordination."""
        # Hierarchical execution uses coordinator agents to manage worker agents
        # For now, implement as parallel - can be enhanced
        return await self._execute_parallel(task_ids, timeout_seconds)

    async def _select_optimal_strategy(
        self, tasks: list[TaskDefinition]
    ) -> CoordinationStrategy:
        """Select optimal coordination strategy based on task characteristics.

        Args:
            tasks: Tasks to analyze

        Returns:
            Optimal coordination strategy
        """
        if not tasks:
            return CoordinationStrategy.SEQUENTIAL

        # Analyze task characteristics
        has_dependencies = any(task.dependencies for task in tasks)
        high_priority_count = sum(
            1
            for task in tasks
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
        )
        avg_duration = sum(task.estimated_duration_ms for task in tasks) / len(tasks)

        # Decision logic for strategy selection
        if len(tasks) == 1:
            return CoordinationStrategy.SEQUENTIAL
        if has_dependencies and len(tasks) > 3:
            return CoordinationStrategy.PIPELINE
        if high_priority_count > len(tasks) / 2:  # >50% high priority
            return CoordinationStrategy.HIERARCHICAL
        if avg_duration < 1000:  # Short tasks benefit from parallel execution
            return CoordinationStrategy.PARALLEL
        return CoordinationStrategy.PARALLEL  # Default for most cases
