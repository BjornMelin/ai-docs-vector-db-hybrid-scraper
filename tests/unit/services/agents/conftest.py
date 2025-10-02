"""Pytest configuration and lightweight infra stubs for unit tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest


# src.config stub
config_mod = cast(Any, types.ModuleType("src.config"))


def get_config() -> dict[str, Any]:
    return {"env": "test"}


config_mod.get_config = get_config
sys.modules["src.config"] = config_mod

# src.infrastructure.client_manager stub
infra_mod = cast(Any, types.ModuleType("src.infrastructure.client_manager"))


class ClientManager:
    def __init__(self) -> None:  # pragma: no cover - trivial
        self.name = "test-client-manager"


infra_mod.ClientManager = ClientManager
sys.modules["src.infrastructure.client_manager"] = infra_mod

# src.services.cache.patterns stub (CircuitBreakerPattern)
cb_mod = cast(Any, types.ModuleType("src.services.cache.patterns"))


class CircuitBreakerPattern:
    """Minimal async-friendly circuit breaker stub."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        expected_exception: type[BaseException] = Exception,
    ) -> None:
        self._open = False
        self._expected = expected_exception

    def is_open(self) -> bool:
        return self._open

    async def call(self, func: Callable[[], Any]) -> Any:
        try:
            res = func()
            if hasattr(res, "__await__"):
                return await res  # coroutine support
            return res
        except self._expected:
            self._open = True
            raise


cb_mod.CircuitBreakerPattern = CircuitBreakerPattern
sys.modules["src.services.cache.patterns"] = cb_mod

# src.services.observability.tracking stub
obs_mod = cast(Any, types.ModuleType("src.services.observability.tracking"))


class PerformanceTracker:  # pragma: no cover - trivial
    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, name: str) -> None:
        self.events.append(name)


obs_mod.PerformanceTracker = PerformanceTracker
sys.modules["src.services.observability.tracking"] = obs_mod

# src.services.agents.core stub
core_mod = cast(Any, types.ModuleType("src.services.agents.core"))


@dataclass(slots=True)
class AgentState:
    """Lightweight session state container mirroring production surface."""

    session_id: str
    user_id: str | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    tool_usage_stats: dict[str, int] = field(default_factory=dict)

    def add_interaction(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.conversation_history.append(entry)

    def update_metrics(self, metrics: dict[str, float]) -> None:
        self.performance_metrics.update(metrics)

    def increment_tool_usage(self, name: str) -> None:
        self.tool_usage_stats[name] = self.tool_usage_stats.get(name, 0) + 1


@dataclass(slots=True)
class BaseAgentDependencies:  # pragma: no cover - trivial plumbing
    client_manager: Any
    config: Any
    session_state: AgentState


class BaseAgent:
    """Deterministic BaseAgent fallback implementation for tests."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = False
        self.is_initialized = False
        self.agent = None  # pydantic-ai Agent in production

    def get_system_prompt(self) -> str:  # pragma: no cover - overridden in tests
        return ""

    async def initialize(self, deps: BaseAgentDependencies) -> None:
        await self.initialize_tools(deps)
        deps.session_state.add_interaction(
            "system",
            f"agent:{self.name}:initialized",
            {"model": self.model},
        )
        self._initialized = True
        self.is_initialized = True

    async def initialize_tools(
        self, deps: BaseAgentDependencies
    ) -> None:  # pragma: no cover - overridden in tests
        del deps  # pragma: no cover - placeholder to satisfy signature

    async def execute(
        self,
        task: str,
        deps: BaseAgentDependencies,
        _context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        deps.session_state.add_interaction(
            "assistant",
            f"fallback-executed:{task}",
            {"agent": self.name},
        )
        return {
            "success": True,
            "agent": self.name,
            "result": "fallback response generated",
            "fallback_used": True,
        }


def create_agent_dependencies(
    client_manager: Any,
    *,
    config: Any | None = None,
    session_state: AgentState | None = None,
    user_id: str | None = None,
) -> BaseAgentDependencies:
    """Assemble agent dependencies using local stubs."""

    if session_state is None:
        session_state = AgentState(session_id=str(uuid4()), user_id=user_id)
    elif user_id is not None:
        session_state.user_id = user_id

    return BaseAgentDependencies(
        client_manager=client_manager,
        config=config or get_config(),
        session_state=session_state,
    )


class QueryOrchestrator:
    """Deterministic orchestrator stub mirroring fallback behaviour."""

    def __init__(self) -> None:
        self._initialized = False
        self._deps: BaseAgentDependencies | None = None

    async def initialize(self, deps: BaseAgentDependencies) -> None:
        self._deps = deps
        self._initialized = True
        deps.session_state.add_interaction(
            "system",
            "orchestrator_initialized",
            {"component": "query_orchestrator"},
        )

    async def orchestrate_query(self, query: str, *, collection: str) -> dict[str, Any]:
        if not self._initialized or self._deps is None:
            raise RuntimeError("QueryOrchestrator.initialize must be called first")

        result_payload = {
            "query": query,
            "collection": collection,
            "orchestration_plan": [
                {"step": 1, "action": "tool_discovery"},
                {"step": 2, "action": "vector_search"},
            ],
            "mock_results": [
                {
                    "collection": collection,
                    "score": 0.99,
                    "snippet": "Deterministic mock result for testing.",
                }
            ],
            "fallback_used": True,
        }

        return {
            "success": True,
            "result": result_payload,
        }


@dataclass(slots=True)
class AgenticOrchestrationResult:
    """Structured output returned by the agentic orchestrator stub."""

    success: bool
    tools_used: list[str]
    reasoning: list[str]
    latency_ms: float

    def model_dump(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tools_used": self.tools_used,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


class AgenticOrchestrator:
    """Fallback agentic orchestrator providing deterministic outputs."""

    def __init__(self) -> None:
        self._invocations = 0

    async def orchestrate(
        self,
        goal: str,
        constraints: dict[str, Any],
        deps: BaseAgentDependencies,
    ) -> AgenticOrchestrationResult:
        self._invocations += 1
        deps.session_state.add_interaction(
            "system",
            "agentic_orchestrator_invoked",
            {"goal": goal, "invocations": self._invocations},
        )
        latency_budget = constraints.get("max_latency_ms", 1000)
        return AgenticOrchestrationResult(
            success=True,
            tools_used=["dynamic_tool_discovery", "vector_search"],
            reasoning=[
                "Analyzed goal for fallback execution",
                "Executed deterministic plan",
            ],
            latency_ms=min(latency_budget, 1000.0),
        )


class TaskPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class CoordinationStrategy(Enum):
    SEQUENTIAL = auto()
    PIPELINE = auto()
    PARALLEL = auto()
    HIERARCHICAL = auto()


@dataclass(slots=True)
class TaskDefinition:
    task_id: str
    description: str
    priority: TaskPriority
    estimated_duration_ms: float
    dependencies: list[str]
    required_capabilities: list[str]
    input_data: dict[str, Any]


class ParallelAgentCoordinator:
    """Minimal coordinator supporting basic workflow execution for tests."""

    def __init__(self, *, max_parallel_agents: int = 1) -> None:
        self._max_parallel = max_parallel_agents
        self._agents: dict[str, dict[str, Any]] = {}

    async def register_agent(
        self, agent: BaseAgent, capabilities: list[str] | None = None
    ) -> None:
        self._agents[agent.name] = {
            "agent": agent,
            "capabilities": set(capabilities or []),
        }

    async def execute_coordinated_workflow(
        self,
        tasks: list[TaskDefinition],
        strategy: CoordinationStrategy,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del timeout_seconds  # pragma: no cover - argument retained for parity
        task_results: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_results[task.task_id] = {
                "success": True,
                "strategy": strategy.name,
                "assigned_agent": next(iter(self._agents), None),
            }
        return {
            "success": True,
            "task_results": task_results,
            "max_parallel_agents": self._max_parallel,
        }

    async def stop_coordination(self) -> None:
        self._agents.clear()


@dataclass(slots=True)
class DiscoveredTool:
    name: str
    capability: str
    confidence_score: float


class DynamicToolDiscovery:
    """Deterministic tool discovery engine used in tests."""

    def __init__(self) -> None:
        self._initialized = False
        self._deps: BaseAgentDependencies | None = None

    async def initialize_discovery(self, deps: BaseAgentDependencies) -> None:
        self._deps = deps
        self._initialized = True

    async def discover_tools_for_task(
        self, task_description: str, constraints: dict[str, Any] | None = None
    ) -> list[DiscoveredTool]:
        del task_description, constraints  # deterministic output for tests
        if not self._initialized or self._deps is None:
            raise RuntimeError(
                "DynamicToolDiscovery.initialize_discovery must run first"
            )
        return [
            DiscoveredTool("vector_search", "search", 0.9),
            DiscoveredTool("summarize", "generation", 0.75),
        ]


def choose_coordination_strategy(tasks: list[TaskDefinition]) -> CoordinationStrategy:
    """Heuristic mirroring production behaviour for tests."""

    if len(tasks) <= 1:
        return CoordinationStrategy.SEQUENTIAL

    high_priority = sum(
        1
        for task in tasks
        if task.priority in {TaskPriority.HIGH, TaskPriority.CRITICAL}
    )
    if high_priority >= len(tasks) / 2:
        return CoordinationStrategy.HIERARCHICAL

    dependency_rich = sum(1 for task in tasks if task.dependencies)
    if dependency_rich >= len(tasks) - 1:
        return CoordinationStrategy.PIPELINE

    if all(task.estimated_duration_ms <= 300.0 for task in tasks):
        return CoordinationStrategy.PARALLEL

    return CoordinationStrategy.SEQUENTIAL


class ToolCapability(Enum):
    SEARCH = auto()
    GENERATION = auto()
    ANALYSIS = auto()


class ToolPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass(slots=True)
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    capabilities: set[ToolCapability]
    priority: ToolPriority
    estimated_duration_ms: float
    resource_requirements: dict[str, Any]
    dependencies: list[str]
    success_rate: float
    executor: Callable[[dict[str, Any]], Any]


class AdvancedToolOrchestrator:
    """Minimal asynchronous tool orchestrator for deterministic tests."""

    def __init__(
        self,
        client_manager: ClientManager,
        *,
        max_parallel_executions: int = 1,
    ) -> None:
        self._client_manager = client_manager
        self._max_parallel = max_parallel_executions
        self._tools: dict[str, ToolDefinition] = {}

    async def register_tool(self, definition: ToolDefinition) -> None:
        self._tools[definition.tool_id] = definition

    async def compose_tool_chain(
        self,
        *,
        goal: str,
        constraints: dict[str, Any],
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        del goal  # pragma: no cover - deterministic output
        ordered_tools = sorted(
            self._tools.values(),
            key=lambda d: d.priority.value,
        )
        nodes = [
            {
                "tool_id": tool.tool_id,
                "priority": tool.priority.name,
            }
            for tool in ordered_tools
        ]
        return {
            "nodes": nodes,
            "constraints": constraints,
            "preferences": preferences,
        }

    async def execute_tool_chain(
        self,
        plan: dict[str, Any],
        input_payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del timeout_seconds  # pragma: no cover - deterministic output
        results: dict[str, Any] = {}
        for node in plan.get("nodes", []):
            tool = self._tools[node["tool_id"]]
            outcome = tool.executor(input_payload)
            if hasattr(outcome, "__await__"):
                outcome = await outcome
            results[tool.tool_id] = outcome
        metadata = {
            "total_nodes": len(plan.get("nodes", [])),
            "tools_used": list(results.keys()),
            "max_parallel_executions": self._max_parallel,
        }
        return {
            "success": True,
            "metadata": metadata,
            "results": results,
        }


core_mod.AgentState = AgentState
core_mod.BaseAgentDependencies = BaseAgentDependencies
core_mod.BaseAgent = BaseAgent
core_mod.create_agent_dependencies = create_agent_dependencies

sys.modules["src.services.agents.core"] = core_mod

agentic_orchestrator_mod = cast(
    Any, types.ModuleType("src.services.agents.agentic_orchestrator")
)
agentic_orchestrator_mod.AgenticOrchestrator = AgenticOrchestrator
agentic_orchestrator_mod.AgenticOrchestrationResult = AgenticOrchestrationResult
sys.modules["src.services.agents.agentic_orchestrator"] = agentic_orchestrator_mod

coordination_mod = cast(Any, types.ModuleType("src.services.agents.coordination"))
coordination_mod.CoordinationStrategy = CoordinationStrategy
coordination_mod.ParallelAgentCoordinator = ParallelAgentCoordinator
coordination_mod.TaskDefinition = TaskDefinition
coordination_mod.TaskPriority = TaskPriority
sys.modules["src.services.agents.coordination"] = coordination_mod

dynamic_discovery_mod = cast(
    Any, types.ModuleType("src.services.agents.dynamic_tool_discovery")
)
dynamic_discovery_mod.DynamicToolDiscovery = DynamicToolDiscovery
dynamic_discovery_mod.DiscoveredTool = DiscoveredTool
sys.modules["src.services.agents.dynamic_tool_discovery"] = dynamic_discovery_mod

shared_mod = cast(Any, types.ModuleType("src.services.agents._shared"))
shared_mod.choose_coordination_strategy = choose_coordination_strategy
sys.modules["src.services.agents._shared"] = shared_mod

tool_orchestration_mod = cast(
    Any, types.ModuleType("src.services.agents.tool_orchestration")
)
tool_orchestration_mod.AdvancedToolOrchestrator = AdvancedToolOrchestrator
tool_orchestration_mod.ToolCapability = ToolCapability
tool_orchestration_mod.ToolDefinition = ToolDefinition
tool_orchestration_mod.ToolPriority = ToolPriority
sys.modules["src.services.agents.tool_orchestration"] = tool_orchestration_mod

compat_mod = cast(Any, types.ModuleType("src.services.agents._compat"))


def load_pydantic_ai() -> tuple[bool, Any | None, Any | None]:
    module = sys.modules.get("pydantic_ai")
    if module is None:
        return False, None, None
    return True, getattr(module, "Agent", None), getattr(module, "RunContext", None)


compat_mod.load_pydantic_ai = load_pydantic_ai
sys.modules["src.services.agents._compat"] = compat_mod

# src.services.agents aggregate stub for imports in tests
agents_mod = cast(Any, types.ModuleType("src.services.agents"))
agents_mod.__path__ = []
agents_mod.AgentState = AgentState
agents_mod.BaseAgent = BaseAgent
agents_mod.BaseAgentDependencies = BaseAgentDependencies
agents_mod.QueryOrchestrator = QueryOrchestrator
agents_mod.create_agent_dependencies = create_agent_dependencies
agents_mod.AgenticOrchestrator = AgenticOrchestrator
agents_mod.AgenticOrchestrationResult = AgenticOrchestrationResult
agents_mod.TaskPriority = TaskPriority
agents_mod.CoordinationStrategy = CoordinationStrategy
agents_mod.ParallelAgentCoordinator = ParallelAgentCoordinator
agents_mod.TaskDefinition = TaskDefinition
agents_mod.DynamicToolDiscovery = DynamicToolDiscovery
agents_mod.DiscoveredTool = DiscoveredTool
agents_mod.AdvancedToolOrchestrator = AdvancedToolOrchestrator
agents_mod.ToolCapability = ToolCapability
agents_mod.ToolDefinition = ToolDefinition
agents_mod.ToolPriority = ToolPriority
agents_mod.choose_coordination_strategy = choose_coordination_strategy
agents_mod.load_pydantic_ai = load_pydantic_ai
agents_mod.core = core_mod
agents_mod.agentic_orchestrator = agentic_orchestrator_mod
agents_mod.coordination = coordination_mod
agents_mod.dynamic_tool_discovery = dynamic_discovery_mod
agents_mod.tool_orchestration = tool_orchestration_mod
agents_mod._shared = shared_mod
agents_mod._compat = compat_mod
agents_mod.__all__ = [
    "AgentState",
    "BaseAgent",
    "BaseAgentDependencies",
    "QueryOrchestrator",
    "AgenticOrchestrator",
    "AgenticOrchestrationResult",
    "TaskPriority",
    "CoordinationStrategy",
    "ParallelAgentCoordinator",
    "TaskDefinition",
    "DynamicToolDiscovery",
    "DiscoveredTool",
    "AdvancedToolOrchestrator",
    "ToolCapability",
    "ToolDefinition",
    "ToolPriority",
    "create_agent_dependencies",
    "choose_coordination_strategy",
    "load_pydantic_ai",
]

sys.modules["src.services.agents"] = agents_mod


@pytest.fixture
def mock_dependencies() -> BaseAgentDependencies:
    """Return agent dependencies backed by lightweight stubs."""
    mock_client_manager = Mock(spec=ClientManager)
    config = get_config()
    session_state = AgentState(session_id=str(uuid4()))

    return BaseAgentDependencies(
        client_manager=mock_client_manager,
        config=config,
        session_state=session_state,
    )
