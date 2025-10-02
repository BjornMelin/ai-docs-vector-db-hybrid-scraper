"""Core agent architecture for a Pydantic-AI based agent RAG system."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


try:
    from pydantic_ai import Agent as PydanticAgent  # type: ignore[import-untyped]
except ImportError:
    PydanticAgent = None  # type: ignore[misc,assignment]

from src.config import get_config
from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AgentConfig:
    """Configuration for an agent."""

    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclasses.dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    execution_count: int = 0
    total_execution_time: float = 0.0
    success_count: int = 0
    error_count: int = 0


def _check_api_key_availability() -> bool:
    """Return True if a supported API key is available."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here":
        return True

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if (
        anthropic_key
        and anthropic_key.strip()
        and anthropic_key != "your_anthropic_api_key_here"
    ):
        return True

    logger.debug("No valid API keys found; agents will use fallback mode")
    return False


class AgentState(BaseModel):
    """Shared state across all agents in the system."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str | None = Field(None, description="User identifier")
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_base: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, float] = Field(default_factory=dict)
    tool_usage_stats: dict[str, int] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)

    def add_interaction(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record an interaction without mutating in place."""
        entry = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.conversation_history = [*self.conversation_history, entry]

    def update_metrics(self, metrics: dict[str, float]) -> None:
        """Merge metrics without dict.update to satisfy static analyzers."""
        self.performance_metrics = {**self.performance_metrics, **metrics}

    def increment_tool_usage(self, tool_name: str) -> None:
        """Increment usage counter without dict.get to satisfy E1101."""
        base = 0
        if tool_name in self.tool_usage_stats:
            base = self.tool_usage_stats[tool_name]
        self.tool_usage_stats = {**self.tool_usage_stats, tool_name: base + 1}


class BaseAgentDependencies(BaseModel):
    """Base dependencies injected into all agents."""

    client_manager: Any = Field(..., description="ClientManager instance")
    config: Any = Field(..., description="Unified configuration")
    session_state: AgentState = Field(..., description="Session state")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> None:
        self.name = name
        self.config = AgentConfig(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
        self._initialized = False
        self.agent = None

        # Fallback reasoning
        self._fallback_reason: str | None = None
        try:
            if PydanticAgent is not None and _check_api_key_availability():
                self.agent = PydanticAgent(
                    model=self.config.model,
                    system_prompt=self.get_system_prompt(),
                    deps_type=BaseAgentDependencies,
                )
            else:
                self._fallback_reason = "no_api_keys"
        except Exception as exc:  # pragma: no cover - optional dep
            self._fallback_reason = f"pydantic_ai_unavailable: {exc}"
        # Metrics
        self.metrics = AgentMetrics()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""

    @abstractmethod
    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize agent-specific tools."""

    async def initialize(self, deps: BaseAgentDependencies) -> None:
        """Initialize the agent with dependencies."""
        if self._initialized:
            return
        await self.initialize_tools(deps)
        self._initialized = True
        logger.info("Agent %s initialized successfully", self.name)

    @property
    def is_initialized(self) -> bool:
        """True if the agent has been initialized."""
        return self._initialized

    async def execute(
        self,
        task: str,
        deps: BaseAgentDependencies,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a task using the agent with graceful fallback."""
        if not self._initialized:
            await self.initialize(deps)

        start_time = time.time()
        self.metrics.execution_count += 1

        try:
            if self.agent is None:
                return await self._fallback_execute(task, deps, context)

            result = await self.agent.run(task, deps=deps)  # type: ignore[attr-defined]

            self.metrics.success_count += 1
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time

            deps.session_state.update_metrics(
                {
                    f"{self.name}_last_execution_time": execution_time,
                    f"{self.name}_success_rate": self.metrics.success_count
                    / self.metrics.execution_count,
                }
            )
            deps.session_state.add_interaction(
                role=f"agent_{self.name}",
                content=str(result.data),
                metadata={
                    "execution_time": execution_time,
                    "model": self.config.model,
                    "context": context,
                },
            )

            return {
                "success": True,
                "result": result.data,
                "metadata": {
                    "agent": self.name,
                    "execution_time": execution_time,
                    "model": self.config.model,
                },
            }
        except Exception as exc:  # pragma: no cover - safety
            self.metrics.error_count += 1
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            logger.exception("Agent %s execution failed", self.name)
            return {
                "success": False,
                "error": str(exc),
                "metadata": {
                    "agent": self.name,
                    "execution_time": execution_time,
                    "error_type": type(exc).__name__,
                },
            }

    async def _fallback_execute(
        self,
        task: str,
        _deps: BaseAgentDependencies,
        _context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deterministic fallback behavior."""
        reason = self._fallback_reason or "unknown"
        tl = task.lower()
        if "search" in tl:
            out = f"Mock search results for: {task} (fallback)"
        elif "analyze" in tl:
            out = f"Mock analysis of: {task} (fallback)"
        elif "generate" in tl:
            out = f"Mock generation for: {task} (fallback)"
        else:
            out = f"Mock response for task: {task} (fallback)"
        return {
            "result": out,
            "fallback_used": True,
            "fallback_reason": reason,
            "agent": self.name,
            "success": True,
        }

    def get_performance_metrics(self) -> dict[str, float]:
        """Return performance metrics."""
        if self.metrics.execution_count == 0:
            return {
                "execution_count": 0.0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "error_rate": 0.0,
            }
        return {
            "execution_count": float(self.metrics.execution_count),
            "avg_execution_time": (
                self.metrics.total_execution_time / self.metrics.execution_count
            ),
            "success_rate": self.metrics.success_count / self.metrics.execution_count,
            "error_rate": self.metrics.error_count / self.metrics.execution_count,
            "total_execution_time": self.metrics.total_execution_time,
        }

    async def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = AgentMetrics()
        logger.info("Metrics reset for agent %s", self.name)


class AgentRegistry:
    """Registry for managing multiple agents."""

    def __init__(self) -> None:
        self.agents: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self.agents[agent.name] = agent
        logger.info("Agent %s registered", agent.name)

    def get_agent(self, name: str) -> BaseAgent | None:
        """Return an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """Return names of registered agents."""
        return list(self.agents.keys())

    async def initialize_all(self, deps: BaseAgentDependencies) -> None:
        """Initialize all agents."""
        for agent in self.agents.values():
            await agent.initialize(deps)

    def get_all_metrics(self) -> dict[str, dict[str, float]]:
        """Collect metrics from all agents."""
        return {
            name: agent.get_performance_metrics() for name, agent in self.agents.items()
        }


# Global registry
agent_registry = AgentRegistry()


def create_agent_dependencies(
    client_manager: ClientManager,
    session_id: str | None = None,
    user_id: str | None = None,
) -> BaseAgentDependencies:
    """Create a ready-to-use dependency container."""
    config = get_config()
    session = AgentState(session_id=session_id or str(uuid4()), user_id=user_id)
    return BaseAgentDependencies(
        client_manager=client_manager, config=config, session_state=session
    )
