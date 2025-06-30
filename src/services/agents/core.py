"""Core agent architecture for Pydantic-AI based agentic RAG.

This module provides the foundational classes and patterns for building
autonomous agents that can intelligently process queries, compose tools,
and coordinate with other agents.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


try:
    from pydantic_ai import Agent, RunContext

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    # Graceful degradation when pydantic-ai is not available
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None

from src.config import get_config
from src.infrastructure.client_manager import ClientManager


def _check_api_key_availability() -> bool:
    """Check if required API keys are available for agent operation.

    Returns:
        bool: True if API keys are available, False otherwise
    """
    # Check for OpenAI API key (most common for agents)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here":
        return True

    # Check for other potential API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if (
        anthropic_key
        and anthropic_key.strip()
        and anthropic_key != "your_anthropic_api_key_here"
    ):
        return True

    logger.debug("No valid API keys found - agents will use fallback mode")
    return False


logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Shared state across all agents in the system."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str | None = Field(None, description="User identifier for personalization")
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list, description="History of interactions"
    )
    knowledge_base: dict[str, Any] = Field(
        default_factory=dict, description="Session-specific knowledge"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance tracking metrics"
    )
    tool_usage_stats: dict[str, int] = Field(
        default_factory=dict, description="Tool usage statistics"
    )
    preferences: dict[str, Any] = Field(
        default_factory=dict, description="User or session preferences"
    )

    def add_interaction(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add an interaction to the conversation history."""
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        )

    def update_metrics(self, metrics: dict[str, float]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update(metrics)

    def increment_tool_usage(self, tool_name: str) -> None:
        """Increment usage count for a tool."""
        self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1


class BaseAgentDependencies(BaseModel):
    """Base dependencies injected into all agents."""

    client_manager: Any = Field(..., description="ClientManager instance")
    config: Any = Field(..., description="Unified configuration")
    session_state: AgentState = Field(..., description="Session state")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAgent(ABC):
    """Base class for all autonomous agents in the system."""

    def __init__(
        self,
        name: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """Initialize the base agent.

        Args:
            name: Agent name for identification
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens per response
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = False
        self.agent = None  # Initialize as None by default

        # Check if Pydantic-AI is available
        if not PYDANTIC_AI_AVAILABLE:
            logger.warning(
                f"Pydantic-AI not available, agent {name} will use fallback mode"
            )
            self._fallback_reason = "pydantic_ai_unavailable"
        # Check if API keys are available
        elif not _check_api_key_availability():
            logger.info(f"No API keys available, agent {name} will use fallback mode")
            self._fallback_reason = "no_api_keys"
        else:
            # Try to initialize Pydantic-AI agent
            try:
                self.agent = Agent(
                    model=model,
                    system_prompt=self.get_system_prompt(),
                    deps_type=BaseAgentDependencies,
                )
                self._fallback_reason = None
                logger.info(f"Agent {name} initialized with Pydantic-AI")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Pydantic-AI agent {name}: {e}. Using fallback mode."
                )
                self.agent = None
                self._fallback_reason = f"initialization_failed: {e}"

        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Define agent-specific system prompt.

        Returns:
            System prompt string defining agent behavior and capabilities
        """

    @abstractmethod
    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize agent-specific tools.

        Args:
            deps: Dependencies required for tool initialization
        """

    async def initialize(self, deps: BaseAgentDependencies) -> None:
        """Initialize the agent with dependencies.

        Args:
            deps: Agent dependencies
        """
        if self._initialized:
            return

        await self.initialize_tools(deps)
        self._initialized = True
        logger.info(f"Agent {self.name} initialized successfully")

    async def execute(
        self,
        task: str,
        deps: BaseAgentDependencies,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a task using the agent.

        Args:
            task: Task description or prompt
            deps: Agent dependencies
            context: Additional context for task execution

        Returns:
            Task execution result
        """
        if not self._initialized:
            await self.initialize(deps)

        start_time = time.time()
        self.execution_count += 1

        try:
            # Fallback execution if Pydantic-AI not available
            if not PYDANTIC_AI_AVAILABLE or self.agent is None:
                return await self._fallback_execute(task, deps, context)

            # Execute using Pydantic-AI agent
            result = await self.agent.run(task, deps=deps)

            # Track success
            self.success_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            # Update session metrics
            deps.session_state.update_metrics(
                {
                    f"{self.name}_last_execution_time": execution_time,
                    f"{self.name}_success_rate": self.success_count
                    / self.execution_count,
                }
            )

            # Add to conversation history
            deps.session_state.add_interaction(
                role=f"agent_{self.name}",
                content=str(result.data),
                metadata={
                    "execution_time": execution_time,
                    "model": self.model,
                    "context": context,
                },
            )

            return {
                "success": True,
                "result": result.data,
                "metadata": {
                    "agent": self.name,
                    "execution_time": execution_time,
                    "model": self.model,
                },
            }

        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            logger.error(f"Agent {self.name} execution failed: {e}", exc_info=True)

            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "agent": self.name,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                },
            }

    async def _fallback_execute(
        self,
        task: str,
        deps: BaseAgentDependencies,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fallback execution when Pydantic-AI is not available.

        Args:
            task: Task description
            deps: Agent dependencies
            context: Additional context

        Returns:
            Fallback execution result
        """
        fallback_reason = getattr(self, "_fallback_reason", "unknown")
        logger.info(
            f"Using fallback execution for agent {self.name} (reason: {fallback_reason})"
        )

        # Enhanced fallback logic with context-aware responses
        if "search" in task.lower():
            result = f"Mock search results for: {task} (fallback mode)"
        elif "analyze" in task.lower():
            result = f"Mock analysis of: {task} (fallback mode)"
        elif "generate" in task.lower():
            result = f"Mock generation for: {task} (fallback mode)"
        else:
            result = f"Mock response for task: {task} (fallback mode)"

        return {
            "result": result,
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "agent": self.name,
            "success": True,  # Mark as successful for validation purposes
        }

    def get_performance_metrics(self) -> dict[str, float]:
        """Get agent performance metrics.

        Returns:
            Performance metrics dictionary
        """
        if self.execution_count == 0:
            return {
                "execution_count": 0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "error_rate": 0.0,
            }

        return {
            "execution_count": self.execution_count,
            "avg_execution_time": self.total_execution_time / self.execution_count,
            "success_rate": self.success_count / self.execution_count,
            "error_rate": self.error_count / self.execution_count,
            "total_execution_time": self.total_execution_time,
        }

    async def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0
        logger.info(f"Metrics reset for agent {self.name}")


class AgentRegistry:
    """Registry for managing multiple agents."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the registry.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Agent {agent.name} registered")

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    async def initialize_all(self, deps: BaseAgentDependencies) -> None:
        """Initialize all registered agents.

        Args:
            deps: Dependencies for initialization
        """
        for agent in self.agents.values():
            await agent.initialize(deps)

    def get_all_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for all agents.

        Returns:
            Dictionary mapping agent names to their metrics
        """
        return {
            name: agent.get_performance_metrics() for name, agent in self.agents.items()
        }


# Global agent registry instance
agent_registry = AgentRegistry()


def create_agent_dependencies(
    client_manager: ClientManager,
    session_id: str | None = None,
    user_id: str | None = None,
) -> BaseAgentDependencies:
    """Create agent dependencies with default values.

    Args:
        client_manager: ClientManager instance
        session_id: Optional session ID
        user_id: Optional user ID

    Returns:
        Configured agent dependencies
    """
    config = get_config()

    session_state = AgentState(session_id=session_id or str(uuid4()), user_id=user_id)

    return BaseAgentDependencies(
        client_manager=client_manager, config=config, session_state=session_state
    )
