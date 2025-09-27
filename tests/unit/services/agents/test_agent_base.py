"""Deterministic coverage for the shared agent base abstractions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    QueryOrchestrator,
    create_agent_dependencies,
)


class DummyAgent(BaseAgent):
    """Minimal BaseAgent implementation for fallback surface testing."""

    def get_system_prompt(self) -> str:
        """Return a short system prompt for the dummy agent."""
        return "You are a deterministic test agent."

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Record initialization with a predictable side effect."""
        deps.session_state.add_interaction(
            role="system",
            content="initialized",
            metadata={"agent": self.name},
        )


class TestAgentState:
    """Exercise the session state container used by all agents."""

    def test_add_interaction_and_metrics(self) -> None:
        """Ensure interactions, metrics, and tool usage accumulate correctly."""
        state = AgentState(session_id="session-1", user_id="user-7")

        state.add_interaction("user", "Hello", {"channel": "cli"})
        state.update_metrics({"latency": 42.0})
        state.increment_tool_usage("hybrid_search")
        state.increment_tool_usage("hybrid_search")

        assert len(state.conversation_history) == 1
        entry = state.conversation_history[0]
        assert entry["role"] == "user"
        assert entry["content"] == "Hello"
        assert entry["metadata"]["channel"] == "cli"
        assert state.performance_metrics["latency"] == 42.0
        assert state.tool_usage_stats["hybrid_search"] == 2


class TestCreateAgentDependencies:
    """Validate the helper that wires the dependency container."""

    def test_create_dependencies_infers_session(self) -> None:
        """When session ID is omitted a UUID-backed default should be generated."""
        manager = MagicMock(spec=ClientManager)

        deps = create_agent_dependencies(manager, user_id="user-42")
        assert isinstance(deps.session_state, AgentState)
        state = deps.session_state

        assert isinstance(deps, BaseAgentDependencies)
        assert deps.client_manager is manager
        # Pylint cannot follow Pydantic's runtime attributes.
        assert state.user_id == "user-42"  # pylint: disable=no-member
        assert state.session_id  # pylint: disable=no-member


class TestBaseAgentFallback:
    """Cover the fallback execution path used without pydantic-ai."""

    @pytest.mark.asyncio
    async def test_initialize_and_execute_without_pydantic_ai(self) -> None:
        """The agent should initialize and return structured fallback metadata."""
        agent = DummyAgent(name="dummy", model="gpt-4o-mini")
        manager = MagicMock(spec=ClientManager)
        deps = BaseAgentDependencies(
            client_manager=manager,
            config=MagicMock(),
            session_state=AgentState(session_id="session-2", user_id=None),
        )

        await agent.initialize(deps)
        response = await agent.execute("perform task", deps)

        assert agent._initialized is True
        assert response["agent"] == "dummy"
        assert response.get("fallback_used", False) is True
        assert "fallback" in response["result"].lower()


class TestQueryOrchestratorFallback:
    """Smoke test the query orchestrator in fallback execution mode."""

    @pytest.mark.asyncio
    async def test_orchestrate_returns_reasoned_response(self) -> None:
        """Fallback orchestration should produce deterministic tool summaries."""
        orchestrator = QueryOrchestrator()
        manager = MagicMock(spec=ClientManager)
        manager.get_dynamic_tool_discovery = AsyncMock()
        deps = BaseAgentDependencies(
            client_manager=manager,
            config=MagicMock(),
            session_state=AgentState(session_id="session-3", user_id=None),
        )

        await orchestrator.initialize(deps)
        response = await orchestrator.orchestrate_query(
            "Search and summarize docs", collection="documentation"
        )

        assert response["success"] is True
        assert "mock_results" in response["result"]
        assert "orchestration_plan" in response["result"]
        assert response["result"]["fallback_used"] is True
