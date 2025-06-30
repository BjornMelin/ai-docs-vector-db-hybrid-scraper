"""Comprehensive coverage tests for agentic core module.

This test module provides thorough coverage of the agentic core functionality,
focusing on behavior validation rather than implementation details.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.agents.core import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    create_agent_dependencies,
    PYDANTIC_AI_AVAILABLE,
)


class TestAgentState:
    """Test AgentState model behavior."""

    def test_agent_state_initialization_with_required_fields(self):
        """Agent state should initialize with required session_id."""
        session_id = "test-session-123"
        state = AgentState(session_id=session_id)

        assert state.session_id == session_id
        assert state.user_id is None
        assert state.conversation_history == []
        assert state.knowledge_base == {}
        assert state.performance_metrics == {}
        assert state.tool_usage_stats == {}

    def test_agent_state_with_optional_fields(self):
        """Agent state should handle optional fields correctly."""
        state = AgentState(
            session_id="test-session",
            user_id="user-123",
            conversation_history=[{"role": "user", "content": "hello"}],
            knowledge_base={"key": "value"},
            performance_metrics={"response_time": 0.5},
            tool_usage_stats={"search": 3},
        )

        assert state.user_id == "user-123"
        assert len(state.conversation_history) == 1
        assert state.knowledge_base["key"] == "value"
        assert state.performance_metrics["response_time"] == 0.5
        assert state.tool_usage_stats["search"] == 3

    def test_agent_state_metrics_operations(self):
        """Agent state should support metrics operations."""
        state = AgentState(session_id="test")

        # Test adding performance metrics
        state.performance_metrics["latency"] = 100.0
        state.performance_metrics["accuracy"] = 0.95

        assert state.performance_metrics["latency"] == 100.0
        assert state.performance_metrics["accuracy"] == 0.95

    def test_agent_state_conversation_history_operations(self):
        """Agent state should support conversation history operations."""
        state = AgentState(session_id="test")

        # Add conversation entries
        state.conversation_history.append({"role": "user", "content": "query"})
        state.conversation_history.append({"role": "assistant", "content": "response"})

        assert len(state.conversation_history) == 2
        assert state.conversation_history[0]["role"] == "user"
        assert state.conversation_history[1]["role"] == "assistant"


class TestBaseAgentDependencies:
    """Test BaseAgentDependencies model behavior."""

    def test_base_agent_dependencies_initialization(self):
        """BaseAgentDependencies should initialize with required fields."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config
        )

        assert deps.client_manager == mock_client_manager
        assert deps.config == mock_config

    def test_base_agent_dependencies_with_optional_fields(self):
        """BaseAgentDependencies should handle optional fields."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()
        mock_logger = MagicMock()

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, logger=mock_logger
        )

        assert deps.logger == mock_logger


class TestBaseAgent:
    """Test BaseAgent abstract class behavior."""

    def test_base_agent_is_abstract(self):
        """BaseAgent should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseAgent()

    def test_base_agent_subclass_must_implement_abstract_methods(self):
        """BaseAgent subclasses must implement abstract methods."""

        class IncompleteAgent(BaseAgent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_base_agent_concrete_implementation(self):
        """BaseAgent can be subclassed with concrete implementations."""

        class ConcreteAgent(BaseAgent):
            async def process_query(self, query: str, state: AgentState) -> dict:
                return {"response": f"Processed: {query}"}

            async def get_capabilities(self) -> list[str]:
                return ["test_capability"]

        # Should be able to instantiate concrete implementation
        agent = ConcreteAgent()
        assert agent is not None

    @pytest.mark.asyncio
    async def test_base_agent_interface_methods(self):
        """BaseAgent interface methods should work correctly."""

        class TestAgent(BaseAgent):
            async def process_query(self, query: str, state: AgentState) -> dict:
                return {"query": query, "session": state.session_id}

            async def get_capabilities(self) -> list[str]:
                return ["search", "analyze"]

        agent = TestAgent()
        state = AgentState(session_id="test-session")

        # Test process_query
        result = await agent.process_query("test query", state)
        assert result["query"] == "test query"
        assert result["session"] == "test-session"

        # Test get_capabilities
        capabilities = await agent.get_capabilities()
        assert "search" in capabilities
        assert "analyze" in capabilities


class TestCreateAgentDependencies:
    """Test create_agent_dependencies factory function."""

    @patch("src.services.agents.core.get_config")
    def test_create_agent_dependencies_success(self, mock_get_config):
        """create_agent_dependencies should create valid dependencies."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        deps = create_agent_dependencies(mock_client_manager)

        assert deps.client_manager == mock_client_manager
        assert deps.config == mock_config
        mock_get_config.assert_called_once()

    @patch("src.services.agents.core.get_config")
    def test_create_agent_dependencies_with_config_error(self, mock_get_config):
        """create_agent_dependencies should handle config errors gracefully."""
        mock_client_manager = MagicMock()
        mock_get_config.side_effect = Exception("Config error")

        with pytest.raises(Exception, match="Config error"):
            create_agent_dependencies(mock_client_manager)

    @patch("src.services.agents.core.get_config")
    def test_create_agent_dependencies_type_validation(self, mock_get_config):
        """create_agent_dependencies should validate input types."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        # Test with valid client_manager
        mock_client_manager = MagicMock()
        deps = create_agent_dependencies(mock_client_manager)
        assert deps is not None

        # Test with None client_manager should fail validation
        with pytest.raises((TypeError, ValueError)):
            create_agent_dependencies(None)


class TestPydanticAIIntegration:
    """Test Pydantic-AI integration behavior."""

    def test_pydantic_ai_availability_flag(self):
        """PYDANTIC_AI_AVAILABLE should reflect actual availability."""
        assert isinstance(PYDANTIC_AI_AVAILABLE, bool)

        if PYDANTIC_AI_AVAILABLE:
            # If available, Agent and RunContext should be imported
            from src.services.agents.core import Agent, RunContext

            assert Agent is not None
            assert RunContext is not None
        else:
            # If not available, they should be None
            from src.services.agents.core import Agent, RunContext

            assert Agent is None
            assert RunContext is None

    @pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not available")
    def test_pydantic_ai_agent_integration(self):
        """Test Pydantic-AI Agent integration when available."""
        from src.services.agents.core import Agent

        # Should be able to reference Agent class
        assert Agent is not None
        assert hasattr(Agent, "__name__")

    @pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not available")
    def test_pydantic_ai_run_context_integration(self):
        """Test Pydantic-AI RunContext integration when available."""
        from src.services.agents.core import RunContext

        # Should be able to reference RunContext class
        assert RunContext is not None
        assert hasattr(RunContext, "__name__")


class TestAgentStateEdgeCases:
    """Test edge cases and error conditions for AgentState."""

    def test_agent_state_empty_session_id(self):
        """Agent state should handle empty session_id."""
        state = AgentState(session_id="")
        assert state.session_id == ""

    def test_agent_state_large_conversation_history(self):
        """Agent state should handle large conversation history."""
        large_history = [
            {"role": "user", "content": f"message {i}"} for i in range(1000)
        ]
        state = AgentState(session_id="test", conversation_history=large_history)

        assert len(state.conversation_history) == 1000
        assert state.conversation_history[0]["content"] == "message 0"
        assert state.conversation_history[999]["content"] == "message 999"

    def test_agent_state_complex_knowledge_base(self):
        """Agent state should handle complex knowledge base structures."""
        complex_kb = {
            "nested": {"deep": {"value": "test"}},
            "list": [1, 2, 3],
            "mixed": {"numbers": [1, 2], "text": "hello"},
        }

        state = AgentState(session_id="test", knowledge_base=complex_kb)

        assert state.knowledge_base["nested"]["deep"]["value"] == "test"
        assert state.knowledge_base["list"] == [1, 2, 3]
        assert state.knowledge_base["mixed"]["numbers"] == [1, 2]

    def test_agent_state_metrics_with_various_types(self):
        """Agent state should handle various metric value types."""
        state = AgentState(session_id="test")

        # Add various metric types
        state.performance_metrics.update(
            {
                "float_metric": 3.14,
                "int_metric": 42,
                "zero_metric": 0.0,
                "negative_metric": -1.5,
            }
        )

        assert state.performance_metrics["float_metric"] == 3.14
        assert state.performance_metrics["int_metric"] == 42
        assert state.performance_metrics["zero_metric"] == 0.0
        assert state.performance_metrics["negative_metric"] == -1.5


@pytest.mark.integration
class TestAgentCoreIntegration:
    """Integration tests for agent core functionality."""

    @patch("src.services.agents.core.get_config")
    def test_full_agent_lifecycle(self, mock_get_config):
        """Test complete agent lifecycle with dependencies."""
        # Setup mocks
        mock_config = MagicMock()
        mock_client_manager = MagicMock()
        mock_get_config.return_value = mock_config

        # Create dependencies
        deps = create_agent_dependencies(mock_client_manager)

        # Create agent state
        state = AgentState(session_id="integration-test")

        # Verify integration
        assert deps.client_manager == mock_client_manager
        assert deps.config == mock_config
        assert state.session_id == "integration-test"

    @pytest.mark.asyncio
    async def test_agent_state_with_real_workflow(self):
        """Test agent state through a realistic workflow."""
        state = AgentState(session_id="workflow-test")

        # Simulate conversation
        state.conversation_history.append({"role": "user", "content": "Hello"})
        state.conversation_history.append(
            {"role": "assistant", "content": "Hi! How can I help?"}
        )

        # Simulate tool usage
        state.tool_usage_stats["search"] = 1
        state.tool_usage_stats["summarize"] = 2

        # Simulate performance tracking
        state.performance_metrics["response_time"] = 0.5
        state.performance_metrics["accuracy"] = 0.95

        # Add knowledge
        state.knowledge_base["user_preference"] = "detailed answers"

        # Verify workflow state
        assert len(state.conversation_history) == 2
        assert state.tool_usage_stats["search"] == 1
        assert state.tool_usage_stats["summarize"] == 2
        assert state.performance_metrics["response_time"] == 0.5
        assert state.knowledge_base["user_preference"] == "detailed answers"
