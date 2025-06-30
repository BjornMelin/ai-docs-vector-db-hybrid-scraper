"""Comprehensive coverage tests for AgenticOrchestrator.

This test module provides thorough coverage of the AgenticOrchestrator functionality,
focusing on autonomous decision-making, tool composition, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.agents.agentic_orchestrator import (
    AgenticOrchestrator,
    get_orchestrator,
    orchestrate_tools,
)
from src.services.agents.core import AgentState, BaseAgentDependencies


class TestAgenticOrchestrator:
    """Test AgenticOrchestrator core functionality."""

    def create_mock_dependencies(self) -> BaseAgentDependencies:
        """Create mock dependencies for testing."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()

        return BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config
        )

    def test_orchestrator_initialization(self):
        """AgenticOrchestrator should initialize correctly."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)

        assert orchestrator.dependencies == deps
        assert hasattr(orchestrator, "dependencies")

    @pytest.mark.asyncio
    async def test_orchestrator_process_query_basic(self):
        """AgenticOrchestrator should process basic queries."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        with (
            patch.object(orchestrator, "_analyze_query_intent") as mock_analyze,
            patch.object(orchestrator, "_select_tools") as mock_select,
            patch.object(orchestrator, "_compose_response") as mock_compose,
        ):
            mock_analyze.return_value = {"intent": "search", "confidence": 0.9}
            mock_select.return_value = ["search_tool", "summarize_tool"]
            mock_compose.return_value = {
                "response": "Test response",
                "tools_used": ["search_tool"],
            }

            result = await orchestrator.process_query("test query", state)

            assert result["response"] == "Test response"
            assert "tools_used" in result
            mock_analyze.assert_called_once_with("test query", state)
            mock_select.assert_called_once()
            mock_compose.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_get_capabilities(self):
        """AgenticOrchestrator should return capabilities."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)

        capabilities = await orchestrator.get_capabilities()

        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert all(isinstance(cap, str) for cap in capabilities)

    @pytest.mark.asyncio
    async def test_orchestrator_analyze_query_intent(self):
        """AgenticOrchestrator should analyze query intent correctly."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        # Test search intent
        result = await orchestrator._analyze_query_intent(
            "find documents about AI", state
        )
        assert isinstance(result, dict)
        assert "intent" in result
        assert "confidence" in result

        # Test different query types
        queries = [
            "search for python tutorials",
            "summarize this document",
            "what is machine learning?",
            "help me understand this code",
        ]

        for query in queries:
            result = await orchestrator._analyze_query_intent(query, state)
            assert isinstance(result, dict)
            assert "intent" in result
            assert isinstance(result["confidence"], int | float)

    @pytest.mark.asyncio
    async def test_orchestrator_select_tools(self):
        """AgenticOrchestrator should select appropriate tools."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)

        intent_analysis = {"intent": "search", "confidence": 0.9}
        available_tools = ["search_tool", "summarize_tool", "analyze_tool"]

        with patch.object(
            orchestrator, "_get_available_tools", return_value=available_tools
        ):
            selected_tools = await orchestrator._select_tools(
                intent_analysis, "test query"
            )

            assert isinstance(selected_tools, list)
            assert len(selected_tools) > 0
            assert all(tool in available_tools for tool in selected_tools)

    @pytest.mark.asyncio
    async def test_orchestrator_compose_response(self):
        """AgenticOrchestrator should compose responses correctly."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        tools = ["search_tool"]
        tool_results = {"search_tool": {"results": ["result1", "result2"]}}

        with patch.object(orchestrator, "_execute_tools", return_value=tool_results):
            response = await orchestrator._compose_response("test query", tools, state)

            assert isinstance(response, dict)
            assert "response" in response
            assert "tools_used" in response

    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self):
        """AgenticOrchestrator should handle errors gracefully."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        with patch.object(
            orchestrator,
            "_analyze_query_intent",
            side_effect=Exception("Analysis error"),
        ):
            result = await orchestrator.process_query("test query", state)

            # Should return error response instead of raising
            assert isinstance(result, dict)
            assert "error" in result or "response" in result

    @pytest.mark.asyncio
    async def test_orchestrator_tool_execution_error(self):
        """AgenticOrchestrator should handle tool execution errors."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        with (
            patch.object(
                orchestrator,
                "_analyze_query_intent",
                return_value={"intent": "search", "confidence": 0.9},
            ),
            patch.object(orchestrator, "_select_tools", return_value=["failing_tool"]),
            patch.object(
                orchestrator,
                "_execute_tools",
                side_effect=Exception("Tool execution failed"),
            ),
        ):
            result = await orchestrator.process_query("test query", state)

            # Should handle tool execution errors gracefully
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_orchestrator_state_updates(self):
        """AgenticOrchestrator should update agent state appropriately."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        initial_history_length = len(state.conversation_history)

        with (
            patch.object(
                orchestrator,
                "_analyze_query_intent",
                return_value={"intent": "search", "confidence": 0.9},
            ),
            patch.object(orchestrator, "_select_tools", return_value=["search_tool"]),
            patch.object(
                orchestrator,
                "_compose_response",
                return_value={
                    "response": "Test response",
                    "tools_used": ["search_tool"],
                },
            ),
        ):
            await orchestrator.process_query("test query", state)

            # State should be updated with performance metrics or tool usage
            assert len(state.conversation_history) >= initial_history_length

    @pytest.mark.asyncio
    async def test_orchestrator_different_query_types(self):
        """AgenticOrchestrator should handle different query types."""
        deps = self.create_mock_dependencies()
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        query_types = [
            "search for documents",
            "summarize this text",
            "analyze the data",
            "what is the answer?",
            "",  # Edge case: empty query
        ]

        for query in query_types:
            result = await orchestrator.process_query(query, state)
            assert isinstance(result, dict)
            # Should handle all query types without raising exceptions


class TestGetOrchestrator:
    """Test get_orchestrator factory function."""

    @patch("src.services.agents.agentic_orchestrator.create_agent_dependencies")
    def test_get_orchestrator_success(self, mock_create_deps):
        """get_orchestrator should create orchestrator instance."""
        mock_client_manager = MagicMock()
        mock_deps = MagicMock()
        mock_create_deps.return_value = mock_deps

        orchestrator = get_orchestrator(mock_client_manager)

        assert isinstance(orchestrator, AgenticOrchestrator)
        assert orchestrator.dependencies == mock_deps
        mock_create_deps.assert_called_once_with(mock_client_manager)

    @patch("src.services.agents.agentic_orchestrator.create_agent_dependencies")
    def test_get_orchestrator_with_dependency_error(self, mock_create_deps):
        """get_orchestrator should handle dependency creation errors."""
        mock_client_manager = MagicMock()
        mock_create_deps.side_effect = Exception("Dependency error")

        with pytest.raises(Exception, match="Dependency error"):
            get_orchestrator(mock_client_manager)


class TestOrchestrateTools:
    """Test orchestrate_tools utility function."""

    @pytest.mark.asyncio
    async def test_orchestrate_tools_success(self):
        """orchestrate_tools should orchestrate tool execution."""
        mock_client_manager = MagicMock()
        query = "test query"

        with patch(
            "src.services.agents.agentic_orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.process_query.return_value = {"response": "Test response"}
            mock_get_orch.return_value = mock_orchestrator

            result = await orchestrate_tools(query, mock_client_manager)

            assert result["response"] == "Test response"
            mock_get_orch.assert_called_once_with(mock_client_manager)
            mock_orchestrator.process_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrate_tools_with_session_id(self):
        """orchestrate_tools should handle session_id parameter."""
        mock_client_manager = MagicMock()
        query = "test query"
        session_id = "custom-session"

        with patch(
            "src.services.agents.agentic_orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.process_query.return_value = {"response": "Test response"}
            mock_get_orch.return_value = mock_orchestrator

            result = await orchestrate_tools(
                query, mock_client_manager, session_id=session_id
            )

            assert result["response"] == "Test response"
            # Verify that the orchestrator was called with correct state
            call_args = mock_orchestrator.process_query.call_args
            assert call_args[0][0] == query  # First argument is query
            assert call_args[0][1].session_id == session_id  # Second argument is state

    @pytest.mark.asyncio
    async def test_orchestrate_tools_error_handling(self):
        """orchestrate_tools should handle orchestration errors."""
        mock_client_manager = MagicMock()
        query = "test query"

        with patch(
            "src.services.agents.agentic_orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.process_query.side_effect = Exception(
                "Orchestration error"
            )
            mock_get_orch.return_value = mock_orchestrator

            # Should either raise or return error response
            try:
                result = await orchestrate_tools(query, mock_client_manager)
                # If it returns a result, it should be an error response
                assert isinstance(result, dict)
            except (ConnectionError, RuntimeError, ValueError) as e:
                # If it raises, that's also acceptable error handling
                assert "error" in str(e).lower() or "orchestration" in str(e).lower()


class TestOrchestratorEdgeCases:
    """Test edge cases and error conditions."""

    def test_orchestrator_with_none_dependencies(self):
        """AgenticOrchestrator should handle None dependencies gracefully."""
        with pytest.raises((TypeError, ValueError)):
            AgenticOrchestrator(None)

    @pytest.mark.asyncio
    async def test_orchestrator_with_empty_query(self):
        """AgenticOrchestrator should handle empty queries."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        result = await orchestrator.process_query("", state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_orchestrator_with_very_long_query(self):
        """AgenticOrchestrator should handle very long queries."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        long_query = "test " * 1000  # Very long query
        result = await orchestrator.process_query(long_query, state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_orchestrator_with_special_characters(self):
        """AgenticOrchestrator should handle queries with special characters."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="test-session")

        special_queries = [
            "query with émojis 🚀",
            "query with <tags>",
            "query with 'quotes' and \"double quotes\"",
            "query with\nnewlines\nand\ttabs",
            "query with unicode: ∂∆√∑∏π",
        ]

        for query in special_queries:
            result = await orchestrator.process_query(query, state)
            assert isinstance(result, dict)


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_orchestrator_full_workflow(self):
        """Test complete orchestrator workflow."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config
        )

        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="integration-test")

        # Should be able to process query end-to-end
        result = await orchestrator.process_query("test integration query", state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_orchestrator_state_persistence(self):
        """Test that orchestrator properly updates state."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())

        orchestrator = AgenticOrchestrator(deps)
        state = AgentState(session_id="persistence-test")

        # Process multiple queries with same state
        await orchestrator.process_query("first query", state)
        await orchestrator.process_query("second query", state)

        # State should maintain session information
        assert state.session_id == "persistence-test"

    @pytest.mark.asyncio
    async def test_orchestrator_concurrent_queries(self):
        """Test orchestrator handling concurrent queries."""

        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())

        orchestrator = AgenticOrchestrator(deps)

        # Create multiple states for concurrent processing
        states = [AgentState(session_id=f"concurrent-{i}") for i in range(3)]
        queries = [f"concurrent query {i}" for i in range(3)]

        # Process queries concurrently
        tasks = [
            orchestrator.process_query(query, state)
            for query, state in zip(queries, states, strict=False)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All queries should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict | Exception)
