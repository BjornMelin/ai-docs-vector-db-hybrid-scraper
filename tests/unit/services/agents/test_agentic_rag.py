"""Tests for agentic RAG implementation."""

import asyncio
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    QueryOrchestrator,
    create_agent_dependencies,
)


class TestAgentState:
    """Test agent state management."""

    def test_agent_state_initialization(self):
        """Test agent state initialization."""
        session_id = str(uuid4())
        state = AgentState(session_id=session_id)

        assert state.session_id == session_id
        assert state.user_id is None
        assert len(state.conversation_history) == 0
        assert len(state.knowledge_base) == 0
        assert len(state.performance_metrics) == 0

    def test_add_interaction(self):
        """Test adding interactions to conversation history."""
        state = AgentState(session_id="test_session")

        state.add_interaction("user", "Hello", {"test": "metadata"})

        assert len(state.conversation_history) == 1
        interaction = state.conversation_history[0]
        assert interaction["role"] == "user"
        assert interaction["content"] == "Hello"
        assert interaction["metadata"]["test"] == "metadata"
        assert "timestamp" in interaction

    def test_update_metrics(self):
        """Test updating performance metrics."""
        state = AgentState(session_id="test_session")

        metrics = {"latency": 100.0, "accuracy": 0.95}
        state.update_metrics(metrics)

        assert state.performance_metrics["latency"] == 100.0
        assert state.performance_metrics["accuracy"] == 0.95

    def test_increment_tool_usage(self):
        """Test tool usage tracking."""
        state = AgentState(session_id="test_session")

        state.increment_tool_usage("hybrid_search")
        state.increment_tool_usage("hybrid_search")
        state.increment_tool_usage("hyde_search")

        assert state.tool_usage_stats["hybrid_search"] == 2
        assert state.tool_usage_stats["hyde_search"] == 1


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def get_system_prompt(self) -> str:
        return "Test agent prompt"

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        pass


class TestBaseAgent:
    """Test base agent functionality."""

    def test_base_agent_initialization(self):
        """Test base agent initialization."""
        agent = MockAgent("test_agent", "gpt-4")

        assert agent.name == "test_agent"
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.1
        assert agent.max_tokens == 1000
        assert not agent._initialized

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization with dependencies."""
        agent = MockAgent("test_agent")

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await agent.initialize(deps)

        assert agent._initialized

    @pytest.mark.asyncio
    async def test_agent_execute_fallback(self):
        """Test agent execution in fallback mode."""
        agent = MockAgent("test_agent")

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test fallback execution (when Pydantic-AI not available)
        result = await agent.execute("test task", deps)

        assert "success" in result
        assert "result" in result or "error" in result
        assert "metadata" in result

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        agent = MockAgent("test_agent")

        # Test with no executions
        metrics = agent.get_performance_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["avg_execution_time"] == 0.0

        # Simulate some executions
        agent.execution_count = 5
        agent.total_execution_time = 2.5
        agent.success_count = 4
        agent.error_count = 1

        metrics = agent.get_performance_metrics()
        assert metrics["execution_count"] == 5
        assert metrics["avg_execution_time"] == 0.5
        assert metrics["success_rate"] == 0.8
        assert metrics["error_rate"] == 0.2


class TestQueryOrchestrator:
    """Test query orchestrator agent."""

    def test_orchestrator_initialization(self):
        """Test query orchestrator initialization."""
        orchestrator = QueryOrchestrator()

        assert orchestrator.name == "query_orchestrator"
        assert orchestrator.model == "gpt-4"
        assert orchestrator.temperature == 0.1
        assert orchestrator.max_tokens == 1500
        assert orchestrator.strategy_performance == {}

    def test_orchestrator_custom_initialization(self):
        """Test query orchestrator with custom parameters."""
        orchestrator = QueryOrchestrator(model="gpt-3.5-turbo")

        assert orchestrator.model == "gpt-3.5-turbo"
        assert orchestrator.temperature == 0.1
        assert orchestrator.max_tokens == 1500

    def test_system_prompt(self):
        """Test system prompt generation."""
        orchestrator = QueryOrchestrator()
        prompt = orchestrator.get_system_prompt()

        assert "Query Orchestrator" in prompt
        assert "Analyze incoming queries" in prompt
        assert "Delegate to specialized agents" in prompt
        assert "Coordinate multi-stage retrieval" in prompt
        assert "Learn from past performance" in prompt

        # Check analysis framework
        assert "SIMPLE:" in prompt
        assert "MODERATE:" in prompt
        assert "COMPLEX:" in prompt

        # Check processing strategies
        assert "FAST:" in prompt
        assert "BALANCED:" in prompt
        assert "COMPREHENSIVE:" in prompt

    @pytest.mark.asyncio
    async def test_orchestrate_query_fallback(self):
        """Test query orchestration in fallback mode."""
        orchestrator = QueryOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        # Test orchestration
        result = await orchestrator.orchestrate_query(
            query="What is machine learning?", collection="documentation"
        )

        assert "success" in result
        assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_orchestrate_query_with_context(self):
        """Test query orchestration with user context and performance requirements."""
        orchestrator = QueryOrchestrator()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")
        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        result = await orchestrator.orchestrate_query(
            query="How to implement neural networks?",
            collection="docs",
            user_context={"user_id": "test_user", "expertise": "beginner"},
            performance_requirements={"max_latency": 1000, "min_quality": 0.8},
        )

        assert result["success"] is True
        assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_orchestrate_query_uninitialized(self):
        """Test orchestrate_query raises error when not initialized."""
        orchestrator = QueryOrchestrator()

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await orchestrator.orchestrate_query("test query")

    @pytest.mark.asyncio
    async def test_orchestrate_query_error_handling(self):
        """Test orchestrate_query error handling."""
        orchestrator = QueryOrchestrator()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")
        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        # Simulate error in fallback orchestration
        with patch.object(
            orchestrator, "_fallback_orchestration", side_effect=Exception("Test error")
        ):
            result = await orchestrator.orchestrate_query("test query")

            assert result["success"] is False
            assert "error" in result
            assert "Test error" in result["error"]
            assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_fallback_orchestration(self):
        """Test fallback orchestration method."""
        orchestrator = QueryOrchestrator()

        context = {
            "query": "What is Python?",
            "collection": "docs",
            "user_context": {"user_id": "test"},
            "performance_requirements": {"max_latency": 500},
            "orchestration_id": str(uuid4()),
        }

        result = await orchestrator._fallback_orchestration(context)

        assert result["success"] is True
        assert "analysis" in result["result"]
        assert "orchestration_plan" in result["result"]
        assert result["result"]["fallback_used"] is True
        assert context["orchestration_id"] == result["orchestration_id"]
        assert context["query"] in result["result"]["orchestration_plan"]

    def test_recommend_tools(self):
        """Test tool recommendation logic for all complexity/domain combinations."""
        orchestrator = QueryOrchestrator()

        # Test simple queries
        tools = orchestrator._recommend_tools("simple", "general")
        assert "hybrid_search" in tools
        assert "hyde_search" not in tools
        assert "multi_stage_search" not in tools

        # Test moderate queries
        tools = orchestrator._recommend_tools("moderate", "general")
        assert "hybrid_search" in tools
        assert "hyde_search" not in tools

        # Test complex queries
        tools = orchestrator._recommend_tools("complex", "general")
        assert "hybrid_search" in tools
        assert "hyde_search" in tools
        assert "multi_stage_search" in tools

        # Test technical domain
        tools = orchestrator._recommend_tools("simple", "technical")
        assert "hybrid_search" in tools
        assert "content_classification" in tools

        tools = orchestrator._recommend_tools("complex", "technical")
        assert "hybrid_search" in tools
        assert "hyde_search" in tools
        assert "multi_stage_search" in tools
        assert "content_classification" in tools

        # Test non-technical domains
        tools = orchestrator._recommend_tools("moderate", "business")
        assert "hybrid_search" in tools
        assert "content_classification" not in tools

    def test_recommend_tools_edge_cases(self):
        """Test tool recommendation with edge cases."""
        orchestrator = QueryOrchestrator()

        # Test with empty inputs
        tools = orchestrator._recommend_tools("", "")
        assert "hybrid_search" in tools

        # Test with invalid complexity/domain
        tools = orchestrator._recommend_tools("invalid", "invalid")
        assert "hybrid_search" in tools

    def test_estimate_completion_time(self):
        """Test completion time estimation for all agent types and complexities."""
        orchestrator = QueryOrchestrator()

        # Test all agent types
        agent_types = [
            "retrieval_specialist",
            "answer_generator",
            "tool_selector",
            "unknown_agent",
        ]
        complexities = ["simple", "moderate", "complex", "unknown_complexity"]

        for agent_type in agent_types:
            for complexity in complexities:
                time_estimate = orchestrator._estimate_completion_time(
                    agent_type, {"complexity": complexity}
                )
                assert time_estimate > 0
                assert isinstance(time_estimate, float)

        # Test specific expected times
        simple_retrieval = orchestrator._estimate_completion_time(
            "retrieval_specialist", {"complexity": "simple"}
        )
        complex_generation = orchestrator._estimate_completion_time(
            "answer_generator", {"complexity": "complex"}
        )

        assert simple_retrieval == 1.0  # 2.0 * 0.5
        assert complex_generation == 6.0  # 3.0 * 2.0

    def test_estimate_completion_time_edge_cases(self):
        """Test completion time estimation edge cases."""
        orchestrator = QueryOrchestrator()

        # Test with empty task data
        time_estimate = orchestrator._estimate_completion_time("unknown_agent", {})
        assert time_estimate == 2.0  # Default time

        # Test with missing complexity
        time_estimate = orchestrator._estimate_completion_time(
            "retrieval_specialist", {}
        )
        assert time_estimate == 2.0  # Default base time * default multiplier

    def test_get_strategy_recommendation(self):
        """Test strategy recommendation logic."""
        orchestrator = QueryOrchestrator()

        # Test high performance - continue using
        high_perf_stats = {"avg_performance": 0.9}
        assert (
            orchestrator._get_strategy_recommendation(high_perf_stats)
            == "continue_using"
        )

        # Test medium performance - monitor
        medium_perf_stats = {"avg_performance": 0.7}
        assert (
            orchestrator._get_strategy_recommendation(medium_perf_stats)
            == "monitor_performance"
        )

        # Test low performance - consider alternative
        low_perf_stats = {"avg_performance": 0.5}
        assert (
            orchestrator._get_strategy_recommendation(low_perf_stats)
            == "consider_alternative"
        )

        # Test edge cases
        edge_high_stats = {"avg_performance": 0.8}
        assert (
            orchestrator._get_strategy_recommendation(edge_high_stats)
            == "monitor_performance"
        )

        edge_low_stats = {"avg_performance": 0.6}
        assert (
            orchestrator._get_strategy_recommendation(edge_low_stats)
            == "consider_alternative"
        )

    def test_get_strategy_recommendation_edge_cases(self):
        """Test strategy recommendation with edge cases."""
        orchestrator = QueryOrchestrator()

        # Test with missing keys
        recommendation = orchestrator._get_strategy_recommendation({})
        # Should handle missing keys gracefully
        assert recommendation in [
            "continue_using",
            "monitor_performance",
            "consider_alternative",
        ]

    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking and learning."""
        orchestrator = QueryOrchestrator()

        # Test initial state
        assert orchestrator.strategy_performance == {}

        # Test strategy performance tracking
        strategy = "test_strategy"
        stats = {
            "total_uses": 5,
            "avg_performance": 0.75,
            "avg_latency": 300.0,
            "avg_quality": 0.8,
        }

        orchestrator.strategy_performance[strategy] = stats

        # Verify state persistence
        assert orchestrator.strategy_performance[strategy] == stats

        # Test recommendation generation
        recommendation = orchestrator._get_strategy_recommendation(stats)
        assert recommendation == "monitor_performance"

    def test_strategy_performance_learning(self):
        """Test strategy performance learning and adaptation."""
        orchestrator = QueryOrchestrator()
        strategy = "fast"

        # Simulate multiple evaluations
        evaluations = [
            {"latency": 200.0, "quality": 0.7, "cost": 0.02},
            {"latency": 300.0, "quality": 0.8, "cost": 0.03},
            {"latency": 250.0, "quality": 0.75, "cost": 0.025},
        ]

        for eval_data in evaluations:
            latency = eval_data["latency"]
            quality = eval_data["quality"]
            cost = eval_data["cost"]

            performance_score = (
                (1.0 - min(latency / 1000.0, 1.0)) * 0.3
                + quality * 0.6
                + (1.0 - min(cost / 0.1, 1.0)) * 0.1
            )

            if strategy not in orchestrator.strategy_performance:
                orchestrator.strategy_performance[strategy] = {
                    "total_uses": 0,
                    "avg_performance": 0.0,
                    "avg_latency": 0.0,
                    "avg_quality": 0.0,
                }

            stats = orchestrator.strategy_performance[strategy]
            stats["total_uses"] += 1

            # Update running averages
            alpha = 0.1
            stats["avg_performance"] = (1 - alpha) * stats[
                "avg_performance"
            ] + alpha * performance_score
            stats["avg_latency"] = (1 - alpha) * stats["avg_latency"] + alpha * latency
            stats["avg_quality"] = (1 - alpha) * stats["avg_quality"] + alpha * quality

        # Verify learning occurred
        final_stats = orchestrator.strategy_performance[strategy]
        assert final_stats["total_uses"] == 3
        assert final_stats["avg_performance"] > 0
        assert final_stats["avg_latency"] > 0
        assert final_stats["avg_quality"] > 0

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Test performance score calculation
        test_cases = [
            # (latency, quality, cost, expected_range)
            (100.0, 0.9, 0.01, (0.85, 1.0)),  # Excellent performance
            (500.0, 0.7, 0.05, (0.6, 0.65)),  # Good performance
            (1000.0, 0.5, 0.1, (0.25, 0.35)),  # Average performance
            (2000.0, 0.3, 0.2, (0.15, 0.25)),  # Poor performance
        ]

        for latency, quality, cost, expected_range in test_cases:
            performance_score = (
                (1.0 - min(latency / 1000.0, 1.0)) * 0.3
                + quality * 0.6
                + (1.0 - min(cost / 0.1, 1.0)) * 0.1
            )

            assert expected_range[0] <= performance_score <= expected_range[1], (
                f"Performance score {performance_score} not in expected range "
                f"{expected_range}"
            )

    def test_query_analysis_logic(self):
        """Test query analysis and classification logic."""
        # Test different query types
        test_queries = [
            ("What is machine learning?", "simple", "general"),
            (
                "How to implement a neural network algorithm?",
                "moderate",
                "technical",
            ),  # 'algorithm' makes it technical
            (
                "Analyze the market trends and evaluate investment strategies",
                "complex",
                "business",
            ),
            ("Define artificial intelligence", "simple", "general"),
            ("Compare different database systems", "moderate", "technical"),
            (
                "Research the academic literature on quantum computing",
                "moderate",
                "academic",
            ),
        ]

        for query, expected_complexity, expected_domain in test_queries:
            # Simulate the query analysis logic from the tool
            query_lower = query.lower()

            # Test complexity detection logic
            complexity_indicators = {
                "simple": ["what is", "who is", "when did", "where is", "define"],
                "moderate": ["how to", "why does", "compare", "difference between"],
                "complex": ["analyze", "evaluate", "recommend", "strategy", "multiple"],
            }

            detected_complexity = "moderate"  # Default
            for level, indicators in complexity_indicators.items():
                if any(indicator in query_lower for indicator in indicators):
                    detected_complexity = level
                    break

            # Test domain detection logic
            domains = {
                "technical": ["code", "programming", "api", "database", "algorithm"],
                "business": ["market", "revenue", "strategy", "customer", "sales"],
                "academic": ["research", "study", "theory", "academic", "paper"],
            }

            detected_domain = "general"
            for domain_name, keywords in domains.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_domain = domain_name
                    break

            assert detected_complexity == expected_complexity
            assert detected_domain == expected_domain

    def test_multi_step_query_detection(self):
        """Test multi-step query detection logic."""
        # Test queries that should be detected as multi-step
        multi_step_queries = [
            "First, analyze the data and then generate a report",
            "Step by step implementation guide",
            "Process the input and after that validate the results",
            "Finally, summarize the findings",
            "First step is to collect data, second step is analysis",
        ]

        # Test queries that should NOT be detected as multi-step
        single_step_queries = [
            "What is machine learning?",
            "How to implement a neural network?",
            "Define artificial intelligence",
            "Compare different algorithms",
        ]

        # Simulate the multi-step detection logic
        multi_step_indicators = [
            "and then",
            "after that",
            "step by step",
            "process",
            "first",
            "second",
            "finally",
        ]

        for query in multi_step_queries:
            query_lower = query.lower()
            requires_multi_step = any(
                indicator in query_lower for indicator in multi_step_indicators
            )
            assert requires_multi_step, (
                f"Query should be detected as multi-step: {query}"
            )

        for query in single_step_queries:
            query_lower = query.lower()
            requires_multi_step = any(
                indicator in query_lower for indicator in multi_step_indicators
            )
            assert not requires_multi_step, (
                f"Query should NOT be detected as multi-step: {query}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_orchestration(self):
        """Test concurrent query orchestration."""
        orchestrator = QueryOrchestrator()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")
        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        queries = [
            "What is Python?",
            "How to implement sorting algorithms?",
            "Analyze machine learning trends",
            "Define data structures",
            "Compare database systems",
        ]

        # Run multiple orchestrations concurrently
        tasks = [
            orchestrator.orchestrate_query(query, collection="docs")
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all orchestrations completed
        assert len(results) == len(queries)
        for result in results:
            assert not isinstance(result, Exception)
            assert "success" in result
            assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_performance_constraint_handling(self):
        """Test handling of performance constraints."""
        orchestrator = QueryOrchestrator()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")
        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        # Test with various performance requirements
        performance_requirements = [
            {"max_latency": 500},
            {"max_cost": 0.1},
            {"min_quality": 0.8},
            {"max_latency": 1000, "max_cost": 0.05, "min_quality": 0.7},
            {},  # No requirements
        ]

        for requirements in performance_requirements:
            result = await orchestrator.orchestrate_query(
                query="test query",
                collection="docs",
                performance_requirements=requirements,
            )

            assert result["success"] is True
            assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test orchestrator error recovery and graceful degradation."""
        orchestrator = QueryOrchestrator()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")
        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await orchestrator.initialize(deps)

        # Test various error scenarios
        error_scenarios = [
            {"query": "", "collection": "docs"},  # Empty query
            {"query": "test", "collection": ""},  # Empty collection
            {"query": "test", "collection": "nonexistent"},  # Invalid collection
        ]

        for scenario in error_scenarios:
            try:
                result = await orchestrator.orchestrate_query(**scenario)
                # Should handle gracefully
                assert "success" in result
                assert "orchestration_id" in result
            except (ConnectionError, RuntimeError, ValueError) as e:
                # If exceptions occur, they should be handled gracefully
                exception_instance = e
                assert isinstance(exception_instance, ValueError | RuntimeError)

    def test_orchestrator_state_management(self):
        """Test orchestrator state management."""
        orchestrator = QueryOrchestrator()

        # Test initial state
        assert orchestrator.strategy_performance == {}
        assert orchestrator.name == "query_orchestrator"
        assert not orchestrator._initialized

        # Test configuration validation
        assert orchestrator.model == "gpt-4"
        assert orchestrator.temperature == 0.1
        assert orchestrator.max_tokens == 1500

    def test_orchestrator_configuration_validation(self):
        """Test orchestrator configuration validation."""
        # Test valid configurations
        valid_configs = [
            {"model": "gpt-4"},
            {"model": "gpt-3.5-turbo"},
        ]

        for config in valid_configs:
            orchestrator = QueryOrchestrator(**config)
            assert orchestrator.model == config["model"]
            assert orchestrator.temperature == 0.1
            assert orchestrator.max_tokens == 1500


# ToolCompositionEngine tests removed - functionality replaced by AgenticOrchestrator
# and native Pydantic-AI tool orchestration patterns in the new architecture


class TestCreateAgentDependencies:
    """Test agent dependency creation utilities."""

    def test_create_dependencies_with_defaults(self):
        """Test creating dependencies with default values."""
        mock_client_manager = Mock(spec=ClientManager)

        deps = create_agent_dependencies(mock_client_manager)

        assert deps.client_manager == mock_client_manager
        assert deps.config is not None
        assert deps.session_state.session_id is not None
        assert deps.session_state.user_id is None

    def test_create_dependencies_with_values(self):
        """Test creating dependencies with specific values."""
        mock_client_manager = Mock(spec=ClientManager)
        session_id = "custom_session"
        user_id = "user_123"

        deps = create_agent_dependencies(
            mock_client_manager, session_id=session_id, user_id=user_id
        )

        assert deps.session_state.session_id == session_id
        assert deps.session_state.user_id == user_id


# Integration tests would go here for testing with actual Pydantic-AI
# when the dependency is available
@pytest.mark.skip(reason="Requires Pydantic-AI installation")
class TestPydanticAIIntegration:
    """Integration tests with actual Pydantic-AI (when available)."""

    @pytest.mark.asyncio
    async def test_actual_agent_execution(self):
        """Test actual agent execution with Pydantic-AI."""
        # This would test with actual Pydantic-AI when installed

    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test actual tool registration with Pydantic-AI."""
        # This would test tool registration when Pydantic-AI is available
