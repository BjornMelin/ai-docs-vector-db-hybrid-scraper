"""Tests for agentic RAG implementation."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    QueryOrchestrator,
    ToolCompositionEngine,
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

    def test_system_prompt(self):
        """Test system prompt generation."""
        orchestrator = QueryOrchestrator()
        prompt = orchestrator.get_system_prompt()

        assert "Query Orchestrator" in prompt
        assert "analyze incoming queries" in prompt
        assert "delegate to specialized agents" in prompt

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

    def test_recommend_tools(self):
        """Test tool recommendation logic."""
        orchestrator = QueryOrchestrator()

        # Test simple query
        tools = orchestrator._recommend_tools("simple", "general")
        assert "hybrid_search" in tools

        # Test complex technical query
        tools = orchestrator._recommend_tools("complex", "technical")
        assert "hybrid_search" in tools
        assert "hyde_search" in tools
        assert "content_classification" in tools

    def test_estimate_completion_time(self):
        """Test completion time estimation."""
        orchestrator = QueryOrchestrator()

        # Test different agent types
        time1 = orchestrator._estimate_completion_time(
            "retrieval_specialist", {"complexity": "simple"}
        )
        time2 = orchestrator._estimate_completion_time(
            "answer_generator", {"complexity": "complex"}
        )

        assert time1 > 0
        assert time2 > time1  # Complex tasks should take longer


class TestToolCompositionEngine:
    """Test tool composition engine."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test tool composition engine initialization."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)

        await engine.initialize()

        assert len(engine.tool_registry) > 0
        assert "hybrid_search" in engine.tool_registry
        assert "hyde_search" in engine.tool_registry

    @pytest.mark.asyncio
    async def test_analyze_goal(self):
        """Test goal analysis functionality."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)

        # Test search goal
        analysis = await engine._analyze_goal(
            "search for information about AI", {"max_latency_ms": 500}
        )

        assert analysis["primary_action"] == "search"
        assert "search" in analysis["required_capabilities"]
        assert (
            analysis["performance_priority"] == "speed"
        )  # Due to low latency requirement

        # Test generation goal
        analysis = await engine._analyze_goal(
            "generate a comprehensive analysis", {"min_quality_score": 0.95}
        )

        assert analysis["primary_action"] == "generate"
        assert "rag" in analysis["required_capabilities"]
        assert analysis["performance_priority"] == "quality"

    @pytest.mark.asyncio
    async def test_select_optimal_tools(self):
        """Test optimal tool selection."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)
        await engine.initialize()

        goal_analysis = {
            "primary_action": "search",
            "required_capabilities": ["search"],
            "performance_priority": "speed",
            "complexity": "simple",
        }

        tools = await engine._select_optimal_tools(
            goal_analysis, {"max_latency_ms": 200}, None
        )

        assert len(tools) > 0
        assert any("search" in tool for tool in tools)

    @pytest.mark.asyncio
    async def test_create_execution_chain(self):
        """Test execution chain creation."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)
        await engine.initialize()

        selected_tools = ["hybrid_search", "generate_rag_answer"]
        goal_analysis = {"primary_action": "generate"}

        chain = await engine._create_execution_chain(selected_tools, goal_analysis)

        assert len(chain) == 2
        assert chain[0].tool_name == "hybrid_search"
        assert chain[1].tool_name == "generate_rag_answer"

        # Check that second tool uses first tool's results
        assert "search_results" in chain[1].input_mapping

    @pytest.mark.asyncio
    async def test_execute_tool_chain(self):
        """Test tool chain execution."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)
        await engine.initialize()

        # Create simple chain
        from src.services.agents.tool_composition import ToolChainStep

        chain = [
            ToolChainStep(
                tool_name="hybrid_search",
                input_mapping={
                    "query": "input_data.query",
                    "collection": "input_data.collection",
                },
                output_key="search_results",
                parallel=False,
                optional=False,
            )
        ]

        input_data = {"query": "test query", "collection": "test_collection"}

        result = await engine.execute_tool_chain(chain, input_data, timeout_seconds=5.0)

        assert "success" in result
        assert "execution_id" in result
        assert "metadata" in result

    def test_get_tool_metadata(self):
        """Test tool metadata retrieval."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)

        # Add a test tool
        from src.services.agents.tool_composition import (
            ToolCategory,
            ToolMetadata,
            ToolPriority,
        )

        test_metadata = ToolMetadata(
            name="test_tool",
            category=ToolCategory.SEARCH,
            description="Test tool",
            input_schema={},
            output_schema={},
            performance_metrics={},
            dependencies=[],
            priority=ToolPriority.NORMAL,
        )
        engine.tool_registry["test_tool"] = test_metadata

        metadata = engine.get_tool_metadata("test_tool")
        assert metadata is not None
        assert metadata.name == "test_tool"

        # Test non-existent tool
        metadata = engine.get_tool_metadata("nonexistent")
        assert metadata is None

    def test_list_tools_by_category(self):
        """Test listing tools by category."""
        mock_client_manager = Mock(spec=ClientManager)
        engine = ToolCompositionEngine(mock_client_manager)

        # Add test tools
        from src.services.agents.tool_composition import (
            ToolCategory,
            ToolMetadata,
            ToolPriority,
        )

        search_tool = ToolMetadata(
            name="search_tool",
            category=ToolCategory.SEARCH,
            description="Search tool",
            input_schema={},
            output_schema={},
            performance_metrics={},
            dependencies=[],
        )
        rag_tool = ToolMetadata(
            name="rag_tool",
            category=ToolCategory.RAG,
            description="RAG tool",
            input_schema={},
            output_schema={},
            performance_metrics={},
            dependencies=[],
        )

        engine.tool_registry["search_tool"] = search_tool
        engine.tool_registry["rag_tool"] = rag_tool

        search_tools = engine.list_tools_by_category(ToolCategory.SEARCH)
        rag_tools = engine.list_tools_by_category(ToolCategory.RAG)

        assert "search_tool" in search_tools
        assert "rag_tool" in rag_tools
        assert "rag_tool" not in search_tools


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
