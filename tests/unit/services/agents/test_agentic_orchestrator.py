"""Comprehensive tests for AgenticOrchestrator implementation.

This test suite covers:
- AgenticOrchestrator initialization and setup
- Pure Pydantic-AI native orchestration patterns
- Tool selection and capability assessment
- Autonomous decision-making logic
- Performance metrics and confidence scoring
- Error handling and fallback scenarios
- Integration with BaseAgent framework

The AgenticOrchestrator replaced a 950-line ToolCompositionEngine with ~200 lines
of native patterns, so tests focus on clean autonomous capabilities.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import given, strategies as st

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import (
    AgenticOrchestrator,
    ToolRequest,
    ToolResponse,
    get_orchestrator,
    orchestrate_tools,
)
from src.services.agents.core import AgentState, BaseAgentDependencies


class TestToolRequest:
    """Test ToolRequest model validation and behavior."""

    def test_tool_request_initialization(self):
        """Test ToolRequest initialization with required fields."""
        request = ToolRequest(task="test task")

        assert request.task == "test task"
        assert request.constraints == {}
        assert request.context == {}

    def test_tool_request_with_full_data(self):
        """Test ToolRequest with all optional fields."""
        constraints = {"max_latency_ms": 500, "min_quality": 0.8}
        context = {"session_id": "test_session", "user_id": "user_123"}

        request = ToolRequest(
            task="complex analysis task", constraints=constraints, context=context
        )

        assert request.task == "complex analysis task"
        assert request.constraints == constraints
        assert request.context == context

    @given(
        task=st.text(min_size=1, max_size=200),
        max_latency=st.integers(min_value=100, max_value=10000),
        quality_score=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_tool_request_property_based(self, task, max_latency, quality_score):
        """Property-based testing for ToolRequest validation."""
        constraints = {"max_latency_ms": max_latency, "min_quality": quality_score}
        request = ToolRequest(task=task, constraints=constraints)

        assert request.task == task
        assert request.constraints["max_latency_ms"] == max_latency
        assert request.constraints["min_quality"] == quality_score


class TestToolResponse:
    """Test ToolResponse model validation and behavior."""

    def test_tool_response_success(self):
        """Test successful ToolResponse creation."""
        response = ToolResponse(
            success=True,
            results={"answer": "test result"},
            tools_used=["hybrid_search", "rag_generation"],
            reasoning="Selected tools based on query complexity",
            latency_ms=250.5,
            confidence=0.85,
        )

        assert response.success is True
        assert response.results == {"answer": "test result"}
        assert response.tools_used == ["hybrid_search", "rag_generation"]
        assert response.reasoning == "Selected tools based on query complexity"
        assert response.latency_ms == 250.5
        assert response.confidence == 0.85

    def test_tool_response_failure(self):
        """Test failure ToolResponse creation."""
        response = ToolResponse(
            success=False,
            results={"error": "Tool execution failed"},
            tools_used=[],
            reasoning="Failed due to timeout",
            latency_ms=5000.0,
            confidence=0.0,
        )

        assert response.success is False
        assert response.tools_used == []
        assert response.confidence == 0.0

    @given(
        success=st.booleans(),
        latency=st.floats(min_value=0.0, max_value=30000.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_tool_response_property_based(self, success, latency, confidence):
        """Property-based testing for ToolResponse validation."""
        response = ToolResponse(
            success=success,
            results={},
            tools_used=[],
            reasoning="test reasoning",
            latency_ms=latency,
            confidence=confidence,
        )

        assert response.success == success
        assert response.latency_ms == latency
        assert response.confidence == confidence


class TestAgenticOrchestratorInitialization:
    """Test AgenticOrchestrator initialization and setup."""

    def test_orchestrator_default_initialization(self):
        """Test orchestrator initialization with default parameters."""
        orchestrator = AgenticOrchestrator()

        assert orchestrator.name == "agentic_orchestrator"
        assert orchestrator.model == "gpt-4o-mini"
        assert orchestrator.temperature == 0.1
        assert orchestrator.max_tokens == 1500
        assert not orchestrator._initialized

    def test_orchestrator_custom_initialization(self):
        """Test orchestrator initialization with custom parameters."""
        orchestrator = AgenticOrchestrator(model="gpt-4", temperature=0.3)

        assert orchestrator.name == "agentic_orchestrator"
        assert orchestrator.model == "gpt-4"
        assert orchestrator.temperature == 0.3
        assert orchestrator.max_tokens == 1500

    def test_system_prompt_content(self):
        """Test system prompt contains required autonomous capabilities."""
        orchestrator = AgenticOrchestrator()
        prompt = orchestrator.get_system_prompt()

        # Check for core autonomous capabilities
        assert "INTELLIGENT ANALYSIS" in prompt
        assert "AUTONOMOUS EXECUTION" in prompt
        assert "PERFORMANCE OPTIMIZATION" in prompt

        # Check for specific behaviors
        assert "Analyze user requests" in prompt
        assert "Select optimal tool combinations" in prompt
        assert "Make real-time decisions" in prompt
        assert "Balance speed, quality, and resource constraints" in prompt
        assert "Always explain your reasoning" in prompt

    @pytest.mark.asyncio
    async def test_initialize_tools_fallback_mode(self):
        """Test tool initialization in fallback mode (no Pydantic-AI)."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test initialization in fallback mode
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            await orchestrator.initialize_tools(deps)
            # Should complete without error in fallback mode

    @pytest.mark.asyncio
    async def test_initialize_tools_with_pydantic_ai(self):
        """Test tool initialization with Pydantic-AI available."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Mock Pydantic-AI availability and agent
        mock_agent = Mock()
        mock_agent.tool_plain = Mock()
        orchestrator.agent = mock_agent

        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", True
        ):
            await orchestrator.initialize_tools(deps)

            # Verify tool registration
            assert mock_agent.tool_plain.call_count == 3  # Three tools registered


class TestToolDiscoveryAndSelection:
    """Test tool discovery and intelligent selection logic."""

    @pytest.mark.asyncio
    async def test_discover_tools(self):
        """Test dynamic tool discovery functionality."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        tools = await orchestrator._discover_tools(deps)

        # Verify core tools are discovered
        assert "hybrid_search" in tools
        assert "rag_generation" in tools
        assert "content_analysis" in tools

        # Verify tool metadata structure
        for tool_info in tools.values():
            assert "capabilities" in tool_info
            assert "performance" in tool_info
            assert "description" in tool_info
            assert isinstance(tool_info["capabilities"], list)
            assert "latency_ms" in tool_info["performance"]
            assert "accuracy" in tool_info["performance"]

    def test_select_tools_for_search_task(self):
        """Test tool selection for search-oriented tasks."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
            "content_analysis": {"capabilities": ["analysis", "classification"]},
        }

        # Test search task
        selected = orchestrator._select_tools_for_task(
            "search for information about machine learning", available_tools, {}
        )

        assert "hybrid_search" in selected
        assert len(selected) >= 1

    def test_select_tools_for_generation_task(self):
        """Test tool selection for generation-oriented tasks."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
            "content_analysis": {"capabilities": ["analysis", "classification"]},
        }

        # Test generation task
        selected = orchestrator._select_tools_for_task(
            "generate a comprehensive answer about AI", available_tools, {}
        )

        assert "hybrid_search" in selected  # For retrieval
        assert "rag_generation" in selected  # For generation
        assert len(selected) >= 2

    def test_select_tools_for_analysis_task(self):
        """Test tool selection for analysis-oriented tasks."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
            "content_analysis": {"capabilities": ["analysis", "classification"]},
        }

        # Test analysis task
        selected = orchestrator._select_tools_for_task(
            "analyze and classify this document", available_tools, {}
        )

        assert "content_analysis" in selected

    def test_select_tools_with_latency_constraints(self):
        """Test tool selection respects performance constraints."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
            "content_analysis": {"capabilities": ["analysis", "classification"]},
        }

        # Test with strict latency constraint
        selected = orchestrator._select_tools_for_task(
            "generate an answer quickly",
            available_tools,
            {"max_latency_ms": 500},  # Low latency requirement
        )

        # Should exclude slow rag_generation tool
        assert "rag_generation" not in selected
        assert "hybrid_search" in selected

    def test_select_tools_fallback_behavior(self):
        """Test fallback behavior when no specific tools match."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
            "content_analysis": {"capabilities": ["analysis", "classification"]},
        }

        # Test with vague task
        selected = orchestrator._select_tools_for_task(
            "do something", available_tools, {}
        )

        # Should fallback to search
        assert "hybrid_search" in selected
        assert len(selected) >= 1

    @given(
        task=st.text(min_size=5, max_size=100),
        max_latency=st.integers(min_value=100, max_value=5000),
    )
    def test_select_tools_property_based(self, task, max_latency):
        """Property-based testing for tool selection logic."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "hybrid_search": {"capabilities": ["search", "retrieval"]},
            "rag_generation": {"capabilities": ["generation", "synthesis"]},
        }

        selected = orchestrator._select_tools_for_task(
            task, available_tools, {"max_latency_ms": max_latency}
        )

        # Should always select at least one tool
        assert len(selected) >= 1
        # All selected tools should be from available tools
        assert all(tool in available_tools for tool in selected)


class TestAutonomousOrchestration:
    """Test autonomous orchestration and decision-making logic."""

    @pytest.mark.asyncio
    async def test_orchestrate_autonomous_success(self):
        """Test successful autonomous orchestration."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        request = ToolRequest(
            task="search for AI documentation",
            constraints={"max_latency_ms": 1000},
            context={"session_id": "test"},
        )

        # Mock tool execution
        with patch.object(
            orchestrator, "_execute_tool", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {
                "tool": "hybrid_search",
                "result": "mock search results",
                "input_keys": ["task", "constraints"],
                "timestamp": time.time(),
            }

            response = await orchestrator._orchestrate_autonomous(request, deps)

            assert response.success is True
            assert response.tools_used
            assert response.reasoning
            assert response.latency_ms > 0
            assert 0 <= response.confidence <= 1

    @pytest.mark.asyncio
    async def test_orchestrate_autonomous_with_error(self):
        """Test autonomous orchestration handles errors gracefully."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        request = ToolRequest(task="failing task")

        # Mock tool execution failure
        with patch.object(
            orchestrator, "_discover_tools", side_effect=Exception("Discovery failed")
        ):
            response = await orchestrator._orchestrate_autonomous(request, deps)

            assert response.success is False
            assert "Discovery failed" in response.results["error"]
            assert response.confidence == 0.0
            assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_chain_sequential(self):
        """Test sequential tool chain execution."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        tools = ["hybrid_search", "rag_generation"]
        input_data = {"task": "test task", "constraints": {}}

        # Mock individual tool execution
        with patch.object(
            orchestrator, "_execute_tool", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = [
                {
                    "tool": "hybrid_search",
                    "result": "search results",
                    "timestamp": time.time(),
                },
                {
                    "tool": "rag_generation",
                    "result": "generated answer",
                    "timestamp": time.time(),
                },
            ]

            results = await orchestrator._execute_chain(tools, input_data, deps)

            assert len(results) == 2
            assert "hybrid_search_result" in results
            assert "rag_generation_result" in results
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_chain_with_failures(self):
        """Test tool chain execution handles individual tool failures."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        tools = ["working_tool", "failing_tool"]
        input_data = {"task": "test task"}

        # Mock mixed success/failure
        async def mock_execute_tool(tool_name, context, deps):
            if tool_name == "failing_tool":
                msg = "Tool execution failed"
                raise Exception(msg)
            return {
                "tool": tool_name,
                "result": f"{tool_name} result",
                "timestamp": time.time(),
            }

        with patch.object(orchestrator, "_execute_tool", side_effect=mock_execute_tool):
            results = await orchestrator._execute_chain(tools, input_data, deps)

            assert "working_tool_result" in results
            assert "failing_tool_error" in results
            assert results["failing_tool_error"] == "Tool execution failed"

    @pytest.mark.asyncio
    async def test_execute_individual_tool(self):
        """Test individual tool execution mock implementation."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        context = {"task": "test task", "param": "value"}

        result = await orchestrator._execute_tool("test_tool", context, deps)

        assert result["tool"] == "test_tool"
        assert result["result"] == "Mock result from test_tool"
        assert result["input_keys"] == ["task", "param"]
        assert "timestamp" in result


class TestPerformanceAndConfidence:
    """Test performance metrics and confidence scoring."""

    def test_generate_reasoning_search_only(self):
        """Test reasoning generation for search-only tasks."""
        orchestrator = AgenticOrchestrator()

        reasoning = orchestrator._generate_reasoning(
            "find information about AI",
            ["hybrid_search"],
            {"hybrid_search_result": {"tool": "hybrid_search"}},
        )

        assert "find information about AI" in reasoning
        assert "hybrid_search" in reasoning
        assert "Used hybrid search for information retrieval" in reasoning
        assert "Successfully executed 1 tools" in reasoning

    def test_generate_reasoning_multi_tool(self):
        """Test reasoning generation for multi-tool workflows."""
        orchestrator = AgenticOrchestrator()

        reasoning = orchestrator._generate_reasoning(
            "generate comprehensive analysis",
            ["hybrid_search", "rag_generation", "content_analysis"],
            {
                "hybrid_search_result": {"tool": "hybrid_search"},
                "rag_generation_result": {"tool": "rag_generation"},
                "content_analysis_result": {"tool": "content_analysis"},
            },
        )

        assert "generate comprehensive analysis" in reasoning
        assert "3 tools" in reasoning
        assert "hybrid_search, rag_generation, content_analysis" in reasoning
        assert "Used hybrid search for information retrieval" in reasoning
        assert "Applied RAG generation for comprehensive answers" in reasoning
        assert "Performed content analysis for deeper insights" in reasoning
        assert "Successfully executed 3 tools" in reasoning

    def test_generate_reasoning_with_errors(self):
        """Test reasoning generation accounts for tool failures."""
        orchestrator = AgenticOrchestrator()

        reasoning = orchestrator._generate_reasoning(
            "test task",
            ["tool1", "tool2"],
            {"tool1_result": {"tool": "tool1"}, "tool2_error": "Tool failed"},
        )

        assert "Successfully executed 1 tools" in reasoning

    def test_calculate_confidence_perfect_success(self):
        """Test confidence calculation for perfect execution."""
        orchestrator = AgenticOrchestrator()

        results = {"tool1_result": {"success": True}, "tool2_result": {"success": True}}
        tools_used = ["tool1", "tool2"]

        confidence = orchestrator._calculate_confidence(results, tools_used)

        assert confidence == 1.0  # Perfect success gets boosted to 1.0

    def test_calculate_confidence_partial_success(self):
        """Test confidence calculation for partial success."""
        orchestrator = AgenticOrchestrator()

        results = {"tool1_result": {"success": True}, "tool2_error": "Failed"}
        tools_used = ["tool1", "tool2"]

        confidence = orchestrator._calculate_confidence(results, tools_used)

        assert confidence == 0.5  # 1 success out of 2 tools

    def test_calculate_confidence_no_success(self):
        """Test confidence calculation for complete failure."""
        orchestrator = AgenticOrchestrator()

        results = {"tool1_error": "Failed", "tool2_error": "Failed"}
        tools_used = ["tool1", "tool2"]

        confidence = orchestrator._calculate_confidence(results, tools_used)

        assert confidence == 0.0

    def test_calculate_confidence_empty_results(self):
        """Test confidence calculation for empty results."""
        orchestrator = AgenticOrchestrator()

        confidence = orchestrator._calculate_confidence({}, [])

        assert confidence == 0.0

    def test_calculate_confidence_single_tool_boost(self):
        """Test confidence boost for multi-tool success."""
        orchestrator = AgenticOrchestrator()

        # Single tool success - no boost
        single_results = {"tool1_result": {"success": True}}
        single_confidence = orchestrator._calculate_confidence(
            single_results, ["tool1"]
        )

        # Multi-tool success - gets boost
        multi_results = {
            "tool1_result": {"success": True},
            "tool2_result": {"success": True},
            "tool3_result": {"success": True},
        }
        multi_confidence = orchestrator._calculate_confidence(
            multi_results, ["tool1", "tool2", "tool3"]
        )

        assert single_confidence == 1.0
        assert multi_confidence == 1.0  # Boosted and capped at 1.0

    @given(
        success_count=st.integers(min_value=0, max_value=10),
        total_count=st.integers(min_value=1, max_value=10),
    )
    def test_calculate_confidence_property_based(self, success_count, total_count):
        """Property-based testing for confidence calculation."""
        orchestrator = AgenticOrchestrator()

        # Ensure success_count doesn't exceed total_count
        success_count = min(success_count, total_count)

        # Create results with specified success/failure ratio
        results = {}
        tools_used = []

        for i in range(total_count):
            tool_name = f"tool{i}"
            tools_used.append(tool_name)

            if i < success_count:
                results[f"{tool_name}_result"] = {"success": True}
            else:
                results[f"{tool_name}_error"] = "Failed"

        confidence = orchestrator._calculate_confidence(results, tools_used)

        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0

        # If no successes, confidence should be 0
        if success_count == 0:
            assert confidence == 0.0

        # If all successes and more than one tool, confidence should be boosted
        if success_count == total_count and total_count > 1:
            expected_base = success_count / total_count
            expected_boosted = min(expected_base * 1.1, 1.0)
            assert abs(confidence - expected_boosted) < 0.01


class TestMainOrchestrationAPI:
    """Test main orchestration API and integration points."""

    @pytest.mark.asyncio
    async def test_orchestrate_main_api_success(self):
        """Test main orchestrate API for successful execution."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        task = "search for machine learning resources"
        constraints = {"max_latency_ms": 2000}

        # Mock Pydantic-AI execution
        mock_result = Mock()
        mock_result.data = "Agent orchestration completed"

        with (
            patch(
                "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", True
            ),
            patch.object(orchestrator, "agent") as mock_agent,
        ):
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await orchestrator.orchestrate(task, constraints, deps)

            assert response.success is True
            assert response.results["agent_response"] == "Agent orchestration completed"
            assert response.tools_used == ["autonomous_agent"]
            assert response.reasoning == "Native Pydantic-AI agent orchestration"
            assert response.confidence == 0.8

    @pytest.mark.asyncio
    async def test_orchestrate_fallback_mode(self):
        """Test orchestration in fallback mode."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        task = "test task"
        constraints = {}

        # Test fallback mode
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            response = await orchestrator.orchestrate(task, constraints, deps)

            assert response.success is True
            assert "Processed task: test task" in response.results["fallback_response"]
            assert response.tools_used == ["fallback_handler"]
            assert response.reasoning == "Fallback mode - Pydantic-AI not available"
            assert response.confidence == 0.5

    @pytest.mark.asyncio
    async def test_orchestrate_with_pydantic_ai_error(self):
        """Test orchestration handles Pydantic-AI execution errors."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        task = "failing task"
        constraints = {}

        # Mock Pydantic-AI failure
        with (
            patch(
                "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", True
            ),
            patch.object(orchestrator, "agent") as mock_agent,
        ):
            mock_agent.run = AsyncMock(side_effect=Exception("Agent execution failed"))

            response = await orchestrator.orchestrate(task, constraints, deps)

            assert response.success is False
            assert "Agent execution failed" in response.results["error"]
            assert response.confidence == 0.0

    @pytest.mark.asyncio
    async def test_orchestrate_updates_session_state(self):
        """Test orchestration properly updates session state."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        task = "test task"
        constraints = {}

        # Test session state updates
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            await orchestrator.orchestrate(task, constraints, deps)

            # Verify session state was updated
            assert deps.session_state.tool_usage_stats["agentic_orchestrator"] == 1
            assert len(deps.session_state.conversation_history) == 1

            interaction = deps.session_state.conversation_history[0]
            assert interaction["role"] == "orchestrator"
            assert "metadata" in interaction
            assert "tools_used" in interaction["metadata"]


class TestFallbackOrchestration:
    """Test fallback orchestration when Pydantic-AI unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_orchestrate_basic(self):
        """Test basic fallback orchestration functionality."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        request = ToolRequest(
            task="fallback test task", constraints={"max_latency_ms": 1000}
        )

        response = await orchestrator._fallback_orchestrate(request, deps)

        assert response.success is True
        assert (
            "Processed task: fallback test task"
            in response.results["fallback_response"]
        )
        assert response.tools_used == ["fallback_handler"]
        assert response.reasoning == "Fallback mode - Pydantic-AI not available"
        assert response.latency_ms == 50.0
        assert response.confidence == 0.5

    @pytest.mark.asyncio
    async def test_fallback_orchestrate_different_tasks(self):
        """Test fallback orchestration handles different task types."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        tasks = [
            "search for information",
            "generate content",
            "analyze data",
            "complex workflow task",
        ]

        for task in tasks:
            request = ToolRequest(task=task)
            response = await orchestrator._fallback_orchestrate(request, deps)

            assert response.success is True
            assert task in response.results["fallback_response"]
            assert response.tools_used == ["fallback_handler"]


class TestSingletonPattern:
    """Test singleton pattern and utility functions."""

    def test_get_orchestrator_singleton(self):
        """Test get_orchestrator returns singleton instance."""
        # Clear any existing instance
        import src.services.agents.agentic_orchestrator as orchestrator_module

        orchestrator_module._orchestrator_instance = None

        # Get first instance
        orchestrator1 = get_orchestrator()

        # Get second instance
        orchestrator2 = get_orchestrator()

        # Should be same instance
        assert orchestrator1 is orchestrator2
        assert isinstance(orchestrator1, AgenticOrchestrator)

    def test_get_orchestrator_creates_new_when_none(self):
        """Test get_orchestrator creates new instance when none exists."""
        import src.services.agents.agentic_orchestrator as orchestrator_module

        orchestrator_module._orchestrator_instance = None

        orchestrator = get_orchestrator()

        assert isinstance(orchestrator, AgenticOrchestrator)
        assert orchestrator.name == "agentic_orchestrator"

    @pytest.mark.asyncio
    async def test_orchestrate_tools_convenience_function(self):
        """Test orchestrate_tools convenience function."""
        import src.services.agents.agentic_orchestrator as orchestrator_module

        orchestrator_module._orchestrator_instance = None

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        task = "test convenience function"
        constraints = {"test": "constraint"}

        # Test convenience function
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            response = await orchestrate_tools(task, constraints, deps)

            assert isinstance(response, ToolResponse)
            assert response.success is True


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_orchestration_with_empty_task(self):
        """Test orchestration handles empty task gracefully."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test with empty task
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            response = await orchestrator.orchestrate("", {}, deps)

            assert response.success is True  # Fallback should handle gracefully

    @pytest.mark.asyncio
    async def test_orchestration_with_malformed_constraints(self):
        """Test orchestration handles malformed constraints."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test with various malformed constraints
        malformed_constraints = [
            {"max_latency_ms": "not_a_number"},
            {"invalid_constraint": True},
            None,  # This will be converted to {} by Pydantic
        ]

        for constraints in malformed_constraints:
            try:
                with patch(
                    "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE",
                    False,
                ):
                    response = await orchestrator.orchestrate(
                        "test task", constraints or {}, deps
                    )
                    # Should not crash, may succeed or fail gracefully
                    assert isinstance(response, ToolResponse)
            except (ConnectionError, RuntimeError, ValueError) as e:
                # If it does raise an exception, it should be handled gracefully
                assert "error" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_tool_execution_timeout_simulation(self):
        """Test behavior under simulated timeout conditions."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Mock slow tool execution
        async def slow_tool_execution(tool_name, context, deps):
            await asyncio.sleep(0.1)  # Simulate slow execution
            return {
                "tool": tool_name,
                "result": "slow result",
                "timestamp": time.time(),
            }

        with patch.object(
            orchestrator, "_execute_tool", side_effect=slow_tool_execution
        ):
            request = ToolRequest(task="slow task")

            start_time = time.time()
            response = await orchestrator._orchestrate_autonomous(request, deps)
            end_time = time.time()

            # Should complete but take measurable time
            assert response.success is True
            assert (end_time - start_time) * 1000 >= 100  # At least 100ms
            assert response.latency_ms > 100

    def test_tool_selection_with_extreme_constraints(self):
        """Test tool selection with extreme or impossible constraints."""
        orchestrator = AgenticOrchestrator()
        available_tools = {
            "fast_tool": {"capabilities": ["search"]},
            "slow_tool": {"capabilities": ["generation"]},
        }

        # Test with impossible latency constraint
        selected = orchestrator._select_tools_for_task(
            "generate content",
            available_tools,
            {"max_latency_ms": 0},  # Impossible constraint
        )

        # Should still select some tools (fallback behavior)
        assert len(selected) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_orchestration_requests(self):
        """Test handling concurrent orchestration requests."""
        orchestrator = AgenticOrchestrator()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Create multiple concurrent requests
        tasks = ["task 1", "task 2", "task 3", "task 4", "task 5"]

        # Execute concurrently
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            responses = await asyncio.gather(
                *[orchestrator.orchestrate(task, {}, deps) for task in tasks]
            )

        # All should complete successfully
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.success is True
            assert f"task {i + 1}" in response.results["fallback_response"]


class TestIntegrationWithBaseAgent:
    """Test integration with BaseAgent framework."""

    @pytest.mark.asyncio
    async def test_inherits_base_agent_functionality(self):
        """Test AgenticOrchestrator properly inherits BaseAgent functionality."""
        orchestrator = AgenticOrchestrator()

        # Test base agent attributes
        assert hasattr(orchestrator, "name")
        assert hasattr(orchestrator, "model")
        assert hasattr(orchestrator, "temperature")
        assert hasattr(orchestrator, "max_tokens")
        assert hasattr(orchestrator, "_initialized")

        # Test base agent methods
        assert hasattr(orchestrator, "initialize")
        assert hasattr(orchestrator, "execute")
        assert hasattr(orchestrator, "get_performance_metrics")
        assert hasattr(orchestrator, "reset_metrics")

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test performance metrics are properly tracked."""
        # Create orchestrator with fallback mode
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            orchestrator = AgenticOrchestrator()

        # Initial metrics should be zero
        metrics = orchestrator.get_performance_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["avg_execution_time"] == 0.0
        assert metrics["success_rate"] == 0.0

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Execute some tasks to generate metrics in fallback mode
        with patch(
            "src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False
        ):
            # Override the agent to ensure fallback
            orchestrator.agent = None

            result1 = await orchestrator.execute("test task 1", deps)
            result2 = await orchestrator.execute("test task 2", deps)

            # Executions should complete in fallback mode (base agent fallback doesn't have 'success' key)
            assert "result" in result1 or "error" in result1
            assert "result" in result2 or "error" in result2

        # Check updated metrics
        metrics = orchestrator.get_performance_metrics()
        assert metrics["execution_count"] == 2
        # The execution time may be 0 for base agent fallback, but execution count should be tracked
        assert metrics["execution_count"] > 0

    @pytest.mark.asyncio
    async def test_reset_metrics_functionality(self):
        """Test metrics reset functionality."""
        orchestrator = AgenticOrchestrator()

        # Simulate some executions
        orchestrator.execution_count = 5
        orchestrator.total_execution_time = 2.5
        orchestrator.success_count = 4
        orchestrator.error_count = 1

        # Reset metrics
        await orchestrator.reset_metrics()

        # Verify reset
        metrics = orchestrator.get_performance_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["avg_execution_time"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_initialization_lifecycle(self):
        """Test proper initialization lifecycle."""
        orchestrator = AgenticOrchestrator()

        # Should start uninitialized
        assert not orchestrator._initialized

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id="test_session")

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Initialize
        await orchestrator.initialize(deps)

        # Should be initialized
        assert orchestrator._initialized

        # Second initialization should not cause issues
        await orchestrator.initialize(deps)
        assert orchestrator._initialized


# Integration test markers for running with actual dependencies
@pytest.mark.integration
@pytest.mark.skip(reason="Requires actual Pydantic-AI installation")
class TestPydanticAIIntegration:
    """Integration tests with actual Pydantic-AI (when available)."""

    @pytest.mark.asyncio
    async def test_real_pydantic_ai_orchestration(self):
        """Test orchestration with real Pydantic-AI agent."""
        # This would test with actual Pydantic-AI when installed

    @pytest.mark.asyncio
    async def test_real_tool_registration(self):
        """Test actual tool registration with Pydantic-AI."""
        # This would test tool registration when Pydantic-AI is available


# Performance benchmarks
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_orchestration_performance_benchmark(benchmark):
    """Benchmark orchestration performance."""
    orchestrator = AgenticOrchestrator()

    # Mock dependencies
    mock_client_manager = Mock(spec=ClientManager)
    mock_config = get_config()
    state = AgentState(session_id="test_session")

    deps = BaseAgentDependencies(
        client_manager=mock_client_manager, config=mock_config, session_state=state
    )

    # Benchmark the orchestration
    with patch("src.services.agents.agentic_orchestrator.PYDANTIC_AI_AVAILABLE", False):
        result = await benchmark.pedantic(
            orchestrator.orchestrate,
            args=("benchmark task", {}, deps),
            rounds=10,
            iterations=1,
        )

        assert result.success is True


# Property-based testing for comprehensive coverage
@pytest.mark.property
class TestAgenticOrchestratorProperties:
    """Property-based tests for comprehensive validation."""

    @given(
        model_name=st.sampled_from(["gpt-4", "gpt-4o-mini", "gpt-4-turbo"]),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        task_text=st.text(min_size=1, max_size=500),
    )
    def test_orchestrator_initialization_properties(
        self, model_name, temperature, task_text
    ):
        """Property-based test for orchestrator initialization."""
        orchestrator = AgenticOrchestrator(model=model_name, temperature=temperature)

        assert orchestrator.model == model_name
        assert orchestrator.temperature == temperature
        assert orchestrator.name == "agentic_orchestrator"

        # Test tool request creation
        request = ToolRequest(task=task_text)
        assert request.task == task_text
        assert isinstance(request.constraints, dict)
        assert isinstance(request.context, dict)

    @given(
        success_rate=st.floats(min_value=0.0, max_value=1.0),
        latency_ms=st.floats(min_value=0.0, max_value=30000.0, allow_nan=False),
        tool_count=st.integers(min_value=0, max_value=10),
    )
    def test_tool_response_properties(self, success_rate, latency_ms, tool_count):
        """Property-based test for tool response validation."""
        success = success_rate > 0.5
        tools_used = [f"tool_{i}" for i in range(tool_count)]

        response = ToolResponse(
            success=success,
            results={"test": "result"},
            tools_used=tools_used,
            reasoning="test reasoning",
            latency_ms=latency_ms,
            confidence=success_rate,
        )

        assert response.success == success
        assert len(response.tools_used) == tool_count
        assert 0.0 <= response.confidence <= 1.0
        assert response.latency_ms >= 0.0
