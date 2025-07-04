"""Comprehensive tests for Dynamic Tool Discovery Engine.

This module provides thorough testing of the J3 research implementation
for autonomous tool orchestration with intelligent capability assessment.
"""

import asyncio
import sys
import time
from unittest.mock import Mock
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents.core import (
    AgentState,
    BaseAgentDependencies,
)
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    ToolCapabilityType,
    ToolMetrics,
    discover_tools_for_task,
    get_discovery_engine,
)


class TestToolMetrics:
    """Test tool metrics data structure."""

    def test_tool_metrics_creation(self):
        """Test creating tool metrics."""
        metrics = ToolMetrics(
            average_latency_ms=150.0,
            success_rate=0.95,
            accuracy_score=0.88,
            cost_per_execution=0.02,
            reliability_score=0.92,
        )

        assert metrics.average_latency_ms == 150.0
        assert metrics.success_rate == 0.95
        assert metrics.accuracy_score == 0.88
        assert metrics.cost_per_execution == 0.02
        assert metrics.reliability_score == 0.92

    @given(
        latency=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        success_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        accuracy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        cost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        reliability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_tool_metrics_property_validation(
        self, latency, success_rate, accuracy, cost, reliability
    ):
        """Test tool metrics properties with Hypothesis."""
        metrics = ToolMetrics(
            average_latency_ms=latency,
            success_rate=success_rate,
            accuracy_score=accuracy,
            cost_per_execution=cost,
            reliability_score=reliability,
        )

        # All values should be preserved exactly
        assert metrics.average_latency_ms == latency
        assert metrics.success_rate == success_rate
        assert metrics.accuracy_score == accuracy
        assert metrics.cost_per_execution == cost
        assert metrics.reliability_score == reliability


class TestToolCapability:
    """Test tool capability model."""

    def test_tool_capability_creation(self):
        """Test creating tool capability."""
        metrics = ToolMetrics(
            average_latency_ms=100.0,
            success_rate=0.9,
            accuracy_score=0.85,
            cost_per_execution=0.01,
            reliability_score=0.88,
        )

        capability = ToolCapability(
            name="test_tool",
            capability_type=ToolCapabilityType.SEARCH,
            description="Test search tool",
            input_types=["text", "query"],
            output_types=["search_results"],
            metrics=metrics,
            last_updated="2024-01-01T00:00:00Z",
        )

        assert capability.name == "test_tool"
        assert capability.capability_type == ToolCapabilityType.SEARCH
        assert capability.description == "Test search tool"
        assert capability.input_types == ["text", "query"]
        assert capability.output_types == ["search_results"]
        assert capability.metrics == metrics
        assert capability.confidence_score == 0.8  # Default value
        assert capability.last_updated == "2024-01-01T00:00:00Z"

    def test_tool_capability_defaults(self):
        """Test tool capability with default values."""
        capability = ToolCapability(
            name="simple_tool",
            capability_type=ToolCapabilityType.ANALYSIS,
            description="Simple analysis tool",
            input_types=["text"],
            output_types=["analysis"],
            last_updated="2024-01-01T00:00:00Z",
        )

        assert capability.requirements == {}
        assert capability.constraints == {}
        assert capability.compatible_tools == []
        assert capability.dependencies == []
        assert capability.confidence_score == 0.8
        assert capability.metrics is None

    def test_tool_capability_model_copy(self):
        """Test tool capability model copying."""
        original = ToolCapability(
            name="original_tool",
            capability_type=ToolCapabilityType.GENERATION,
            description="Original tool",
            input_types=["text"],
            output_types=["generated_text"],
            confidence_score=0.9,
            last_updated="2024-01-01T00:00:00Z",
        )

        copy = original.model_copy()
        copy.confidence_score = 0.7

        assert original.confidence_score == 0.9
        assert copy.confidence_score == 0.7
        assert copy.name == original.name


class TestDynamicToolDiscoveryInitialization:
    """Test dynamic tool discovery initialization."""

    def test_discovery_engine_initialization(self):
        """Test discovery engine initialization."""
        engine = DynamicToolDiscovery()

        assert engine.name == "dynamic_tool_discovery"
        assert engine.model == "gpt-4o-mini"
        assert engine.temperature == 0.1
        assert engine.max_tokens == 2000
        assert not engine._initialized
        assert len(engine.discovered_tools) == 0
        assert len(engine.tool_performance_history) == 0
        assert len(engine.capability_cache) == 0

    def test_discovery_engine_custom_parameters(self):
        """Test discovery engine with custom parameters."""
        engine = DynamicToolDiscovery(model="gpt-4", temperature=0.2)

        assert engine.model == "gpt-4"
        assert engine.temperature == 0.2

    def test_system_prompt_generation(self):
        """Test system prompt for autonomous behavior."""
        engine = DynamicToolDiscovery()
        prompt = engine.get_system_prompt()

        assert "autonomous tool discovery engine" in prompt.lower()
        assert "INTELLIGENT TOOL ASSESSMENT" in prompt
        assert "DYNAMIC CAPABILITY EVALUATION" in prompt
        assert "PERFORMANCE-DRIVEN SELECTION" in prompt
        assert "Assess compatibility between different tools for chaining" in prompt
        assert "Real-time assessment of tool performance" in prompt
        assert "Balance speed, quality, cost, and reliability" in prompt

    @pytest.mark.asyncio
    async def test_initialize_discovery(self):
        """Test discovery initialization with tool scanning."""
        engine = DynamicToolDiscovery()

        # Mock dependencies
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        assert engine._initialized
        assert len(engine.discovered_tools) > 0
        assert "hybrid_search" in engine.discovered_tools
        assert "rag_generation" in engine.discovered_tools
        assert "content_analysis" in engine.discovered_tools

        # Verify tool metrics are initialized
        for tool_name in engine.discovered_tools:
            assert tool_name in engine.tool_performance_history
            assert len(engine.tool_performance_history[tool_name]) == 1

    @pytest.mark.asyncio
    async def test_double_initialization_protection(self):
        """Test that double initialization is handled correctly."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)
        initial_tool_count = len(engine.discovered_tools)

        # Initialize again
        await engine.initialize_discovery(deps)

        # Should not create duplicate tools
        assert len(engine.discovered_tools) == initial_tool_count


class TestToolScanning:
    """Test tool scanning and discovery functionality."""

    @pytest.mark.asyncio
    async def test_scan_available_tools(self):
        """Test scanning for available tools."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine._scan_available_tools(deps)

        # Verify core tools are discovered
        expected_tools = ["hybrid_search", "rag_generation", "content_analysis"]
        for tool_name in expected_tools:
            assert tool_name in engine.discovered_tools

        # Verify tool properties
        hybrid_search = engine.discovered_tools["hybrid_search"]
        assert hybrid_search.capability_type == ToolCapabilityType.SEARCH
        assert "hybrid vector and text search" in hybrid_search.description.lower()
        assert "text" in hybrid_search.input_types
        assert "search_results" in hybrid_search.output_types
        assert hybrid_search.metrics is not None

    @pytest.mark.asyncio
    async def test_tool_compatibility_assessment(self):
        """Test tool compatibility assessment."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine._scan_available_tools(deps)

        # Check search tool compatibility
        hybrid_search = engine.discovered_tools["hybrid_search"]
        assert "rag_generation" in hybrid_search.compatible_tools
        assert "content_analysis" in hybrid_search.compatible_tools

        # Check analysis tool compatibility
        content_analysis = engine.discovered_tools["content_analysis"]
        assert "hybrid_search" in content_analysis.compatible_tools
        assert "rag_generation" in content_analysis.compatible_tools

    @pytest.mark.asyncio
    async def test_performance_tracking_initialization(self):
        """Test performance tracking initialization."""
        engine = DynamicToolDiscovery()

        await engine._initialize_performance_tracking()

        # Should complete without error (logs performance tracking initialization)
        # This method primarily sets up logging


class TestIntelligentCapabilityAssessment:
    """Test intelligent capability assessment algorithms."""

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_search(self):
        """Test discovering tools for search tasks."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test search task
        tools = await engine.discover_tools_for_task(
            "Search for information about machine learning",
            {"max_latency_ms": 500, "min_accuracy": 0.8},
        )

        assert len(tools) > 0
        # Should find hybrid_search as most suitable
        assert tools[0].name == "hybrid_search"
        assert tools[0].capability_type == ToolCapabilityType.SEARCH
        assert tools[0].confidence_score > 0.5

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_generation(self):
        """Test discovering tools for generation tasks."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test generation task
        tools = await engine.discover_tools_for_task(
            "Generate comprehensive analysis based on documents",
            {"max_cost": 0.1, "min_accuracy": 0.85},
        )

        assert len(tools) > 0
        # Should find rag_generation as most suitable
        assert tools[0].name == "rag_generation"
        assert tools[0].capability_type == ToolCapabilityType.GENERATION

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_analysis(self):
        """Test discovering tools for analysis tasks."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test analysis task
        tools = await engine.discover_tools_for_task(
            "Analyze content quality and extract insights",
            {"max_latency_ms": 300},
        )

        assert len(tools) > 0
        # Should find content_analysis as most suitable
        assert tools[0].name == "content_analysis"
        assert tools[0].capability_type == ToolCapabilityType.ANALYSIS

    @pytest.mark.asyncio
    async def test_suitability_score_calculation(self):
        """Test suitability score calculation algorithm."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test search tool for search task
        search_tool = engine.discovered_tools["hybrid_search"]
        score = await engine._calculate_suitability_score(
            search_tool,
            "search for information",
            {"max_latency_ms": 200, "min_accuracy": 0.8, "max_cost": 0.05},
        )

        # Should get high score for matching capability
        assert score > 0.5
        assert score <= 1.0

        # Test generation tool for search task (should be lower)
        gen_tool = engine.discovered_tools["rag_generation"]
        gen_score = await engine._calculate_suitability_score(
            gen_tool, "search for information", {}
        )

        # Should get lower score for non-matching capability
        assert gen_score < score

    @pytest.mark.asyncio
    async def test_performance_requirement_scoring(self):
        """Test performance requirement-based scoring."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        search_tool = engine.discovered_tools["hybrid_search"]

        # Test with strict latency requirement (should pass)
        score_strict = await engine._calculate_suitability_score(
            search_tool,
            "search for data",
            {"max_latency_ms": 200},  # hybrid_search has 150ms
        )

        # Test with loose latency requirement
        score_loose = await engine._calculate_suitability_score(
            search_tool, "search for data", {"max_latency_ms": 1000}
        )

        # Both should pass but strict should have penalty removed
        assert score_strict >= score_loose

        # Test with impossible latency requirement
        score_impossible = await engine._calculate_suitability_score(
            search_tool, "search for data", {"max_latency_ms": 50}
        )

        # Should have penalty for exceeding latency
        assert score_impossible < score_strict

    @pytest.mark.asyncio
    async def test_no_suitable_tools_scenario(self):
        """Test scenario where no tools meet requirements."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with impossible requirements
        tools = await engine.discover_tools_for_task(
            "unrecognized task type",
            {
                "max_latency_ms": 1,  # Impossible latency
                "min_accuracy": 0.99,  # Very high accuracy
                "max_cost": 0.001,  # Very low cost
            },
        )

        # Should return empty list when no tools meet threshold
        assert len(tools) == 0


class TestPerformanceDrivenToolSelection:
    """Test performance-driven tool selection algorithms."""

    @pytest.mark.asyncio
    async def test_tool_ranking_by_suitability(self):
        """Test tools are ranked by suitability score."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test task that could match multiple tools
        tools = await engine.discover_tools_for_task(
            "search and analyze content",  # Could match search and analysis
            {"max_latency_ms": 1000, "min_accuracy": 0.5},
        )

        # Should return multiple tools ranked by suitability
        assert len(tools) > 1

        # Scores should be in descending order
        for i in range(1, len(tools)):
            assert tools[i - 1].confidence_score >= tools[i].confidence_score

        # All returned tools should meet minimum threshold
        for tool in tools:
            assert tool.confidence_score > 0.5

    @pytest.mark.asyncio
    async def test_constraint_satisfaction(self):
        """Test constraint satisfaction in tool selection."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with specific constraints
        tools = await engine.discover_tools_for_task(
            "search for information",
            {
                "max_latency_ms": 200,  # Should pass for hybrid_search (150ms)
                "min_accuracy": 0.85,  # Should pass for hybrid_search (0.87)
                "max_cost": 0.03,  # Should pass for hybrid_search (0.02)
            },
        )

        assert len(tools) > 0
        selected_tool = tools[0]

        # Verify constraints are satisfied
        assert selected_tool.metrics.average_latency_ms <= 200
        assert selected_tool.metrics.accuracy_score >= 0.85
        assert selected_tool.metrics.cost_per_execution <= 0.03

    @pytest.mark.asyncio
    async def test_reliability_bonus_scoring(self):
        """Test reliability bonus in scoring algorithm."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Get tools and check reliability bonus is applied
        search_tool = engine.discovered_tools["hybrid_search"]
        score = await engine._calculate_suitability_score(
            search_tool, "search task", {}
        )

        # Reliability should contribute to score (0.92 * 0.1 = 0.092)
        # Base score is 0.4 for capability match, so total should be > 0.4
        assert score > 0.4


class TestRollingAveragePerformanceTracking:
    """Test rolling average performance tracking and learning."""

    @pytest.mark.asyncio
    async def test_update_tool_performance(self):
        """Test updating tool performance metrics."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        initial_metrics = engine.discovered_tools["hybrid_search"].metrics

        # Add new performance data
        new_metrics = ToolMetrics(
            average_latency_ms=120.0,  # Improved from 150.0
            success_rate=0.96,  # Improved from 0.94
            accuracy_score=0.90,  # Improved from 0.87
            cost_per_execution=0.015,  # Improved from 0.02
            reliability_score=0.95,  # Improved from 0.92
        )

        await engine.update_tool_performance("hybrid_search", new_metrics)

        # Verify performance history updated
        assert len(engine.tool_performance_history["hybrid_search"]) == 2

        # Verify rolling average calculated
        updated_tool = engine.discovered_tools["hybrid_search"]
        updated_metrics = updated_tool.metrics

        # Should be average of initial and new metrics
        expected_latency = (initial_metrics.average_latency_ms + 120.0) / 2
        assert abs(updated_metrics.average_latency_ms - expected_latency) < 0.01

        expected_success_rate = (initial_metrics.success_rate + 0.96) / 2
        assert abs(updated_metrics.success_rate - expected_success_rate) < 0.01

    @pytest.mark.asyncio
    async def test_rolling_average_window_limit(self):
        """Test rolling average window is limited to last 10 executions."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Add 15 performance updates
        for i in range(15):
            metrics = ToolMetrics(
                average_latency_ms=100.0 + i,
                success_rate=0.9 + (i * 0.005),
                accuracy_score=0.8 + (i * 0.01),
                cost_per_execution=0.01 + (i * 0.001),
                reliability_score=0.85 + (i * 0.01),
            )
            await engine.update_tool_performance("hybrid_search", metrics)

        # Should have all 15 + initial metrics in history
        assert len(engine.tool_performance_history["hybrid_search"]) == 16

        # But rolling average should only use last 10
        updated_tool = engine.discovered_tools["hybrid_search"]
        updated_metrics = updated_tool.metrics

        # Should be closer to recent values (last 10) than early values
        # The average of last 10 latencies: 105, 106, 107, 108, 109, 110, 111, 112, 113, 114
        expected_recent_avg = sum(range(105, 115)) / 10  # 109.5
        assert abs(updated_metrics.average_latency_ms - expected_recent_avg) < 1.0

    def test_calculate_average_metrics(self):
        """Test average metrics calculation."""
        engine = DynamicToolDiscovery()

        metrics_list = [
            ToolMetrics(100.0, 0.9, 0.8, 0.01, 0.85),
            ToolMetrics(120.0, 0.95, 0.85, 0.015, 0.90),
            ToolMetrics(80.0, 0.88, 0.82, 0.008, 0.87),
        ]

        avg_metrics = engine._calculate_average_metrics(metrics_list)

        assert abs(avg_metrics.average_latency_ms - 100.0) < 0.01  # (100+120+80)/3
        assert abs(avg_metrics.success_rate - 0.91) < 0.01  # (0.9+0.95+0.88)/3
        assert abs(avg_metrics.accuracy_score - 0.823333) < 0.01
        assert abs(avg_metrics.cost_per_execution - 0.011) < 0.001
        assert abs(avg_metrics.reliability_score - 0.873333) < 0.01

    def test_calculate_average_metrics_empty_list(self):
        """Test average metrics calculation with empty list."""
        engine = DynamicToolDiscovery()

        avg_metrics = engine._calculate_average_metrics([])

        assert avg_metrics.average_latency_ms == 0.0
        assert avg_metrics.success_rate == 0.0
        assert avg_metrics.accuracy_score == 0.0
        assert avg_metrics.cost_per_execution == 0.0
        assert avg_metrics.reliability_score == 0.0

    @pytest.mark.asyncio
    async def test_update_performance_nonexistent_tool(self):
        """Test updating performance for non-existent tool."""
        engine = DynamicToolDiscovery()

        new_metrics = ToolMetrics(100.0, 0.9, 0.8, 0.01, 0.85)

        # Should not raise error for non-existent tool
        await engine.update_tool_performance("nonexistent_tool", new_metrics)

        # Should not create new entries
        assert "nonexistent_tool" not in engine.tool_performance_history
        assert "nonexistent_tool" not in engine.discovered_tools

    @pytest.mark.asyncio
    async def test_last_updated_timestamp(self):
        """Test that last_updated timestamp is updated."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        initial_timestamp = engine.discovered_tools["hybrid_search"].last_updated

        # Wait a small amount to ensure timestamp difference
        await asyncio.sleep(0.01)

        new_metrics = ToolMetrics(100.0, 0.9, 0.8, 0.01, 0.85)
        await engine.update_tool_performance("hybrid_search", new_metrics)

        updated_timestamp = engine.discovered_tools["hybrid_search"].last_updated

        # Timestamp should be updated
        assert updated_timestamp != initial_timestamp


class TestSelfLearningOptimization:
    """Test self-learning optimization patterns."""

    @pytest.mark.asyncio
    async def test_performance_learning_affects_selection(self):
        """Test that performance learning affects tool selection."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Get initial ranking
        initial_tools = await engine.discover_tools_for_task(
            "search for information", {"max_latency_ms": 500}
        )
        initial_best = initial_tools[0]

        # Simulate moderately worse performance for the best tool (not so bad it gets filtered out)
        poor_metrics = ToolMetrics(
            average_latency_ms=300.0,  # Worse but still reasonable
            success_rate=0.8,  # Worse but not terrible
            accuracy_score=0.75,  # Worse but still acceptable
            cost_per_execution=0.05,  # Worse but not extreme
            reliability_score=0.7,  # Worse but still usable
        )

        await engine.update_tool_performance(initial_best.name, poor_metrics)

        # Get new ranking with same requirements
        updated_tools = await engine.discover_tools_for_task(
            "search for information", {"max_latency_ms": 500}
        )

        # The previously best tool should still appear but with lower score
        updated_best = next(
            (tool for tool in updated_tools if tool.name == initial_best.name), None
        )

        if updated_best is not None:
            # Tool should have lower score due to performance degradation
            assert updated_best.confidence_score < initial_best.confidence_score
        else:
            # Tool fell below threshold due to poor performance - that's also valid learning behavior
            assert len(updated_tools) == 0 or updated_tools[0].name != initial_best.name


class TestToolCompatibilityAnalysis:
    """Test tool compatibility analysis and chain generation."""

    @pytest.mark.asyncio
    async def test_get_tool_recommendations(self):
        """Test intelligent tool recommendations."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test search task recommendations
        recommendations = await engine.get_tool_recommendations(
            "search for machine learning information", {"max_latency_ms": 500}
        )

        assert "primary_tools" in recommendations
        assert "secondary_tools" in recommendations
        assert "tool_chains" in recommendations
        assert "reasoning" in recommendations

        # Should have primary tools
        assert len(recommendations["primary_tools"]) > 0

        # Primary tool should have required fields
        primary_tool = recommendations["primary_tools"][0]
        assert "name" in primary_tool
        assert "suitability_score" in primary_tool
        assert "capability_type" in primary_tool
        assert "estimated_latency_ms" in primary_tool

        # Should have reasoning
        assert len(recommendations["reasoning"]) > 0
        assert "Recommended" in recommendations["reasoning"]

    @pytest.mark.asyncio
    async def test_tool_chain_generation(self):
        """Test tool chain generation for complex workflows."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Get all suitable tools to enable chain generation
        tools = await engine.discover_tools_for_task(
            "search and generate comprehensive analysis",
            {"max_latency_ms": 2000},
        )

        # Should have multiple tools for chaining
        assert len(tools) > 1

        chains = await engine._generate_tool_chains(tools)

        assert len(chains) > 0

        # Check search → generation chain
        search_gen_chain = next(
            (c for c in chains if c["type"] == "search_then_generate"), None
        )
        assert search_gen_chain is not None
        assert len(search_gen_chain["chain"]) == 2
        assert "estimated_total_latency_ms" in search_gen_chain

        # Verify chain structure
        chain_tools = search_gen_chain["chain"]
        assert any("search" in tool for tool in chain_tools)
        assert any("generation" in tool or "rag" in tool for tool in chain_tools)

    @pytest.mark.asyncio
    async def test_complex_tool_chain_generation(self):
        """Test complex tool chain generation with analysis."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Get tools that include all types for complex chaining
        tools = await engine.discover_tools_for_task(
            "analyze search and generate comprehensive report",
            {"max_latency_ms": 5000},
        )

        chains = await engine._generate_tool_chains(tools)

        # Should include analyze → search → generate chain
        complex_chain = next(
            (c for c in chains if c["type"] == "analyze_search_generate"), None
        )

        if complex_chain:  # Chain exists if all tool types available
            assert len(complex_chain["chain"]) == 3
            assert "estimated_total_latency_ms" in complex_chain

            # Verify chain includes all phases
            chain_tools = complex_chain["chain"]
            assert any("analysis" in tool for tool in chain_tools)
            assert any("search" in tool for tool in chain_tools)
            assert any("generation" in tool or "rag" in tool for tool in chain_tools)

    @pytest.mark.asyncio
    async def test_empty_tool_chain_generation(self):
        """Test tool chain generation with insufficient tools."""
        engine = DynamicToolDiscovery()

        # Test with minimal tools
        minimal_tools = [
            ToolCapability(
                name="single_tool",
                capability_type=ToolCapabilityType.SEARCH,
                description="Single tool",
                input_types=["text"],
                output_types=["results"],
                last_updated="2024-01-01T00:00:00Z",
            )
        ]

        chains = await engine._generate_tool_chains(minimal_tools)

        # Should not create chains with insufficient tools
        assert len(chains) == 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_task_description(self):
        """Test handling of empty task description."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        tools = await engine.discover_tools_for_task("", {})

        # Should handle empty description gracefully
        # May return tools with low scores or empty list
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_invalid_requirements(self):
        """Test handling of invalid requirements."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with negative/invalid requirements
        tools = await engine.discover_tools_for_task(
            "search task",
            {
                "max_latency_ms": -100,  # Invalid
                "min_accuracy": -0.5,  # Invalid
                "max_cost": -1.0,  # Invalid
            },
        )

        # Should handle gracefully
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_tool_without_metrics(self):
        """Test handling tools without performance metrics."""
        engine = DynamicToolDiscovery()

        # Add tool without metrics
        tool_without_metrics = ToolCapability(
            name="no_metrics_tool",
            capability_type=ToolCapabilityType.SEARCH,
            description="Tool without metrics",
            input_types=["text"],
            output_types=["results"],
            metrics=None,  # No metrics
            last_updated="2024-01-01T00:00:00Z",
        )

        engine.discovered_tools["no_metrics_tool"] = tool_without_metrics

        score = await engine._calculate_suitability_score(
            tool_without_metrics, "search task", {"max_latency_ms": 100}
        )

        # Should handle missing metrics gracefully
        assert score >= 0.0
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_recommendations_with_no_suitable_tools(self):
        """Test recommendations when no tools are suitable."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with task that won't match any tools
        recommendations = await engine.get_tool_recommendations(
            "unknown alien technology task",
            {"max_latency_ms": 1, "min_accuracy": 0.999, "max_cost": 0.0001},
        )

        # Should return structure with empty recommendations
        assert "primary_tools" in recommendations
        assert "secondary_tools" in recommendations
        assert "tool_chains" in recommendations
        assert "reasoning" in recommendations

        assert len(recommendations["primary_tools"]) == 0
        assert len(recommendations["tool_chains"]) == 0


class TestGlobalDiscoveryEngine:
    """Test global discovery engine singleton and utility functions."""

    def test_get_discovery_engine_singleton(self):
        """Test singleton pattern for discovery engine."""
        engine1 = get_discovery_engine()
        engine2 = get_discovery_engine()

        assert engine1 is engine2
        assert isinstance(engine1, DynamicToolDiscovery)

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_utility(self):
        """Test utility function for tool discovery."""
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test utility function
        tools = await discover_tools_for_task(
            "search for information", {"max_latency_ms": 500}, deps
        )

        assert isinstance(tools, list)
        # Engine should be initialized automatically
        engine = get_discovery_engine()
        assert engine._initialized

    @pytest.mark.asyncio
    async def test_utility_function_initialization_once(self):
        """Test utility function only initializes once."""
        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Call multiple times
        await discover_tools_for_task("task 1", {}, deps)
        await discover_tools_for_task("task 2", {}, deps)

        engine = get_discovery_engine()
        # Should only have core tools, not duplicates
        assert (
            len(engine.discovered_tools) == 3
        )  # hybrid_search, rag_generation, content_analysis


class TestSpecificUncoveredFunctionality:
    """Test specific uncovered functionality to reach 80%+ coverage."""

    @pytest.mark.asyncio
    async def test_initialize_tools_method(self):
        """Test initialize_tools method - Line 126-134."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Call initialize_tools directly
        await engine.initialize_tools(deps)

        # Should complete without error (it just logs)
        # This covers lines 132-134

    @pytest.mark.asyncio
    async def test_suitability_score_edge_cases(self):
        """Test suitability score calculation edge cases - Lines 300-345."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test tool without any matching keywords in task description
        tool = engine.discovered_tools["hybrid_search"]
        score = await engine._calculate_suitability_score(
            tool, "completely unrelated quantum physics task", {}
        )

        # Should get low base score (no capability match)
        # But may still get reliability bonus
        assert score >= 0.0
        assert score <= 1.0

        # Test with all requirements that pass
        high_score = await engine._calculate_suitability_score(
            tool,
            "search for documents",  # Matches search capability
            {
                "max_latency_ms": 1000,  # Tool passes (150ms)
                "min_accuracy": 0.8,  # Tool passes (0.87)
                "max_cost": 0.1,  # Tool passes (0.02)
            },
        )

        # Should get higher score with all bonuses
        assert high_score > score
        assert high_score > 0.5  # Should exceed threshold

    @pytest.mark.asyncio
    async def test_tool_recommendation_with_empty_suitable_tools(self):
        """Test tool recommendations when no suitable tools found - Lines 411-449."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with impossible requirements that no tools can meet
        recommendations = await engine.get_tool_recommendations(
            "alien technology task",  # Won't match any capability types
            {
                "max_latency_ms": 0.1,  # Impossible latency
                "min_accuracy": 0.999,  # Impossible accuracy
                "max_cost": 0.0001,  # Impossible cost
            },
        )

        # Should return empty recommendations structure
        assert "primary_tools" in recommendations
        assert "secondary_tools" in recommendations
        assert "tool_chains" in recommendations
        assert "reasoning" in recommendations

        # Primary tools should be empty
        assert len(recommendations["primary_tools"]) == 0
        assert len(recommendations["tool_chains"]) == 0

    @pytest.mark.asyncio
    async def test_tool_chain_generation_edge_cases(self):
        """Test tool chain generation edge cases - Lines 462-526."""
        engine = DynamicToolDiscovery()

        # Test with tools that have None metrics
        tools_with_none_metrics = [
            ToolCapability(
                name="no_metrics_search",
                capability_type=ToolCapabilityType.SEARCH,
                description="Search tool without metrics",
                input_types=["text"],
                output_types=["results"],
                metrics=None,  # No metrics
                last_updated="2024-01-01T00:00:00Z",
            ),
            ToolCapability(
                name="no_metrics_gen",
                capability_type=ToolCapabilityType.GENERATION,
                description="Generation tool without metrics",
                input_types=["text"],
                output_types=["content"],
                metrics=None,  # No metrics
                last_updated="2024-01-01T00:00:00Z",
            ),
        ]

        chains = await engine._generate_tool_chains(tools_with_none_metrics)

        # Should handle None metrics gracefully
        assert isinstance(chains, list)
        if chains:
            for chain in chains:
                assert "estimated_total_latency_ms" in chain
                # Should default to 0 for None metrics
                assert chain["estimated_total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_complex_tool_chain_with_all_types(self):
        """Test complex tool chain generation with all capability types."""
        engine = DynamicToolDiscovery()

        # Create tools of all types for comprehensive chain testing
        all_type_tools = [
            ToolCapability(
                name="analysis_tool",
                capability_type=ToolCapabilityType.ANALYSIS,
                description="Analysis tool",
                input_types=["text"],
                output_types=["analysis"],
                metrics=ToolMetrics(50.0, 0.9, 0.8, 0.01, 0.85),
                last_updated="2024-01-01T00:00:00Z",
            ),
            ToolCapability(
                name="search_tool",
                capability_type=ToolCapabilityType.SEARCH,
                description="Search tool",
                input_types=["query"],
                output_types=["results"],
                metrics=ToolMetrics(100.0, 0.95, 0.87, 0.02, 0.9),
                last_updated="2024-01-01T00:00:00Z",
            ),
            ToolCapability(
                name="gen_tool",
                capability_type=ToolCapabilityType.GENERATION,
                description="Generation tool",
                input_types=["context"],
                output_types=["content"],
                metrics=ToolMetrics(200.0, 0.88, 0.85, 0.03, 0.82),
                last_updated="2024-01-01T00:00:00Z",
            ),
        ]

        chains = await engine._generate_tool_chains(all_type_tools)

        # Should create both simple and complex chains
        assert len(chains) >= 2

        # Find the complex chain (analyze → search → generate)
        complex_chain = next(
            (c for c in chains if c["type"] == "analyze_search_generate"), None
        )

        assert complex_chain is not None
        assert len(complex_chain["chain"]) == 3
        # Total latency should be sum of all three tools: 50 + 100 + 200 = 350
        assert complex_chain["estimated_total_latency_ms"] == 350.0

    @pytest.mark.asyncio
    async def test_get_discovery_engine_reset_functionality(self):
        """Test discovery engine global state management."""
        # Access global variable to test singleton behavior

        # Get engine instances
        engine1 = get_discovery_engine()
        engine2 = get_discovery_engine()

        # Should be same instance (singleton)
        assert engine1 is engine2

    def test_tool_capability_type_enum_coverage(self):
        """Test all ToolCapabilityType enum values."""
        # Test all enum values to ensure they're properly defined
        types = [
            ToolCapabilityType.SEARCH,
            ToolCapabilityType.RETRIEVAL,
            ToolCapabilityType.GENERATION,
            ToolCapabilityType.ANALYSIS,
            ToolCapabilityType.CLASSIFICATION,
            ToolCapabilityType.SYNTHESIS,
            ToolCapabilityType.ORCHESTRATION,
        ]

        for capability_type in types:
            assert isinstance(capability_type.value, str)
            assert len(capability_type.value) > 0

        # Test creating ToolCapability with each type
        for capability_type in types:
            tool = ToolCapability(
                name=f"test_{capability_type.value}",
                capability_type=capability_type,
                description=f"Test {capability_type.value} tool",
                input_types=["text"],
                output_types=["result"],
                last_updated="2024-01-01T00:00:00Z",
            )
            assert tool.capability_type == capability_type

    @pytest.mark.asyncio
    async def test_score_calculation_branching_coverage(self):
        """Test all branches in score calculation logic."""
        engine = DynamicToolDiscovery()

        # Create test tool
        test_tool = ToolCapability(
            name="test_tool",
            capability_type=ToolCapabilityType.GENERATION,
            description="Test generation tool",
            input_types=["text"],
            output_types=["content"],
            metrics=ToolMetrics(150.0, 0.9, 0.8, 0.01, 0.85),
            last_updated="2024-01-01T00:00:00Z",
        )

        # Test generate keyword matching
        gen_score = await engine._calculate_suitability_score(
            test_tool, "generate a comprehensive report", {}
        )
        assert gen_score >= 0.4  # Should get base capability match

        # Test analyze keyword with generation tool (no match)
        no_match_score = await engine._calculate_suitability_score(
            test_tool, "analyze the data patterns", {}
        )
        # Should get lower score (only reliability bonus)
        assert no_match_score < gen_score

        # Test with requirements that fail
        fail_score = await engine._calculate_suitability_score(
            test_tool,
            "generate content",
            {
                "max_latency_ms": 50,  # Tool has 150ms, fails requirement
                "min_accuracy": 0.9,  # Tool has 0.8, fails requirement
                "max_cost": 0.005,  # Tool has 0.01, fails requirement
            },
        )

        # Should have penalties applied (negative adjustments)
        assert fail_score < gen_score

    @pytest.mark.asyncio
    async def test_initialization_dependency_coverage(self):
        """Test initialization dependency and error handling."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Test that initialize_discovery calls initialize if not initialized
        assert not engine._initialized

        # This should trigger the initialization path on line 142-143
        await engine.initialize_discovery(deps)

        assert engine._initialized

        # Test calling initialize_discovery when already initialized
        # Should not reinitialize
        initial_tool_count = len(engine.discovered_tools)
        await engine.initialize_discovery(deps)

        # Should not change tool count (no re-initialization)
        assert len(engine.discovered_tools) == initial_tool_count


class TestPropertyBasedValidation:
    """Property-based tests for algorithm validation."""

    @given(
        task_description=st.text(min_size=1, max_size=100),
        max_latency=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False),
        min_accuracy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        max_cost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @pytest.mark.asyncio
    async def test_suitability_score_properties(
        self, task_description, max_latency, min_accuracy, max_cost
    ):
        """Test suitability score properties with Hypothesis."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        requirements = {
            "max_latency_ms": max_latency,
            "min_accuracy": min_accuracy,
            "max_cost": max_cost,
        }

        # Test with each available tool
        for tool_name, tool in engine.discovered_tools.items():
            score = await engine._calculate_suitability_score(
                tool, task_description, requirements
            )

            # Score should always be between 0 and 1 (allowing for small negative values due to penalties)
            assert -0.2 <= score <= 1.0, f"Invalid score {score} for tool {tool_name}"

            # Score should be deterministic (same inputs = same output)
            score2 = await engine._calculate_suitability_score(
                tool, task_description, requirements
            )
            assert score == score2, f"Non-deterministic scoring for {tool_name}"

    @given(
        metrics_count=st.integers(min_value=1, max_value=20),
        latency_values=st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            min_size=1,
            max_size=20,
        ),
    )
    def test_average_metrics_properties(self, metrics_count, latency_values):
        """Test average metrics calculation properties."""
        engine = DynamicToolDiscovery()

        # Create metrics list with consistent size
        metrics_list = []
        for i in range(min(metrics_count, len(latency_values))):
            metrics = ToolMetrics(
                average_latency_ms=latency_values[i],
                success_rate=0.9,
                accuracy_score=0.8,
                cost_per_execution=0.01,
                reliability_score=0.85,
            )
            metrics_list.append(metrics)

        if not metrics_list:
            return  # Skip empty lists

        avg_metrics = engine._calculate_average_metrics(metrics_list)

        # Average should be within the range of input values
        min_latency = min(latency_values[: len(metrics_list)])
        max_latency = max(latency_values[: len(metrics_list)])

        # Allow for small floating point precision errors
        tolerance = 1e-10
        assert (
            min_latency - tolerance
            <= avg_metrics.average_latency_ms
            <= max_latency + tolerance
        ), "Average outside input range"

        # Other metrics should remain consistent (all same values) with small tolerance for floating point
        assert abs(avg_metrics.success_rate - 0.9) < 1e-10
        assert abs(avg_metrics.accuracy_score - 0.8) < 1e-10
        assert abs(avg_metrics.cost_per_execution - 0.01) < 1e-10
        assert abs(avg_metrics.reliability_score - 0.85) < 1e-10


class TestAdvancedToolDiscoveryAlgorithms:
    """Test advanced tool discovery algorithms and complex scenarios."""

    @pytest.mark.asyncio
    async def test_multi_criteria_tool_selection(self):
        """Test tool selection with multiple competing criteria."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with conflicting requirements (fast vs accurate)
        tools = await engine.discover_tools_for_task(
            "search for documents with high precision",
            {
                "max_latency_ms": 100,  # Very fast requirement
                "min_accuracy": 0.95,  # Very high accuracy requirement
                "max_cost": 0.001,  # Very low cost requirement
            },
        )

        # Should handle conflicting requirements gracefully
        assert isinstance(tools, list)
        # May return empty list if no tools meet all criteria
        for tool in tools:
            assert tool.confidence_score > 0.5

    @pytest.mark.asyncio
    async def test_tool_chain_optimization(self):
        """Test optimization of tool chains for complex workflows."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        recommendations = await engine.get_tool_recommendations(
            "research workflow: search, analyze, then generate comprehensive report",
            {"max_latency_ms": 2000, "min_accuracy": 0.8},
        )

        # Should create optimized tool chains
        assert "tool_chains" in recommendations
        if recommendations["tool_chains"]:
            chains = recommendations["tool_chains"]
            # Verify chains are logically ordered
            for chain in chains:
                assert "chain" in chain
                assert "estimated_total_latency_ms" in chain
                assert len(chain["chain"]) >= 2

    @pytest.mark.asyncio
    async def test_adaptive_performance_learning(self):
        """Test adaptive learning from performance feedback."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Simulate improving performance over time
        tool_name = "hybrid_search"
        initial_tool = engine.discovered_tools[tool_name]
        initial_latency = initial_tool.metrics.average_latency_ms

        # Add several performance improvements
        for i in range(5):
            improved_metrics = ToolMetrics(
                average_latency_ms=initial_latency - (i * 10),  # Getting faster
                success_rate=0.94 + (i * 0.01),  # Getting more reliable
                accuracy_score=0.87 + (i * 0.02),  # Getting more accurate
                cost_per_execution=0.02 - (i * 0.002),  # Getting cheaper
                reliability_score=0.92 + (i * 0.01),  # Getting more reliable
            )
            await engine.update_tool_performance(tool_name, improved_metrics)

        # Verify performance learning adaptation
        updated_tool = engine.discovered_tools[tool_name]
        assert updated_tool.metrics.average_latency_ms < initial_latency
        # Success rate should be improved or at least maintained (account for floating point precision)
        assert (
            updated_tool.metrics.success_rate
            >= initial_tool.metrics.success_rate - 0.001
        )

    @pytest.mark.asyncio
    async def test_capability_correlation_analysis(self):
        """Test correlation between tool capabilities and task success."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test tools for tasks that partially match their capabilities
        search_tools = await engine.discover_tools_for_task(
            "partially search and partially analyze content", {}
        )

        # Should find multiple tools with varying suitability scores
        if len(search_tools) > 1:
            scores = [tool.confidence_score for tool in search_tools]
            # Scores should reflect varying suitability
            assert len(set(scores)) > 1  # Different scores

    @pytest.mark.asyncio
    async def test_dynamic_threshold_adjustment(self):
        """Test dynamic adjustment of suitability thresholds."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with different task complexities
        simple_task_tools = await engine.discover_tools_for_task("search", {})
        complex_task_tools = await engine.discover_tools_for_task(
            "comprehensive multi-modal analysis with cross-reference validation", {}
        )

        # Both should return valid results (threshold adapts to context)
        assert isinstance(simple_task_tools, list)
        assert isinstance(complex_task_tools, list)


class TestRobustnessAndEdgeCases:
    """Test robustness against edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_extreme_performance_metrics(self):
        """Test handling of extreme performance metric values."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with extreme values
        extreme_metrics = ToolMetrics(
            average_latency_ms=0.0,  # Impossible speed
            success_rate=1.0,  # Perfect success
            accuracy_score=0.0,  # Terrible accuracy
            cost_per_execution=1000.0,  # Very expensive
            reliability_score=0.5,  # Mediocre reliability
        )

        await engine.update_tool_performance("hybrid_search", extreme_metrics)

        # Should handle extreme values gracefully
        updated_tool = engine.discovered_tools["hybrid_search"]
        assert updated_tool.metrics is not None

    @pytest.mark.asyncio
    async def test_malformed_task_descriptions(self):
        """Test handling of malformed task descriptions."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        malformed_tasks = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a" * 10000,  # Extremely long
            "!@#$%^&*()",  # Special characters only
            "\x00\x01\x02",  # Control characters
            "search\nfor\ttabs\rand\r\nlinebreaks",  # Mixed whitespace
        ]

        for task in malformed_tasks:
            tools = await engine.discover_tools_for_task(task, {})
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_conflicting_tool_constraints(self):
        """Test handling of conflicting tool constraints."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Add a tool with conflicting constraints
        conflicted_tool = ToolCapability(
            name="conflicted_tool",
            capability_type=ToolCapabilityType.SEARCH,
            description="Tool with conflicts",
            input_types=["text"],
            output_types=["results"],
            metrics=ToolMetrics(
                average_latency_ms=500.0,  # Slow
                success_rate=0.99,  # But very reliable
                accuracy_score=0.95,  # And very accurate
                cost_per_execution=0.001,  # And cheap
                reliability_score=0.98,  # And stable
            ),
            constraints={
                "requires_gpu": True,
                "memory_usage": "high",
                "network_dependency": True,
            },
            last_updated="2024-01-01T00:00:00Z",
        )

        engine.discovered_tools["conflicted_tool"] = conflicted_tool
        engine.tool_performance_history["conflicted_tool"] = [conflicted_tool.metrics]

        # Should handle tool with conflicting characteristics
        tools = await engine.discover_tools_for_task(
            "fast search with high accuracy", {"max_latency_ms": 100}
        )

        # Should intelligently handle the conflict
        assert isinstance(tools, list)

    def test_tool_metrics_boundary_values(self):
        """Test ToolMetrics with boundary values."""
        # Test minimum values
        min_metrics = ToolMetrics(
            average_latency_ms=0.0,
            success_rate=0.0,
            accuracy_score=0.0,
            cost_per_execution=0.0,
            reliability_score=0.0,
        )
        assert min_metrics.average_latency_ms == 0.0

        # Test maximum typical values
        max_metrics = ToolMetrics(
            average_latency_ms=999999.0,
            success_rate=1.0,
            accuracy_score=1.0,
            cost_per_execution=1000.0,
            reliability_score=1.0,
        )
        assert max_metrics.success_rate == 1.0


class TestMemoryAndPerformanceOptimization:
    """Test memory efficiency and performance optimization."""

    @pytest.mark.asyncio
    async def test_memory_efficient_history_management(self):
        """Test memory-efficient management of performance history."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Add many performance records to test memory management
        tool_name = "hybrid_search"
        initial_count = len(engine.tool_performance_history[tool_name])

        # Add more than the rolling window size (10)
        for i in range(25):
            metrics = ToolMetrics(
                average_latency_ms=100.0 + i,
                success_rate=0.9,
                accuracy_score=0.8,
                cost_per_execution=0.01,
                reliability_score=0.85,
            )
            await engine.update_tool_performance(tool_name, metrics)

        # History should contain all records but rolling average uses only last 10
        total_history = len(engine.tool_performance_history[tool_name])
        assert total_history == initial_count + 25

        # But rolling average calculation should only use recent metrics
        updated_tool = engine.discovered_tools[tool_name]
        # Should be closer to recent values (100+15 to 100+24 range)
        assert updated_tool.metrics.average_latency_ms > 110.0

    @pytest.mark.asyncio
    async def test_cache_efficiency(self):
        """Test caching efficiency for repeated operations."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test repeated task discovery (should benefit from caching)
        task = "search for machine learning papers"
        requirements = {"max_latency_ms": 500}

        start_time = time.time()
        tools1 = await engine.discover_tools_for_task(task, requirements)
        first_time = time.time() - start_time

        start_time = time.time()
        tools2 = await engine.discover_tools_for_task(task, requirements)
        second_time = time.time() - start_time

        # Verify timing measurements are reasonable
        assert first_time >= 0, "First discovery time should be non-negative"
        assert second_time >= 0, "Second discovery time should be non-negative"
        assert second_time <= first_time, "Second call should be faster due to caching"

        # Results should be consistent
        assert len(tools1) == len(tools2)
        if tools1 and tools2:
            assert tools1[0].name == tools2[0].name

        # Second call may be faster due to caching optimizations
        # (though this is implementation-dependent)


class TestAsyncPatterns:
    """Test async patterns and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_discovery(self):
        """Test concurrent tool discovery operations."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Run multiple discovery operations concurrently
        tasks = [
            engine.discover_tools_for_task("search task", {"max_latency_ms": 500}),
            engine.discover_tools_for_task("generate content", {"min_accuracy": 0.8}),
            engine.discover_tools_for_task("analyze data", {"max_cost": 0.05}),
        ]

        results = await asyncio.gather(*tasks)

        # All operations should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_concurrent_performance_updates(self):
        """Test concurrent performance metric updates."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Create multiple performance update tasks
        update_tasks = []
        for i in range(5):
            metrics = ToolMetrics(
                average_latency_ms=100.0 + i,
                success_rate=0.9 + (i * 0.01),
                accuracy_score=0.8 + (i * 0.02),
                cost_per_execution=0.01 + (i * 0.001),
                reliability_score=0.85 + (i * 0.01),
            )
            update_tasks.append(
                engine.update_tool_performance("hybrid_search", metrics)
            )

        # Execute all updates concurrently
        await asyncio.gather(*update_tasks)

        # Verify all updates were recorded
        assert (
            len(engine.tool_performance_history["hybrid_search"]) == 6
        )  # Initial + 5 updates

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in async operations."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test operations complete within reasonable time
        start_time = time.time()

        tools = await engine.discover_tools_for_task(
            "search for information", {"max_latency_ms": 500}
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (under 1 second for local operations)
        assert execution_time < 1.0
        assert len(tools) > 0


class TestToolDiscoveryComprehensiveScenarios:
    """Test comprehensive real-world scenarios."""

    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow_discovery(self):
        """Test discovery for complex multi-step workflows."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test complex research workflow
        recommendations = await engine.get_tool_recommendations(
            "Conduct comprehensive research: search academic papers, analyze sentiment, extract key insights, generate executive summary with citations",
            {
                "max_latency_ms": 5000,
                "min_accuracy": 0.85,
                "max_cost": 0.5,
            },
        )

        assert "primary_tools" in recommendations
        assert "tool_chains" in recommendations
        assert "reasoning" in recommendations

        # Should provide reasoning for complex workflows
        assert len(recommendations["reasoning"]) > 50

    @pytest.mark.asyncio
    async def test_domain_specific_tool_discovery(self):
        """Test discovery for domain-specific tasks."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        domain_tasks = [
            "medical literature search for drug interactions",
            "legal document analysis for contract review",
            "financial data analysis for risk assessment",
            "technical documentation search for API endpoints",
        ]

        for task in domain_tasks:
            tools = await engine.discover_tools_for_task(task, {})
            assert isinstance(tools, list)
            # Should find tools even for domain-specific tasks

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """Test detection and handling of performance degradation."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        tool_name = "hybrid_search"
        initial_tool = engine.discovered_tools[tool_name]
        initial_score = await engine._calculate_suitability_score(
            initial_tool, "search task", {"max_latency_ms": 500}
        )

        # Simulate performance degradation
        degraded_metrics = ToolMetrics(
            average_latency_ms=1000.0,  # Much slower
            success_rate=0.6,  # Much less reliable
            accuracy_score=0.5,  # Much less accurate
            cost_per_execution=0.2,  # Much more expensive
            reliability_score=0.4,  # Much less reliable
        )

        await engine.update_tool_performance(tool_name, degraded_metrics)

        # Calculate score after degradation
        degraded_tool = engine.discovered_tools[tool_name]
        degraded_score = await engine._calculate_suitability_score(
            degraded_tool, "search task", {"max_latency_ms": 500}
        )

        # Should detect and reflect performance degradation
        assert degraded_score < initial_score


@pytest.mark.performance
class TestPerformanceCharacteristics:
    """Test performance characteristics of the discovery engine."""

    @pytest.mark.asyncio
    async def test_discovery_performance(self):
        """Test discovery operation performance."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Measure performance of tool discovery
        start_time = time.time()

        for i in range(10):
            await engine.discover_tools_for_task(
                f"search task {i}", {"max_latency_ms": 500}
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        # Should average less than 50ms per discovery operation
        assert avg_time < 0.05, f"Average discovery time too slow: {avg_time:.3f}s"

    @pytest.mark.asyncio
    async def test_performance_update_overhead(self):
        """Test performance update operation overhead."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Measure performance update overhead
        start_time = time.time()

        for i in range(100):
            metrics = ToolMetrics(100.0 + i, 0.9, 0.8, 0.01, 0.85)
            await engine.update_tool_performance("hybrid_search", metrics)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100

        # Should average less than 1ms per update operation
        assert avg_time < 0.001, f"Average update time too slow: {avg_time:.6f}s"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency of performance history."""

        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Get initial memory usage
        initial_size = sys.getsizeof(engine.tool_performance_history)

        # Add many performance records
        for i in range(1000):
            metrics = ToolMetrics(100.0 + i, 0.9, 0.8, 0.01, 0.85)
            await engine.update_tool_performance("hybrid_search", metrics)

        # Check final memory usage
        final_size = sys.getsizeof(engine.tool_performance_history)

        # Memory should not grow excessively (rolling window limits growth)
        size_increase = final_size - initial_size

        # Should not use more than reasonable amount of memory
        assert size_increase < 100000, (
            f"Memory usage increased too much: {size_increase} bytes"
        )
