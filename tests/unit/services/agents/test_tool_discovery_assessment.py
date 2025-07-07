"""Assessment tests for Dynamic Tool Discovery Engine.

Tests for capability assessment, performance-driven selection and performance tracking.
"""

import asyncio
import time
from unittest.mock import Mock
from uuid import uuid4

import pytest

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
)


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
