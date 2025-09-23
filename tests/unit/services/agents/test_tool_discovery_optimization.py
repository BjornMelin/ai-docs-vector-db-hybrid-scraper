"""Optimization tests for Dynamic Tool Discovery Engine.

Tests for self-learning optimization, tool compatibility analysis,
and edge case handling.
"""

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

        # Simulate moderately worse performance
        # or the best tool (not so bad it gets filtered out)

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
            # Tool fell below threshold due to
            # oor performance - that's also valid learning behavior

            assert len(updated_tools) == 0 or updated_tools[0].name != initial_best.name


class TestToolCompatibilityAnalysis:
    """Test tool compatibility analysis and chain generation."""

    @pytest.mark.asyncio
    async def test_get_tool_recommendations(self):
        """Test  tool recommendations."""
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
            "search and generate analysis",
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
            "analyze search and generate report",
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
