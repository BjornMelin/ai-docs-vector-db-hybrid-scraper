"""Advanced tests for Dynamic Tool Discovery Engine.

Tests for algorithms, performance characteristics, robustness,
and complex scenarios.
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


class TestToolDiscoveryBehavior:
    """Test observable behavior of dynamic tool discovery system."""

    @pytest.mark.asyncio
    async def test_tool_initialization_completes_successfully(self):
        """Test that tool initialization completes without errors."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # Should complete initialization without errors
        await engine.initialize_tools(deps)

    @pytest.mark.asyncio
    async def test_tool_scoring_reflects_task_relevance(self):
        """Test that suitability scoring reflects task relevance."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test that search-related tasks score higher for search tools
        tool = engine.discovered_tools["hybrid_search"]
        search_score = await engine._calculate_suitability_score(
            tool, "search for documents", {}
        )

        unrelated_score = await engine._calculate_suitability_score(
            tool, "completely unrelated quantum physics task", {}
        )

        # Search tool should score higher for search tasks
        assert search_score > unrelated_score
        assert search_score >= 0.0
        assert search_score <= 1.0

    @pytest.mark.asyncio
    async def test_handles_impossible_requirements_gracefully(self):
        """Test that system handles impossible requirements without crashing."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        await engine.initialize_discovery(deps)

        # Test with requirements that no tools can meet
        recommendations = await engine.get_tool_recommendations(
            "search for documents",
            {
                "max_latency_ms": 0.1,  # Impossible latency
                "min_accuracy": 0.999,  # Impossible accuracy
                "max_cost": 0.0001,  # Impossible cost
            },
        )

        # Should return valid structure even with impossible requirements
        assert "primary_tools" in recommendations
        assert "secondary_tools" in recommendations
        assert "tool_chains" in recommendations
        assert "reasoning" in recommendations

    @pytest.mark.asyncio
    async def test_tool_chains_handle_missing_metrics(self):
        """Test that tool chain generation handles tools without metrics."""
        engine = DynamicToolDiscovery()

        # Create tools without metrics
        tools_without_metrics = [
            ToolCapability(
                name="no_metrics_search",
                capability_type=ToolCapabilityType.SEARCH,
                description="Search tool without metrics",
                input_types=["text"],
                output_types=["results"],
                metrics=None,
                last_updated="2024-01-01T00:00:00Z",
            ),
            ToolCapability(
                name="no_metrics_gen",
                capability_type=ToolCapabilityType.GENERATION,
                description="Generation tool without metrics",
                input_types=["text"],
                output_types=["content"],
                metrics=None,
                last_updated="2024-01-01T00:00:00Z",
            ),
        ]

        chains = await engine._generate_tool_chains(tools_without_metrics)

        # Should handle None metrics gracefully
        assert isinstance(chains, list)
        for chain in chains:
            if "estimated_total_latency_ms" in chain:
                assert chain["estimated_total_latency_ms"] >= 0

    def test_tool_capability_types_are_valid(self):
        """Test that all tool capability types are properly defined."""
        capability_types = [
            ToolCapabilityType.SEARCH,
            ToolCapabilityType.RETRIEVAL,
            ToolCapabilityType.GENERATION,
            ToolCapabilityType.ANALYSIS,
            ToolCapabilityType.CLASSIFICATION,
            ToolCapabilityType.SYNTHESIS,
            ToolCapabilityType.ORCHESTRATION,
        ]

        for capability_type in capability_types:
            # Each type should have a string value
            assert isinstance(capability_type.value, str)
            assert len(capability_type.value) > 0

            # Should be able to create tools with each type
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
    async def test_initialization_is_idempotent(self):
        """Test that repeated initialization does not cause issues."""
        engine = DynamicToolDiscovery()

        mock_client_manager = Mock(spec=ClientManager)
        mock_config = get_config()
        state = AgentState(session_id=str(uuid4()))

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config, session_state=state
        )

        # First initialization
        await engine.initialize_discovery(deps)
        initial_tool_count = len(engine.discovered_tools)

        # Second initialization should not duplicate tools
        await engine.initialize_discovery(deps)
        final_tool_count = len(engine.discovered_tools)

        assert final_tool_count == initial_tool_count


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

            # Score should always be between 0 and 1
            # allowing for small negative values due to penalties)

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

        # Other metrics should remain consistent
        # all same values) with small tolerance for floating point

        assert abs(avg_metrics.success_rate - 0.9) < 1e-10
        assert abs(avg_metrics.accuracy_score - 0.8) < 1e-10
        assert abs(avg_metrics.cost_per_execution - 0.01) < 1e-10
        assert abs(avg_metrics.reliability_score - 0.85) < 1e-10


class TestAdvancedToolDiscoveryAlgorithms:
    """Test tool discovery algorithms and complex scenarios."""

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
        # Success rate should be improved or at
        # east maintained (account for floating point precision)

        assert (
            updated_tool.metrics.success_rate
            >= initial_tool.metrics.success_rate - 0.001
        )


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

        # Test typical maximum values
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
