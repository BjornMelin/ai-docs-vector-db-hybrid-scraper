"""Core tests for Dynamic Tool Discovery Engine.

Tests for tool metrics, capabilities, initialization and scanning functionality.
"""

import asyncio
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
