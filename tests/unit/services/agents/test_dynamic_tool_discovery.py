"""Deterministic tests for the dynamic tool discovery engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.agents import (
    AgentState,
    BaseAgentDependencies,
    DynamicToolDiscovery,
)
from src.services.agents.dynamic_tool_discovery import ToolCapabilityType, ToolMetrics


@pytest.fixture
def agent_dependencies() -> BaseAgentDependencies:
    """Provide lightweight dependencies for discovery tests."""
    manager = MagicMock()
    manager.get_registered_tools = MagicMock(return_value={})
    manager.get_dynamic_tool_discovery = AsyncMock(return_value=None)
    return BaseAgentDependencies(
        client_manager=manager,
        config=MagicMock(),
        session_state=AgentState(session_id="discovery-session", user_id=None),
    )


@pytest.mark.asyncio
async def test_initialize_discovery_populates_core_tools(
    agent_dependencies: BaseAgentDependencies,
) -> None:
    """The discovery engine should register core tools with compatibility sets."""
    engine = DynamicToolDiscovery()

    await engine.initialize_discovery(agent_dependencies)

    assert engine.discovered_tools
    assert {"hybrid_search", "rag_generation", "content_analysis"} <= set(
        engine.discovered_tools.keys()
    )
    for capability in engine.discovered_tools.values():
        assert capability.name
        assert isinstance(capability.capability_type, ToolCapabilityType)
        assert (
            capability.compatible_tools != []
            or capability.capability_type is ToolCapabilityType.GENERATION
        )


@pytest.mark.asyncio
async def test_discover_tools_for_task_ranks_by_requirements(
    agent_dependencies: BaseAgentDependencies,
) -> None:
    """Discovery should surface tools that satisfy latency and accuracy constraints."""
    engine = DynamicToolDiscovery()
    await engine.initialize_discovery(agent_dependencies)

    results = await engine.discover_tools_for_task(
        "Generate a detailed answer",
        {"max_latency_ms": 900, "min_accuracy": 0.85},
    )

    assert results
    assert all(tool.confidence_score <= 1.0 for tool in results)
    assert results[0].capability_type in {
        ToolCapabilityType.GENERATION,
        ToolCapabilityType.SEARCH,
    }
    assert results == sorted(
        results, key=lambda tool: tool.confidence_score, reverse=True
    )


@pytest.mark.asyncio
async def test_update_tool_performance_refreshes_metrics(
    agent_dependencies: BaseAgentDependencies,
) -> None:
    """Recorded executions should update the rolling metrics and timestamp."""
    engine = DynamicToolDiscovery()
    await engine.initialize_discovery(agent_dependencies)

    await engine.update_tool_performance(
        "hybrid_search",
        ToolMetrics(
            average_latency_ms=120.0,
            success_rate=0.98,
            accuracy_score=0.9,
            cost_per_execution=0.015,
            reliability_score=0.95,
        ),
    )

    capability = engine.discovered_tools["hybrid_search"]
    assert capability.metrics is not None
    assert capability.metrics.average_latency_ms <= 150.0
    assert capability.last_updated
