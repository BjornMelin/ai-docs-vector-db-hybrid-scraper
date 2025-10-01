"""Unit tests for dynamic tool discovery heuristics."""

from __future__ import annotations

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.core import BaseAgentDependencies, create_agent_dependencies
from src.services.agents.dynamic_tool_discovery import DynamicToolDiscovery


@pytest.mark.asyncio
async def test_discovery_and_ranking():
    """Ensure discovery pipeline yields scored tools for a given task."""
    engine = DynamicToolDiscovery()
    deps: BaseAgentDependencies = create_agent_dependencies(ClientManager())
    await engine.initialize_discovery(deps)
    tools = await engine.discover_tools_for_task(
        "search and generate an answer", {"max_latency_ms": 1000}
    )
    assert len(tools) >= 1
    assert all(hasattr(t, "confidence_score") for t in tools)
