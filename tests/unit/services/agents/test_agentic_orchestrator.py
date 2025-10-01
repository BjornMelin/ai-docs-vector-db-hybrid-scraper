"""Tests covering the agentic orchestrator fallback pipeline."""

from __future__ import annotations

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import AgenticOrchestrator
from src.services.agents.core import BaseAgentDependencies, create_agent_dependencies


@pytest.mark.asyncio
async def test_fallback_orchestrate() -> None:
    """Ensure fallback execution returns structured orchestration output."""
    orchestrator = AgenticOrchestrator()
    deps: BaseAgentDependencies = create_agent_dependencies(ClientManager())
    response = await orchestrator.orchestrate(
        "search and answer the question",
        {"max_latency_ms": 800},
        deps,
    )
    assert response.success is True
    assert response.tools_used
    assert "reasoning" in response.model_dump()
