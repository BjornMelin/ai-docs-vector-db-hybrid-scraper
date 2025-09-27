"""Deterministic coverage for the agentic orchestrator fallback behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.agents import AgenticOrchestrator, AgentState, BaseAgentDependencies


@pytest.fixture
def orchestrator_dependencies() -> BaseAgentDependencies:
    """Construct dependencies with async client hooks for orchestrator tests."""
    manager = MagicMock()
    manager.get_dynamic_tool_discovery = AsyncMock()
    return BaseAgentDependencies(
        client_manager=manager,
        config=MagicMock(),
        session_state=AgentState(session_id="orch-session", user_id=None),
    )


@pytest.mark.asyncio
async def test_orchestrate_uses_fallback_pipeline(
    orchestrator_dependencies: BaseAgentDependencies,
) -> None:
    """Fallback orchestration should produce a structured tool response."""
    orchestrator = AgenticOrchestrator()

    response = await orchestrator.orchestrate(
        task="Search and generate an executive summary",
        constraints={"max_latency_ms": 1500},
        deps=orchestrator_dependencies,
    )

    assert response.success is True
    assert response.tools_used
    assert response.latency_ms >= 0
    assert response.results["fallback_mode"] is True
    assert "fallback" in response.reasoning.lower()


def test_select_tools_for_task_applies_keywords() -> None:
    """Keyword heuristics should map tasks to concrete tool selections."""
    orchestrator = AgenticOrchestrator()

    tools = orchestrator._select_tools_for_task(
        "Analyze and generate insights",
        {},
        {},
    )

    assert tools[:2] == ["hybrid_search", "rag_generation"]
    assert "content_analysis" in tools


def test_select_tools_respects_latency_constraints() -> None:
    """Latency caps should remove slower generators from the final plan."""
    orchestrator = AgenticOrchestrator()

    tools = orchestrator._select_tools_for_task(
        "Generate answer from documents",
        {},
        {"max_latency_ms": 500},
    )

    assert "rag_generation" not in tools
    assert tools == ["hybrid_search"]


def test_calculate_confidence_handles_partial_errors() -> None:
    """Confidence metric should scale with the proportion of successful steps."""
    orchestrator = AgenticOrchestrator()

    confidence = orchestrator._calculate_confidence(
        {"hybrid_search_result": {}, "rag_generation_error": "timeout"},
        ["hybrid_search", "rag_generation"],
    )

    assert 0.0 < confidence <= 1.0
