"""Tests for the LangGraph-backed agentic RAG MCP tools."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from fastmcp import FastMCP

from contracts.retrieval import SearchRecord
from src.mcp_tools.tools import agentic_rag
from src.services.agents import GraphAnalysisOutcome, GraphRunner, GraphSearchOutcome


class StubGraphRunner:
    """Stub GraphRunner returning predetermined outcomes."""

    def __init__(
        self,
        search_outcome: GraphSearchOutcome,
        analysis_outcome: GraphAnalysisOutcome,
    ) -> None:
        self._search_outcome = search_outcome
        self._analysis_outcome = analysis_outcome
        self.run_search_calls: list[dict[str, Any]] = []
        self.run_analysis_calls: list[dict[str, Any]] = []

    async def run_search(self, **kwargs: Any) -> GraphSearchOutcome:
        self.run_search_calls.append(kwargs)
        return self._search_outcome

    async def run_analysis(self, **kwargs: Any) -> GraphAnalysisOutcome:
        self.run_analysis_calls.append(kwargs)
        return self._analysis_outcome


class StubMCP:
    """Stub FastMCP server recording registered tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}

    def tool(self, *_, name: str, **__):  # pragma: no cover - decorator wiring
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[name] = func
            return func

        return decorator


@pytest.fixture(autouse=True)
def reset_tracker() -> Iterator[None]:
    """Reset agent telemetry before and after each test."""

    agentic_rag.get_ai_tracker().reset()
    yield
    agentic_rag.get_ai_tracker().reset()


@pytest.fixture()
def stub_runner() -> StubGraphRunner:
    """Provide a stub GraphRunner with deterministic outcomes."""

    search_outcome = GraphSearchOutcome(
        success=True,
        session_id="session-1",
        answer="Result",
        confidence=0.75,
        results=[SearchRecord(id="1", content="", score=0.9)],
        tools_used=["semantic_search"],
        reasoning=["step"],
        metrics={"total_latency_ms": 10.0},
        errors=[],
    )
    analysis_outcome = GraphAnalysisOutcome(
        success=True,
        analysis_id="analysis-1",
        summary="Summary",
        insights={"key": "value"},
        recommendations=["follow-up"],
        confidence=0.9,
        metrics={"latency_ms": 12.0},
        errors=[],
    )
    return StubGraphRunner(search_outcome, analysis_outcome)


@pytest.fixture()
def registered_tools(stub_runner: StubGraphRunner) -> dict[str, Callable[..., Any]]:
    """Register agentic tools against the stub MCP server."""

    mcp = StubMCP()
    agentic_rag.register_tools(
        cast(FastMCP[Any], mcp),
        graph_runner=cast(GraphRunner, stub_runner),
    )
    return mcp.tools


@pytest.mark.asyncio
async def test_agentic_search_returns_runner_payload(
    registered_tools: dict[str, Callable[..., Any]],
    stub_runner: StubGraphRunner,
) -> None:
    """agentic search should surface GraphRunner results and telemetry."""

    request = agentic_rag.AgenticSearchRequest(
        query="q",
        collection="docs",
        max_results=None,
        session_id=None,
        user_context=None,
        filters=None,
    )
    handler = registered_tools["agentic_search"]

    response = await handler(request)

    assert response.success is True
    assert response.answer == "Result"
    assert response.tools_used == ["semantic_search"]
    assert response.metrics["total_latency_ms"] == 10.0
    assert stub_runner.run_search_calls[0]["query"] == "q"


@pytest.mark.asyncio
async def test_agentic_analysis_returns_runner_payload(
    registered_tools: dict[str, Callable[..., Any]],
    stub_runner: StubGraphRunner,
) -> None:
    """agentic analysis should surface GraphRunner analysis outcomes."""

    request = agentic_rag.AgenticAnalysisRequest(
        query="analyse",
        data=[{}],
        session_id=None,
        user_context=None,
    )
    handler = registered_tools["agentic_analysis"]

    response = await handler(request)

    assert response.success is True
    assert response.summary == "Summary"
    assert response.insights["key"] == "value"
    assert stub_runner.run_analysis_calls[0]["query"] == "analyse"


@pytest.mark.asyncio
async def test_metrics_tools_expose_tracker_snapshot(
    registered_tools: dict[str, Callable[..., Any]],
) -> None:
    """Telemetry tools should reflect tracker state."""

    tracker = agentic_rag.get_ai_tracker()
    tracker.record_operation(
        operation="agent.graph.run",
        provider="search",
        model="run",
        duration_s=0.5,
        success=True,
        tokens=120,
        cost_usd=0.001,
    )

    metrics_handler = registered_tools["agent_performance_metrics"]
    orchestration_handler = registered_tools["agentic_orchestration_metrics"]
    reset_handler = registered_tools["reset_agent_learning"]

    metrics_response = await metrics_handler()
    assert metrics_response.success is True
    assert metrics_response.operations  # ensure snapshot populated

    orchestration_response = await orchestration_handler()
    assert orchestration_response.run_summary.total_runs >= 0

    with pytest.raises(agentic_rag.ToolError):
        await reset_handler()

    reset_result = await reset_handler(confirm=True)
    assert reset_result["success"] is True
