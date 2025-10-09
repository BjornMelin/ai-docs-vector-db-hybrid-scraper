"""Tests for the LangGraph-backed agentic RAG MCP tools."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, relative: str) -> ModuleType:
    module_path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)  # type: ignore[arg-type]
    return loaded


_load_module("src.services.errors", "src/services/errors.py")
_load_module("src.services.agents", "src/services/agents/__init__.py")
graph_module = _load_module(
    "src.services.agents.langgraph_runner",
    "src/services/agents/langgraph_runner.py",
)

MODULE_PATH = ROOT / "src/mcp_tools/tools/agentic_rag.py"
spec = importlib.util.spec_from_file_location("agentic_rag_under_test", MODULE_PATH)
assert spec and spec.loader
agentic_rag: Any = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agentic_rag
spec.loader.exec_module(agentic_rag)  # type: ignore[arg-type]

GraphAnalysisOutcome = cast(Any, graph_module.GraphAnalysisOutcome)
GraphSearchOutcome = cast(Any, graph_module.GraphSearchOutcome)


class DummyClientManager:
    """Minimal client manager stub for GraphRunner initialisation."""

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(
            mcp_client=types.SimpleNamespace(request_timeout_ms=1000, servers=[])
        )

    async def get_mcp_client(self) -> Any:  # pragma: no cover - not used in tests
        raise RuntimeError("get_mcp_client should not be called in tests")


class StubGraphRunner:
    """Stub GraphRunner returning predetermined outcomes."""

    def __init__(self, search_outcome: Any, analysis_outcome: Any) -> None:
        self._search_outcome = search_outcome
        self._analysis_outcome = analysis_outcome
        self.run_search_calls: list[dict[str, Any]] = []
        self.run_analysis_calls: list[dict[str, Any]] = []

    async def run_search(self, **kwargs: Any) -> Any:  # noqa: D401
        self.run_search_calls.append(kwargs)
        return self._search_outcome

    async def run_analysis(self, **kwargs: Any) -> Any:  # noqa: D401
        self.run_analysis_calls.append(kwargs)
        return self._analysis_outcome


class StubMCP:
    """Stub FastMCP server recording registered tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str, tags: set[str]):  # type: ignore[override]
        def decorator(func: Any) -> Any:
            self.tools[name] = func
            return func

        return decorator


@pytest.fixture(autouse=True)
def reset_agentic_state():
    agentic_rag.get_ai_tracker().reset()
    agentic_rag._runner_cache.clear()
    agentic_rag._runner_locks.clear()
    yield
    agentic_rag.get_ai_tracker().reset()
    agentic_rag._runner_cache.clear()
    agentic_rag._runner_locks.clear()


@pytest.mark.asyncio
async def test_run_agentic_search() -> None:
    """agentic search should surface GraphRunner results and telemetry."""

    outcome = GraphSearchOutcome(
        success=True,
        session_id="session-1",
        answer="Result",
        confidence=0.75,
        results=[{"id": "1"}],
        tools_used=["semantic_search"],
        reasoning=["step"],
        metrics={"latency_ms": 10.0, "tool_count": 1, "error_count": 0},
        errors=[],
    )
    stub_runner = StubGraphRunner(
        search_outcome=outcome,
        analysis_outcome=GraphAnalysisOutcome(
            success=True,
            analysis_id="analysis-1",
            summary="Summary",
            insights={},
            recommendations=[],
            confidence=0.8,
            metrics={"latency_ms": 5.0},
            errors=[],
        ),
    )
    client_manager = DummyClientManager()
    agentic_rag._runner_cache[client_manager] = stub_runner

    request = agentic_rag.AgenticSearchRequest(query="q", collection="docs")
    response = await agentic_rag._run_search(request, client_manager)

    assert response.success is True
    assert response.answer == "Result"
    assert response.tools_used == ["semantic_search"]
    assert response.total_latency_ms == 10.0
    assert stub_runner.run_search_calls[0]["query"] == "q"
    assert response.errors == []


@pytest.mark.asyncio
async def test_run_agentic_analysis() -> None:
    """agentic analysis should surface GraphRunner analysis outcomes."""

    analysis_outcome = GraphAnalysisOutcome(
        success=True,
        analysis_id="analysis-1",
        summary="Summary",
        insights={"key": "value"},
        recommendations=["follow-up"],
        confidence=0.9,
        metrics={"latency_ms": 12.0},
        errors=[{"code": "warning"}],
    )
    stub_runner = StubGraphRunner(
        search_outcome=GraphSearchOutcome(
            success=False,
            session_id="session-1",
            answer=None,
            confidence=None,
            results=[],
            tools_used=[],
            reasoning=[],
            metrics={},
            errors=[{"code": "stub"}],
        ),
        analysis_outcome=analysis_outcome,
    )
    client_manager = DummyClientManager()
    agentic_rag._runner_cache[client_manager] = stub_runner

    request = agentic_rag.AgenticAnalysisRequest(query="analyse", data=[{}])
    response = await agentic_rag._run_analysis(request, client_manager)

    assert response.success is True
    assert response.summary == "Summary"
    assert response.insights["key"] == "value"
    assert response.metrics["latency_ms"] == 12.0
    assert stub_runner.run_analysis_calls[0]["query"] == "analyse"
    assert response.errors[0]["code"] == "warning"


@pytest.mark.asyncio
async def test_metrics_tools_surface_tracker_snapshot() -> None:
    """Metrics tools should expose aggregated tracker statistics."""

    tracker = agentic_rag.get_ai_tracker()
    tracker.reset()
    tracker.record_operation(
        operation="agent.graph.run",
        provider="search",
        model="run",
        duration_s=0.5,
        success=True,
    )
    tracker.record_operation(
        operation="agent.graph.run",
        provider="search",
        model="run",
        duration_s=0.25,
        success=False,
    )
    tracker.record_operation(
        operation="agent.graph.retrieval",
        provider="docs",
        model="search",
        duration_s=0.2,
    )

    stub_mcp = StubMCP()
    agentic_rag.register_tools(stub_mcp, DummyClientManager())
    metrics_tool = stub_mcp.tools["agent_performance_metrics"]
    orchestration_tool = stub_mcp.tools["agentic_orchestration_metrics"]

    metrics_response = await metrics_tool()
    assert metrics_response.success is True
    assert metrics_response.operations["agent.graph.run"].count == 2

    orchestration_response = await orchestration_tool()
    assert orchestration_response.run_summary.total_runs == 2
    assert orchestration_response.run_summary.successful_runs == 1
    assert orchestration_response.operations["agent.graph.retrieval"].count == 1
