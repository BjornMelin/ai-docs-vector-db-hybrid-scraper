"""Tests for the LangGraph agentic runner."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from mcp.types import CallToolResult


ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str, relative: str) -> ModuleType:
    module_path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)  # type: ignore[arg-type]
    return loaded


dynamic_module = _load_module(
    "src.services.agents.dynamic_tool_discovery",
    "src/services/agents/dynamic_tool_discovery.py",
)
retrieval_module = _load_module(
    "src.services.agents.retrieval",
    "src/services/agents/retrieval.py",
)
tool_execution_module = _load_module(
    "src.services.agents.tool_execution_service",
    "src/services/agents/tool_execution_service.py",
)
module = _load_module(
    "src.services.agents.langgraph_runner",
    "src/services/agents/langgraph_runner.py",
)

GraphRunner = cast(Any, module.GraphRunner)
GraphSearchOutcome = cast(Any, module.GraphSearchOutcome)
ToolCapability = cast(Any, module.ToolCapability)
ToolCapabilityType = cast(Any, module.ToolCapabilityType)
ToolExecutionResult = cast(Any, module.ToolExecutionResult)
RetrievedDocument = cast(Any, retrieval_module.RetrievedDocument)
ToolExecutionFailure = cast(Any, tool_execution_module.ToolExecutionFailure)


class StubDiscovery:
    def __init__(self, capabilities: list[Any]) -> None:
        self._capabilities = capabilities
        self.refresh_count = 0

    async def refresh(self, *, force: bool = False) -> None:  # noqa: D401
        self.refresh_count += 1

    def get_capabilities(self) -> tuple[Any, ...]:  # noqa: D401
        return tuple(self._capabilities)


class StubToolService:
    async def execute_tool(
        self, tool_name, *, arguments, server_name, read_timeout_ms=None
    ):  # noqa: D401, ANN001
        return ToolExecutionResult(
            tool_name=tool_name,
            server_name=server_name,
            duration_ms=5.0,
            result=CallToolResult(
                content=[],
                structuredContent={"result": f"executed:{tool_name}"},
                isError=False,
            ),
        )


class StubRetrievalHelper:
    async def fetch(self, query):  # noqa: D401, ANN001
        return (
            RetrievedDocument(
                id="doc-1",
                score=0.9,
                payload={"title": "Doc"},
                raw=None,
            ),
        )


class FailingToolService:
    async def execute_tool(self, *_, **__):  # noqa: D401, ANN001
        raise ToolExecutionFailure("boom")


@pytest.mark.asyncio
async def test_run_search_produces_outcome() -> None:
    capabilities = [
        ToolCapability(
            name="semantic_search",
            server="primary",
            description="Semantic search",
            capability_type=ToolCapabilityType.SEARCH,
            input_schema=("query",),
            output_schema=(),
            metadata={},
            last_refreshed="2024-01-01T00:00:00+00:00",
        )
    ]
    runner = GraphRunner(
        client_manager=None,
        discovery=StubDiscovery(capabilities),
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
        max_parallel_tools=2,
    )

    outcome = await runner.run_search(query="find langgraph docs", collection="docs", top_k=2)

    assert outcome.success
    assert outcome.answer is not None
    assert outcome.results
    assert outcome.tools_used == ["semantic_search"]
    assert outcome.metrics["tool_count"] == 1
    assert outcome.errors == []


@pytest.mark.asyncio
async def test_run_analysis_returns_summary() -> None:
    capabilities = [
        ToolCapability(
            name="report_analysis",
            server="primary",
            description="Analyse reports",
            capability_type=ToolCapabilityType.ANALYSIS,
            input_schema=("query",),
            output_schema=("insights",),
            metadata={},
            last_refreshed="2024-01-01T00:00:00+00:00",
        )
    ]
    runner = GraphRunner(
        client_manager=None,
        discovery=StubDiscovery(capabilities),
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
    )

    outcome = await runner.run_analysis(
        query="summarise metrics",
        context_documents=[{"id": "1", "score": 0.7, "payload": {"title": "Metrics"}}],
    )

    assert outcome.success
    assert outcome.summary
    assert outcome.insights["documents_considered"]


@pytest.mark.asyncio
async def test_run_search_surfaces_structured_tool_error() -> None:
    capabilities = [
        ToolCapability(
            name="semantic_search",
            server="primary",
            description="Semantic search",
            capability_type=ToolCapabilityType.SEARCH,
            input_schema=("query",),
            output_schema=(),
            metadata={},
            last_refreshed="2024-01-01T00:00:00+00:00",
        )
    ]
    runner = GraphRunner(
        client_manager=None,
        discovery=StubDiscovery(capabilities),
        tool_service=FailingToolService(),
        retrieval_helper=StubRetrievalHelper(),
        max_parallel_tools=1,
    )

    outcome = await runner.run_search(query="find", collection="docs")

    assert outcome.success is False
    assert outcome.errors
    error = outcome.errors[0]
    assert error["source"] == "tool_execution"
    assert error["code"] == "ToolExecutionFailure"


@pytest.mark.asyncio
async def test_run_search_times_out() -> None:
    capabilities = []

    # Use existing runner but with long-running tool discovery to trigger timeout
    runner = GraphRunner(
        client_manager=None,
        discovery=StubDiscovery(capabilities),
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
        run_timeout_seconds=0.01,
    )

    async def sleepy_refresh(*_, **__):
        await asyncio.sleep(0.05)

    runner._discovery.refresh = sleepy_refresh  # type: ignore[attr-defined]

    outcome = await runner.run_search(query="slow", collection="docs")

    assert outcome.success is False
    assert any(err.get("code") == "RUN_TIMEOUT" for err in outcome.errors)
