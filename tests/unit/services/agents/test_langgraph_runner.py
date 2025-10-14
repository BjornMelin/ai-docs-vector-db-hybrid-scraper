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
AgentErrorCode = cast(Any, module.AgentErrorCode)


class StubDiscovery:
    """Stub implementation of DynamicToolDiscovery for testing."""

    def __init__(self, capabilities: list[Any]) -> None:
        """Initialize the stub discovery with given capabilities."""
        self._capabilities = capabilities
        self.refresh_count = 0

    async def refresh(self, *, force: bool = False) -> None:
        """Simulate refreshing the discovery capabilities."""
        self.refresh_count += 1

    def get_capabilities(self) -> tuple[Any, ...]:
        """Get the current capabilities."""
        return tuple(self._capabilities)


class StubToolService:
    """Stub implementation of ToolService for testing."""

    async def execute_tool(
        self, tool_name, *, arguments, server_name, read_timeout_ms=None
    ):
        """Simulate executing a tool."""
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
    """Stub implementation of RetrievalHelper for testing."""

    async def fetch(self, query):
        """Simulate fetching documents based on a query."""
        return (
            RetrievedDocument(
                id="doc-1",
                score=0.9,
                metadata={"title": "Doc"},
                raw=None,
            ),
        )


class FailingToolService:
    """Stub implementation of a tool service that always fails."""

    async def execute_tool(self, *_, **__):
        """Simulate a tool execution failure."""
        raise ToolExecutionFailure("boom")


class SlowToolService:
    """Stub implementation of a tool service that simulates a slow execution."""

    def __init__(self) -> None:
        """Initialize the slow tool service."""
        self.cancelled = asyncio.Event()

    async def execute_tool(
        self,
        tool_name,
        *,
        arguments,
        server_name,
        read_timeout_ms=None,
    ):
        """Simulate a slow tool execution that can be cancelled."""
        try:
            await asyncio.sleep(0.05)
            return ToolExecutionResult(
                tool_name=tool_name,
                server_name=server_name,
                duration_ms=50.0,
                result=CallToolResult(
                    content=[], structuredContent=None, isError=False
                ),
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_run_search_produces_outcome() -> None:
    """Test that running a search produces a successful outcome."""
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
        discovery=StubDiscovery(capabilities),
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
        max_parallel_tools=2,
    )

    outcome = await runner.run_search(
        query="find langgraph docs", collection="docs", top_k=2
    )

    assert outcome.success
    assert outcome.answer is not None
    assert outcome.results
    assert outcome.tools_used == ["semantic_search"]
    assert outcome.metrics["tool_count"] == 1
    assert outcome.errors == []


@pytest.mark.asyncio
async def test_run_analysis_returns_summary() -> None:
    """Test that running an analysis produces a summary outcome."""
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
        discovery=StubDiscovery(capabilities),
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
    )

    outcome = await runner.run_analysis(
        query="summarise metrics",
        context_documents=[{"id": "1", "score": 0.7, "metadata": {"title": "Metrics"}}],
    )

    assert outcome.success
    assert outcome.summary
    assert outcome.insights["documents_considered"]


@pytest.mark.asyncio
async def test_run_search_surfaces_structured_tool_error() -> None:
    """Test that a structured tool error is surfaced in the outcome."""
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
    assert error["code"] == AgentErrorCode.TOOL_FAILURE.value


@pytest.mark.asyncio
async def test_run_search_times_out() -> None:
    """Test that a tool execution timeout is handled correctly."""
    capabilities = [
        ToolCapability(
            name="slow_tool",
            server="primary",
            description="Slow tool",
            capability_type=ToolCapabilityType.SEARCH,
            input_schema=(),
            output_schema=(),
            metadata={},
            last_refreshed="2024-01-01T00:00:00+00:00",
        )
    ]
    slow_service = SlowToolService()
    runner = GraphRunner(
        discovery=StubDiscovery(capabilities),
        tool_service=slow_service,
        retrieval_helper=StubRetrievalHelper(),
        run_timeout_seconds=0.01,
    )

    outcome = await runner.run_search(query="slow", collection="docs")

    assert outcome.success is False
    assert any(
        err.get("code") == AgentErrorCode.RUN_TIMEOUT.value for err in outcome.errors
    )
    await asyncio.wait_for(slow_service.cancelled.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_discovery_failure_returns_structured_error() -> None:
    """Test that discovery failure returns a structured error."""

    class BrokenDiscovery:
        def __init__(self) -> None:
            self.calls = 0

        async def refresh(self, *, force: bool = False) -> None:
            self.calls += 1
            raise RuntimeError("discovery unavailable")

        def get_capabilities(self) -> tuple[Any, ...]:
            return ()

    discovery = BrokenDiscovery()
    runner = GraphRunner(
        discovery=discovery,
        tool_service=StubToolService(),
        retrieval_helper=StubRetrievalHelper(),
    )

    outcome = await runner.run_search(query="q", collection="docs")

    assert discovery.calls == 1
    assert outcome.success is False
    assert any(
        err.get("code") == AgentErrorCode.DISCOVERY_ERROR.value
        for err in outcome.errors
    )
    assert not outcome.tools_used
