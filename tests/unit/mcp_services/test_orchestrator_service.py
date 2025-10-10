"""Tests for the LangGraph-backed orchestrator service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.mcp_services.orchestrator_service import (
    OrchestratorDependencies,
    OrchestratorService,
)
from src.services.agents import (
    DynamicToolDiscovery,
    GraphAnalysisOutcome,
    GraphRunner,
    GraphSearchOutcome,
)


if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from src.mcp_services.analytics_service import (
        AnalyticsService as AnalyticsServiceType,
    )
    from src.mcp_services.document_service import DocumentService as DocumentServiceType
    from src.mcp_services.search_service import SearchService as SearchServiceType
    from src.mcp_services.system_service import SystemService as SystemServiceType
else:  # pragma: no cover - runtime fallbacks for optional imports
    AnalyticsServiceType = Any
    DocumentServiceType = Any
    SearchServiceType = Any
    SystemServiceType = Any


@dataclass
class StubService:
    """Stub service exposing get_service_info."""

    name: str

    async def get_service_info(self) -> dict[str, Any]:
        return {"status": "ok", "name": self.name}


class StubDiscovery:
    """Stub dynamic tool discovery."""

    def __init__(self) -> None:
        self.refresh_calls: list[bool] = []

    async def refresh(self, *, force: bool = False) -> None:
        self.refresh_calls.append(force)

    def get_capabilities(self) -> tuple[Any, ...]:
        # Return a mock object with model_dump method
        class MockCapability:
            def __init__(self):
                self.name = "demo"
                self.server = "primary"
                self.description = "stub"
                self.capability_type = "search"
                self.input_schema = ("query",)
                self.output_schema = ()
                self.metadata = {}
                self.last_refreshed = "2025-10-10T00:00:00Z"

            def model_dump(self):
                return {
                    "name": self.name,
                    "server": self.server,
                    "description": self.description,
                    "capability_type": self.capability_type,
                    "input_schema": self.input_schema,
                    "output_schema": self.output_schema,
                    "metadata": self.metadata,
                    "last_refreshed": self.last_refreshed,
                }

        return (MockCapability(),)


ToolCallable = Callable[..., Awaitable[Any]]


def _get_tool(orchestrator: OrchestratorService, tool_name: str) -> ToolCallable:
    """Retrieve a registered FastMCP tool with precise typing."""
    mcp_server = orchestrator.get_mcp_server()
    mcp_server_any = cast(Any, mcp_server)
    raw_tools = getattr(mcp_server_any, "tools", None)
    if raw_tools is None:
        msg = "FastMCP server missing tool registry"
        raise AttributeError(msg)
    tools = cast(Mapping[str, ToolCallable], raw_tools)
    return tools[tool_name]


class StubGraphRunner:
    """Graph runner returning predetermined search results."""

    def __init__(self) -> None:
        # Skip parent __init__; only override required methods
        self.run_search_calls: list[dict[str, Any]] = []

    async def run_search(self, **kwargs: Any) -> GraphSearchOutcome:
        self.run_search_calls.append(kwargs)
        return GraphSearchOutcome(
            success=True,
            session_id="sid",
            answer="answer",
            confidence=0.9,
            results=[{"id": "doc-1"}],
            tools_used=["search"],
            reasoning=["step"],
            metrics={"latency_ms": 42.0},
            errors=[],
        )

    async def run_analysis(
        self, **kwargs: Any
    ) -> GraphAnalysisOutcome:  # pragma: no cover - unused
        return GraphAnalysisOutcome(
            success=True,
            analysis_id="aid",
            summary="summary",
            insights={},
            recommendations=[],
            confidence=0.8,
            metrics={},
            errors=[],
        )


@pytest.fixture()
def orchestrator_components() -> tuple[
    OrchestratorService, StubDiscovery, StubGraphRunner
]:
    """Instantiate orchestrator service with stubbed dependencies."""

    discovery = StubDiscovery()
    graph_runner = StubGraphRunner()
    deps = OrchestratorDependencies(
        search_service=cast(SearchServiceType, StubService("search")),
        document_service=cast(DocumentServiceType, StubService("document")),
        analytics_service=cast(AnalyticsServiceType, StubService("analytics")),
        system_service=cast(SystemServiceType, StubService("system")),
        graph_runner=cast(GraphRunner, graph_runner),
        discovery=cast(DynamicToolDiscovery, discovery),
    )
    orchestrator = OrchestratorService(dependencies=deps)

    # Replace the real FastMCP with a mock that has tools registry
    from unittest.mock import MagicMock

    mock_mcp = MagicMock()
    mock_mcp.name = "orchestrator-service"
    mock_mcp.instructions = "Mock instructions"
    mock_mcp.tools = {}

    # Mock the tool decorator to capture registered tools
    def mock_tool_decorator(*args, **kwargs):
        def decorator(func):
            mock_mcp.tools[func.__name__] = func
            return func

        return decorator

    mock_mcp.tool = mock_tool_decorator

    # Re-register tools with the mock
    orchestrator.mcp = mock_mcp
    orchestrator._register_orchestrator_tools()

    return orchestrator, discovery, graph_runner


@pytest.mark.asyncio
async def test_orchestrate_multi_service_workflow_invokes_graph_runner(
    orchestrator_components: tuple[OrchestratorService, StubDiscovery, StubGraphRunner],
) -> None:
    """Multi-service workflow should proxy through the GraphRunner."""

    orchestrator, _, graph_runner = orchestrator_components
    tool = _get_tool(orchestrator, "orchestrate_multi_service_workflow")
    response = await tool(
        "Describe workflow", services_required=["search"], performance_constraints=None
    )

    assert response["success"] is True
    assert response["workflow_results"]["answer"] == "answer"
    assert graph_runner.run_search_calls[0]["query"] == "Describe workflow"
    assert graph_runner.run_search_calls[0]["user_context"]["services_required"] == [
        "search"
    ]


@pytest.mark.asyncio
async def test_get_service_capabilities_returns_dependency_info(
    orchestrator_components: tuple[OrchestratorService, StubDiscovery, StubGraphRunner],
) -> None:
    """Capability tool should return status for each orchestrated service."""

    orchestrator, _, _ = orchestrator_components
    tool = _get_tool(orchestrator, "get_service_capabilities")

    response = await tool()

    assert response["services"]["search"]["status"] == "ok"
    assert response["services"]["document"]["name"] == "document"


@pytest.mark.asyncio
async def test_optimize_service_performance_refreshes_discovery(
    orchestrator_components: tuple[OrchestratorService, StubDiscovery, StubGraphRunner],
) -> None:
    """Optimisation tool should refresh discovery and return capabilities."""

    orchestrator, discovery, _ = orchestrator_components
    tool = _get_tool(orchestrator, "optimize_service_performance")

    response = await tool()

    assert response["success"] is True
    assert discovery.refresh_calls == [True]
    assert response["tool_capabilities"]
