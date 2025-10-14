"""Unit tests for :mod:`src.mcp_services.orchestrator_service`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_services.orchestrator_service import OrchestratorService


@pytest.fixture()
def orchestrator_fixture() -> tuple[
    OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
]:
    """Build an orchestrator with fully mocked dependencies."""
    search_service = SimpleNamespace(
        get_service_info=AsyncMock(
            return_value={"service": "search", "status": "active"}
        )
    )
    document_service = SimpleNamespace(
        get_service_info=AsyncMock(
            return_value={"service": "document", "status": "active"}
        )
    )
    analytics_service = SimpleNamespace(
        get_service_info=AsyncMock(
            return_value={"service": "analytics", "status": "active"}
        )
    )
    system_service = SimpleNamespace(
        get_service_info=AsyncMock(
            return_value={"service": "system", "status": "active"}
        )
    )

    discovery = MagicMock()
    discovery.refresh = AsyncMock()
    discovery.get_capabilities.return_value = (
        SimpleNamespace(model_dump=lambda: {"name": "workflow-agent"}),  # type: ignore[arg-type]
    )

    graph_runner = MagicMock()
    graph_runner.run_search = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            answer="Vector DB overview",
            results=[{"id": "doc-1"}],
            reasoning=["used reranker"],
            tools_used=["vector-search"],
            metrics={"latency_ms": 123},
            errors=[],
        )
    )

    orchestrator = OrchestratorService(
        search_service=search_service,  # type: ignore[arg-type]
        document_service=document_service,  # type: ignore[arg-type]
        analytics_service=analytics_service,  # type: ignore[arg-type]
        system_service=system_service,  # type: ignore[arg-type]
        discovery=discovery,
        graph_runner=graph_runner,
    )

    services = {
        "search": search_service.get_service_info,
        "document": document_service.get_service_info,
        "analytics": analytics_service.get_service_info,
        "system": system_service.get_service_info,
    }

    return orchestrator, discovery, graph_runner, services


@pytest.mark.asyncio()
async def test_orchestrate_workflow_delegates_to_graph_runner(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify workflow orchestration delegates to the graph runner."""
    orchestrator, _, graph_runner, _ = orchestrator_fixture

    server = orchestrator.get_mcp_server()
    tool = await server.get_tool("orchestrate_multi_service_workflow")
    result = await tool.run(
        {
            "workflow_description": "Summarise latest vector search updates",
            "services_required": ["search", "analytics"],
            "performance_constraints": {"latency_budget_ms": 500},
        }
    )

    graph_runner.run_search.assert_awaited_once_with(
        query="Summarise latest vector search updates",
        collection="documentation",
        user_context={
            "services_required": ["search", "analytics"],
            "performance_constraints": {"latency_budget_ms": 500},
        },
    )
    payload = result.structured_content
    assert payload is not None
    assert payload["success"] is True
    assert payload["workflow_results"]["answer"] == "Vector DB overview"
    assert payload["workflow_results"]["results"] == [{"id": "doc-1"}]


@pytest.mark.asyncio()
async def test_orchestrate_workflow_handles_missing_runner(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify error handling when graph runner is not initialized."""
    orchestrator, _, _, _ = orchestrator_fixture
    orchestrator._graph_runner = None  # type: ignore[attr-defined]

    tool = await orchestrator.get_mcp_server().get_tool(
        "orchestrate_multi_service_workflow"
    )
    result = await tool.run({"workflow_description": "anything"})

    payload = result.structured_content
    assert payload is not None
    assert payload == {
        "success": False,
        "error": "graph_runner_not_initialized",
    }


@pytest.mark.asyncio()
async def test_get_service_capabilities_aggregates_service_metadata(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify aggregation of capabilities across all services."""
    orchestrator, _, _, services = orchestrator_fixture
    # Force analytics to raise to exercise error branch
    services["analytics"].side_effect = RuntimeError("telemetry offline")

    tool = await orchestrator.get_mcp_server().get_tool("get_service_capabilities")
    result = await tool.run({})

    payload = result.structured_content
    assert payload is not None
    services = payload["services"]
    assert services["search"]["service"] == "search"
    assert services["analytics"]["status"] == "error"
    assert "telemetry offline" in services["analytics"]["error"]


@pytest.mark.asyncio()
async def test_optimize_service_performance_refreshes_discovery(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify performance optimization triggers discovery refresh."""
    orchestrator, discovery, _, _ = orchestrator_fixture

    tool = await orchestrator.get_mcp_server().get_tool("optimize_service_performance")
    result = await tool.run({})

    discovery.refresh.assert_awaited_once_with(force=True)
    payload = result.structured_content
    assert payload is not None
    assert payload["success"] is True
    assert payload["tool_capabilities"] == [{"name": "workflow-agent"}]


@pytest.mark.asyncio()
async def test_optimize_service_performance_reports_missing_discovery(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify error reporting when discovery is not initialized."""
    orchestrator, _, _, _ = orchestrator_fixture
    orchestrator._discovery = None  # type: ignore[attr-defined]

    tool = await orchestrator.get_mcp_server().get_tool("optimize_service_performance")
    result = await tool.run({})

    payload = result.structured_content
    assert payload is not None
    assert payload == {
        "success": False,
        "error": "discovery_not_initialized",
    }


@pytest.mark.asyncio()
async def test_get_all_services_returns_metadata(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify retrieval of metadata for all registered services."""
    orchestrator, _, _, _ = orchestrator_fixture

    info = await orchestrator.get_all_services()

    assert set(info.keys()) == {"search", "document", "analytics", "system"}
    assert info["search"]["status"] == "active"


@pytest.mark.asyncio()
async def test_get_service_info_static_payload(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify static service info payload structure."""
    orchestrator, _, _, _ = orchestrator_fixture

    info = await orchestrator.get_service_info()

    assert info["service"] == "orchestrator"
    assert "workflow_orchestration" in info["capabilities"]


def test_get_mcp_server_returns_fastmcp_instance(
    orchestrator_fixture: tuple[
        OrchestratorService, MagicMock, MagicMock, dict[str, AsyncMock]
    ],
) -> None:
    """Verify MCP server instance is returned correctly."""
    orchestrator, _, _, _ = orchestrator_fixture

    server = orchestrator.get_mcp_server()

    assert hasattr(server, "tool")
    assert server.name == "orchestrator-service"
