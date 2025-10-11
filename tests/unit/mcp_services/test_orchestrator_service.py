"""Tests for OrchestratorService dependency integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_services.orchestrator_service import OrchestratorService


@pytest.fixture()
def orchestrator_components() -> tuple[OrchestratorService, MagicMock, MagicMock]:
    """Return an orchestrator wired with mocked services and discovery components."""

    search_service = MagicMock()
    document_service = MagicMock()
    analytics_service = MagicMock()
    system_service = MagicMock()

    discovery = MagicMock()
    discovery.refresh = AsyncMock()
    discovery.get_capabilities.return_value = (SimpleNamespace(model_dump=lambda: {}),)

    graph_runner = MagicMock()
    graph_runner.run_search = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            answer="ok",
            results=[],
            reasoning=[],
            tools_used=[],
            metrics={},
            errors=[],
        )
    )

    orchestrator = OrchestratorService(
        search_service=search_service,
        document_service=document_service,
        analytics_service=analytics_service,
        system_service=system_service,
        discovery=discovery,
        graph_runner=graph_runner,
    )
    return orchestrator, discovery, graph_runner


@pytest.mark.asyncio()
async def test_orchestrate_workflow_uses_graph_runner(
    orchestrator_components: tuple[OrchestratorService, MagicMock, MagicMock],
) -> None:
    """The orchestrate workflow tool should delegate to the injected graph runner."""

    orchestrator, _, runner = orchestrator_components
    tool = orchestrator.get_mcp_server().tools["orchestrate_multi_service_workflow"]
    result = await tool("search docs")

    assert result["success"] is True
    runner.run_search.assert_awaited_once()


@pytest.mark.asyncio()
async def test_optimize_service_performance_triggers_refresh(
    orchestrator_components: tuple[OrchestratorService, MagicMock, MagicMock],
) -> None:
    """The optimise tool should call discovery.refresh."""

    orchestrator, discovery, _ = orchestrator_components
    tool = orchestrator.get_mcp_server().tools["optimize_service_performance"]
    await tool()

    discovery.refresh.assert_awaited_once_with(force=True)
