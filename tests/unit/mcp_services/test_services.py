"""Focused tests for MCP service orchestration."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_services.analytics_service import AnalyticsService
from src.mcp_services.document_service import DocumentService
from src.mcp_services.orchestrator_service import OrchestratorService
from src.mcp_services.search_service import SearchService
from src.mcp_services.system_service import SystemService
from src.services.observability.ai_tracking import AIOperationTracker


@pytest.fixture
async def client_manager():
    """Provide a reusable async client manager mock."""
    manager = AsyncMock()
    manager.get_qdrant_service = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_search_service_initialization(client_manager):
    service = SearchService()
    with patch.object(
        service, "_register_search_tools", new_callable=AsyncMock
    ) as register:
        await service.initialize(client_manager)

    assert service.client_manager is client_manager
    register.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_service_registers_modules(client_manager):
    service = SearchService()
    service.client_manager = client_manager

    module = SimpleNamespace(register_tools=MagicMock())
    with patch.multiple(
        "src.mcp_services.search_service",
        hybrid_search=module,
        hyde_search=module,
        multi_stage_search=module,
        search_with_reranking=module,
        web_search=module,
    ):
        await service._register_search_tools()

    assert module.register_tools.call_count == 5


@pytest.mark.asyncio
async def test_document_service_registers_modules(client_manager):
    service = DocumentService()
    service.client_manager = client_manager

    module = SimpleNamespace(register_tools=MagicMock())
    with patch.multiple(
        "src.mcp_services.document_service",
        document_management=module,
        collections=module,
        projects=module,
        crawling=module,
        content_intelligence=module,
    ):
        await service._register_document_tools()

    assert module.register_tools.call_count == 5


@pytest.mark.asyncio
async def test_system_service_registers_modules(client_manager):
    service = SystemService()
    service.client_manager = client_manager

    module = SimpleNamespace(register_tools=MagicMock())
    with patch.multiple(
        "src.mcp_services.system_service",
        system_health=module,
        configuration=module,
        cost_estimation=module,
        embeddings=module,
        filtering=module,
    ):
        await service._register_system_tools()

    assert module.register_tools.call_count == 5


@pytest.mark.asyncio
async def test_analytics_service_registers_modules(client_manager):
    service = AnalyticsService()
    service.client_manager = client_manager

    module = SimpleNamespace(register_tools=MagicMock())
    with (
        patch.object(
            service, "_register_enhanced_observability_tools", new_callable=AsyncMock
        ) as enhanced,
        patch.multiple(
            "src.mcp_services.analytics_service",
            analytics=module,
            query_processing=module,
            agentic_rag=module,
        ),
    ):
        await service._register_analytics_tools()

    assert module.register_tools.call_count == 3
    enhanced.assert_awaited_once()


@pytest.mark.asyncio
async def test_analytics_service_observability_tools():
    service = AnalyticsService()
    registered: dict[str, Callable[..., Any]] = {}

    def capture_tool(*_, **__):
        def decorator(func):
            registered[func.__name__] = func
            return func

        return decorator

    service.ai_tracker = MagicMock(spec=AIOperationTracker)
    service.correlation_manager = MagicMock()

    with patch.object(service.mcp, "tool", side_effect=capture_tool):
        await service._register_enhanced_observability_tools()

    decision_tool = registered["get_agentic_decision_metrics"]
    workflow_tool = registered["get_multi_agent_workflow_visualization"]

    decision_metrics = await decision_tool(agent_id="agent-1", time_range_minutes=15)
    workflow_metrics = await workflow_tool()

    assert decision_metrics["integration_status"] == "using_existing_ai_tracker"
    assert (
        workflow_metrics["integration_status"] == "using_existing_correlation_manager"
    )


@pytest.mark.asyncio
async def test_orchestrator_service_initialize(client_manager):
    service = OrchestratorService()
    with (
        patch.object(
            service, "_initialize_domain_services", new_callable=AsyncMock
        ) as init_domain,
        patch.object(
            service, "_initialize_agentic_orchestration", new_callable=AsyncMock
        ) as init_agentic,
        patch.object(
            service, "_register_orchestrator_tools", new_callable=AsyncMock
        ) as register,
    ):
        await service.initialize(client_manager)

    assert service.client_manager is client_manager
    init_domain.assert_awaited_once()
    init_agentic.assert_awaited_once()
    register.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_service_registers_tool(client_manager):
    service = OrchestratorService()
    service.client_manager = client_manager

    orchestrator = AsyncMock()
    orchestrator.orchestrate = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            results={"status": "ok"},
            tools_used=["search"],
            reasoning="done",
            latency_ms=120.0,
            confidence=0.9,
        )
    )
    service.agentic_orchestrator = orchestrator

    registered: dict[str, Callable[..., Any]] = {}

    def capture_tool(*_, **__):
        def decorator(func):
            registered[func.__name__] = func
            return func

        return decorator

    with (
        patch(
            "src.mcp_services.orchestrator_service.create_agent_dependencies",
            return_value={"deps": True},
        ),
        patch.object(service.mcp, "tool", side_effect=capture_tool),
    ):
        await service._register_orchestrator_tools()

    tool = registered["orchestrate_multi_service_workflow"]
    result = await tool(
        workflow_description="research",
        services_required=["search"],
        performance_constraints={"latency_ms": 200},
    )

    assert result["success"] is True
    orchestrator.orchestrate.assert_awaited_once()

    service.agentic_orchestrator = None
    error = await tool(workflow_description="fail")
    assert error == {"error": "Agentic orchestrator not initialized"}


@pytest.mark.asyncio
async def test_orchestrator_agentic_initialization(client_manager):
    service = OrchestratorService()
    service.client_manager = client_manager
    service.discovery_engine = AsyncMock()

    orchestrator_instance = AsyncMock()
    with (
        patch(
            "src.mcp_services.orchestrator_service.AgenticOrchestrator",
            return_value=orchestrator_instance,
        ),
        patch(
            "src.mcp_services.orchestrator_service.create_agent_dependencies",
            return_value={"deps": True},
        ),
    ):
        await service._initialize_agentic_orchestration()

    orchestrator_instance.initialize.assert_awaited_once()
    service.discovery_engine.initialize_discovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_methods_raise_when_uninitialized():
    search_service = SearchService()
    document_service = DocumentService()
    analytics_service = AnalyticsService()
    system_service = SystemService()

    with pytest.raises(RuntimeError):
        await search_service._register_search_tools()
    with pytest.raises(RuntimeError):
        await document_service._register_document_tools()
    with pytest.raises(RuntimeError):
        await analytics_service._register_analytics_tools()
    with pytest.raises(RuntimeError):
        await system_service._register_system_tools()
