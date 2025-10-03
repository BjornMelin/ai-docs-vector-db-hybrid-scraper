# pylint: disable=duplicate-code,R0801

"""Tests for the LangGraph-backed orchestrator service."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, cast

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative: str) -> types.ModuleType:
    module_path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)  # type: ignore[arg-type]
    return loaded


_load_module("src.services.errors", "src/services/errors.py")
_load_module("src.services.agents", "src/services/agents/__init__.py")
_load_module(
    "src.services.agents.dynamic_tool_discovery",
    "src/services/agents/dynamic_tool_discovery.py",
)
_load_module("src.services.agents.retrieval", "src/services/agents/retrieval.py")
_load_module(
    "src.services.agents.tool_execution_service",
    "src/services/agents/tool_execution_service.py",
)
graph_module = _load_module(
    "src.services.agents.langgraph_runner",
    "src/services/agents/langgraph_runner.py",
)


class _ToolNamespace(types.SimpleNamespace):
    def register_tools(self, *_, **__) -> None:  # pragma: no cover - stub
        return None


class _ToolsModule(types.ModuleType):
    def __getattr__(self, name: str) -> _ToolNamespace:  # pragma: no cover - stub
        stub = _ToolNamespace()
        setattr(self, name, stub)
        return stub


tools_stub = _ToolsModule("src.mcp_tools.tools")
sys.modules["src.mcp_tools.tools"] = tools_stub
services_pkg = types.ModuleType("src.mcp_services")
services_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules["src.mcp_services"] = services_pkg
analytics_module = _load_module(
    "src.mcp_services.analytics_service",
    "src/mcp_services/analytics_service.py",
)
document_module = _load_module(
    "src.mcp_services.document_service",
    "src/mcp_services/document_service.py",
)
search_module = _load_module(
    "src.mcp_services.search_service",
    "src/mcp_services/search_service.py",
)
system_module = _load_module(
    "src.mcp_services.system_service",
    "src/mcp_services/system_service.py",
)
setattr(services_pkg, "analytics_service", analytics_module)
setattr(services_pkg, "document_service", document_module)
setattr(services_pkg, "search_service", search_module)
setattr(services_pkg, "system_service", system_module)
orchestrator_module = _load_module(
    "src.mcp_services.orchestrator_service",
    "src/mcp_services/orchestrator_service.py",
)

OrchestratorService = cast(Any, orchestrator_module.OrchestratorService)
GraphSearchOutcome = cast(Any, graph_module.GraphSearchOutcome)


class DummyClientManager:
    async def get_mcp_client(self):  # pragma: no cover - not used in tests
        raise AssertionError("get_mcp_client should not be called")


def _install_agentic_stubs() -> Dict[str, Any]:
    created: Dict[str, Any] = {"originals": {}}

    class DiscoveryStub:
        def __init__(self, client_manager, **_kwargs) -> None:
            created["discovery_client"] = client_manager
            created["discovery_instance"] = self
            self.refresh_calls: list[bool] = []

        async def refresh(self, *, force: bool = False) -> None:
            self.refresh_calls.append(force)

        def get_capabilities(self):  # pragma: no cover - not needed
            return ()

    class ToolServiceStub:
        def __init__(self, client_manager, **_kwargs) -> None:
            created["tool_service_client"] = client_manager

    class RetrievalStub:
        def __init__(self, client_manager, **_kwargs) -> None:
            created["retrieval_client"] = client_manager

    class GraphRunnerStub:
        def __init__(self, **kwargs) -> None:
            created["graph_kwargs"] = kwargs

        async def run_search(self, **_kwargs):  # pragma: no cover - set later
            raise AssertionError("run_search not configured")

    originals = cast(Dict[str, Any], created["originals"])
    originals["DynamicToolDiscovery"] = orchestrator_module.DynamicToolDiscovery
    originals["ToolExecutionService"] = orchestrator_module.ToolExecutionService
    originals["RetrievalHelper"] = orchestrator_module.RetrievalHelper
    originals["GraphRunner"] = orchestrator_module.GraphRunner
    setattr(orchestrator_module, "DynamicToolDiscovery", DiscoveryStub)
    setattr(orchestrator_module, "ToolExecutionService", ToolServiceStub)
    setattr(orchestrator_module, "RetrievalHelper", RetrievalStub)
    setattr(orchestrator_module, "GraphRunner", GraphRunnerStub)
    return created


@pytest.mark.asyncio
async def test_initialize_agentic_components_builds_graph_runner():
    created = _install_agentic_stubs()
    try:
        service = OrchestratorService()
        service.client_manager = DummyClientManager()

        await service._initialize_agentic_components()

        assert isinstance(service._graph_runner, object)
        assert created["discovery_client"] is service.client_manager
        assert created["tool_service_client"] is service.client_manager
        assert created["retrieval_client"] is service.client_manager
        assert created["graph_kwargs"]["run_timeout_seconds"] == 30.0
        assert created["discovery_instance"].refresh_calls == [True]
    finally:
        originals = cast(Dict[str, Any], created["originals"])
        for attr, original in originals.items():
            setattr(orchestrator_module, attr, original)


@pytest.mark.asyncio
async def test_orchestrate_multi_service_workflow_uses_graph_runner():
    created = _install_agentic_stubs()
    service = OrchestratorService()
    service.client_manager = DummyClientManager()

    async def no_refresh(**_kwargs):
        return None

    service._discovery = types.SimpleNamespace(
        refresh=no_refresh,
        get_capabilities=lambda: (),
    )

    async def fake_run_search(**_kwargs):
        return GraphSearchOutcome(
            success=True,
            session_id="sid",
            answer="done",
            confidence=0.8,
            results=[{"id": "1"}],
            tools_used=["semantic_search"],
            reasoning=["step"],
            metrics={"latency_ms": 12.0, "tool_count": 1, "error_count": 0},
            errors=[],
        )

    service._graph_runner = types.SimpleNamespace(run_search=fake_run_search)

    registered: Dict[str, Any] = {}

    def fake_tool_decorator(func=None):
        def _register(inner):
            registered[inner.__name__] = inner
            return inner

        if func is None:
            return _register
        return _register(func)

    service.mcp.tool = fake_tool_decorator

    try:
        await service._register_orchestrator_tools()

        workflow = cast(
            Any, registered["orchestrate_multi_service_workflow"]
        )
        result = await workflow(
            workflow_description="desc",
            services_required=["search"],
            performance_constraints=None,
        )

        assert result["success"] is True
        assert result["workflow_results"]["answer"] == "done"
        assert result["workflow_results"]["results"]
    finally:
        originals = cast(Dict[str, Any], created["originals"])
        for attr, original in originals.items():
            setattr(orchestrator_module, attr, original)
