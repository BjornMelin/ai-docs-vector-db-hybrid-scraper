"""Orchestrator service providing LangGraph-driven coordination tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from fastmcp import FastMCP

from src.services.agents import (
    DynamicToolDiscovery,
    GraphRunner,
)

from .analytics_service import AnalyticsService
from .document_service import DocumentService
from .search_service import SearchService
from .system_service import SystemService


logger = logging.getLogger(__name__)


class OrchestratorService:  # pylint: disable=too-many-instance-attributes
    """FastMCP service coordinating domain-specific MCP tools."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str = "orchestrator-service",
        *,
        search_service: SearchService,
        document_service: DocumentService,
        analytics_service: AnalyticsService,
        system_service: SystemService,
        graph_runner: GraphRunner | None = None,
        discovery: DynamicToolDiscovery | None = None,
    ) -> None:
        instructions = (
            "Coordinate domain tools, surface orchestration telemetry, and expose "
            "LangGraph-based workflows for multi-service requests."
        )
        self.mcp = FastMCP(name, instructions=instructions)
        self.search_service = search_service
        self.document_service = document_service
        self.analytics_service = analytics_service
        self.system_service = system_service
        if discovery is None or graph_runner is None:
            discovery, _, _, graph_runner = GraphRunner.build_components(
                discovery=discovery
            )
        self._discovery: DynamicToolDiscovery | None = discovery
        self._graph_runner: GraphRunner | None = graph_runner
        if self._discovery:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._discovery.refresh(force=True))
            except RuntimeError:
                logger.debug("Event loop not running; discovery refresh deferred")
        else:  # pragma: no cover - defensive branch
            logger.warning("Dynamic tool discovery not initialised for orchestrator")
        logger.info("Initialized LangGraph components for orchestrator")

        self._register_orchestrator_tools()

    def _register_orchestrator_tools(self) -> None:
        """Register FastMCP tool handlers exposed by the orchestrator."""

        @self.mcp.tool()  # pylint: disable=no-value-for-parameter
        async def orchestrate_multi_service_workflow(
            workflow_description: str,
            services_required: list[str] | None = None,
            performance_constraints: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Execute a LangGraph search workflow with orchestration context."""

            if not self._graph_runner:
                return {"success": False, "error": "graph_runner_not_initialized"}

            user_context = {
                "services_required": services_required or [],
                "performance_constraints": performance_constraints or {},
            }
            graph_runner = cast(GraphRunner, self._graph_runner)
            outcome = await graph_runner.run_search(  # pylint: disable=no-member
                query=workflow_description,
                collection="documentation",
                user_context=user_context,
            )
            return {
                "success": outcome.success,
                "workflow_results": {
                    "answer": outcome.answer,
                    "results": outcome.results,
                    "reasoning": outcome.reasoning,
                },
                "tools_used": outcome.tools_used,
                "metrics": outcome.metrics,
                "errors": outcome.errors,
            }

        @self.mcp.tool()  # pylint: disable=no-value-for-parameter
        async def get_service_capabilities() -> dict[str, Any]:
            """Return capabilities for each domain service."""

            capabilities = {"services": {}}
            services = [
                ("search", self.search_service),
                ("document", self.document_service),
                ("analytics", self.analytics_service),
                ("system", self.system_service),
            ]
            for service_name, service in services:
                if service is None:
                    capabilities["services"][service_name] = {
                        "status": "not_initialized"
                    }
                    continue
                try:
                    capabilities["services"][
                        service_name
                    ] = await service.get_service_info()
                except Exception as exc:  # pragma: no cover - defensive log
                    capabilities["services"][service_name] = {
                        "status": "error",
                        "error": str(exc),
                    }
            return capabilities

        @self.mcp.tool()  # pylint: disable=no-value-for-parameter
        async def optimize_service_performance() -> dict[str, Any]:
            """Refresh tool discovery information for optimisation workflows."""

            if not self._discovery:
                return {"success": False, "error": "discovery_not_initialized"}
            await self._discovery.refresh(force=True)
            capabilities = [
                cap.model_dump() for cap in self._discovery.get_capabilities()
            ]
            return {
                "success": True,
                "tool_capabilities": capabilities,
            }

    async def get_service_info(self) -> dict[str, Any]:
        """Return orchestrator service metadata."""

        return {
            "service": "orchestrator",
            "version": "2.0",
            "capabilities": [
                "multi_service_coordination",
                "workflow_orchestration",
                "service_composition",
                "cross_service_management",
                "error_handling",
            ],
            "status": "active",
        }

    async def get_all_services(self) -> dict[str, Any]:
        """Return metadata for all coordinated services."""

        services_info: dict[str, Any] = {}
        services = [
            ("search", self.search_service),
            ("document", self.document_service),
            ("analytics", self.analytics_service),
            ("system", self.system_service),
        ]
        for name, service in services:
            if service is None:
                services_info[name] = {"status": "not_initialized"}
                continue
            try:
                services_info[name] = await service.get_service_info()
            except Exception as exc:  # pragma: no cover - defensive log
                services_info[name] = {"status": "error", "error": str(exc)}
        return services_info

    def get_mcp_server(self) -> FastMCP:
        """Return the configured FastMCP server instance."""

        return self.mcp
