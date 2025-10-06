"""Orchestrator service providing LangGraph-driven coordination tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    DynamicToolDiscovery,
    GraphRunner,
    RetrievalHelper,
    ToolExecutionService,
)

from .analytics_service import AnalyticsService
from .document_service import DocumentService
from .search_service import SearchService
from .system_service import SystemService


logger = logging.getLogger(__name__)


class OrchestratorService:  # pylint: disable=too-many-instance-attributes
    """FastMCP service coordinating domain-specific MCP tools."""

    def __init__(self, name: str = "orchestrator-service") -> None:
        instructions = (
            "Coordinate domain tools, surface orchestration telemetry, and expose "
            "LangGraph-based workflows for multi-service requests."
        )
        self.mcp = FastMCP(name, instructions=instructions)
        self.client_manager: ClientManager | None = None

        self.search_service: SearchService | None = None
        self.document_service: DocumentService | None = None
        self.analytics_service: AnalyticsService | None = None
        self.system_service: SystemService | None = None

        self._discovery: DynamicToolDiscovery | None = None
        self._graph_runner: GraphRunner | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialise domain services and register orchestrator tools."""

        self.client_manager = client_manager
        await self._initialize_domain_services()
        await self._initialize_agentic_components()
        await self._register_orchestrator_tools()
        logger.info("OrchestratorService initialized")

    async def _initialize_domain_services(self) -> None:
        if not self.client_manager:
            msg = "OrchestratorService not initialized"
            raise RuntimeError(msg)

        services = [
            ("search", SearchService()),
            ("document", DocumentService()),
            ("analytics", AnalyticsService()),
            ("system", SystemService()),
        ]

        tasks = [
            (
                name,
                service,
                asyncio.create_task(
                    service.initialize(self.client_manager), name=f"init_{name}"
                ),
            )
            for name, service in services
        ]

        for service_name, service, task in tasks:
            try:
                await task
            except Exception:  # pragma: no cover - defensive log
                logger.exception("Failed to initialize %s service", service_name)
                continue
            if service_name == "search":
                self.search_service = service
            elif service_name == "document":
                self.document_service = service
            elif service_name == "analytics":
                self.analytics_service = service
            elif service_name == "system":
                self.system_service = service
            logger.info("Initialized %s service", service_name)

    async def _initialize_agentic_components(self) -> None:
        if not self.client_manager:
            return

        discovery = DynamicToolDiscovery(self.client_manager)
        tool_service = ToolExecutionService(self.client_manager)
        retrieval_helper = RetrievalHelper(self.client_manager)
        agentic_cfg = getattr(self.client_manager.config, "agentic", None)
        max_parallel = getattr(agentic_cfg, "max_parallel_tools", 3)
        run_timeout = getattr(agentic_cfg, "run_timeout_seconds", 30.0)
        retrieval_limit = getattr(agentic_cfg, "retrieval_limit", 8)
        self._discovery = discovery
        self._graph_runner = GraphRunner(
            client_manager=self.client_manager,
            discovery=discovery,
            tool_service=tool_service,
            retrieval_helper=retrieval_helper,
            max_parallel_tools=max_parallel,
            run_timeout_seconds=run_timeout,
            retrieval_limit=retrieval_limit,
        )
        await discovery.refresh(force=True)
        logger.info("initialized LangGraph runner for orchestrator")

    async def _register_orchestrator_tools(self) -> None:
        if not self.client_manager:
            msg = "OrchestratorService not initialized"
            raise RuntimeError(msg)

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
            outcome = await self._graph_runner.run_search(
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
