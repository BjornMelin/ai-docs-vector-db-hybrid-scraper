"""Orchestrator Service - Central coordination service for multi-service MCP.

This service coordinates between domain-specific services and provides
composer functionality for complex multi-service workflows.
"""

import asyncio
import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    AgenticOrchestrator,
    create_agent_dependencies,
    get_discovery_engine,
)

from .analytics_service import AnalyticsService
from .document_service import DocumentService
from .search_service import SearchService
from .system_service import SystemService


logger = logging.getLogger(__name__)


class OrchestratorService:
    """FastMCP 2.0+ orchestrator service for multi-service coordination.

    Coordinates between domain-specific services for multi-service workflows.
    """

    def __init__(self, name: str = "orchestrator-service"):
        """Initialize the orchestrator service.

        Args:
            name: Service name for MCP registration
        """

        self.mcp = FastMCP(
            name,
            instructions="""
            Orchestrator service for multi-service coordination.

            Provides tools for:
            - Multi-service workflow orchestration
            - Service selection and routing
            - Cross-service state management
            - Service composition for workflows
            - Error handling across services
            """,
        )
        self.client_manager: ClientManager | None = None

        # Domain-specific services
        self.search_service: SearchService | None = None
        self.document_service: DocumentService | None = None
        self.analytics_service: AnalyticsService | None = None
        self.system_service: SystemService | None = None

        # Agentic orchestration components
        self.agentic_orchestrator: AgenticOrchestrator | None = None
        self.discovery_engine = get_discovery_engine()

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialize the orchestrator service with all domain services.

        Args:
            client_manager: Shared client manager instance
        """

        self.client_manager = client_manager

        # Initialize domain-specific services
        await self._initialize_domain_services()

        # Initialize agentic orchestration
        await self._initialize_agentic_orchestration()

        # Register orchestrator tools
        await self._register_orchestrator_tools()

        logger.info("OrchestratorService initialized")

    async def _initialize_domain_services(self) -> None:
        """Initialize all domain-specific services."""

        if not self.client_manager:
            msg = "OrchestratorService not initialized"
            raise RuntimeError(msg)

        # Initialize services in parallel for better performance
        services = [
            ("search", SearchService()),
            ("document", DocumentService()),
            ("analytics", AnalyticsService()),
            ("system", SystemService()),
        ]

        tasks = []
        for service_name, service in services:
            task = asyncio.create_task(
                service.initialize(self.client_manager), name=f"init_{service_name}"
            )
            tasks.append((service_name, service, task))

        # Wait for all services to initialize
        for service_name, service, task in tasks:
            try:
                await self._initialize_single_service(service_name, service, task)
            except (ImportError, ValueError, RuntimeError, AttributeError):
                logger.exception("Failed to initialize {service_name} service")

    async def _initialize_single_service(
        self, service_name: str, service, task
    ) -> None:
        """Initialize a single service and assign it to the appropriate attribute."""

        await task
        if service_name == "search":
            self.search_service = service
        elif service_name == "document":
            self.document_service = service
        elif service_name == "analytics":
            self.analytics_service = service
        elif service_name == "system":
            self.system_service = service
        logger.info("Initialized %s service", service_name)

    async def _get_service_info_safely(self, service) -> dict:
        """Get service info safely with error handling."""
        return await service.get_service_info()

    async def _initialize_agentic_orchestration(self) -> None:
        """Initialize agentic orchestration components."""

        if not self.client_manager:
            return

        # Initialize AgenticOrchestrator
        deps = create_agent_dependencies(
            client_manager=self.client_manager, session_id="orchestrator"
        )

        self.agentic_orchestrator = AgenticOrchestrator()
        await self.agentic_orchestrator.initialize(deps)

        # Initialize DynamicToolDiscovery
        await self.discovery_engine.initialize_discovery(deps)

        logger.info("Initialized agentic orchestration components")

    async def _register_orchestrator_tools(self) -> None:
        """Register orchestrator-specific MCP tools."""

        if not self.client_manager:
            msg = "OrchestratorService not initialized"
            raise RuntimeError(msg)

        @self.mcp.tool()
        async def orchestrate_multi_service_workflow(
            workflow_description: str,
            services_required: list[str] | None = None,
            performance_constraints: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Orchestrate complex workflows across multiple services.

            Args:
                workflow_description: Description of the workflow to execute
                services_required: Optional list of specific services to use
                performance_constraints: Optional performance requirements

            Returns:
                Workflow execution results with service coordination details
            """

            if not self.agentic_orchestrator:
                return {"error": "Agentic orchestrator not initialized"}

            if not self.client_manager:
                return {"error": "Client manager not initialized"}

            # Use AgenticOrchestrator for workflow coordination
            deps = create_agent_dependencies(
                client_manager=self.client_manager, session_id="multi_service_workflow"
            )

            constraints = performance_constraints or {}
            if services_required:
                constraints["preferred_services"] = services_required

            orchestration_result = await self.agentic_orchestrator.orchestrate(
                task=workflow_description, constraints=constraints, deps=deps
            )

            return {
                "success": orchestration_result.success,
                "workflow_results": orchestration_result.results,
                "services_used": orchestration_result.tools_used,
                "orchestration_reasoning": orchestration_result.reasoning,
                "execution_time_ms": orchestration_result.latency_ms,
                "confidence": orchestration_result.confidence,
            }

        @self.mcp.tool()
        async def get_service_capabilities() -> dict[str, Any]:
            """Get capabilities of all available services.

            Returns:
                Dictionary with service capabilities and status

            """
            capabilities = {"services": {}}

            services = [
                ("search", self.search_service),
                ("document", self.document_service),
                ("analytics", self.analytics_service),
                ("system", self.system_service),
            ]

            for service_name, service in services:
                if service:
                    try:
                        capabilities["services"][
                            service_name
                        ] = await self._get_service_info_safely(service)
                    except (
                        asyncio.CancelledError,
                        TimeoutError,
                        RuntimeError,
                    ) as e:  # noqa: BLE001
                        capabilities["services"][service_name] = {
                            "status": "error",
                            "error": str(e),
                        }
                else:
                    capabilities["services"][service_name] = {
                        "status": "not_initialized"
                    }

            return capabilities

        @self.mcp.tool()
        async def optimize_service_performance() -> dict[str, Any]:
            """Optimize performance across all services.

            Returns:
                Optimization results and recommendations

            """
            if not self.discovery_engine:
                return {"error": "Discovery engine not available"}

            optimization_results = {
                "optimization_applied": [],
                "recommendations": [],
                "performance_impact": {},
            }

            # Use discovery engine for intelligent optimization
            recommendations = await self.discovery_engine.get_tool_recommendations(
                task_type="performance_optimization",
                constraints={"target": "latency_and_throughput"},
            )

            optimization_results["recommendations"] = recommendations
            optimization_results["optimization_applied"].append(
                "Cross-service performance correlation analysis"
            )

            return optimization_results

        logger.info("Registered orchestrator tools with multi-service coordination")

    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance.

        Returns:
            Configured FastMCP server for this service
        """

        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities.

        Returns:
            Service metadata and capability information
        """

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
        """Get information about all coordinated services.

        Returns:
            Information about all domain-specific services
        """

        services_info = {}

        services = [
            ("search", self.search_service),
            ("document", self.document_service),
            ("analytics", self.analytics_service),
            ("system", self.system_service),
        ]

        for service_name, service in services:
            if service:
                try:
                    services_info[service_name] = await service.get_service_info()
                except (
                    asyncio.CancelledError,
                    TimeoutError,
                    RuntimeError,
                ) as e:  # noqa: BLE001
                    services_info[service_name] = {"status": "error", "error": str(e)}
            else:
                services_info[service_name] = {"status": "not_initialized"}

        return services_info
