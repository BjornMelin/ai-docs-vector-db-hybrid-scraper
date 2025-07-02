"""System Service - Domain-specific MCP server for system operations.

This service handles all system-related functionality including health monitoring,
metrics collection, and infrastructure management.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    configuration,
    cost_estimation,
    embeddings,
    filtering,
    system_health,
)


logger = logging.getLogger(__name__)


class SystemService:
    """FastMCP 2.0+ system service with self-healing capabilities.

    Implements autonomous system monitoring and management with
    self-healing capabilities and predictive maintenance.
    """

    def __init__(self, name: str = "system-service"):
        """Initialize the system service.

        Args:
            name: Service name for MCP registration
        """
        self.mcp = FastMCP(
            name,
            instructions="""
            Advanced system service with self-healing and monitoring capabilities.

            Features:
            - Real-time system health monitoring and alerting
            - Autonomous resource management and optimization
            - Configuration management with intelligent defaults
            - Cost estimation and budget optimization
            - Embedding management with provider optimization
            - Advanced filtering and data processing

            Autonomous Capabilities:
            - Self-healing system recovery and fault tolerance
            - Predictive maintenance and resource scaling
            - Intelligent configuration optimization
            - Cost and performance correlation analysis
            """,
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialize the system service with client manager.

        Args:
            client_manager: Shared client manager instance
        """
        self.client_manager = client_manager

        # Register system tools
        await self._register_system_tools()

        logger.info("SystemService initialized with self-healing capabilities")

    async def _register_system_tools(self) -> None:
        """Register all system-related MCP tools."""
        if not self.client_manager:
            msg = "SystemService not initialized"
            raise RuntimeError(msg)

        # Register core system tools
        system_health.register_tools(self.mcp, self.client_manager)
        configuration.register_tools(self.mcp, self.client_manager)
        cost_estimation.register_tools(self.mcp, self.client_manager)
        embeddings.register_tools(self.mcp, self.client_manager)
        filtering.register_tools(self.mcp, self.client_manager)

        logger.info("Registered system tools with self-healing capabilities")

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
            "service": "system",
            "version": "2.0",
            "capabilities": [
                "health_monitoring",
                "resource_management",
                "configuration_management",
                "cost_estimation",
                "embedding_optimization",
                "data_filtering",
                "self_healing",
            ],
            "autonomous_features": [
                "fault_tolerance",
                "predictive_maintenance",
                "configuration_optimization",
                "cost_intelligence",
            ],
            "status": "active",
            "research_basis": "SYSTEM_INFRASTRUCTURE",
        }
