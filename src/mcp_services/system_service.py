"""System Service - Domain-specific MCP server for system operations.

This service handles all system-related functionality including health monitoring,
metrics collection, and infrastructure management.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.mcp_tools.tools import (
    configuration,
    cost_estimation,
    embeddings,
    system_health,
)
from src.services.embeddings.manager import EmbeddingManager
from src.services.observability.health_manager import HealthCheckManager
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


class SystemService:
    """FastMCP 2.0+ system service for system operations.

    Provides system monitoring, configuration management, and resource tracking.
    """

    def __init__(
        self,
        name: str = "system-service",
        *,
        vector_service: VectorStoreService | None = None,
        embedding_manager: EmbeddingManager | None = None,
        health_manager: HealthCheckManager | None = None,
    ):
        """Initialize the system service.

        Args:
            name: Service name for MCP registration
        """

        self.mcp = FastMCP(
            name,
            instructions="""
            System service for monitoring and configuration management.

            Provides tools for:
            - System health monitoring and metrics collection
            - Resource usage tracking and management
            - Configuration management and validation
            - Cost estimation for operations
            - Embedding provider management
            """,
        )
        system_health.register_tools(self.mcp, health_manager=health_manager)
        configuration.register_tools(self.mcp)
        cost_estimation.register_tools(self.mcp)
        embeddings.register_tools(self.mcp, embedding_manager=embedding_manager)
        logger.info("System tools registered")

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
                "embedding_management",
            ],
            "status": "active",
        }
