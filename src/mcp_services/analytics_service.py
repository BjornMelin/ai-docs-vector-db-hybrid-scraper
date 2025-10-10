"""Analytics Service — metrics and observability tools."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.mcp_tools.tools import analytics
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


class AnalyticsService:
    """Monitoring and metrics with observability integration."""

    def __init__(
        self,
        name: str = "analytics-service",
        vector_service: VectorStoreService | None = None,
    ):
        """Initialize the analytics service with MCP tools."""

        self.mcp = FastMCP(
            name,
            instructions=(
                "Analytics for collections and performance. Integrates with "
                "observability backends."
            ),
        )
        analytics.register_tools(self.mcp, vector_service=vector_service)
        logger.info("Analytics tools registered")

    def get_mcp_server(self) -> FastMCP:
        """Return the MCP server instance."""

        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
        """Return metadata about the analytics service capabilities."""

        return {
            "service": "analytics",
            "version": "3.0",
            "capabilities": [
                "collection_analytics",
                "performance_estimates",
                "cost_estimates",
            ],
            "status": "active",
        }
