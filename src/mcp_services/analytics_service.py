"""Analytics Service — metrics and observability tools."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import analytics


logger = logging.getLogger(__name__)


class AnalyticsService:
    """Monitoring and metrics with observability integration."""

    def __init__(self, name: str = "analytics-service"):
        self.mcp = FastMCP(
            name,
            instructions=(
                "Analytics for collections and performance. Integrates with "
                "observability backends."
            ),
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        self.client_manager = client_manager
        await self._register_tools()
        logger.info("AnalyticsService initialized")

    async def _register_tools(self) -> None:
        if not self.client_manager:
            raise RuntimeError("AnalyticsService not initialized")
        analytics.register_tools(self.mcp, self.client_manager)
        logger.info("Registered analytics tools")

    def get_mcp_server(self) -> FastMCP:
        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
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
