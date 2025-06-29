"""Analytics Service - Domain-specific MCP server for analytics and monitoring.

This service handles all analytics and monitoring functionality including
performance metrics, observability, and agentic decision tracking based on
J1 research findings.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    agentic_rag,
    analytics,
    query_processing,
)


logger = logging.getLogger(__name__)


class AnalyticsService:
    """FastMCP 2.0+ analytics service with agentic observability.

    Implements enterprise agentic observability with agent decision metrics
    and multi-agent workflow visualization based on J1 research findings.
    """

    def __init__(self, name: str = "analytics-service"):
        """Initialize the analytics service.

        Args:
            name: Service name for MCP registration
        """
        self.mcp = FastMCP(
            name,
            instructions="""
            Advanced analytics service with agentic observability capabilities.

            Features:
            - Agent decision metrics with confidence and quality tracking
            - Multi-agent workflow visualization and dependency mapping
            - Auto-RAG performance monitoring with convergence analysis
            - Self-healing capabilities with predictive failure detection
            - Cost estimation and optimization analytics
            - Performance benchmarking and trend analysis

            Autonomous Capabilities:
            - Predictive failure detection and autonomous remediation
            - Self-learning performance optimization patterns
            - Intelligent cost and resource optimization
            - Agent decision quality correlation and improvement
            """,
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialize the analytics service with client manager.

        Args:
            client_manager: Shared client manager instance
        """
        self.client_manager = client_manager

        # Register analytics tools
        await self._register_analytics_tools()

        logger.info("AnalyticsService initialized with agentic observability")

    async def _register_analytics_tools(self) -> None:
        """Register all analytics-related MCP tools."""
        if not self.client_manager:
            msg = "AnalyticsService not initialized"
            raise RuntimeError(msg)

        # Register core analytics tools
        analytics.register_tools(self.mcp, self.client_manager)
        query_processing.register_tools(self.mcp, self.client_manager)

        # Register agentic RAG analytics tools (J1 research)
        agentic_rag.register_tools(self.mcp, self.client_manager)

        logger.info(
            "Registered analytics tools with agentic observability capabilities"
        )

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
            "service": "analytics",
            "version": "2.0",
            "capabilities": [
                "agent_decision_metrics",
                "workflow_visualization",
                "auto_rag_monitoring",
                "performance_analytics",
                "cost_optimization",
                "predictive_monitoring",
                "agentic_observability",
            ],
            "autonomous_features": [
                "failure_prediction",
                "performance_optimization",
                "cost_intelligence",
                "decision_quality_tracking",
            ],
            "status": "active",
            "research_basis": "J1_ENTERPRISE_AGENTIC_OBSERVABILITY",
        }
