"""Search Service - Domain-specific MCP server for search operations.

This service handles all search-related functionality including hybrid search,
vector search, and web search orchestration based on I5 research findings.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    hybrid_search,
    hyde_search,
    multi_stage_search,
    query_processing,
    rag,
    search,
    search_tools,
    search_with_reranking,
    utilities,
    web_search,
)


logger = logging.getLogger(__name__)


class SearchService:
    """FastMCP 2.0+ search service with autonomous capabilities.

    Implements intelligent search orchestration with multi-provider support
    and autonomous web search agents based on I5 research findings.
    """

    def __init__(self, name: str = "search-service"):
        """Initialize the search service.

        Args:
            name: Service name for MCP registration
        """
        self.mcp = FastMCP(
            name,
            instructions="""
            Advanced search service with intelligent orchestration capabilities.

            Features:
            - Hybrid vector + text search with DBSF score fusion
            - HyDE (Hypothetical Document Embeddings) search
            - Multi-stage search with progressive refinement
            - Autonomous web search orchestration (I5 research)
            - Search result reranking and quality assessment
            - Real-time search strategy optimization

            Autonomous Capabilities:
            - Intelligent search provider selection
            - Dynamic strategy adaptation based on query type
            - Self-learning search pattern optimization
            - Multi-provider result fusion and synthesis
            """,
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialize the search service with client manager.

        Args:
            client_manager: Shared client manager instance
        """
        self.client_manager = client_manager

        # Register search tools
        await self._register_search_tools()

        logger.info("SearchService initialized with autonomous capabilities")

    async def _register_search_tools(self) -> None:
        """Register all search-related MCP tools."""
        if not self.client_manager:
            msg = "SearchService not initialized"
            raise RuntimeError(msg)

        # Register core search tools
        hybrid_search.register_tools(self.mcp, self.client_manager)
        hyde_search.register_tools(self.mcp, self.client_manager)
        multi_stage_search.register_tools(self.mcp, self.client_manager)
        search_with_reranking.register_tools(self.mcp, self.client_manager)

        # Register autonomous web search tools (I5 research)
        web_search.register_tools(self.mcp, self.client_manager)

        logger.info("Registered search tools with autonomous web search capabilities")

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
            "service": "search",
            "version": "2.0",
            "capabilities": [
                "hybrid_search",
                "hyde_search",
                "multi_stage_search",
                "search_reranking",
                "autonomous_web_search",
                "multi_provider_orchestration",
                "intelligent_result_fusion",
            ],
            "autonomous_features": [
                "provider_optimization",
                "strategy_adaptation",
                "quality_assessment",
                "self_learning_patterns",
            ],
            "status": "active",
            "research_basis": "I5_WEB_SEARCH_TOOL_ORCHESTRATION",
        }
