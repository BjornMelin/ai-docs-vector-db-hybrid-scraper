"""Search Service - Domain-specific MCP server for search operations.

This service handles all search-related functionality including hybrid search,
vector search, and web search orchestration based on I5 research findings.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    filtering_tools,
    query_processing_tools,
    search,
    search_tools,
    web_search,
)


logger = logging.getLogger(__name__)


class SearchService:
    """FastMCP 2.0+ search service for search operations.

    Provides search functionality with multi-provider support.
    """

    def __init__(self, name: str = "search-service"):
        """Initialize the search service.

        Args:
            name: Service name for MCP registration

        """
        self.mcp = FastMCP(
            name,
            instructions="""
            Search service for vector and text search operations.

            Provides tools for:
            - Vector similarity search
            - Hybrid vector + text search
            - Multi-stage search with refinement
            - Search result filtering and reranking
            - Web search integration
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

        logger.info("SearchService initialized")

    async def _register_search_tools(self) -> None:
        """Register all search-related MCP tools."""
        if not self.client_manager:
            msg = "SearchService not initialized"
            raise RuntimeError(msg)

        # Register core search tools (basic Qdrant operations)
        search.register_tools(self.mcp, self.client_manager)

        # Register advanced search tools (HyDE, A/B testing, multi-stage)
        search_tools.register_tools(self.mcp, self.client_manager)

        # Register query processing and orchestration tools
        query_processing_tools.register_query_processing_tools(
            self.mcp, self.client_manager
        )

        # Register advanced filtering tools
        filtering_tools.register_filtering_tools(self.mcp, self.client_manager)

        # Register web search tools
        web_search.register_tools(self.mcp, self.client_manager)

        logger.info("Registered search tools")

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
                # Core Qdrant search operations
                "vector_search",
                "hybrid_search",
                "scroll_collection",
                "recommend_similar",
                "filtered_search",
                # Advanced search tools
                "hyde_search",
                "multi_stage_search",
                "search_reranking",
                "ab_test_search",
                # Query processing
                "query_expansion",
                "clustered_search",
                "federated_search",
                "personalized_search",
                "orchestrated_search",
                # Advanced filtering
                "temporal_filter",
                "content_type_filter",
                "metadata_filter",
                "similarity_filter",
                "composite_filter",
                # Web search
                "web_search",
                "multi_provider_search",
            ],
            "status": "active",
        }
