"""Search Service — unified retrieval + web search."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.mcp_tools.tools import retrieval, web_search
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


class SearchService:
    """FastMCP service for retrieval operations."""

    def __init__(
        self,
        name: str = "search-service",
        *,
        vector_service: VectorStoreService,
    ):
        self.mcp = FastMCP(
            name,
            instructions=(
                "Unified retrieval service. Exposes vector/hybrid search, filtering, "
                "multi-stage, recommendations, reranking, and web search."
            ),
        )
        retrieval.register_tools(self.mcp, vector_service=vector_service)
        web_search.register_tools(self.mcp)
        logger.info("Search tools registered")

    def get_mcp_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
        """Return service metadata and capabilities."""
        return {
            "service": "search",
            "version": "3.0",
            "capabilities": [
                "vector_search",
                "filtered_search",
                "multi_stage_search",
                "search_with_context",
                "recommend_similar",
                "reranked_search",
                "scroll_collection",
                "web_search",
            ],
            "status": "active",
        }
