"""Search Service — unified retrieval + web search."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import retrieval, web_search


logger = logging.getLogger(__name__)


class SearchService:
    """FastMCP service for retrieval operations."""

    def __init__(self, name: str = "search-service"):
        self.mcp = FastMCP(
            name,
            instructions=(
                "Unified retrieval service. Exposes vector/hybrid search, filtering, "
                "multi-stage, recommendations, reranking, and web search."
            ),
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        self.client_manager = client_manager
        await self._register_tools()
        logger.info("SearchService initialized")

    async def _register_tools(self) -> None:
        if not self.client_manager:
            raise RuntimeError("SearchService not initialized")
        retrieval.register_tools(self.mcp, self.client_manager)
        web_search.register_tools(self.mcp, self.client_manager)
        logger.info("Registered retrieval + web_search tools")

    def get_mcp_server(self) -> FastMCP:
        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
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
