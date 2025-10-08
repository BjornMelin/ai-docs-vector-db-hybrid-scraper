"""Document Service — ingestion and management."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    collection_management,
    content_intelligence,
    crawling,
    documents,
    projects,
)


logger = logging.getLogger(__name__)


class DocumentService:
    """Document ingestion and management."""

    def __init__(self, name: str = "document-service"):
        self.mcp = FastMCP(
            name,
            instructions=(
                "Document ingestion and management. Provides crawling, analysis, "
                "collections, and project tools."
            ),
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        self.client_manager = client_manager
        await self._register_tools()
        logger.info("DocumentService initialized")

    async def _register_tools(self) -> None:
        if not self.client_manager:
            raise RuntimeError("DocumentService not initialized")

        documents.register_tools(self.mcp, self.client_manager)
        collection_management.register_tools(self.mcp, self.client_manager)
        projects.register_tools(self.mcp, self.client_manager)
        crawling.register_tools(self.mcp, self.client_manager)
        content_intelligence.register_tools(self.mcp, self.client_manager)

        logger.info("Registered document tools")

    def get_mcp_server(self) -> FastMCP:
        return self.mcp

    async def get_service_info(self) -> dict[str, Any]:
        return {
            "service": "document",
            "version": "3.0",
            "capabilities": [
                "ingestion",
                "web_crawling",
                "collection_management",
                "project_organization",
                "content_processing",
            ],
            "status": "active",
        }
