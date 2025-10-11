"""Document Service — ingestion and management."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.mcp_tools.tools import (
    collection_management,
    content_intelligence,
    crawling,
    documents,
    projects,
)
from src.services.cache.manager import CacheManager
from src.services.core.project_storage import ProjectStorage
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


class DocumentService:
    """Document ingestion and management."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str = "document-service",
        *,
        vector_service: VectorStoreService,
        cache_manager: CacheManager,
        crawl_manager: Any,
        content_intelligence_service: Any,
        project_storage: ProjectStorage,
    ) -> None:
        self.mcp = FastMCP(
            name,
            instructions=(
                "Document ingestion and management. Provides crawling, analysis, "
                "collections, and project tools."
            ),
        )

        documents.register_tools(
            self.mcp,
            vector_service=vector_service,
            cache_manager=cache_manager,
            crawl_manager=crawl_manager,
            content_intelligence_service=content_intelligence_service,
        )
        collection_management.register_tools(
            self.mcp,
            vector_service=vector_service,
            cache_manager=cache_manager,
        )
        projects.register_tools(
            self.mcp,
            vector_service=vector_service,
            project_storage=project_storage,
        )
        crawling.register_tools(self.mcp, crawl_manager=crawl_manager)
        content_intelligence.register_tools(
            self.mcp,
            content_service=content_intelligence_service,
        )
        logger.info("Document tools registered")

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
