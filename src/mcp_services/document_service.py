"""Document Service - Domain-specific MCP server for document operations.

This service handles document-related functionality including document
management, processing, and crawling.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    collection_management,
    content_intelligence,
    crawling,
    document_management,
    projects,
)


logger = logging.getLogger(__name__)


class DocumentService:
    """FastMCP 2.0+ document service for document processing.

    Provides document management, processing, and 5-tier crawling capabilities.
    """

    def __init__(self, name: str = "document-service"):
        """Initialize the document service.

        Args:
            name: Service name for MCP registration
        """

        self.mcp = FastMCP(
            name,
            instructions="""
            Document service for document processing and management.

            Provides tools for:
            - Document management and organization
            - Web crawling and content extraction
            - Project-based document organization
            - Collection management
            - Content processing and analysis
            """,
        )
        self.client_manager: ClientManager | None = None

    async def initialize(self, client_manager: ClientManager) -> None:
        """Initialize the document service with client manager.

        Args:
            client_manager: Shared client manager instance
        """

        self.client_manager = client_manager

        # Register document tools
        await self._register_document_tools()

        logger.info("DocumentService initialized")

    async def _register_document_tools(self) -> None:
        """Register all document-related MCP tools."""

        if not self.client_manager:
            msg = "DocumentService not initialized"
            raise RuntimeError(msg)

        # Register core document tools
        document_management.register_tools(self.mcp, self.client_manager)
        collection_management.register_tools(self.mcp, self.client_manager)
        projects.register_tools(self.mcp, self.client_manager)

        # Register crawling tools
        crawling.register_tools(self.mcp, self.client_manager)

        # Register content processing tools
        content_intelligence.register_tools(self.mcp, self.client_manager)

        logger.info("Registered document tools")

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
            "service": "document",
            "version": "2.0",
            "capabilities": [
                "document_management",
                "web_crawling",
                "collection_management",
                "project_organization",
                "content_processing",
            ],
            "status": "active",
        }
