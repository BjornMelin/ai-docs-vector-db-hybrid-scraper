"""Document Service - Domain-specific MCP server for document operations.

This service handles all document-related functionality including document
management, processing, and 5-tier crawling based on I3 research findings.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tools import (
    collections,
    content_intelligence,
    crawling,
    document_management,
    documents,
    lightweight_scrape,
    projects,
)


logger = logging.getLogger(__name__)


class DocumentService:
    """FastMCP 2.0+ document service with intelligent processing capabilities.

    Implements autonomous document management with 5-tier crawling enhancement
    and intelligent content processing based on I3 research findings.
    """

    def __init__(self, name: str = "document-service"):
        """Initialize the document service.

        Args:
            name: Service name for MCP registration
        """
        self.mcp = FastMCP(
            name,
            instructions="""
            Advanced document service with intelligent processing capabilities.

            Features:
            - 5-tier intelligent crawling with ML-powered tier selection
            - Autonomous document processing and content extraction
            - Project-based document organization and management
            - Collection management with agentic optimization
            - Content intelligence and quality assessment
            - Advanced chunking strategies with AST-based processing

            Autonomous Capabilities:
            - Intelligent tier selection for crawling optimization
            - Dynamic content quality assessment and filtering
            - Self-learning document processing patterns
            - Autonomous collection provisioning and management
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

        logger.info("DocumentService initialized with 5-tier crawling capabilities")

    async def _register_document_tools(self) -> None:
        """Register all document-related MCP tools."""
        if not self.client_manager:
            msg = "DocumentService not initialized"
            raise RuntimeError(msg)

        # Register core document tools
        document_management.register_tools(self.mcp, self.client_manager)
        collections.register_tools(self.mcp, self.client_manager)
        projects.register_tools(self.mcp, self.client_manager)

        # Register intelligent crawling tools (I3 research)
        crawling.register_tools(self.mcp, self.client_manager)

        # Register content intelligence tools
        content_intelligence.register_tools(self.mcp, self.client_manager)

        logger.info("Registered document tools with intelligent crawling capabilities")

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
                "intelligent_crawling",
                "5_tier_crawling",
                "collection_management",
                "project_organization",
                "content_intelligence",
                "autonomous_processing",
            ],
            "autonomous_features": [
                "tier_selection_optimization",
                "content_quality_assessment",
                "processing_pattern_learning",
                "collection_provisioning",
            ],
            "status": "active",
            "research_basis": "I3_5_TIER_CRAWLING_ENHANCEMENT",
        }
