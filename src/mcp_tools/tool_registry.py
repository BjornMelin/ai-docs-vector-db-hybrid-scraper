"""Dynamic tool registration system for FastMCP server.

This module provides a clean, modular way to register all MCP tools
with the server, following FastMCP 2.0 best practices.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


async def register_all_tools(mcp: "FastMCP", client_manager: "ClientManager") -> None:
    """Register all tool modules with the MCP server.

    This function dynamically loads and registers all tool modules,
    allowing for easy addition/removal of functionality.

    Args:
        mcp: The FastMCP server instance
        client_manager: The unified client manager for all services
    """
    from . import tools

    # Track registration for logging
    registered_tools = []

    # Core functionality
    logger.info("Registering core tools...")
    tools.search.register_tools(mcp, client_manager)
    registered_tools.append("search")

    tools.documents.register_tools(mcp, client_manager)
    registered_tools.append("documents")

    tools.embeddings.register_tools(mcp, client_manager)
    registered_tools.append("embeddings")

    tools.lightweight_scrape.register_tools(mcp, client_manager)
    registered_tools.append("lightweight_scrape")

    # Collection and project management
    logger.info("Registering management tools...")
    tools.collections.register_tools(mcp, client_manager)
    registered_tools.append("collections")

    tools.projects.register_tools(mcp, client_manager)
    registered_tools.append("projects")

    # Advanced features
    logger.info("Registering advanced tools...")
    tools.advanced_search.register_tools(mcp, client_manager)
    registered_tools.append("advanced_search")

    tools.query_processing.register_tools(mcp, client_manager)
    registered_tools.append("query_processing")

    tools.payload_indexing.register_tools(mcp, client_manager)
    registered_tools.append("payload_indexing")

    tools.deployment.register_tools(mcp, client_manager)
    registered_tools.append("deployment")

    # Utilities and monitoring
    logger.info("Registering utility tools...")
    tools.analytics.register_tools(mcp, client_manager)
    registered_tools.append("analytics")

    tools.cache.register_tools(mcp, client_manager)
    registered_tools.append("cache")

    tools.utilities.register_tools(mcp, client_manager)
    registered_tools.append("utilities")

    # Content Intelligence
    logger.info("Registering content intelligence tools...")
    tools.content_intelligence.register_tools(mcp, client_manager)
    registered_tools.append("content_intelligence")

    logger.info(
        f"Successfully registered {len(registered_tools)} tool modules: {', '.join(registered_tools)}"
    )
