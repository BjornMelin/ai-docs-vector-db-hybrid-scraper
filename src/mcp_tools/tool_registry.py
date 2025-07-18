"""Dynamic tool registration system for FastMCP server.

This module provides a clean, modular way to register all MCP tools
with the server, following FastMCP 2.0 best practices.
"""

import logging
from typing import TYPE_CHECKING

from . import tools


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
    tools.search_tools.register_tools(mcp, client_manager)
    registered_tools.append("search_tools")

    tools.query_processing.register_tools(mcp, client_manager)
    registered_tools.append("query_processing")

    # New advanced filtering and query processing capabilities
    tools.filtering_tools.register_filtering_tools(mcp, client_manager)
    registered_tools.append("filtering_tools")

    tools.query_processing_tools.register_query_processing_tools(mcp, client_manager)
    registered_tools.append("query_processing_tools")

    tools.payload_indexing.register_tools(mcp, client_manager)
    registered_tools.append("payload_indexing")

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

    # Agentic RAG (NEW - Pydantic-AI based autonomous agents)
    logger.info("Registering agentic RAG tools...")
    try:
        tools.agentic_rag.register_tools(mcp, client_manager)
        registered_tools.append("agentic_rag")
    except ImportError as e:
        logger.warning(f"Agentic RAG tools not available (missing dependencies): {e}")
    except Exception:
        logger.exception("Failed to register agentic RAG tools")

    logger.info(
        f"Successfully registered {len(registered_tools)} tool modules: {', '.join(registered_tools)}"
    )
