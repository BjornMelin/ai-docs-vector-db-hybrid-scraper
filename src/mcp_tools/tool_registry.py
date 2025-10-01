"""Dynamic tool registration system for FastMCP server.

This module provides a clean, modular way to register all MCP tools
with the server, following FastMCP 2.0 best practices.
"""

import logging

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager

from . import tools


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
    tools.collection_management.register_tools(mcp, client_manager)
    registered_tools.append("collection_management")

    tools.projects.register_tools(mcp, client_manager)
    registered_tools.append("projects")

    # Additional features
    logger.info("Registering additional tools...")
    tools.search_tools.register_tools(mcp, client_manager)
    registered_tools.append("search_tools")

    tools.query_processing.register_tools(mcp, client_manager)
    registered_tools.append("query_processing")

    # Additional filtering and query processing capabilities
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

    # RAG (Pydantic-AI based agents)
    logger.info("Registering RAG tools...")
    try:
        tools.agentic_rag.register_tools(mcp, client_manager)
        registered_tools.append("agentic_rag")
    except ImportError as e:
        logger.warning("RAG tools not available (missing dependencies): %s", e)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to register RAG tools")

    tool_list = ", ".join(registered_tools)
    logger.info(
        "Successfully registered %d tool modules: %s", len(registered_tools), tool_list
    )
