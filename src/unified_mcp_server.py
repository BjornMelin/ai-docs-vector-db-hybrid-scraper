#!/usr/bin/env python3
"""Unified MCP Server for AI Documentation Vector DB.

This is the main entry point for the MCP server. It follows FastMCP 2.0
best practices with lazy initialization and modular tool registration.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Setup import paths
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

from infrastructure.client_manager import ClientManager
from mcp.tool_registry import register_all_tools
from services.logging_config import configure_logging

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    "ai-docs-vector-db-unified",
    instructions="""
    This server provides advanced vector database functionality for AI documentation.

    Features:
    - Hybrid search with dense+sparse vectors and reranking
    - Multi-provider embedding generation
    - Advanced chunking strategies (Basic, Enhanced, AST-based)
    - Project-based document management
    - Two-tier caching with metrics
    - Batch processing and streaming support
    - Cost estimation and optimization
    - Analytics and monitoring
    """,
)

# Global client manager (initialized lazily)
client_manager = None


def validate_configuration():
    """Validate configuration at startup.

    Checks for required API keys and validates critical settings.
    """
    from config import get_config

    config = get_config()
    warnings = []
    errors = []

    # Check API keys based on enabled providers
    if "openai" in config.get_active_providers():
        if not config.openai.api_key:
            errors.append("OpenAI API key is required when OpenAI provider is enabled")

    if "firecrawl" in config.crawling.providers:
        if not config.firecrawl.api_key:
            warnings.append(
                "Firecrawl API key not set - premium features will be unavailable"
            )

    # Check Qdrant connection
    if not config.qdrant.url:
        errors.append("Qdrant URL is required")

    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    # Raise on errors
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    logger.info("Configuration validation passed")


@asynccontextmanager
async def lifespan():
    """Server lifecycle management with lazy initialization."""
    global client_manager

    try:
        # Validate configuration first
        validate_configuration()

        # Initialize client manager
        logger.info("Initializing AI Documentation Vector DB MCP Server...")
        client_manager = ClientManager.from_unified_config()
        await client_manager.initialize()

        # Register all tools
        logger.info("Registering MCP tools...")
        await register_all_tools(mcp, client_manager)

        logger.info("Server initialization complete")
        yield

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down server...")
        if client_manager:
            await client_manager.cleanup()
        logger.info("Server shutdown complete")


# Set the lifespan
mcp.lifespan = lifespan


if __name__ == "__main__":
    # Run the server
    # Transport can be overridden via environment variables or CLI
    transport = os.getenv("FASTMCP_TRANSPORT", "stdio")

    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("FASTMCP_HOST", "127.0.0.1"),
            port=int(os.getenv("FASTMCP_PORT", "8000")),
        )
    else:
        # Default to stdio for Claude Desktop compatibility
        mcp.run(transport="stdio")
