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

from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tool_registry import register_all_tools
from src.services.logging_config import configure_logging

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastMCP server with streaming support
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
    - Batch processing and enhanced streaming support
    - Cost estimation and optimization
    - Analytics and monitoring

    Streaming Support:
    - Uses streamable-http transport by default for optimal performance
    - Supports large search results with configurable response buffers
    - Environment variables: FASTMCP_TRANSPORT, FASTMCP_HOST, FASTMCP_PORT
    - Automatic fallback to stdio for Claude Desktop compatibility
    """,
)


def _validate_streaming_config(errors: list, warnings: list) -> None:
    """Validate streaming configuration parameters."""
    transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
    if transport != "streamable-http":
        return

    # Validate port
    try:
        port = int(os.getenv("FASTMCP_PORT", "8000"))
        if port <= 0 or port > 65535:
            errors.append(f"Invalid port number: {port}. Must be between 1 and 65535")
    except ValueError:
        errors.append(
            f"Invalid port value: {os.getenv('FASTMCP_PORT')}. Must be a valid integer"
        )

    # Validate buffer size
    try:
        buffer_size = int(os.getenv("FASTMCP_BUFFER_SIZE", "8192"))
        if buffer_size <= 0:
            warnings.append(
                f"Buffer size {buffer_size} is very small and may impact performance"
            )
    except ValueError:
        errors.append(
            f"Invalid buffer size: {os.getenv('FASTMCP_BUFFER_SIZE')}. Must be a valid integer"
        )

    # Validate max response size
    try:
        max_response_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760"))
        if max_response_size <= 0:
            errors.append("Max response size must be positive")
    except ValueError:
        errors.append(
            f"Invalid max response size: {os.getenv('FASTMCP_MAX_RESPONSE_SIZE')}. Must be a valid integer"
        )


def validate_configuration():
    """Validate configuration at startup.

    Checks for required API keys and validates critical settings.
    """
    from src.config import get_config

    config = get_config()
    warnings = []
    errors = []

    # Check API keys based on enabled providers
    if "openai" in config.get_active_providers() and not config.openai.api_key:
        errors.append("OpenAI API key is required when OpenAI provider is enabled")

    if "firecrawl" in config.crawling.providers and not config.firecrawl.api_key:
        warnings.append(
            "Firecrawl API key not set - premium features will be unavailable"
        )

    # Check Qdrant connection
    if not config.qdrant.url:
        errors.append("Qdrant URL is required")

    # Validate streaming configuration
    _validate_streaming_config(errors, warnings)

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
    try:
        # Validate configuration first
        validate_configuration()

        # Initialize client manager with unified config
        logger.info("Initializing AI Documentation Vector DB MCP Server...")
        from src.config import get_config

        config = get_config()
        lifespan.client_manager = ClientManager(config)
        await lifespan.client_manager.initialize()

        # Register all tools
        logger.info("Registering MCP tools...")
        await register_all_tools(mcp, lifespan.client_manager)

        logger.info("Server initialization complete")
        yield

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down server...")
        if hasattr(lifespan, "client_manager"):
            await lifespan.client_manager.cleanup()
        logger.info("Server shutdown complete")


# Set the lifespan
mcp.lifespan = lifespan


if __name__ == "__main__":
    # Run the server with enhanced streaming support
    # Default to streamable-http for better performance and streaming capabilities
    transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")

    logger.info(f"Starting MCP server with transport: {transport}")

    if transport == "streamable-http":
        # Enhanced streaming configuration
        host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        port = int(os.getenv("FASTMCP_PORT", "8000"))

        logger.info(f"Starting streamable HTTP server on {host}:{port}")
        logger.info("Enhanced streaming support enabled for large search results")

        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
        )
    elif transport == "stdio":
        # Fallback to stdio for Claude Desktop compatibility
        logger.info("Using stdio transport for Claude Desktop compatibility")
        mcp.run(transport="stdio")
    else:
        # Support for other transport types
        logger.info(f"Using {transport} transport")
        mcp.run(transport=transport)
