#!/usr/bin/env python3
"""Unified MCP Server for AI Docs Vector DB.

This is the main entry point for the MCP server. It follows FastMCP 2.0
best practices with lazy initialization and modular tool registration.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal, cast

from fastmcp import FastMCP

from src.config.loader import get_settings
from src.config.models import CrawlProvider, EmbeddingProvider
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tool_registry import register_all_tools
from src.services.dependencies import get_cache_manager
from src.services.logging_config import configure_logging
from src.services.monitoring.initialization import (
    initialize_monitoring_system,
    run_periodic_health_checks,
    setup_fastmcp_monitoring,
    update_cache_metrics_periodically,
    update_system_metrics_periodically,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def managed_lifespan(server: FastMCP[Any]) -> AsyncIterator[None]:  # pylint: disable=too-many-locals
    """Server lifecycle management with lazy initialization."""

    config = get_settings()
    configure_logging(settings=config)
    monitoring_tasks: list[asyncio.Task[Any]] = []
    client_manager: ClientManager | None = None
    metrics_registry = None
    health_manager = None
    try:
        validate_configuration()

        logger.info("Initializing AI Documentation Vector DB MCP Server...")

        client_manager = ClientManager()
        await client_manager.initialize()

        logger.info("Initializing monitoring system...")
        qdrant_client = getattr(client_manager, "qdrant_client", None)

        cache_config = getattr(config, "cache", None)
        enable_dragonfly = bool(getattr(cache_config, "enable_dragonfly_cache", False))
        dragonfly_url_value = getattr(cache_config, "dragonfly_url", None)
        redis_url = (
            str(dragonfly_url_value)
            if enable_dragonfly and dragonfly_url_value
            else None
        )

        metrics_registry, health_manager = initialize_monitoring_system(
            config, qdrant_client, redis_url
        )

        if metrics_registry and health_manager:
            setup_fastmcp_monitoring(server, config, metrics_registry, health_manager)

            if config.monitoring.enabled:
                health_check_task = asyncio.create_task(
                    run_periodic_health_checks(health_manager, interval_seconds=60.0)
                )
                monitoring_tasks.append(health_check_task)

                if config.monitoring.include_system_metrics:
                    metrics_task = asyncio.create_task(
                        update_system_metrics_periodically(
                            metrics_registry,
                            interval_seconds=config.monitoring.system_metrics_interval,
                        )
                    )
                    monitoring_tasks.append(metrics_task)

                cache_config = config.cache
                if metrics_registry and (
                    getattr(cache_config, "enable_local_cache", False)
                    or getattr(cache_config, "enable_dragonfly_cache", False)
                ):
                    cache_manager = await get_cache_manager(client_manager)
                    cache_metrics_task = asyncio.create_task(
                        update_cache_metrics_periodically(
                            metrics_registry,
                            cache_manager,
                            interval_seconds=30.0,
                        )
                    )
                    monitoring_tasks.append(cache_metrics_task)

                logger.info("Started background monitoring tasks")

        logger.info("Registering MCP tools...")
        await register_all_tools(server, client_manager)

        logger.info("Server initialization complete")
        yield

    finally:
        logger.info("Shutting down server...")

        for task in monitoring_tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        if client_manager:
            await client_manager.cleanup()

        logger.info("Server shutdown complete")


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
    lifespan=managed_lifespan,
)

TransportName = Literal["stdio", "http", "sse", "streamable-http"]

DEFAULT_TRANSPORT: TransportName = "streamable-http"
SUPPORTED_TRANSPORTS: frozenset[TransportName] = frozenset(
    {"stdio", "http", "sse", "streamable-http"}
)


def _normalize_transport(value: str) -> TransportName:
    """Normalize FASTMCP transport value to a supported literal."""
    if value not in SUPPORTED_TRANSPORTS:
        logger.warning(
            "Unsupported FASTMCP_TRANSPORT '%s'; defaulting to 'stdio'", value
        )
        return "stdio"
    return cast(TransportName, value)


def _get_int_env(var_name: str, default: int) -> int:
    """Fetch an integer environment variable with safe fallback."""
    env_value = os.getenv(var_name)
    if env_value is None:
        return default
    try:
        return int(env_value)
    except ValueError:
        logger.warning(
            "Invalid integer value '%s' for %s; using default %s",
            env_value,
            var_name,
            default,
        )
        return default


def _validate_streaming_config(errors: list, warnings: list) -> None:
    """Validate streaming configuration parameters."""
    transport = os.getenv("FASTMCP_TRANSPORT", DEFAULT_TRANSPORT)
    if transport != DEFAULT_TRANSPORT:
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
            f"Invalid buffer size: {os.getenv('FASTMCP_BUFFER_SIZE')}. "
            "Must be a valid integer"
        )

    # Validate max response size
    try:
        max_response_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760"))
        if max_response_size <= 0:
            errors.append("Max response size must be positive")
    except ValueError:
        errors.append(
            f"Invalid max response size: {os.getenv('FASTMCP_MAX_RESPONSE_SIZE')}. "
            "Must be a valid integer"
        )


def validate_configuration() -> None:
    """Validate configuration at startup."""
    config = get_settings()
    warnings: list[str] = []
    errors: list[str] = []

    embedding_provider = getattr(config.embedding, "provider", None)
    if embedding_provider == EmbeddingProvider.OPENAI and not config.openai.api_key:
        errors.append(
            "OpenAI API key is required when the OpenAI embedding provider is enabled"
        )

    if (
        config.crawl_provider == CrawlProvider.FIRECRAWL
        and not config.firecrawl.api_key
    ):
        warnings.append(
            "Firecrawl API key not set - premium crawling features will be unavailable"
        )

    if not config.qdrant.url:
        errors.append("Qdrant URL is required")

    _validate_streaming_config(errors, warnings)

    for warning in warnings:
        logger.warning("Configuration warning: %s", warning)

    if errors:
        for error in errors:
            logger.error("Configuration error: %s", error)
        msg = f"Configuration validation failed: {'; '.join(errors)}"
        raise ValueError(msg)

    logger.info("Configuration validation passed")


if __name__ == "__main__":
    configure_logging()
    transport = _normalize_transport(os.getenv("FASTMCP_TRANSPORT", DEFAULT_TRANSPORT))

    logger.info("Starting MCP server with transport: %s", transport)

    if transport == "stdio":
        logger.info("Using stdio transport for Claude Desktop compatibility")
        mcp.run(transport=transport)
    else:
        host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        port = _get_int_env("FASTMCP_PORT", 8000)
        logger.info("Starting %s server on %s:%s", transport, host, port)
        if transport == DEFAULT_TRANSPORT:
            logger.info("Enhanced streaming support enabled for large search results")
        mcp.run(transport=transport, host=host, port=port)
