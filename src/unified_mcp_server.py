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
from src.infrastructure.bootstrap import container_session
from src.mcp_tools.tool_registry import register_all_tools
from src.services.logging_config import configure_logging
from src.services.observability.health_manager import (
    HealthCheckConfig,
    HealthCheckManager,
    build_health_manager,
)


logger = logging.getLogger(__name__)
_MONITORING_STATE: dict[int, HealthCheckManager] = {}


def initialize_monitoring_system(
    settings: Any,
    qdrant_client: Any,
    redis_url: str | None,
) -> HealthCheckManager | None:
    """Set up the health monitoring system using application configuration."""

    monitoring_config = getattr(settings, "monitoring", None)
    monitoring_enabled = bool(getattr(monitoring_config, "enabled", True))

    if not monitoring_enabled:
        return None

    base_config = HealthCheckConfig.from_unified_config(settings)
    health_checks_enabled = bool(
        getattr(monitoring_config, "enable_health_checks", False)
    )

    if not health_checks_enabled:
        return HealthCheckManager(base_config)

    return build_health_manager(
        settings,
        qdrant_client=qdrant_client,
        redis_url=redis_url,
    )


def setup_fastmcp_monitoring(
    server: FastMCP[Any],
    config: Any,
    health_manager: HealthCheckManager,
) -> None:
    """Attach monitoring metadata to the FastMCP server."""

    if not getattr(config.monitoring, "include_system_metrics", False):
        return
    _MONITORING_STATE[id(server)] = health_manager
    logger.info("Registered MCP health manager; monitoring enabled.")


async def run_periodic_health_checks(
    health_manager: HealthCheckManager,
    *,
    interval_seconds: float,
) -> None:
    """Periodically execute all registered health checks."""

    while True:
        await health_manager.check_all()
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def managed_lifespan(server: FastMCP[Any]) -> AsyncIterator[None]:  # pylint: disable=too-many-locals
    """Server lifecycle management with lazy initialization."""

    config = get_settings()
    configure_logging(settings=config)
    monitoring_tasks: list[asyncio.Task[Any]] = []
    validate_configuration()
    logger.info("Initializing AI Documentation Vector DB MCP Server...")

    async with container_session(settings=config, force_reload=True) as container:
        try:
            logger.info("Initializing monitoring system...")
            qdrant_client = container.qdrant_client()

            cache_config = getattr(config, "cache", None)
            enable_dragonfly = bool(
                getattr(cache_config, "enable_dragonfly_cache", False)
            )
            dragonfly_url_value = getattr(cache_config, "dragonfly_url", None)
            redis_url = (
                str(dragonfly_url_value)
                if enable_dragonfly and dragonfly_url_value
                else None
            )

            health_manager = initialize_monitoring_system(
                config, qdrant_client, redis_url
            )

            monitoring_config = getattr(config, "monitoring", None)
            health_checks_requested = bool(
                getattr(monitoring_config, "enable_health_checks", False)
            )

            if health_manager is None:
                logger.info(
                    "Monitoring disabled via configuration; skipping "
                    "health manager registration"
                )
            else:
                manager_config = getattr(health_manager, "config", None)
                health_checks_enabled = bool(getattr(manager_config, "enabled", False))

                if health_checks_requested and health_checks_enabled:
                    setup_fastmcp_monitoring(server, config, health_manager)

                    interval = getattr(
                        monitoring_config, "system_metrics_interval", 60
                    )
                    health_check_task = asyncio.create_task(
                        run_periodic_health_checks(
                            health_manager, interval_seconds=float(interval)
                        )
                    )
                    monitoring_tasks.append(health_check_task)
                    logger.info("Started background health monitoring task")
                else:
                    logger.info(
                        "Health checks disabled; skipping endpoint registration "
                        "and background task"
                    )

            logger.info("Registering MCP tools...")
            vector_service = container.vector_store_service()
            cache_manager = container.cache_manager()
            crawl_manager = container.browser_manager()
            content_service = container.content_intelligence_service()
            project_storage = container.project_storage()
            embedding_manager = container.embedding_manager()

            await register_all_tools(
                server,
                vector_service=vector_service,
                cache_manager=cache_manager,
                crawl_manager=crawl_manager,
                content_intelligence_service=content_service,
                project_storage=project_storage,
                embedding_manager=embedding_manager,
                health_manager=health_manager,
            )

            logger.info("Server initialization complete")
            yield
        finally:
            logger.info("Shutting down server...")

            for task in monitoring_tasks:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

            removed = _MONITORING_STATE.pop(id(server), None)
            if removed is not None:
                logger.info("Monitoring state cleared for MCP server.")
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
