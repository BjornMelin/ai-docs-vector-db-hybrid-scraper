"""FastAPI dependency injection functions for service layer.

Provides centralized dependency management with resource lifecycle handling.
Uses yield dependencies for proper cleanup and circuit breaker patterns.
"""

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from src.config import Config


if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


# Configuration dependency
async def get_config() -> Config:
    """Get application configuration.

    Returns:
        Config: Unified application configuration

    """
    return Config()


# Client manager dependency with resource lifecycle
async def get_client_manager() -> AsyncGenerator["ClientManager"]:
    """Get ClientManager instance with proper lifecycle management.

    Yields:
        ClientManager: Initialized client manager

    """
    from src.infrastructure.client_manager import ClientManager  # noqa: PLC0415

    client_manager = ClientManager()
    try:
        await client_manager.initialize()
        yield client_manager
    finally:
        await client_manager.cleanup()


# Cache client dependency
async def get_cache_client(
    config: Annotated[Config, Depends(get_config)],
) -> AsyncGenerator[object]:
    """Get cache client with lifecycle management.

    Args:
        config: Application configuration

    Yields:
        CacheManager: Initialized cache manager

    """
    from src.services.cache.manager import CacheManager  # noqa: PLC0415

    cache_manager = CacheManager(
        dragonfly_url=config.cache.dragonfly_url,
        enable_local_cache=config.cache.enable_local_cache,
        enable_distributed_cache=config.cache.enable_dragonfly_cache,
        local_max_size=config.cache.local_max_size,
        local_max_memory_mb=config.cache.local_max_memory_mb,
        distributed_ttl_seconds={},  # Use defaults
        enable_metrics=True,
    )

    try:
        yield cache_manager
    finally:
        await cache_manager.close()


# Embedding client dependency
async def get_embedding_client(
    config: Annotated[Config, Depends(get_config)],
    client_manager: Annotated["ClientManager", Depends(get_client_manager)],
) -> AsyncGenerator[object]:
    """Get embedding client with lifecycle management.

    Args:
        config: Application configuration
        client_manager: Client manager for dependency injection

    Yields:
        EmbeddingManager: Initialized embedding manager

    """
    from src.services.embeddings.manager import EmbeddingManager  # noqa: PLC0415

    embedding_manager = EmbeddingManager(
        config=config,
        client_manager=client_manager,
        budget_limit=None,
        rate_limiter=None,
    )

    try:
        await embedding_manager.initialize()
        yield embedding_manager
    finally:
        await embedding_manager.cleanup()


# Vector database client dependency
async def get_vector_db_client(
    config: Annotated[Config, Depends(get_config)],
    client_manager: Annotated["ClientManager", Depends(get_client_manager)],
) -> AsyncGenerator[object]:
    """Get vector database client with lifecycle management.

    Args:
        config: Application configuration
        client_manager: Client manager for dependency injection

    Yields:
        QdrantService: Initialized vector database client

    """
    from src.services.vector_db.service import QdrantService  # noqa: PLC0415

    qdrant_manager = QdrantService(config, client_manager)

    try:
        await qdrant_manager.initialize()
        yield qdrant_manager
    finally:
        await qdrant_manager.cleanup()


# Rate limiter dependency (optional)
async def get_rate_limiter(
    _config: Annotated[Config, Depends(get_config)],
) -> object | None:
    """Get rate limiter if configured.

    Args:
        config: Application configuration

    Returns:
        RateLimitManager | None: Rate limiter or None if disabled

    """
    # Rate limiter implementation would go here
    # For now, return None to maintain compatibility
    return None


# Crawling client dependency
async def get_crawling_client(
    config: Annotated[Config, Depends(get_config)],
    rate_limiter: Annotated[object | None, Depends(get_rate_limiter)],
) -> AsyncGenerator[object]:
    """Get crawling client with lifecycle management.

    Args:
        config: Application configuration
        rate_limiter: Optional rate limiter

    Yields:
        CrawlManager: Initialized crawl manager

    """
    from src.services.crawling.manager import CrawlManager  # noqa: PLC0415

    crawl_manager = CrawlManager(
        config=config,
        rate_limiter=rate_limiter,
    )

    try:
        await crawl_manager.initialize()
        yield crawl_manager
    finally:
        await crawl_manager.cleanup()
