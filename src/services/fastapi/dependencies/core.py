"""Core dependency injection components for FastAPI production deployment.

This module provides essential dependencies for database sessions,
configuration management, and other production services.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException
from fastapi import Request
from src.config import Config
from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.cache.manager import CacheManager
from src.services.embeddings.manager import EmbeddingManager
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.vector_db.service import QdrantService
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Container for managing production dependencies and their lifecycle."""

    def __init__(self):
        """Initialize dependency container."""
        self._config: Config | None = None
        self._vector_service: QdrantService | None = None
        self._embedding_manager: EmbeddingManager | None = None
        self._cache_manager: CacheManager | None = None
        self._initialized = False

    async def initialize(self, config: Config | None = None) -> None:
        """Initialize all dependencies.

        Args:
            config: Application configuration (loads default if None)
        """
        if self._initialized:
            return

        try:
            # Load configurations
            if config is None:
                config = get_config()

            self._config = config

            # Initialize core services
            client_manager = ClientManager(config)
            self._vector_service = QdrantService(config, client_manager)
            await self._vector_service.initialize()

            self._embedding_manager = EmbeddingManager(config.embeddings)
            await self._embedding_manager.initialize()

            self._cache_manager = CacheManager(config.cache)
            await self._cache_manager.initialize()

            self._initialized = True
            logger.info("Dependency container initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize dependency container: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up all dependencies."""
        try:
            if self._cache_manager:
                await self._cache_manager.cleanup()

            if self._embedding_manager:
                await self._embedding_manager.cleanup()

            if self._vector_service:
                await self._vector_service.cleanup()

            self._initialized = False
            logger.info("Dependency container cleaned up")

        except Exception as e:
            logger.error(f"Error during dependency cleanup: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized

    @property
    def config(self) -> Config:
        """Get configuration."""
        if not self._config:
            raise RuntimeError("Dependency container not initialized")
        return self._config

    @property
    def vector_service(self) -> QdrantService:
        """Get vector database service."""
        if not self._vector_service:
            raise RuntimeError("Vector service not initialized")
        return self._vector_service

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Get embedding manager."""
        if not self._embedding_manager:
            raise RuntimeError("Embedding manager not initialized")
        return self._embedding_manager

    @property
    def cache_manager(self) -> CacheManager:
        """Get cache manager."""
        if not self._cache_manager:
            raise RuntimeError("Cache manager not initialized")
        return self._cache_manager


# Global dependency container instance
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global dependency container.

    Returns:
        Global dependency container instance

    Raises:
        RuntimeError: If container is not initialized
    """
    if _container is None:
        raise RuntimeError(
            "Dependency container not initialized. Call initialize_dependencies() first."
        )
    return _container


async def initialize_dependencies(config: Config | None = None) -> None:
    """Initialize the global dependency container.

    Args:
        config: Application configuration
    """
    global _container  # noqa: PLW0603
    if _container is None:
        _container = DependencyContainer()
    await _container.initialize(config)


async def cleanup_dependencies() -> None:
    """Clean up the global dependency container."""
    global _container  # noqa: PLW0603
    if _container:
        await _container.cleanup()
        _container = None


# FastAPI dependency functions


def get_config_dependency() -> Config:
    """FastAPI dependency for configuration.

    Returns:
        Configuration instance
    """
    return get_container().config


async def get_vector_service() -> QdrantService:
    """FastAPI dependency for vector database service.

    Returns:
        Vector database service instance

    Raises:
        HTTPException: If service is not available
    """
    try:
        container = get_container()
        if not container.is_initialized:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector service not available",
            )
        return container.vector_service
    except Exception as e:
        logger.error(f"Failed to get vector service: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from e


async def get_embedding_manager() -> EmbeddingManager:
    """FastAPI dependency for embedding manager.

    Returns:
        Embedding manager instance

    Raises:
        HTTPException: If service is not available
    """
    try:
        container = get_container()
        if not container.is_initialized:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding service not available",
            )
        return container.embedding_manager
    except Exception as e:
        logger.error(f"Failed to get embedding manager: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not available",
        ) from e


async def get_cache_manager() -> CacheManager:
    """FastAPI dependency for cache manager.

    Returns:
        Cache manager instance

    Raises:
        HTTPException: If service is not available
    """
    try:
        container = get_container()
        if not container.is_initialized:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available",
            )
        return container.cache_manager
    except Exception as e:
        logger.error(f"Failed to get cache manager: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service not available",
        ) from e


def get_correlation_id_dependency(request: Request) -> str:
    """FastAPI dependency for correlation ID from request.

    Args:
        request: HTTP request

    Returns:
        Correlation ID string
    """
    return get_correlation_id(request)


@asynccontextmanager
async def database_session() -> AsyncGenerator[Any]:
    """Async context manager for database sessions.

    This is a placeholder for when database sessions are needed.
    Currently returns None as we're using Qdrant which doesn't
    require traditional database sessions.

    Yields:
        Database session (currently None)
    """
    # Placeholder for future database session implementation
    yield None


def get_request_context(request: Request) -> dict[str, Any]:
    """FastAPI dependency for request context information.

    Args:
        request: HTTP request

    Returns:
        Dictionary with request context data
    """
    return {
        "correlation_id": get_correlation_id(request),
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }


class ServiceHealthChecker:
    """Health checker for production services."""

    def __init__(self, container: DependencyContainer):
        """Initialize health checker.

        Args:
            container: Dependency container to check
        """
        self.container = container

    async def check_health(self) -> dict[str, Any]:
        """Check health of all services.

        Returns:
            Health status dictionary
        """
        health = {"status": "healthy", "services": {}, "timestamp": None}

        from datetime import datetime

        health["timestamp"] = datetime.utcnow().isoformat()

        # Check container initialization
        if not self.container.is_initialized:
            health["status"] = "unhealthy"
            health["services"]["container"] = {"status": "not_initialized"}
            return health

        # Check vector service
        try:
            # Attempt a simple health check operation
            await self.container.vector_service.list_collections()
            health["services"]["vector_db"] = {"status": "healthy"}
        except Exception as e:
            health["status"] = "degraded"
            health["services"]["vector_db"] = {"status": "unhealthy", "error": str(e)}

        # Check embedding service
        try:
            # Check if embedding manager is responsive
            health["services"]["embeddings"] = {"status": "healthy"}
        except Exception as e:
            health["status"] = "degraded"
            health["services"]["embeddings"] = {"status": "unhealthy", "error": str(e)}

        # Check cache service
        try:
            # Check cache manager health
            health["services"]["cache"] = {"status": "healthy"}
        except Exception as e:
            health["status"] = "degraded"
            health["services"]["cache"] = {"status": "unhealthy", "error": str(e)}

        return health


def get_health_checker() -> ServiceHealthChecker:
    """FastAPI dependency for service health checker.

    Returns:
        Service health checker instance
    """
    return ServiceHealthChecker(get_container())


# Export all dependency functions
__all__ = [
    "DependencyContainer",
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_config_dependency",
    "get_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
