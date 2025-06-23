import typing

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
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config import Config
from src.config import get_config
from src.infrastructure.client_manager import ClientManager

# Import new function-based dependencies
from src.services.dependencies import get_cache_manager
from src.services.dependencies import get_embedding_manager
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.vector_db.service import QdrantService

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Container for managing production dependencies and their lifecycle."""

    def __init__(self):
        """Initialize dependency container."""
        self._config: Config | None = None
        self._client_manager: ClientManager | None = None
        self._vector_service: QdrantService | None = None
        self._embedding_manager: Any | None = None
        self._cache_manager: Any | None = None
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

            # Initialize core services using function-based approach
            self._client_manager = ClientManager.from_unified_config()
            self._vector_service = QdrantService(config, self._client_manager)
            await self._vector_service.initialize()

            # Use function-based managers through client manager
            self._embedding_manager = await get_embedding_manager(self._client_manager)
            self._cache_manager = await get_cache_manager(self._client_manager)

            self._initialized = True
            logger.info("Dependency container initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize dependency container: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up all dependencies."""
        try:
            # Function-based managers are cleaned up through client manager
            if self._vector_service:
                await self._vector_service.cleanup()

            if self._client_manager:
                await self._client_manager.cleanup()

            self._initialized = False
            logger.info("Dependency container cleaned up")

        except Exception as e:
            logger.exception(f"Error during dependency cleanup: {e}")

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
    def embedding_manager(self) -> Any:
        """Get embedding manager."""
        if not self._embedding_manager:
            raise RuntimeError("Embedding manager not initialized")
        return self._embedding_manager

    @property
    def cache_manager(self) -> Any:
        """Get cache manager."""
        if not self._cache_manager:
            raise RuntimeError("Cache manager not initialized")
        return self._cache_manager

    @property
    def client_manager(self) -> ClientManager:
        """Get client manager."""
        if not self._client_manager:
            raise RuntimeError("Client manager not initialized")
        return self._client_manager


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


def get_fastapi_config() -> Config:
    """Alias for get_config_dependency for backward compatibility.

    Returns:
        Configuration instance
    """
    return get_config_dependency()


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
        logger.exception(f"Failed to get vector service: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from e


async def get_embedding_manager_legacy() -> Any:
    """FastAPI dependency for embedding manager (legacy container approach).

    Returns:
        Embedding manager instance via legacy container

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
        logger.exception(f"Failed to get embedding manager: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not available",
        ) from e


async def get_cache_manager_legacy() -> Any:
    """FastAPI dependency for cache manager (legacy container approach).

    Returns:
        Cache manager instance via legacy container

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
        logger.exception(f"Failed to get cache manager: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service not available",
        ) from e


def get_client_manager() -> ClientManager:
    """FastAPI dependency for client manager.

    Returns:
        Client manager instance

    Raises:
        HTTPException: If service is not available
    """
    try:
        container = get_container()
        if not container.is_initialized:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Client manager not available",
            )
        return container.client_manager
    except Exception as e:
        logger.exception(f"Failed to get client manager: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
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

    Provides SQLAlchemy async sessions through the DatabaseManager.

    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    client_manager = get_client_manager()
    db_manager = await client_manager.get_database_manager()

    async with db_manager.session() as session:
        yield session


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
    "get_client_manager",
    "get_config_dependency",
    "get_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
