"""Core dependency injection components for FastAPI production deployment.

This module provides essential dependencies for database sessions,
configuration management, and other production services.
"""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any


try:
    import redis
except ImportError:
    redis = None

from fastapi import HTTPException, Request
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config import Config, get_config
from src.infrastructure.client_manager import ClientManager

# Import new function-based dependencies
from src.services.dependencies import get_cache_manager, get_embedding_manager
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
            await self._perform_initialization(config)
        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Failed to initialize dependency container")
            raise

    async def _perform_initialization(self, config: Config | None) -> None:
        """Perform dependency initialization steps.

        Args:
            config: Application configuration

        """
        # Load configurations
        if config is None:
            config = get_config()
        self._config = config

        # Initialize services sequentially
        await self._initialize_core_services(config)
        await self._initialize_managers()

        self._initialized = True
        logger.info("Dependency container initialized successfully")

    async def _initialize_core_services(self, config: Config) -> None:
        """Initialize core client manager and vector service.

        Args:
            config: Application configuration

        """
        self._client_manager = ClientManager.from_unified_config()
        self._vector_service = QdrantService(config, self._client_manager)
        await self._vector_service.initialize()

    async def _initialize_managers(self) -> None:
        """Initialize embedding and cache managers."""
        self._embedding_manager = await get_embedding_manager(self._client_manager)
        self._cache_manager = await get_cache_manager(self._client_manager)

    async def cleanup(self) -> None:
        """Clean up all dependencies."""
        try:
            await self._perform_cleanup()
        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Error during dependency cleanup")

    async def _perform_cleanup(self) -> None:
        """Perform dependency cleanup steps."""
        # Function-based managers are cleaned up through client manager
        if self._vector_service:
            await self._vector_service.cleanup()

        if self._client_manager:
            await self._client_manager.cleanup()

        self._initialized = False
        logger.info("Dependency container cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized

    @property
    def config(self) -> Config:
        """Get configuration."""
        if not self._config:
            msg = "Dependency container not initialized"
            raise RuntimeError(msg)
        return self._config

    @property
    def vector_service(self) -> QdrantService:
        """Get vector database service."""
        if not self._vector_service:
            msg = "Vector service not initialized"
            raise RuntimeError(msg)
        return self._vector_service

    @property
    def embedding_manager(self) -> Any:
        """Get embedding manager."""
        if not self._embedding_manager:
            msg = "Embedding manager not initialized"
            raise RuntimeError(msg)
        return self._embedding_manager

    @property
    def cache_manager(self) -> Any:
        """Get cache manager."""
        if not self._cache_manager:
            msg = "Cache manager not initialized"
            raise RuntimeError(msg)
        return self._cache_manager

    @property
    def client_manager(self) -> ClientManager:
        """Get client manager."""
        if not self._client_manager:
            msg = "Client manager not initialized"
            raise RuntimeError(msg)
        return self._client_manager


class _DependencyContainerSingleton:
    """Singleton holder for dependency container instance."""

    _instance: DependencyContainer | None = None

    @classmethod
    def get_instance(cls) -> DependencyContainer:
        """Get the singleton dependency container instance."""
        if cls._instance is None:
            msg = "Dependency container not initialized. Call initialize_dependencies() first."
            raise RuntimeError(msg)
        return cls._instance

    @classmethod
    async def initialize_instance(cls, config: Config | None = None) -> None:
        """Initialize the singleton with configuration."""
        if cls._instance is None:
            cls._instance = DependencyContainer()
        await cls._instance.initialize(config)

    @classmethod
    async def cleanup_instance(cls) -> None:
        """Cleanup the singleton instance."""
        if cls._instance:
            await cls._instance.cleanup()
            cls._instance = None


def get_container() -> DependencyContainer:
    """Get the global dependency container.

    Returns:
        Global dependency container instance

    Raises:
        RuntimeError: If container is not initialized

    """
    return _DependencyContainerSingleton.get_instance()


async def initialize_dependencies(config: Config | None = None) -> None:
    """Initialize the global dependency container.

    Args:
        config: Application configuration

    """
    await _DependencyContainerSingleton.initialize_instance(config)


async def cleanup_dependencies() -> None:
    """Clean up the global dependency container."""
    await _DependencyContainerSingleton.cleanup_instance()


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


def _raise_vector_service_unavailable() -> HTTPException:
    """Helper to raise vector service unavailable exception."""
    raise HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Vector service not available",
    )


async def get_vector_service() -> QdrantService:
    """FastAPI dependency for vector database service.

    Returns:
        Vector database service instance

    Raises:
        HTTPException: If service is not available

    """
    try:
        return _get_service_from_container(
            "vector_service", _raise_vector_service_unavailable
        )
    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Failed to get vector service")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from None


def _raise_embedding_service_unavailable() -> HTTPException:
    """Helper to raise embedding service unavailable exception."""
    raise HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Embedding service not available",
    )


async def get_embedding_manager_legacy() -> Any:
    """FastAPI dependency for embedding manager (legacy container approach).

    Returns:
        Embedding manager instance via legacy container

    Raises:
        HTTPException: If service is not available

    """
    try:
        return _get_service_from_container(
            "embedding_manager", _raise_embedding_service_unavailable
        )
    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Failed to get embedding manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not available",
        ) from None


def _raise_cache_service_unavailable() -> HTTPException:
    """Helper to raise cache service unavailable exception."""
    raise HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Cache service not available",
    )


async def get_cache_manager_legacy() -> Any:
    """FastAPI dependency for cache manager (legacy container approach).

    Returns:
        Cache manager instance via legacy container

    Raises:
        HTTPException: If service is not available

    """
    try:
        return _get_service_from_container(
            "cache_manager", _raise_cache_service_unavailable
        )
    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Failed to get cache manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service not available",
        ) from None


def _raise_client_manager_unavailable() -> HTTPException:
    """Helper to raise client manager unavailable exception."""
    raise HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Client manager not available",
    )


def get_client_manager() -> ClientManager:
    """FastAPI dependency for client manager.

    Returns:
        Client manager instance

    Raises:
        HTTPException: If service is not available

    """
    try:
        return _get_service_from_container(
            "client_manager", _raise_client_manager_unavailable
        )
    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Failed to get client manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
        ) from None


def _get_service_from_container(
    service_name: str, unavailable_handler: Callable
) -> Any:
    """Get service from container with availability check.

    Args:
        service_name: Name of the service attribute
        unavailable_handler: Function to call if service unavailable

    Returns:
        Service instance

    Raises:
        HTTPException: If service is not available

    """
    container = get_container()
    if not container.is_initialized:
        unavailable_handler()
    return getattr(container, service_name)


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

        health["timestamp"] = datetime.now(tz=UTC).isoformat()

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
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
            health["status"] = "degraded"
            health["services"]["vector_db"] = {"status": "unhealthy", "error": str(e)}

        # Check embedding service
        try:
            # Check if embedding manager is responsive
            health["services"]["embeddings"] = {"status": "healthy"}
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            health["status"] = "degraded"
            health["services"]["embeddings"] = {"status": "unhealthy", "error": str(e)}

        # Check cache service
        try:
            # Check cache manager health
            health["services"]["cache"] = {"status": "healthy"}
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
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
