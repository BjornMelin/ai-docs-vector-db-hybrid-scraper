"""Core dependency injection components for FastAPI production deployment.

This module provides essential dependencies for database sessions,
configuration management, and other production services.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any


try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config import Config, get_config
from src.infrastructure.client_manager import ClientManager

# Import new function-based dependencies
from src.services.dependencies import (
    cleanup_services as cleanup_dependency_services,
    get_cache_manager,
    get_database_manager,
    get_embedding_manager,
    get_ready_client_manager,
)
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

RedisErrorType = getattr(redis, "RedisError", Exception)


class DependencyContainer:
    """Container for managing production dependencies and their lifecycle."""

    def __init__(self):
        """Initialize dependency container."""
        self._config: Config | None = None
        self._client_manager: ClientManager | None = None
        self._vector_service: VectorStoreService | None = None
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

        self._client_manager = await get_ready_client_manager()
        self._vector_service = await self._client_manager.get_vector_store_service()

    async def _initialize_managers(self) -> None:
        """Initialize embedding and cache managers."""

        ready_manager = self._client_manager or await get_ready_client_manager()
        self._embedding_manager = await get_embedding_manager(ready_manager)
        self._cache_manager = await get_cache_manager(ready_manager)

    async def cleanup(self) -> None:
        """Clean up all dependencies."""

        try:
            await self._perform_cleanup()
        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Error during dependency cleanup")

    async def _perform_cleanup(self) -> None:
        """Perform dependency cleanup steps."""

        await cleanup_dependency_services()
        self._vector_service = None
        self._embedding_manager = None
        self._cache_manager = None
        self._client_manager = None

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
    def vector_service(self) -> VectorStoreService:
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
            msg = (
                "Dependency container not initialized. "
                "Call initialize_dependencies() first."
            )
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


def get_app_dependency_container() -> DependencyContainer:
    """Return the FastAPI dependency container singleton."""

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

    return get_app_dependency_container().config


def get_fastapi_config() -> Config:
    """Alias for get_config_dependency for backward compatibility.

    Returns:
        Configuration instance
    """

    return get_config_dependency()


async def get_vector_service() -> VectorStoreService:
    """Return an initialized vector service suitable for FastAPI dependencies."""

    try:
        manager = await get_ready_client_manager()
        return await manager.get_vector_store_service()
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to obtain vector service")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from exc


async def get_client_manager() -> ClientManager:
    """Return an initialized ClientManager for FastAPI dependencies."""

    try:
        return await get_ready_client_manager()
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to obtain client manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
        ) from exc


def get_correlation_id_dependency(request: Request) -> str:
    """FastAPI dependency for correlation ID from request."""

    return get_correlation_id(request)


@asynccontextmanager
async def database_session() -> AsyncGenerator[Any]:
    """Async context manager for database sessions.

    Provides SQLAlchemy async sessions through the DatabaseManager.

    Returns:
        AsyncSession: SQLAlchemy async database session
    """

    client_manager = await get_ready_client_manager()
    db_manager = await get_database_manager(client_manager)

    async with db_manager.get_session() as session:
        yield session


def get_request_context(request: Request) -> dict[str, Any]:
    """FastAPI dependency for request context information."""

    return {
        "correlation_id": get_correlation_id(request),
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }


class ServiceHealthChecker:
    """Health checker for production services."""

    HEALTH_TIMEOUT_SECONDS = 2.0

    def __init__(self, container: DependencyContainer):
        """Initialize health checker.

        Args:
            container: Dependency container to check
        """

        self.container = container

    async def check_health(self) -> dict[str, Any]:
        """Check health of all services."""

        health = {"status": "healthy", "services": {}, "timestamp": None}

        health["timestamp"] = datetime.now(tz=UTC).isoformat()

        # Check container initialization
        if not self.container.is_initialized:
            health["status"] = "unhealthy"
            health["services"]["container"] = {"status": "not_initialized"}
            return health

        # Check vector service readiness
        try:
            vector_service = self.container.vector_service
        except RuntimeError:
            vector_service = await get_vector_service()

        try:
            await asyncio.wait_for(
                vector_service.list_collections(),
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            )
            health["services"]["vector_db"] = {"status": "healthy"}
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            health["status"] = "degraded"
            health["services"]["vector_db"] = {
                "status": "unhealthy",
                "error": str(exc),
            }

        # Check embedding service readiness
        try:
            embedding_manager = self.container.embedding_manager
        except RuntimeError:
            embedding_manager = await get_embedding_manager(
                await get_ready_client_manager()
            )

        try:
            await asyncio.wait_for(
                asyncio.to_thread(embedding_manager.get_provider_info),
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            )
            health["services"]["embeddings"] = {"status": "healthy"}
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            health["status"] = "degraded"
            health["services"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(exc),
            }

        # Check cache service readiness
        try:
            cache_manager = self.container.cache_manager
        except RuntimeError:
            cache_manager = await get_cache_manager(await get_ready_client_manager())

        try:
            await asyncio.wait_for(
                cache_manager.get_stats(),
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            )
            health["services"]["cache"] = {"status": "healthy"}
        except (RedisErrorType, ConnectionError, TimeoutError, ValueError) as exc:
            health["status"] = "degraded"
            health["services"]["cache"] = {
                "status": "unhealthy",
                "error": str(exc),
            }

        return health


def get_health_checker() -> ServiceHealthChecker:
    """FastAPI dependency for service health checker.

    Returns:
        Service health checker instance
    """

    return ServiceHealthChecker(get_app_dependency_container())


# Export all dependency functions
__all__ = [
    "DependencyContainer",
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_config_dependency",
    "get_app_dependency_container",
    "get_correlation_id_dependency",
    "get_client_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
