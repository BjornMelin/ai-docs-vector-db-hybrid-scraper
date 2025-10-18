"""FastAPI dependency helpers resolved through the DI container."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends  # type: ignore[attr-defined]
from fastapi.exceptions import HTTPException
from fastapi.requests import Request  # type: ignore
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config.loader import Settings, get_settings
from src.infrastructure.bootstrap import ensure_container
from src.infrastructure.container import get_container, shutdown_container
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.observability.health_manager import (
    HealthCheckManager,
    build_health_manager,
)
from src.services.service_resolver import (
    get_cache_manager as resolve_cache_manager,
    get_embedding_manager as resolve_embedding_manager,
    get_rag_generator as resolve_rag_generator,
    get_vector_store_service as resolve_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.services.cache.manager import CacheManager
    from src.services.embeddings.manager import EmbeddingManager


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatabaseSessionContext:
    """Lightweight context exposing cache and vector helpers.

    Attributes:
        dragonfly: The optional Dragonfly client instance.
        cache_manager: Cache manager resolved for the request lifecycle.
        vector_service: Vector store service resolved from the container.
    """

    dragonfly: Any | None
    cache_manager: CacheManager | None
    vector_service: VectorStoreService | None


async def initialize_dependencies() -> None:
    """Prime long-lived services used by the FastAPI application.

    Raises:
        RuntimeError: If the dependency container cannot be initialised.
    """
    await ensure_container(settings=get_settings(), force_reload=True)


async def cleanup_dependencies() -> None:
    """Release shared services created for FastAPI usage.

    Raises:
        RuntimeError: If the dependency container cannot be shut down.
    """
    await shutdown_container()


def get_config_dependency() -> Settings:
    """Return application configuration for FastAPI routes.

    Returns:
        Settings: Pydantic settings instance representing the app configuration.
    """
    return get_settings()


async def get_vector_service() -> VectorStoreService:
    """Return the vector service instance via the DI container.

    Returns:
        VectorStoreService: Vector service resolved from the dependency container.

    Raises:
        HTTPException: If the service cannot be obtained from the container.
    """
    try:
        return await resolve_vector_store_service()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain vector service from container")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from exc


async def get_embedding_manager() -> EmbeddingManager:
    """Expose the shared embedding manager instance.

    Returns:
        EmbeddingManager: Embedding manager provided by the container.

    Raises:
        HTTPException: If the embedding manager cannot be resolved.
    """
    try:
        return await resolve_embedding_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain embedding manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding manager not available",
        ) from exc


async def get_cache_manager() -> CacheManager:
    """Expose the cache manager maintained by the DI container.

    Returns:
        CacheManager: Cache manager shared across the FastAPI application.

    Raises:
        HTTPException: If the cache manager cannot be resolved.
    """
    try:
        return await resolve_cache_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain cache manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager not available",
        ) from exc


CacheManagerDep = Annotated["CacheManager", Depends(get_cache_manager)]
EmbeddingManagerDep = Annotated["EmbeddingManager", Depends(get_embedding_manager)]
VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_service)]
ConfigDep = Annotated[Settings, Depends(get_config_dependency)]


async def get_rag_generator() -> Any:
    """Return the configured RAG generator instance.

    Returns:
        Any: The retrieval-augmented generation pipeline configured in the container.

    Raises:
        HTTPException: If the generator cannot be resolved.
    """
    try:
        return await resolve_rag_generator()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain RAG generator")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG generator not available",
        ) from exc


RAGGeneratorDep = Annotated[Any, Depends(get_rag_generator)]


@asynccontextmanager
async def database_session() -> AsyncGenerator[DatabaseSessionContext, None]:
    """Provide a database session context with automatic cleanup.

    Yields:
        AsyncGenerator[DatabaseSessionContext, None]: Context with resolved clients
        used during the request lifecycle.

    Raises:
        HTTPException: If the dependency container is not initialised.
    """
    container = get_container()
    if container is None:
        logger.error("Dependency container is not initialised for database session")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service container unavailable",
        )

    dragonfly_client = None
    try:
        dragonfly_client = container.dragonfly_client()
    except (
        RuntimeError,
        AttributeError,
        ValueError,
    ):  # pragma: no cover - defensive logging
        logger.debug("Dragonfly client unavailable", exc_info=True)

    cache_manager: CacheManager | None = None
    try:
        cache_manager = await resolve_cache_manager()
    except (
        RuntimeError,
        ImportError,
        ValueError,
        AttributeError,
    ):  # pragma: no cover - defensive logging
        logger.debug("Cache manager unavailable during session", exc_info=True)

    vector_service: VectorStoreService | None = None
    try:
        vector_service = await resolve_vector_store_service()
    except (
        RuntimeError,
        ImportError,
        ValueError,
        AttributeError,
    ):  # pragma: no cover - defensive logging
        logger.debug("Vector service unavailable during session", exc_info=True)

    context = DatabaseSessionContext(
        dragonfly=dragonfly_client,
        cache_manager=cache_manager,
        vector_service=vector_service,
    )

    try:
        yield context
    finally:
        if dragonfly_client is not None and hasattr(dragonfly_client, "close"):
            close_result = dragonfly_client.close()
            if asyncio.iscoroutine(close_result):  # pragma: no cover - defensive
                await close_result


def get_correlation_id_dependency(request: Request) -> str:
    """Return the correlation identifier extracted from the request.

    Args:
        request: Incoming FastAPI request containing header metadata.

    Returns:
        str: The correlation identifier associated with the request.
    """
    return get_correlation_id(request)


def get_request_context(request: Request) -> dict[str, Any]:
    """Assemble lightweight request diagnostics for logging.

    Args:
        request: Incoming FastAPI request containing metadata and headers.

    Returns:
        dict[str, Any]: Structured diagnostics captured from the request.
    """
    return {
        "correlation_id": get_correlation_id(request),
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }


_health_manager: HealthCheckManager | None = None
_health_manager_lock = asyncio.Lock()


async def get_health_checker() -> HealthCheckManager:
    """Return a configured :class:`HealthCheckManager` singleton.

    Returns:
        HealthCheckManager: The shared health manager used by the API.
    """
    global _health_manager  # pylint: disable=global-statement
    if _health_manager is not None:
        return _health_manager

    async with _health_manager_lock:
        if _health_manager is not None:
            return _health_manager

        settings = get_settings()
        _health_manager = build_health_manager(settings)
        return _health_manager


HealthCheckerDep = Annotated[HealthCheckManager, Depends(get_health_checker)]


__all__ = [
    "CacheManagerDep",
    "ConfigDep",
    "DatabaseSessionContext",
    "EmbeddingManagerDep",
    "HealthCheckerDep",
    "RAGGeneratorDep",
    "VectorStoreServiceDep",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_config_dependency",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
