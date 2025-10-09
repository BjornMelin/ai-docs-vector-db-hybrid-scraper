"""FastAPI dependency helpers wired directly to service singletons."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends
from fastapi.exceptions import HTTPException
from fastapi.requests import Request  # type: ignore
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config.loader import Settings, get_settings
from src.infrastructure.client_manager import (
    ClientManager,
    ensure_client_manager,
    shutdown_client_manager,
)
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.health.manager import HealthCheckManager, build_health_manager
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.services.cache.manager import CacheManager
    from src.services.embeddings.manager import EmbeddingManager


logger = logging.getLogger(__name__)


async def initialize_dependencies() -> None:
    """Prime long-lived services used by the FastAPI application."""

    await ensure_client_manager()


async def cleanup_dependencies() -> None:
    """Release shared services created for FastAPI usage."""

    await shutdown_client_manager()


def get_config_dependency() -> Settings:
    """Return application configuration for FastAPI routes."""

    return get_settings()


async def get_vector_service() -> VectorStoreService:
    """Return the vector service instance via the client manager."""

    try:
        client_manager = await get_client_manager()
        return await client_manager.get_vector_store_service()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain vector service via ClientManager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from exc


async def get_client_manager() -> ClientManager:
    """Return the initialized ClientManager singleton."""

    try:
        return await ensure_client_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain client manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
        ) from exc


async def get_embedding_manager() -> EmbeddingManager:
    """Expose the shared embedding manager instance."""

    try:
        client_manager = await ensure_client_manager()
        return await client_manager.get_embedding_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain embedding manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding manager not available",
        ) from exc


async def get_cache_manager() -> CacheManager:
    """Expose the cache manager maintained by the registry."""

    try:
        client_manager = await ensure_client_manager()
        return await client_manager.get_cache_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain cache manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager not available",
        ) from exc


async def get_vector_store_service() -> VectorStoreService:
    """Backward-compatible alias for get_vector_service."""

    return await get_vector_service()


def get_correlation_id_dependency(request: Request) -> str:
    """Return the correlation identifier extracted from the request."""

    return get_correlation_id(request)


@asynccontextmanager
async def database_session() -> AsyncGenerator[Any]:
    """Provide a database session with automatic cleanup."""

    client_manager = await ensure_client_manager()
    async with client_manager.database_session() as session:
        yield session


def get_request_context(request: Request) -> dict[str, Any]:
    """Assemble lightweight request diagnostics for logging."""

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
    """Return a configured :class:`HealthCheckManager` singleton."""

    global _health_manager  # pylint: disable=global-statement
    if _health_manager is not None:
        return _health_manager

    async with _health_manager_lock:
        if _health_manager is not None:
            return _health_manager

        settings = get_settings()
        qdrant_client = None
        try:
            client_manager = await ensure_client_manager()
            qdrant_client = await client_manager.get_qdrant_client()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to obtain Qdrant client for health checks: %s", exc)

        _health_manager = build_health_manager(
            settings,
            qdrant_client=qdrant_client,
        )
        return _health_manager


HealthCheckerDep = Annotated[HealthCheckManager, Depends(get_health_checker)]


__all__ = [
    "HealthCheckerDep",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_client_manager",
    "get_config_dependency",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_health_checker",
    "get_request_context",
    "get_vector_store_service",
    "get_vector_service",
    "initialize_dependencies",
]
