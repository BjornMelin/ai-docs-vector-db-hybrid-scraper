"""FastAPI dependency helpers wired directly to service singletons."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends  # type: ignore[attr-defined]
from fastapi.exceptions import HTTPException
from fastapi.requests import Request  # type: ignore
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config.loader import Settings, get_settings
from src.infrastructure.bootstrap import ensure_container
from src.infrastructure.container import shutdown_container
from src.services.dependencies import (
    get_cache_manager as core_get_cache_manager,
    get_database_session as core_get_database_session,
    get_embedding_manager as core_get_embedding_manager,
    get_vector_store_service as core_get_vector_store_service,
)
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.observability.health_manager import (
    HealthCheckManager,
    build_health_manager,
)
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.services.cache.manager import CacheManager
    from src.services.embeddings.manager import EmbeddingManager


logger = logging.getLogger(__name__)


async def initialize_dependencies() -> None:
    """Prime long-lived services used by the FastAPI application."""

    await ensure_container(settings=get_settings(), force_reload=True)


async def cleanup_dependencies() -> None:
    """Release shared services created for FastAPI usage."""

    await shutdown_container()


def get_config_dependency() -> Settings:
    """Return application configuration for FastAPI routes."""

    return get_settings()


async def get_vector_service() -> VectorStoreService:
    """Return the vector service instance via the DI container."""

    try:
        return await core_get_vector_store_service()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain vector service from container")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from exc


async def get_embedding_manager() -> EmbeddingManager:
    """Expose the shared embedding manager instance."""

    try:
        return await core_get_embedding_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain embedding manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding manager not available",
        ) from exc


async def get_cache_manager() -> CacheManager:
    """Expose the cache manager maintained by the registry."""

    try:
        return await core_get_cache_manager()
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

    generator = core_get_database_session()
    try:
        context = await generator.__anext__()
        yield context
    finally:
        await generator.aclose()


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
        _health_manager = build_health_manager(
            settings,
        )
        return _health_manager


HealthCheckerDep = Annotated[HealthCheckManager, Depends(get_health_checker)]


__all__ = [
    "HealthCheckerDep",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_config_dependency",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_health_checker",
    "get_request_context",
    "get_vector_store_service",
    "get_vector_service",
    "initialize_dependencies",
]
