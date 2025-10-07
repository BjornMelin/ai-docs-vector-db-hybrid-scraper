"""FastAPI dependency helpers wired directly to service singletons.

This module exposes the final dependency surface for the FastAPI stack.
It avoids the previous container-style wrapper and delegates to the
function-based services defined under ``src.services.dependencies``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any


try:  # Optional redis dependency for cache health checks.
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config import Config, get_config
from src.infrastructure.client_manager import ClientManager
from src.services.dependencies import (
    cleanup_services,
    get_cache_manager,
    get_database_manager,
    get_embedding_manager,
    get_ready_client_manager,
    get_vector_store_service,
)
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

RedisErrorType = getattr(redis, "RedisError", Exception)


async def initialize_dependencies(config: Config | None = None) -> None:
    """Prime long-lived services used by the FastAPI application."""
    del config  # Configuration is resolved lazily inside the helpers.
    client_manager = await get_ready_client_manager()
    await client_manager.get_vector_store_service()
    await get_embedding_manager()
    await get_cache_manager()
    await get_database_manager()


async def cleanup_dependencies() -> None:
    """Release shared services created for FastAPI usage."""
    await cleanup_services()


def get_config_dependency() -> Config:
    """Return application configuration for FastAPI routes."""
    return get_config()


async def get_vector_service() -> VectorStoreService:
    """Return the vector service instance, raising 503 on failure."""
    try:
        return await get_vector_store_service()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain vector service")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service not available",
        ) from exc


async def get_client_manager() -> ClientManager:
    """Return the initialized ClientManager singleton."""
    try:
        return await get_ready_client_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain client manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
        ) from exc


def get_correlation_id_dependency(request: Request) -> str:
    """Return the correlation identifier extracted from the request."""
    return get_correlation_id(request)


@asynccontextmanager
async def database_session() -> AsyncGenerator[Any]:
    """Provide a database session with automatic cleanup."""
    db_manager = await get_database_manager()
    async with db_manager.get_session() as session:
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


class ServiceHealthChecker:
    """Perform asynchronous health probes for core services."""

    HEALTH_TIMEOUT_SECONDS = 2.0

    async def _probe_vector_service(self, health: dict[str, Any]) -> None:
        try:
            vector_service = await get_vector_store_service()
            await asyncio.wait_for(
                vector_service.list_collections(),
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            )
            health["services"]["vector_db"] = {"status": "healthy"}
        except Exception as exc:  # pragma: no cover - runtime failure
            health["status"] = "degraded"
            health["services"]["vector_db"] = {
                "status": "unhealthy",
                "error": str(exc),
            }

    async def _probe_embedding_manager(self, health: dict[str, Any]) -> None:
        try:
            manager = await get_embedding_manager()
            await asyncio.wait_for(
                asyncio.to_thread(manager.get_provider_info),
                timeout=self.HEALTH_TIMEOUT_SECONDS,
            )
            health["services"]["embeddings"] = {"status": "healthy"}
        except Exception as exc:  # pragma: no cover - runtime failure
            health["status"] = "degraded"
            health["services"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(exc),
            }

    async def _probe_cache_manager(self, health: dict[str, Any]) -> None:
        try:
            cache_manager = await get_cache_manager()
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

    async def check_health(self) -> dict[str, Any]:
        """Execute health probes and return their aggregated status."""
        health: dict[str, Any] = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        await self._probe_vector_service(health)
        await self._probe_embedding_manager(health)
        await self._probe_cache_manager(health)
        return health


def get_health_checker() -> ServiceHealthChecker:
    """Provide a ServiceHealthChecker instance."""
    return ServiceHealthChecker()


__all__ = [
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_client_manager",
    "get_config_dependency",
    "get_correlation_id_dependency",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
