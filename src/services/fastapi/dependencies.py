"""FastAPI dependency helpers wired directly to service singletons."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi.exceptions import HTTPException
from fastapi.requests import Request  # type: ignore
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from src.config.loader import Settings, get_settings
from src.infrastructure.client_manager import ClientManager
from src.services.fastapi.middleware.correlation import get_correlation_id
from src.services.registry import ensure_service_registry, shutdown_service_registry
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.services.cache.manager import CacheManager
    from src.services.managers.embedding_manager import EmbeddingManager


logger = logging.getLogger(__name__)

T = TypeVar("T")
MIN_HEALTH_PROBE_TIMEOUT = 0.05


async def initialize_dependencies() -> None:
    """Prime long-lived services used by the FastAPI application."""

    await ensure_service_registry()


async def cleanup_dependencies() -> None:
    """Release shared services created for FastAPI usage."""

    await shutdown_service_registry()


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
        registry = await ensure_service_registry()
        return registry.client_manager
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain client manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client manager not available",
        ) from exc


async def get_embedding_manager() -> EmbeddingManager:
    """Expose the shared embedding manager instance."""

    try:
        registry = await ensure_service_registry()
        return registry.embedding_manager
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to obtain embedding manager")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding manager not available",
        ) from exc


async def get_cache_manager() -> CacheManager:
    """Expose the cache manager maintained by the registry."""

    try:
        registry = await ensure_service_registry()
        return registry.cache_manager
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

    registry = await ensure_service_registry()
    async with registry.database_manager.get_session() as session:
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

    @staticmethod
    async def _await_with_budget(
        awaitable: Awaitable[T],
        remaining: float,
        *,
        minimum_timeout: float = MIN_HEALTH_PROBE_TIMEOUT,
    ) -> tuple[T, float]:
        """Await a coroutine within the remaining timeout budget."""

        timeout = max(remaining, minimum_timeout)
        start_time = perf_counter()
        result = await asyncio.wait_for(awaitable, timeout=timeout)
        elapsed = perf_counter() - start_time
        new_remaining = max(0.0, remaining - elapsed)
        return result, new_remaining

    @staticmethod
    def _mark_healthy(
        health: dict[str, Any], service_key: str, duration_ms: float
    ) -> None:
        health["services"][service_key] = {
            "service": service_key,
            "status": "healthy",
            "duration_ms": duration_ms,
        }

    @staticmethod
    def _mark_unhealthy(
        health: dict[str, Any], service_key: str, exc: Exception, duration_ms: float
    ) -> None:
        health["status"] = "degraded"
        degraded = health.setdefault("degraded_services", [])
        if service_key not in degraded:
            degraded.append(service_key)
        health["services"][service_key] = {
            "service": service_key,
            "status": "unhealthy",
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "duration_ms": duration_ms,
        }

    async def _probe_vector_service(self, health: dict[str, Any]) -> None:
        service_key = "vector_db"
        start_time = perf_counter()
        try:
            remaining = self.HEALTH_TIMEOUT_SECONDS
            client_manager, remaining = await self._await_with_budget(
                get_client_manager(),
                remaining,
            )
            vector_service, remaining = await self._await_with_budget(
                client_manager.get_vector_store_service(),
                remaining,
            )
            _, remaining = await self._await_with_budget(
                vector_service.list_collections(),
                remaining,
            )
            duration_ms = (perf_counter() - start_time) * 1000
            self._mark_healthy(health, service_key, duration_ms)
            logger.debug(
                "Health probe for %s succeeded in %.2fms",
                service_key,
                duration_ms,
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            duration_ms = (perf_counter() - start_time) * 1000
            self._mark_unhealthy(health, service_key, exc, duration_ms)
            logger.warning(
                "Health probe for %s failed after %.2fms: %s",
                service_key,
                duration_ms,
                exc,
                exc_info=True,
            )

    async def _probe_embedding_manager(self, health: dict[str, Any]) -> None:
        service_key = "embeddings"
        start_time = perf_counter()
        try:
            remaining = self.HEALTH_TIMEOUT_SECONDS
            client_manager, remaining = await self._await_with_budget(
                get_client_manager(),
                remaining,
            )
            manager, remaining = await self._await_with_budget(
                client_manager.get_embedding_manager(),
                remaining,
            )
            _, remaining = await self._await_with_budget(
                asyncio.to_thread(manager.get_provider_info),
                remaining,
            )
            duration_ms = (perf_counter() - start_time) * 1000
            self._mark_healthy(health, service_key, duration_ms)
            logger.debug(
                "Health probe for %s succeeded in %.2fms",
                service_key,
                duration_ms,
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            duration_ms = (perf_counter() - start_time) * 1000
            self._mark_unhealthy(health, service_key, exc, duration_ms)
            logger.warning(
                "Health probe for %s failed after %.2fms: %s",
                service_key,
                duration_ms,
                exc,
                exc_info=True,
            )

    async def _probe_cache_manager(self, health: dict[str, Any]) -> None:
        service_key = "cache"
        start_time = perf_counter()
        try:
            remaining = self.HEALTH_TIMEOUT_SECONDS
            client_manager, remaining = await self._await_with_budget(
                get_client_manager(),
                remaining,
            )
            cache_manager, remaining = await self._await_with_budget(
                client_manager.get_cache_manager(),
                remaining,
            )
            stats_callable = getattr(cache_manager, "get_stats", None) or getattr(
                cache_manager, "get_performance_stats", None
            )
            if stats_callable is None or not callable(stats_callable):
                msg = "Cache manager missing callable stats method"
                raise RuntimeError(msg)
            if asyncio.iscoroutinefunction(stats_callable):
                _, remaining = await self._await_with_budget(
                    stats_callable(),
                    remaining,
                )
            else:
                result, remaining = await self._await_with_budget(
                    asyncio.to_thread(stats_callable),
                    remaining,
                )
                if inspect.isawaitable(result):
                    if remaining <= 0.0:
                        msg = "Cache stats awaitable exceeded timeout budget"
                        raise TimeoutError(msg)
                    _, remaining = await self._await_with_budget(
                        result,
                        remaining,
                    )
            duration_ms = (perf_counter() - start_time) * 1000
            stats_name = getattr(
                stats_callable, "__name__", stats_callable.__class__.__name__
            )
            self._mark_healthy(health, service_key, duration_ms)
            health["services"][service_key]["stats_method"] = stats_name
            logger.debug(
                "Health probe for %s succeeded in %.2fms via %s",
                service_key,
                duration_ms,
                stats_name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            duration_ms = (perf_counter() - start_time) * 1000
            self._mark_unhealthy(health, service_key, exc, duration_ms)
            logger.warning(
                "Health probe for %s failed after %.2fms: %s",
                service_key,
                duration_ms,
                exc,
                exc_info=True,
            )

    async def check_health(self) -> dict[str, Any]:
        """Execute health probes and return their aggregated status."""
        health: dict[str, Any] = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "degraded_services": [],
        }

        await asyncio.gather(
            self._probe_vector_service(health),
            self._probe_embedding_manager(health),
            self._probe_cache_manager(health),
        )
        return health


def get_health_checker() -> ServiceHealthChecker:
    """Provide a ServiceHealthChecker instance."""

    return ServiceHealthChecker()


__all__ = [
    "ServiceHealthChecker",
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
