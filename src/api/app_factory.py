"""FastAPI application factory configured for a unified deployment path."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from src.api.lifespan import container_lifespan
from src.config.loader import Settings, get_settings
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.dependencies import (
    get_cache_manager as core_get_cache_manager,
    get_content_intelligence_service as core_get_content_intelligence_service,
    get_embedding_manager as core_get_embedding_manager,
    get_vector_store_service as core_get_vector_store_service,
)
from src.services.fastapi.dependencies import HealthCheckerDep
from src.services.fastapi.middleware.manager import apply_defaults
from src.services.health.manager import HealthStatus


try:
    from .routers import config_router
except ImportError:  # pragma: no cover - optional configuration router
    config_router = None


logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]


def create_app() -> FastAPI:
    """Create a FastAPI application using the cached settings."""

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Hybrid AI documentation scraping system with vector search",
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.router.lifespan_context = _build_app_lifespan()

    _configure_cors(app)
    _apply_middleware(app)
    _configure_routes(app, settings)

    logger.info("Created FastAPI app with unified configuration")

    return app


def _configure_cors(app: FastAPI) -> None:
    """Attach a permissive CORS policy for local development."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEFAULT_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


def _apply_middleware(app: FastAPI) -> None:
    """Install the default middleware stack."""

    apply_defaults(app)


def _configure_routes(app: FastAPI, settings: Settings) -> None:
    """Register application and utility routes."""

    if config_router:
        app.include_router(config_router, prefix="/api/v1")

    _install_application_routes(app)
    _configure_common_routes(app, settings)


def _install_application_routes(app: FastAPI) -> None:
    """Mount the canonical application routers."""

    required_modules = {
        "search": "src.api.routers.v1.search",
        "documents": "src.api.routers.v1.documents",
    }
    routers: dict[str, Any] = {}
    missing: list[str] = []

    for key, module_path in required_modules.items():
        try:
            routers[key] = import_module(module_path)
        except ImportError as exc:  # pragma: no cover - optional import failure
            missing.append(f"{module_path} ({exc})")

    if missing:
        message = ", ".join(missing)
        logger.error("Application routes unavailable: %s", message)
        raise RuntimeError(
            "Application routes require canonical routers to be installed"
        ) from ImportError(message)

    app.include_router(routers["search"].router, prefix="/api/v1", tags=["search"])
    app.include_router(
        routers["documents"].router, prefix="/api/v1", tags=["documents"]
    )
    logger.debug("Configured application routes")


def _configure_common_routes(app: FastAPI, settings: Settings) -> None:
    """Configure informational endpoints for the service."""

    feature_flags = settings.get_feature_flags()

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Return service summary and active features."""

        return {
            "message": f"{settings.app_name} API",
            "version": settings.version,
            "status": "running",
            "environment": settings.environment.value,
            "features": feature_flags,
        }

    @app.get("/health")
    async def health_check(checker: HealthCheckerDep) -> JSONResponse:
        """Return aggregated health status for dependencies."""

        await checker.check_all()
        summary = checker.get_health_summary()
        overall_status = summary.get("overall_status", HealthStatus.UNKNOWN.value)

        status_code = status.HTTP_200_OK
        if overall_status == HealthStatus.UNHEALTHY.value:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        payload = {
            "status": overall_status,
            "services": summary.get("checks", {}),
            "healthy_count": summary.get("healthy_count", 0),
            "total_count": summary.get("total_count", 0),
            "timestamp": summary.get("timestamp", datetime.now(UTC).isoformat()),
        }

        return JSONResponse(payload, status_code=status_code)

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Return descriptive information about the API."""

        return {
            "name": settings.app_name,
            "version": settings.version,
            "description": (
                "Hybrid AI documentation scraping system "
                "with vector database integration"
            ),
            "python_version": "3.11+",
            "framework": "FastAPI",
            "environment": settings.environment.value,
            "features": feature_flags,
        }

    @app.get("/features")
    async def features() -> dict[str, Any]:
        """Expose feature flag configuration for observability."""

        return feature_flags


def _build_app_lifespan():
    """Return a lifespan context manager for FastAPI startup and shutdown."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting application")

        async with container_lifespan(app):
            await _initialize_services()

            logger.info("Application startup complete")

            try:
                yield
            finally:
                logger.info("Shutting down application")
                logger.info("Application shutdown complete")

    return lifespan


async def _ping_redis() -> None:
    container = get_container()
    if container is None:
        return
    try:
        redis_client = container.redis_client()
        ping_result = redis_client.ping()
        if asyncio.iscoroutine(ping_result):
            await ping_result
    except Exception:  # pragma: no cover - optional dependency not available
        logger.debug("Redis ping failed during startup", exc_info=True)


async def _init_qdrant_client() -> None:
    container = get_container()
    if container is None:
        return
    try:
        qdrant_client = container.qdrant_client()
        get_collections = getattr(qdrant_client, "get_collections", None)
        if callable(get_collections):
            result = get_collections()
            if asyncio.iscoroutine(result):
                await result
    except Exception:  # pragma: no cover - optional dependency not available
        logger.debug("Qdrant readiness check failed during startup", exc_info=True)


async def _ensure_database_ready() -> None:
    await core_get_vector_store_service()
    await core_get_cache_manager()
    await _ping_redis()


async def _initialize_services() -> None:
    """Initialize critical services required for the application."""

    service_initializers: dict[str, Callable[[], Awaitable[Any]]] = {
        "embedding_service": core_get_embedding_manager,
        "vector_db_service": core_get_vector_store_service,
        "qdrant_client": _init_qdrant_client,
        "cache_manager": core_get_cache_manager,
        "content_intelligence": core_get_content_intelligence_service,
        "database_ready": _ensure_database_ready,
    }

    for service_name, initializer in service_initializers.items():
        try:
            await initializer()
            logger.info("Initialized critical service: %s", service_name)
        except Exception:  # pragma: no cover - defensive log
            logger.exception("Failed to initialize critical service %s", service_name)


def get_app_container(app: FastAPI) -> ApplicationContainer:
    """Return the dependency-injector container from a FastAPI application."""

    container = getattr(app.state, "container", None)
    if container is None:
        msg = "DI container is not attached to application state"
        raise RuntimeError(msg)
    if not isinstance(container, ApplicationContainer):
        msg = "Application state container is not an ApplicationContainer instance"
        raise TypeError(msg)
    return container


__all__ = [
    "create_app",
    "get_app_container",
]
