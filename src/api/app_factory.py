"""FastAPI application factory configured for a unified deployment path."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
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
from src.services.observability.health_manager import HealthStatus


try:
    from .routers import config_router
except ImportError:  # pragma: no cover - optional configuration router
    config_router = None


logger = logging.getLogger(__name__)

LOCAL_DEV_CORS_ORIGINS = [
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
    app.router.lifespan_context = _build_app_lifespan(settings)

    _configure_cors(app, settings=settings)
    _apply_middleware(app)
    _configure_routes(app, settings)

    logger.info("Created FastAPI app with unified configuration")

    return app


def _parse_allowed_origins(raw_value: str) -> list[str]:
    """Return a normalized list of origins extracted from a raw string."""

    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def _resolve_allowed_cors_origins(settings: Settings) -> list[str]:
    """Determine the active CORS allow-list from configuration sources."""

    env_value = os.getenv("CORS_ALLOWED_ORIGINS")
    if env_value:
        parsed = _parse_allowed_origins(env_value)
        if parsed:
            return parsed

    security_origins = getattr(settings.security, "cors_allowed_origins", None) or []
    if security_origins:
        return list(security_origins)

    direct_setting = getattr(settings, "cors_allowed_origins", None) or []
    if direct_setting:
        return list(direct_setting)

    if settings.is_development():
        logger.debug("Using permissive CORS policy for development environment")
        return ["*"]

    logger.debug(
        "No explicit CORS origins configured; falling back to localhost allow list",
    )
    return LOCAL_DEV_CORS_ORIGINS


def _configure_cors(app: FastAPI, *, settings: Settings) -> None:
    """Attach a configurable CORS policy with secure defaults."""

    allowed_origins = _resolve_allowed_cors_origins(settings)
    allow_all = "*" in allowed_origins
    allow_credentials = not allow_all

    if allow_all:
        logger.debug(
            "Disabling CORS credentials because wildcard origins are permitted",
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"]
        if allow_all
        else ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"]
        if allow_all
        else ["Authorization", "Content-Type", "X-API-Key"],
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
        "cache": "src.api.routers.v1.cache",
    }
    routers: dict[str, Any] = {}
    missing: list[str] = []

    for key, module_path in required_modules.items():
        try:
            module = import_module(module_path)
            router = module.router
        except (
            ImportError,
            AttributeError,
        ) as exc:  # pragma: no cover - optional import failure or missing router
            missing.append(f"{module_path} ({exc})")
            continue

        routers[key] = router
    if missing:
        message = ", ".join(missing)
        logger.error("Application routes unavailable: %s", message)

    if "search" in routers:
        app.include_router(routers["search"], prefix="/api/v1", tags=["search"])
    if "documents" in routers:
        app.include_router(routers["documents"], prefix="/api/v1", tags=["documents"])
    if "cache" in routers:
        app.include_router(routers["cache"], prefix="/api/v1", tags=["cache"])

    if routers:
        logger.debug("Configured application routes: %s", ", ".join(sorted(routers)))
    else:
        logger.warning("No canonical application routes were registered")


def _configure_common_routes(app: FastAPI, settings: Settings) -> None:
    """Configure informational endpoints for the service."""

    @app.get("/")
    async def root() -> JSONResponse:
        """Return service summary and active features.

        Returns:
            JSONResponse containing app metadata and enabled features.
        """

        try:
            feature_flags = settings.get_feature_flags()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to build root endpoint payload")
            return JSONResponse(
                {"detail": "Configuration is unavailable"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        payload = {
            "message": f"{settings.app_name} API",
            "version": settings.version,
            "status": "running",
            "environment": settings.environment.value,
            "features": feature_flags,
        }
        return JSONResponse(payload)

    @app.get("/health")
    async def health_check(checker: HealthCheckerDep) -> JSONResponse:
        """Return aggregated health status for dependencies.

        Args:
            checker: Shared :class:`HealthCheckManager` instance.

        Returns:
            JSONResponse describing aggregate dependency health.
        """

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
        """Return descriptive information about the API.

        Returns:
            Dictionary with descriptive metadata about the API surface.
        """

        try:
            feature_flags = settings.get_feature_flags()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to include feature flags in info endpoint")
            feature_flags = {}

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
    async def features() -> JSONResponse:
        """Expose feature flag configuration for observability.

        Returns:
            JSONResponse enumerating all available feature toggles.
        """

        try:
            feature_flags = settings.get_feature_flags()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to expose feature flags")
            return JSONResponse(
                {"detail": "Feature flags are unavailable"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return JSONResponse(feature_flags)


def _build_app_lifespan(settings: Settings):
    """Return a lifespan context manager for FastAPI startup and shutdown."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting application")

        async with container_lifespan(app):
            await _initialize_services(settings)

            logger.info("Application startup complete")

            try:
                yield
            finally:
                logger.info("Shutting down application")
                logger.info("Application shutdown complete")

    return lifespan


async def _ping_dragonfly() -> None:
    container = get_container()
    if container is None:
        return
    try:
        dragonfly_client = container.dragonfly_client()
        ping_result = dragonfly_client.ping()
        if asyncio.iscoroutine(ping_result):
            await ping_result
    except Exception:  # pragma: no cover - optional dependency not available
        logger.debug("Dragonfly ping failed during startup", exc_info=True)


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


def _cache_initialization_enabled(settings: Settings) -> bool:
    """Return True when cache-related services should initialize."""

    cache_config = getattr(settings, "cache", None)
    if cache_config is None:
        return False
    return bool(getattr(cache_config, "enable_caching", False))


def _content_intelligence_available() -> bool:
    """Return True when the optional content intelligence module is importable."""

    return (
        importlib.util.find_spec("src.services.content_intelligence.service")
        is not None
    )


async def _ensure_database_ready(settings: Settings) -> None:
    await core_get_vector_store_service()

    if not _cache_initialization_enabled(settings):
        return

    await core_get_cache_manager()
    await _ping_dragonfly()


async def _initialize_services(settings: Settings) -> None:
    """Initialize critical services required for the application."""

    service_initializers: dict[str, Callable[[], Awaitable[Any]]] = {
        "embedding_service": core_get_embedding_manager,
        "vector_db_service": core_get_vector_store_service,
        "qdrant_client": _init_qdrant_client,
        "database_ready": lambda: _ensure_database_ready(settings),
    }

    if _cache_initialization_enabled(settings):
        service_initializers["cache_manager"] = core_get_cache_manager
        service_initializers["dragonfly_client"] = _ping_dragonfly
    else:
        logger.debug("Skipping cache initialization; caching disabled in configuration")

    if _content_intelligence_available():
        service_initializers["content_intelligence"] = (
            core_get_content_intelligence_service
        )
    else:
        logger.debug("Content intelligence module unavailable; skipping initializer")

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
