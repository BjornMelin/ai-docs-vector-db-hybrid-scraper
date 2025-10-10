"""Mode-aware FastAPI application factory with profile-driven composition."""

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

from src.api.app_profiles import AppProfile, detect_profile
from src.api.lifespan import container_lifespan
from src.architecture.modes import ApplicationMode, ModeConfig, get_mode_config
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.dependencies import (
    get_cache_manager as core_get_cache_manager,
    get_content_intelligence_service as core_get_content_intelligence_service,
    get_embedding_manager as core_get_embedding_manager,
    get_vector_store_service as core_get_vector_store_service,
)
from src.services.fastapi.dependencies import HealthCheckerDep
from src.services.fastapi.middleware.manager import apply_defaults, apply_named_stack
from src.services.observability.health_manager import HealthStatus


try:
    from .routers import config_router
except ImportError:
    config_router = None


logger = logging.getLogger(__name__)


def _coerce_profile(
    profile: AppProfile | ApplicationMode | str | None,
) -> AppProfile:
    """Normalize incoming profile/mode inputs to an :class:`AppProfile`."""

    if profile is None:
        return detect_profile()
    if isinstance(profile, AppProfile):
        return profile
    if isinstance(profile, ApplicationMode):
        return AppProfile.from_mode(profile)
    if isinstance(profile, str):
        try:
            return AppProfile(profile.lower())
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown application profile: {profile}") from exc
    msg = f"Unsupported profile type: {type(profile)!r}"
    raise TypeError(msg)


def create_app(
    profile: AppProfile | ApplicationMode | str | None = None,
) -> FastAPI:
    """Create FastAPI app configured for a specific profile.

    Args:
        profile: Desired application profile. When ``None`` the profile is
            detected from environment configuration. ``ApplicationMode`` and
            string values are accepted for backward compatibility.

    Returns:
        Configured FastAPI application instance.
    """

    resolved_profile = _coerce_profile(profile)
    mode = resolved_profile.to_mode()

    mode_config = get_mode_config(mode)

    # Create app with mode-specific configuration
    app = FastAPI(
        title=f"AI Docs Vector DB ({mode.value.title()} Mode)",
        description=_get_app_description(mode),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc" if mode == ApplicationMode.ENTERPRISE else None,
        openapi_url="/openapi.json",
    )

    # Store mode information in app state
    app.state.profile = resolved_profile
    app.state.mode = mode
    app.state.mode_config = mode_config

    # Register lifespan handler (temporary bridge retains global factory usage)
    app.router.lifespan_context = _build_app_lifespan(app)

    # Configure CORS based on mode
    _configure_cors(app, mode)

    # Apply mode-specific middleware stack
    _apply_middleware_stack(app, mode_config.middleware_stack)

    # Add profile-specific routes
    _configure_routes(app, resolved_profile, mode)

    logger.info("Created FastAPI app in %s mode", mode.value)

    return app


def _get_app_description(mode: ApplicationMode) -> str:
    """Get mode-specific application description."""
    base_description = (
        "Hybrid AI documentation scraping system with vector database integration"
    )

    if mode == ApplicationMode.SIMPLE:
        return (
            f"{base_description} - Optimized for solo developers "
            "with minimal complexity"
        )
    if mode == ApplicationMode.ENTERPRISE:
        return (
            f"{base_description} - Full enterprise feature set "
            "with advanced capabilities"
        )
    return base_description


def _configure_cors(app: FastAPI, mode: ApplicationMode) -> None:
    """Configure CORS middleware based on mode."""
    if mode == ApplicationMode.SIMPLE:
        # Secure CORS for development with domain whitelist
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )
    else:
        # More restrictive CORS for enterprise
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8000"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )


def _apply_middleware_stack(app: FastAPI, middleware_stack: list[str]) -> None:
    """Apply mode-specific middleware stack."""

    if not middleware_stack:
        apply_defaults(app)
        return

    applied = apply_named_stack(app, middleware_stack)
    if not applied:
        apply_defaults(app)


def _configure_routes(app: FastAPI, profile: AppProfile, mode: ApplicationMode) -> None:
    """Configure routes based on application profile."""
    # Always include basic config router
    if config_router:
        app.include_router(config_router, prefix="/api/v1")

    _install_application_routes(app)

    # Add common routes
    _configure_common_routes(app)


def _install_application_routes(app: FastAPI) -> None:
    """Mount routers for the application profile with lazy imports."""

    required_modules = {
        "search": "src.api.routers.v1.search",
        "documents": "src.api.routers.v1.documents",
    }
    routers: dict[str, Any] = {}
    missing: list[str] = []

    for key, module_path in required_modules.items():
        try:
            routers[key] = import_module(module_path)
        except ImportError as exc:
            missing.append(f"{module_path} ({exc})")

    if missing:
        message = ", ".join(missing)
        logger.error(
            "Application routes unavailable: %s",
            message,
        )
        raise RuntimeError(
            "Application routes require canonical routers to be installed"
        ) from ImportError(message)

    app.include_router(routers["search"].router, prefix="/api/v1", tags=["search"])
    app.include_router(
        routers["documents"].router, prefix="/api/v1", tags=["documents"]
    )
    logger.debug("Configured application routes")


def _configure_common_routes(app: FastAPI) -> None:
    """Configure common routes available in both modes."""

    @app.get("/")
    async def root():
        """Root endpoint with mode information."""
        mode = app.state.mode
        return {
            "message": (
                f"AI Docs Vector DB Hybrid Scraper API ({mode.value.title()} Mode)"
            ),
            "version": "0.1.0",
            "status": "running",
            "mode": mode.value,
            "features": _get_mode_features(mode),
        }

    @app.get("/health")
    async def health_check(checker: HealthCheckerDep):
        """Health check endpoint."""

        mode = app.state.mode
        await checker.check_all()
        summary = checker.get_health_summary()
        overall_status = summary.get("overall_status", HealthStatus.UNKNOWN.value)

        status_code = status.HTTP_200_OK
        if overall_status == HealthStatus.UNHEALTHY.value:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        payload = {
            "status": overall_status,
            "mode": mode.value,
            "services": summary.get("checks", {}),
            "healthy_count": summary.get("healthy_count", 0),
            "total_count": summary.get("total_count", 0),
            "timestamp": summary.get("timestamp", datetime.now(UTC).isoformat()),
        }

        return JSONResponse(payload, status_code=status_code)

    @app.get("/info")
    async def info():
        """Information about the API and current mode."""
        mode = app.state.mode
        mode_config = app.state.mode_config

        return {
            "name": f"AI Docs Vector DB Hybrid Scraper ({mode.value.title()} Mode)",
            "version": "0.1.0",
            "description": _get_app_description(mode),
            "python_version": "3.11+",
            "framework": "FastAPI",
            "mode": {
                "name": mode.value,
                "enabled_services": mode_config.enabled_services,
                "resource_limits": mode_config.resource_limits,
                "features": _get_mode_features(mode),
            },
        }

    @app.get("/mode")
    async def mode_info():
        """Detailed mode configuration information."""
        mode_config: ModeConfig = app.state.mode_config
        mode_value = app.state.mode.value

        return {
            "mode": mode_value,
            "enabled_services": mode_config.enabled_services,
            "resource_limits": mode_config.resource_limits,
            "middleware_stack": mode_config.middleware_stack,
            "advanced_monitoring": mode_config.enable_advanced_monitoring,
            "deployment_features": mode_config.enable_deployment_features,
            "a_b_testing": mode_config.enable_a_b_testing,
        }


def _get_mode_features(mode: ApplicationMode) -> dict[str, Any]:
    """Get feature summary for the current mode."""
    if mode == ApplicationMode.SIMPLE:
        return {
            "advanced_monitoring": False,
            "deployment_features": False,
            "a_b_testing": False,
            "comprehensive_observability": False,
            "max_concurrent_crawls": 5,
            "target_users": "solo developers",
            "complexity": "minimal",
        }
    return {
        "advanced_monitoring": True,
        "deployment_features": True,
        "a_b_testing": True,
        "comprehensive_observability": True,
        "max_concurrent_crawls": 50,
        "target_users": "enterprise teams",
        "complexity": "full",
    }


def _build_app_lifespan(app: FastAPI):
    """Return a lifespan context manager for FastAPI startup/shutdown."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mode = app.state.mode
        mode_config: ModeConfig = app.state.mode_config

        logger.info("Starting application in %s mode", mode.value)

        async with container_lifespan(app):
            await _initialize_mode_services(mode_config)

            logger.info("Application startup complete in %s mode", mode.value)

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


async def _initialize_mode_services(mode_config: ModeConfig) -> None:
    """Initialize services required for the active application mode."""

    service_initializers: dict[str, Callable[[], Awaitable[Any]]] = {
        "embedding_service": core_get_embedding_manager,
        "vector_db_service": core_get_vector_store_service,
        "qdrant_client": _init_qdrant_client,
        "simple_caching": core_get_cache_manager,
        "multi_tier_caching": core_get_cache_manager,
        "advanced_search": core_get_vector_store_service,
        "basic_search": core_get_vector_store_service,
        "deployment_services": _ensure_database_ready,
        "advanced_analytics": core_get_content_intelligence_service,
    }

    for service_name in mode_config.enabled_services:
        initializer = service_initializers.get(service_name)
        if initializer is None:
            continue
        try:
            await initializer()
            logger.info("Initialized critical service: %s", service_name)
        except Exception:  # pragma: no cover - defensive log
            logger.exception("Failed to initialize critical service %s", service_name)


def get_app_mode(app: FastAPI) -> ApplicationMode:
    """Get the mode of a FastAPI application."""
    return app.state.mode  # type: ignore[attr-defined]


def get_app_container(app: FastAPI) -> ApplicationContainer:
    """Get the dependency-injector container from a FastAPI application."""

    container = getattr(app.state, "container", None)
    if container is None:
        msg = "DI container is not attached to application state"
        raise RuntimeError(msg)
    if not isinstance(container, ApplicationContainer):
        msg = "Application state container is not an ApplicationContainer instance"
        raise TypeError(msg)
    return container
