"""Mode-aware FastAPI application factory with profile-driven composition."""

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from src.api.app_profiles import AppProfile, detect_profile
from src.api.lifespan import service_registry_lifespan
from src.architecture.modes import ApplicationMode, get_mode_config
from src.architecture.service_factory import (
    ModeAwareServiceFactory,
    set_service_factory,
)
from src.services.fastapi.dependencies import get_health_checker
from src.services.fastapi.middleware.manager import apply_defaults, apply_named_stack


try:
    from .routers import config_router
except ImportError:
    config_router = None


logger = logging.getLogger(__name__)


def _resolve_optional_class(module_path: str, attribute: str) -> Any | None:
    """Return attribute from module when available, suppressing import errors."""

    try:
        module = import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attribute, None)


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
    app.state.service_factory = ModeAwareServiceFactory(mode)
    set_service_factory(app.state.service_factory)

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

    # Add profile-specific routes
    if profile is AppProfile.SIMPLE:
        _install_simple_routes(app)
    elif profile is AppProfile.ENTERPRISE:
        _install_enterprise_routes(app)

    # Add common routes
    _configure_common_routes(app)


def _install_simple_routes(app: FastAPI) -> None:
    """Mount routers for the simple profile with lazy imports."""

    try:
        simple_search = import_module("src.api.routers.simple.search")
        simple_documents = import_module("src.api.routers.simple.documents")
    except ImportError as exc:  # pragma: no cover - defensive
        logger.error("Simple profile requires simple routers: %s", exc)
        raise RuntimeError(
            "Simple profile requires simple routers to be installed"
        ) from exc
    app.include_router(simple_search.router, prefix="/api/v1", tags=["search"])
    app.include_router(simple_documents.router, prefix="/api/v1", tags=["documents"])
    logger.debug("Configured simple profile routes")


def _install_enterprise_routes(app: FastAPI) -> None:
    """Mount routers for the enterprise profile with lazy imports."""

    required_modules = {
        "search": "src.api.routers.enterprise.search",
        "documents": "src.api.routers.enterprise.documents",
        "analytics": "src.api.routers.enterprise.analytics",
        "deployment": "src.api.routers.enterprise.deployment",
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
            "Enterprise profile selected but enterprise routers are unavailable: %s",
            message,
        )
        raise RuntimeError(
            "Enterprise profile requires enterprise routers to be installed"
        ) from ImportError(message)

    app.include_router(routers["search"].router, prefix="/api/v1", tags=["search"])
    app.include_router(
        routers["documents"].router, prefix="/api/v1", tags=["documents"]
    )
    app.include_router(
        routers["analytics"].router, prefix="/api/v1", tags=["analytics"]
    )
    app.include_router(
        routers["deployment"].router, prefix="/api/v1", tags=["deployment"]
    )
    logger.debug("Configured enterprise profile routes")


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
    async def health_check():
        """Health check endpoint."""
        mode = app.state.mode
        checker = get_health_checker()
        health_report = await checker.check_health()

        status_code = (
            status.HTTP_200_OK
            if health_report.get("status") == "healthy"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )

        payload = {
            "status": health_report["status"],
            "mode": mode.value,
            "services": health_report.get("services", {}),
            "timestamp": health_report.get("timestamp", datetime.now(UTC).isoformat()),
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
        service_factory: ModeAwareServiceFactory = app.state.service_factory

        return service_factory.get_mode_info()


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
        service_factory: ModeAwareServiceFactory = app.state.service_factory

        logger.info("Starting application in %s mode", mode.value)

        async with service_registry_lifespan(app):
            _register_mode_services(service_factory)
            await _initialize_critical_services(service_factory)

            logger.info("Application startup complete in %s mode", mode.value)

            try:
                yield
            finally:
                logger.info("Shutting down application")
                await service_factory.cleanup_all_services()
                logger.info("Application shutdown complete")

    return lifespan


def _register_mode_services(factory: ModeAwareServiceFactory) -> None:
    """Register services for the specified mode."""
    embedding_manager_cls = _resolve_optional_class(
        "src.services.embeddings.manager", "EmbeddingManager"
    )
    if embedding_manager_cls:
        factory.register_universal_service("embedding_service", embedding_manager_cls)
    else:
        logger.warning("Embedding manager unavailable; embedding service disabled")

    vector_store_cls = _resolve_optional_class(
        "src.services.vector_db.service", "VectorStoreService"
    )
    if vector_store_cls:
        factory.register_universal_service("vector_db_service", vector_store_cls)
    else:
        logger.warning("Vector store service unavailable; vector DB service disabled")


async def _initialize_critical_services(factory: ModeAwareServiceFactory) -> None:
    """Initialize critical services required for basic operation."""
    critical_services = list(factory.mode_config.enabled_services)

    for service_name in critical_services:
        try:
            service = await factory.get_service_optional(service_name)
            if service:
                logger.info("Initialized critical service: %s", service_name)
            else:
                logger.warning("Critical service not available: %s", service_name)
        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Failed to initialize critical service %s", service_name)


def get_app_mode(app: FastAPI) -> ApplicationMode:
    """Get the mode of a FastAPI application."""
    return app.state.mode  # type: ignore[attr-defined]


def get_app_service_factory(app: FastAPI) -> ModeAwareServiceFactory:
    """Get the service factory from a FastAPI application."""
    return app.state.service_factory  # type: ignore[attr-defined]
