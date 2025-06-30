"""Mode-aware FastAPI application factory.

This module creates FastAPI applications configured for specific modes,
implementing the dual-mode architecture for complexity management.
"""

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.architecture.modes import ApplicationMode, get_current_mode, get_mode_config
from src.architecture.service_factory import ModeAwareServiceFactory
from src.services.fastapi.middleware.manager import get_middleware_manager


# Import routers and services (conditional imports moved to top level)
try:
    from .routers import config_router
except ImportError:
    config_router = None

try:
    from .routers.simple import (
        documents as simple_documents,
        search as simple_search,
    )
except ImportError:
    simple_documents = None
    simple_search = None

try:
    from .routers.enterprise import (
        analytics as enterprise_analytics,
        deployment as enterprise_deployment,
        documents as enterprise_documents,
        search as enterprise_search,
    )
except ImportError:
    enterprise_analytics = None
    enterprise_deployment = None
    enterprise_documents = None
    enterprise_search = None

try:
    from src.services.enterprise import (
        cache as enterprise_cache,
        search as enterprise_search_service,
    )
    from src.services.simple import (
        cache as simple_cache,
        search as simple_search_service,
    )
except ImportError:
    enterprise_cache = None
    enterprise_search_service = None
    simple_cache = None
    simple_search_service = None

try:
    from src.services.embeddings.manager import EmbeddingManager
    from src.services.vector_db.service import VectorDBService
except ImportError:
    EmbeddingManager = None
    VectorDBService = None


logger = logging.getLogger(__name__)


def create_app(mode: ApplicationMode | None = None) -> FastAPI:
    """Create FastAPI app configured for specific mode.

    Args:
        mode: Application mode to use. If None, detects from environment.

    Returns:
        Configured FastAPI application instance
    """
    if mode is None:
        mode = get_current_mode()

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
    app.state.mode = mode
    app.state.mode_config = mode_config
    app.state.service_factory = ModeAwareServiceFactory(mode)

    # Configure CORS based on mode
    _configure_cors(app, mode)

    # Apply mode-specific middleware stack
    _apply_middleware_stack(app, mode_config.middleware_stack)

    # Add mode-specific routes
    _configure_routes(app, mode)

    # Add startup and shutdown events
    _configure_lifecycle_events(app)

    logger.info("Created FastAPI app in %s mode", mode.value)

    return app


def _get_app_description(mode: ApplicationMode) -> str:
    """Get mode-specific application description."""
    base_description = (
        "Hybrid AI documentation scraping system with vector database integration"
    )

    if mode == ApplicationMode.SIMPLE:
        return f"{base_description} - Optimized for solo developers with minimal complexity"
    if mode == ApplicationMode.ENTERPRISE:
        return f"{base_description} - Full enterprise feature set with advanced capabilities"
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
            allow_headers=["Authorization", "Content-Type"],
        )


def _apply_middleware_stack(app: FastAPI, middleware_stack: list[str]) -> None:
    """Apply mode-specific middleware stack."""
    middleware_manager = get_middleware_manager()
    middleware_manager.apply_middleware(app, middleware_stack)


def _configure_routes(app: FastAPI, mode: ApplicationMode) -> None:
    """Configure routes based on application mode."""
    # Always include basic config router
    if config_router:
        app.include_router(config_router, prefix="/api/v1")

    # Add mode-specific routes
    if mode == ApplicationMode.SIMPLE:
        _configure_simple_routes(app)
    elif mode == ApplicationMode.ENTERPRISE:
        _configure_enterprise_routes(app)

    # Add common routes
    _configure_common_routes(app)


def _configure_simple_routes(app: FastAPI) -> None:
    """Configure routes for simple mode."""
    # Use pre-imported simple mode routers
    if simple_search and simple_documents:
        app.include_router(simple_search.router, prefix="/api/v1", tags=["search"])
        app.include_router(
            simple_documents.router, prefix="/api/v1", tags=["documents"]
        )
        logger.debug("Configured simple mode routes")
    else:
        logger.warning("Simple mode routers not available")


def _configure_enterprise_routes(app: FastAPI) -> None:
    """Configure routes for enterprise mode."""
    # Use pre-imported enterprise mode routers
    if all(
        [
            enterprise_search,
            enterprise_documents,
            enterprise_analytics,
            enterprise_deployment,
        ]
    ):
        app.include_router(enterprise_search.router, prefix="/api/v1", tags=["search"])
        app.include_router(
            enterprise_documents.router, prefix="/api/v1", tags=["documents"]
        )
        app.include_router(
            enterprise_analytics.router, prefix="/api/v1", tags=["analytics"]
        )
        app.include_router(
            enterprise_deployment.router, prefix="/api/v1", tags=["deployment"]
        )
        logger.debug("Configured enterprise mode routes")
    else:
        logger.warning("Enterprise mode routers not available")


def _configure_common_routes(app: FastAPI) -> None:
    """Configure common routes available in both modes."""

    @app.get("/")
    async def root():
        """Root endpoint with mode information."""
        mode = app.state.mode
        return {
            "message": f"AI Docs Vector DB Hybrid Scraper API ({mode.value.title()} Mode)",
            "version": "0.1.0",
            "status": "running",
            "mode": mode.value,
            "features": _get_mode_features(mode),
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        mode = app.state.mode
        service_factory: ModeAwareServiceFactory = app.state.service_factory

        # Get available services for health check
        available_services = service_factory.get_available_services()

        return {
            "status": "healthy",
            "mode": mode.value,
            "available_services": available_services,
            "timestamp": "2025-06-28T00:00:00Z",
        }

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


def _configure_lifecycle_events(app: FastAPI) -> None:
    """Configure startup and shutdown events."""

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        mode = app.state.mode
        service_factory: ModeAwareServiceFactory = app.state.service_factory

        logger.info("Starting application in %s mode", mode.value)

        # Register mode-specific services
        _register_mode_services(service_factory)

        # Initialize critical services
        await _initialize_critical_services(service_factory)

        logger.info("Application startup complete in %s mode", mode.value)

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up services on shutdown."""
        service_factory: ModeAwareServiceFactory = app.state.service_factory

        logger.info("Shutting down application")
        await service_factory.cleanup_all_services()
        logger.info("Application shutdown complete")


def _register_mode_services(factory: ModeAwareServiceFactory) -> None:
    """Register services for the specified mode."""
    # Use pre-imported service implementations
    if all(
        [
            enterprise_cache,
            enterprise_search_service,
            simple_cache,
            simple_search_service,
        ]
    ):
        # Register search services
        factory.register_service(
            "search_service",
            simple_search_service.SimpleSearchService,
            enterprise_search_service.EnterpriseSearchService,
        )

        # Register cache services
        factory.register_service(
            "cache_service",
            simple_cache.SimpleCacheService,
            enterprise_cache.EnterpriseCacheService,
        )
    else:
        logger.warning("Some service implementations not available")

    # Register universal services (work in both modes)
    if EmbeddingManager and VectorDBService:
        factory.register_universal_service("embedding_service", EmbeddingManager)
        factory.register_universal_service("vector_db_service", VectorDBService)
    else:
        logger.warning("Universal services not available")


async def _initialize_critical_services(factory: ModeAwareServiceFactory) -> None:
    """Initialize critical services required for basic operation."""
    critical_services = [
        "embedding_service",
        "vector_db_service",
        "search_service",
        "cache_service",
    ]

    for service_name in critical_services:
        try:
            service = await factory.get_service_optional(service_name)
            if service:
                logger.info("Initialized critical service: %s", service_name)
            else:
                logger.warning("Critical service not available: %s", service_name)
        except Exception:
            logger.exception("Failed to initialize critical service %s", service_name)


def get_app_mode(app: FastAPI) -> ApplicationMode:
    """Get the mode of a FastAPI application."""
    return app.state.mode  # type: ignore[attr-defined]


def get_app_service_factory(app: FastAPI) -> ModeAwareServiceFactory:
    """Get the service factory from a FastAPI application."""
    return app.state.service_factory  # type: ignore[attr-defined]
