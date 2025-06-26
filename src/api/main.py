"""Main FastAPI application for the AI Docs Vector DB Hybrid Scraper.

This module provides the main FastAPI application instance and core API endpoints
with modern error handling and middleware integration.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import get_settings, reload_settings
from src.services.config_watcher import get_config_watcher

from .exceptions import (
    APIException,
    ConfigurationException,
    api_exception_handler,
    generic_exception_handler,
    http_exception_handler_override,
    validation_exception_handler,
)
from .middleware import (
    CircuitBreakerMiddleware,
    ErrorHandlingMiddleware,
    RateLimitingMiddleware,
    SecurityMiddleware,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application lifecycle."""
    # Startup
    settings = get_settings()

    # Start config watcher in development mode only
    if settings.environment == "development":
        try:
            watcher = get_config_watcher()
            watcher.start_watching()
            logger.info("Configuration watcher started in development mode")
        except Exception as e:
            logger.warning(f"Failed to start config watcher: {e}")

    yield

    # Shutdown
    if settings.environment == "development":
        try:
            watcher = get_config_watcher()
            watcher.stop_watching()
            logger.info("Configuration watcher stopped")
        except Exception as e:
            logger.warning(f"Error stopping config watcher: {e}")


# Create FastAPI application
app = FastAPI(
    title="AI Docs Vector DB Hybrid Scraper",
    description="Hybrid AI documentation scraping system with vector database integration",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware stack (order matters - first added is outermost)
settings = get_settings()

# Security middleware
app.add_middleware(SecurityMiddleware, enable_security_headers=True)

# Rate limiting middleware
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=100,
    enable_rate_limiting=settings.environment != "development",
)

# Circuit breaker middleware
app.add_middleware(
    CircuitBreakerMiddleware,
    enable_circuit_breaker=True,
)

# Error handling middleware
app.add_middleware(
    ErrorHandlingMiddleware,
    enable_detailed_errors=settings.environment == "development",
    enable_performance_tracking=True,
)

# CORS middleware (should be last in the stack)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler_override)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Docs Vector DB Hybrid Scraper API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-06-23T23:55:00Z"}


@app.get("/info")
async def info():
    """Information about the API."""
    return {
        "name": "AI Docs Vector DB Hybrid Scraper",
        "version": "0.1.0",
        "description": "Hybrid AI documentation scraping system with vector database integration",
        "python_version": "3.13+",
        "framework": "FastAPI",
    }


@app.post("/api/config/reload")
async def reload_config():
    """Manually reload configuration (development only)."""
    settings = get_settings()
    if settings.environment != "development":
        raise ConfigurationException(
            "Configuration reload is only available in development mode",
            context={"environment": settings.environment},
        )

    try:
        new_settings = reload_settings()
        logger.info("Configuration reloaded manually")
        return {
            "status": "success",
            "message": "Configuration reloaded",
            "environment": new_settings.environment,
        }
    except Exception as e:
        logger.exception("Failed to reload configuration")
        raise ConfigurationException(
            f"Failed to reload configuration: {e!s}",
            context={"operation": "reload_config"},
        ) from e


@app.get("/api/middleware/metrics")
async def get_middleware_metrics():
    """Get middleware performance and error metrics."""
    metrics = {}

    # Get error handling middleware metrics
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(
            middleware.cls, ErrorHandlingMiddleware
        ):
            # Access the middleware instance (this is a simplified approach)
            metrics["error_handling"] = {
                "enabled": True,
                "note": "Metrics collection active",
            }
            break

    # Add circuit breaker status
    try:
        from ..services.errors import CircuitBreakerRegistry

        circuit_status = CircuitBreakerRegistry.get_all_status()
        metrics["circuit_breakers"] = circuit_status
    except Exception as e:
        logger.debug(f"Could not get circuit breaker status: {e}")
        metrics["circuit_breakers"] = {
            "error": "Circuit breaker registry not available"
        }

    return {
        "status": "success",
        "middleware_metrics": metrics,
        "timestamp": "2025-06-26T23:55:00Z",
    }


# Include routers
# Note: metrics_endpoints temporarily disabled in test environment
# from .metrics_endpoints import router as metrics_router
# app.include_router(metrics_router, prefix="/api/v1")

__all__ = ["app"]
