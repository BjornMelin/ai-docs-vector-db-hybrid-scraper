"""FastAPI production service package.

This package provides FastAPI middleware, dependency injection, and background
task management for the unified MCP Server.
"""

from .dependencies import (
    cleanup_dependencies,
    get_client_manager,
    get_config_dependency as get_config,
    get_health_checker,
    get_vector_service,
    initialize_dependencies,
)
from .middleware import (
    CompressionMiddleware,
    PerformanceMiddleware,
    SecurityMiddleware,
    TimeoutMiddleware,
    TracingMiddleware,
    get_correlation_id,
)
from .middleware.manager import MiddlewareManager, get_middleware_manager
from .production_server import (
    ProductionMCPServer,
    create_production_server,
    run_production_server_async,
)


__all__ = [
    # Background tasks
    # Middleware components
    "CompressionMiddleware",
    # Middleware management
    "MiddlewareManager",
    "get_middleware_manager",
    "PerformanceMiddleware",
    # Production server
    "ProductionMCPServer",
    "SecurityMiddleware",
    "TimeoutMiddleware",
    "TracingMiddleware",
    "cleanup_dependencies",
    "create_production_server",
    "get_client_manager",
    "get_config",
    "get_correlation_id",
    "get_health_checker",
    "get_vector_service",
    # Dependency injection
    "initialize_dependencies",
    "run_production_server_async",
]
