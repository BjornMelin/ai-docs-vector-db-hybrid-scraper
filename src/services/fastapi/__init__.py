import typing


"""FastAPI production service package.

This package provides production-ready FastAPI enhancements for the
unified MCP server, including middleware, dependency injection,
and background task management.
"""

from .background import (
    BackgroundTaskManager,
    cleanup_task_manager,
    get_task_manager,
    initialize_task_manager,
    submit_managed_task,
)
from .dependencies import (
    cleanup_dependencies,
    get_cache_manager,
    get_config_dependency as get_config,
    get_embedding_manager,
    get_fastapi_config,
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
from .middleware.manager import MiddlewareManager, create_middleware_manager
from .production_server import (
    ProductionMCPServer,
    create_production_server,
    run_production_server_async,
)


__all__ = [
    # Background tasks
    "BackgroundTaskManager",
    # Middleware components
    "CompressionMiddleware",
    # Middleware management
    "MiddlewareManager",
    "PerformanceMiddleware",
    # Production server
    "ProductionMCPServer",
    "SecurityMiddleware",
    "TimeoutMiddleware",
    "TracingMiddleware",
    "cleanup_dependencies",
    "cleanup_task_manager",
    "create_middleware_manager",
    "create_production_server",
    "get_cache_manager",
    "get_config",
    "get_correlation_id",
    "get_embedding_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_task_manager",
    "get_vector_service",
    # Dependency injection
    "initialize_dependencies",
    "initialize_task_manager",
    "run_production_server_async",
    "submit_managed_task",
]
