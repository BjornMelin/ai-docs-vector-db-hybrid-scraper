"""FastAPI production service package.

This package provides production-ready FastAPI enhancements for the
unified MCP server, including middleware, dependency injection,
and background task management.
"""

from .background import BackgroundTaskManager
from .background import cleanup_task_manager
from .background import get_task_manager
from .background import initialize_task_manager
from .background import submit_managed_task
from .dependencies import cleanup_dependencies
from .dependencies import get_cache_manager
from .dependencies import get_config
from .dependencies import get_embedding_manager
from .dependencies import get_fastapi_config
from .dependencies import get_health_checker
from .dependencies import get_vector_service
from .dependencies import initialize_dependencies
from .middleware import CompressionMiddleware
from .middleware import PerformanceMiddleware
from .middleware import SecurityMiddleware
from .middleware import TimeoutMiddleware
from .middleware import TracingMiddleware
from .middleware import get_correlation_id
from .middleware.manager import MiddlewareManager
from .middleware.manager import create_middleware_manager
from .production_server import ProductionMCPServer
from .production_server import create_production_server
from .production_server import run_production_server_async

__all__ = [
    # Middleware components
    "CompressionMiddleware",
    "PerformanceMiddleware",
    "SecurityMiddleware",
    "TimeoutMiddleware",
    "TracingMiddleware",
    "get_correlation_id",
    # Middleware management
    "MiddlewareManager",
    "create_middleware_manager",
    # Dependency injection
    "initialize_dependencies",
    "cleanup_dependencies",
    "get_config",
    "get_fastapi_config",
    "get_vector_service",
    "get_embedding_manager",
    "get_cache_manager",
    "get_health_checker",
    # Background tasks
    "BackgroundTaskManager",
    "get_task_manager",
    "initialize_task_manager",
    "cleanup_task_manager",
    "submit_managed_task",
    # Production server
    "ProductionMCPServer",
    "create_production_server",
    "run_production_server_async",
]
