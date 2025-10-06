"""Convenience re-exports for FastAPI dependency helpers."""

from .core import (
    DependencyContainer,
    ServiceHealthChecker,
    cleanup_dependencies,
    database_session,
    get_cache_manager,
    get_client_manager,
    get_config_dependency,
    get_container,
    get_correlation_id_dependency,
    get_embedding_manager,
    get_fastapi_config,
    get_health_checker,
    get_request_context,
    get_vector_service,
    initialize_dependencies,
)


__all__ = [
    "DependencyContainer",
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_client_manager",
    "get_config_dependency",
    "get_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
