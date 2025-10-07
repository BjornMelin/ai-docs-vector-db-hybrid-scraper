"""Convenience re-exports for FastAPI dependency helpers."""

from src.services.dependencies import (
    get_cache_manager as dependency_get_cache_manager,
    get_client_manager as get_client_manager_sync,
    get_embedding_manager as dependency_get_embedding_manager,
)

from .core import (
    DependencyContainer,
    ServiceHealthChecker,
    cleanup_dependencies,
    database_session,
    get_app_dependency_container,
    get_client_manager,
    get_config_dependency,
    get_correlation_id_dependency,
    get_fastapi_config,
    get_health_checker,
    get_request_context,
    get_vector_service,
    initialize_dependencies,
)


get_cache_manager = dependency_get_cache_manager
get_embedding_manager = dependency_get_embedding_manager


__all__ = [
    "DependencyContainer",
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_client_manager",
    "get_client_manager_sync",
    "get_config_dependency",
    "get_app_dependency_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
