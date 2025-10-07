"""Convenience re-exports for FastAPI dependency helpers."""

from src.services.dependencies import (
    get_cache_manager,
    get_embedding_manager,
)

from .core import (
    ServiceHealthChecker,
    cleanup_dependencies,
    database_session,
    get_client_manager,
    get_config_dependency,
    get_correlation_id_dependency,
    get_health_checker,
    get_request_context,
    get_vector_service,
    initialize_dependencies,
)


__all__ = [
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_client_manager",
    "get_config_dependency",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
