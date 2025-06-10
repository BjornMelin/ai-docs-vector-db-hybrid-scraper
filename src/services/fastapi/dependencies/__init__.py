"""FastAPI dependency injection components for production environments."""

from .core import DependencyContainer
from .core import ServiceHealthChecker
from .core import cleanup_dependencies
from .core import database_session
from .core import get_cache_manager
from .core import get_config
from .core import get_container
from .core import get_correlation_id_dependency
from .core import get_embedding_manager
from .core import get_fastapi_config
from .core import get_health_checker
from .core import get_request_context
from .core import get_vector_service
from .core import initialize_dependencies

__all__ = [
    "DependencyContainer",
    "ServiceHealthChecker",
    "cleanup_dependencies",
    "database_session",
    "get_cache_manager",
    "get_config",
    "get_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    "get_vector_service",
    "initialize_dependencies",
]
