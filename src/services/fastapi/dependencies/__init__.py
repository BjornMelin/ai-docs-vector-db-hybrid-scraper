import typing
"""FastAPI dependency injection components for production environments."""

# Import existing core dependencies
from src.services.dependencies import BrowserAutomationRouterDep
from src.services.dependencies import CacheManagerDep
from src.services.dependencies import CacheRequest
from src.services.dependencies import ClientManagerDep

# Import new function-based dependencies
from src.services.dependencies import ConfigDep  # Configuration Dependencies
from src.services.dependencies import ContentIntelligenceServiceDep
from src.services.dependencies import CrawlManagerDep
from src.services.dependencies import CrawlRequest
from src.services.dependencies import CrawlResponse
from src.services.dependencies import DatabaseManagerDep
from src.services.dependencies import DatabaseSessionDep
from src.services.dependencies import EmbeddingManagerDep  # Service Dependencies
from src.services.dependencies import EmbeddingRequest  # Request/Response Models
from src.services.dependencies import EmbeddingResponse
from src.services.dependencies import HyDEEngineDep
from src.services.dependencies import ProjectStorageDep
from src.services.dependencies import QdrantServiceDep
from src.services.dependencies import TaskQueueManagerDep
from src.services.dependencies import TaskRequest
from src.services.dependencies import cache_delete
from src.services.dependencies import cache_get
from src.services.dependencies import cache_set
from src.services.dependencies import cleanup_services
from src.services.dependencies import crawl_site
from src.services.dependencies import enqueue_task
from src.services.dependencies import generate_embeddings  # Service Functions
from src.services.dependencies import get_client_manager  # Core Dependency Functions
from src.services.dependencies import get_service_health  # Utility Functions
from src.services.dependencies import get_service_metrics
from src.services.dependencies import get_task_status
from src.services.dependencies import scrape_url

from .core import DependencyContainer
from .core import ServiceHealthChecker
from .core import cleanup_dependencies
from .core import database_session
from .core import get_cache_manager
from .core import get_cache_manager_legacy
from .core import get_config_dependency
from .core import get_config_dependency as get_config
from .core import get_container
from .core import get_correlation_id_dependency
from .core import get_embedding_manager
from .core import get_embedding_manager_legacy
from .core import get_fastapi_config
from .core import get_health_checker
from .core import get_request_context
from .core import get_vector_service
from .core import initialize_dependencies

__all__ = [
    "BrowserAutomationRouterDep",
    "CacheManagerDep",
    "CacheRequest",
    "ClientManagerDep",
    # New function-based dependencies
    "ConfigDep",
    "ContentIntelligenceServiceDep",
    "CrawlManagerDep",
    "CrawlRequest",
    "CrawlResponse",
    "DatabaseManagerDep",
    "DatabaseSessionDep",
    # Legacy core dependencies
    "DependencyContainer",
    "EmbeddingManagerDep",
    # Pydantic Models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "HyDEEngineDep",
    "ProjectStorageDep",
    "QdrantServiceDep",
    "ServiceHealthChecker",
    "TaskQueueManagerDep",
    "TaskRequest",
    "cache_delete",
    "cache_get",
    "cache_set",
    "cleanup_dependencies",
    "cleanup_services",
    "crawl_site",
    "database_session",
    "enqueue_task",
    # Service Functions
    "generate_embeddings",
    "get_cache_manager",
    "get_cache_manager_legacy",
    "get_client_manager",
    "get_config",
    "get_config_dependency",
    "get_container",
    "get_correlation_id_dependency",
    "get_embedding_manager",
    "get_embedding_manager_legacy",
    "get_fastapi_config",
    "get_health_checker",
    "get_request_context",
    # Utility Functions
    "get_service_health",
    "get_service_metrics",
    "get_task_status",
    "get_vector_service",
    "initialize_dependencies",
    "scrape_url",
]
