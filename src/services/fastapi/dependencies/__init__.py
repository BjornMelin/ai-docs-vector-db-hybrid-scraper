import typing


"""FastAPI dependency injection components for production environments."""

# Import existing core dependencies
# Import new function-based dependencies
from src.services.dependencies import (
    BrowserAutomationRouterDep,
    CacheManagerDep,
    CacheRequest,
    ClientManagerDep,
    ConfigDep,  # Configuration Dependencies
    ContentIntelligenceServiceDep,
    CrawlManagerDep,
    CrawlRequest,
    CrawlResponse,
    DatabaseManagerDep,
    DatabaseSessionDep,
    EmbeddingManagerDep,  # Service Dependencies
    EmbeddingRequest,  # Request/Response Models
    EmbeddingResponse,
    HyDEEngineDep,
    ProjectStorageDep,
    QdrantServiceDep,
    TaskQueueManagerDep,
    TaskRequest,
    cache_delete,
    cache_get,
    cache_set,
    cleanup_services,
    crawl_site,
    enqueue_task,
    generate_embeddings,  # Service Functions
    get_client_manager,  # Core Dependency Functions
    get_service_health,  # Utility Functions
    get_service_metrics,
    get_task_status,
    scrape_url,
)

from .core import (
    DependencyContainer,
    ServiceHealthChecker,
    cleanup_dependencies,
    database_session,
    get_cache_manager,
    get_cache_manager_legacy,
    get_config_dependency,
    get_config_dependency as get_config,
    get_container,
    get_correlation_id_dependency,
    get_embedding_manager,
    get_embedding_manager_legacy,
    get_fastapi_config,
    get_health_checker,
    get_request_context,
    get_vector_service,
    initialize_dependencies,
)


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
