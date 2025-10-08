"""Client coordination layer using function-based dependencies."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, cast

from dependency_injector.wiring import Provide, inject
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from src.config import Settings, get_settings
from src.config.models import CacheType, MCPClientConfig, MCPServerConfig, MCPTransport
from src.infrastructure.clients import (
    FirecrawlClientProvider,
    HTTPClientProvider,
    OpenAIClientProvider,
    QdrantClientProvider,
    RedisClientProvider,
)
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.circuit_breaker import CircuitBreakerManager
from src.services.core.project_storage import ProjectStorage
from src.services.embeddings.fastembed_provider import FastEmbedProvider


if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from src.services.cache.manager import CacheManager
    from src.services.content_intelligence.service import ContentIntelligenceService
    from src.services.managers.crawling_manager import CrawlingManager
    from src.services.managers.database_manager import DatabaseManager
    from src.services.managers.embedding_manager import EmbeddingManager
    from src.services.rag.generator import RAGGenerator
    from src.services.vector_db.service import VectorStoreService
else:  # pragma: no cover - runtime fallbacks keep optional extras optional
    CacheManager = Any  # type: ignore[assignment]
    ContentIntelligenceService = Any  # type: ignore[assignment]
    CrawlingManager = Any  # type: ignore[assignment]
    DatabaseManager = Any  # type: ignore[assignment]
    EmbeddingManager = Any  # type: ignore[assignment]


HealthStatusCallable = Callable[[], Awaitable[dict[str, dict[str, Any]]]]
OverallHealthCallable = Callable[[], Awaitable[dict[str, Any]]]


logger = logging.getLogger(__name__)

ProviderKey = Literal["firecrawl", "http", "openai", "qdrant", "redis"]
_PROVIDER_ERROR_MESSAGES: Final[dict[ProviderKey, str]] = {
    "openai": "OpenAI client provider not available",
    "qdrant": "Qdrant client provider not available",
    "redis": "Redis client provider not available",
    "firecrawl": "Firecrawl client provider not available",
    "http": "HTTP client provider not available",
}


def _load_automation_router() -> type[Any]:
    """Dynamically import the automation router implementation."""

    try:
        module = import_module("src.services.browser.router")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        msg = "Automation router is unavailable; browser features are disabled"
        raise RuntimeError(msg) from exc
    except ImportError as exc:  # pragma: no cover - defensive
        msg = "Failed to import automation router"
        raise RuntimeError(msg) from exc

    try:
        router_cls = module.AutomationRouter
    except AttributeError as exc:  # pragma: no cover - defensive
        msg = "Automation router module does not define AutomationRouter"
        raise RuntimeError(msg) from exc

    return cast(type[Any], router_cls)


_API_ERROR_TYPE: type[Exception] | None = None
_HEALTH_FUNCS: (
    tuple[HealthStatusCallable | None, OverallHealthCallable | None] | None
) = None

_OPTIONAL_IMPORT_CACHE: dict[str, type[Any]] = {}


def _import_optional_class(
    module_path: str, attribute: str, feature_name: str
) -> type[Any]:
    """Import an optional service class lazily with helpful error messages."""

    cache_key = f"{module_path}:{attribute}"
    cached = _OPTIONAL_IMPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        msg = (
            f"{feature_name} requires optional dependency '{module_path}'. "
            "Install the corresponding extras or disable the feature."
        )
        raise RuntimeError(msg) from exc
    except ImportError as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected import error when loading %s from %s", attribute, module_path
        )
        msg = f"Failed to import {feature_name}; see logs for details."
        raise RuntimeError(msg) from exc

    try:
        resolved = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        msg = f"{feature_name} module '{module_path}' does not define '{attribute}'."
        raise RuntimeError(msg) from exc

    resolved_cls = cast(type[Any], resolved)
    _OPTIONAL_IMPORT_CACHE[cache_key] = resolved_cls
    return resolved_cls


def _resolve_cache_manager_class() -> type[Any]:
    return _import_optional_class(
        "src.services.cache.manager", "CacheManager", "Cache manager"
    )


def _resolve_embedding_manager_class() -> type[Any]:
    return _import_optional_class(
        "src.services.managers.embedding_manager",
        "EmbeddingManager",
        "Embedding manager",
    )


def _resolve_crawling_manager_class() -> type[Any]:
    return _import_optional_class(
        "src.services.managers.crawling_manager",
        "CrawlingManager",
        "Crawling manager",
    )


def _resolve_content_intelligence_service_class() -> type[Any]:
    return _import_optional_class(
        "src.services.content_intelligence.service",
        "ContentIntelligenceService",
        "Content intelligence service",
    )


def _resolve_database_manager_class() -> type[Any]:
    return _import_optional_class(
        "src.services.managers.database_manager",
        "DatabaseManager",
        "Database manager",
    )


def _resolve_api_error_type() -> type[Exception]:
    """Return the service-layer APIError type without creating import cycles."""

    global _API_ERROR_TYPE  # pylint: disable=global-statement
    if _API_ERROR_TYPE is None:
        module = import_module("src.services.errors")
        _API_ERROR_TYPE = cast(type[Exception], module.APIError)
    return _API_ERROR_TYPE


def _raise_api_error(message: str) -> NoReturn:
    """Dynamically raise APIError without importing services.errors at module scope."""

    error_type = _resolve_api_error_type()
    raise error_type(message)


def _resolve_health_dependencies() -> tuple[
    HealthStatusCallable | None, OverallHealthCallable | None
]:
    """Dynamically resolve health dependency functions to avoid import cycles."""

    global _HEALTH_FUNCS  # pylint: disable=global-statement
    if _HEALTH_FUNCS is not None:
        return _HEALTH_FUNCS

    try:
        module = import_module("src.services.dependencies")
    except ModuleNotFoundError:
        return (None, None)
    except ImportError:
        logger.exception(
            "Failed to import health dependencies from src.services.dependencies"
        )
        return (None, None)

    status_func = getattr(module, "get_health_status", None)
    overall_func = getattr(module, "get_overall_health", None)

    resolved_status = (
        cast(HealthStatusCallable | None, status_func)
        if callable(status_func)
        else None
    )
    resolved_overall = (
        cast(OverallHealthCallable | None, overall_func)
        if callable(overall_func)
        else None
    )

    if resolved_status is None and resolved_overall is None:
        return (None, None)

    _HEALTH_FUNCS = (resolved_status, resolved_overall)
    return _HEALTH_FUNCS


class ClientManager:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """Client coordination layer using function-based dependencies."""

    _instance: ClientManager | None = None
    _lock = asyncio.Lock()
    _init_lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton instance with thread safety."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize client manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Provider compatibility layer
        self._providers: dict[ProviderKey, Any] = {}
        self._parallel_processing_system: Any | None = None
        self._initialized = False
        self._vector_store_service: VectorStoreService | None = None
        self._cache_manager: CacheManager | None = None
        self._embedding_manager: EmbeddingManager | None = None
        self._crawl_manager: CrawlingManager | None = None
        self._content_intelligence: ContentIntelligenceService | None = None
        self._project_storage: ProjectStorage | None = None
        self._rag_generator: RAGGenerator | None = None
        self._database_manager: DatabaseManager | None = None
        self._circuit_breaker_manager: CircuitBreakerManager | None = None
        self._config = get_settings()
        self._automation_router: Any | None = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_client_lock = asyncio.Lock()

    @property
    def config(self) -> Settings:
        """Return the lazily loaded application configuration."""

        return self._config

    @inject
    def initialize_providers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        openai_provider: OpenAIClientProvider = Provide[
            ApplicationContainer.openai_provider
        ],
        qdrant_provider: QdrantClientProvider = Provide[
            ApplicationContainer.qdrant_provider
        ],
        redis_provider: RedisClientProvider = Provide[
            ApplicationContainer.redis_provider
        ],
        firecrawl_provider: FirecrawlClientProvider = Provide[
            ApplicationContainer.firecrawl_provider
        ],
        http_provider: HTTPClientProvider = Provide[ApplicationContainer.http_provider],
    ) -> None:
        """Initialize client providers using dependency injection."""
        self._providers = {
            "openai": openai_provider,
            "qdrant": qdrant_provider,
            "redis": redis_provider,
            "firecrawl": firecrawl_provider,
            "http": http_provider,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if the client manager is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize client manager with function-based dependencies."""
        async with self._lock:
            if self._initialized:
                return
            container = get_container()
            if container:
                container.wire(modules=[__name__])
                self.initialize_providers()
            await self._initialize_parallel_processing_system()
            self._initialized = True
            logger.info("ClientManager initialized with function-based dependencies")

    async def _initialize_parallel_processing_system(self) -> None:
        """Initialize the parallel processing system using dependency injection."""

        try:
            container = get_container()
            if container:
                # Get the parallel processing system from the container
                # Note: embedding manager is now accessed via
                # function-based dependencies
                self._parallel_processing_system = (
                    container.parallel_processing_system()
                )
                logger.info("Parallel processing system initialized")
            else:
                logger.warning(
                    "Cannot initialize parallel processing system: "
                    "container not available"
                )
        except (ImportError, AttributeError, RuntimeError):
            logger.exception("Failed to initialize parallel processing system")
            # Continue without parallel processing
            self._parallel_processing_system = None

    async def cleanup(self) -> None:
        """Cleanup resources (function-based dependencies are stateless)."""

        if self._vector_store_service:
            try:
                await self._vector_store_service.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup vector store service")
            finally:
                self._vector_store_service = None
        if self._rag_generator:
            try:
                await self._rag_generator.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup RAG generator")
            self._rag_generator = None
        if self._automation_router is not None:
            try:
                await self._automation_router.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup automation router")
            self._automation_router = None
        if self._database_manager is not None:
            try:
                await self._database_manager.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup database manager")
            self._database_manager = None
        if self._embedding_manager is not None:
            try:
                await self._embedding_manager.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup embedding manager")
            self._embedding_manager = None
        if self._cache_manager is not None:
            try:
                await self._cache_manager.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to close cache manager")
            self._cache_manager = None
        if self._crawl_manager is not None:
            try:
                await self._crawl_manager.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup crawl manager")
            self._crawl_manager = None
        if self._content_intelligence is not None:
            try:
                await self._content_intelligence.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup content intelligence service")
            self._content_intelligence = None
        if self._project_storage is not None:
            try:
                await self._project_storage.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to cleanup project storage")
            self._project_storage = None
        if self._circuit_breaker_manager is not None:
            try:
                await self._circuit_breaker_manager.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to close circuit breaker manager")
            self._circuit_breaker_manager = None
        await self._close_mcp_client()
        self._providers.clear()
        self._parallel_processing_system = None
        self._initialized = False
        logger.info("ClientManager cleaned up")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes."""

        with cls._init_lock:
            cls._instance = None
        global _API_ERROR_TYPE, _HEALTH_FUNCS  # pylint: disable=global-statement
        _API_ERROR_TYPE = None
        _HEALTH_FUNCS = None

    def _get_provider_client(self, key: ProviderKey) -> Any:
        provider = self._providers.get(key)
        if provider is None:
            _raise_api_error(_PROVIDER_ERROR_MESSAGES[key])
        return provider.client

    async def _close_mcp_client(self) -> None:
        """Close the MCP client if it exists."""

        async with self._mcp_client_lock:
            client = self._mcp_client
            if client is None:
                return

            close_async = getattr(client, "aclose", None)
            if close_async is not None:
                async_closer = cast(Callable[[], Awaitable[Any]], close_async)
                await async_closer()
            else:
                close_sync = getattr(client, "close", None)
                if close_sync is not None:
                    sync_closer = cast(Callable[[], Any], close_sync)
                    sync_closer()

            self._mcp_client = None

    async def get_openai_client(self):
        """Return the OpenAI client."""

        return self._get_provider_client("openai")

    async def get_qdrant_client(self):
        """Return the Qdrant client."""

        return self._get_provider_client("qdrant")

    async def get_mcp_client(self) -> MultiServerMCPClient:
        """Return a cached MultiServerMCPClient built from configuration."""

        async with self._mcp_client_lock:
            if self._mcp_client is not None:
                return self._mcp_client

            config = getattr(self._config, "mcp_client", None)
            if not isinstance(config, MCPClientConfig) or not config.enabled:
                msg = "MCP client integration is disabled"
                _raise_api_error(msg)
            if not config.servers:
                msg = "No MCP servers configured"
                _raise_api_error(msg)

            connections = self._build_mcp_connections(config)
            self._mcp_client = MultiServerMCPClient(connections)
            return self._mcp_client

    def _build_mcp_connections(self, config: MCPClientConfig) -> dict[str, Connection]:
        """Translate MCP client config to connection mappings."""

        connections: dict[str, Connection] = {}
        for server in config.servers:
            connections[server.name] = self._serialise_mcp_server(server, config)
        return connections

    @staticmethod
    def _serialise_mcp_server(
        server: MCPServerConfig, config: MCPClientConfig
    ) -> Connection:
        """Return serialised configuration for a single MCP server."""

        timeout_ms = (
            server.timeout_ms
            if server.timeout_ms is not None
            else config.request_timeout_ms
        )
        timeout_seconds = timeout_ms / 1000.0

        if server.transport == MCPTransport.STDIO:
            payload = {
                "transport": "stdio",
                "command": server.command,
                "args": list(server.args),
            }
            if server.env:
                payload["env"] = dict(server.env)
            return cast(Connection, payload)

        if server.transport == MCPTransport.STREAMABLE_HTTP:
            payload = {
                "transport": "streamable_http",
                "url": str(server.url),
                "timeout": timeout_seconds,
            }
            if server.headers:
                payload["headers"] = dict(server.headers)
            return cast(Connection, payload)

        payload = {
            "transport": "sse",
            "url": str(server.url),
            "timeout": timeout_seconds,
            "sse_read_timeout": timeout_seconds,
        }
        if server.headers:
            payload["headers"] = dict(server.headers)
        return cast(Connection, payload)

    async def get_vector_store_service(self) -> VectorStoreService:
        """Return the vector store service instance."""

        if self._vector_store_service:
            return self._vector_store_service

        model_name = getattr(self._config.fastembed, "model", "BAAI/bge-small-en-v1.5")
        provider = FastEmbedProvider(model_name=model_name)
        vector_module = import_module("src.services.vector_db.service")
        vector_cls = vector_module.VectorStoreService
        service = cast(
            VectorStoreService,
            vector_cls(
                config=self._config, client_manager=self, embeddings_provider=provider
            ),
        )
        await service.initialize()
        self._vector_store_service = service
        return service

    async def get_cache_manager(self) -> CacheManager:
        """Return the cache manager configured for the application."""

        cache_manager = self._cache_manager
        if cache_manager is not None:
            return cache_manager

        cache_config = self._config.cache
        distributed_ttl_seconds = {
            CacheType.REDIS: max(
                cache_config.ttl_embeddings,
                cache_config.ttl_search_results,
            ),
            CacheType.EMBEDDINGS: cache_config.ttl_embeddings,
            CacheType.SEARCH: cache_config.ttl_search_results,
            CacheType.CRAWL: cache_config.ttl_crawl,
            CacheType.HYBRID: max(
                cache_config.ttl_embeddings,
                cache_config.ttl_search_results,
            ),
        }

        cache_manager_cls = _resolve_cache_manager_class()
        cache_manager = cast(
            CacheManager,
            cache_manager_cls(
                dragonfly_url=cache_config.redis_url,
                enable_local_cache=cache_config.enable_local_cache,
                enable_distributed_cache=cache_config.enable_redis_cache,
                local_max_size=cache_config.local_max_size,
                local_max_memory_mb=float(cache_config.local_max_memory_mb),
                distributed_ttl_seconds=distributed_ttl_seconds,
                local_cache_path=self._config.cache_dir / "services",
                memory_pressure_threshold=cache_config.memory_pressure_threshold,
            ),
        )
        self._cache_manager = cache_manager
        return cache_manager

    async def get_embedding_manager(self) -> EmbeddingManager:
        """Return the embedding manager, initializing it on first use."""

        embedding_manager = self._embedding_manager
        if embedding_manager is not None:
            return embedding_manager

        manager_cls = _resolve_embedding_manager_class()
        manager = cast(EmbeddingManager, manager_cls())
        await manager.initialize(config=self._config, client_manager=self)
        self._embedding_manager = manager
        return manager

    async def get_crawl_manager(self) -> CrawlingManager:
        """Return the crawling manager, creating it as needed."""

        crawling_manager = self._crawl_manager
        if crawling_manager is not None:
            return crawling_manager

        manager_cls = _resolve_crawling_manager_class()
        manager = cast(CrawlingManager, manager_cls())
        await manager.initialize(config=self._config)
        self._crawl_manager = manager
        return manager

    async def get_crawling_manager(self) -> CrawlingManager:
        """Alias for get_crawl_manager for compatibility with new naming."""

        return await self.get_crawl_manager()

    async def get_content_intelligence_service(
        self,
    ) -> ContentIntelligenceService:
        """Return the content intelligence service instance."""

        content_intelligence = self._content_intelligence
        if content_intelligence is not None:
            return content_intelligence

        embedding_manager = await self.get_embedding_manager()
        cache_manager = await self.get_cache_manager()
        service_cls = _resolve_content_intelligence_service_class()
        service = cast(
            ContentIntelligenceService,
            service_cls(
                config=self._config,
                embedding_manager=embedding_manager,
                cache_manager=cache_manager,
            ),
        )
        await service.initialize()
        self._content_intelligence = service
        return service

    async def get_project_storage(self) -> ProjectStorage:
        """Return the project storage service."""

        project_storage = self._project_storage
        if project_storage is not None:
            return project_storage

        data_dir = getattr(self._config, "data_dir", None)
        if data_dir is None:
            msg = "Configuration missing data_dir for project storage"
            raise RuntimeError(msg)

        storage = ProjectStorage(data_dir=Path(data_dir))
        await storage.initialize()
        self._project_storage = storage
        return storage

    async def get_database_manager(self) -> DatabaseManager:
        """Return the database manager composed from shared services."""

        database_manager = self._database_manager
        if database_manager is not None:
            return database_manager

        redis_client = await self.get_redis_client()
        cache_manager = await self.get_cache_manager()
        vector_service = await self.get_vector_store_service()
        manager_cls = _resolve_database_manager_class()
        manager = cast(
            DatabaseManager,
            manager_cls(
                redis_client=redis_client,
                cache_manager=cache_manager,
                vector_service=vector_service,
            ),
        )
        await manager.initialize()
        self._database_manager = manager
        return manager

    async def get_circuit_breaker_manager(self) -> CircuitBreakerManager:
        """Return the circuit breaker manager instance."""

        circuit_manager = self._circuit_breaker_manager
        if circuit_manager is not None:
            return circuit_manager

        manager = CircuitBreakerManager(
            redis_url=self._config.cache.redis_url,
            config=self._config,
        )
        self._circuit_breaker_manager = manager
        return manager

    async def get_rag_generator(self) -> RAGGenerator:
        if self._rag_generator:
            return self._rag_generator

        from src.services.rag.generator import (  # pylint: disable=import-outside-toplevel
            RAGGenerator,
        )
        from src.services.rag.models import (  # pylint: disable=import-outside-toplevel
            RAGConfig as ServiceRAGConfig,
        )
        from src.services.rag.retriever import (  # pylint: disable=import-outside-toplevel
            VectorServiceRetriever,
        )

        vector_service = await self.get_vector_store_service()
        rag_config_model = getattr(self._config, "rag", None)
        rag_config = ServiceRAGConfig.model_validate(
            rag_config_model.model_dump() if rag_config_model is not None else {}
        )

        collection_name = getattr(
            getattr(self._config, "qdrant", None),
            "collection_name",
            "documents",
        )

        retriever = VectorServiceRetriever(
            vector_service=vector_service,
            collection=collection_name,
            k=rag_config.retriever_top_k,
            rag_config=rag_config,
        )

        generator = RAGGenerator(rag_config, retriever)
        await generator.initialize()
        self._rag_generator = generator
        return generator

    async def get_redis_client(self):
        return self._get_provider_client("redis")

    async def get_firecrawl_client(self):
        return self._get_provider_client("firecrawl")

    async def get_http_client(self):
        return self._get_provider_client("http")

    # Function-based dependency access methods (backward compatibility)

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status using function-based dependencies."""
        status_callable, _ = _resolve_health_dependencies()
        if status_callable:
            return await status_callable()
        logger.warning(
            "Health status monitoring not available - function-based dependency "
            "not found"
        )
        return {}

    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall health using function-based dependencies."""
        _, overall_callable = _resolve_health_dependencies()
        if overall_callable:
            return await overall_callable()
        return {
            "overall_healthy": False,
            "error": "Health monitoring not available",
        }

    async def get_service_status(self) -> dict[str, Any]:
        """Get service status using function-based dependencies."""
        return {
            "initialized": self._initialized,
            "mode": "function_based_dependencies",
            "providers": list(self._providers.keys()),
            "parallel_processing": self._parallel_processing_system is not None,
            "note": "Using function-based dependencies instead of Manager classes",
        }

    async def get_parallel_processing_system(self):
        """Get parallel processing system instance."""
        return self._parallel_processing_system

    async def get_browser_automation_router(self) -> Any:
        """Get browser automation router for intelligent scraping."""

        if self._automation_router is None:
            router_cls = _load_automation_router()
            router = router_cls(self._config)
            await router.initialize()
            self._automation_router = router
        return self._automation_router

    def get_browser_automation_metrics(self) -> dict[str, dict[str, int]]:
        """Return router metrics if the router has been initialized."""

        if self._automation_router is None:
            return {}
        return self._automation_router.get_metrics_snapshot()

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
            "http": self.get_http_client,
            "parallel_processing": self.get_parallel_processing_system,
        }
        if client_type not in getters:
            msg = (
                f"Unknown client type: {client_type}. Available: {list(getters.keys())}"
            )
            raise ValueError(msg)
        api_error = _resolve_api_error_type()
        try:
            yield await getters[client_type]()
        except (
            ConnectionError,
            TimeoutError,
            api_error,
            ValueError,
            RuntimeError,
        ):
            logger.exception("Error using %s client", client_type)
            raise

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    @classmethod
    def from_unified_config(cls) -> ClientManager:
        """Create ClientManager instance from unified config.

        Used by function-based dependencies for singleton pattern.
        """
        return cls()


_GLOBAL_CLIENT_MANAGER: ClientManager | None = None
_GLOBAL_CLIENT_LOCK = asyncio.Lock()


async def ensure_client_manager(force: bool = False) -> ClientManager:
    """Return an initialized global ClientManager instance."""

    global _GLOBAL_CLIENT_MANAGER  # pylint: disable=global-statement
    if _GLOBAL_CLIENT_MANAGER is not None and not force:
        return _GLOBAL_CLIENT_MANAGER

    async with _GLOBAL_CLIENT_LOCK:
        if _GLOBAL_CLIENT_MANAGER is None or force:
            if _GLOBAL_CLIENT_MANAGER is not None:
                await _GLOBAL_CLIENT_MANAGER.cleanup()
            manager = ClientManager.from_unified_config()
            await manager.initialize()
            _GLOBAL_CLIENT_MANAGER = manager
    return _GLOBAL_CLIENT_MANAGER


def get_client_manager() -> ClientManager:
    """Return the initialized global ClientManager.

    Raises:
        RuntimeError: If the client manager has not been initialized yet.
    """

    if _GLOBAL_CLIENT_MANAGER is None:
        msg = "ClientManager has not been initialized"
        raise RuntimeError(msg)
    return _GLOBAL_CLIENT_MANAGER


async def shutdown_client_manager() -> None:
    """Shutdown the global ClientManager and release resources."""

    global _GLOBAL_CLIENT_MANAGER  # pylint: disable=global-statement
    if _GLOBAL_CLIENT_MANAGER is None:
        return
    await _GLOBAL_CLIENT_MANAGER.cleanup()
    _GLOBAL_CLIENT_MANAGER = None
