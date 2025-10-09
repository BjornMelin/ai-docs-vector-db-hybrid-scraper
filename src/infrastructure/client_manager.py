# pylint: disable=too-many-lines
"""Client coordination layer using function-based dependencies."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, cast

from dependency_injector.wiring import (  # type: ignore[reportMissingImports]
    Provide,
    inject,
)
from langchain_mcp_adapters.client import (  # type: ignore[reportMissingImports]
    MultiServerMCPClient,
)
from langchain_mcp_adapters.sessions import (  # type: ignore[reportMissingImports]
    Connection,
)

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
from src.services.health.manager import (
    HealthCheck,
    HealthCheckManager,
    HealthStatus,
    build_health_manager,
)
from src.services.observability.performance import (
    get_operation_statistics,
    get_performance_monitor,
    get_system_performance_summary,
    monitor_operation,
)
from src.services.vector_db.types import VectorRecord


if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from redis.asyncio import Redis  # type: ignore[reportMissingImports]

    from src.services.browser.unified_manager import UnifiedBrowserManager
    from src.services.cache.manager import CacheManager
    from src.services.content_intelligence.service import ContentIntelligenceService
    from src.services.embeddings.manager import EmbeddingManager as CoreEmbeddingManager
    from src.services.rag.generator import RAGGenerator
    from src.services.vector_db.service import VectorStoreService
else:  # pragma: no cover - runtime fallbacks keep optional extras optional
    CacheManager = Any  # type: ignore[assignment]
    ContentIntelligenceService = Any  # type: ignore[assignment]
    CoreEmbeddingManager = Any  # type: ignore[assignment]
    UnifiedBrowserManager = Any  # type: ignore[assignment]
    Redis = Any  # type: ignore[assignment]


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


def _resolve_content_intelligence_service_class() -> type[Any]:
    return _import_optional_class(
        "src.services.content_intelligence.service",
        "ContentIntelligenceService",
        "Content intelligence service",
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


@dataclass(slots=True)
class DatabaseSessionContext:
    """Lightweight view of database resources exposed to callers."""

    redis: Redis | None
    cache_manager: CacheManager | None
    vector_service: VectorStoreService | None


class _BasicSpan:
    """Simple span used when OpenTelemetry is unavailable."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._start_time = time.time()

    def __enter__(self) -> _BasicSpan:
        logger.debug("Starting span: %s", self._name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        duration_ms = (time.time() - self._start_time) * 1000
        logger.debug("Completed span: %s in %.2fms", self._name, duration_ms)
        if exc_type is not None:
            logger.error("Span %s failed: %s", self._name, exc_val)


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
        self._embedding_manager: CoreEmbeddingManager | None = None
        self._crawl_manager: UnifiedBrowserManager | None = None
        self._content_intelligence: ContentIntelligenceService | None = None
        self._project_storage: ProjectStorage | None = None
        self._rag_generator: RAGGenerator | None = None
        self._circuit_breaker_manager: CircuitBreakerManager | None = None
        self._config = get_settings()
        self._automation_router: Any | None = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_client_lock = asyncio.Lock()
        self._monitoring_initialized = False
        self._health_manager: HealthCheckManager | None = None
        self._metrics_registry: Any | None = None

    @property
    def config(self) -> Settings:
        """Return the lazily loaded application configuration."""

        return self._config

    @property
    def _cleanup_plan(self) -> tuple[tuple[str, str, str], ...]:
        """Return the ordered list of resources to cleanup."""

        return (
            (
                "_vector_store_service",
                "cleanup",
                "Failed to cleanup vector store service",
            ),
            ("_rag_generator", "cleanup", "Failed to cleanup RAG generator"),
            (
                "_automation_router",
                "cleanup",
                "Failed to cleanup automation router",
            ),
            (
                "_embedding_manager",
                "cleanup",
                "Failed to cleanup embedding manager",
            ),
            ("_cache_manager", "close", "Failed to close cache manager"),
            ("_crawl_manager", "cleanup", "Failed to cleanup crawl manager"),
            (
                "_content_intelligence",
                "cleanup",
                "Failed to cleanup content intelligence service",
            ),
            (
                "_project_storage",
                "cleanup",
                "Failed to cleanup project storage",
            ),
            (
                "_circuit_breaker_manager",
                "close",
                "Failed to close circuit breaker manager",
            ),
        )

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

        for attr_name, method_name, message in self._cleanup_plan:
            await self._cleanup_resource(attr_name, method_name, message)
        await self._shutdown_monitoring()
        await self._close_mcp_client()
        self._providers.clear()
        self._parallel_processing_system = None
        self._initialized = False
        logger.info("ClientManager cleaned up")

    async def _cleanup_resource(
        self, attr_name: str, method_name: str, error_message: str
    ) -> None:
        """Safely cleanup a lazily cached resource.

        Args:
            attr_name: Name of the attribute storing the resource.
            method_name: Cleanup or close method to invoke when available.
            error_message: Message used if cleanup raises.
        """

        resource = getattr(self, attr_name, None)
        if resource is None:
            return

        cleanup_callable = getattr(resource, method_name, None)
        if cleanup_callable is None:
            logger.debug(
                "Resource %s has no %s method; skipping cleanup", attr_name, method_name
            )
            setattr(self, attr_name, None)
            return

        try:
            result = cleanup_callable()
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                await result
        except Exception:  # pragma: no cover - defensive
            logger.exception(error_message)
        finally:
            setattr(self, attr_name, None)

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes."""

        with cls._init_lock:
            cls._instance = None
        global _API_ERROR_TYPE  # pylint: disable=global-statement
        _API_ERROR_TYPE = None

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

    async def get_embedding_manager(self) -> CoreEmbeddingManager:
        """Return the embedding manager, initializing it on first use."""

        embedding_manager = self._embedding_manager
        if embedding_manager is not None:
            return embedding_manager

        from src.services.embeddings.manager import (  # pylint: disable=import-outside-toplevel
            EmbeddingManager as CoreEmbeddingManager,
        )

        manager = CoreEmbeddingManager(config=self._config, client_manager=self)
        await manager.initialize()
        self._embedding_manager = manager
        return manager

    async def get_crawl_manager(self) -> UnifiedBrowserManager:
        """Return the crawling manager, creating it as needed."""

        crawling_manager = self._crawl_manager
        if crawling_manager is not None:
            return crawling_manager

        from src.services.browser.unified_manager import (  # pylint: disable=import-outside-toplevel
            UnifiedBrowserManager,
        )

        manager = UnifiedBrowserManager(self._config)
        await manager.initialize()
        self._crawl_manager = manager
        return manager

    async def get_crawling_manager(self) -> UnifiedBrowserManager:
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

    async def ensure_database_ready(self) -> None:
        """Ensure shared database services are initialized."""

        await self.get_vector_store_service()
        await self.get_cache_manager()
        await self.get_redis_client()

    @asynccontextmanager
    async def database_session(self) -> AsyncIterator[DatabaseSessionContext]:
        """Expose a lightweight context with database-related resources."""

        redis_client = cast("Redis | None", await self.get_redis_client())
        cache_manager = await self.get_cache_manager()
        vector_service = await self.get_vector_store_service()
        context = DatabaseSessionContext(
            redis=redis_client,
            cache_manager=cache_manager,
            vector_service=vector_service,
        )
        yield context

    async def list_vector_collections(self) -> list[str]:
        """Return the available vector store collections."""

        vector_service = await self.get_vector_store_service()
        try:
            return await vector_service.list_collections()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to list vector collections")
            _raise_api_error(f"Failed to get collections: {exc}")

    async def upsert_vector_records(
        self, collection_name: str, points: list[dict[str, Any]]
    ) -> bool:
        """Store embeddings in the specified vector collection."""

        vector_service = await self.get_vector_store_service()
        try:
            records = [
                VectorRecord(
                    id=str(point.get("id")),
                    vector=list(point["vector"]),
                    payload=point.get("payload"),
                    sparse_vector=point.get("sparse_vector"),
                )
                for point in points
                if "vector" in point
            ]
        except KeyError as exc:
            msg = "Each point must include a dense vector"
            raise ValueError(msg) from exc

        try:
            await vector_service.upsert_vectors(collection_name, records)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to upsert vector records")
            _raise_api_error(f"Failed to store embeddings: {exc}")
        return True

    async def search_vector_records(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using the shared vector service."""

        vector_service = await self.get_vector_store_service()
        try:
            matches = await vector_service.search_vector(
                collection_name,
                query_vector,
                limit=limit,
                filters=filter_conditions,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to search vector records")
            _raise_api_error(f"Failed to search vectors: {exc}")

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": dict(match.metadata or {}),
            }
            for match in matches
        ]

    async def get_database_status(self) -> dict[str, Any]:
        """Return status information for shared database components."""

        status = {
            "vector_store": {"available": False, "collections": []},
            "redis": {"available": False, "connected": False},
            "cache": {"available": False},
        }

        try:
            vector_service = await self.get_vector_store_service()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Vector store service unavailable")
        else:
            status["vector_store"]["available"] = True
            try:
                status["vector_store"][
                    "collections"
                ] = await vector_service.list_collections()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to enumerate vector collections")

        try:
            cache_manager = await self.get_cache_manager()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Cache manager unavailable")
        else:
            status["cache"]["available"] = cache_manager is not None
            if cache_manager is not None:
                try:
                    status["cache"]["stats"] = await cache_manager.get_stats()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to collect cache statistics")

        try:
            redis_client = await self.get_redis_client()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Redis client unavailable")
            redis_client = None

        status["redis"]["available"] = redis_client is not None
        if redis_client is not None:
            try:
                await redis_client.ping()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Redis ping failed")
            else:
                status["redis"]["connected"] = True

        return status

    def _ensure_monitoring_ready(self) -> None:
        """Initialize monitoring helpers lazily."""

        if self._monitoring_initialized:
            return

        try:
            from src.services.monitoring.metrics import (  # pylint: disable=import-outside-toplevel
                get_metrics_registry,
            )
        except ImportError:
            logger.warning("Monitoring metrics registry not available")
            self._metrics_registry = None
        else:
            self._metrics_registry = get_metrics_registry()

        try:
            self._health_manager = build_health_manager(
                self._config,
                metrics_registry=self._metrics_registry,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to initialize health manager")
            self._health_manager = None

        self._monitoring_initialized = True

    def _get_health_manager(self) -> HealthCheckManager:
        """Return the lazily initialised health manager."""

        self._ensure_monitoring_ready()
        if self._health_manager is None:
            msg = "Health monitoring is disabled"
            raise RuntimeError(msg)
        return self._health_manager

    def get_health_manager(self) -> HealthCheckManager:
        """Expose the configured health manager instance."""

        return self._get_health_manager()

    async def _shutdown_monitoring(self) -> None:
        """Cleanup monitoring resources."""

        try:
            monitor = get_performance_monitor()
            await monitor.cleanup()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to cleanup performance monitor", exc_info=True)

        self._metrics_registry = None
        self._health_manager = None
        self._monitoring_initialized = False

    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register an additional health check with the central manager."""

        manager = self._get_health_manager()
        manager.add_health_check(health_check)
        logger.info("Registered health check for %s", health_check.name)

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

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Return health status for all registered services."""

        manager = self._get_health_manager()
        results = await manager.check_all()
        status: dict[str, dict[str, Any]] = {}
        for name, result in results.items():
            status[name] = {
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp,
                "duration_ms": result.duration_ms,
                "metadata": result.metadata,
                "is_healthy": result.status == HealthStatus.HEALTHY,
            }
        return status

    async def get_overall_health(self) -> dict[str, Any]:
        """Aggregate health summary across registered services."""

        manager = self._get_health_manager()
        return await manager.get_overall_health()

    async def get_service_status(self) -> dict[str, Any]:
        """Summarize the client manager service state."""

        return {
            "initialized": self._initialized,
            "providers": list(self._providers.keys()),
            "parallel_processing": self._parallel_processing_system is not None,
            "monitoring_initialized": self._monitoring_initialized,
        }

    def record_metric(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a metric value via the monitoring registry."""

        self._ensure_monitoring_ready()
        if self._metrics_registry is None:
            return
        try:
            if hasattr(self._metrics_registry, "record_metric"):
                self._metrics_registry.record_metric(metric_name, value, labels or {})
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to record metric %s", metric_name, exc_info=True)

    def increment_counter(
        self, counter_name: str, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""

        self._ensure_monitoring_ready()
        if self._metrics_registry is None:
            return
        try:
            if hasattr(self._metrics_registry, "increment_counter"):
                self._metrics_registry.increment_counter(counter_name, labels or {})
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Failed to increment counter %s", counter_name, exc_info=True
            )

    def record_histogram(
        self, histogram_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram value."""

        self._ensure_monitoring_ready()
        if self._metrics_registry is None:
            return
        try:
            if hasattr(self._metrics_registry, "record_histogram"):
                self._metrics_registry.record_histogram(
                    histogram_name, value, labels or {}
                )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Failed to record histogram %s", histogram_name, exc_info=True
            )

    async def track_performance(
        self,
        operation_name: str,
        operation_func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Track performance of an asynchronous operation."""

        start_time = time.perf_counter()
        with monitor_operation(operation_name, metadata={"source": "client_manager"}):
            try:
                result = await operation_func(*args, **kwargs)
            except Exception:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.record_histogram(f"{operation_name}_duration_ms", duration_ms)
                self.increment_counter(f"{operation_name}_total", {"status": "error"})
                self.increment_counter(f"{operation_name}_errors")
                logger.exception("Operation %s failed", operation_name)
                raise
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.record_histogram(f"{operation_name}_duration_ms", duration_ms)
            self.increment_counter(f"{operation_name}_total", {"status": "success"})
            return result

    def get_performance_metrics(self) -> dict[str, Any]:
        """Return captured performance metrics."""

        self._ensure_monitoring_ready()
        summary = get_operation_statistics()
        system_summary = get_system_performance_summary()
        metrics: dict[str, Any] = {"operations": summary}
        if system_summary:
            metrics["system"] = system_summary
        return metrics

    async def log_operation(
        self,
        operation: str,
        details: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        """Log an operation with structured details."""

        log_data = {"operation": operation, "timestamp": time.time(), **details}
        if level == "debug":
            logger.debug("Operation: %s", operation, extra=log_data)
        elif level == "info":
            logger.info("Operation: %s", operation, extra=log_data)
        elif level == "warning":
            logger.warning("Operation: %s", operation, extra=log_data)
        elif level == "error":
            logger.error("Operation: %s", operation, extra=log_data)
        else:
            logger.info("Operation: %s", operation, extra=log_data)

    def create_span(self, span_name: str) -> _BasicSpan:
        """Create a lightweight tracing span context manager."""

        return _BasicSpan(span_name)

    async def get_monitoring_status(self) -> dict[str, Any]:
        """Return monitoring subsystem status details."""

        try:
            manager = self._get_health_manager()
        except RuntimeError:
            manager = None
            overall_health: dict[str, Any] = {
                "overall_status": HealthStatus.UNKNOWN.value,
                "timestamp": time.time(),
                "checks": {},
                "healthy_count": 0,
                "total_count": 0,
            }
        else:
            overall_health = await manager.get_overall_health()

        check_names = list(manager.list_checks()) if manager is not None else []

        return {
            "initialized": self._monitoring_initialized,
            "health_checks": {
                "registered": len(check_names),
                "services": check_names,
            },
            "metrics_registry": {"available": self._metrics_registry is not None},
            "performance_monitor": {
                "available": True,
                "provider": "opentelemetry",
            },
            "overall_health": overall_health,
            "performance_metrics": self.get_performance_metrics(),
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
