"""Client coordination layer using function-based dependencies."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from dependency_injector.wiring import Provide, inject
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from src.config import Settings, get_settings
from src.config.models import MCPClientConfig, MCPServerConfig, MCPTransport
from src.infrastructure.clients import (
    FirecrawlClientProvider,
    HTTPClientProvider,
    OpenAIClientProvider,
    QdrantClientProvider,
    RedisClientProvider,
)
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import APIError
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from src.services.cache.manager import CacheManager
    from src.services.content_intelligence.service import ContentIntelligenceService
    from src.services.core.project_storage import ProjectStorage
    from src.services.embeddings.manager import EmbeddingManager
    from src.services.managers.crawling_manager import CrawlingManager
    from src.services.rag.generator import RAGGenerator
else:  # pragma: no cover - runtime fallback for optional services
    CacheManager = ContentIntelligenceService = ProjectStorage = EmbeddingManager = (
        CrawlingManager
    ) = RAGGenerator = Any


# Import dependencies for health checks
if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    HealthStatusCallable = Callable[[], Awaitable[dict[str, dict[str, Any]]]]
    OverallHealthCallable = Callable[[], Awaitable[dict[str, Any]]]
else:
    HealthStatusCallable = Callable[[], Awaitable[dict[str, dict[str, Any]]]]
    OverallHealthCallable = Callable[[], Awaitable[dict[str, Any]]]

_dependencies_module = None
try:  # pragma: no cover - optional dependency
    _dependencies_module = import_module("src.services.dependencies")
except ImportError:
    _dependencies_module = None

if _dependencies_module is not None:
    _deps_get_health_status = getattr(_dependencies_module, "get_health_status", None)
    _deps_get_overall_health = getattr(_dependencies_module, "get_overall_health", None)
else:
    _deps_get_health_status = None
    _deps_get_overall_health = None

deps_get_health_status: HealthStatusCallable | None = (
    cast(HealthStatusCallable, _deps_get_health_status)
    if _deps_get_health_status is not None
    else None
)
deps_get_overall_health: OverallHealthCallable | None = (
    cast(OverallHealthCallable, _deps_get_overall_health)
    if _deps_get_overall_health is not None
    else None
)


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


async def _ensure_service_registry():  # pragma: no cover - thin wrapper
    # pylint: disable=import-outside-toplevel
    from src.services.registry import (
        ensure_service_registry as _ensure_service_registry,
    )

    return await _ensure_service_registry()


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
        self._config = get_settings()
        self._automation_router: Any | None = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_client_lock = asyncio.Lock()
        self._service_registry: Any | None = None

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
            await self._vector_store_service.cleanup()
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
        self._cache_manager = None
        self._embedding_manager = None
        self._crawl_manager = None
        self._content_intelligence = None
        self._project_storage = None
        await self._close_mcp_client()
        self._providers.clear()
        self._parallel_processing_system = None
        self._service_registry = None
        self._initialized = False
        logger.info("ClientManager cleaned up")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._init_lock:
            cls._instance = None

    def _get_provider_client(self, key: ProviderKey) -> Any:
        provider = self._providers.get(key)
        if provider is None:
            raise APIError(_PROVIDER_ERROR_MESSAGES[key])
        return provider.client

    async def _close_mcp_client(self) -> None:
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
        return self._get_provider_client("openai")

    async def get_qdrant_client(self):
        return self._get_provider_client("qdrant")

    async def get_mcp_client(self) -> MultiServerMCPClient:
        """Return a cached MultiServerMCPClient built from configuration."""

        async with self._mcp_client_lock:
            if self._mcp_client is not None:
                return self._mcp_client

            config = getattr(self._config, "mcp_client", None)
            if not isinstance(config, MCPClientConfig) or not config.enabled:
                msg = "MCP client integration is disabled"
                raise APIError(msg)
            if not config.servers:
                msg = "No MCP servers configured"
                raise APIError(msg)

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

        timeout_ms = server.timeout_ms or config.request_timeout_ms
        timeout = timedelta(milliseconds=timeout_ms)

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
                "timeout": timeout,
            }
            if server.headers:
                payload["headers"] = dict(server.headers)
            return cast(Connection, payload)

        payload = {
            "transport": "sse",
            "url": str(server.url),
            "timeout": timeout,
            "sse_read_timeout": timeout,
        }
        if server.headers:
            payload["headers"] = dict(server.headers)
        return cast(Connection, payload)

    async def get_vector_store_service(self) -> VectorStoreService:
        if self._vector_store_service:
            return self._vector_store_service

        model_name = getattr(self._config.fastembed, "model", "BAAI/bge-small-en-v1.5")
        provider = FastEmbedProvider(model_name=model_name)
        service = VectorStoreService(self._config, self, provider)
        await service.initialize()
        self._vector_store_service = service
        return service

    async def get_cache_manager(self) -> CacheManager:
        cache_manager = self._cache_manager
        if cache_manager is None:
            registry = await self._get_service_registry()
            cache_manager = registry.cache_manager
            self._cache_manager = cache_manager
        return cache_manager

    async def get_embedding_manager(self) -> EmbeddingManager:
        embedding_manager = self._embedding_manager
        if embedding_manager is None:
            registry = await self._get_service_registry()
            embedding_manager = cast(EmbeddingManager, registry.embedding_manager)
            self._embedding_manager = embedding_manager
        return embedding_manager

    async def get_crawl_manager(self) -> CrawlingManager:
        crawling_manager = self._crawl_manager
        if crawling_manager is None:
            registry = await self._get_service_registry()
            crawling_manager = registry.crawl_manager
            self._crawl_manager = crawling_manager
        return crawling_manager

    async def get_crawling_manager(self) -> CrawlingManager:
        """Alias for get_crawl_manager for compatibility with new naming."""

        return await self.get_crawl_manager()

    async def get_content_intelligence_service(
        self,
    ) -> ContentIntelligenceService:
        content_intelligence = self._content_intelligence
        if content_intelligence is None:
            registry = await self._get_service_registry()
            content_intelligence = registry.content_intelligence
            self._content_intelligence = content_intelligence
        return content_intelligence

    async def get_project_storage(self) -> ProjectStorage:
        project_storage = self._project_storage
        if project_storage is None:
            registry = await self._get_service_registry()
            project_storage = registry.project_storage
            self._project_storage = project_storage
        return project_storage

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

    async def _get_service_registry(self) -> Any:
        if self._service_registry is None:
            self._service_registry = await _ensure_service_registry()
        return self._service_registry

    # Function-based dependency access methods (backward compatibility)

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status using function-based dependencies."""
        if deps_get_health_status:
            return await deps_get_health_status()
        logger.warning(
            "Health status monitoring not available - function-based dependency "
            "not found"
        )
        return {}

    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall health using function-based dependencies."""
        if deps_get_overall_health:
            return await deps_get_overall_health()
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
        try:
            yield await getters[client_type]()
        except (ConnectionError, TimeoutError, APIError, ValueError, RuntimeError):
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
