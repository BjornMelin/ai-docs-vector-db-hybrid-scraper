"""Dependency injection container wiring for the AI Docs services."""

# pylint: disable=c-extension-no-member

import asyncio
import importlib
import logging
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import aiohttp
import redis.asyncio as redis
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide
from firecrawl import AsyncFirecrawlApp  # type: ignore
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
from langchain_mcp_adapters.sessions import Connection  # type: ignore
from qdrant_client import AsyncQdrantClient

from src.config.models import CacheType, MCPClientConfig, MCPServerConfig, MCPTransport
from src.services.cache.embedding_cache import EmbeddingCache
from src.services.cache.manager import CacheManager
from src.services.cache.search_cache import SearchResultCache
from src.services.circuit_breaker import CircuitBreakerManager
from src.services.core.project_storage import ProjectStorage
from src.services.embeddings.manager import EmbeddingManager
from src.services.hyde.config import (
    HyDEConfig as ServiceHyDEConfig,
    HyDEMetricsConfig,
    HyDEPromptConfig,
)
from src.services.hyde.engine import HyDEQueryEngine
from src.services.vector_db.service import VectorStoreService


Configuration = providers.Configuration  # pylint: disable=c-extension-no-member
Singleton = providers.Singleton  # pylint: disable=c-extension-no-member
Factory = providers.Factory  # pylint: disable=c-extension-no-member
List = providers.List  # pylint: disable=c-extension-no-member
Resource = providers.Resource  # pylint: disable=c-extension-no-member
DeclarativeContainer = containers.DeclarativeContainer  # pylint: disable=c-extension-no-member
Provider = providers.Provider  # pylint: disable=c-extension-no-member

logger = logging.getLogger(__name__)


def _create_qdrant_client(config: Any) -> AsyncQdrantClient:
    """Create Qdrant client with configuration."""

    try:
        qdrant_config = getattr(config, "qdrant", None)
        url = getattr(qdrant_config, "url", None) or "http://localhost:6333"
        api_key = getattr(qdrant_config, "api_key", None)
        timeout = int(getattr(qdrant_config, "timeout", None) or 30)
        prefer_grpc = getattr(qdrant_config, "prefer_grpc", None) or False
        return AsyncQdrantClient(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create Qdrant client with config: %s", e)
        return AsyncQdrantClient(url="http://localhost:6333")


def _create_dragonfly_client(config: Any) -> redis.Redis:
    """Create a Redis-compatible Dragonfly client from configuration."""

    try:
        cache_config = getattr(config, "cache", None)
        url = getattr(cache_config, "dragonfly_url", None) or "redis://localhost:6379"
        pool_size = getattr(cache_config, "redis_pool_size", None) or 20
        return redis.from_url(url, max_connections=pool_size, decode_responses=True)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed to create Dragonfly client with config: %s", exc)
        return redis.from_url(
            "redis://localhost:6379", max_connections=20, decode_responses=True
        )


def _create_firecrawl_client(config: Any) -> AsyncFirecrawlApp:
    """Create Firecrawl client with configuration."""

    try:
        firecrawl_config = getattr(config, "firecrawl", None)
        api_key = getattr(firecrawl_config, "api_key", None) or ""
        return AsyncFirecrawlApp(api_key=api_key)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create Firecrawl client with config: %s", e)
        return AsyncFirecrawlApp(api_key="")


async def _create_http_client() -> AsyncGenerator[Any]:
    """Create HTTP client with proper lifecycle management."""

    timeout_config = aiohttp.ClientTimeout(total=30.0)
    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        yield session


def _create_parallel_processing_system(*, embedding_manager: Any) -> Any:
    """Construct the parallel processing system if available.

    Falls back to a lightweight stub when the optional dependency graph is absent.
    """

    manager = embedding_manager() if callable(embedding_manager) else embedding_manager

    try:  # Lazy import to avoid mandatory dependency.
        module = importlib.import_module(
            "src.services.processing.parallel_processing_system"
        )
        factory = module.create_parallel_processing_system
    except (ModuleNotFoundError, AttributeError):
        logger.debug("Parallel processing factory unavailable; using embedding manager")
        return manager

    return factory(embedding_manager=manager)


def _create_cache_manager(config: Any) -> CacheManager:
    """Instantiate the CacheManager from application configuration."""

    cache_config = getattr(config, "cache", None)
    dragonfly_url = "redis://localhost:6379"
    enable_distributed_cache = True
    ttl_overrides: dict[CacheType, int] = {}

    if cache_config is not None:
        dragonfly_url = getattr(cache_config, "dragonfly_url", dragonfly_url)
        enable_caching = bool(getattr(cache_config, "enable_caching", True))
        enable_dragonfly = bool(getattr(cache_config, "enable_dragonfly_cache", True))
        enable_distributed_cache = enable_caching and enable_dragonfly

        ttl_overrides = {
            CacheType.EMBEDDINGS: int(getattr(cache_config, "ttl_embeddings", 86400)),
            CacheType.SEARCH: int(getattr(cache_config, "ttl_search_results", 3600)),
            CacheType.CRAWL: int(getattr(cache_config, "ttl_crawl", 3600)),
            CacheType.QUERIES: int(getattr(cache_config, "ttl_queries", 7200)),
        }

        # Allow arbitrary overrides via cache_ttl_seconds mapping.
        raw_overrides = getattr(cache_config, "cache_ttl_seconds", {})
        override_map = {
            "embeddings": CacheType.EMBEDDINGS,
            "search_results": CacheType.SEARCH,
            "collections": CacheType.CRAWL,
            "queries": CacheType.QUERIES,
        }
        for name, ttl in raw_overrides.items():
            cache_type = override_map.get(name)
            if cache_type is not None:
                ttl_overrides[cache_type] = int(ttl)

    return CacheManager(
        dragonfly_url=dragonfly_url,
        enable_distributed_cache=enable_distributed_cache,
        distributed_ttl_seconds=ttl_overrides,
    )


def _create_embedding_manager(
    config: Any,
    cache_manager: CacheManager | None,
) -> EmbeddingManager:
    """Instantiate the EmbeddingManager with DI-provided dependencies."""

    return EmbeddingManager(
        config=config,
        cache_manager=cache_manager,
    )


def _create_vector_store_service(
    config: Any,
    async_qdrant_client: AsyncQdrantClient,
) -> VectorStoreService:
    """Instantiate VectorStoreService backed by LangChain's Qdrant adapter."""

    return VectorStoreService(
        config=config,
        async_qdrant_client=async_qdrant_client,
    )


def _create_hyde_query_engine(
    config: Any,
    embedding_manager: EmbeddingManager,
    vector_store: VectorStoreService,
    embedding_cache: EmbeddingCache | None,
    search_cache: SearchResultCache | None,
) -> HyDEQueryEngine:
    """Build a HyDEQueryEngine wired to shared cache services."""

    hyde_config_source = getattr(config, "hyde", None)
    hyde_config = (
        ServiceHyDEConfig.from_unified_config(hyde_config_source)
        if hyde_config_source is not None
        else ServiceHyDEConfig()
    )
    prompt_config = HyDEPromptConfig()
    metrics_config = HyDEMetricsConfig()
    openai_config = getattr(config, "openai", None)
    openai_api_key = getattr(openai_config, "api_key", None)

    return HyDEQueryEngine(
        config=hyde_config,
        prompt_config=prompt_config,
        metrics_config=metrics_config,
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        embedding_cache=embedding_cache,
        search_cache=search_cache,
        openai_api_key=openai_api_key,
    )


def _create_circuit_breaker_manager(config: Any) -> CircuitBreakerManager | None:
    """Instantiate the CircuitBreakerManager if purgatory is available."""

    cache_config = getattr(config, "cache", None)
    redis_url = "redis://localhost:6379"
    if cache_config is not None:
        candidate = getattr(cache_config, "dragonfly_url", None)
        if candidate:
            redis_url = candidate

    try:
        return CircuitBreakerManager(
            redis_url=redis_url,
            config=config,
        )
    except RuntimeError as exc:
        logger.warning(
            "CircuitBreakerManager unavailable (purgatory missing?): %s", exc
        )
        return None


def _create_project_storage(config: Any) -> ProjectStorage:
    """Instantiate project storage backed by filesystem."""

    data_dir = getattr(config, "data_dir", None)
    if data_dir is None:
        msg = "Configuration missing data_dir for project storage"
        raise RuntimeError(msg)
    return ProjectStorage(data_dir=Path(data_dir))


def _create_content_intelligence_service(
    config: Any,
    embedding_manager: Any,
    cache_manager: Any,
) -> Any | None:
    """Lazily instantiate the ContentIntelligenceService if available."""

    try:
        module = importlib.import_module("src.services.content_intelligence.service")
    except ModuleNotFoundError:
        logger.debug(
            "Content intelligence service unavailable; optional dependency missing"
        )
        return None

    service_cls = getattr(module, "ContentIntelligenceService", None)
    if service_cls is None:
        logger.warning(
            "Content intelligence module does not expose ContentIntelligenceService"
        )
        return None

    return service_cls(
        config=config,
        embedding_manager=embedding_manager,
        cache_manager=cache_manager,
    )


def _create_browser_manager(config: Any) -> Any | None:
    """Instantiate the UnifiedBrowserManager if optional dependencies exist."""

    try:
        module = importlib.import_module("src.services.browser.unified_manager")
    except ModuleNotFoundError:
        logger.debug("UnifiedBrowserManager unavailable; optional dependency missing")
        return None

    manager_cls = getattr(module, "UnifiedBrowserManager", None)
    if manager_cls is None:
        logger.warning(
            "Unified browser module does not define UnifiedBrowserManager class"
        )
        return None

    return manager_cls(config)


def _create_rag_generator(
    config: Any,
    vector_service: VectorStoreService,
) -> Any | None:
    """Instantiate the RAG generator if the optional module is installed."""

    try:
        rag_module = importlib.import_module("src.services.rag.generator")
        rag_models = importlib.import_module("src.services.rag.models")
        retriever_module = importlib.import_module("src.services.rag.retriever")
    except ModuleNotFoundError:
        logger.debug("RAG generator dependencies unavailable; skipping initialization")
        return None

    rag_config_model = getattr(config, "rag", None)
    rag_config_cls = getattr(rag_models, "RAGConfig", None)
    if rag_config_cls is None:
        logger.warning("RAG models module missing RAGConfig; generator disabled")
        return None

    payload = {}
    if rag_config_model is not None:
        if hasattr(rag_config_model, "model_dump"):
            payload = rag_config_model.model_dump()
        elif isinstance(rag_config_model, dict):
            payload = rag_config_model
    rag_config = rag_config_cls.model_validate(payload)

    collection_name = getattr(
        getattr(config, "qdrant", None),
        "collection_name",
        "documents",
    )

    retriever_cls = getattr(retriever_module, "VectorServiceRetriever", None)
    if retriever_cls is None:
        logger.warning("RAG retriever class missing; generator disabled")
        return None

    retriever = retriever_cls(
        vector_service=vector_service,
        collection=collection_name,
        k=getattr(rag_config, "retriever_top_k", 5),
        rag_config=rag_config,
    )

    generator_cls = getattr(rag_module, "RAGGenerator", None)
    if generator_cls is None:
        logger.warning("RAG generator class missing; generator disabled")
        return None

    return generator_cls(rag_config, retriever)


def _build_mcp_connections(config: MCPClientConfig) -> dict[str, Connection]:
    """Translate MCP client configuration into session connections."""

    connections: dict[str, Connection] = {}
    for server in config.servers:
        connections[server.name] = _serialise_mcp_server(server, config)
    return connections


def _serialise_mcp_server(
    server: MCPServerConfig, config: MCPClientConfig
) -> Connection:
    timeout_ms = (
        server.timeout_ms
        if server.timeout_ms is not None
        else config.request_timeout_ms
    )
    timeout_seconds = timeout_ms / 1000.0

    if server.transport == MCPTransport.STDIO:
        payload: dict[str, Any] = {
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


def _create_mcp_client(config: Any) -> MultiServerMCPClient | None:
    """Instantiate MultiServerMCPClient when enabled in configuration."""

    mcp_config = getattr(config, "mcp_client", None)
    if not isinstance(mcp_config, MCPClientConfig) or not mcp_config.enabled:
        return None
    if not mcp_config.servers:
        logger.warning("MCP client enabled but no servers configured")
        return None
    connections = _build_mcp_connections(mcp_config)
    return MultiServerMCPClient(connections)


async def _maybe_initialize(service: Any, name: str, *, required: bool = True) -> None:
    """Execute service.initialize() if available."""

    if service is None:
        return

    initializer = getattr(service, "initialize", None)
    if initializer is None:
        return

    try:
        result = initializer()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # pragma: no cover - defensive
        if required:
            msg = f"Failed to initialize core service '{name}': {exc}"
            raise RuntimeError(msg) from exc
        logger.warning("Optional service '%s' failed to initialize: %s", name, exc)


async def _maybe_cleanup(service: Any, name: str) -> None:
    """Execute service.cleanup() if available."""

    if service is None:
        return

    cleaner = getattr(service, "cleanup", None)
    if cleaner is None:
        return

    try:
        result = cleaner()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # pragma: no cover - defensive
        logger.debug("Error during cleanup for service '%s'", name, exc_info=True)


async def _initialize_service_graph(container: "ApplicationContainer") -> None:
    """Initialize core and optional services managed by the container."""

    await _maybe_initialize(
        container.cache_manager(),
        "cache_manager",
        required=False,
    )
    await _maybe_initialize(container.embedding_manager(), "embedding_manager")
    await _maybe_initialize(container.vector_store_service(), "vector_store_service")
    await _maybe_initialize(container.project_storage(), "project_storage")
    await _maybe_initialize(
        container.circuit_breaker_manager(),
        "circuit_breaker_manager",
        required=False,
    )
    await _maybe_initialize(
        container.content_intelligence_service(),
        "content_intelligence_service",
        required=False,
    )
    await _maybe_initialize(
        container.browser_manager(),
        "browser_manager",
        required=False,
    )
    await _maybe_initialize(
        container.rag_generator(),
        "rag_generator",
        required=False,
    )


async def _cleanup_service_graph(container: "ApplicationContainer") -> None:
    """Cleanup services managed by the container in reverse order."""

    await _maybe_cleanup(container.rag_generator(), "rag_generator")
    await _maybe_cleanup(container.browser_manager(), "browser_manager")
    await _maybe_cleanup(
        container.content_intelligence_service(),
        "content_intelligence_service",
    )
    await _maybe_cleanup(container.vector_store_service(), "vector_store_service")
    await _maybe_cleanup(container.embedding_manager(), "embedding_manager")
    await _maybe_cleanup(container.cache_manager(), "cache_manager")
    await _maybe_cleanup(container.project_storage(), "project_storage")
    await _maybe_cleanup(
        container.circuit_breaker_manager(),
        "circuit_breaker_manager",
    )


async def _run_task_factories(factories: list[Any]) -> None:
    """Execute callables returned by container task registries."""

    for factory in factories:
        try:
            result = factory()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover - defensive
            logger.debug("Container task execution failed", exc_info=True)


class ApplicationContainer(DeclarativeContainer):
    """Application dependency injection container."""

    # Configuration
    config = Configuration()

    qdrant_client = Singleton(
        _create_qdrant_client,
        config=config,
    )

    dragonfly_client = Singleton(
        _create_dragonfly_client,
        config=config,
    )

    firecrawl_client = Singleton(
        _create_firecrawl_client,
        config=config,
    )

    # HTTP client with session management
    http_client = Resource(
        _create_http_client,
    )

    cache_manager = Singleton(
        _create_cache_manager,
        config=config,
    )

    embedding_manager = Singleton(
        _create_embedding_manager,
        config=config,
        cache_manager=cache_manager,
    )

    vector_store_service = Singleton(
        _create_vector_store_service,
        config=config,
        async_qdrant_client=qdrant_client,
    )

    hyde_query_engine = Singleton(
        _create_hyde_query_engine,
        config=config,
        embedding_manager=embedding_manager,
        vector_store=vector_store_service,
        embedding_cache=cache_manager.provided.embedding_cache,
        search_cache=cache_manager.provided.search_cache,
    )

    circuit_breaker_manager = Singleton(
        _create_circuit_breaker_manager,
        config=config,
    )

    project_storage = Singleton(
        _create_project_storage,
        config=config,
    )

    content_intelligence_service = Singleton(
        _create_content_intelligence_service,
        config=config,
        embedding_manager=embedding_manager,
        cache_manager=cache_manager,
    )

    browser_manager = Singleton(
        _create_browser_manager,
        config=config,
    )

    rag_generator = Singleton(
        _create_rag_generator,
        config=config,
        vector_service=vector_store_service,
    )

    mcp_client = Singleton(
        _create_mcp_client,
        config=config,
    )

    # Parallel processing system
    parallel_processing_system = Factory(
        _create_parallel_processing_system,
        embedding_manager=embedding_manager,
    )

    # Lifecycle management
    startup_tasks = List()
    shutdown_tasks = List()


class ContainerManager:
    """Manager for dependency injection container lifecycle."""

    def __init__(self):
        self.container: ApplicationContainer | None = None
        self._initialized = False

    async def initialize(self, config: Any) -> ApplicationContainer:
        """Initialize the container with configuration."""

        if self._initialized:
            if self.container is None:
                raise RuntimeError("Container manager in inconsistent state")
            return self.container

        self.container = ApplicationContainer()
        self.container.config.from_dict(self._config_to_dict(config))

        # Initialize resource providers
        await self.container.init_resources()  # pyright: ignore[reportGeneralTypeIssues]

        await _initialize_service_graph(self.container)
        await _run_task_factories(list(self.container.startup_tasks()))

        self._initialized = True
        logger.info("Dependency injection container initialized")
        return self.container

    async def shutdown(self) -> None:
        """Shutdown the container and cleanup resources."""

        if self._initialized and self.container is not None:
            await _run_task_factories(list(self.container.shutdown_tasks()))
            await _cleanup_service_graph(self.container)
            await self.container.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
            self.container = None
            self._initialized = False
            logger.info("Dependency injection container shutdown")

    def _config_to_dict(self, config: Any) -> dict:
        """Convert config object to dictionary for dependency-injector."""

        try:
            # Try to convert using model_dump if it's a Pydantic model
            if hasattr(config, "model_dump"):
                return config.model_dump()
            # Try to convert using dict() if it's a dataclass or similar
            if hasattr(config, "__dict__"):
                return self._serialize_config_dict(config.__dict__)
            # Fallback to basic attributes
            return {
                key: getattr(config, key)
                for key in dir(config)
                if not key.startswith("_") and not callable(getattr(config, key))
            }
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Failed to convert config to dict: %s", e)
            return {}

    def _serialize_config_dict(self, data: Any) -> Any:
        """Recursively serialize configuration data."""

        if hasattr(data, "model_dump"):
            return data.model_dump()
        if hasattr(data, "__dict__"):
            return {
                key: self._serialize_config_dict(value)
                for key, value in data.__dict__.items()
                if not key.startswith("_")
            }
        if isinstance(data, dict):
            return {
                key: self._serialize_config_dict(value) for key, value in data.items()
            }
        if isinstance(data, list | tuple):
            return [self._serialize_config_dict(item) for item in data]
        return data


# Global container manager instance
_container_manager = ContainerManager()


@lru_cache(maxsize=1)
def get_container() -> ApplicationContainer | None:
    """Get the global container instance."""

    return _container_manager.container


async def initialize_container(config: Any) -> ApplicationContainer:
    """Initialize the global container."""

    container = await _container_manager.initialize(config)
    get_container.cache_clear()
    return container


async def shutdown_container() -> None:
    """Shutdown the global container."""

    await _container_manager.shutdown()
    get_container.cache_clear()


# Dependency injection decorators and functions for easy access
def inject_parallel_processing_system():
    """Inject parallel processing system dependency."""

    return Provide[ApplicationContainer.parallel_processing_system]


def inject_cache_manager():
    """Inject cache manager dependency."""

    return Provide[ApplicationContainer.cache_manager]


def inject_embedding_manager():
    """Inject embedding manager dependency."""

    return Provide[ApplicationContainer.embedding_manager]


def inject_vector_store_service():
    """Inject vector store service dependency."""

    return Provide[ApplicationContainer.vector_store_service]


def inject_hyde_query_engine():
    """Inject HyDE query engine dependency."""

    return Provide[ApplicationContainer.hyde_query_engine]


def inject_circuit_breaker_manager():
    """Inject circuit breaker manager dependency."""

    return Provide[ApplicationContainer.circuit_breaker_manager]


def inject_project_storage():
    """Inject project storage dependency."""

    return Provide[ApplicationContainer.project_storage]


def inject_content_intelligence_service():
    """Inject content intelligence service dependency."""

    return Provide[ApplicationContainer.content_intelligence_service]


def inject_browser_manager():
    """Inject unified browser manager dependency."""

    return Provide[ApplicationContainer.browser_manager]


def inject_rag_generator():
    """Inject RAG generator dependency."""

    return Provide[ApplicationContainer.rag_generator]


def inject_qdrant() -> Provider[AsyncQdrantClient]:
    """Inject raw Qdrant client dependency."""

    return Provide[ApplicationContainer.qdrant_client]


def inject_dragonfly_client() -> Provider[redis.Redis]:
    """Inject raw Dragonfly cache client dependency."""

    return Provide[ApplicationContainer.dragonfly_client]


def inject_firecrawl() -> Provider[AsyncFirecrawlApp]:
    """Inject raw Firecrawl client dependency."""

    return Provide[ApplicationContainer.firecrawl_client]


def inject_http() -> Provider[Any]:
    """Inject raw HTTP client dependency."""

    return Provide[ApplicationContainer.http_client]


# Context manager for automatic dependency injection setup
class DependencyContext:
    """Context manager for dependency injection setup."""

    def __init__(self, config: Any):
        self.config = config
        self.container: ApplicationContainer | None = None

    async def __aenter__(self) -> ApplicationContainer:
        """Initialize dependencies."""

        self.container = await initialize_container(self.config)
        return self.container

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup dependencies."""

        await shutdown_container()


# Wire modules for automatic dependency injection
def wire_modules() -> None:
    """Wire modules for dependency injection."""

    container = get_container()
    if container:
        # Wire commonly used modules
        modules = [
            "src.services.embeddings",
            "src.services.vector_db",
            "src.services.crawling",
            "src.services.cache",
            "src.api.routers",
            "src.mcp_tools",
        ]
        container.wire(modules=modules)
        logger.info("Dependency injection wiring completed")
