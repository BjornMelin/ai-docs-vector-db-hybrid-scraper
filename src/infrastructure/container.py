"""Dependency injection container wiring for the AI Docs services."""

# pylint: disable=c-extension-no-member

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiohttp
import redis.asyncio as redis
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
from langchain_mcp_adapters.sessions import Connection  # type: ignore
from qdrant_client import AsyncQdrantClient

from src.config.loader import Settings
from src.config.models import CacheType, MCPClientConfig, MCPServerConfig, MCPTransport
from src.infrastructure.project_storage import ProjectStorage
from src.services.cache.embedding_cache import EmbeddingCache
from src.services.cache.manager import CacheManager
from src.services.cache.search_cache import SearchResultCache
from src.services.circuit_breaker import CircuitBreakerManager
from src.services.content_intelligence.service import ContentIntelligenceService
from src.services.embeddings.manager import EmbeddingManager
from src.services.hyde.config import (
    HyDEConfig as ServiceHyDEConfig,
    HyDEMetricsConfig,
    HyDEPromptConfig,
)
from src.services.hyde.engine import HyDEQueryEngine
from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:
    from firecrawl import AsyncFirecrawlApp  # type: ignore[attr-defined]

    from src.services.browser.unified_manager import UnifiedBrowserManager
else:  # pragma: no cover - optional dependency
    UnifiedBrowserManager = Any  # type: ignore[assignment]


Dependency = providers.Dependency  # pylint: disable=c-extension-no-member
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


def _create_firecrawl_client(config: Any) -> AsyncFirecrawlApp | None:
    """Create Firecrawl client when the SDK is available."""
    try:
        module = importlib.import_module("firecrawl")
    except ModuleNotFoundError:
        logger.info("Firecrawl SDK not installed; skipping client creation")
        return None

    client_cls = getattr(module, "AsyncFirecrawlApp", None)
    if client_cls is None:
        client_cls = getattr(module, "AsyncFirecrawl", None)
    if client_cls is None:
        logger.warning("Firecrawl SDK missing async client; skipping initialization")
        return None

    client_factory = cast("type[AsyncFirecrawlApp]", client_cls)

    try:
        firecrawl_config = getattr(config, "firecrawl", None)
        api_key = getattr(firecrawl_config, "api_key", None) or ""
        return client_factory(api_key=api_key)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed to create Firecrawl client with config: %s", exc)
        return client_factory(api_key="")


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
        enable_dragonfly = bool(getattr(cache_config, "enable_dragonfly_cache", False))
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
    distributed_state = bool(
        cache_config is not None
        and getattr(cache_config, "enable_caching", False)
        and getattr(cache_config, "enable_dragonfly_cache", False)
    )
    redis_url = "redis://localhost:6379"
    if cache_config is not None:
        candidate = getattr(cache_config, "dragonfly_url", None)
        if candidate:
            redis_url = candidate

    try:
        if not distributed_state:
            return CircuitBreakerManager.in_memory(config=config)
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
) -> ContentIntelligenceService:
    """Instantiate the ContentIntelligenceService using required dependencies."""
    return ContentIntelligenceService(
        config=config,
        embedding_manager=embedding_manager,
        cache_manager=cache_manager,
    )


def _create_browser_manager(config: Any) -> UnifiedBrowserManager | None:
    """Instantiate the UnifiedBrowserManager when optional deps are available."""
    try:
        module = importlib.import_module("src.services.browser.unified_manager")
    except ModuleNotFoundError as exc:
        logger.info(
            "Browser integrations unavailable (missing dependency: %s); "
            "skipping UnifiedBrowserManager initialization",
            exc.name,
        )
        return None

    manager_cls = getattr(module, "UnifiedBrowserManager", None)
    if manager_cls is None or not callable(manager_cls):
        logger.warning(
            "Unified browser module missing manager class; skipping initialization",
        )
        return None

    manager_type = cast("type[UnifiedBrowserManager]", manager_cls)
    return manager_type(config)


def _create_rag_generator(
    config: Any,
    vector_service: VectorStoreService,
) -> Any | None:
    """Instantiate the RAG generator if the optional module is installed."""
    rag_config_model = getattr(config, "rag", None)
    if not getattr(rag_config_model, "enable_rag", False):
        return None

    try:
        rag_module = importlib.import_module("src.services.rag.generator")
        rag_models = importlib.import_module("src.services.rag.models")
        retriever_module = importlib.import_module("src.services.rag.retriever")
    except ModuleNotFoundError:
        logger.debug("RAG generator dependencies unavailable; skipping initialization")
        return None

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
    """Execute the service's canonical cleanup hook if available."""
    if service is None:
        return

    cleaner = getattr(service, "cleanup", None)
    if cleaner is None:
        cleaner = getattr(service, "close", None)
        if cleaner is None:
            return

    try:
        result = cleaner()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # noqa: BLE001  # pragma: no cover - isolate service cleanup
        logger.debug("Error during cleanup for service '%s'", name, exc_info=True)


@dataclass(frozen=True, slots=True)
class _ResolvedService:
    """A service instance resolved during this container generation."""

    name: str
    instance: Any


async def _initialize_service_graph(
    container: ApplicationContainer,
    resolved_services: list[_ResolvedService],
) -> None:
    """Resolve and initialize services while recording rollback ownership."""
    service_specs = (
        (container.cache_manager, "cache_manager", False),
        (container.embedding_manager, "embedding_manager", True),
        (container.vector_store_service, "vector_store_service", True),
        (container.project_storage, "project_storage", True),
        (container.circuit_breaker_manager, "circuit_breaker_manager", False),
        (
            container.content_intelligence_service,
            "content_intelligence_service",
            True,
        ),
        (container.browser_manager, "browser_manager", True),
        (container.rag_generator, "rag_generator", False),
    )
    for provider, name, required in service_specs:
        service = provider()
        if service is not None:
            resolved_services.append(_ResolvedService(name, service))
        await _maybe_initialize(service, name, required=required)


async def _cleanup_service_graph(
    resolved_services: list[_ResolvedService],
) -> None:
    """Cleanup only resolved services, in exact reverse resolution order."""
    for service in reversed(resolved_services):
        await _maybe_cleanup(service.instance, service.name)


async def _run_task_factories(
    factories: list[Any],
    *,
    suppress_errors: bool = False,
) -> None:
    """Execute callables returned by container task registries."""
    for factory in factories:
        try:
            result = factory()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            if not suppress_errors:
                raise
            logger.debug("Container task execution failed", exc_info=True)


class ApplicationContainer(DeclarativeContainer):
    """Application dependency injection container."""

    # Configuration
    config = Dependency(instance_of=Settings)

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
        EmbeddingManager,
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

    def __init__(self) -> None:
        """Initialize container manager."""
        self.container: ApplicationContainer | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._generation = 0
        self._next_lease_id = 0
        self._active_lease_ids: set[int] = set()
        self._lease_managed_generation: int | None = None
        self._resolved_services: list[_ResolvedService] = []
        self._qdrant_client: AsyncQdrantClient | None = None

    async def initialize(self, config: Settings) -> ApplicationContainer:
        """Initialize the container with configuration."""
        async with self._lock:
            if self._initialized:
                if self.container is None:
                    raise RuntimeError("Container manager in inconsistent state")
                return self.container
            return await self._initialize_locked(config)

    async def acquire(
        self,
        config: Settings,
        *,
        force_reload: bool = False,
    ) -> ContainerLease:
        """Acquire a generation-scoped lease on the active container."""
        async with self._lock:
            if force_reload and self.container is not None:
                if self._active_lease_ids:
                    msg = "Cannot force-reload the container while sessions are active"
                    raise RuntimeError(msg)
                await self._shutdown_locked()

            created = self.container is None
            if created:
                container = await self._initialize_locked(config)
                self._lease_managed_generation = self._generation
            else:
                container = self.container
                if container is None:  # pragma: no cover - narrowed above
                    raise RuntimeError("Container manager in inconsistent state")

            self._next_lease_id += 1
            lease_id = self._next_lease_id
            self._active_lease_ids.add(lease_id)
            return ContainerLease(
                container=container,
                generation=self._generation,
                lease_id=lease_id,
            )

    async def release(self, lease: ContainerLease) -> None:
        """Release a lease and stop lease-owned containers after the last holder."""
        async with self._lock:
            if (
                lease.generation != self._generation
                or lease.container is not self.container
                or lease.lease_id not in self._active_lease_ids
            ):
                raise RuntimeError("Container lease is no longer active")

            self._active_lease_ids.remove(lease.lease_id)
            if (
                not self._active_lease_ids
                and self._lease_managed_generation == lease.generation
            ):
                await self._shutdown_locked()

    async def shutdown(self) -> None:
        """Shutdown the container and cleanup resources."""
        async with self._lock:
            if self._active_lease_ids:
                msg = "Cannot shut down the container while sessions are active"
                raise RuntimeError(msg)
            await self._shutdown_locked()

    async def _initialize_locked(self, config: Settings) -> ApplicationContainer:
        """Initialize and publish one container while ``_lock`` is held."""
        candidate = ApplicationContainer(config=config)
        resolved_services: list[_ResolvedService] = []
        qdrant_client: AsyncQdrantClient | None = None
        try:
            await candidate.init_resources()  # pyright: ignore[reportGeneralTypeIssues]
            qdrant_client = candidate.qdrant_client()
            await _initialize_service_graph(candidate, resolved_services)
            await _run_task_factories(list(candidate.startup_tasks()))
        except BaseException:
            try:
                await _cleanup_service_graph(resolved_services)
            except Exception:  # pragma: no cover - best-effort rollback
                logger.exception("Failed to roll back partially initialized services")
            if qdrant_client is not None:
                try:
                    await qdrant_client.close()
                except Exception:  # pragma: no cover - best-effort rollback
                    logger.exception("Failed to release shared Qdrant client")
            try:
                await candidate.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
            except Exception:  # pragma: no cover - best-effort rollback
                logger.exception("Failed to release partially initialized resources")
            raise

        self.container = candidate
        self._initialized = True
        self._resolved_services = resolved_services
        self._qdrant_client = qdrant_client
        self._generation += 1
        logger.info("Dependency injection container initialized")
        return candidate

    async def _shutdown_locked(self) -> None:
        """Shutdown the active container while ``_lock`` is held."""
        container = self.container
        if container is None:
            self._initialized = False
            self._resolved_services = []
            self._lease_managed_generation = None
            self._qdrant_client = None
            return
        try:
            if self._initialized:
                await _run_task_factories(
                    list(container.shutdown_tasks()),
                    suppress_errors=True,
                )
                await _cleanup_service_graph(self._resolved_services)
        finally:
            try:
                qdrant_client = self._qdrant_client
                self._qdrant_client = None
                if qdrant_client is not None:
                    await qdrant_client.close()
            finally:
                try:
                    await container.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
                finally:
                    self.container = None
                    self._initialized = False
                    self._resolved_services = []
                    self._lease_managed_generation = None
        logger.info("Dependency injection container shutdown")


@dataclass(frozen=True, slots=True)
class ContainerLease:
    """Ownership token for one container generation."""

    container: ApplicationContainer
    generation: int
    lease_id: int


# Global container manager instance
_container_manager = ContainerManager()


@lru_cache(maxsize=1)
def get_container() -> ApplicationContainer | None:
    """Get the global container instance."""
    return _container_manager.container


async def initialize_container(config: Settings) -> ApplicationContainer:
    """Initialize the global container."""
    container = await _container_manager.initialize(config)
    get_container.cache_clear()
    return container


async def shutdown_container() -> None:
    """Shutdown the global container."""
    try:
        await _container_manager.shutdown()
    finally:
        get_container.cache_clear()


async def acquire_container(
    config: Settings,
    *,
    force_reload: bool = False,
) -> ContainerLease:
    """Acquire a lease on the global container."""
    lease = await _container_manager.acquire(config, force_reload=force_reload)
    get_container.cache_clear()
    return lease


async def release_container(lease: ContainerLease) -> None:
    """Release a lease on the global container."""
    try:
        await _container_manager.release(lease)
    finally:
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


def inject_firecrawl() -> Provider[AsyncFirecrawlApp | None]:
    """Inject raw Firecrawl client dependency."""
    return Provide[ApplicationContainer.firecrawl_client]


def inject_http() -> Provider[Any]:
    """Inject raw HTTP client dependency."""
    return Provide[ApplicationContainer.http_client]


# Context manager for automatic dependency injection setup
class DependencyContext:
    """Context manager for dependency injection setup."""

    def __init__(self, config: Settings):
        """Store configuration for deferred container initialization.

        Args:
            config: Runtime settings for dependency construction.
        """
        self.config = config
        self.container: ApplicationContainer | None = None
        self.lease: ContainerLease | None = None

    async def __aenter__(self) -> ApplicationContainer:
        """Initialize dependencies."""
        self.lease = await acquire_container(self.config)
        self.container = self.lease.container
        return self.container

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup dependencies."""
        lease = self.lease
        if lease is None:
            return
        try:
            await release_container(lease)
        finally:
            self.lease = None
            self.container = None


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
