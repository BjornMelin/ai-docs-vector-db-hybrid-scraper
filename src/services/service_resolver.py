"""Helpers to resolve core services from the dependency injector container."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

from src.infrastructure.container import ApplicationContainer, get_container
from src.services.circuit_breaker.decorators import circuit_breaker


if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore

    from src.services.cache.manager import CacheManager
    from src.services.embeddings.manager import EmbeddingManager
    from src.services.vector_db.service import VectorStoreService
else:  # pragma: no cover - runtime fallbacks for typing imports
    CacheManager = EmbeddingManager = VectorStoreService = MultiServerMCPClient = Any


T = TypeVar("T")


def _require_container() -> ApplicationContainer:
    """Return the globally initialised container or raise an error.

    Returns:
        ApplicationContainer: The globally configured dependency injector container.

    Raises:
        RuntimeError: If the container has not been initialised yet.
    """

    container = get_container()
    if container is None:
        msg = "Dependency injector container is not initialized"
        raise RuntimeError(msg)
    return container


async def _ensure_initialized(service: Any, *, name: str) -> Any:
    """Ensure lazily initialised services are ready for use.

    Args:
        service: The service instance resolved from the container.
        name: Human readable identifier for error reporting.

    Returns:
        Any: The same service instance, initialised if required.
    """

    initializer = getattr(service, "initialize", None)
    is_initialized = getattr(service, "is_initialized", None)

    needs_initialization = False
    if callable(is_initialized):
        try:
            needs_initialization = not bool(is_initialized())
        except Exception:  # pragma: no cover - defensive guard
            needs_initialization = True
    elif callable(initializer):
        needs_initialization = True

    if needs_initialization and callable(initializer):
        result = initializer()
        if inspect.isawaitable(result):
            await result
    return service


@overload
async def _resolve_service(  # noqa: UP047
    *,
    name: str,
    supplier: Callable[[ApplicationContainer], T],
    ensure_ready: bool = True,
    optional: Literal[False] = False,
) -> T: ...


@overload
async def _resolve_service(  # noqa: UP047
    *,
    name: str,
    supplier: Callable[[ApplicationContainer], T],
    ensure_ready: bool = True,
    optional: Literal[True],
) -> T | None: ...


async def _resolve_service(  # noqa: UP047
    *,
    name: str,
    supplier: Callable[[ApplicationContainer], T],
    ensure_ready: bool = True,
    optional: bool = False,
) -> T | None:
    """Resolve a service from the container and optionally initialise it.

    Args:
        name: Identifier for the service used in error messages.
        supplier: Callable that retrieves the service from the container.
        ensure_ready: Whether to call ``initialize`` on the service when available.
        optional: Whether ``None`` is an acceptable result from the container.

    Returns:
        T | None: The resolved and initialised service instance when available.

    Raises:
        RuntimeError: If a required service is not available from the container.
    """

    container = _require_container()
    instance = supplier(container)
    if instance is None:
        if optional:
            return None
        msg = f"Container returned None for required service '{name}'"
        raise RuntimeError(msg)

    value = cast(T, instance)
    if ensure_ready:
        await _ensure_initialized(value, name=name)
    return value


@circuit_breaker(
    service_name="cache_manager",
    failure_threshold=2,
    recovery_timeout=10.0,
)
async def get_cache_manager() -> CacheManager:
    """Return the cache manager singleton provided by the container.

    Returns:
        CacheManager: The cache manager maintained by the container.

    Raises:
        RuntimeError: If the cache manager cannot be resolved.
    """

    manager = await _resolve_service(
        name="cache_manager",
        supplier=lambda container: container.cache_manager(),
    )
    return cast(CacheManager, manager)


@circuit_breaker(
    service_name="embedding_manager",
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def get_embedding_manager() -> EmbeddingManager:
    """Return the embedding manager configured in the container.

    Returns:
        EmbeddingManager: The shared embedding manager instance.

    Raises:
        RuntimeError: If the embedding manager cannot be resolved.
    """

    manager = await _resolve_service(
        name="embedding_manager",
        supplier=lambda container: container.embedding_manager(),
    )
    return cast(EmbeddingManager, manager)


@circuit_breaker(
    service_name="vector_store_service",
    failure_threshold=3,
    recovery_timeout=15.0,
)
async def get_vector_store_service() -> VectorStoreService:
    """Return the vector store service resolved from the container.

    Returns:
        VectorStoreService: The vector store service instance.

    Raises:
        RuntimeError: If the vector store service cannot be resolved.
    """

    service = await _resolve_service(
        name="vector_store_service",
        supplier=lambda container: container.vector_store_service(),
    )
    return cast(VectorStoreService, service)


@circuit_breaker(
    service_name="content_intelligence_service",
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def get_content_intelligence_service() -> Any:
    """Return the content intelligence service instance.

    Returns:
        Any: The content intelligence service registered in the container.

    Raises:
        RuntimeError: If the content intelligence service is unavailable.
    """

    service = await _resolve_service(
        name="content_intelligence_service",
        supplier=lambda container: container.content_intelligence_service(),
    )
    return service


async def get_mcp_client() -> MultiServerMCPClient:
    """Return the shared MultiServerMCPClient instance.

    Returns:
        MultiServerMCPClient: The multi-server MCP client managed by the container.

    Raises:
        RuntimeError: If the MCP client integration is disabled.
    """

    client = await _resolve_service(
        name="mcp_client",
        supplier=lambda container: container.mcp_client(),
        ensure_ready=False,
        optional=True,
    )
    if client is None:
        msg = "MCP client integration is disabled"
        raise RuntimeError(msg)
    return cast(MultiServerMCPClient, client)


@circuit_breaker(
    service_name="browser_manager",
    failure_threshold=5,
    recovery_timeout=60.0,
)
async def get_crawl_manager() -> Any:
    """Return the unified browser manager used for crawling tasks.

    Returns:
        Any: The browser manager registered with the container.

    Raises:
        RuntimeError: If the browser manager is unavailable.
    """

    manager = await _resolve_service(
        name="browser_manager",
        supplier=lambda container: container.browser_manager(),
    )
    return manager


@circuit_breaker(
    service_name="rag_generator",
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def get_rag_generator() -> Any:
    """Return the configured RAG generator.

    Returns:
        Any: The configured retrieval-augmented generation pipeline.

    Raises:
        RuntimeError: If the RAG generator is not available.
    """

    generator = await _resolve_service(
        name="rag_generator",
        supplier=lambda container: container.rag_generator(),
        optional=True,
    )
    if generator is None:
        msg = (
            "RAG generator unavailable; ensure optional RAG dependencies are "
            "installed and enabled"
        )
        raise RuntimeError(msg)
    return generator


__all__ = [
    "get_cache_manager",
    "get_content_intelligence_service",
    "get_crawl_manager",
    "get_embedding_manager",
    "get_mcp_client",
    "get_rag_generator",
    "get_vector_store_service",
]
