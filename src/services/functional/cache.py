"""Function-based cache service with FastAPI dependency injection.

Transforms the CacheManager class into pure functions with dependency injection.
Provides simple cache operations with circuit breaker patterns.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends  # pyright: ignore

from src.config import CacheType
from src.services.cache import CacheManager

from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_cache_client


logger = logging.getLogger(__name__)


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_get(
    key: str,
    cache_type: CacheType = CacheType.LOCAL,
    default: Any = None,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> Any:
    """Get value from cache with L1 -> L2 fallback.

    Pure function replacement for CacheManager.get().

    Args:
        key: Cache key
        cache_type: Type of cached data for TTL selection
        default: Default value if not found
        cache_client: Injected cache manager

    Returns:
        Cached value or default
    """

    try:
        if not cache_client:
            logger.warning("Cache client not available, returning default")
            return default

        result = await cache_client.get(key, cache_type, default)

        if result != default:
            logger.debug("Cache hit for key: %s", key)
        else:
            logger.debug("Cache miss for key: %s", key)

    except (AttributeError, ConnectionError, OSError):
        logger.exception("Cache get failed for key: %s", key)
        # Return default on cache failure (graceful degradation)
        return default

    return result  # Return cached value or default


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_set(
    key: str,
    value: Any,
    cache_type: CacheType = CacheType.LOCAL,
    ttl: int | None = None,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,
) -> bool:
    """Set value in both cache layers.

    Pure function replacement for CacheManager.set().

    Args:
        key: Cache key
        value: Value to cache (must be serializable)
        cache_type: Type of cached data for TTL selection
        ttl: Custom TTL in seconds (overrides cache_type default)
        cache_client: Injected cache manager

    Returns:
        True if successful in at least one cache layer
    """

    success = False
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache set")
            return False

        success = await cache_client.set(key, value, cache_type, ttl)

        if success:
            logger.debug("Cache set successful for key: %s", key)
        else:
            logger.warning("Cache set failed for key: %s", key)

        return success

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Cache set failed for key: %s", key)
        # Don't raise exception for cache failures
        return False


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_delete(
    key: str,
    cache_type: CacheType = CacheType.LOCAL,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> bool:
    """Delete value from both cache layers.

    Pure function replacement for CacheManager.delete().

    Args:
        key: Cache key
        cache_type: Type of cached data
        cache_client: Injected cache manager

    Returns:
        True if successful
    """

    success = False
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache delete")
            return False

        success = await cache_client.delete(key, cache_type)

        if success:
            logger.debug("Cache delete successful for key: %s", key)
        else:
            logger.warning("Cache delete failed for key: %s", key)

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Cache delete failed for key: %s", key)
        return False

    return success  # Return success status


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_clear(
    cache_type: CacheType | None = None,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> bool:
    """Clear cache layers.

    Pure function replacement for CacheManager.clear().

    Args:
        cache_type: Specific cache type to clear (None for all)
        cache_client: Injected cache manager

    Returns:
        True if successful
    """

    success = False
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache clear")
            return False

        success = await cache_client.clear(cache_type)

        if success:
            scope = cache_type.value if cache_type else "all"
            logger.info(
                "Cache clear successful for scope: %s",
                scope,
            )
        else:
            logger.warning(
                "Cache clear failed for cache_type: %s",
                cache_type,
            )

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Cache clear failed")
        return False

    return success  # Return success status


async def get_cache_stats(
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> dict[str, Any]:
    """Get comprehensive cache statistics.

    Pure function replacement for CacheManager.get_stats().

    Args:
        cache_client: Injected cache manager

    Returns:
        Cache statistics dictionary
    """

    try:
        if not cache_client:
            return {
                "manager": {"enabled_layers": []},
                "error": "Cache client not available",
            }

        stats = await cache_client.get_stats()
        logger.debug("Retrieved cache statistics")

    except Exception as e:
        logger.exception("Cache stats retrieval failed")
        return {
            "manager": {"enabled_layers": []},
            "error": f"Stats retrieval failed: {e!s}",
        }
    return stats  # Return stats dictionary


async def get_performance_stats(
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> dict[str, Any]:
    """Get performance-focused cache statistics.

    Args:
        cache_client: Injected cache manager

    Returns:
        Performance metrics dictionary
    """

    try:
        if not cache_client:
            return {}

        stats = await cache_client.get_performance_stats()
        logger.debug("Retrieved cache performance statistics")

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Cache performance stats retrieval failed")
        return {}

    return stats  # Return perf stats dictionary


# Specialized cache functions
async def cache_embedding(
    content_hash: str,
    model: str,
    embedding: list[float],
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> bool:
    """Cache an embedding vector.

    Specialized function for embedding cache operations.

    Args:
        content_hash: Hash of the content that was embedded
        model: Name of the embedding model used
        embedding: Embedding vector to cache
        cache_client: Injected cache manager

    Returns:
        True if successfully cached
    """

    success = False
    try:
        if not cache_client or not hasattr(cache_client, "embedding_cache"):
            return False

        if cache_client.embedding_cache:
            success = await cache_client.embedding_cache.set_embedding(
                text=content_hash,
                provider="default",
                model=model,
                dimensions=0,
                embedding=embedding,
            )
            if success:
                logger.debug(
                    "Cached embedding for model %s",
                    model,
                )

    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Embedding cache failed")
        return False

    # Return success status
    if cache_client.embedding_cache:
        return success
    return False


async def get_cached_embedding(
    content_hash: str,
    model: str,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,
) -> list[float] | None:
    """Get cached embedding vector.

    Specialized function for embedding cache retrieval.

    Args:
        content_hash: Hash of the content that was embedded
        model: Name of the embedding model used
        cache_client: Injected cache manager

    Returns:
        Cached embedding vector or None if not found
    """

    try:
        if not cache_client or not hasattr(cache_client, "embedding_cache"):
            return None

        embedding: list[float] | None = None
        if cache_client.embedding_cache:
            embedding = await cache_client.embedding_cache.get_embedding(
                text=content_hash,
                provider="default",
                model=model,
                dimensions=0,
            )
            if embedding:
                logger.debug(
                    "Retrieved cached embedding for model %s",
                    model,
                )

    except (OSError, AttributeError, ConnectionError, ImportError):
        logger.exception("Cached embedding retrieval failed")
        return None
    else:
        if cache_client.embedding_cache:
            return embedding
        return None


async def cache_search_results(
    query_hash: str,
    collection: str,
    results: list[dict[str, Any]],
    ttl: int | None = None,
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> bool:
    """Cache search results.

    Specialized function for search result caching.

    Args:
        query_hash: Hash of the search query
        collection: Name of the collection searched
        results: Search results to cache
        ttl: Custom TTL in seconds (None uses default)
        cache_client: Injected cache manager

    Returns:
        True if successfully cached
    """

    success = False
    try:
        if not cache_client or not hasattr(cache_client, "search_cache"):
            return False

        if cache_client.search_cache:  # pyright: ignore
            success = await cache_client.search_cache.set_search_results(
                query_hash,
                collection,  # pyright: ignore
                results,  # pyright: ignore
                ttl,  # pyright: ignore
            )
            if success:
                logger.debug(
                    "Cached search results for collection %s",
                    collection,
                )

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Search results cache failed")
    if cache_client.search_cache:  # pyright: ignore
        return success
    return False


async def _execute_cache_operation(
    op: dict[str, Any], cache_client: CacheManager
) -> bool:
    """Execute a single cache operation.

    Args:
        op: Operation dictionary with 'op', 'key', 'value', etc.
        cache_client: Cache client instance

    Returns:
        True if operation succeeded, False otherwise
    """

    op_type = op.get("op")
    key = op.get("key")

    if op_type == "get":
        await cache_get(
            key,
            op.get("cache_type", CacheType.CRAWL),
            op.get("default"),
            cache_client,
        )
        return True

    if op_type == "set":
        return await cache_set(
            key,
            op.get("value"),
            op.get("cache_type", CacheType.CRAWL),
            op.get("ttl"),
            cache_client,
        )

    if op_type == "delete":
        return await cache_delete(
            key,
            op.get("cache_type", CacheType.CRAWL),
            cache_client,
        )

    # Unknown operation type
    return False


def _create_error_result(
    operations: list[dict[str, Any]],
    error_msg: str,
) -> dict[str, Any]:
    """Create error result for bulk operations.

    Args:
        operations: List of operations that failed
        error_msg: Error message

    Returns:
        Error result dictionary
    """

    return {
        "total": len(operations),
        "successful": 0,
        "failed": len(operations),
        "errors": [error_msg],
    }


# Bulk operations function (new functionality)
async def bulk_cache_operations(
    operations: list[dict[str, Any]],
    cache_client: Annotated[CacheManager | None, Depends(get_cache_client)] = None,  # pyright: ignore
) -> dict[str, Any]:
    """Perform multiple cache operations in batch.

    New function-based capability for efficient bulk operations.

    Args:
        operations: List of operation dictionaries with 'op', 'key', 'value', etc.
        cache_client: Injected cache manager

    Returns:
        Results summary with success/failure counts
    """

    try:
        if not cache_client:
            return _create_error_result(operations, "Cache client not available")

        results = {"total": len(operations), "successful": 0, "failed": 0, "errors": []}

        for i, op in enumerate(operations):
            try:
                success = await _execute_cache_operation(op, cache_client)
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    op_type = op.get("op", "unknown")
                    results["errors"].append(f"Unknown operation: {op_type}")

            except (OSError, PermissionError, ValueError) as e:
                results["failed"] += 1
                results["errors"].append(f"Operation {i} failed: {e!s}")

        logger.info(
            "Bulk cache operations completed: %d/%d successful",
            results["successful"],
            results["total"],
        )

    except Exception as e:
        logger.exception("Bulk cache operations failed")
        return _create_error_result(operations, f"Bulk operation failed: {e!s}")
    else:
        return results
