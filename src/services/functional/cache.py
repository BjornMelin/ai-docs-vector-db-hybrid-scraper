"""Function-based cache service with FastAPI dependency injection.

Transforms the CacheManager class into pure functions with dependency injection.
Provides simple cache operations with circuit breaker patterns.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends

from src.config import CacheType

from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_cache_client


logger = logging.getLogger(__name__)


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_get(
    key: str,
    cache_type: CacheType = CacheType.LOCAL,
    default: Any = None,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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

    Raises:
        HTTPException: If cache operation fails critically

    """
    try:
        if not cache_client:
            logger.warning("Cache client not available, returning default")
            return default

        result = await cache_client.get(key, cache_type, default)

        if result != default:
            logger.debug(
                f"Cache hit for key: {key}"
            )  # TODO: Convert f-string to logging format
        else:
            logger.debug(
                f"Cache miss for key: {key}"
            )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception(f"Cache get failed for key {key}")
        # Return default on cache failure (graceful degradation)
        return default
    else:
        return result


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_set(
    key: str,
    value: Any,
    cache_type: CacheType = CacheType.LOCAL,
    ttl: int | None = None,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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

    Raises:
        HTTPException: If cache operation fails critically

    """
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache set")
            return False

        success = await cache_client.set(key, value, cache_type, ttl)

        if success:
            logger.debug(
                f"Cache set successful for key: {key}"
            )  # TODO: Convert f-string to logging format
        else:
            logger.warning(
                f"Cache set failed for key: {key}"
            )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception(f"Cache set failed for key {key}")
        # Don't raise exception for cache failures
        return False
    else:
        return success


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_delete(
    key: str,
    cache_type: CacheType = CacheType.LOCAL,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
) -> bool:
    """Delete value from both cache layers.

    Pure function replacement for CacheManager.delete().

    Args:
        key: Cache key
        cache_type: Type of cached data
        cache_client: Injected cache manager

    Returns:
        True if successful

    Raises:
        HTTPException: If cache operation fails critically

    """
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache delete")
            return False

        success = await cache_client.delete(key, cache_type)

        if success:
            logger.debug(
                f"Cache delete successful for key: {key}"
            )  # TODO: Convert f-string to logging format
        else:
            logger.warning(
                f"Cache delete failed for key: {key}"
            )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception(f"Cache delete failed for key {key}")
        return False
    else:
        return success


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def cache_clear(
    cache_type: CacheType | None = None,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
) -> bool:
    """Clear cache layers.

    Pure function replacement for CacheManager.clear().

    Args:
        cache_type: Specific cache type to clear (None for all)
        cache_client: Injected cache manager

    Returns:
        True if successful

    Raises:
        HTTPException: If cache operation fails critically

    """
    try:
        if not cache_client:
            logger.warning("Cache client not available, skipping cache clear")
            return False

        success = await cache_client.clear(cache_type)

        if success:
            scope = cache_type.value if cache_type else "all"
            logger.info(
                f"Cache clear successful for scope: {scope}"
            )  # TODO: Convert f-string to logging format
        else:
            logger.warning(
                f"Cache clear failed for cache_type: {cache_type}"
            )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception("Cache clear failed")
        return False
    else:
        return success


async def get_cache_stats(
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
) -> dict[str, Any]:
    """Get comprehensive cache statistics.

    Pure function replacement for CacheManager.get_stats().

    Args:
        cache_client: Injected cache manager

    Returns:
        Cache statistics dictionary

    Raises:
        HTTPException: If stats retrieval fails

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
    else:
        return stats


async def get_performance_stats(
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
) -> dict[str, Any]:
    """Get performance-focused cache statistics.

    Pure function replacement for CacheManager.get_performance_stats().

    Args:
        cache_client: Injected cache manager

    Returns:
        Performance metrics dictionary

    Raises:
        HTTPException: If performance stats retrieval fails

    """
    try:
        if not cache_client:
            return {}

        stats = await cache_client.get_performance_stats()
        logger.debug("Retrieved cache performance statistics")

    except Exception:
        logger.exception("Cache performance stats retrieval failed")
        return {}
    else:
        return stats


# Specialized cache functions
async def cache_embedding(
    content_hash: str,
    model: str,
    embedding: list[float],
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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
    try:
        if not cache_client or not hasattr(cache_client, "embedding_cache"):
            return False

        if cache_client.embedding_cache:
            success = await cache_client.embedding_cache.set_embedding(
                content_hash, model, embedding
            )
            if success:
                logger.debug(
                    f"Cached embedding for model {model}"
                )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception("Embedding cache failed")
        return False
    else:
        if cache_client.embedding_cache:
            return success
        return False


async def get_cached_embedding(
    content_hash: str,
    model: str,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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

        if cache_client.embedding_cache:
            embedding = await cache_client.embedding_cache.get_embedding(
                content_hash, model
            )
            if embedding:
                logger.debug(
                    f"Retrieved cached embedding for model {model}"
                )  # TODO: Convert f-string to logging format

    except Exception:
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
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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
    try:
        if not cache_client or not hasattr(cache_client, "search_cache"):
            return False

        if cache_client.search_cache:
            success = await cache_client.search_cache.set_search_results(
                query_hash, collection, results, ttl
            )
            if success:
                logger.debug(
                    f"Cached search results for collection {collection}"
                )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception("Search results cache failed")
        return False
    else:
        if cache_client.search_cache:
            return success
        return False


async def get_cached_search_results(
    query_hash: str,
    collection: str,
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
) -> list[dict[str, Any]] | None:
    """Get cached search results.

    Specialized function for search result cache retrieval.

    Args:
        query_hash: Hash of the search query
        collection: Name of the collection searched
        cache_client: Injected cache manager

    Returns:
        Cached search results or None if not found

    """
    try:
        if not cache_client or not hasattr(cache_client, "search_cache"):
            return None

        if cache_client.search_cache:
            results = await cache_client.search_cache.get_search_results(
                query_hash, collection
            )
            if results:
                logger.debug(
                    f"Retrieved cached search results for collection {collection}"
                )

    except Exception:
        logger.exception("Cached search results retrieval failed")
        return None
    else:
        if cache_client.search_cache:
            return results
        return None


# Bulk operations function (new functionality)
async def bulk_cache_operations(
    operations: list[dict[str, Any]],
    cache_client: Annotated[object, Depends(get_cache_client)] = None,
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
            return {
                "total": len(operations),
                "successful": 0,
                "failed": len(operations),
                "errors": ["Cache client not available"],
            }

        results = {"total": len(operations), "successful": 0, "failed": 0, "errors": []}

        for i, op in enumerate(operations):
            try:
                op_type = op.get("op")
                key = op.get("key")

                if op_type == "get":
                    await cache_get(
                        key,
                        op.get("cache_type", CacheType.CRAWL),
                        op.get("default"),
                        cache_client,
                    )
                    results["successful"] += 1

                elif op_type == "set":
                    success = await cache_set(
                        key,
                        op.get("value"),
                        op.get("cache_type", CacheType.CRAWL),
                        op.get("ttl"),
                        cache_client,
                    )
                    if success:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1

                elif op_type == "delete":
                    success = await cache_delete(
                        key, op.get("cache_type", CacheType.CRAWL), cache_client
                    )
                    if success:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1

                else:
                    results["failed"] += 1
                    results["errors"].append(f"Unknown operation: {op_type}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Operation {i} failed: {e!s}")

        logger.info(
            f"Bulk cache operations completed: "
            f"{results['successful']}/{results['total']} successful"
        )

    except Exception as e:
        logger.exception("Bulk cache operations failed")
        return {
            "total": len(operations),
            "successful": 0,
            "failed": len(operations),
            "errors": [f"Bulk operation failed: {e!s}"],
        }
    else:
        return results
