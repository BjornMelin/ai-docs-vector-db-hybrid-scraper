"""Search result caching utilities backed by Dragonfly."""

import logging
from typing import Any

from ._bulk_delete import delete_in_batches
from .dragonfly_cache import DragonflyCache
from .key_utils import build_search_cache_key


logger = logging.getLogger(__name__)


class SearchResultCache:
    """Cache search results with invalidation and popularity tracking.

    Features:
    - Medium TTL (1 hour) for search results
    - Popularity-based cache adjustment
    - Invalidation by collection
    - Query parameter normalization
    - Efficient batch operations
    """

    def __init__(self, cache: DragonflyCache, default_ttl: int = 3600):
        """Initialize search result cache.

        Args:
            cache: DragonflyDB cache instance.
            default_ttl: Default TTL in seconds (1 hour for search results).
        """
        self.cache = cache
        self.default_ttl = default_ttl

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def get_search_results(
        self,
        query: str,
        collection_name: str = "default",
        filters: dict | None = None,
        limit: int = 10,
        search_type: str = "hybrid",
        params: dict | None = None,
    ) -> list[dict] | None:
        """Get cached search results.

        Args:
            query: Search query text.
            collection_name: Qdrant collection name.
            filters: Search filters.
            limit: Number of results.
            search_type: Type of search (dense, sparse, hybrid).
            params: Additional search parameters.

        Returns:
            Cached search results or None if not found.
        """
        key = self._build_search_key(
            query, collection_name, filters, limit, search_type, params or {}
        )

        try:
            cached = await self.cache.get(key)
            if cached is not None:
                logger.debug("Search cache hit for query: %s...", query[:50])
                await self._increment_query_popularity(query)
                return cached

            logger.debug("Search cache miss for query: %s...", query[:50])
            return None

        except (ConnectionError, RuntimeError, TimeoutError):
            logger.exception("Error retrieving search results from cache")
            return None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def set_search_results(
        self,
        query: str,
        results: list[dict],
        collection_name: str = "default",
        filters: dict | None = None,
        limit: int = 10,
        search_type: str = "hybrid",
        ttl: int | None = None,
        params: dict | None = None,
    ) -> bool:
        """Cache search results with TTL adjustment.

        Args:
            query: Search query text.
            results: Search results to cache.
            collection_name: Qdrant collection name.
            filters: Search filters.
            limit: Number of results.
            search_type: Type of search.
            ttl: Custom TTL (uses popularity-adjusted default if None).
            params: Additional search parameters.

        Returns:
            ``True`` when the cache write succeeded, otherwise ``False``.
        """
        key = self._build_search_key(
            query, collection_name, filters, limit, search_type, params or {}
        )

        try:
            # Adjust TTL based on query popularity if not specified
            if ttl is None:
                popularity = await self._get_query_popularity(query)

                if popularity > 10:
                    # Popular queries get shorter TTL for fresher results
                    cache_ttl = self.default_ttl // 2
                    logger.debug("Popular query (%s hits): shorter TTL", popularity)
                elif popularity > 5:
                    # Moderately popular queries get normal TTL
                    cache_ttl = self.default_ttl
                else:
                    # Unpopular queries get longer TTL
                    cache_ttl = self.default_ttl * 2
                    logger.debug("Unpopular query (%s hits): longer TTL", popularity)
            else:
                cache_ttl = ttl

            success = await self.cache.set(key, results, ttl=cache_ttl)

            if success:
                logger.debug(
                    "Cached %s search results for query: %s... (TTL: %ss)",
                    len(results),
                    query[:50],
                    cache_ttl,
                )

                # Track query for popularity statistics
                await self._increment_query_popularity(query)

            return success

        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error caching search results")
            return False

    async def invalidate_by_collection(self, collection_name: str) -> int:
        """Invalidate all cached searches for a collection.

        Useful when collection data is updated.

        Args:
            collection_name: Collection name to invalidate.

        Returns:
            Number of entries invalidated.
        """
        try:
            pattern = f"search:{collection_name}:*"

            # Use DragonflyDB's efficient SCAN for pattern matching
            keys = await self.cache.scan_keys(pattern)
            deleted_count = await delete_in_batches(self.cache, keys)

            if deleted_count:
                logger.info(
                    "Invalidated %s search cache entries for collection: %s",
                    deleted_count,
                    collection_name,
                )

            return deleted_count

        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error invalidating collection cache")
            return 0

    async def invalidate_by_query_pattern(self, query_pattern: str) -> int:
        """Invalidate cached searches matching a query pattern.

        Args:
            query_pattern: Query pattern to match (supports wildcards).

        Returns:
            Number of entries invalidated.
        """
        try:
            # Create hash pattern for query matching
            digest = build_search_cache_key(query_pattern).partition(":")[2]
            pattern = f"search:*:{digest}*"

            keys = await self.cache.scan_keys(pattern)

            if keys:
                results = await self.cache.delete_many(keys)
                deleted_count = sum(results.values())

                logger.info(
                    "Invalidated %s search cache entries matching: %s",
                    deleted_count,
                    query_pattern,
                )
                return deleted_count

            return 0

        except (ConnectionError, RuntimeError, TimeoutError):
            logger.exception("Error invalidating query pattern cache")
            return 0

    async def get_popular_queries(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most popular queries with their hit counts.

        Args:
            limit: Number of top queries to return.

        Returns:
            List of (query, count) tuples sorted by popularity.
        """
        try:
            pattern = "popular:*"
            keys = await self.cache.scan_keys(pattern)

            # Get counts for all popular queries
            if keys:
                counts = await self.cache.mget(keys)

                # Combine keys and counts, sort by count
                query_counts = []
                for key, count in zip(keys, counts, strict=False):
                    if count is not None:
                        # Extract query from key (remove "popular:" prefix)
                        query_hash = key.replace("popular:", "")
                        query_counts.append((query_hash, int(count)))

                # Sort by count descending
                query_counts.sort(key=lambda x: x[1], reverse=True)

                return query_counts[:limit]

            return []

        except (ConnectionError, RuntimeError, TimeoutError):
            logger.exception("Error getting popular queries")
            return []

    async def cleanup_expired_popularity(self) -> int:
        """Clean up expired popularity counters.

        Returns:
            Number of expired entries cleaned up.
        """
        try:
            pattern = "popular:*"
            keys = await self.cache.scan_keys(pattern)

            expired_keys = []
            for key in keys:
                ttl = await self.cache.ttl(key)
                if ttl <= 0:  # Expired or no TTL
                    expired_keys.append(key)

            if expired_keys:
                results = await self.cache.delete_many(expired_keys)
                deleted_count = sum(results.values())

                logger.info("Cleaned up %s expired popularity counters", deleted_count)
                return deleted_count

            return 0

        except (ConnectionError, RuntimeError, TimeoutError):
            logger.exception("Error cleaning up expired popularity")
            return 0

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get search cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        try:
            # Count search result keys
            search_keys = await self.cache.scan_keys("search:*")
            popularity_keys = await self.cache.scan_keys("popular:*")

            stats: dict[str, Any] = {
                "total_search_results": len(search_keys),
                "popularity_counters": len(popularity_keys),
                "cache_size": await self.cache.size(),
            }

            # Group by collection if possible
            collections = {}

            for key in search_keys:
                try:
                    # Parse key format: search:{collection}:{hash}
                    parts = key.split(":")
                    if len(parts) >= 3:
                        collection = parts[1]
                        collections[collection] = collections.get(collection, 0) + 1

                except (OSError, PermissionError):
                    continue

            stats["by_collection"] = collections

            # Get popular queries
            popular = await self.get_popular_queries(limit=5)
            stats["top_queries"] = popular

            return stats

        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.exception("Error getting cache stats")
            return {"error": str(e)}

    async def get_stats(self) -> dict[str, Any]:
        """Alias for get_cache_stats for compatibility."""
        return await self.get_cache_stats()

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _build_search_key(
        self,
        query: str,
        collection_name: str,
        filters: dict | None,
        limit: int,
        search_type: str,
        params: dict | None,
    ) -> str:
        """Return deterministic key for search cache entries."""
        payload = {
            "collection": collection_name,
            "filters": filters or {},
            "limit": limit,
            "search_type": search_type,
            "params": params or {},
        }
        digest = build_search_cache_key(query, payload).partition(":")[2]
        return f"search:{collection_name}:{digest}"

    def _get_search_key(
        self,
        query: str,
        collection_name: str,
        filters: dict | None,
        limit: int,
        search_type: str,
        **params: Any,
    ) -> str:
        """Backward-compatible wrapper for key generation."""
        return self._build_search_key(
            query, collection_name, filters, limit, search_type, params or {}
        )

    async def _get_query_popularity(self, query: str) -> int:
        """Get query popularity count.

        Args:
            query: Query text.

        Returns:
            Number of times query was accessed.
        """
        try:
            digest = build_search_cache_key(query)
            key = f"popular:{digest.partition(':')[2]}"
            count = await self.cache.get(key)
            return int(count) if count else 0

        except (ConnectionError, OSError, PermissionError) as e:
            logger.debug("Error getting query popularity: %s", e)
            return 0

    async def _increment_query_popularity(self, query: str) -> None:
        """Track query popularity for cache optimization.

        Args:
            query: Query text.
        """
        try:
            digest = build_search_cache_key(query)
            key = f"popular:{digest.partition(':')[2]}"

            # Use atomic increment
            client = self.cache.client
            current = await client.incr(key)

            # Set TTL on first increment (reset daily)
            if current == 1:
                await client.expire(key, 86400)  # 24 hours

        except (ConnectionError, OSError, TimeoutError) as e:
            logger.debug("Error tracking query popularity: %s", e)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def warm_popular_searches(
        self,
        queries: list[str],
        collection_name: str = "default",
        search_func: Any = None,
    ) -> int:
        """Pre-warm cache with popular search queries.

        Args:
            queries: List of popular query texts
            collection_name: Collection to search
            search_func: Function to execute searches (optional)

        Returns:
            Number of queries warmed
        """
        if not queries or not search_func:
            return 0

        warmed_count = 0

        try:
            for query in queries:
                # Check if already cached
                key = self._build_search_key(
                    query, collection_name, None, 10, "hybrid", None
                )
                exists = await self.cache.exists(key)

                if not exists:
                    try:
                        # Execute search and cache result
                        results = await search_func(query, collection_name)
                        if results:
                            await self.set_search_results(
                                query, results, collection_name
                            )
                            warmed_count += 1
                            logger.debug("Warmed search cache for: %s...", query[:50])

                    except (ConnectionError, OSError, PermissionError) as e:
                        logger.warning("Failed to warm search for '%s': %s", query, e)

            logger.info(
                "Search cache warming completed: %s/%s queries",
                warmed_count,
                len(queries),
            )
            return warmed_count

        except (ConnectionError, RuntimeError, TimeoutError):
            logger.exception("Error in search cache warming")
            return warmed_count
