import typing
"""Specialized cache for search results with intelligent invalidation."""

import hashlib
import json  # noqa: PLC0415
import logging  # noqa: PLC0415
from typing import Any

from .dragonfly_cache import DragonflyCache

logger = logging.getLogger(__name__)


class SearchResultCache:
    """Cache search results with intelligent invalidation and popularity tracking.

    Optimized for:
    - Medium TTL (1 hour) for search results
    - Popularity-based cache adjustment
    - Intelligent invalidation by collection
    - Query parameter normalization
    - Efficient batch operations
    """

    def __init__(self, cache: DragonflyCache, default_ttl: int = 3600):
        """Initialize search result cache.

        Args:
            cache: DragonflyDB cache instance
            default_ttl: Default TTL in seconds (1 hour for search results)
        """
        self.cache = cache
        self.default_ttl = default_ttl

    async def get_search_results(
        self,
        query: str,
        collection_name: str = "default",
        filters: dict | None = None,
        limit: int = 10,
        search_type: str = "hybrid",
        **params: Any,
    ) -> list[dict] | None:
        """Get cached search results.

        Args:
            query: Search query text
            collection_name: Qdrant collection name
            filters: Search filters
            limit: Number of results
            search_type: Type of search (dense, sparse, hybrid)
            **params: Additional search parameters

        Returns:
            Cached search results or None if not found
        """
        key = self._get_search_key(
            query, collection_name, filters, limit, search_type, **params
        )

        try:
            cached = await self.cache.get(key)
            if cached:
                logger.debug(f"Search cache hit for query: {query[:50]}...")

                # Track popularity for cache optimization
                await self._increment_query_popularity(query)

                return cached

            logger.debug(f"Search cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error retrieving search results from cache: {e}")
            return None

    async def set_search_results(
        self,
        query: str,
        results: list[dict],
        collection_name: str = "default",
        filters: dict | None = None,
        limit: int = 10,
        search_type: str = "hybrid",
        ttl: int | None = None,
        **params: Any,
    ) -> bool:
        """Cache search results with intelligent TTL adjustment.

        Args:
            query: Search query text
            results: Search results to cache
            collection_name: Qdrant collection name
            filters: Search filters
            limit: Number of results
            search_type: Type of search
            ttl: Custom TTL (uses popularity-adjusted default if None)
            **params: Additional search parameters

        Returns:
            Success status
        """
        key = self._get_search_key(
            query, collection_name, filters, limit, search_type, **params
        )

        try:
            # Adjust TTL based on query popularity if not specified
            if ttl is None:
                popularity = await self._get_query_popularity(query)

                if popularity > 10:
                    # Popular queries get shorter TTL for fresher results
                    cache_ttl = self.default_ttl // 2
                    logger.debug(f"Popular query ({popularity} hits): shorter TTL")
                elif popularity > 5:
                    # Moderately popular queries get normal TTL
                    cache_ttl = self.default_ttl
                else:
                    # Unpopular queries get longer TTL
                    cache_ttl = self.default_ttl * 2
                    logger.debug(f"Unpopular query ({popularity} hits): longer TTL")
            else:
                cache_ttl = ttl

            success = await self.cache.set(key, results, ttl=cache_ttl)

            if success:
                logger.debug(
                    f"Cached {len(results)} search results for query: {query[:50]}... "
                    f"(TTL: {cache_ttl}s)"
                )

                # Track query for popularity statistics
                await self._increment_query_popularity(query)

            return success

        except Exception as e:
            logger.error(f"Error caching search results: {e}")
            return False

    async def invalidate_by_collection(self, collection_name: str) -> int:
        """Invalidate all cached searches for a collection.

        Useful when collection data is updated.

        Args:
            collection_name: Collection name to invalidate

        Returns:
            Number of entries invalidated
        """
        try:
            pattern = f"search:{collection_name}:*"

            # Use DragonflyDB's efficient SCAN for pattern matching
            keys = await self.cache.scan_keys(pattern)

            if keys:
                # Delete in batches for efficiency
                batch_size = 100
                deleted_count = 0

                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    results = await self.cache.delete_many(batch)
                    deleted_count += sum(results.values())

                logger.info(
                    f"Invalidated {deleted_count} search cache entries for collection: {collection_name}"
                )
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error invalidating collection cache: {e}")
            return 0

    async def invalidate_by_query_pattern(self, query_pattern: str) -> int:
        """Invalidate cached searches matching a query pattern.

        Args:
            query_pattern: Query pattern to match (supports wildcards)

        Returns:
            Number of entries invalidated
        """
        try:
            # Create hash pattern for query matching
            pattern_hash = hashlib.md5(query_pattern.encode()).hexdigest()
            pattern = f"search:*:{pattern_hash}*"

            keys = await self.cache.scan_keys(pattern)

            if keys:
                results = await self.cache.delete_many(keys)
                deleted_count = sum(results.values())

                logger.info(
                    f"Invalidated {deleted_count} search cache entries matching: {query_pattern}"
                )
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error invalidating query pattern cache: {e}")
            return 0

    async def get_popular_queries(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most popular queries with their hit counts.

        Args:
            limit: Number of top queries to return

        Returns:
            List of (query, count) tuples sorted by popularity
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

        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []

    async def cleanup_expired_popularity(self) -> int:
        """Clean up expired popularity counters.

        Returns:
            Number of expired entries cleaned up
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

                logger.info(f"Cleaned up {deleted_count} expired popularity counters")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error cleaning up expired popularity: {e}")
            return 0

    async def get_cache_stats(self) -> dict:
        """Get search cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Count search result keys
            search_keys = await self.cache.scan_keys("search:*")
            popularity_keys = await self.cache.scan_keys("popular:*")

            stats = {
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

                except Exception:
                    continue

            stats["by_collection"] = collections

            # Get popular queries
            popular = await self.get_popular_queries(limit=5)
            stats["top_queries"] = popular

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def _get_search_key(
        self,
        query: str,
        collection_name: str,
        filters: dict | None,
        limit: int,
        search_type: str,
        **params: Any,
    ) -> str:
        """Generate deterministic cache key for search.

        Args:
            query: Search query text
            collection_name: Collection name
            filters: Search filters
            limit: Result limit
            search_type: Type of search
            **params: Additional parameters

        Returns:
            Cache key
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Sort filters for consistency
        sorted_filters = json.dumps(filters or {}, sort_keys=True)

        # Sort additional parameters
        sorted_params = json.dumps(params, sort_keys=True)

        # Combine all parameters
        key_data = f"{normalized_query}|{collection_name}|{sorted_filters}|{limit}|{search_type}|{sorted_params}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return f"search:{collection_name}:{key_hash}"

    async def _get_query_popularity(self, query: str) -> int:
        """Get query popularity count.

        Args:
            query: Query text

        Returns:
            Number of times query was accessed
        """
        try:
            key = f"popular:{hashlib.md5(query.encode()).hexdigest()}"
            count = await self.cache.get(key)
            return int(count) if count else 0

        except Exception as e:
            logger.debug(f"Error getting query popularity: {e}")
            return 0

    async def _increment_query_popularity(self, query: str) -> None:
        """Track query popularity for cache optimization.

        Args:
            query: Query text
        """
        try:
            key = f"popular:{hashlib.md5(query.encode()).hexdigest()}"

            # Use atomic increment
            client = await self.cache.client
            current = await client.incr(key)

            # Set TTL on first increment (reset daily)
            if current == 1:
                await client.expire(key, 86400)  # 24 hours

        except Exception as e:
            logger.debug(f"Error tracking query popularity: {e}")

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
                key = self._get_search_key(query, collection_name, None, 10, "hybrid")
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
                            logger.debug(f"Warmed search cache for: {query[:50]}...")

                    except Exception as e:
                        logger.warning(f"Failed to warm search for '{query}': {e}")

            logger.info(
                f"Search cache warming completed: {warmed_count}/{len(queries)} queries"
            )
            return warmed_count

        except Exception as e:
            logger.error(f"Error in search cache warming: {e}")
            return warmed_count
