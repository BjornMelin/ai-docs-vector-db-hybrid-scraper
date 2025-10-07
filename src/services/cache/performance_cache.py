"""High-performance multi-tier caching system.

This module implements an intelligent caching architecture with L1 (in-memory)
and L2 (Redis) tiers, automatic cache warming, and comprehensive performance tracking.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import redis.asyncio as redis


logger = logging.getLogger(__name__)


class SearchService(Protocol):
    """Minimal async search interface for cache warming."""

    async def search(
        self, query: str
    ) -> Any:  # pragma: no cover - behavioural contract
        """Execute a search request for the supplied query."""
        raise NotImplementedError


@dataclass
class CacheMetrics:
    """Cache performance metrics tracking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    avg_retrieval_time: float = 0.0
    l1_hits: int = 0
    l2_hits: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate."""
        return self.l1_hits / self.hits if self.hits > 0 else 0.0


class PerformanceCache:
    """High-performance multi-tier caching system."""

    def __init__(self, redis_url: str, max_l1_size: int = 10000):
        """Initialize the performance cache.

        Args:
            redis_url: Redis connection URL for L2 cache
            max_l1_size: Maximum number of items in L1 cache

        """
        self.l1_cache: dict[str, Any] = {}  # In-memory LRU
        self.l2_redis: redis.Redis | None = None
        self.redis_url = redis_url
        self.metrics = CacheMetrics()
        self.max_l1_size = max_l1_size
        self.l1_access_times: dict[str, float] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection and cache warming."""
        if self._initialized:
            return

        await self._initialize_redis_connection()
        self._initialized = True

    async def _initialize_redis_connection(self) -> None:
        """Initialize Redis connection with error handling."""
        try:
            self.l2_redis = redis.from_url(self.redis_url, decode_responses=True)
            await self.l2_redis.ping()
            logger.info("PerformanceCache initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Redis cache: %s", e)
            # Continue without Redis for graceful degradation
            self.l2_redis = None

    async def get(self, key: str) -> Any | None:
        """Multi-tier cache retrieval with performance tracking.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found, None otherwise

        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # L1 Cache (in-memory) - fastest
        if key in self.l1_cache:
            self.metrics.hits += 1
            self.metrics.l1_hits += 1
            self._update_access_time(key)
            retrieval_time = (time.time() - start_time) * 1000
            self._update_avg_retrieval_time(retrieval_time)
            return self.l1_cache[key]

        # L2 Cache (Redis) - distributed
        l2_result = await self._try_l2_cache_get(key, start_time)
        if l2_result is not None:
            return l2_result

        # Cache miss
        self.metrics.misses += 1
        return None

    async def _try_l2_cache_get(self, key: str, start_time: float) -> Any | None:
        """Try to get value from L2 cache with promotion to L1."""
        if not self.l2_redis:
            return None

        try:
            value = await self.l2_redis.get(key)
        except Exception as e:
            logger.warning("L2 cache retrieval failed for key '%s': %s", key, e)
            return None

        if value:
            return await self._process_l2_cache_hit(key, value, start_time)
        return None

    async def _process_l2_cache_hit(
        self, key: str, value: str, start_time: float
    ) -> Any:
        """Process L2 cache hit by promoting to L1 and updating metrics."""
        self.metrics.hits += 1
        # Promote to L1 cache
        deserialized_value = json.loads(value)
        await self._set_l1(key, deserialized_value)
        retrieval_time = (time.time() - start_time) * 1000
        self._update_avg_retrieval_time(retrieval_time)
        return deserialized_value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        tier_preference: str = "both",
    ) -> None:
        """Set value in appropriate cache tier(s).

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tier_preference: Cache tier preference ('l1', 'l2', 'both')

        """
        if not self._initialized:
            await self.initialize()

        # Serialize value for consistency
        try:
            serialized_value = (
                json.dumps(value) if not isinstance(value, str) else value
            )
        except (TypeError, ValueError):
            logger.warning(
                "Failed to serialize value for key '%s', skipping cache", key
            )
            return

        # L1 Cache
        if tier_preference in ["l1", "both"]:
            await self._set_l1(key, value)

        # L2 Cache (Redis)
        if tier_preference in ["l2", "both"] and self.l2_redis:
            try:
                await self.l2_redis.setex(key, ttl, serialized_value)
            except Exception as e:
                logger.warning("L2 cache set failed for key '%s': %s", key, e)

    async def _set_l1(self, key: str, value: Any) -> None:
        """Set value in L1 cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache

        """
        if len(self.l1_cache) >= self.max_l1_size:
            # LRU eviction
            oldest_key = min(
                self.l1_access_times.keys(),
                key=lambda k: self.l1_access_times[k],
            )
            del self.l1_cache[oldest_key]
            del self.l1_access_times[oldest_key]
            self.metrics.evictions += 1

        self.l1_cache[key] = value
        self._update_access_time(key)

    def _update_access_time(self, key: str) -> None:
        """Update access time for LRU tracking.

        Args:
            key: Cache key that was accessed

        """
        self.l1_access_times[key] = time.time()

    def _update_avg_retrieval_time(self, retrieval_time: float) -> None:
        """Update average retrieval time metric.

        Args:
            retrieval_time: Time taken for retrieval in milliseconds

        """
        total_requests = self.metrics.hits + self.metrics.misses
        if total_requests > 0:
            self.metrics.avg_retrieval_time = (
                self.metrics.avg_retrieval_time * (total_requests - 1) + retrieval_time
            ) / total_requests

    async def warm_cache(
        self,
        popular_queries: list[str],
        *,
        search_service: SearchService,
    ) -> None:
        """Proactively warm cache with popular queries using supplied search service.

        Args:
            popular_queries: List of popular search queries to pre-cache
            search_service: Search service used to fetch uncached results

        """
        if not self._initialized:
            await self.initialize()

        # Warm cache for each query
        warmed_count = await self._warm_queries(popular_queries, search_service)
        logger.info("Cache warming completed for %d queries", warmed_count)

    async def _warm_queries(
        self,
        queries: list[str],
        search_service: SearchService,
    ) -> int:
        """Warm cache for list of queries."""
        warmed_count = 0
        for query in queries:
            if await self._warm_single_query(query, search_service):
                warmed_count += 1
        return warmed_count

    async def _warm_single_query(
        self,
        query: str,
        search_service: SearchService,
    ) -> bool:
        """Warm cache for a single query."""
        cache_key = f"search:{hashlib.sha256(query.encode()).hexdigest()}"

        # Check if already cached
        if await self.get(cache_key) is not None:
            return False

        try:
            result = await search_service.search(query)
            if result is not None:
                await self.set(cache_key, result, ttl=7200)
            return True
        except Exception as e:
            logger.warning("Failed to warm cache for query '%s': %s", query, e)
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dict containing cache performance metrics

        """
        l2_info = {}
        if self.l2_redis:
            try:
                l2_info = await self.l2_redis.info("memory")
            except Exception as e:
                logger.warning("Failed to get L2 cache stats: %s", e)

        return {
            "l1_cache": {
                "size": len(self.l1_cache),
                "max_size": self.max_l1_size,
                "utilization": len(self.l1_cache) / self.max_l1_size,
            },
            "l2_cache": {
                "connected": self.l2_redis is not None,
                "memory_info": l2_info,
            },
            "performance": {
                "total_hits": self.metrics.hits,
                "total_misses": self.metrics.misses,
                "hit_rate": self.metrics.hit_rate,
                "l1_hit_rate": self.metrics.l1_hit_rate,
                "evictions": self.metrics.evictions,
                "avg_retrieval_time_ms": self.metrics.avg_retrieval_time,
            },
        }

    async def clear_cache(self, pattern: str | None = None) -> None:
        """Clear cache entries matching pattern.

        Args:
            pattern: Optional pattern to match keys (None clears all)

        """
        if pattern:
            # Clear matching L1 entries
            keys_to_remove = [key for key in self.l1_cache if pattern in key]
            for key in keys_to_remove:
                del self.l1_cache[key]
                if key in self.l1_access_times:
                    del self.l1_access_times[key]

            # Clear matching L2 entries
            if self.l2_redis:
                try:
                    async for key in self.l2_redis.scan_iter(match=f"*{pattern}*"):
                        await self.l2_redis.delete(key)
                except Exception as e:
                    logger.warning(
                        "Failed to clear L2 cache pattern '%s': %s", pattern, e
                    )
        else:
            # Clear all caches
            self.l1_cache.clear()
            self.l1_access_times.clear()

            if self.l2_redis:
                try:
                    await self.l2_redis.flushdb()
                except Exception as e:
                    logger.warning("Failed to clear L2 cache: %s", e)

        logger.info("Cache cleared with pattern: %s", pattern or "all")

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        if self.l2_redis:
            try:
                await self.l2_redis.close()
            except Exception as e:
                logger.warning("Error closing Redis connection: %s", e)

        self.l1_cache.clear()
        self.l1_access_times.clear()
        self._initialized = False
        logger.info("PerformanceCache cleanup completed")


# Factory function for easy integration
async def create_performance_cache(
    redis_url: str, max_l1_size: int = 10000
) -> PerformanceCache:
    """Create and initialize a PerformanceCache instance.

    Args:
        redis_url: Redis connection URL
        max_l1_size: Maximum L1 cache size

    Returns:
        Initialized PerformanceCache instance

    """
    cache = PerformanceCache(redis_url, max_l1_size)
    await cache.initialize()
    return cache
