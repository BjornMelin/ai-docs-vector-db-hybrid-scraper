"""Two-tier cache manager orchestrating local and distributed caches."""

import asyncio
import hashlib
import logging
from enum import Enum
from typing import Any

from .local_cache import LocalCache
from .metrics import CacheMetrics
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cacheable data."""

    EMBEDDINGS = "embeddings"
    CRAWL_RESULTS = "crawl_results"
    QUERY_RESULTS = "query_results"


class CacheManager:
    """Two-tier cache manager with L1 (local) and L2 (Redis) caches."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        enable_local_cache: bool = True,
        enable_redis_cache: bool = True,
        local_max_size: int = 1000,
        local_max_memory_mb: float = 100.0,
        local_ttl_seconds: int = 300,  # 5 minutes
        redis_ttl_seconds: dict[CacheType, int] | None = None,
        key_prefix: str = "aidocs:",
        enable_metrics: bool = True,
    ):
        """Initialize cache manager.

        Args:
            redis_url: Redis connection URL
            enable_local_cache: Enable L1 local cache
            enable_redis_cache: Enable L2 Redis cache
            local_max_size: Maximum entries in local cache
            local_max_memory_mb: Maximum memory for local cache
            local_ttl_seconds: Default TTL for local cache
            redis_ttl_seconds: TTL mapping for Redis by cache type
            key_prefix: Prefix for all cache keys
            enable_metrics: Enable metrics collection
        """
        self.enable_local_cache = enable_local_cache
        self.enable_redis_cache = enable_redis_cache
        self.key_prefix = key_prefix

        # Default Redis TTLs by cache type
        self.redis_ttl_seconds = redis_ttl_seconds or {
            CacheType.EMBEDDINGS: 86400,  # 24 hours
            CacheType.CRAWL_RESULTS: 3600,  # 1 hour
            CacheType.QUERY_RESULTS: 7200,  # 2 hours
        }

        # Initialize caches
        self.local_cache: LocalCache | None = None
        if enable_local_cache:
            self.local_cache = LocalCache(
                max_size=local_max_size,
                default_ttl=local_ttl_seconds,
                max_memory_mb=local_max_memory_mb,
            )

        self.redis_cache: RedisCache | None = None
        if enable_redis_cache:
            self.redis_cache = RedisCache(
                redis_url=redis_url,
                key_prefix=key_prefix,
            )

        # Initialize metrics
        self.metrics: CacheMetrics | None = None
        if enable_metrics:
            self.metrics = CacheMetrics()

    async def get_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
    ) -> Any | None:
        """Get embedding from cache.

        Args:
            text: Text that was embedded
            provider: Embedding provider
            model: Model name
            dimensions: Embedding dimensions

        Returns:
            Cached embedding or None
        """
        key = self._make_embedding_key(text, provider, model, dimensions)
        return await self._get(key, CacheType.EMBEDDINGS)

    async def set_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
        embedding: Any,
    ) -> bool:
        """Cache embedding.

        Args:
            text: Text that was embedded
            provider: Embedding provider
            model: Model name
            dimensions: Embedding dimensions
            embedding: Embedding vector

        Returns:
            Success status
        """
        key = self._make_embedding_key(text, provider, model, dimensions)
        return await self._set(key, embedding, CacheType.EMBEDDINGS)

    async def get_crawl_result(
        self,
        url: str,
        provider: str,
    ) -> Any | None:
        """Get crawl result from cache.

        Args:
            url: URL that was crawled
            provider: Crawl provider

        Returns:
            Cached crawl result or None
        """
        key = self._make_crawl_key(url, provider)
        return await self._get(key, CacheType.CRAWL_RESULTS)

    async def set_crawl_result(
        self,
        url: str,
        provider: str,
        result: Any,
    ) -> bool:
        """Cache crawl result.

        Args:
            url: URL that was crawled
            provider: Crawl provider
            result: Crawl result data

        Returns:
            Success status
        """
        key = self._make_crawl_key(url, provider)
        return await self._set(key, result, CacheType.CRAWL_RESULTS)

    async def get_query_result(
        self,
        query: str,
        **params: Any,
    ) -> Any | None:
        """Get query result from cache.

        Args:
            query: Search query
            **params: Additional query parameters

        Returns:
            Cached query result or None
        """
        key = self._make_query_key(query, **params)
        return await self._get(key, CacheType.QUERY_RESULTS)

    async def set_query_result(
        self,
        query: str,
        result: Any,
        **params: Any,
    ) -> bool:
        """Cache query result.

        Args:
            query: Search query
            result: Query result data
            **params: Additional query parameters

        Returns:
            Success status
        """
        key = self._make_query_key(query, **params)
        return await self._set(key, result, CacheType.QUERY_RESULTS)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Key pattern (e.g., "crawl:*:firecrawl")

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Clear from local cache (simple pattern matching)
        if self.local_cache:
            # Local cache doesn't support pattern matching well
            # Would need to iterate all keys
            await self.local_cache.clear()
            count += 1  # Approximate

        # Clear from Redis cache
        if self.redis_cache:
            try:
                client = await self.redis_cache.client
                full_pattern = f"{self.key_prefix}{pattern}"

                async for key in client.scan_iter(match=full_pattern, count=100):
                    await client.delete(key)
                    count += 1

            except Exception as e:
                logger.error(f"Error invalidating pattern {pattern}: {e}")

        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": {
                "local": self.enable_local_cache,
                "redis": self.enable_redis_cache,
            },
        }

        if self.local_cache:
            stats["local"] = self.local_cache.get_stats()

        if self.redis_cache:
            stats["redis"] = {
                "size": await self.redis_cache.size(),
            }

        if self.metrics:
            stats["metrics"] = self.metrics.get_summary()

        return stats

    async def close(self) -> None:
        """Close all cache connections."""
        if self.local_cache:
            await self.local_cache.close()

        if self.redis_cache:
            await self.redis_cache.close()

    # Private methods
    async def _get(self, key: str, cache_type: CacheType) -> Any | None:
        """Get value from cache hierarchy."""
        start_time = asyncio.get_event_loop().time()

        # Try L1 (local) cache first
        if self.local_cache:
            value = await self.local_cache.get(key)
            if value is not None:
                if self.metrics:
                    latency = asyncio.get_event_loop().time() - start_time
                    self.metrics.record_hit(cache_type.value, "local", latency)
                return value

        # Try L2 (Redis) cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Populate L1 cache
                if self.local_cache:
                    await self.local_cache.set(key, value)

                if self.metrics:
                    latency = asyncio.get_event_loop().time() - start_time
                    self.metrics.record_hit(cache_type.value, "redis", latency)
                return value

        # Cache miss
        if self.metrics:
            latency = asyncio.get_event_loop().time() - start_time
            self.metrics.record_miss(cache_type.value, latency)

        return None

    async def _set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType,
    ) -> bool:
        """Set value in cache hierarchy."""
        start_time = asyncio.get_event_loop().time()
        success = False

        # Set in both caches concurrently
        tasks = []

        if self.local_cache:
            tasks.append(self.local_cache.set(key, value))

        if self.redis_cache:
            ttl = self.redis_ttl_seconds.get(cache_type, 3600)
            tasks.append(self.redis_cache.set(key, value, ttl))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = any(
                result is True
                for result in results
                if not isinstance(result, Exception)
            )

        if self.metrics:
            latency = asyncio.get_event_loop().time() - start_time
            self.metrics.record_set(cache_type.value, latency, success)

        return success

    def _make_embedding_key(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
    ) -> str:
        """Create cache key for embeddings."""
        # Normalize text for consistent hashing
        normalized_text = text.lower().strip()
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()

        return f"emb:{text_hash}:{provider}:{model}:{dimensions}"

    def _make_crawl_key(self, url: str, provider: str) -> str:
        """Create cache key for crawl results."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"crawl:{url_hash}:{provider}"

    def _make_query_key(self, query: str, **params: Any) -> str:
        """Create cache key for query results."""
        # Include query and parameters in key
        key_parts = [query]
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")

        key_str = "|".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return f"query:{key_hash}"
