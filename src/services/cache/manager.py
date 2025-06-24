import typing
"""Simplified cache manager using DragonflyDB with specialized cache layers."""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Used only for external library compatibility

from src.config.enums import CacheType

from .dragonfly_cache import DragonflyCache
from .embedding_cache import EmbeddingCache
from .local_cache import LocalCache
from .metrics import CacheMetrics
from .search_cache import SearchResultCache

logger = logging.getLogger(__name__)

# Import monitoring registry for Prometheus integration
try:
    from ..monitoring.metrics import get_metrics_registry

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class CacheManager:
    """Simplified two-tier cache manager with DragonflyDB and specialized cache layers."""

    def __init__(
        self,
        dragonfly_url: str = "redis://localhost:6379",
        enable_local_cache: bool = True,
        enable_distributed_cache: bool = True,
        local_max_size: int = 1000,
        local_max_memory_mb: float = 100.0,
        local_ttl_seconds: int = 300,  # 5 minutes
        distributed_ttl_seconds: dict[CacheType, int] | None = None,
        key_prefix: str = "aidocs:",
        enable_metrics: bool = True,
        enable_specialized_caches: bool = True,
    ):
        """Initialize simplified cache manager with DragonflyDB.

        Args:
            dragonfly_url: DragonflyDB connection URL (Redis-compatible)
            enable_local_cache: Enable L1 local cache
            enable_distributed_cache: Enable L2 distributed cache
            local_max_size: Maximum entries in local cache
            local_max_memory_mb: Maximum memory for local cache
            local_ttl_seconds: Default TTL for local cache
            distributed_ttl_seconds: TTL mapping for distributed cache by cache type
            key_prefix: Prefix for all cache keys
            enable_metrics: Enable metrics collection
            enable_specialized_caches: Enable embedding and search result caches
        """
        self.enable_local_cache = enable_local_cache
        self.enable_distributed_cache = enable_distributed_cache
        self.key_prefix = key_prefix
        self.enable_specialized_caches = enable_specialized_caches

        # Default distributed cache TTLs by cache type
        self.distributed_ttl_seconds = distributed_ttl_seconds or {
            CacheType.EMBEDDINGS: 86400 * 7,  # 7 days for embeddings
            CacheType.CRAWL: 3600,  # 1 hour for crawl results
            CacheType.SEARCH: 3600,  # 1 hour for search results
            CacheType.HYDE: 3600,  # 1 hour for HyDE results
        }

        # Initialize local cache (L1)
        self._local_cache = None
        if enable_local_cache:
            self._local_cache = LocalCache(
                max_size=local_max_size,
                max_memory_mb=local_max_memory_mb,
                default_ttl=local_ttl_seconds,
            )

        # Initialize DragonflyDB cache (L2)
        self._distributed_cache = None
        if enable_distributed_cache:
            self._distributed_cache = DragonflyCache(
                redis_url=dragonfly_url,
                key_prefix=key_prefix,
                max_connections=50,  # Optimized for DragonflyDB
                enable_compression=True,
                compression_threshold=1024,
            )

        # Initialize specialized caches
        self._embedding_cache = None
        self._search_cache = None
        if enable_specialized_caches and self._distributed_cache:
            self._embedding_cache = EmbeddingCache(
                cache=self._distributed_cache,
                default_ttl=self.distributed_ttl_seconds[CacheType.EMBEDDINGS],
            )
            self._search_cache = SearchResultCache(
                cache=self._distributed_cache,
                default_ttl=self.distributed_ttl_seconds[CacheType.SEARCH],
            )

        # Initialize metrics
        self._metrics = CacheMetrics() if enable_metrics else None

        # Initialize Prometheus monitoring registry
        self.metrics_registry = None
        if enable_metrics and MONITORING_AVAILABLE:
            try:
                self.metrics_registry = get_metrics_registry()
                logger.info("Cache monitoring enabled with Prometheus integration")
            except Exception as e:
                logger.warning(f"Failed to initialize cache monitoring: {e}")

        logger.info(
            f"CacheManager initialized with DragonflyDB: {dragonfly_url}, "
            f"local={enable_local_cache}, specialized={enable_specialized_caches}"
        )

    @property
    def local_cache(self) -> LocalCache | None:
        """Access to local cache layer.

        Returns:
            LocalCache | None: Local cache instance if enabled, None otherwise
        """
        return self._local_cache

    @property
    def distributed_cache(self) -> DragonflyCache | None:
        """Access to DragonflyDB cache layer.

        Returns:
            DragonflyCache | None: Distributed cache instance if enabled, None otherwise
        """
        return self._distributed_cache

    @property
    def embedding_cache(self) -> EmbeddingCache | None:
        """Access to embedding-specific cache.

        Returns:
            EmbeddingCache | None: Embedding cache instance if enabled, None otherwise
        """
        return self._embedding_cache

    @property
    def search_cache(self) -> SearchResultCache | None:
        """Access to search result cache.

        Returns:
            SearchResultCache | None: Search cache instance if enabled, None otherwise
        """
        return self._search_cache

    @property
    def metrics(self) -> CacheMetrics | None:
        """Access to cache metrics.

        Returns:
            CacheMetrics | None: Metrics collector instance if enabled, None otherwise
        """
        return self._metrics

    async def get(
        self,
        key: str,
        cache_type: CacheType = CacheType.CRAWL,
        default: object = None,
    ) -> object:
        """Get value from cache with L1 -> L2 fallback.

        Args:
            key: Cache key
            cache_type: Type of cached data for TTL selection
            default: Default value if not found

        Returns:
            object: Cached value or default

        Raises:
            Exception: Logged but not raised - returns default on error
        """
        # Monitor cache operations with Prometheus if available
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_cache_performance(
                cache_type=cache_type.value, operation="get"
            )

            async def _monitored_get():
                return await self._execute_cache_get(key, cache_type, default)

            return await decorator(_monitored_get)()
        else:
            return await self._execute_cache_get(key, cache_type, default)

    async def _execute_cache_get(  # noqa: PLR0912
        self,
        key: str,
        cache_type: CacheType,
        default: object = None,
    ) -> object:
        """Execute the actual cache get operation."""
        start_time = asyncio.get_event_loop().time()
        cache_key = self._get_cache_key(key, cache_type)

        try:
            # Try L1 cache first
            if self._local_cache:
                value = await self._local_cache.get(cache_key)
                if value is not None:
                    if self._metrics:
                        latency = (asyncio.get_event_loop().time() - start_time) * 1000
                        self._metrics.record_hit(cache_type.value, "local", latency)

                    # Record hit in Prometheus metrics
                    if self.metrics_registry:
                        self.metrics_registry.record_cache_hit(
                            "local", cache_type.value
                        )
                    return value

            # Try L2 cache (DragonflyDB)
            if self._distributed_cache:
                value = await self._distributed_cache.get(cache_key)
                if value is not None:
                    # Populate L1 cache for future hits
                    if self._local_cache:
                        await self._local_cache.set(cache_key, value)

                    if self._metrics:
                        latency = (asyncio.get_event_loop().time() - start_time) * 1000
                        self._metrics.record_hit(
                            cache_type.value, "distributed", latency
                        )

                    # Record hit in Prometheus metrics
                    if self.metrics_registry:
                        self.metrics_registry.record_cache_hit(
                            "distributed", cache_type.value
                        )
                    return value

            # Cache miss
            if self._metrics:
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                self._metrics.record_miss(cache_type.value, latency)

            # Record miss in Prometheus metrics
            if self.metrics_registry:
                self.metrics_registry.record_cache_miss(cache_type.value)
            return default

        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            if self._metrics:
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                self._metrics.record_miss(cache_type.value, latency)

            # Record miss in Prometheus metrics
            if self.metrics_registry:
                self.metrics_registry.record_cache_miss(cache_type.value)
            return default

    async def set(
        self,
        key: str,
        value: object,
        cache_type: CacheType = CacheType.CRAWL,
        ttl: int | None = None,
    ) -> bool:
        """Set value in both cache layers.

        Args:
            key: Cache key
            value: Value to cache (must be serializable)
            cache_type: Type of cached data for TTL selection
            ttl: Custom TTL in seconds (overrides cache_type default)

        Returns:
            bool: True if successful in at least one cache layer

        Raises:
            Exception: Logged but not raised - returns False on error
        """
        # Monitor cache operations with Prometheus if available
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_cache_performance(
                cache_type=cache_type.value, operation="set"
            )

            async def _monitored_set():
                return await self._execute_cache_set(key, value, cache_type, ttl)

            return await decorator(_monitored_set)()
        else:
            return await self._execute_cache_set(key, value, cache_type, ttl)

    async def _execute_cache_set(
        self,
        key: str,
        value: object,
        cache_type: CacheType,
        ttl: int | None = None,
    ) -> bool:
        """Execute the actual cache set operation."""
        start_time = asyncio.get_event_loop().time()
        cache_key = self._get_cache_key(key, cache_type)
        effective_ttl = ttl or self.distributed_ttl_seconds.get(cache_type, 3600)

        success = True
        try:
            # Set in L1 cache
            if self._local_cache:
                await self._local_cache.set(
                    cache_key, value, ttl=min(effective_ttl, 300)
                )

            # Set in L2 cache (DragonflyDB)
            if self._distributed_cache:
                success = await self._distributed_cache.set(
                    cache_key, value, ttl=effective_ttl
                )

            if self._metrics:
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                self._metrics.record_set(cache_type.value, latency, success)

            return success

        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            if self._metrics:
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                self._metrics.record_set(cache_type.value, latency, False)
            return False

    async def delete(self, key: str, cache_type: CacheType = CacheType.CRAWL) -> bool:
        """Delete value from both cache layers.

        Args:
            key: Cache key
            cache_type: Type of cached data

        Returns:
            True if successful
        """
        cache_key = self._get_cache_key(key, cache_type)
        success = True

        try:
            # Delete from L1 cache
            if self._local_cache:
                await self._local_cache.delete(cache_key)

            # Delete from L2 cache
            if self._distributed_cache:
                success = await self._distributed_cache.delete(cache_key)

            return success

        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
            return False

    async def clear(self, cache_type: CacheType | None = None) -> bool:
        """Clear cache layers.

        Args:
            cache_type: Specific cache type to clear (None for all)

        Returns:
            True if successful
        """
        try:
            if cache_type:
                # Clear specific cache type by pattern
                pattern = f"{self.key_prefix}{cache_type.value}:*"
                if self._distributed_cache:
                    keys = await self._distributed_cache.scan_keys(pattern)
                    for key in keys:
                        await self._distributed_cache.delete(key)
                        if self._local_cache:
                            await self._local_cache.delete(
                                self._get_cache_key(key, cache_type)
                            )
            else:
                # Clear all caches
                if self._local_cache:
                    await self._local_cache.clear()
                if self._distributed_cache:
                    await self._distributed_cache.clear()

            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    async def get_stats(self) -> dict[str, object]:
        """Get comprehensive cache statistics.

        Returns:
            dict[str, object]: Cache statistics including:
                - manager: Enabled cache layers info
                - local: Local cache stats if enabled
                - dragonfly: Distributed cache stats if enabled
                - embedding_cache: Embedding cache stats if enabled
                - search_cache: Search cache stats if enabled
                - metrics: Performance metrics if enabled

        Raises:
            Exception: Logged internally - always returns valid stats dict
        """
        stats = {"manager": {"enabled_layers": []}}

        if self._local_cache:
            stats["manager"]["enabled_layers"].append("local")
            stats["local"] = {
                "size": await self._local_cache.size(),
                "memory_usage": self._local_cache.get_memory_usage(),
                "max_size": self._local_cache.max_size,
                "max_memory_mb": self._local_cache.max_memory_mb,
            }

        if self._distributed_cache:
            stats["manager"]["enabled_layers"].append("dragonfly")
            stats["dragonfly"] = {
                "size": await self._distributed_cache.size(),
                "url": self._distributed_cache.redis_url,
                "compression": self._distributed_cache.enable_compression,
                "max_connections": self._distributed_cache.max_connections,
            }

        if self._embedding_cache:
            stats["embedding_cache"] = await self._embedding_cache.get_stats()

        if self._search_cache:
            stats["search_cache"] = await self._search_cache.get_stats()

        if self._metrics:
            stats["metrics"] = self._metrics.get_summary()

        return stats

    async def close(self) -> None:
        """Clean up cache resources.

        Closes all active cache connections and releases resources.
        Safe to call multiple times.

        Raises:
            Exception: Logged but not raised - cleanup continues on error
        """
        try:
            if self._distributed_cache:
                await self._distributed_cache.close()
            logger.info("Cache manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")

    def _get_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Generate cache key with type prefix.

        Args:
            key: Base key
            cache_type: Cache type for prefix

        Returns:
            Formatted cache key
        """
        # Create content-based hash for consistent keys
        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        return f"{self.key_prefix}{cache_type.value}:{key_hash}"

    # Direct access methods for specialized caches
    async def get_embedding_direct(
        self, content_hash: str, model: str
    ) -> list[float] | None:
        """Direct access to embedding cache.

        Args:
            content_hash: Hash of the content that was embedded
            model: Name of the embedding model used

        Returns:
            list[float] | None: Cached embedding vector or None if not found
        """
        if not self._embedding_cache:
            return None
        return await self._embedding_cache.get_embedding(content_hash, model)

    async def set_search_results_direct(
        self,
        query_hash: str,
        collection: str,
        results: list[dict[str, object]],
        ttl: int | None = None,
    ) -> bool:
        """Direct access to search result cache.

        Args:
            query_hash: Hash of the search query
            collection: Name of the collection searched
            results: Search results to cache
            ttl: Custom TTL in seconds (None uses default)

        Returns:
            bool: True if successfully cached
        """
        if not self._search_cache:
            return False
        return await self._search_cache.set_search_results(
            query_hash, collection, results, ttl
        )

    async def get_performance_stats(self) -> dict[str, object]:
        """Get performance-focused statistics.

        Returns:
            dict[str, object]: Performance metrics including:
                - hit_rates: Cache hit rates by type and layer
                - latency_stats: Operation latency statistics
                - operation_counts: Total operation counts by type

        Note:
            Returns empty dict if metrics are disabled
        """
        if not self._metrics:
            return {}

        return {
            "hit_rates": self._metrics.get_hit_rates(),
            "latency_stats": self._metrics.get_latency_stats(),
            "operation_counts": self._metrics.get_operation_counts(),
        }
