"""Cache manager that combines a persistent local layer with DragonflyDB."""

import asyncio
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import CacheType

from .dragonfly_cache import DragonflyCache
from .embedding_cache import EmbeddingCache
from .persistent_cache import PersistentCacheManager
from .search_cache import SearchResultCache


logger = logging.getLogger(__name__)


try:
    # Optional dependency: monitoring is only needed when metrics are enabled.
    from ..monitoring.metrics import get_metrics_registry as _get_metrics_registry

    MONITORING_AVAILABLE = True
    get_metrics_registry = _get_metrics_registry
except ImportError:  # pragma: no cover - optional monitoring stack
    MONITORING_AVAILABLE = False
    get_metrics_registry = None


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
class CacheManager:
    """Two-tier cache manager with DragonflyDB and specialized cache layers."""

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
        local_cache_path: Path | None = None,
        memory_pressure_threshold: float | None = None,
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
        self.dragonfly_url = dragonfly_url

        # Default distributed cache TTLs by cache type
        self.distributed_ttl_seconds = distributed_ttl_seconds or {
            CacheType.LOCAL: 3600,  # 1 hour for local cache
            CacheType.REDIS: 86400,  # 1 day for redis cache
            CacheType.HYBRID: 3600,  # 1 hour for hybrid cache
        }

        # Initialize local cache (L1)
        self._local_cache: PersistentCacheManager | None = None
        self._local_default_ttl = local_ttl_seconds
        if enable_local_cache:
            base_path = (
                Path(local_cache_path)
                if local_cache_path is not None
                else Path("cache") / "local"
            )
            bytes_limit = int(local_max_memory_mb * 1024 * 1024)
            self._local_cache = PersistentCacheManager(
                base_path=base_path,
                max_entries=local_max_size,
                max_memory_bytes=bytes_limit,
                memory_pressure_threshold=memory_pressure_threshold,
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
                default_ttl=self.distributed_ttl_seconds[CacheType.REDIS],
            )
            self._search_cache = SearchResultCache(
                cache=self._distributed_cache,
                default_ttl=self.distributed_ttl_seconds[CacheType.REDIS],
            )

        # Initialize Prometheus monitoring registry
        self.metrics_registry = self._initialize_metrics_registry(enable_metrics)

    def _initialize_metrics_registry(self, enable_metrics: bool):
        """Initialize metrics registry with error handling."""
        if not (
            enable_metrics and MONITORING_AVAILABLE and get_metrics_registry is not None
        ):
            return None
        try:
            registry = get_metrics_registry()
            logger.info("Cache monitoring enabled with Prometheus integration")
            return registry
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("Failed to initialize cache monitoring: %s", e)
            return None

        logger.info(
            "CacheManager initialized with DragonflyDB: %s, local=%s, specialized=%s",
            self.dragonfly_url,
            self.enable_local_cache,
            self.enable_specialized_caches,
        )

    @property
    def local_cache(self) -> PersistentCacheManager | None:
        """Access to local cache layer.

        Returns:
            PersistentCacheManager | None: Local cache instance if enabled, else None
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

    async def get(
        self,
        key: str,
        cache_type: CacheType = CacheType.LOCAL,
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
        return await self._execute_cache_get(key, cache_type, default)

    async def _execute_cache_get(
        self,
        key: str,
        cache_type: CacheType,
        default: object = None,
    ) -> object:
        """Execute the actual cache get operation."""
        start_time = asyncio.get_event_loop().time()
        cache_key = self._get_cache_key(key, cache_type)

        # Try L1 cache first
        l1_result = await self._try_local_cache_get(cache_key, cache_type, start_time)
        if l1_result is not None:
            return l1_result

        # Try L2 cache (DragonflyDB)
        l2_result = await self._try_distributed_cache_get(
            cache_key, cache_type, start_time
        )
        if l2_result is not None:
            return l2_result

        # Cache miss - record metrics and return default
        self._record_cache_miss(cache_type, start_time)
        return default

    async def _try_local_cache_get(
        self, cache_key: str, cache_type: CacheType, start_time: float
    ) -> object | None:
        """Try to get value from local cache."""
        if not self._local_cache:
            return None

        try:
            value = await self._local_cache.get(cache_key)
        except (ConnectionError, OSError, PermissionError) as e:
            logger.error("Local cache get error for key %s: %s", cache_key, e)
            return None

        if value is not None:
            self._record_cache_hit(cache_type, "local", start_time)
        return value

    async def _try_distributed_cache_get(
        self, cache_key: str, cache_type: CacheType, start_time: float
    ) -> object | None:
        """Try to get value from distributed cache and populate L1 if found."""
        if not self._distributed_cache:
            return None

        try:
            value = await self._distributed_cache.get(cache_key)
        except (ConnectionError, OSError, PermissionError) as e:
            logger.error("Distributed cache get error for key %s: %s", cache_key, e)
            return None

        if value is not None:
            # Populate L1 cache for future hits
            await self._populate_local_cache(cache_key, value)
            self._record_cache_hit(cache_type, "distributed", start_time)
        return value

    async def _populate_local_cache(self, cache_key: str, value: object) -> None:
        """Populate local cache with distributed cache hit."""
        if not self._local_cache:
            return
        try:
            ttl = self._local_default_ttl
            await self._local_cache.set(cache_key, value, ttl=ttl)
        except (ConnectionError, OSError, PermissionError) as e:
            logger.warning(
                "Failed to populate local cache for key %s: %s", cache_key, e
            )

    def _record_cache_hit(
        self, cache_type: CacheType, layer: str, start_time: float
    ) -> None:
        """Record cache hit metrics."""
        if self.metrics_registry:
            self.metrics_registry.record_cache_hit(layer, cache_type.value)

    def _record_cache_miss(self, cache_type: CacheType, start_time: float) -> None:
        """Record cache miss metrics."""
        if self.metrics_registry:
            self.metrics_registry.record_cache_miss(cache_type.value)

    async def set(
        self,
        key: str,
        value: object,
        cache_type: CacheType = CacheType.LOCAL,
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
        return await self._execute_cache_set(key, value, cache_type, ttl)

    async def _execute_cache_set(
        self,
        key: str,
        value: object,
        cache_type: CacheType,
        ttl: int | None = None,
    ) -> bool:
        """Execute the actual cache set operation."""
        cache_key = self._get_cache_key(key, cache_type)
        effective_ttl = ttl or self.distributed_ttl_seconds.get(cache_type, 3600)

        # Set in L1 cache
        await self._set_local_cache(cache_key, value, effective_ttl)

        # Set in L2 cache (DragonflyDB)
        success = await self._set_distributed_cache(cache_key, value, effective_ttl)

        if not success:
            logger.warning("Distributed cache set failed for %s", cache_key)
        return success

    async def _set_local_cache(
        self, cache_key: str, value: object, effective_ttl: int
    ) -> None:
        """Set value in local cache with error handling."""
        if not self._local_cache:
            return
        try:
            ttl_limit = self._local_default_ttl
            ttl_value = (
                min(effective_ttl, ttl_limit)
                if ttl_limit is not None
                else effective_ttl
            )
            await self._local_cache.set(cache_key, value, ttl=ttl_value)
        except (ConnectionError, OSError, PermissionError) as e:
            logger.warning("Local cache set error for key %s: %s", cache_key, e)

    async def _set_distributed_cache(
        self, cache_key: str, value: object, effective_ttl: int
    ) -> bool:
        """Set value in distributed cache with error handling."""
        if not self._distributed_cache:
            return True
        try:
            return await self._distributed_cache.set(
                cache_key, value, ttl=effective_ttl
            )
        except (ConnectionError, OSError, PermissionError) as e:
            logger.error("Distributed cache set error for key %s: %s", cache_key, e)
            return False

    async def delete(self, key: str, cache_type: CacheType = CacheType.LOCAL) -> bool:
        """Delete value from both cache layers.

        Args:
            key: Cache key
            cache_type: Type of cached data

        Returns:
            True if successful
        """
        cache_key = self._get_cache_key(key, cache_type)

        # Delete from L1 cache
        await self._delete_from_local_cache(cache_key)

        # Delete from L2 cache
        return await self._delete_from_distributed_cache(cache_key)

    async def _delete_from_local_cache(self, cache_key: str) -> None:
        """Delete from local cache with error handling."""
        if not self._local_cache:
            return
        try:
            await self._local_cache.delete(cache_key)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning("Local cache delete error for key %s: %s", cache_key, e)

    async def _delete_from_distributed_cache(self, cache_key: str) -> bool:
        """Delete from distributed cache with error handling."""
        if not self._distributed_cache:
            return True
        try:
            return await self._distributed_cache.delete(cache_key)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Distributed cache delete error for key %s: %s", cache_key, e)
            return False

    async def clear(self, cache_type: CacheType | None = None) -> bool:
        """Clear cache layers.

        Args:
            cache_type: Specific cache type to clear (None for all)

        Returns:
            True if successful
        """
        if cache_type:
            return await self._clear_specific_cache_type(cache_type)
        return await self._clear_all_caches()

    async def _clear_specific_cache_type(self, cache_type: CacheType) -> bool:
        """Clear specific cache type by pattern."""
        pattern = f"{self.key_prefix}{cache_type.value}:*"

        if not self._distributed_cache:
            return True

        try:
            keys = await self._distributed_cache.scan_keys(pattern)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Cache scan error for pattern %s: %s", pattern, e)
            return False

        return await self._clear_keys_from_both_caches(keys, cache_type)

    async def _clear_keys_from_both_caches(
        self, keys: list[str], cache_type: CacheType
    ) -> bool:
        """Clear specific keys from both cache layers."""
        for key in keys:
            if not await self._clear_single_key_from_both_caches(key, cache_type):
                return False
        return True

    async def _clear_single_key_from_both_caches(
        self, key: str, cache_type: CacheType
    ) -> bool:
        """Clear a single key from both cache layers."""
        # Clear from distributed cache
        if self._distributed_cache:
            try:
                await self._distributed_cache.delete(key)
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.error("Distributed cache delete error for key %s: %s", key, e)
                return False

        # Clear from local cache
        if self._local_cache:
            try:
                await self._local_cache.delete(self._get_cache_key(key, cache_type))
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.warning("Local cache delete error for key %s: %s", key, e)
        return True

    async def _clear_all_caches(self) -> bool:
        """Clear all cache layers."""
        local_success = await self._clear_local_cache()
        distributed_success = await self._clear_distributed_cache()
        return local_success and distributed_success

    async def _clear_local_cache(self) -> bool:
        """Clear local cache with error handling."""
        if not self._local_cache:
            return True
        try:
            await self._local_cache.clear()
            return True
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Local cache clear error: %s", e)
            return False

    async def _clear_distributed_cache(self) -> bool:
        """Clear distributed cache with error handling."""
        if not self._distributed_cache:
            return True
        try:
            await self._distributed_cache.clear()
            return True
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Distributed cache clear error: %s", e)
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
        stats: dict[str, Any] = {"manager": {"enabled_layers": []}}

        if self._local_cache:
            stats["manager"]["enabled_layers"].append("local")
            local_size = await self._local_cache.size()
            local_stats = asdict(self._local_cache.stats)
            total_requests = local_stats["hits"] + local_stats["misses"]
            hit_rate = (local_stats["hits"] / total_requests) if total_requests else 0.0
            stats.update(
                {
                    "size": local_size,
                    "hit_rate": hit_rate,
                    "total_requests": total_requests,
                }
            )
            stats["local"] = {
                "size": local_size,
                "memory_usage": self._local_cache.get_memory_usage(),
                "max_size": self._local_cache.max_size,
                "max_memory_mb": self._local_cache.max_memory_mb,
                "stats": local_stats,
            }
        else:
            stats.update({"size": 0, "hit_rate": 0.0, "total_requests": 0})

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

        return stats

    async def close(self) -> None:
        """Clean up cache resources.

        Closes all active cache connections and releases resources.
        Safe to call multiple times.

        Raises:
            Exception: Logged but not raised - cleanup continues on error
        """
        await self._close_distributed_cache()
        logger.info("Cache manager closed successfully")

    async def _close_distributed_cache(self) -> None:
        """Close distributed cache connection with error handling."""
        if not self._distributed_cache:
            return
        try:
            await self._distributed_cache.close()
        except (AttributeError, ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Error closing distributed cache: %s", e)

    def _get_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Generate cache key with type prefix.

        Args:
            key: Base key
            cache_type: Cache type for prefix

        Returns:
            Formatted cache key
        """
        # Create content-based hash for consistent keys (using SHA256 for security)
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
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
            query_hash, results, collection, ttl=ttl
        )

    async def get_performance_stats(self) -> dict[str, object]:
        """Return lightweight performance stats for active cache layers."""

        stats: dict[str, object] = {}

        if self._local_cache:
            stats["local"] = asdict(self._local_cache.stats)

        return stats
