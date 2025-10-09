"""Cache manager that combines a persistent local layer with DragonflyDB."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config.models import CacheType
from src.services.observability.tracing import set_span_attributes, trace_function

from .dragonfly_cache import DragonflyCache
from .embedding_cache import EmbeddingCache
from .persistent_cache import PersistentCacheManager
from .search_cache import SearchResultCache


logger = logging.getLogger(__name__)


@runtime_checkable
class _PatternClearableCache(Protocol):
    """Protocol describing pattern-based key management in caches."""

    async def scan_keys(
        self, pattern: str
    ) -> list[str]:  # pragma: no cover - structural
        """Return keys matching the supplied pattern."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:  # pragma: no cover - structural
        """Delete a key from the cache. Returns True if the key was removed."""
        raise NotImplementedError


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

        (
            self._local_cache,
            self._local_default_ttl,
        ) = self._build_local_cache(
            enable_local_cache=enable_local_cache,
            local_cache_path=local_cache_path,
            local_max_size=local_max_size,
            local_max_memory_mb=local_max_memory_mb,
            memory_pressure_threshold=memory_pressure_threshold,
            local_ttl_seconds=local_ttl_seconds,
        )

        (
            self._distributed_cache,
            self._embedding_cache,
            self._search_cache,
        ) = self._build_distributed_cache_layers(
            enable_distributed_cache=enable_distributed_cache,
            enable_specialized_caches=enable_specialized_caches,
            dragonfly_url=dragonfly_url,
            key_prefix=key_prefix,
        )
        logger.info(
            "CacheManager initialized with DragonflyDB: %s, local=%s, specialized=%s",
            self.dragonfly_url,
            self.enable_local_cache,
            self.enable_specialized_caches,
        )

    def _build_local_cache(
        self,
        *,
        enable_local_cache: bool,
        local_cache_path: Path | None,
        local_max_size: int,
        local_max_memory_mb: float,
        memory_pressure_threshold: float | None,
        local_ttl_seconds: int,
    ) -> tuple[PersistentCacheManager | None, int]:
        """Construct the local cache layer if enabled."""

        if not enable_local_cache:
            return None, local_ttl_seconds

        base_path = (
            Path(local_cache_path)
            if local_cache_path is not None
            else Path("cache") / "local"
        )
        bytes_limit = int(local_max_memory_mb * 1024 * 1024)
        manager = PersistentCacheManager(
            base_path=base_path,
            max_entries=local_max_size,
            max_memory_bytes=bytes_limit,
            memory_pressure_threshold=memory_pressure_threshold,
        )
        return manager, local_ttl_seconds

    def _build_distributed_cache_layers(
        self,
        *,
        enable_distributed_cache: bool,
        enable_specialized_caches: bool,
        dragonfly_url: str,
        key_prefix: str,
    ) -> tuple[
        DragonflyCache | None,
        EmbeddingCache | None,
        SearchResultCache | None,
    ]:
        """Construct distributed cache and specialized helpers."""

        distributed_cache: DragonflyCache | None = None
        embedding_cache: EmbeddingCache | None = None
        search_cache: SearchResultCache | None = None

        if enable_distributed_cache:
            distributed_cache = DragonflyCache(
                redis_url=dragonfly_url,
                key_prefix=key_prefix,
                max_connections=50,
                enable_compression=True,
                compression_threshold=1024,
            )

            if enable_specialized_caches:
                redis_ttl = self.distributed_ttl_seconds.get(CacheType.REDIS, 3600)
                embedding_cache = EmbeddingCache(
                    cache=distributed_cache,
                    default_ttl=redis_ttl,
                )
                search_cache = SearchResultCache(
                    cache=distributed_cache,
                    default_ttl=redis_ttl,
                )

        return distributed_cache, embedding_cache, search_cache

    async def initialize(self) -> None:
        """Initialize underlying cache resources."""

        tasks: list[Awaitable[None]] = []
        if self._local_cache is not None:
            tasks.append(self._local_cache.initialize())
        if self._distributed_cache is not None:
            tasks.append(self._distributed_cache.initialize())

        if tasks:
            await asyncio.gather(*tasks)

    async def cleanup(self) -> None:
        """Release cache resources and close connections."""

        tasks: list[Awaitable[None]] = []
        if self._local_cache is not None:
            tasks.append(self._local_cache.cleanup())
        if self._distributed_cache is not None:
            tasks.append(self._distributed_cache.close())

        if tasks:
            await asyncio.gather(*tasks)

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

    @trace_function("cache.manager.get")
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
        return await self._execute_cache_get(key, cache_type, default)

    async def _execute_cache_get(
        self,
        key: str,
        cache_type: CacheType,
        default: object = None,
    ) -> object:
        """Execute the actual cache get operation."""
        cache_key = self._get_cache_key(key, cache_type)

        # Try L1 cache first
        l1_result = await self._try_local_cache_get(cache_key, cache_type)
        if l1_result is not None:
            return l1_result

        # Try L2 cache (DragonflyDB)
        l2_result = await self._try_distributed_cache_get(cache_key, cache_type)
        if l2_result is not None:
            return l2_result

        # Cache miss - record metrics and return default
        self._record_cache_miss(cache_type)
        return default

    async def _try_local_cache_get(
        self, cache_key: str, cache_type: CacheType
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
            self._record_cache_hit(cache_type, "local")
        return value

    async def _try_distributed_cache_get(
        self, cache_key: str, cache_type: CacheType
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
            self._record_cache_hit(cache_type, "distributed")
        return value

    async def _populate_local_cache(self, cache_key: str, value: object) -> None:
        """Populate local cache with distributed cache hit."""
        if not self._local_cache:
            return
        try:
            ttl = self._local_default_ttl
            await self._local_cache.set(cache_key, value, ttl_seconds=ttl)
        except (ConnectionError, OSError, PermissionError) as e:
            logger.warning(
                "Failed to populate local cache for key %s: %s", cache_key, e
            )

    def _record_cache_hit(self, cache_type: CacheType, layer: str) -> None:
        """Annotate active span with cache hit information."""
        set_span_attributes(
            {
                "cache.result": "hit",
                "cache.type": cache_type.value,
                "cache.layer": layer,
            }
        )

    def _record_cache_miss(self, cache_type: CacheType) -> None:
        """Annotate active span with cache miss information."""
        set_span_attributes(
            {
                "cache.result": "miss",
                "cache.type": cache_type.value,
            }
        )

    @trace_function("cache.manager.set")
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
        effective_ttl = (
            ttl
            if ttl is not None
            else self.distributed_ttl_seconds.get(cache_type, 3600)
        )

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
            await self._local_cache.set(cache_key, value, ttl_seconds=ttl_value)
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

    async def clear_all(self) -> int:
        """Clear all cache layers and return cleared entry count from L2."""

        cleared = 0
        if isinstance(self._distributed_cache, DragonflyCache):
            cleared = await self._distributed_cache.clear()
        if self._local_cache:
            await self._local_cache.clear()
        return cleared

    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching a pattern in distributed cache."""

        if not isinstance(self._distributed_cache, DragonflyCache):
            if self._distributed_cache is None:
                return 0
            cleared = await self._clear_keys_matching_pattern(pattern)
        else:
            cleared = await self._distributed_cache.clear_pattern(pattern)
        if self._local_cache:
            await self._local_cache.clear()
        return cleared

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

    async def _clear_keys_matching_pattern(self, pattern: str) -> int:
        """Clear keys matching a Redis glob pattern."""

        distributed_cache = self._distributed_cache
        if not isinstance(distributed_cache, _PatternClearableCache):
            logger.warning(
                "Distributed cache %r missing _PatternClearableCache; skipping clear()",
                type(distributed_cache),
            )
            return 0

        try:
            keys = await distributed_cache.scan_keys(pattern)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Cache scan error for pattern %s: %s", pattern, e)
            return 0

        cleared = 0
        for key in keys:
            try:
                if bool(await distributed_cache.delete(key)):
                    cleared += 1
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.error("Distributed cache delete error for key %s: %s", key, e)
        return cleared

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
                await self._local_cache.delete(key)
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
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"{self.key_prefix}{cache_type.value}:{key_hash}"

    async def get_performance_stats(self) -> dict[str, object]:
        """Return lightweight performance stats for enabled cache layers."""

        stats: dict[str, object] = {}

        if self._local_cache:
            stats["local"] = asdict(self._local_cache.stats)

        if self._distributed_cache:
            stats["distributed"] = {
                "compression_enabled": self._distributed_cache.enable_compression,
                "redis_url": self._distributed_cache.redis_url,
            }

        return stats
