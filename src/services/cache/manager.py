"""Cache management utilities coordinating Dragonfly caches."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, cast, runtime_checkable

from src.config.models import CacheType
from src.services.observability.tracing import (
    log_extra_with_trace,
    set_span_attributes,
    trace_function,
)

from .dragonfly_cache import DragonflyCache
from .embedding_cache import EmbeddingCache
from .search_cache import SearchResultCache


logger = logging.getLogger(__name__)


def _log_extra(event: str, **metadata: Any) -> dict[str, Any]:
    """Return logging extras enriched with trace metadata."""

    metadata.setdefault("component", "cache.manager")
    return log_extra_with_trace(event, **metadata)


DEFAULT_TTLS: dict[CacheType, int] = {
    CacheType.EMBEDDINGS: 86400,
    CacheType.SEARCH: 3600,
    CacheType.CRAWL: 3600,
    CacheType.HYDE: 3600,
    CacheType.DRAGONFLY: 3600,
    CacheType.QUERIES: 3600,
}


@runtime_checkable
class _PatternClearableCache(Protocol):
    """Protocol describing pattern-based key management in caches."""

    async def clear_pattern(self, pattern: str) -> int:  # pragma: no cover - protocol
        """Remove keys matching ``pattern``."""

        ...

    async def scan_keys(self, pattern: str) -> list[str]:  # pragma: no cover - protocol
        """Return keys matching ``pattern``."""

        ...

    async def delete_many(  # pragma: no cover - protocol
        self, keys: list[str]
    ) -> dict[str, bool]:
        """Delete a batch of keys."""

        ...


class CacheManager:
    """Thin wrapper that coordinates Dragonfly cache usage."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        *,
        dragonfly_url: str = "redis://localhost:6379",
        enable_distributed_cache: bool = True,
        distributed_ttl_seconds: dict[CacheType, int] | None = None,
        key_prefix: str = "aidocs:",
        enable_specialized_caches: bool = True,
        distributed_cache: DragonflyCache | None = None,
    ) -> None:
        """Initialise the cache manager.

        Args:
            dragonfly_url: Redis-compatible URL for the Dragonfly deployment.
            enable_distributed_cache: Toggle distributed caching entirely.
            distributed_ttl_seconds: Optional per-cache TTL overrides.
            key_prefix: Prefix applied to all Dragonfly keys.
            enable_specialized_caches: Toggle creation of embedding/search helpers.
            distributed_cache: Preconfigured Dragonfly cache instance.
        """

        self.key_prefix = key_prefix
        self.distributed_ttl_seconds = distributed_ttl_seconds or DEFAULT_TTLS.copy()

        self._distributed_cache: DragonflyCache | None = None
        if enable_distributed_cache:
            self._distributed_cache = (
                distributed_cache
                if distributed_cache is not None
                else DragonflyCache(
                    redis_url=dragonfly_url,
                    key_prefix=key_prefix,
                )
            )
        else:
            logger.warning(
                "Distributed Dragonfly cache disabled; cache operations will no-op",
                extra=_log_extra("cache.manager.init", distributed=False),
            )

        self._embedding_cache: EmbeddingCache | None = None
        self._search_cache: SearchResultCache | None = None

        if enable_specialized_caches and self._distributed_cache is not None:
            ttl = self.distributed_ttl_seconds.get(CacheType.DRAGONFLY, 3600)
            self._embedding_cache = EmbeddingCache(
                cache=self._distributed_cache,
                default_ttl=self.distributed_ttl_seconds.get(CacheType.EMBEDDINGS, ttl),
            )
            self._search_cache = SearchResultCache(
                cache=self._distributed_cache,
                default_ttl=self.distributed_ttl_seconds.get(CacheType.SEARCH, 3600),
            )

        logger.info(
            "CacheManager configured (distributed=%s, specialized=%s)",
            enable_distributed_cache,
            enable_specialized_caches,
            extra=_log_extra(
                "cache.manager.init",
                distributed=enable_distributed_cache,
                specialized=enable_specialized_caches,
            ),
        )

    async def initialize(self) -> None:
        """Initialise the underlying Dragonfly cache connection."""

        if self._distributed_cache is not None:
            await self._distributed_cache.initialize()

    async def cleanup(self) -> None:
        """Cleanup routine delegating to ``close`` for lifecycle symmetry."""

        await self.close()

    @property
    def distributed_cache(self) -> DragonflyCache | None:
        """Return the configured Dragonfly cache instance.

        Returns:
            Dragonfly cache or ``None`` when caching is disabled.
        """

        return self._distributed_cache

    @property
    def embedding_cache(self) -> EmbeddingCache | None:
        """Return specialised embedding cache helper when configured.

        Returns:
            Embedding cache helper or ``None`` when disabled.
        """

        return self._embedding_cache

    @property
    def search_cache(self) -> SearchResultCache | None:
        """Return specialised search cache helper when configured.

        Returns:
            Search cache helper or ``None`` when disabled.
        """

        return self._search_cache

    @trace_function("cache.manager.get")
    async def get(
        self,
        key: str,
        cache_type: CacheType = CacheType.DRAGONFLY,
        default: object = None,
    ) -> object:
        """Fetch value from Dragonfly, returning ``default`` on misses.

        Args:
            key: Logical cache key to fetch.
            cache_type: Cache namespace the key belongs to.
            default: Value to return when the key is absent or unavailable.

        Returns:
            Cached object or the provided default.
        """

        cache_key = self._get_cache_key(key, cache_type)
        distributed = self._distributed_cache
        if distributed is None:
            return default

        try:
            result = await distributed.get(cache_key)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.exception(
                "Distributed cache get error for key %s: %s",
                cache_key,
                exc,
                extra=_log_extra(
                    "cache.manager.get",
                    key=cache_key,
                    cache_type=cache_type.value,
                ),
            )
            return default

        if result is not None:
            self._record_cache_hit(cache_type)
            return result

        self._record_cache_miss(cache_type)
        return default

    @trace_function("cache.manager.set")
    async def set(
        self,
        key: str,
        value: object,
        cache_type: CacheType = CacheType.DRAGONFLY,
        ttl: int | None = None,
    ) -> bool:
        """Store value within Dragonfly.

        Args:
            key: Logical cache key to populate.
            value: Serializable value to store.
            cache_type: Cache namespace governing TTL selection.
            ttl: Optional explicit TTL override in seconds.

        Returns:
            ``True`` if the operation succeeded, otherwise ``False``.
        """

        cache_key = self._get_cache_key(key, cache_type)
        distributed = self._distributed_cache
        if distributed is None:
            return False

        effective_ttl = ttl or self.distributed_ttl_seconds.get(cache_type, 3600)
        try:
            return await distributed.set(cache_key, value, ttl=effective_ttl)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.exception(
                "Distributed cache set error for key %s: %s",
                cache_key,
                exc,
                extra=_log_extra(
                    "cache.manager.set",
                    key=cache_key,
                    cache_type=cache_type.value,
                    ttl=effective_ttl,
                ),
            )
            return False

    async def delete(
        self, key: str, cache_type: CacheType = CacheType.DRAGONFLY
    ) -> bool:
        """Delete cached value from Dragonfly.

        Args:
            key: Logical cache key to remove.
            cache_type: Cache namespace the key belongs to.

        Returns:
            ``True`` when the entry was deleted successfully.
        """

        cache_key = self._get_cache_key(key, cache_type)
        distributed = self._distributed_cache
        if distributed is None:
            return False

        try:
            return await distributed.delete(cache_key)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.exception(
                "Distributed cache delete error for key %s: %s",
                cache_key,
                exc,
                extra=_log_extra(
                    "cache.manager.delete",
                    key=cache_key,
                    cache_type=cache_type.value,
                ),
            )
            return False

    async def clear(self, cache_type: CacheType | None = None) -> bool:
        """Clear Dragonfly cache entries.

        Args:
            cache_type: Optional namespace to clear; clears all when ``None``.

        Returns:
            ``True`` when the clear request completed without errors.
        """

        if cache_type is None:
            return await self._clear_all_distributed()
        return await self._clear_specific_cache_type(cache_type)

    async def clear_all(self) -> int:
        """Clear all distributed cache entries and return cleared count.

        Returns:
            Number of keys removed (``-1`` when unknown).
        """

        distributed = self._distributed_cache
        if distributed is None:
            return 0

        if isinstance(distributed, DragonflyCache):
            return await distributed.clear()

        if isinstance(distributed, _PatternClearableCache):
            clearable = cast(_PatternClearableCache, distributed)
            keys = await clearable.scan_keys("*")
            if not keys:
                return 0
            result = await clearable.delete_many(keys)
            return sum(result.values())

        logger.debug(
            "Distributed cache does not support full clear",
            extra=_log_extra("cache.manager.clear_all", supported=False),
        )
        return 0

    async def clear_pattern(self, pattern: str) -> int:
        """Clear entries matching ``pattern`` via Dragonfly pattern support.

        Args:
            pattern: Redis match pattern without the global key prefix.

        Returns:
            Number of keys deleted.
        """

        distributed = self._distributed_cache
        if distributed is None:
            return 0

        if isinstance(distributed, DragonflyCache):
            prefixed_pattern = (
                f"{distributed.key_prefix}{pattern}"
                if distributed.key_prefix
                else pattern
            )
            return await distributed.clear_pattern(prefixed_pattern)

        if isinstance(distributed, _PatternClearableCache):
            clearable = cast(_PatternClearableCache, distributed)
            keys = await clearable.scan_keys(pattern)
            if not keys:
                return 0
            result = await clearable.delete_many(keys)
            return sum(result.values())

        return 0

    async def get_stats(self) -> dict[str, Any]:
        """Return stats describing the configured cache layers.

        Returns:
            Dictionary with manager metadata and cache statistics.
        """

        stats: dict[str, Any] = {"manager": {"enabled_layers": []}}

        distributed = self._distributed_cache
        if isinstance(distributed, DragonflyCache):
            stats["manager"]["enabled_layers"].append("dragonfly")
            stats["dragonfly"] = {
                "size": await distributed.size(),
                "url": getattr(distributed, "redis_url", None),
                "compression": getattr(distributed, "compression_enabled", False),
                "max_connections": getattr(distributed, "max_connections", 0),
            }
        else:
            stats["dragonfly"] = {
                "size": 0,
                "url": None,
                "compression": False,
                "max_connections": 0,
            }

        if self._embedding_cache is not None:
            stats["embedding_cache"] = await self._embedding_cache.get_stats()

        if self._search_cache is not None:
            stats["search_cache"] = await self._search_cache.get_stats()

        return stats

    async def get_performance_stats(self) -> dict[str, Any]:
        """Return lightweight performance indicators.

        Returns:
            Dictionary containing compression and connection details.
        """

        stats: dict[str, Any] = {}
        distributed = self._distributed_cache
        if isinstance(distributed, DragonflyCache):
            stats["distributed"] = {
                "compression_enabled": getattr(
                    distributed, "compression_enabled", False
                ),
                "redis_url": getattr(distributed, "redis_url", None),
            }
        return stats

    async def close(self) -> None:
        """Close Dragonfly connection."""

        distributed = self._distributed_cache
        if not isinstance(distributed, DragonflyCache):
            return
        try:
            await distributed.close()
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.exception(
                "Error closing distributed cache: %s",
                exc,
                extra=_log_extra("cache.manager.close"),
            )

    async def _clear_all_distributed(self) -> bool:
        """Clear entire distributed cache safely."""

        distributed = self._distributed_cache
        if distributed is None:
            return True

        if isinstance(distributed, DragonflyCache):
            try:
                await distributed.clear()
                return True
            except (ConnectionError, OSError, TimeoutError) as exc:
                logger.exception(
                    "Distributed cache clear error: %s",
                    exc,
                    extra=_log_extra("cache.manager.clear_all"),
                )
                return False

        if isinstance(distributed, _PatternClearableCache):
            clearable = cast(_PatternClearableCache, distributed)
            keys = await clearable.scan_keys("*")
            if keys:
                await clearable.delete_many(keys)
            return True

        logger.debug(
            "Distributed cache clear skipped; unsupported cache type",
            extra=_log_extra("cache.manager.clear_all", supported=False),
        )
        return False

    async def _clear_specific_cache_type(self, cache_type: CacheType) -> bool:
        """Clear keys matching the supplied cache type."""

        distributed = self._distributed_cache
        if distributed is None:
            return True

        pattern = f"{cache_type.value}:*"
        if isinstance(distributed, DragonflyCache):
            prefixed_pattern = (
                f"{distributed.key_prefix}{pattern}"
                if distributed.key_prefix
                else pattern
            )
            await distributed.clear_pattern(prefixed_pattern)
            return True

        if isinstance(distributed, _PatternClearableCache):
            clearable = cast(_PatternClearableCache, distributed)
            keys = await clearable.scan_keys(pattern)
            if keys:
                await clearable.delete_many(keys)
            return True

        logger.debug(
            "Distributed cache does not expose pattern clearing; skip clear for %s",
            cache_type.value,
            extra=_log_extra(
                "cache.manager.clear_specific", cache_type=cache_type.value
            ),
        )
        return False

    def _get_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Return hashed cache key scoped by ``cache_type``."""

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{cache_type.value}:{digest}"

    def _record_cache_hit(self, cache_type: CacheType) -> None:
        """Annotate tracing spans for cache hits."""

        set_span_attributes(
            {
                "cache.result": "hit",
                "cache.type": cache_type.value,
                "cache.layer": "distributed",
            }
        )

    def _record_cache_miss(self, cache_type: CacheType) -> None:
        """Annotate tracing spans for cache misses."""

        set_span_attributes(
            {
                "cache.result": "miss",
                "cache.type": cache_type.value,
            }
        )
