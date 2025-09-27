"""Modern caching implementation using aiocache.

This module provides a modernized caching implementation that replaces
custom caching patterns with the battle-tested aiocache library.
It isolates cache aliases per manager instance, defaults to safe JSON
serialization, and centralizes TTL management with configuration overrides.
"""

# pylint: disable=too-many-return-statements

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast
from urllib.parse import urlparse

from aiocache import cached, caches
from aiocache.base import BaseCache
from aiocache.serializers import BaseSerializer, JsonSerializer, PickleSerializer

from src.config import CacheType, Config


logger = logging.getLogger(__name__)

_KEY_HASH_LENGTH = 12
_LOG_KEY_HASH_LENGTH = 8


@dataclass(frozen=True)
class CacheAliasConfig:
    """Configuration metadata for a cache alias."""

    cache_type: CacheType
    alias: str
    namespace: str
    serializer: BaseSerializer


def _hash_text(value: str, length: int = _KEY_HASH_LENGTH) -> str:
    """Return a stable truncated SHA-256 hash for the given text."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:length]


def _json_default(value: Any) -> Any:
    """Best-effort serializer for non-JSON-native types."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "_asdict"):
        return value._asdict()  # type: ignore[no-any-return]
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _canonical_json(value: Any) -> str:
    """Serialize value into canonical JSON for hashing / key generation."""

    return json.dumps(
        value,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    )


class ModernCacheManager:  # pylint: disable=too-many-instance-attributes,too-many-arguments
    """Modern cache manager using aiocache.

    Provides declarative caching with decorators and automatic
    serialization, TTL management, and cache invalidation while ensuring
    per-instance alias isolation and sanitized logging.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aidocs:",
        enable_compression: bool = True,
        config: Config | None = None,
        *,
        serializer_map: Mapping[str | CacheType, BaseSerializer] | None = None,
        use_pickle_for_embeddings: bool = False,
        command_timeout: float = 1.0,
    ):
        """Initialize modern cache manager.

        Args:
            redis_url: Redis connection URL or DSN.
            key_prefix: Prefix for all cache keys.
            enable_compression: Retained for compatibility; no-op for now.
            config: Application configuration for cache settings.
            serializer_map: Optional map overriding serializers per cache type.
            use_pickle_for_embeddings: Enable legacy Pickle serializer for
                embeddings (default is JSON for safety).
            command_timeout: Timeout for cache commands in seconds.
        """

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.enable_compression = enable_compression
        self.config = config
        self.command_timeout = command_timeout
        self._serializer_overrides = serializer_map or {}
        self._closed = False

        self._backend_options, self._backend_cache_path = self._build_backend_options(
            redis_url
        )
        self._alias_suffix = self._generate_alias_suffix(key_prefix)

        self.ttl_settings = (
            getattr(getattr(config, "cache", None), "cache_ttl_seconds", {})
            if config is not None
            else {}
        )

        self.default_ttls = {
            CacheType.EMBEDDINGS: self._resolve_ttl(CacheType.EMBEDDINGS, 86400 * 7),
            CacheType.SEARCH: self._resolve_ttl(CacheType.SEARCH, 3600),
            CacheType.CRAWL: self._resolve_ttl(CacheType.CRAWL, 3600),
            CacheType.HYDE: self._resolve_ttl(CacheType.HYDE, 3600),
        }

        self._alias_order = (
            CacheType.EMBEDDINGS,
            CacheType.SEARCH,
            CacheType.CRAWL,
            CacheType.HYDE,
        )
        self._alias_configs: dict[CacheType, CacheAliasConfig] = {}

        for cache_type in self._alias_order:
            serializer = self._select_serializer(cache_type, use_pickle_for_embeddings)
            alias_config = self._register_alias(cache_type, serializer)
            self._alias_configs[cache_type] = alias_config

        logger.info(
            "ModernCacheManager initialized with Redis host %s (aliases suffix=%s)",
            self._backend_options.get("endpoint")
            or self._backend_options.get("host")
            or self.redis_url,
            self._alias_suffix,
        )

    def _build_backend_options(self, redis_url: str) -> tuple[dict[str, Any], str]:
        """Build backend configuration for aiocache."""

        parsed = urlparse(redis_url)
        options: dict[str, Any]

        if parsed.scheme in {"memory", "simple"}:
            options = {}
            backend = "aiocache.SimpleMemoryCache"
        elif parsed.scheme in {"redis", "rediss", ""}:
            options = {"url": redis_url} if parsed.scheme else {"endpoint": redis_url}
            backend = "aiocache.RedisCache"
        else:
            # Unknown scheme - delegate to aiocache; assume DSN-compatible backend
            options = {"url": redis_url}
            backend = "aiocache.RedisCache"

        if backend == "aiocache.RedisCache":
            options.setdefault("timeout", self.command_timeout)
            options.setdefault("retry_on_timeout", True)
        else:
            options.setdefault("timeout", self.command_timeout)
        return options, backend

    def _generate_alias_suffix(self, key_prefix: str) -> str:
        """Create a deterministic alias suffix to avoid global collisions."""

        base = key_prefix or "aidocs"
        environment = getattr(getattr(self.config, "environment", None), "value", None)
        if environment:
            base = f"{base}:{environment}"
        return _hash_text(base, length=_KEY_HASH_LENGTH)

    def _resolve_ttl(self, cache_type: CacheType, fallback: int) -> int:
        """Determine TTL for a cache type using config overrides."""

        search_keys = (
            cache_type.value,
            cache_type.name,
        )
        for key in search_keys:
            ttl_value = self.ttl_settings.get(key)
            if ttl_value is not None:
                try:
                    return int(ttl_value)
                except (TypeError, ValueError):
                    logger.debug(
                        "Invalid TTL override for %s=%s; falling back to %s",
                        key,
                        ttl_value,
                        fallback,
                    )
                    break
        return fallback

    def _serializer_override_for(self, cache_type: CacheType) -> BaseSerializer | None:
        """Look up serializer override for a cache type."""

        if cache_type in self._serializer_overrides:
            return self._serializer_overrides[cache_type]  # type: ignore[index]
        return cast(
            BaseSerializer | None,
            self._serializer_overrides.get(cache_type.value),
        )

    def _select_serializer(
        self,
        cache_type: CacheType,
        use_pickle_for_embeddings: bool,
    ) -> BaseSerializer:
        """Choose serializer for cache type honoring overrides and safety."""

        override = self._serializer_override_for(cache_type)
        if override is not None:
            return override

        if cache_type is CacheType.EMBEDDINGS and use_pickle_for_embeddings:
            return PickleSerializer()

        return JsonSerializer()

    def _register_alias(
        self,
        cache_type: CacheType,
        serializer: BaseSerializer,
    ) -> CacheAliasConfig:
        """Register cache alias safely within aiocache registry."""

        alias_name = f"{cache_type.value}:{self._alias_suffix}"
        namespace = f"{self.key_prefix}{cache_type.value}:"
        cache_config = {
            "cache": self._backend_cache_path,
            **self._backend_options,
            "serializer": self._serializer_to_config(serializer),
            "namespace": namespace,
            "timeout": self.command_timeout,
        }
        caches.add(alias_name, cache_config)
        return CacheAliasConfig(
            cache_type=cache_type,
            alias=alias_name,
            namespace=namespace,
            serializer=serializer,
        )

    def _resolve_alias(self, cache_type: CacheType) -> CacheAliasConfig:
        """Resolve alias config for cache type with sensible default."""

        if cache_type in self._alias_configs:
            return self._alias_configs[cache_type]
        return self._alias_configs[CacheType.SEARCH]

    def _mask_key(self, key: str) -> str:
        """Produce a short hash representation for logging sensitive keys."""

        if not key:
            return "empty"
        return _hash_text(key, length=_LOG_KEY_HASH_LENGTH)

    @staticmethod
    def _serializer_to_config(serializer: BaseSerializer) -> dict[str, Any]:
        """Convert serializer instance into aiocache configuration mapping."""

        return {"class": f"{serializer.__module__}.{serializer.__class__.__name__}"}

    def get_cache_for_type(self, cache_type: CacheType) -> BaseCache:
        """Get the appropriate cache instance for a cache type.

        Args:
            cache_type: Type of cache needed

        Returns:
            Cache instance for the specified type
        """
        return cast(BaseCache, caches.get(self._resolve_alias(cache_type).alias))

    def cache_embeddings(
        self,
        ttl: int | None = None,
        key_builder: Callable | None = None,
    ):
        """Decorator for caching embedding results.

        Args:
            ttl: Time to live in seconds (None uses default)
            key_builder: Custom function to build cache keys

        Returns:
            Cached decorator for embedding functions
        """
        effective_ttl = ttl or self.default_ttls[CacheType.EMBEDDINGS]

        return cached(
            alias=self._alias_configs[CacheType.EMBEDDINGS].alias,
            ttl=effective_ttl,
            key_builder=key_builder or self._embedding_key_builder,
        )

    def cache_search_results(
        self,
        ttl: int | None = None,
        key_builder: Callable | None = None,
    ):
        """Decorator for caching search results.

        Args:
            ttl: Time to live in seconds (None uses default)
            key_builder: Custom function to build cache keys

        Returns:
            Cached decorator for search functions
        """
        effective_ttl = ttl or self.default_ttls[CacheType.SEARCH]

        return cached(
            alias=self._alias_configs[CacheType.SEARCH].alias,
            ttl=effective_ttl,
            key_builder=key_builder or self._search_key_builder,
        )

    def cache_crawl_results(
        self,
        ttl: int | None = None,
        key_builder: Callable | None = None,
    ):
        """Decorator for caching crawl results.

        Args:
            ttl: Time to live in seconds (None uses default)
            key_builder: Custom function to build cache keys

        Returns:
            Cached decorator for crawl functions
        """
        effective_ttl = ttl or self.default_ttls[CacheType.CRAWL]

        return cached(
            alias=self._alias_configs[CacheType.CRAWL].alias,
            ttl=effective_ttl,
            key_builder=key_builder or self._crawl_key_builder,
        )

    def cache_hyde_results(
        self,
        ttl: int | None = None,
        key_builder: Callable | None = None,
    ):
        """Decorator for caching HyDE results.

        Args:
            ttl: Time to live in seconds (None uses default)
            key_builder: Custom function to build cache keys

        Returns:
            Cached decorator for HyDE functions
        """
        effective_ttl = ttl or self.default_ttls[CacheType.HYDE]

        return cached(
            alias=self._alias_configs[CacheType.HYDE].alias,
            ttl=effective_ttl,
            key_builder=key_builder or self._hyde_key_builder,
        )

    async def get(
        self,
        key: str,
        cache_type: CacheType = CacheType.SEARCH,
        default: Any = None,
    ) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache to use
            default: Default value if not found

        Returns:
            Cached value or default
        """
        alias = self._resolve_alias(cache_type)
        try:
            cache = self.get_cache_for_type(cache_type)
            value = await cache.get(key)
            return value if value is not None else default
        except Exception as exc:
            logger.warning(
                "Cache get error (alias=%s, key=%s): %s",
                alias.alias,
                self._mask_key(key),
                exc,
            )
            return default

    async def set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType = CacheType.SEARCH,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        alias = self._resolve_alias(cache_type)
        try:
            cache = self.get_cache_for_type(cache_type)
            effective_ttl = ttl or self.default_ttls[cache_type]
            await cache.set(key, value, ttl=effective_ttl)
            return True
        except Exception as exc:
            logger.error(
                "Cache set error (alias=%s, key=%s): %s",
                alias.alias,
                self._mask_key(key),
                exc,
            )
            return False

    async def delete(
        self,
        key: str,
        cache_type: CacheType = CacheType.SEARCH,
    ) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache to use

        Returns:
            True if successful, False otherwise
        """
        alias = self._resolve_alias(cache_type)
        try:
            cache = self.get_cache_for_type(cache_type)
            await cache.delete(key)
            return True
        except Exception as exc:
            logger.error(
                "Cache delete error (alias=%s, key=%s): %s",
                alias.alias,
                self._mask_key(key),
                exc,
            )
            return False

    async def clear(self, cache_type: CacheType | None = None) -> bool:
        """Clear cache.

        Args:
            cache_type: Specific cache type to clear (None clears all)

        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_type:
                alias = self._resolve_alias(cache_type)
                await self._clear_cache_namespace(
                    self.get_cache_for_type(cache_type), alias
                )
            else:
                tasks = [
                    self._clear_cache_namespace(
                        self.get_cache_for_type(ct), self._alias_configs[ct]
                    )
                    for ct in self._alias_order
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            return True
        except Exception as exc:
            logger.error("Cache clear error: %s", exc)
            return False

    async def _clear_cache_namespace(
        self, cache: BaseCache, alias: CacheAliasConfig
    ) -> None:
        """Clear all keys in a cache namespace."""

        try:
            if hasattr(cache, "clear"):
                await cache.clear(namespace=alias.namespace)
                return

            backend = cast(Any, getattr(cache, "_backend", None))
            if backend is not None and hasattr(backend, "clear"):
                await backend.clear(namespace=alias.namespace)  # type: ignore
            else:
                logger.warning(
                    "Cannot clear cache alias=%s namespace=%s - no clear method",
                    alias.alias,
                    alias.namespace,
                )
        except Exception as exc:
            logger.warning(
                "Failed to clear cache namespace alias=%s: %s",
                alias.alias,
                exc,
            )

    async def invalidate_pattern(
        self,
        pattern: str,
        cache_type: CacheType = CacheType.SEARCH,
    ) -> int:
        """Invalidate cache keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "user:*")
            cache_type: Type of cache to search

        Returns:
            Number of keys invalidated
        """
        alias = self._resolve_alias(cache_type)
        full_pattern = (
            pattern
            if pattern.startswith(alias.namespace)
            else f"{alias.namespace}{pattern}"
        )
        try:
            cache = self.get_cache_for_type(cache_type)
            if hasattr(cache, "delete_matched"):
                result = await cache.delete_matched(full_pattern)  # type: ignore
                return int(result or 0)
            if hasattr(cache, "delete_pattern"):
                result = await cache.delete_pattern(full_pattern)  # type: ignore
                return int(result or 0)
            logger.warning(
                "Pattern invalidation not supported for alias=%s", alias.alias
            )
            return 0
        except Exception as exc:
            logger.error(
                "Cache pattern invalidation error for alias=%s pattern=%s: %s",
                alias.alias,
                pattern,
                exc,
            )
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            stats: dict[str, Any] = {
                "manager": {
                    "redis_url": self.redis_url,
                    "key_prefix": self.key_prefix,
                    "compression_enabled": self.enable_compression,
                    "aliases": {
                        ct.value: cfg.alias for ct, cfg in self._alias_configs.items()
                    },
                },
                "ttl_settings": self.default_ttls,
                "cache_types": {
                    cfg.cache_type.value: {"namespace": cfg.namespace}
                    for cfg in self._alias_configs.values()
                },
            }

            for cache_type, alias in self._alias_configs.items():
                try:
                    cache_obj = caches.get(alias.alias)
                    if cache_obj is None:
                        continue
                    cache = cast(BaseCache, cache_obj)
                    cache_any = cast(Any, cache)
                    stats_value = None
                    if hasattr(cache_any, "get_stats"):
                        stats_result = cache_any.get_stats()
                        if inspect.isawaitable(stats_result):
                            stats_value = await stats_result
                        else:
                            stats_value = stats_result
                    if stats_value is not None:
                        stats["cache_types"][cache_type.value]["stats"] = stats_value
                except (ConnectionError, OSError, PermissionError) as exc:
                    logger.debug(
                        "Cache stats unavailable for alias=%s: %s",
                        alias.alias,
                        exc,
                    )

            return stats
        except Exception as exc:
            logger.error("Error getting cache stats: %s", exc)
            return {"error": str(exc)}

    async def close(self) -> None:
        """Clean up cache resources."""
        if self._closed:
            return

        for cache_type in self._alias_order:
            alias = self._alias_configs[cache_type]
            cache_obj = caches.get(alias.alias)
            if cache_obj is None:
                continue
            cache = cast(BaseCache, cache_obj)
            try:
                if hasattr(cache, "close"):
                    await cache.close()
                elif hasattr(cache, "disconnect"):
                    await cache.disconnect()  # type: ignore[call-arg]
            except Exception as exc:
                logger.debug(
                    "Error closing cache alias=%s: %s",
                    alias.alias,
                    exc,
                )

        self._closed = True
        logger.info("ModernCacheManager closed (aliases suffix=%s)", self._alias_suffix)

    # Key builder functions for different cache types
    def _embedding_key_builder(
        self,
        _func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Build cache key for embedding functions."""
        text = args[0] if args else kwargs.get("text", "")
        model = kwargs.get("model", "default")
        payload = {"text": text, "model": model}
        text_hash = _hash_text(_canonical_json(payload))
        return f"embed:{model}:{text_hash}"

    def _search_key_builder(
        self,
        _func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Build cache key for search functions."""
        query = args[0] if args else kwargs.get("query", "")
        filters = kwargs.get("filters", {})
        payload = {"query": query, "filters": filters}
        query_hash = _hash_text(_canonical_json(payload))
        return f"search:{query_hash}"

    def _crawl_key_builder(
        self,
        _func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Build cache key for crawl functions."""
        url = args[0] if args else kwargs.get("url", "")
        url_hash = _hash_text(url)
        return f"crawl:{url_hash}"

    def _hyde_key_builder(
        self,
        _func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Build cache key for HyDE functions."""
        query = args[0] if args else kwargs.get("query", "")
        query_hash = _hash_text(query)
        return f"hyde:{query_hash}"


# Convenience function for creating cache manager
def create_modern_cache_manager(  # pylint: disable=too-many-arguments
    redis_url: str = "redis://localhost:6379",
    key_prefix: str = "aidocs:",
    enable_compression: bool = True,
    config: Config | None = None,
    *,
    serializer_map: Mapping[str | CacheType, BaseSerializer] | None = None,
    use_pickle_for_embeddings: bool = False,
    command_timeout: float = 1.0,
) -> ModernCacheManager:
    """Create a modern cache manager instance.

    Args:
        redis_url: Redis connection URL
        key_prefix: Prefix for all cache keys
        enable_compression: Enable compression for cached values
        config: Application configuration
        serializer_map: Optional serializer overrides per cache type
        use_pickle_for_embeddings: Use Pickle serializer for embeddings cache
        command_timeout: Timeout for cache commands in seconds

    Returns:
        ModernCacheManager instance
    """
    return ModernCacheManager(
        redis_url=redis_url,
        key_prefix=key_prefix,
        enable_compression=enable_compression,
        config=config,
        serializer_map=serializer_map,
        use_pickle_for_embeddings=use_pickle_for_embeddings,
        command_timeout=command_timeout,
    )
