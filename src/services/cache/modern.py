"""Modern caching implementation using aiocache.

This module provides a modernized caching implementation that replaces
custom caching patterns with the battle-tested aiocache library.
Provides zero-boilerplate caching with automatic serialization and TTL management.
"""

import asyncio
import hashlib
import logging
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from aiocache import Cache, cached, multi_cached
from aiocache.backends.redis import RedisBackend
from aiocache.serializers import JsonSerializer, PickleSerializer

from src.config import CacheType, Config


logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModernCacheManager:
    """Modern cache manager using aiocache.

    Provides declarative caching with decorators and automatic
    serialization, TTL management, and cache invalidation.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aidocs:",
        enable_compression: bool = True,
        config: Config | None = None,
    ):
        """Initialize modern cache manager.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            enable_compression: Enable compression for cached values
            config: Application configuration for cache settings
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.enable_compression = enable_compression
        self.config = config

        # Configure cache backends
        self.cache_config = {
            Cache.REDIS: {
                "endpoint": redis_url,
                "serializer": PickleSerializer()
                if enable_compression
                else JsonSerializer(),
                "namespace": key_prefix,
                "timeout": 1,  # Connection timeout
                "retry_on_timeout": True,
            }
        }

        # Set up different cache instances for different purposes
        self.embedding_cache = Cache(
            Cache.REDIS,
            endpoint=redis_url,
            serializer=PickleSerializer(),
            namespace=f"{key_prefix}embeddings:",
            timeout=1,
        )

        self.search_cache = Cache(
            Cache.REDIS,
            endpoint=redis_url,
            serializer=JsonSerializer(),
            namespace=f"{key_prefix}search:",
            timeout=1,
        )

        self.crawl_cache = Cache(
            Cache.REDIS,
            endpoint=redis_url,
            serializer=JsonSerializer(),
            namespace=f"{key_prefix}crawl:",
            timeout=1,
        )

        self.hyde_cache = Cache(
            Cache.REDIS,
            endpoint=redis_url,
            serializer=JsonSerializer(),
            namespace=f"{key_prefix}hyde:",
            timeout=1,
        )

        # TTL settings from config or defaults
        if config and hasattr(config, "cache"):
            self.ttl_settings = getattr(config.cache, "cache_ttl_seconds", {})
        else:
            self.ttl_settings = {}

        # Default TTLs
        self.default_ttls = {
            CacheType.EMBEDDINGS: self.ttl_settings.get(
                CacheType.EMBEDDINGS, 86400 * 7
            ),  # 7 days
            CacheType.SEARCH: self.ttl_settings.get(CacheType.SEARCH, 3600),  # 1 hour
            CacheType.CRAWL: self.ttl_settings.get(CacheType.CRAWL, 3600),  # 1 hour
            CacheType.HYDE: self.ttl_settings.get(CacheType.HYDE, 3600),  # 1 hour
        }

        logger.info(f"ModernCacheManager initialized with Redis: {redis_url}")

    def get_cache_for_type(self, cache_type: CacheType) -> Cache:
        """Get the appropriate cache instance for a cache type.

        Args:
            cache_type: Type of cache needed

        Returns:
            Cache instance for the specified type
        """
        cache_map = {
            CacheType.EMBEDDINGS: self.embedding_cache,
            CacheType.SEARCH: self.search_cache,
            CacheType.CRAWL: self.crawl_cache,
            CacheType.HYDE: self.hyde_cache,
        }
        return cache_map.get(cache_type, self.search_cache)

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
            ttl=effective_ttl,
            cache=self.embedding_cache,
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
            ttl=effective_ttl,
            cache=self.search_cache,
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
            ttl=effective_ttl,
            cache=self.crawl_cache,
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
            ttl=effective_ttl,
            cache=self.hyde_cache,
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
        try:
            cache = self.get_cache_for_type(cache_type)
            value = await cache.get(key)
            return value if value is not None else default
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
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
        try:
            cache = self.get_cache_for_type(cache_type)
            effective_ttl = ttl or self.default_ttls[cache_type]
            await cache.set(key, value, ttl=effective_ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
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
        try:
            cache = self.get_cache_for_type(cache_type)
            await cache.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
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
                cache = self.get_cache_for_type(cache_type)
                await self._clear_cache_namespace(cache)
            else:
                # Clear all caches - use delete_many with pattern matching
                await asyncio.gather(
                    self._clear_cache_namespace(self.embedding_cache),
                    self._clear_cache_namespace(self.search_cache),
                    self._clear_cache_namespace(self.crawl_cache),
                    self._clear_cache_namespace(self.hyde_cache),
                    return_exceptions=True,
                )
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    async def _clear_cache_namespace(self, cache: Cache) -> None:
        """Clear all keys in a cache namespace."""
        try:
            # For aiocache, we can try to clear using the backend directly
            if hasattr(cache, "clear"):
                await cache.clear()
            elif hasattr(cache, "_backend") and hasattr(cache._backend, "clear"):
                await cache._backend.clear(namespace=cache.namespace)
            else:
                # Fallback: no-op for now - V2 will have proper implementation
                logger.warning(
                    f"Cannot clear cache {cache.namespace} - no clear method available"
                )
        except Exception as e:
            logger.warning(f"Failed to clear cache namespace {cache.namespace}: {e}")

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
        try:
            cache = self.get_cache_for_type(cache_type)
            # This would require extending aiocache or using Redis directly
            # For now, we'll implement a basic version
            if hasattr(cache, "delete_pattern"):
                return await cache.delete_pattern(pattern)
            logger.warning(f"Pattern invalidation not supported for {cache_type}")
            return 0
        except Exception as e:
            logger.error(f"Cache pattern invalidation error for {pattern}: {e}")
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            stats = {
                "manager": {
                    "redis_url": self.redis_url,
                    "key_prefix": self.key_prefix,
                    "compression_enabled": self.enable_compression,
                },
                "ttl_settings": self.default_ttls,
                "cache_types": {
                    "embeddings": {"namespace": f"{self.key_prefix}embeddings:"},
                    "search": {"namespace": f"{self.key_prefix}search:"},
                    "crawl": {"namespace": f"{self.key_prefix}crawl:"},
                    "hyde": {"namespace": f"{self.key_prefix}hyde:"},
                },
            }

            # Try to get cache-specific stats if available
            for cache_type, cache in [
                ("embeddings", self.embedding_cache),
                ("search", self.search_cache),
                ("crawl", self.crawl_cache),
                ("hyde", self.hyde_cache),
            ]:
                try:
                    if hasattr(cache, "get_stats"):
                        stats["cache_types"][cache_type][
                            "stats"
                        ] = await cache.get_stats()
                except (ConnectionError, OSError, PermissionError) as e:
                    pass  # Stats not available for this cache type

            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Clean up cache resources."""
        try:
            # Close all cache connections
            await asyncio.gather(
                self._close_cache(self.embedding_cache),
                self._close_cache(self.search_cache),
                self._close_cache(self.crawl_cache),
                self._close_cache(self.hyde_cache),
                return_exceptions=True,
            )
            logger.info("ModernCacheManager closed successfully")
        except Exception as e:
            logger.error(f"Error closing ModernCacheManager: {e}")

    async def _close_cache(self, cache: Cache) -> None:
        """Close a single cache instance."""
        try:
            if hasattr(cache, "close"):
                await cache.close()
            elif hasattr(cache, "_pool") and hasattr(cache._pool, "disconnect"):
                await cache._pool.disconnect()
        except Exception as e:
            logger.debug(f"Error closing cache: {e}")

    # Key builder functions for different cache types
    def _embedding_key_builder(self, func, *args, **kwargs) -> str:
        """Build cache key for embedding functions."""
        text = args[0] if args else kwargs.get("text", "")
        model = kwargs.get("model", "default")
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"embed:{model}:{text_hash}"

    def _search_key_builder(self, func, *args, **kwargs) -> str:
        """Build cache key for search functions."""
        query = args[0] if args else kwargs.get("query", "")
        filters = kwargs.get("filters", {})
        query_hash = hashlib.sha256(f"{query}:{filters}".encode()).hexdigest()[:12]
        return f"search:{query_hash}"

    def _crawl_key_builder(self, func, *args, **kwargs) -> str:
        """Build cache key for crawl functions."""
        url = args[0] if args else kwargs.get("url", "")
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        return f"crawl:{url_hash}"

    def _hyde_key_builder(self, func, *args, **kwargs) -> str:
        """Build cache key for HyDE functions."""
        query = args[0] if args else kwargs.get("query", "")
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
        return f"hyde:{query_hash}"


# Convenience function for creating cache manager
def create_modern_cache_manager(
    redis_url: str = "redis://localhost:6379",
    key_prefix: str = "aidocs:",
    enable_compression: bool = True,
    config: Config | None = None,
) -> ModernCacheManager:
    """Create a modern cache manager instance.

    Args:
        redis_url: Redis connection URL
        key_prefix: Prefix for all cache keys
        enable_compression: Enable compression for cached values
        config: Application configuration

    Returns:
        ModernCacheManager instance
    """
    return ModernCacheManager(redis_url, key_prefix, enable_compression, config)
