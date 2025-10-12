"""Browser automation result caching for UnifiedBrowserManager.

This module provides caching functionality for browser automation results
to avoid redundant scrapes and improve performance.
"""

import hashlib
import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

from src.services.browser.models import ProviderKind

from .base import CacheInterface
from .persistent_cache import PersistentCacheManager


logger = logging.getLogger(__name__)


class BrowserCacheEntry:
    """Entry for browser automation cache."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        url: str,
        content: str,
        metadata: dict[str, Any],
        provider: ProviderKind | str,
        timestamp: float | None = None,
    ) -> None:
        """Initialize cache entry.

        Args:
            url: Source URL
            content: Scraped content
            metadata: Page metadata
            provider: Which provider fulfilled the scrape
            timestamp: When the content was scraped
        """

        self.url = url
        self.content = content
        self.metadata = metadata
        self.provider = (
            provider if isinstance(provider, ProviderKind) else ProviderKind(provider)
        )
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for caching."""

        return {
            "url": self.url,
            "content": self.content,
            "metadata": self.metadata,
            "provider": self.provider.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserCacheEntry":
        """Create from dictionary."""

        return cls(
            url=data["url"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            provider=data.get("provider", ProviderKind.LIGHTWEIGHT.value),
            timestamp=data.get("timestamp", time.time()),
        )


class BrowserCache(CacheInterface[BrowserCacheEntry]):
    """Cache for browser automation results.

    This cache stores scraped content to avoid redundant browser automation
    calls. It uses URL and tier as cache keys, with configurable TTLs
    based on content type and domain.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        local_cache: PersistentCacheManager | None = None,
        distributed_cache: CacheInterface[str] | None = None,
        default_ttl: int = 3600,  # 1 hour default
        dynamic_content_ttl: int = 300,  # 5 minutes for dynamic content
        static_content_ttl: int = 86400,  # 24 hours for static content
    ) -> None:
        """Initialize browser cache.

        Args:
        local_cache: Local persistent cache implementation
            distributed_cache: Distributed cache implementation
            default_ttl: Default TTL in seconds
            dynamic_content_ttl: TTL for dynamic content
            static_content_ttl: TTL for static content
        """

        self.local_cache = local_cache
        self.distributed_cache = distributed_cache
        self.default_ttl = default_ttl
        self.dynamic_content_ttl = dynamic_content_ttl
        self.static_content_ttl = static_content_ttl
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

    def _generate_cache_key(self, url: str, provider: str | None = None) -> str:
        """Generate cache key from URL and tier.

        Args:
            url: URL to cache
            tier: Tier used for scraping

        Returns:
            Cache key string
        """
        # Normalize URL
        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            # Sort query parameters for consistency
            query_params = sorted(parsed.query.split("&"))
            normalized_url += f"?{'&'.join(query_params)}"

        # Include tier in key if specified
        key_parts = [normalized_url]
        if provider:
            key_parts.append(f"provider:{provider}")

        # Generate hash for consistent key length (using SHA256 for security)
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"browser:{key_hash}:{parsed.netloc}"

    def generate_cache_key(self, url: str, provider: str | None = None) -> str:
        """Generate cache key from URL and tier (public interface).

        Args:
            url: URL to cache
            tier: Tier used for scraping

        Returns:
            Cache key string
        """

        return self._generate_cache_key(url, provider)

    def _determine_ttl(self, url: str, content_length: int) -> int:
        """Determine appropriate TTL based on URL and content.

        Args:
            url: URL that was scraped
            content_length: Length of scraped content

        Returns:
            TTL in seconds
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Static content patterns
        static_patterns = [
            ".pdf",
            ".txt",
            ".json",
            ".xml",
            ".csv",
            "/docs/",
            "/documentation/",
            "/api/reference/",
            "raw.githubusercontent.com",
            "gist.github.com",
        ]

        # Dynamic content patterns
        dynamic_patterns = [
            "/search",
            "/query",
            "/api/",
            "/graphql",
            "api.",
            "twitter.com",
            "x.com",
            "linkedin.com",
            "/feed",
            "/stream",
            "/live",
        ]

        # Check for static content
        for pattern in static_patterns:
            if pattern in path or pattern in domain:
                return self.static_content_ttl

        # Check for dynamic content
        for pattern in dynamic_patterns:
            if pattern in path or pattern in domain:
                return self.dynamic_content_ttl

        # Default TTL
        return self.default_ttl

    async def get(self, key: str) -> BrowserCacheEntry | None:
        """Get browser cache entry.

        Args:
            key: Cache key (use _generate_cache_key)

        Returns:
            Cached entry or None
        """
        # Try local cache first
        local_result = await self._try_local_cache_get(key)
        if local_result is not None:
            return local_result

        # Try distributed cache
        distributed_result = await self._try_distributed_cache_get(key)
        if distributed_result is not None:
            return distributed_result

        self._cache_stats["misses"] += 1
        return None

    async def _try_local_cache_get(self, key: str) -> BrowserCacheEntry | None:
        """Try to get entry from local cache."""

        if not self.local_cache:
            return None
        try:
            cached_json = await self.local_cache.get(key)
            if cached_json:
                self._cache_stats["hits"] += 1
                data = json.loads(cached_json)
                logger.debug("Browser cache hit (local): %s", key)
                return BrowserCacheEntry.from_dict(data)
        except (ConnectionError, FileNotFoundError, OSError, PermissionError) as e:
            logger.warning("Error reading from local browser cache: %s", e)
        return None

    async def _try_distributed_cache_get(self, key: str) -> BrowserCacheEntry | None:
        """Try to get entry from distributed cache with local promotion."""

        if not self.distributed_cache:
            return None
        try:
            cached_json = await self.distributed_cache.get(key)
        except (ConnectionError, FileNotFoundError, OSError, PermissionError) as e:
            logger.warning("Error reading from distributed browser cache: %s", e)
            return None

        if cached_json:
            return await self._process_distributed_cache_hit(key, cached_json)
        return None

    async def _process_distributed_cache_hit(
        self, key: str, cached_json: str
    ) -> BrowserCacheEntry:
        """Process distributed cache hit by parsing data + promoting to local cache."""
        self._cache_stats["hits"] += 1
        data = json.loads(cached_json)
        entry = BrowserCacheEntry.from_dict(data)

        # Promote to local cache
        await self._promote_to_local_cache(key, cached_json)

        logger.debug("Browser cache hit (distributed): %s", key)
        return entry

    async def _promote_to_local_cache(self, key: str, cached_json: str) -> None:
        """Promote distributed cache hit to local cache."""
        if not self.local_cache:
            return
        try:
            await self.local_cache.set(key, cached_json, ttl_seconds=300)  # 5 min local
        except (ConnectionError, FileNotFoundError, OSError, PermissionError) as e:
            logger.warning("Error promoting to local cache for key %s: %s", key, e)

    async def set(
        self,
        key: str,
        value: BrowserCacheEntry,
        ttl: int | None = None,
    ) -> bool:
        """Cache browser automation result.

        Args:
            key: Cache key (use _generate_cache_key)
            value: Cache entry to store
            ttl: Override TTL in seconds

        Returns:
            Success status
        """
        if ttl is None:
            ttl = self._determine_ttl(value.url, len(value.content))

        # Serialize the cache entry
        try:
            cached_json = json.dumps(value.to_dict())
            self._cache_stats["sets"] += 1
        except (TypeError, ValueError) as e:
            logger.error("Error serializing browser cache entry: %s", e)
            return False

        # Store in both caches
        success = await self._store_in_both_caches(key, cached_json, ttl)

        if success:
            logger.debug(
                "Cached browser result: %s (TTL: %ds, size: %d)",
                key,
                ttl,
                len(cached_json),
            )

        return success

    async def _store_in_both_caches(self, key: str, cached_json: str, ttl: int) -> bool:
        """Store cached JSON in both local and distributed caches."""
        results = []

        # Store in local cache
        local_result = await self._store_in_local_cache(key, cached_json, ttl)
        if local_result is not None:
            results.append(local_result)

        # Store in distributed cache
        distributed_result = await self._store_in_distributed_cache(
            key, cached_json, ttl
        )
        if distributed_result is not None:
            results.append(distributed_result)

        return any(results) if results else False

    async def _store_in_local_cache(
        self, key: str, cached_json: str, ttl: int
    ) -> bool | None:
        """Store in local cache with error handling."""
        if not self.local_cache:
            return None
        try:
            return await self.local_cache.set(
                key, cached_json, ttl_seconds=min(ttl, 3600)
            )
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning("Error storing in local cache: %s", e)
            return False

    async def _store_in_distributed_cache(
        self, key: str, cached_json: str, ttl: int
    ) -> bool | None:
        """Store in distributed cache with error handling."""
        if not self.distributed_cache:
            return None
        try:
            return await self.distributed_cache.set(key, cached_json, ttl=ttl)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning("Error storing in distributed cache: %s", e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete cached entry.

        Args:
            key: Cache key to delete

        Returns:
            Success status
        """
        results = []

        # Delete from local cache
        local_result = await self._delete_from_local_cache(key)
        if local_result is not None:
            results.append(local_result)

        # Delete from distributed cache
        distributed_result = await self._delete_from_distributed_cache(key)
        if distributed_result is not None:
            results.append(distributed_result)

        return any(results) if results else False

    async def _delete_from_local_cache(self, key: str) -> bool | None:
        """Delete from local cache with error handling."""
        if not self.local_cache:
            return None
        try:
            return await self.local_cache.delete(key)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning("Error deleting from local cache: %s", e)
            return False

    async def _delete_from_distributed_cache(self, key: str) -> bool | None:
        """Delete from distributed cache with error handling."""
        if not self.distributed_cache:
            return None
        try:
            return await self.distributed_cache.delete(key)
        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.warning("Error deleting from distributed cache: %s", e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match (e.g., domain name)

        Returns:
            Number of entries invalidated
        """
        count = 0

        # For now, we can only invalidate if using distributed cache + pattern support
        if self.distributed_cache and hasattr(
            self.distributed_cache, "invalidate_pattern"
        ):
            count = await self._invalidate_distributed_pattern(pattern)

        return count

    async def _invalidate_distributed_pattern(self, pattern: str) -> int:
        """Invalidate pattern in distributed cache with error handling."""

        try:
            count = await self.distributed_cache.invalidate_pattern(  # type: ignore
                f"browser:*{pattern}*"
            )
        except (ConnectionError, OSError, PermissionError) as e:
            logger.error("Error invalidating browser cache pattern: %s", e)
            return 0

        self._cache_stats["evictions"] += count
        logger.info("Invalidated %d browser cache entries matching %s", count, pattern)
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""

        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        return {
            **self._cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    async def get_or_fetch(
        self,
        url: str,
        provider: str | ProviderKind | None,
        fetch_func,
    ) -> tuple[BrowserCacheEntry, bool]:
        """Get from cache or fetch if not cached.

        Args:
            url: URL to get/fetch
            provider: Provider to use for cache key differentiation
            fetch_func: Async function to fetch content

        Returns:
            Tuple of (cache entry, was_cached)
        """
        # Generate cache key
        provider_key = (
            provider.value if isinstance(provider, ProviderKind) else provider
        )
        cache_key = self._generate_cache_key(url, provider_key)

        # Try cache first
        cached_entry = await self.get(cache_key)
        if cached_entry:
            logger.info(
                "Browser cache hit for %s (tier: %s, age: %.1fs)",
                url,
                cached_entry.provider.value,
                time.time() - cached_entry.timestamp,
            )
            return cached_entry, True

        # Fetch fresh content
        logger.info("Browser cache miss for %s, fetching fresh content", url)

        try:
            result = await fetch_func()

            # Create cache entry
            entry = BrowserCacheEntry(
                url=url,
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                provider=result.get(
                    "provider",
                    provider_key or ProviderKind.LIGHTWEIGHT.value,
                ),
            )

            # Cache the result
            await self.set(cache_key, entry)

            return entry, False

        except (ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Error fetching content for %s: %s", url, e)
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if exists, False otherwise
        """
        # Check local cache first
        if self.local_cache:
            try:
                if await self.local_cache.exists(key):
                    return True
            except (ConnectionError, RuntimeError, TimeoutError):
                pass

        # Check distributed cache
        if self.distributed_cache:
            try:
                return await self.distributed_cache.exists(key)
            except (ConnectionError, RuntimeError, TimeoutError):
                pass

        return False

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """

        count = 0

        if self.local_cache:
            try:
                local_cleared = await self.local_cache.clear()
                if isinstance(local_cleared, int):
                    count += local_cleared
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.warning("Error clearing local cache: %s", e)

        if self.distributed_cache:
            try:
                # Don't double count if same entries in both
                await self.distributed_cache.clear()
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.warning("Error clearing distributed cache: %s", e)

        # Reset stats
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        return count

    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache
        """
        # Return size of distributed cache if available, otherwise local
        if self.distributed_cache:
            try:
                return await self.distributed_cache.size()
            except (ConnectionError, RuntimeError, TimeoutError):
                pass

        if self.local_cache:
            try:
                return await self.local_cache.size()
            except (ConnectionError, RuntimeError, TimeoutError):
                pass

        return 0

    async def close(self) -> None:
        """Close cache connections and cleanup resources."""
        if self.local_cache:
            try:
                await self.local_cache.close()
            except (ConnectionError, OSError, PermissionError) as e:
                logger.warning("Error closing local cache: %s", e)

        if self.distributed_cache:
            try:
                await self.distributed_cache.close()
            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.warning("Error closing distributed cache: %s", e)
