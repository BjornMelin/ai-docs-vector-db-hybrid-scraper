import typing
"""Browser automation result caching for UnifiedBrowserManager.

This module provides caching functionality for browser automation results
to avoid redundant scrapes and improve performance.
"""

import hashlib
import json  # noqa: PLC0415
import logging  # noqa: PLC0415
import time  # noqa: PLC0415
from typing import Any
from urllib.parse import urlparse

from .base import CacheInterface

logger = logging.getLogger(__name__)


class BrowserCacheEntry:
    """Entry for browser automation cache."""

    def __init__(
        self,
        url: str,
        content: str,
        metadata: dict[str, Any],
        tier_used: str,
        timestamp: float | None = None,
    ):
        """Initialize cache entry.

        Args:
            url: Source URL
            content: Scraped content
            metadata: Page metadata
            tier_used: Which tier was used for scraping
            timestamp: When the content was scraped
        """
        self.url = url
        self.content = content
        self.metadata = metadata
        self.tier_used = tier_used
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "url": self.url,
            "content": self.content,
            "metadata": self.metadata,
            "tier_used": self.tier_used,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserCacheEntry":
        """Create from dictionary."""
        return cls(
            url=data["url"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            tier_used=data.get("tier_used", "unknown"),
            timestamp=data.get("timestamp", time.time()),
        )


class BrowserCache(CacheInterface[BrowserCacheEntry]):
    """Cache for browser automation results.

    This cache stores scraped content to avoid redundant browser automation
    calls. It uses URL and tier as cache keys, with configurable TTLs
    based on content type and domain.
    """

    def __init__(
        self,
        local_cache: CacheInterface[str] | None = None,
        distributed_cache: CacheInterface[str] | None = None,
        default_ttl: int = 3600,  # 1 hour default
        dynamic_content_ttl: int = 300,  # 5 minutes for dynamic content
        static_content_ttl: int = 86400,  # 24 hours for static content
    ):
        """Initialize browser cache.

        Args:
            local_cache: Local cache implementation
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

    def _generate_cache_key(self, url: str, tier: str | None = None) -> str:
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
        if tier:
            key_parts.append(f"tier:{tier}")

        # Generate hash for consistent key length
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]

        return f"browser:{key_hash}:{parsed.netloc}"

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
        if self.local_cache:
            try:
                cached_json = await self.local_cache.get(key)
                if cached_json:
                    self._cache_stats["hits"] += 1
                    data = json.loads(cached_json)
                    logger.debug(f"Browser cache hit (local): {key}")
                    return BrowserCacheEntry.from_dict(data)
            except Exception as e:
                logger.warning(f"Error reading from local browser cache: {e}")

        # Try distributed cache
        if self.distributed_cache:
            try:
                cached_json = await self.distributed_cache.get(key)
                if cached_json:
                    self._cache_stats["hits"] += 1
                    data = json.loads(cached_json)
                    entry = BrowserCacheEntry.from_dict(data)

                    # Promote to local cache
                    if self.local_cache:
                        await self.local_cache.set(
                            key, cached_json, ttl=300
                        )  # 5 min local

                    logger.debug(f"Browser cache hit (distributed): {key}")
                    return entry
            except Exception as e:
                logger.warning(f"Error reading from distributed browser cache: {e}")

        self._cache_stats["misses"] += 1
        return None

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

        try:
            cached_json = json.dumps(value.to_dict())
            self._cache_stats["sets"] += 1

            # Store in both caches
            results = []

            if self.local_cache:
                result = await self.local_cache.set(
                    key, cached_json, ttl=min(ttl, 3600)
                )
                results.append(result)

            if self.distributed_cache:
                result = await self.distributed_cache.set(key, cached_json, ttl=ttl)
                results.append(result)

            success = any(results) if results else False
            if success:
                logger.debug(
                    f"Cached browser result: {key} (TTL: {ttl}s, size: {len(cached_json)})"
                )

            return success

        except Exception as e:
            logger.error(f"Error caching browser result: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cached entry.

        Args:
            key: Cache key to delete

        Returns:
            Success status
        """
        results = []

        if self.local_cache:
            try:
                result = await self.local_cache.delete(key)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error deleting from local cache: {e}")

        if self.distributed_cache:
            try:
                result = await self.distributed_cache.delete(key)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error deleting from distributed cache: {e}")

        return any(results) if results else False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match (e.g., domain name)

        Returns:
            Number of entries invalidated
        """
        count = 0

        # For now, we can only invalidate if using distributed cache with pattern support
        if self.distributed_cache and hasattr(
            self.distributed_cache, "invalidate_pattern"
        ):
            try:
                count = await self.distributed_cache.invalidate_pattern(
                    f"browser:*{pattern}*"
                )
                self._cache_stats["evictions"] += count
                logger.info(
                    f"Invalidated {count} browser cache entries matching {pattern}"
                )
            except Exception as e:
                logger.error(f"Error invalidating browser cache pattern: {e}")

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
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
        tier: str | None,
        fetch_func,
    ) -> tuple[BrowserCacheEntry, bool]:
        """Get from cache or fetch if not cached.

        Args:
            url: URL to get/fetch
            tier: Tier to use
            fetch_func: Async function to fetch content

        Returns:
            Tuple of (cache entry, was_cached)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(url, tier)

        # Try cache first
        cached_entry = await self.get(cache_key)
        if cached_entry:
            logger.info(
                f"Browser cache hit for {url} "
                f"(tier: {cached_entry.tier_used}, age: {time.time() - cached_entry.timestamp:.1f}s)"
            )
            return cached_entry, True

        # Fetch fresh content
        logger.info(f"Browser cache miss for {url}, fetching fresh content")

        try:
            result = await fetch_func()

            # Create cache entry
            entry = BrowserCacheEntry(
                url=url,
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                tier_used=result.get("tier_used", tier or "unknown"),
            )

            # Cache the result
            await self.set(cache_key, entry)

            return entry, False

        except Exception as e:
            logger.error(f"Error fetching content for {url}: {e}")
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
            except Exception:
                pass

        # Check distributed cache
        if self.distributed_cache:
            try:
                return await self.distributed_cache.exists(key)
            except Exception:
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
                count += await self.local_cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing local cache: {e}")

        if self.distributed_cache:
            try:
                # Don't double count if same entries in both
                await self.distributed_cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing distributed cache: {e}")

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
            except Exception:
                pass

        if self.local_cache:
            try:
                return await self.local_cache.size()
            except Exception:
                pass

        return 0

    async def close(self) -> None:
        """Close cache connections and cleanup resources."""
        if self.local_cache:
            try:
                await self.local_cache.close()
            except Exception as e:
                logger.warning(f"Error closing local cache: {e}")

        if self.distributed_cache:
            try:
                await self.distributed_cache.close()
            except Exception as e:
                logger.warning(f"Error closing distributed cache: {e}")
