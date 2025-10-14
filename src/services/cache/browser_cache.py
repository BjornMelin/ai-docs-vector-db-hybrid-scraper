"""Browser automation result caching backed by Dragonfly."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast
from urllib.parse import urlparse

from src.services.browser.models import ProviderKind

from .base import CacheInterface


logger = logging.getLogger(__name__)


class BrowserCacheEntry:
    """Entry describing cached browser automation results."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        url: str,
        content: str,
        metadata: dict[str, Any],
        provider: ProviderKind | str,
        timestamp: float | None = None,
    ) -> None:
        """Initialise a cache entry for browser content.

        Args:
            url: Source URL for the cached artefact.
            content: Scraped content payload.
            metadata: Additional metadata captured during scraping.
            provider: Provider responsible for producing the scrape.
            timestamp: Optional epoch timestamp when cached.
        """

        self.url = url
        self.content = content
        self.metadata = metadata
        self.provider = (
            provider if isinstance(provider, ProviderKind) else ProviderKind(provider)
        )
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the cache entry into a JSON-friendly payload."""

        return {
            "url": self.url,
            "content": self.content,
            "metadata": self.metadata,
            "provider": self.provider.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrowserCacheEntry:
        """Deserialise a cache entry from persisted JSON data."""

        return cls(
            url=data["url"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            provider=data.get("provider", ProviderKind.LIGHTWEIGHT.value),
            timestamp=data.get("timestamp", time.time()),
        )


class BrowserCache(CacheInterface[BrowserCacheEntry]):
    """Cache for browser automation results backed solely by Dragonfly."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        distributed_cache: CacheInterface[str] | None = None,
        default_ttl: int = 3600,
        dynamic_content_ttl: int = 300,
        static_content_ttl: int = 86400,
    ) -> None:
        """Create a browser cache instance.

        Args:
            distributed_cache: Distributed cache facade (Dragonfly-backed).
            default_ttl: Default TTL applied to cached entries.
            dynamic_content_ttl: TTL for dynamic pages.
            static_content_ttl: TTL for static artefacts.
        """

        self.distributed_cache = distributed_cache
        self.default_ttl = default_ttl
        self.dynamic_content_ttl = dynamic_content_ttl
        self.static_content_ttl = static_content_ttl
        self._cache_stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

    def _generate_cache_key(self, url: str, provider: str | None = None) -> str:
        """Return deterministic cache key derived from URL and provider."""

        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            query_params = sorted(parsed.query.split("&"))
            normalized_url += f"?{'&'.join(query_params)}"

        key_parts = [normalized_url]
        if provider:
            key_parts.append(f"provider:{provider}")

        key_hash = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()
        return f"browser:{key_hash}:{parsed.netloc}"

    def generate_cache_key(self, url: str, provider: str | None = None) -> str:
        """Generate a deterministic cache key for browser payloads.

        Args:
            url: Page URL that produced the cached payload.
            provider: Optional provider identifier used for routing.

        Returns:
            Canonical cache key string.
        """

        return self._generate_cache_key(url, provider)

    def _determine_ttl(self, url: str, content_length: int) -> int:
        """Derive TTL based on URL characteristics and payload size."""

        del content_length
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

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
        for pattern in static_patterns:
            if pattern in path or pattern in domain:
                return self.static_content_ttl

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
        for pattern in dynamic_patterns:
            if pattern in path or pattern in domain:
                return self.dynamic_content_ttl

        return self.default_ttl

    async def get(self, key: str) -> BrowserCacheEntry | None:
        """Return cached browser entry if present.

        Args:
            key: Cache key generated via :meth:`generate_cache_key`.

        Returns:
            Cached entry if found, otherwise ``None``.
        """

        if self.distributed_cache is None:
            self._cache_stats["misses"] += 1
            return None

        try:
            cached_json = await self.distributed_cache.get(key)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("Distributed browser cache read failed: %s", exc)
            self._cache_stats["misses"] += 1
            return None

        if not cached_json:
            self._cache_stats["misses"] += 1
            return None

        self._cache_stats["hits"] += 1
        data = json.loads(cached_json)
        logger.debug("Browser cache hit: %s", key)
        return BrowserCacheEntry.from_dict(data)

    async def set(
        self, key: str, value: BrowserCacheEntry, ttl: int | None = None
    ) -> bool:
        """Store browser cache entry in the distributed cache.

        Args:
            key: Cache key generated via :meth:`generate_cache_key`.
            value: Cache entry to serialise and store.
            ttl: Optional explicit TTL override in seconds.

        Returns:
            ``True`` when the value was persisted successfully.
        """

        if self.distributed_cache is None:
            logger.debug("Browser cache disabled; skipping set for %s", key)
            return False

        if ttl is None:
            ttl = self._determine_ttl(value.url, len(value.content))

        try:
            cached_json = json.dumps(value.to_dict())
        except (TypeError, ValueError) as exc:
            logger.exception("Failed serialising browser cache entry: %s", exc)
            return False

        try:
            success = await self.distributed_cache.set(key, cached_json, ttl=ttl)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("Distributed browser cache write failed: %s", exc)
            return False

        if success:
            self._cache_stats["sets"] += 1
            logger.debug("Cached browser result: %s (ttl=%s)", key, ttl)
        return success

    async def delete(self, key: str) -> bool:
        """Remove cached entry from the distributed cache.

        Args:
            key: Cache key to remove.

        Returns:
            ``True`` if the key was deleted, ``False`` otherwise.
        """

        if self.distributed_cache is None:
            return False
        try:
            return await self.distributed_cache.delete(key)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("Distributed browser cache delete failed: %s", exc)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern in the distributed cache.

        Args:
            pattern: Substring or pattern identifying affected keys.

        Returns:
            Number of cache entries invalidated.
        """

        if self.distributed_cache is None:
            return 0

        try:
            clear_fn = getattr(self.distributed_cache, "clear_pattern", None)
            if not callable(clear_fn):
                return 0

            clear_callable: Callable[[str], Awaitable[int]] = cast(
                Callable[[str], Awaitable[int]],
                clear_fn,
            )
            count = await clear_callable(f"browser:*{pattern}*")
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.exception("Browser cache pattern invalidation failed: %s", exc)
            return 0

        self._cache_stats["evictions"] += count
        return count

    def get_stats(self) -> dict[str, Any]:
        """Return counters describing cache behaviour.

        Returns:
            Dictionary containing hit/miss counters and derived metrics.
        """

        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests else 0.0
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
        """Return cached entry or fetch, cache, and return fresh content.

        Args:
            url: URL to retrieve from cache or fetch fresh content for.
            provider: Provider identifier used for cache key partitioning.
            fetch_func: Awaitable that returns a dictionary payload from a scrape.

        Returns:
            Tuple of the cache entry and a flag indicating whether it was cached.
        """

        provider_key = (
            provider.value if isinstance(provider, ProviderKind) else provider
        )
        cache_key = self._generate_cache_key(url, provider_key)

        cached_entry = await self.get(cache_key)
        if cached_entry is not None:
            logger.info(
                "Browser cache hit for %s (provider=%s, age=%.1fs)",
                url,
                cached_entry.provider.value,
                time.time() - cached_entry.timestamp,
            )
            return cached_entry, True

        logger.info("Browser cache miss for %s; invoking fetcher", url)
        result = await fetch_func()
        entry = BrowserCacheEntry(
            url=url,
            content=result.get("content", ""),
            metadata=result.get("metadata", {}),
            provider=result.get(
                "provider", provider_key or ProviderKind.LIGHTWEIGHT.value
            ),
        )
        await self.set(cache_key, entry)
        return entry, False

    async def exists(self, key: str) -> bool:
        """Return whether ``key`` is present in the distributed cache.

        Args:
            key: Cache key to check.

        Returns:
            ``True`` if the key exists, otherwise ``False``.
        """

        if self.distributed_cache is None:
            return False
        try:
            return await self.distributed_cache.exists(key)
        except (ConnectionError, OSError, TimeoutError):
            return False

    async def clear(self) -> int:
        """Clear distributed cache entries tracked by this cache.

        Returns:
            Number of entries removed from the distributed cache.
        """

        cleared = 0
        if self.distributed_cache is not None:
            try:
                cleared = await self.distributed_cache.clear()
            except (ConnectionError, OSError, TimeoutError) as exc:
                logger.warning("Distributed browser cache clear failed: %s", exc)

        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }
        return cleared

    async def size(self) -> int:
        """Return number of cached entries stored in Dragonfly.

        Returns:
            Estimated number of cached entries.
        """

        if self.distributed_cache is None:
            return 0
        try:
            return await self.distributed_cache.size()
        except (ConnectionError, OSError, TimeoutError):
            return 0

    async def close(self) -> None:
        """Close the distributed cache connection if supported."""

        if self.distributed_cache is None:
            return
        try:
            await self.distributed_cache.close()
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("Distributed browser cache close failed: %s", exc)
