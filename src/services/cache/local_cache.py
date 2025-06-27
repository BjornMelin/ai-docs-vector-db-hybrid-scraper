import typing
"""Local in-memory LRU cache implementation with TTL support."""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415
import sys
import time  # noqa: PLC0415
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .base import CacheInterface

logger = logging.getLogger(__name__)

# Import monitoring registry for metrics integration
try:
    from ..monitoring.metrics import get_metrics_registry

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""

    value: Any
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class LocalCache(CacheInterface[Any]):
    """Thread-safe local LRU cache with TTL support."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = 300,  # 5 minutes
        max_memory_mb: float = 100.0,
    ):
        """Initialize local cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None for no expiration)
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        self._cache: Ordereddict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory = 0

        # Initialize metrics registry if available
        self.metrics_registry = None
        if MONITORING_AVAILABLE:
            try:
                self.metrics_registry = get_metrics_registry()
                logger.debug("Local cache monitoring enabled")
            except Exception as e:
                logger.debug(f"Local cache monitoring disabled: {e}")

    async def get(self, key: str) -> Any | None:
        """Get value from cache with LRU update."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._update_memory_usage(key, entry.value, remove=True)
                self._misses += 1
                self._evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache with TTL."""
        # Calculate expiration time
        if ttl is None:
            ttl = self.default_ttl

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        async with self._lock:
            # Check memory constraint
            value_size = self._estimate_size(value)
            if value_size > self.max_memory_bytes:
                return False  # Value too large

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._update_memory_usage(key, old_entry.value, remove=True)

            # Evict entries if needed
            await self._evict_if_needed(value_size)

            # Add new entry
            self._cache[key] = CacheEntry(value, expires_at)
            self._update_memory_usage(key, value, remove=False)

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            entry = self._cache.pop(key, None)
            if entry is not None:
                self._update_memory_usage(key, entry.value, remove=True)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                self._update_memory_usage(key, entry.value, remove=True)
                self._evictions += 1
                return False

            return True

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_memory = 0
            self._evictions += count
            return count

    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            # Clean up expired entries
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._update_memory_usage(key, entry.value, remove=True)
                self._evictions += 1

            return len(self._cache)

    async def close(self) -> None:
        """Close cache (no-op for local cache)."""
        await self.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "memory_bytes": self._current_memory,
            "memory_mb": self._current_memory / (1024 * 1024),
        }

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._current_memory / (1024 * 1024)

    async def _evict_if_needed(self, new_value_size: int) -> None:
        """Evict entries if cache is full or memory limit exceeded."""
        # Evict by size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # Evict by memory limit
        while (
            self._current_memory + new_value_size > self.max_memory_bytes
            and len(self._cache) > 0
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Pop first item (least recently used)
        key, entry = self._cache.popitem(last=False)
        self._update_memory_usage(key, entry.value, remove=True)
        self._evictions += 1

        # Record eviction in monitoring metrics
        if self.metrics_registry:
            self.metrics_registry._metrics["cache_evictions"].labels(
                cache_type="local", cache_name="main"
            ).inc()

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        # Simple estimation - can be improved
        if isinstance(value, str | bytes):
            return len(value)
        elif isinstance(value, list | tuple):
            return sum(self._estimate_size(item) for item in value) + sys.getsizeof(
                value
            )
        elif isinstance(value, dict):
            size = sys.getsizeof(value)
            for k, v in value.items():
                size += self._estimate_size(k) + self._estimate_size(v)
            return size
        else:
            # Default size estimate
            return sys.getsizeof(value)

    def _update_memory_usage(self, key: str, value: Any, remove: bool) -> None:
        """Update current memory usage tracking."""
        key_size = sys.getsizeof(key)
        value_size = self._estimate_size(value)
        total_size = key_size + value_size + sys.getsizeof(CacheEntry)

        if remove:
            self._current_memory -= total_size
        else:
            self._current_memory += total_size

        # Ensure non-negative
        self._current_memory = max(0, self._current_memory)

        # Update memory usage metrics
        if self.metrics_registry:
            self.metrics_registry.update_cache_memory_usage(
                "local", "main", self._current_memory
            )
