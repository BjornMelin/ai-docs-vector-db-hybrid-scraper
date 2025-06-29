"""Simple mode cache service implementation.

Optimized for solo developer use with in-memory LRU cache and minimal overhead.
"""

import logging
import time
from typing import Any, Dict

from src.architecture.service_factory import BaseService


logger = logging.getLogger(__name__)


class SimpleCacheService(BaseService):
    """Simplified cache service for solo developer deployments.

    Features:
    - In-memory LRU cache only
    - No distributed caching
    - Simple TTL management
    - Minimal memory footprint
    """

    def __init__(self):
        super().__init__()
        self.cache: dict[str, dict[str, Any]] = {}
        self.max_size = 500  # Reduced from enterprise 10000
        self.max_memory_mb = 50  # Reduced from enterprise 1000
        self.default_ttl = 1800  # 30 minutes default TTL
        self._access_order: list[str] = []  # For LRU eviction

    async def initialize(self) -> None:
        """Initialize the simple cache service."""
        logger.info("Initializing simple cache service")
        self._mark_initialized()
        logger.info("Simple cache service initialized successfully")

    async def cleanup(self) -> None:
        """Clean up cache service resources."""
        self.cache.clear()
        self._access_order.clear()
        self._mark_cleanup()
        logger.info("Simple cache service cleaned up")

    def get_service_name(self) -> str:
        """Get the service name."""
        return "simple_cache_service"

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL
        if self._is_expired(entry):
            await self.delete(key)
            return None

        # Update access order for LRU
        self._update_access_order(key)

        return entry["value"]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        if ttl is None:
            ttl = self.default_ttl

        # Check if we need to evict entries
        await self._evict_if_needed()

        # Store the entry
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }

        # Update access order
        self._update_access_order(key)

        logger.debug(
            f"Cached value for key: {key}"
        )  # TODO: Convert f-string to logging format

    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        if key not in self.cache:
            return False

        entry = self.cache[key]
        if self._is_expired(entry):
            await self.delete(key)
            return False

        return True

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() > entry["expires_at"]

    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    async def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        # Clean up expired entries first
        await self._cleanup_expired()

        # If still over size limit, evict LRU entries
        while len(self.cache) >= self.max_size and self._access_order:
            lru_key = self._access_order[0]
            await self.delete(lru_key)
            logger.debug(
                f"Evicted LRU entry: {lru_key}"
            )  # TODO: Convert f-string to logging format

    async def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        expired_keys = []
        current_time = time.time()

        for key, entry in self.cache.items():
            if current_time > entry["expires_at"]:
                expired_keys.append(key)

        for key in expired_keys:
            await self.delete(key)

        if expired_keys:
            logger.debug(
                f"Cleaned up {len(expired_keys)} expired entries"
            )  # TODO: Convert f-string to logging format

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)

        # Calculate approximate memory usage (simplified)
        estimated_memory_kb = total_entries * 2  # Rough estimate: 2KB per entry

        return {
            "service_type": "simple",
            "total_entries": total_entries,
            "max_size": self.max_size,
            "estimated_memory_kb": estimated_memory_kb,
            "max_memory_mb": self.max_memory_mb,
            "default_ttl": self.default_ttl,
            "hit_rate": "N/A",  # Simplified - no hit rate tracking
            "features": {
                "distributed": False,
                "persistence": False,
                "compression": False,
                "analytics": False,
            },
        }

    async def get_cache_health(self) -> dict[str, Any]:
        """Get cache health information."""
        await self._cleanup_expired()

        stats = self.get_cache_stats()
        memory_usage_pct = (
            (stats["estimated_memory_kb"] / 1024) / self.max_memory_mb * 100
        )
        size_usage_pct = stats["total_entries"] / self.max_size * 100

        return {
            "status": "healthy",
            "memory_usage_percent": min(memory_usage_pct, 100),
            "size_usage_percent": size_usage_pct,
            "total_entries": stats["total_entries"],
            "estimated_memory_mb": stats["estimated_memory_kb"] / 1024,
        }
