"""Base cache interface for all cache implementations."""

from abc import ABC
from abc import abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class CacheInterface(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get value from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: T,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close cache connections and cleanup resources."""
        pass

    async def get_many(self, keys: list[str]) -> dict[str, T | None]:
        """Get multiple values from cache.

        Default implementation calls get() for each key.
        Subclasses can override for batch operations.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (None if not found)
        """
        results = {}
        for key in keys:
            results[key] = await self.get(key)
        return results

    async def set_many(
        self,
        items: dict[str, T],
        ttl: int | None = None,
    ) -> dict[str, bool]:
        """Set multiple values in cache.

        Default implementation calls set() for each item.
        Subclasses can override for batch operations.

        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            Dictionary mapping keys to success status
        """
        results = {}
        for key, value in items.items():
            results[key] = await self.set(key, value, ttl)
        return results

    async def delete_many(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values from cache.

        Default implementation calls delete() for each key.
        Subclasses can override for batch operations.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to deletion status
        """
        results = {}
        for key in keys:
            results[key] = await self.delete(key)
        return results
