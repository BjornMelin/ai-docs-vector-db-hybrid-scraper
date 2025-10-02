"""Caching utilities shared across query processing services."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from cachetools import LRUCache


@dataclass
class CacheTracker:
    """Track cache usage statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
    """

    hits: int = 0
    misses: int = 0

    def record_hit(self) -> None:
        """Increment the hit counter."""
        self.hits += 1

    def record_miss(self) -> None:
        """Increment the miss counter."""
        self.misses += 1


def _clone_value(value: Any) -> Any:
    """Return a defensive copy when possible.

    Args:
        value: Value to clone.

    Returns:
        Cloned value or original if cloning not supported.
    """

    if hasattr(value, "model_copy"):
        return value.model_copy(deep=True)
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:  # pragma: no cover - some types disallow copy kwargs
            return value
    return value


class CacheManager:
    """LRU cache wrapper that can return defensive copies."""

    def __init__(self, maxsize: int = 256, *, clone: bool = True) -> None:
        """Initialize the cache manager.

        Args:
            maxsize: Maximum number of cached items.
            clone: Whether to clone values on get/set operations.
        """
        self._store: LRUCache[str, Any] = LRUCache(maxsize=maxsize)
        self.tracker = CacheTracker()
        self._clone_enabled = clone

    def _prepare_for_store(self, value: Any) -> Any:
        """Prepare value for storage in cache.

        Args:
            value: Value to store.

        Returns:
            Cloned value if cloning is enabled.
        """
        if not self._clone_enabled:
            return value
        return _clone_value(value)

    def _prepare_for_return(self, value: Any) -> Any:
        """Prepare value for return from cache.

        Args:
            value: Value to return.

        Returns:
            Cloned value if cloning is enabled.
        """
        if not self._clone_enabled:
            return value
        return _clone_value(value)

    def get(self, key: str) -> Any | None:
        """Return a cached value if present.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """

        value = self._store.get(key)
        if value is None:
            self.tracker.record_miss()
            return None
        self.tracker.record_hit()
        return self._prepare_for_return(value)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to store.
        """

        self._store[key] = self._prepare_for_store(value)

    def delete(self, key: str) -> None:
        """Remove a key from the cache if present.

        Args:
            key: Cache key to remove.
        """

        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()
        self.tracker = CacheTracker()

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists and is a string.
        """
        if not isinstance(key, str):
            return False
        return key in self._store

    def __getitem__(self, key: str) -> Any:
        """Get item with dict-like access.

        Args:
            key: Cache key.

        Returns:
            Cached value.

        Raises:
            KeyError: If key not found.
        """
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item with dict-like access.

        Args:
            key: Cache key.
            value: Value to store.
        """
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item with dict-like access.

        Args:
            key: Cache key to delete.

        Raises:
            KeyError: If key not found.
        """
        if key not in self._store:
            raise KeyError(key)
        self.delete(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over cache keys."""
        return iter(self._store.keys())

    def keys(self) -> Iterable[str]:
        """Return all cache keys."""
        return list(self._store.keys())

    def items(self) -> Iterable[tuple[str, Any]]:
        """Return all cache items as key-value pairs."""
        return [(key, self.get(key)) for key in self._store]

    def values(self) -> Iterable[Any]:
        """Return all cache values."""
        return [self.get(key) for key in self._store]

    def __eq__(self, other: object) -> bool:
        """Compare cache managers by content.

        Args:
            other: Object to compare with.

        Returns:
            True if contents are equal.
        """
        if isinstance(other, CacheManager):
            return self.snapshot() == other.snapshot()
        if isinstance(other, Mapping):
            return self.snapshot() == dict(other)
        return NotImplemented

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the cache contents.

        Returns:
            Dictionary of cache contents.
        """

        return {
            key: self._prepare_for_return(value) for key, value in self._store.items()
        }


def build_cache_key(*parts: str) -> str:
    """Build a stable cache key using a Blake2b digest.

    Args:
        *parts: String parts to include in the key.

    Returns:
        Hexadecimal digest of the combined parts.
    """

    normalized = "|".join(parts)
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16)
    return digest.hexdigest()
