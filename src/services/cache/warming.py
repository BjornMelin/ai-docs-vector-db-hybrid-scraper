"""Helpers for warming cache keys via asynchronous loaders."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .persistent_cache import PersistentCacheManager


class CacheWarmer:
    """Warm cache entries by invoking asynchronous loader callbacks."""

    def __init__(self, cache_manager: PersistentCacheManager) -> None:
        """Store cache manager reference for warming operations."""

        self.cache_manager = cache_manager

    async def warm_key(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        *,
        ttl: int | None = None,
    ) -> bool:
        """Populate a single key using ``loader`` if it is missing."""

        return await self.cache_manager.warm_one_key(key, loader, ttl_seconds=ttl)

    async def warm_batch(
        self,
        items: Iterable[tuple[str, Callable[[], Awaitable[Any]], int | None]],
    ) -> int:
        """Warm multiple keys sequentially.

        Args:
            items: Iterable of ``(key, loader, ttl)`` tuples.

        Returns:
            Number of keys populated by this call.
        """

        populated = 0
        for key, loader, ttl in items:
            if await self.warm_key(key, loader, ttl=ttl):
                populated += 1
        return populated
