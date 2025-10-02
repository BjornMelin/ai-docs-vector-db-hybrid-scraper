"""Shared helpers for cache key invalidation operations."""

from collections.abc import Sequence
from typing import Any

from .base import CacheInterface


async def delete_in_batches(
    cache: CacheInterface[Any],
    keys: Sequence[str],
    batch_size: int = 100,
) -> int:
    """Delete cache keys in fixed-size batches.

    Args:
        cache: Cache implementation that exposes ``delete_many``.
        keys: Keys scheduled for deletion.
        batch_size: Maximum number of keys per ``delete_many`` invocation.

    Returns:
        Number of cache entries deleted successfully.
    """

    if not keys:
        return 0

    deleted_count = 0
    for index in range(0, len(keys), batch_size):
        batch = list(keys[index : index + batch_size])
        results = await cache.delete_many(batch)
        deleted_count += sum(bool(result) for result in results.values())

    return deleted_count
