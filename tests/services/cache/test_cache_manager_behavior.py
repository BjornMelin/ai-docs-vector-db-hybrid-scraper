"""Behavioural tests for the cache manager and helpers."""

import asyncio
from pathlib import Path
from typing import Any, cast

from src.services.cache._bulk_delete import delete_in_batches
from src.services.cache.base import CacheInterface
from src.services.cache.manager import CacheManager, CacheType
from src.services.cache.persistent_cache import cache_path_for_key


def run(coro):
    """Execute an async coroutine synchronously for test convenience.

    Args:
        coro: Awaitable under test.

    Returns:
        Result produced by the awaited coroutine.
    """

    return asyncio.run(coro)


def test_cache_manager_delete_removes_hashed_local_entry(tmp_path: Path) -> None:
    """Test that deleting a local cache entry removes its persisted file."""

    manager = CacheManager(
        enable_local_cache=True,
        enable_distributed_cache=False,
        enable_specialized_caches=False,
        enable_metrics=False,
        local_cache_path=tmp_path,
    )

    key = "user:123"
    payload = {"value": 42}

    run(manager.set(key, payload, cache_type=CacheType.LOCAL))

    hashed_key = manager._get_cache_key(  # pylint: disable=protected-access
        key, CacheType.LOCAL
    )
    storage_path = cache_path_for_key(tmp_path, hashed_key)
    assert storage_path.exists()

    assert run(manager.delete(key, cache_type=CacheType.LOCAL)) is True
    assert not storage_path.exists()

    run(manager.close())


def test_delete_in_batches_counts_successful_deletions() -> None:
    """Test that delete_in_batches correctly counts successful deletions."""

    class DummyCache:
        """Dummy cache with batch delete simulation."""

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def delete_many(self, keys: list[str]) -> dict[str, bool]:
            self.calls.append(keys)
            return {key: key.endswith("ok") for key in keys}

    cache = DummyCache()
    keys = ["alpha_ok", "beta", "gamma_ok", "delta"]

    typed_cache = cast(CacheInterface[Any], cache)
    deleted = run(delete_in_batches(typed_cache, keys, batch_size=2))

    assert deleted == 2
    assert cache.calls == [["alpha_ok", "beta"], ["gamma_ok", "delta"]]
