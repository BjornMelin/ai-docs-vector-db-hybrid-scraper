"""Behavioural tests for the simplified CacheManager implementation."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.services.cache._bulk_delete import delete_in_batches
from src.services.cache.base import CacheInterface
from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.cache.manager import CacheManager, CacheType


@pytest.mark.asyncio
async def test_cache_manager_round_trip(fakeredis_cache: DragonflyCache) -> None:
    """CacheManager should perform distributed set/get/delete operations."""

    manager = CacheManager(
        distributed_cache=fakeredis_cache,
        enable_distributed_cache=True,
        enable_specialized_caches=False,
    )

    key = "user:123"
    payload = {"value": 42}

    assert await manager.set(key, payload, cache_type=CacheType.SEARCH) is True
    cached = await manager.get(key, cache_type=CacheType.SEARCH)
    assert cached == payload

    assert await manager.delete(key, cache_type=CacheType.SEARCH) is True
    assert await manager.get(key, cache_type=CacheType.SEARCH) is None

    await manager.close()


@pytest.mark.asyncio
async def test_specialized_cache_ttl_overrides(fakeredis_cache: DragonflyCache) -> None:
    """Specialised caches inherit TTL overrides provided at construction time."""

    manager = CacheManager(
        distributed_cache=fakeredis_cache,
        enable_specialized_caches=True,
        distributed_ttl_seconds={
            CacheType.EMBEDDINGS: 120,
            CacheType.SEARCH: 45,
        },
    )

    assert manager.embedding_cache is not None
    assert manager.embedding_cache.default_ttl == 120
    assert manager.search_cache is not None
    assert manager.search_cache.default_ttl == 45

    await manager.close()


@pytest.mark.asyncio
async def test_clear_specific_cache_type_removes_distributed_keys(
    fakeredis_cache: DragonflyCache,
) -> None:
    """Clearing a cache type should evict matching Dragonfly keys."""

    manager = CacheManager(
        distributed_cache=fakeredis_cache,
        enable_specialized_caches=False,
    )

    logical_key = "session:42"
    assert await manager.set(logical_key, {"value": 42}, cache_type=CacheType.SEARCH)

    hashed_key = manager._get_cache_key(  # pylint: disable=protected-access
        logical_key, CacheType.SEARCH
    )
    assert await fakeredis_cache.exists(hashed_key)

    assert await manager.clear(CacheType.SEARCH) is True
    assert not await fakeredis_cache.exists(hashed_key)

    await manager.close()


@pytest.mark.asyncio
async def test_delete_in_batches_counts_successful_deletions() -> None:
    """delete_in_batches should tally successful deletion results."""

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

    deleted = await delete_in_batches(typed_cache, keys, batch_size=2)

    assert deleted == 2
    assert cache.calls == [["alpha_ok", "beta"], ["gamma_ok", "delta"]]


def test_cache_manager_warns_when_distributed_disabled(caplog) -> None:
    """Disabling Dragonfly caching should emit a configuration warning."""

    with caplog.at_level("WARNING"):
        CacheManager(enable_distributed_cache=False)

    assert "Distributed Dragonfly cache disabled" in caplog.text
