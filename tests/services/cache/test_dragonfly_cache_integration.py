"""Integration tests for DragonflyCache backed by fakeredis."""

import pytest

from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.cache.manager import CacheManager, CacheType


@pytest.mark.asyncio
async def test_dragonfly_cache_round_trip(fakeredis_cache: DragonflyCache) -> None:
    """Ensure values stored in DragonflyCache round-trip correctly."""

    payload = {"value": 42, "nested": {"flag": True}}
    assert await fakeredis_cache.set("integration:test", payload) is True

    cached = await fakeredis_cache.get("integration:test")
    assert cached == payload

    ttl = await fakeredis_cache.ttl("integration:test")
    assert ttl == fakeredis_cache.default_ttl


@pytest.mark.asyncio
async def test_cache_manager_clear_specific_cache_type_uses_correct_prefix(
    fakeredis_cache: DragonflyCache,
) -> None:
    """CacheManager.clear(cache_type=...) should remove prefixed keys exactly once."""

    manager = CacheManager(
        enable_local_cache=False,
        enable_distributed_cache=True,
        enable_specialized_caches=False,
        distributed_cache=fakeredis_cache,
    )

    logical_key = "session:777"
    hashed_key = manager._get_cache_key(  # pylint: disable=protected-access
        logical_key, CacheType.REDIS
    )
    distributed_key = hashed_key[len(manager.key_prefix) :]

    assert await manager.set(
        logical_key,
        {"payload": "value"},
        cache_type=CacheType.REDIS,
    )

    assert await fakeredis_cache.exists(distributed_key)

    assert await manager.clear(CacheType.REDIS) is True
    assert not await fakeredis_cache.exists(distributed_key)

    await manager.close()
