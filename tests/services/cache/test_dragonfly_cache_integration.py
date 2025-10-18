"""Integration tests for DragonflyCache backed by fakeredis."""

import pytest

from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.cache.manager import CacheManager, CacheType


pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


@pytest.mark.skip(
    reason=(
        "Dragonfly cache integration requires redis service not present in test harness"
    )
)
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
        distributed_cache=fakeredis_cache,
        enable_specialized_caches=False,
    )

    logical_key = "session:777"
    hashed_key = manager._get_cache_key(  # pylint: disable=protected-access
        logical_key, CacheType.SEARCH
    )

    assert await manager.set(
        logical_key,
        {"payload": "value"},
        cache_type=CacheType.SEARCH,
    )

    assert await fakeredis_cache.exists(hashed_key)

    assert await manager.clear(CacheType.SEARCH) is True
    assert not await fakeredis_cache.exists(hashed_key)

    await manager.close()


@pytest.mark.asyncio
async def test_dragonfly_cache_mset_supports_ttl(
    fakeredis_cache: DragonflyCache,
) -> None:
    """Mset should accept TTL overrides and apply them to each key."""
    assert await fakeredis_cache.mset({"ttl:test": {"value": 1}}, ttl=5) is True
    ttl = await fakeredis_cache.ttl("ttl:test")
    assert 0 < ttl <= 5


@pytest.mark.asyncio
async def test_dragonfly_cache_delete_many_pipelines_requests(
    fakeredis_cache: DragonflyCache,
) -> None:
    """delete_many should pipeline deletions for better throughput."""
    await fakeredis_cache.set("batch:a", 1)
    await fakeredis_cache.set("batch:b", 2)

    result = await fakeredis_cache.delete_many(["batch:a", "batch:b"])

    assert result == {"batch:a": True, "batch:b": True}
    assert await fakeredis_cache.get("batch:a") is None
    assert await fakeredis_cache.get("batch:b") is None
