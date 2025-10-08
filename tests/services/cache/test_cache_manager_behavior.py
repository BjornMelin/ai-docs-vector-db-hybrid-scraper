"""Behavioural tests for the cache manager and helpers."""

import asyncio
from dataclasses import dataclass
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


def test_specialized_cache_defaults_to_safe_ttl(monkeypatch) -> None:
    """Embedding caches fall back to safe defaults but honor explicit TTLs."""

    class StubDistributedCache:
        def __init__(self, **_kwargs):
            self.enable_compression = True
            self.redis_url = "mock://redis"

        async def close(self) -> None:  # pragma: no cover - simple stub
            return None

    monkeypatch.setattr(
        "src.services.cache.manager.DragonflyCache",
        StubDistributedCache,
    )

    manager = CacheManager(
        enable_local_cache=False,
        enable_distributed_cache=True,
        enable_specialized_caches=True,
        enable_metrics=False,
        distributed_ttl_seconds={CacheType.EMBEDDINGS: 120},
    )

    assert manager.embedding_cache is not None
    assert manager.search_cache is not None
    assert manager.embedding_cache.default_ttl == 3600
    assert manager.search_cache.default_ttl == 3600

    run(manager.close())

    manager_with_explicit_ttl = CacheManager(
        enable_local_cache=False,
        enable_distributed_cache=True,
        enable_specialized_caches=True,
        enable_metrics=False,
        distributed_ttl_seconds={CacheType.REDIS: 7200},
    )

    assert manager_with_explicit_ttl.embedding_cache is not None
    assert manager_with_explicit_ttl.search_cache is not None
    assert manager_with_explicit_ttl.embedding_cache.default_ttl == 7200
    assert manager_with_explicit_ttl.search_cache.default_ttl == 7200

    run(manager_with_explicit_ttl.close())


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


def test_set_with_zero_ttl_evicts_entry(tmp_path: Path) -> None:
    """Ensure that explicit zero TTL invalidates the entry immediately."""

    manager = CacheManager(
        enable_local_cache=True,
        enable_distributed_cache=False,
        enable_specialized_caches=False,
        enable_metrics=False,
        local_cache_path=tmp_path,
    )

    run(
        manager.set(
            "ephemeral",
            {"value": "v"},
            cache_type=CacheType.LOCAL,
            ttl=0,
        )
    )
    assert run(manager.get("ephemeral", cache_type=CacheType.LOCAL)) is None

    run(manager.close())


@dataclass(slots=True)
class _StubDistributedCache:
    """Lightweight distributed cache stub for pattern clearing tests."""

    keys: list[str]
    deleted: list[str]

    async def scan_keys(self, pattern: str) -> list[str]:
        return list(self.keys)

    async def delete(self, key: str) -> bool:
        self.deleted.append(key)
        return True

    async def get(self, key: str) -> None:  # pragma: no cover - behaviourless stub
        return None


def test_clear_specific_cache_type_removes_local_and_distributed(
    tmp_path: Path,
) -> None:
    """Clear operations must purge both distributed and local storage layers."""

    manager = CacheManager(
        enable_local_cache=True,
        enable_distributed_cache=False,
        enable_specialized_caches=False,
        enable_metrics=False,
        local_cache_path=tmp_path,
    )

    run(
        manager.set(
            "session:42",
            {"value": 42},
            cache_type=CacheType.LOCAL,
            ttl=60,
        )
    )
    hashed_key = manager._get_cache_key(  # pylint: disable=protected-access
        "session:42", CacheType.LOCAL
    )

    persisted_path = cache_path_for_key(tmp_path, hashed_key)
    assert persisted_path.exists()

    stub = _StubDistributedCache(keys=[hashed_key], deleted=[])
    manager._distributed_cache = cast(Any, stub)  # pylint: disable=protected-access

    assert run(manager._clear_specific_cache_type(CacheType.LOCAL)) is True  # pylint: disable=protected-access
    assert stub.deleted == [hashed_key]
    assert run(manager.get("session:42", cache_type=CacheType.LOCAL)) is None
    assert not persisted_path.exists()

    run(manager.close())
