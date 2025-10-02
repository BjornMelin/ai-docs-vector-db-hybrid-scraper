"""Tests covering PersistentCacheManager semantics."""

import asyncio
import os
import time
from pathlib import Path

import pytest

import src.services.cache.persistent_cache as persistent_module
from src.services.cache.persistent_cache import (
    PersistentCacheManager,
    cache_path_for_key,
)


def run(coro):
    """Execute an async coroutine for test assertions.

    Args:
        coro: Awaitable providing the value under test.

    Returns:
        Result of awaiting ``coro``.
    """

    return asyncio.run(coro)


def test_set_get_respects_ttl(tmp_path: Path, monkeypatch) -> None:
    """Test that setting and getting respects the TTL expiration."""

    cache = PersistentCacheManager(base_path=tmp_path, persistence_enabled=True)
    run(cache.set("alpha", {"value": 1}, ttl_seconds=1))
    assert run(cache.get("alpha")) == {"value": 1}

    current = time.time()
    monkeypatch.setattr(persistent_module.time, "time", lambda: current + 2)
    assert run(cache.get("alpha")) is None


def test_persistence_round_trip(tmp_path: Path) -> None:
    """Test that a persisted entry can be rehydrated from disk."""

    cache = PersistentCacheManager(base_path=tmp_path, persistence_enabled=True)
    run(cache.set("beta", "payload", ttl_seconds=30))

    rehydrated = PersistentCacheManager(base_path=tmp_path, persistence_enabled=True)
    assert run(rehydrated.get("beta")) == "payload"


def test_delete_removes_persisted_entry(tmp_path: Path) -> None:
    """Test that deleting a key also removes its persisted file."""

    cache = PersistentCacheManager(base_path=tmp_path, persistence_enabled=True)
    run(cache.set("gamma", {"value": 2}, ttl_seconds=10))

    path = cache_path_for_key(tmp_path, "gamma")
    assert path.exists()

    run(cache.delete("gamma"))
    assert not path.exists()


def test_warm_one_key_loader(tmp_path: Path) -> None:
    """Test warming a single cache key using a loader function."""

    cache = PersistentCacheManager(base_path=tmp_path, persistence_enabled=False)

    async def loader() -> str:
        return "data"

    assert run(cache.warm_one_key("delta", loader, ttl_seconds=60)) is True
    assert run(cache.get("delta")) == "data"
    assert run(cache.warm_one_key("delta", loader)) is False


def test_stats_tracking(tmp_path: Path) -> None:
    """Test that cache hit/miss statistics are tracked correctly."""

    cache = PersistentCacheManager(base_path=tmp_path, persistence_enabled=False)
    run(cache.set("hit", "value"))
    assert run(cache.get("hit")) == "value"
    assert run(cache.get("miss")) is None

    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hits + stats.misses == 2


def test_memory_pressure_threshold(tmp_path: Path) -> None:
    """Test that memory pressure threshold evicts entries as expected."""

    cache = PersistentCacheManager(
        base_path=tmp_path,
        max_entries=5,
        max_memory_bytes=64,
        memory_pressure_threshold=0.5,
        persistence_enabled=False,
    )

    run(cache.set("large", "x" * 64))
    assert run(cache.get("large")) is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only permission hardening")
def test_persistent_cache_sets_directory_permissions(tmp_path: Path) -> None:
    """Persistent cache should harden directory permissions on POSIX hosts."""

    cache_dir = tmp_path / "local"
    cache = PersistentCacheManager(base_path=cache_dir, persistence_enabled=False)
    run(cache.close())

    mode = cache_dir.stat().st_mode & 0o777
    assert mode == 0o700
