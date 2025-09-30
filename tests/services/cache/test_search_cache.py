"""Tests for search result cache behaviour."""

import asyncio
from typing import Any, cast

from src.services.cache.search_cache import SearchResultCache


def run(coro):
    """Run an async coroutine in a synchronous context.

    Args:
        coro: Awaitable scheduled for execution.

    Returns:
        Result yielded by the awaited coroutine.
    """

    return asyncio.run(coro)


class FakeCache:
    """In-memory fake cache for testing."""

    def __init__(self) -> None:
        """Initialize in-memory fake cache."""

        self.storage: dict[str, object] = {}

    async def get(self, key: str):
        """Get a value from the cache."""

        return self.storage.get(key)

    async def set(self, key: str, value: object, ttl: int | None = None) -> bool:
        """Set a value in the cache."""

        self.storage[key] = value
        return True


async def _noop_increment(_self, _query: str) -> None:
    """No-op increment function for monkeypatching."""

    return None


async def _zero_popularity(_self, _query: str) -> int:
    """Return zero popularity for monkeypatching."""

    return 0


def test_search_cache_returns_cached_empty_results(monkeypatch) -> None:
    """Test that empty search results are cached and returned correctly."""

    fake_cache = FakeCache()
    cache = SearchResultCache(cast(Any, fake_cache), default_ttl=60)

    monkeypatch.setattr(
        SearchResultCache,
        "_increment_query_popularity",
        _noop_increment,
    )
    monkeypatch.setattr(
        SearchResultCache,
        "_get_query_popularity",
        _zero_popularity,
    )

    run(cache.set_search_results("no hits", [], collection_name="docs"))
    result = run(cache.get_search_results("no hits", collection_name="docs"))

    assert result == []
