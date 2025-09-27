"""Tests for ModernCacheManager behavior."""

import pytest
from aiocache.serializers import JsonSerializer

from src.config import CacheType
from src.config.settings import Settings
from src.services.cache.modern import ModernCacheManager


@pytest.mark.asyncio
async def test_aliases_are_namespace_scoped() -> None:
    manager_a = ModernCacheManager(redis_url="memory://", key_prefix="aidocs:")
    manager_b = ModernCacheManager(redis_url="memory://", key_prefix="aidocs_v2:")

    try:
        alias_a = manager_a._alias_configs[CacheType.SEARCH].alias  # pylint: disable=protected-access
        alias_b = manager_b._alias_configs[CacheType.SEARCH].alias  # pylint: disable=protected-access
        assert alias_a != alias_b
        assert alias_a.startswith("search:")
        assert alias_b.startswith("search:")
    finally:
        await manager_a.close()
        await manager_b.close()


@pytest.mark.asyncio
async def test_default_serializer_is_json() -> None:
    manager = ModernCacheManager(redis_url="memory://")
    try:
        alias = manager._alias_configs[CacheType.SEARCH]  # pylint: disable=protected-access
        assert isinstance(alias.serializer, JsonSerializer)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_ttl_override_accepts_string_keys() -> None:
    settings = Settings()
    settings.cache.cache_ttl_seconds = {"SEARCH": 42}

    manager = ModernCacheManager(redis_url="memory://", config=settings)
    try:
        assert manager.default_ttls[CacheType.SEARCH] == 42
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_clear_removes_entries() -> None:
    manager = ModernCacheManager(redis_url="memory://", key_prefix="aidocs:")
    try:
        await manager.set("foo", {"value": 1}, cache_type=CacheType.SEARCH)
        assert await manager.get("foo", cache_type=CacheType.SEARCH) == {"value": 1}

        await manager.clear()
        assert await manager.get("foo", cache_type=CacheType.SEARCH) is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_search_key_builder_is_deterministic() -> None:
    manager = ModernCacheManager(redis_url="memory://", key_prefix="aidocs:")
    try:
        filters_a = {"b": 2, "a": 1}
        filters_b = {"a": 1, "b": 2}
        hash_a = manager._search_key_builder(None, "query", filters=filters_a)  # pylint: disable=protected-access
        hash_b = manager._search_key_builder(None, "query", filters=filters_b)  # pylint: disable=protected-access
        assert hash_a == hash_b
    finally:
        await manager.close()
