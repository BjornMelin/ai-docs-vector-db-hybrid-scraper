from unittest.mock import AsyncMock

import pytest
from src.services.cache.dragonfly_cache import DragonflyCache
from src.services.cache.embedding_cache import EmbeddingCache
from src.services.cache.search_cache import SearchResultCache


@pytest.mark.asyncio
async def test_embedding_cache_warm_and_invalidate():
    df_cache = DragonflyCache()
    df_cache.exists = AsyncMock(side_effect=[False, True])
    df_cache.set = AsyncMock(return_value=True)
    df_cache.scan_keys = AsyncMock(
        return_value=[
            "emb:openai:model:hash1",
            "emb:openai:model:hash2",
        ]
    )
    df_cache.delete_many = AsyncMock(
        return_value={
            "emb:openai:model:hash1": True,
            "emb:openai:model:hash2": True,
        }
    )

    cache = EmbeddingCache(cache=df_cache)

    missing = await cache.warm_cache(["text1", "text2"], "model")
    assert missing == ["text1"]

    success = await cache.set_embedding("text1", "model", [0.1, 0.2])
    assert success is True

    deleted = await cache.invalidate_model("model")
    assert deleted == 2


@pytest.mark.asyncio
async def test_search_cache_operations():
    df_cache = DragonflyCache()
    df_cache.get = AsyncMock(return_value=None)
    df_cache.set = AsyncMock(return_value=True)
    df_cache.scan_keys = AsyncMock(return_value=["search:default:hash"])
    df_cache.delete_many = AsyncMock(return_value={"search:default:hash": True})

    cache = SearchResultCache(cache=df_cache)

    assert await cache.get_search_results("query") is None

    success = await cache.set_search_results("query", [{"id": 1}], ttl=60)
    assert success is True

    deleted = await cache.invalidate_by_collection("default")
    assert deleted == 1
