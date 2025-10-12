from __future__ import annotations

from typing import Any

import pytest

from src.config.models import EmbeddingModel, EmbeddingProvider
from src.services.cache.manager import CacheManager
from src.services.cache.warmup import warm_caches


class _StubEmbeddingConfig:
    """Minimal embedding config used for cache warmup tests."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    dense_model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL


class _StubSettings:
    """Container exposing the embedding config required by the stub manager."""

    embedding = _StubEmbeddingConfig()


class _StubEmbeddingManager:
    """Embedding manager stub warming caches by returning fixed vectors."""

    def __init__(self) -> None:
        self.config = _StubSettings()
        self._initialized = False
        self.generated: list[list[float]] = []

    async def initialize(self) -> None:
        self._initialized = True

    async def generate_embeddings(
        self, texts: list[str], **_: object
    ) -> dict[str, object]:
        self.generated = [[float(index)] for index, _ in enumerate(texts, start=1)]
        return {"embeddings": self.generated, "cache_hit": False}


@pytest.mark.asyncio()
async def test_warm_caches_generates_missing_embeddings(fakeredis_cache) -> None:
    """warm_caches should generate embeddings for uncached queries."""

    manager = CacheManager(distributed_cache=fakeredis_cache)
    stub_embedding_manager = _StubEmbeddingManager()

    summary = await warm_caches(
        manager,
        embedding_manager=stub_embedding_manager,
        embedding_queries=["alpha", "beta"],
    )

    assert summary["embeddings"]["requested"] == 2
    assert summary["embeddings"]["generated"] == 2
    assert stub_embedding_manager.generated


@pytest.mark.asyncio()
async def test_warm_caches_skips_when_manager_missing() -> None:
    """warm_caches should report skipped embeddings when manager is absent."""

    manager = CacheManager(enable_distributed_cache=False)
    summary = await warm_caches(
        manager,
        embedding_manager=None,
        embedding_queries=["alpha"],
    )

    assert summary["embeddings"]["skipped"] == 1


@pytest.mark.asyncio()
async def test_warm_caches_warms_search_with_executor(
    fakeredis_cache,
) -> None:
    """warm_caches should warm search cache entries when executor is provided."""

    manager = CacheManager(distributed_cache=fakeredis_cache)

    async def execute(query: str, collection: str) -> list[dict[str, Any]]:
        del collection
        return [{"id": query, "content": query, "score": 0.5}]

    summary = await warm_caches(
        manager,
        embedding_manager=None,
        search_queries=["dragonfly"],
        search_executor=execute,
    )

    assert summary["search"]["warmed"] == 1
