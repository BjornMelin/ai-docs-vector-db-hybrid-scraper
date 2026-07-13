"""Tests for cache warmup functionality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from src.config.models import EmbeddingProvider
from src.services.cache.manager import CacheManager
from src.services.cache.warmup import warm_caches
from src.services.embeddings.manager import EmbeddingManager


class _StubOpenAIConfig:
    """Minimal OpenAI config exposed to warmup logic."""

    model = "configured-openai-model"


class _StubFastEmbedConfig:
    """Minimal FastEmbed config exposed to warmup logic."""

    dense_model = "configured-fastembed-model"


class _StubSettings:
    """Container exposing embedding metadata for the stub manager."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.embedding_provider = provider
        self.openai = _StubOpenAIConfig()
        self.fastembed = _StubFastEmbedConfig()


class _StubEmbeddingManager:
    """Embedding manager stub returning deterministic vectors."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        generate_limit: int | None = None,
    ) -> None:
        """Initialize the stub embedding manager."""
        self.config = _StubSettings(provider)
        self.generate_limit = generate_limit
        self.generated_vectors: list[list[float]] = []
        self.requested_texts: Sequence[str] | None = None
        self.request_kwargs: dict[str, Any] = {}
        self.initialize_calls = 0

    async def initialize(self) -> None:
        """Simulate initialization logic."""
        self.initialize_calls += 1

    def get_provider_info(self) -> dict[str, dict[str, object]]:
        """Expose the same public provider metadata as EmbeddingManager."""
        provider = self.config.embedding_provider
        model = (
            self.config.openai.model
            if provider is EmbeddingProvider.OPENAI
            else self.config.fastembed.dense_model
        )
        return {provider.value: {"model": model, "dimensions": 1}}

    async def generate_embeddings(
        self, texts: list[str], *args: Any, **kwargs: Any
    ) -> dict[str, object]:
        """Generate embeddings for the given texts."""
        self.requested_texts = list(texts)
        self.request_kwargs = kwargs
        limit = len(texts) if self.generate_limit is None else self.generate_limit
        vectors = [[float(index + 1)] for index, _ in enumerate(texts[:limit])]
        self.generated_vectors = vectors
        return {"embeddings": vectors, "cache_hit": False}


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        (EmbeddingProvider.OPENAI, "configured-openai-model"),
        (EmbeddingProvider.FASTEMBED, "configured-fastembed-model"),
    ],
)
async def test_warm_caches_generates_embeddings_for_missing_queries(
    fakeredis_cache,
    provider: EmbeddingProvider,
    model: str,
) -> None:
    """Verify that warm_caches generates embeddings for uncached queries."""
    manager = CacheManager(distributed_cache=fakeredis_cache)
    stub_manager = _StubEmbeddingManager(provider=provider)
    embedding_cache = manager.embedding_cache
    assert embedding_cache is not None

    await embedding_cache.set_embedding(
        text="alpha",
        model=model,
        embedding=[0.1],
        provider=provider.value,
        dimensions=1,
    )

    summary = await warm_caches(
        manager,
        embedding_manager=cast(EmbeddingManager, stub_manager),
        embedding_queries=["alpha", "beta"],
    )

    embeddings_report = summary["embeddings"]
    assert embeddings_report["requested"] == 2
    assert embeddings_report["already_cached"] == 1
    assert embeddings_report["generated"] == 1
    assert embeddings_report["skipped"] == 0
    assert stub_manager.initialize_calls == 1
    assert stub_manager.generated_vectors == [[1.0]]
    assert stub_manager.request_kwargs == {
        "provider_name": provider.value,
        "auto_select": False,
    }
    assert await embedding_cache.get_embedding(
        text="beta",
        model=model,
        provider=provider.value,
        dimensions=1,
    ) == [1.0]

    await manager.close()


@pytest.mark.asyncio()
async def test_warm_caches_records_skipped_when_generation_partial(
    fakeredis_cache,
) -> None:
    """If embedding generation is limited, excess queries are skipped."""
    manager = CacheManager(distributed_cache=fakeredis_cache)
    stub_manager = _StubEmbeddingManager(generate_limit=1)

    summary = await warm_caches(
        manager,
        embedding_manager=cast(EmbeddingManager, stub_manager),
        embedding_queries=["first", "second"],
    )

    embeddings_report = summary["embeddings"]
    assert embeddings_report["requested"] == 2
    assert embeddings_report["already_cached"] == 0
    assert embeddings_report["generated"] == 1
    assert embeddings_report["skipped"] == 1
    assert stub_manager.requested_texts == ["first", "second"]

    await manager.close()


@pytest.mark.asyncio()
async def test_warm_caches_skips_when_caches_disabled(fakeredis_cache) -> None:
    """If specialized caches are disabled, nothing is done."""
    manager = CacheManager(
        distributed_cache=fakeredis_cache,
        enable_specialized_caches=False,
    )
    stub_manager = _StubEmbeddingManager()
    search_call_tracker: dict[str, bool] = {"called": False}

    async def search_executor(query: str, collection: str) -> list[dict[str, Any]]:
        search_call_tracker["called"] = True
        return [{"id": f"{collection}:{query}", "content": query, "score": 1.0}]

    summary = await warm_caches(
        manager,
        embedding_manager=cast(EmbeddingManager, stub_manager),
        embedding_queries=["gamma"],
        search_queries=["delta"],
        search_executor=search_executor,
    )

    embeddings_report = summary["embeddings"]
    assert embeddings_report["requested"] == 1
    assert embeddings_report["generated"] == 0
    assert embeddings_report["skipped"] == 1
    assert stub_manager.initialize_calls == 1
    assert stub_manager.generated_vectors == []

    search_report = summary["search"]
    assert search_report["requested"] == 1
    assert search_report["warmed"] == 0
    assert search_report["skipped"] == 1
    assert search_call_tracker["called"] is False

    await manager.close()


@pytest.mark.asyncio()
async def test_warm_caches_skips_when_manager_missing(fakeredis_cache) -> None:
    """If no cache manager is provided, nothing is done."""
    manager = CacheManager(distributed_cache=fakeredis_cache)

    summary = await warm_caches(
        manager,
        embedding_manager=None,
        embedding_queries=["alpha"],
    )

    embeddings_report = summary["embeddings"]
    assert embeddings_report["requested"] == 1
    assert embeddings_report["generated"] == 0
    assert embeddings_report["skipped"] == 1

    await manager.close()


@pytest.mark.asyncio()
async def test_warm_caches_warms_search_results(fakeredis_cache) -> None:
    """Verify that warm_caches correctly warms search results in the cache."""
    manager = CacheManager(distributed_cache=fakeredis_cache)
    search_cache = manager.search_cache
    assert search_cache is not None

    existing_results = [{"id": "existing", "content": "existing", "score": 1.0}]
    await search_cache.set_search_results(
        query="existing",
        results=existing_results,
        collection_name="default",
    )

    async def execute(query: str, collection: str) -> list[dict[str, Any]]:
        return [{"id": query, "content": query, "score": 0.5, "collection": collection}]

    summary = await warm_caches(
        manager,
        embedding_manager=None,
        search_queries=["dragonfly", "existing"],
        search_executor=execute,
    )

    search_report = summary["search"]
    assert search_report["requested"] == 2
    assert search_report["warmed"] == 1
    assert search_report["skipped"] == 0

    cached = await search_cache.get_search_results(
        "dragonfly", collection_name="default"
    )
    assert cached and cached[0]["id"] == "dragonfly"

    await manager.close()
