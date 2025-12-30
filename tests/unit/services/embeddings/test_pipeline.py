"""Tests for the embedding pipeline orchestration logic."""
# pylint: too-many-arguments

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.manager.pipeline import (
    EmbeddingPipeline,
    GenerationOptions,
    PipelineContext,
)
from src.services.embeddings.manager.selection import TextAnalysis
from src.services.embeddings.manager.types import QualityTier
from src.services.embeddings.manager.usage import UsageTracker
from src.services.errors import EmbeddingServiceError


@dataclass
class _FakeConfig:
    """Minimal configuration stub satisfying pipeline requirements."""

    class OpenAIConfig:
        """OpenAI configuration stub."""

        model: str = "text-embedding-3-small"
        dimensions: int = 1536

    class FastEmbedConfig:
        """FastEmbed configuration stub."""

        dense_model: str = "BAAI/bge-small-en-v1.5"

    class ProviderChoice:
        """Provider choice stub."""

        value: str = "fastembed"

    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    fastembed: FastEmbedConfig = field(default_factory=FastEmbedConfig)
    embedding_provider: ProviderChoice = field(default_factory=ProviderChoice)


class _FakeCache:
    """Simple in-memory cache with optional capacity for eviction tests."""

    def __init__(self, capacity: int = 2) -> None:
        """Initialize the fake cache."""
        self.capacity = capacity
        self.entries: dict[tuple[str, str, str, int], list[float]] = {}
        self.embedding_cache = self  # mimic cache manager attribute

    async def get_embedding(
        self, *, text: str, provider: str, model: str, dimensions: int
    ) -> list[float] | None:
        """Get the embedding from the cache."""
        return self.entries.get((text, provider, model, dimensions))

    async def set_embedding(
        self,
        *,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
        embedding: list[float],
    ) -> None:
        """Set the embedding in the cache."""
        if len(self.entries) >= self.capacity:
            self.entries.pop(next(iter(self.entries)))
        self.entries[(text, provider, model, dimensions)] = embedding


class _FakeSelectionEngine:
    """Deterministic selection engine stub."""

    def __init__(self, analysis: TextAnalysis, recommendation: dict[str, Any]):
        """Initialize the fake selection engine."""
        self._analysis = analysis
        self._recommendation = recommendation
        self.analyze_calls: list[list[str]] = []
        self.recommend_calls: list[
            tuple[dict[str, EmbeddingProvider], dict[str, Any]]
        ] = []

    def analyze(self, texts: list[str]) -> TextAnalysis:
        """Analyze the texts."""
        self.analyze_calls.append(texts)
        return self._analysis

    def recommend(
        self,
        providers: dict[str, EmbeddingProvider],
        text_analysis: TextAnalysis,
        params: Any,
    ) -> dict[str, Any]:
        """Recommend the provider."""
        self.recommend_calls.append(
            (providers, {"params": params, "analysis": text_analysis})
        )
        return self._recommendation

    def set_recommendation(self, recommendation: dict[str, Any]) -> None:
        """Set the recommendation."""
        self._recommendation = recommendation


class _FakeProviderRegistry:
    """Provider registry stub exposing minimal resolve behaviour."""

    def __init__(self, providers: dict[str, EmbeddingProvider]):
        """Initialize the fake provider registry."""
        self.providers = providers

    def resolve(
        self,
        provider_name: str | None,
        quality_tier: QualityTier | None,
    ) -> EmbeddingProvider:
        """Resolve the provider."""
        if provider_name is not None:
            return self.providers[provider_name]
        return self.providers[next(iter(self.providers))]


class _TestProvider(EmbeddingProvider):
    """Concrete provider emitting deterministic vectors."""

    def __init__(
        self,
        name: str,
        *,
        cost_per_token: float,
        supports_sparse: bool = False,
    ) -> None:
        """Initialize the test provider."""
        super().__init__(model_name=f"{name}-model")
        self.dimensions = 3
        self._normalize_embeddings = False
        self._cost = cost_per_token
        self._supports_sparse = supports_sparse
        self.calls: list[tuple[list[str], int | None]] = []
        self.raise_error: Exception | None = None

    async def initialize(self) -> None:  # pragma: no cover - unused
        """Initialize the test provider."""
        return

    async def cleanup(self) -> None:  # pragma: no cover - unused
        """Cleanup the test provider."""
        return

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate the embeddings."""
        self.calls.append((texts, batch_size))
        if self.raise_error is not None:
            raise self.raise_error
        return [[float(len(text)), 0.0, -0.0] for text in texts]

    async def generate_sparse_embeddings(
        self, texts: list[str]
    ) -> list[dict[str, list[float]]]:
        """Generate the sparse embeddings."""
        if not self._supports_sparse:
            raise AssertionError("Sparse embeddings not supported")
        return [{"indices": [0, 1], "values": [len(text), 1.0]} for text in texts]

    @property
    def cost_per_token(self) -> float:
        """Get the cost per token."""
        return self._cost

    @property
    def max_tokens_per_request(self) -> int:
        """Get the maximum tokens per request."""
        return 8191


def _build_pipeline(
    *,
    usage_tracker: UsageTracker | None = None,
    cache_manager: Any | None = None,
    selection_engine: _FakeSelectionEngine | None = None,
    providers: dict[str, EmbeddingProvider] | None = None,
) -> EmbeddingPipeline:
    """Build the pipeline."""
    config = _FakeConfig()
    usage = usage_tracker or UsageTracker()
    selection = selection_engine or _FakeSelectionEngine(
        analysis=TextAnalysis(
            total_length=10,
            avg_length=5,
            complexity_score=0.2,
            estimated_tokens=2,
            text_type="short",
            requires_high_quality=False,
        ),
        recommendation={
            "provider": "fastembed",
            "model": "fastembed-model",
            "estimated_cost": 0.0,
            "score": 80.0,
            "reasoning": "default",
        },
    )
    provider_instances: dict[str, EmbeddingProvider] = providers or {
        "fastembed": _TestProvider("fastembed", cost_per_token=0.0),
        "openai": _TestProvider("openai", cost_per_token=0.0002),
    }
    registry = _FakeProviderRegistry(provider_instances)
    context = PipelineContext(
        config=config,
        usage=usage,
        selection=cast(Any, selection),
        providers=cast(Any, registry),
        cache_manager=cache_manager,
        smart_config=None,
    )
    return EmbeddingPipeline(context)


def _default_options(**overrides: Any) -> GenerationOptions:
    """Get the default options."""
    return GenerationOptions(
        quality_tier=overrides.get("quality_tier"),
        provider_name=overrides.get("provider_name"),
        max_cost=overrides.get("max_cost"),
        speed_priority=overrides.get("speed_priority", False),
        auto_select=overrides.get("auto_select", True),
        generate_sparse=overrides.get("generate_sparse", False),
    )


@pytest.mark.asyncio
async def test_generate_returns_cached_embedding() -> None:
    """Single-text cache hits should bypass provider execution."""
    cache = _FakeCache()
    embedding = [1.0, 2.0, 3.0]
    cache.entries[("hello", "fastembed", "fastembed-model", 3)] = embedding

    pipeline = _build_pipeline(cache_manager=cache)
    result = await pipeline.generate(["hello"], _default_options())

    assert result["cache_hit"] is True
    assert result["embeddings"] == [embedding]
    assert result["cost"] == 0.0


@pytest.mark.asyncio
async def test_generate_caches_when_miss_and_updates_usage() -> None:
    """Cache misses should store the new embedding and update usage stats."""
    cache = _FakeCache(capacity=1)
    usage = UsageTracker()
    pipeline = _build_pipeline(cache_manager=cache, usage_tracker=usage)
    provider = cast(_TestProvider, pipeline._providers.providers["fastembed"])  # type: ignore[attr-defined]

    result = await pipeline.generate(["alpha"], _default_options())

    assert result["cache_hit"] is False
    assert provider.calls == [(["alpha"], 32)]
    assert cache.entries  # New embedding stored
    stats = usage.report()
    assert stats["summary"]["_total_requests"] == 1


@pytest.mark.asyncio
async def test_generate_uses_direct_cache_api_when_no_embedding_cache_attr() -> None:
    """Cache manager exposing get/set methods directly should be supported."""

    class _DirectCache:
        """Direct cache stub."""

        def __init__(self) -> None:
            """Initialize the direct cache."""
            self._store: dict[tuple[str, str, int], list[float]] = {}

        async def get_embedding(
            self, *, text: str, provider: str, model: str, dimensions: int
        ):
            """Get the embedding from the direct cache."""
            return self._store.get((text, provider, dimensions))

        async def set_embedding(
            self,
            *,
            text: str,
            provider: str,
            model: str,
            dimensions: int,
            embedding: list[float],
        ):
            """Set the embedding in the direct cache."""
            self._store[(text, provider, dimensions)] = embedding

    cache = _DirectCache()
    pipeline = _build_pipeline(cache_manager=cache)
    # Align expected dimension with provider output (3)
    pipeline._config.openai.dimensions = 3  # type: ignore[attr-defined]  # pylint: disable=protected-access
    # First call caches
    result1 = await pipeline.generate(
        ["alpha"], _default_options(provider_name="fastembed")
    )
    assert result1["cache_hit"] is False
    # Second call should be a cache hit
    result2 = await pipeline.generate(
        ["alpha"], _default_options(provider_name="fastembed")
    )
    assert result2["cache_hit"] is True


@pytest.mark.asyncio
async def test_get_embedding_cache_returns_none_when_manager_incompatible() -> None:
    """Managers without the expected API should be ignored gracefully."""

    class _BadCache:  # missing expected methods
        """Bad cache stub."""

    pipeline = _build_pipeline(cache_manager=_BadCache())
    # With incompatible cache, generate proceeds without cache hit
    result = await pipeline.generate(["alpha"], _default_options())
    assert result["cache_hit"] is False


@pytest.mark.asyncio
async def test_parallel_safety_with_shared_cache() -> None:
    """Concurrent generate calls should be safe and share the cache."""
    cache = _FakeCache(capacity=10)
    usage = UsageTracker()
    pipeline = _build_pipeline(cache_manager=cache, usage_tracker=usage)
    pipeline._config.openai.dimensions = 3  # type: ignore[attr-defined]  # pylint: disable=protected-access

    async def _run_once(text: str) -> dict[str, Any]:
        """Run the pipeline once."""
        return await pipeline.generate(
            [text], _default_options(provider_name="fastembed")
        )

    # First wave: all misses
    texts = [f"t{i}" for i in range(5)]
    results1 = await _run_parallel([_run_once(t) for t in texts])
    assert all(r["cache_hit"] is False for r in results1)

    # Second wave: all hits
    results2 = await _run_parallel([_run_once(t) for t in texts])
    assert all(r["cache_hit"] is True for r in results2)


async def _run_parallel(coros: list[Any]) -> list[Any]:
    """Run the coroutines in parallel."""
    import asyncio

    return await asyncio.gather(*coros)


@pytest.mark.asyncio
async def test_generate_respects_budget_constraints() -> None:
    """Estimated cost exceeding budget should raise EmbeddingServiceError."""
    usage = UsageTracker(budget_limit=0.001)
    expensive_provider = _TestProvider("openai", cost_per_token=0.1)
    pipeline = _build_pipeline(
        usage_tracker=usage,
        providers={
            "openai": expensive_provider,
        },
        selection_engine=_FakeSelectionEngine(
            analysis=TextAnalysis(
                total_length=40,
                avg_length=20,
                complexity_score=0.5,
                estimated_tokens=10,
                text_type="docs",
                requires_high_quality=False,
            ),
            recommendation={
                "provider": "openai",
                "model": "openai-model",
                "estimated_cost": 1.0,
                "score": 90.0,
                "reasoning": "expensive",
            },
        ),
    )

    with pytest.raises(EmbeddingServiceError, match="Budget constraint violated"):
        await pipeline.generate(["alpha"], _default_options())


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_when_supported() -> None:
    """Providers implementing sparse embeddings should be invoked when requested."""
    sparse_provider = _TestProvider(
        "fastembed",
        cost_per_token=0.0,
        supports_sparse=True,
    )
    pipeline = _build_pipeline(
        providers={"fastembed": sparse_provider},
        selection_engine=_FakeSelectionEngine(
            analysis=TextAnalysis(
                total_length=20,
                avg_length=20,
                complexity_score=0.5,
                estimated_tokens=5,
                text_type="docs",
                requires_high_quality=False,
            ),
            recommendation={
                "provider": "fastembed",
                "model": "fastembed-model",
                "estimated_cost": 0.0,
                "score": 80.0,
                "reasoning": "supports sparse",
            },
        ),
    )

    result = await pipeline.generate(
        ["alpha"], _default_options(generate_sparse=True, auto_select=True)
    )
    assert "sparse_embeddings" in result
    assert result["sparse_embeddings"][0]["indices"] == [0, 1]


@pytest.mark.asyncio
async def test_select_provider_manual_override() -> None:
    """Explicit provider name should bypass smart selection logic."""
    pipeline = _build_pipeline()
    provider = pipeline._providers.providers["openai"]  # type: ignore[attr-defined]
    text_analysis = TextAnalysis(
        total_length=20,
        avg_length=20,
        complexity_score=0.1,
        estimated_tokens=5,
        text_type="docs",
        requires_high_quality=False,
    )
    selection = pipeline.select_provider(
        text_analysis,
        _default_options(provider_name="openai", auto_select=False),
    )

    assert selection.provider is provider
    assert selection.reasoning == "Manual provider override"


@pytest.mark.asyncio
async def test_cache_eviction_occurs_on_capacity() -> None:
    """Cache should evict oldest entry when capacity exceeded."""
    cache = _FakeCache(capacity=1)
    pipeline = _build_pipeline(cache_manager=cache)

    await pipeline.generate(["first"], _default_options())
    await pipeline.generate(["second"], _default_options())

    keys = list(cache.entries.keys())
    assert len(keys) == 1
    assert keys[0][0] == "second"
