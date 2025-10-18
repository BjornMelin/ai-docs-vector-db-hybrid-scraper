"""Integration tests for EmbeddingManager using lightweight fakes."""
# pylint: too-many-instance-attributes

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.manager import EmbeddingManager, QualityTier, TextAnalysis
from src.services.errors import EmbeddingServiceError


@dataclass
class _SmartSelection:
    """Simple smart selection configuration for manager tests."""

    chars_per_token: int = 4
    code_keywords: list[str] = field(default_factory=lambda: ["def", "class"])
    long_text_threshold: int = 1000
    short_text_threshold: int = 50
    quality_weight: float = 0.4
    speed_weight: float = 0.3
    cost_weight: float = 0.3
    speed_balanced_threshold: int = 100
    cost_expensive_threshold: float = 0.1
    cost_cheap_threshold: float = 0.02
    quality_best_threshold: float = 0.85
    quality_balanced_threshold: float = 0.65
    budget_warning_threshold: float = 0.8
    budget_critical_threshold: float = 0.95


@dataclass
class _EmbeddingConfig:
    """Embedding configuration stub."""

    smart_selection: _SmartSelection = field(default_factory=_SmartSelection)
    model_benchmarks: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "local-model": {
                "model_name": "local-model",
                "provider": "fastembed",
                "avg_latency_ms": 20,
                "quality_score": 75,
                "cost_per_million_tokens": 0.0,
            },
            "openai-model": {
                "model_name": "openai-model",
                "provider": "openai",
                "avg_latency_ms": 80,
                "quality_score": 92,
                "cost_per_million_tokens": 40.0,
            },
        }
    )


@dataclass
class _CacheConfig:
    """Cache configuration stub."""

    enable_caching: bool = False


@dataclass
class _OpenAIConfig:
    """OpenAI configuration stub."""

    api_key: str = "sk-test"
    model: str = "openai-model"
    dimensions: int = 1536


@dataclass
class _FastEmbedConfig:
    """FastEmbed configuration stub."""

    dense_model: str = "local-model"
    generate_sparse: bool = False


@dataclass
class _EmbeddingProviderChoice:
    """Embedding provider choice stub."""

    value: str = "fastembed"


@dataclass
class _SettingsStub:
    """Configuration stub compatible with EmbeddingManager requirements."""

    cache: _CacheConfig = field(default_factory=_CacheConfig)
    openai: _OpenAIConfig = field(default_factory=_OpenAIConfig)
    fastembed: _FastEmbedConfig = field(default_factory=_FastEmbedConfig)
    embedding: _EmbeddingConfig = field(default_factory=_EmbeddingConfig)
    embedding_provider: _EmbeddingProviderChoice = field(
        default_factory=_EmbeddingProviderChoice
    )


class _StubProvider(EmbeddingProvider):
    """Common behaviour for fake embedding providers."""

    def __init__(self, model_name: str, cost_per_token: float):
        """Initialize the stub provider."""
        super().__init__(model_name)
        self.dimensions = 3
        self._cost = cost_per_token
        self.init_calls = 0
        self.cleanup_calls = 0
        self.generate_calls: list[list[str]] = []
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the stub provider."""
        self.initialized = True
        self.init_calls += 1

    async def cleanup(self) -> None:
        """Cleanup the stub provider."""
        self.initialized = False
        self.cleanup_calls += 1

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate the embeddings."""
        self.generate_calls.append(texts)
        return [[float(len(text)), 0.0, -0.0] for text in texts]

    @property
    def cost_per_token(self) -> float:
        """Get the cost per token."""
        return self._cost

    @property
    def max_tokens_per_request(self) -> int:
        """Get the maximum tokens per request."""
        return 8191


class _StubOpenAIProvider(_StubProvider):
    """Stub OpenAI provider."""

    def __init__(self, api_key: str, model_name: str, dimensions: int) -> None:
        """Initialize the stub OpenAI provider."""
        super().__init__(model_name, cost_per_token=0.00004)
        self.api_key = api_key
        self.dimensions = dimensions


class _StubFastEmbedProvider(_StubProvider):
    """Stub FastEmbed provider."""

    def __init__(self, model_name: str) -> None:
        """Initialize the stub FastEmbed provider."""
        super().__init__(model_name, cost_per_token=0.0)


@pytest.fixture
def manager_config() -> _SettingsStub:
    """Return configuration stub for manager tests."""
    return _SettingsStub()


@pytest.fixture(autouse=True)
def stub_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace real providers with lightweight stubs."""
    monkeypatch.setattr(
        "src.services.embeddings.fastembed_provider.FastEmbedProvider",
        _StubFastEmbedProvider,
    )
    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.OpenAIEmbeddingProvider",
        _StubOpenAIProvider,
    )
    monkeypatch.setattr(
        "src.services.embeddings.manager.FastEmbedProvider",
        _StubFastEmbedProvider,
    )
    monkeypatch.setattr(
        "src.services.embeddings.manager.OpenAIEmbeddingProvider",
        _StubOpenAIProvider,
    )


@pytest.mark.asyncio
async def test_initialize_registers_providers(manager_config: _SettingsStub) -> None:
    """Manager should initialize available providers based on config."""
    manager = EmbeddingManager(config=cast(Any, manager_config))

    await manager.initialize()

    providers = manager._provider_registry.providers
    assert "fastembed" in providers
    fastembed_provider = cast(_StubFastEmbedProvider, providers["fastembed"])
    assert fastembed_provider.initialized is True
    assert manager._initialized is True


@pytest.mark.asyncio
async def test_generate_embeddings_delegates_to_pipeline(
    manager_config: _SettingsStub,
) -> None:
    """Manager.generate_embeddings should delegate to pipeline."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    await manager.initialize()

    async def _fake_generate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "embeddings": [[1.0, 0.0, 0.0]],
            "provider": "fastembed",
            "model": "local-model",
            "cost": 0.0,
            "latency_ms": 5.0,
            "tokens": 4,
            "reasoning": "stubbed",
            "quality_tier": "fast",
            "usage_stats": {},
            "cache_hit": False,
        }

    manager._pipeline.generate = _fake_generate  # type: ignore[assignment]
    result = await manager.generate_embeddings(["hello world"], auto_select=True)

    assert result["provider"] == "fastembed"
    embeddings = cast(list[list[float]], result["embeddings"])
    assert embeddings[0] == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_generate_embeddings_empty_input(manager_config: _SettingsStub) -> None:
    """Empty input should short-circuit and return metadata."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    await manager.initialize()

    result = await manager.generate_embeddings([])

    assert result["embeddings"] == []
    assert result["provider"] is None
    assert result["cost"] == 0.0


def test_estimate_cost_reports_per_provider(manager_config: _SettingsStub) -> None:
    """Cost estimation should return breakdown per provider."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    manager._initialized = True  # pylint: disable=protected-access
    manager._provider_registry.providers = cast(
        dict[str, EmbeddingProvider],
        {
            "fastembed": _StubFastEmbedProvider("local-model"),
            "openai": _StubOpenAIProvider("sk", "openai-model", 1536),
        },
    )

    report = manager.estimate_cost(["sample text"])

    assert "fastembed" in report
    assert report["fastembed"]["total_cost"] == 0.0
    assert report["openai"]["total_cost"] > 0.0


def test_get_provider_info(manager_config: _SettingsStub) -> None:
    """Provider info should reflect registered providers."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    manager._initialized = True  # pylint: disable=protected-access
    manager._provider_registry.providers = cast(
        dict[str, EmbeddingProvider],
        {"fastembed": _StubFastEmbedProvider("local-model")},
    )

    info = manager.get_provider_info()
    assert info["fastembed"]["model"] == "local-model"
    assert info["fastembed"]["dimensions"] == 3


@pytest.mark.asyncio
async def test_optimal_provider_selection(manager_config: _SettingsStub) -> None:
    """Heuristic optimal provider should consider budget and quality flags."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    await manager.initialize()
    manager._provider_registry.providers = cast(
        dict[str, EmbeddingProvider],
        {
            "fastembed": _StubFastEmbedProvider("local-model"),
            "openai": _StubOpenAIProvider("sk", "openai-model", 1536),
        },
    )

    best = await manager.get_optimal_provider(text_length=10000, quality_required=True)
    assert best == "openai"

    cheap = await manager.get_optimal_provider(text_length=100, budget_limit=0.001)
    assert cheap == "fastembed"


def test_usage_stats_update_and_report(manager_config: _SettingsStub) -> None:
    """Usage stats should accumulate requests by provider and tier."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    manager.update_usage_stats(
        "fastembed", "local-model", tokens=10, cost=0.0, tier="fast"
    )
    manager.update_usage_stats(
        "openai", "openai-model", tokens=20, cost=0.01, tier="best"
    )

    report = manager.get_usage_report()
    assert report["summary"]["_total_requests"] == 2
    assert report["by_provider"]["openai"] == 1


def test_budget_constraint_helpers(manager_config: _SettingsStub) -> None:
    """Budget evaluation should reuse usage tracker rules."""
    manager = EmbeddingManager(config=cast(Any, manager_config), budget_limit=1.0)
    manager.update_usage_stats(
        "openai", "openai-model", tokens=0, cost=0.8, tier="best"
    )

    review = manager.check_budget_constraints(0.3)
    assert review["within_budget"] is False or review["warnings"]


@pytest.mark.asyncio
async def test_get_smart_provider_recommendation(manager_config: _SettingsStub) -> None:
    """Smart recommendation should return provider metadata using benchmarks."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    await manager.initialize()
    manager._provider_registry.providers = cast(
        dict[str, EmbeddingProvider],
        {
            "fastembed": _StubFastEmbedProvider("local-model"),
            "openai": _StubOpenAIProvider("sk", "openai-model", 1536),
        },
    )
    analysis = TextAnalysis(
        total_length=200,
        avg_length=200,
        complexity_score=0.5,
        estimated_tokens=50,
        text_type="docs",
        requires_high_quality=False,
    )

    recommendation = manager.get_smart_provider_recommendation(
        text_analysis=analysis,
        quality_tier=QualityTier.BEST,
        max_cost=5.0,
        speed_priority=False,
    )

    assert recommendation["provider"] in {"fastembed", "openai"}
    assert "reasoning" in recommendation


def test_load_custom_benchmarks_accepts_minimal_empty_models(
    manager_config: _SettingsStub, tmp_path: Path
) -> None:
    """Loading a config with empty model_benchmarks results in zero candidates."""
    manager = EmbeddingManager(config=cast(Any, manager_config))
    file_path = tmp_path / "empty_models.json"
    file_path.write_text(
        json.dumps({"embedding": {"model_benchmarks": {}}}), encoding="utf-8"
    )

    manager.load_custom_benchmarks(file_path)

    # Benchmarks replaced with the empty mapping from file
    assert manager._benchmarks == {}  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_generate_embeddings_requires_initialization(
    manager_config: _SettingsStub,
) -> None:
    """Manager should guard against usage before initialization."""
    manager = EmbeddingManager(config=cast(Any, manager_config))

    with pytest.raises(EmbeddingServiceError, match="Manager not initialized"):
        await manager.generate_embeddings(["sample"])
