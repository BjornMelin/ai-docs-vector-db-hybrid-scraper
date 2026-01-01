"""Tests covering selection heuristics and usage tracking."""
# pylint: too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.manager.selection import (
    RecommendationParams,
    SelectionEngine,
)
from src.services.embeddings.manager.types import QualityTier
from src.services.embeddings.manager.usage import UsageRecord, UsageTracker


@dataclass
class _SmartConfig:
    """Simple smart selection configuration for manager tests."""

    chars_per_token: int = 4
    code_keywords: list[str] = field(default_factory=lambda: ["def", "class", "import"])
    long_text_threshold: int = 1000
    short_text_threshold: int = 50
    quality_weight: float = 0.6
    speed_weight: float = 0.2
    cost_weight: float = 0.2
    quality_best_threshold: float = 0.8
    quality_balanced_threshold: float = 0.6
    cost_expensive_threshold: float = 0.1
    cost_cheap_threshold: float = 0.02
    speed_balanced_threshold: int = 120
    budget_warning_threshold: float = 0.8
    budget_critical_threshold: float = 0.95


class _StubProvider(EmbeddingProvider):
    """Provider that exposes cost metadata for selection tests."""

    def __init__(self, name: str, cost_per_token: float):
        super().__init__(model_name=name)
        self.dimensions = 3
        self._cost = cost_per_token

    async def initialize(self) -> None:  # pragma: no cover - unused
        """Initialize the provider."""
        return

    async def cleanup(self) -> None:  # pragma: no cover - unused
        """Cleanup the provider."""
        return

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for the given texts."""
        return [[1.0] * self.dimensions for _ in texts]

    @property
    def cost_per_token(self) -> float:
        """Get the cost per token for the provider."""
        return self._cost

    @property
    def max_tokens_per_request(self) -> int:
        """Get the maximum tokens per request for the provider."""
        return 8191


def test_analyze_empty_payload() -> None:
    """Empty text lists should produce zeroed analysis."""
    engine = SelectionEngine(_SmartConfig(), benchmarks={})
    analysis = engine.analyze([])
    assert analysis.text_type == "empty"
    assert analysis.requires_high_quality is False


def test_analyze_detects_code_content() -> None:
    """Presence of code keywords should classify as code and require quality."""
    engine = SelectionEngine(_SmartConfig(), benchmarks={})
    analysis = engine.analyze(["def foo():\n    pass"])
    assert analysis.text_type == "code"
    assert analysis.requires_high_quality is True


def test_analyze_long_text() -> None:
    """Very long documents should be categorised as long."""
    engine = SelectionEngine(_SmartConfig(), benchmarks={})
    analysis = engine.analyze(["word " * 400])
    assert analysis.text_type == "long"
    assert analysis.avg_length >= 2000


def test_recommend_prefers_cost_effective_models() -> None:
    """Recommendation should respect max cost constraint."""
    benchmarks = {
        "text-embedding-3-small": {
            "model_name": "text-embedding-3-small",
            "provider": "openai",
            "avg_latency_ms": 80,
            "quality_score": 90,
            "cost_per_million_tokens": 40.0,
        },
        "BAAI/bge-small-en-v1.5": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "provider": "fastembed",
            "avg_latency_ms": 20,
            "quality_score": 80,
            "cost_per_million_tokens": 0.0,
        },
    }
    engine = SelectionEngine(_SmartConfig(), benchmarks=benchmarks)
    providers: dict[str, EmbeddingProvider] = {
        # Use model names that exist in the benchmarks for candidate generation
        "openai": _StubProvider("text-embedding-3-small", cost_per_token=0.00004),
        "fastembed": _StubProvider("BAAI/bge-small-en-v1.5", cost_per_token=0.0),
    }
    analysis = engine.analyze(["short text"])
    recommendation = engine.recommend(
        providers=providers,
        text_analysis=analysis,
        params=RecommendationParams(
            quality_tier=QualityTier.FAST,
            max_cost=0.5,
            speed_priority=True,
        ),
    )

    assert recommendation["provider"] == "fastembed"
    assert recommendation["score"] > 0


def test_recommend_raises_when_no_models_satisfy_constraints() -> None:
    """Engine should raise ValueError when no candidates meet constraints."""
    benchmarks = {
        "expensive-model": {
            "model_name": "expensive-model",
            "provider": "openai",
            "avg_latency_ms": 200,
            "quality_score": 95,
            "cost_per_million_tokens": 500.0,
        },
    }
    engine = SelectionEngine(_SmartConfig(), benchmarks=benchmarks)
    providers: dict[str, EmbeddingProvider] = {
        "openai": _StubProvider("openai", cost_per_token=0.0005)
    }
    analysis = engine.analyze(["short"])
    with pytest.raises(ValueError, match="No models available"):
        engine.recommend(
            providers=providers,
            text_analysis=analysis,
            params=RecommendationParams(
                quality_tier=QualityTier.BEST,
                max_cost=0.0001,
                speed_priority=False,
            ),
        )


def test_usage_tracker_records_and_reports() -> None:
    """Usage tracker should accumulate counts across events."""
    tracker = UsageTracker(_SmartConfig())
    tracker.record(
        UsageRecord(provider="fastembed", model="m1", tokens=10, cost=0.0, tier="fast")
    )
    tracker.record(
        UsageRecord(provider="openai", model="m2", tokens=20, cost=0.01, tier="best")
    )

    report = tracker.report()
    assert report["summary"]["_total_requests"] == 2
    assert report["summary"]["_total_tokens"] == 30
    assert report["summary"]["_total_cost"] == pytest.approx(0.01)
    assert report["by_provider"]["openai"] == 1


def test_usage_tracker_budget_warnings() -> None:
    """Budget checks should emit warnings when near limits."""
    tracker = UsageTracker(_SmartConfig(), budget_limit=1.0)
    tracker.record(UsageRecord("openai", "m1", tokens=0, cost=0.8, tier="best"))

    evaluation = tracker.check_budget(0.15)
    assert evaluation["within_budget"] is True
    assert evaluation["warnings"]  # Should emit at least one warning


def test_usage_tracker_budget_exceeded() -> None:
    """Budget exceeding limit should return within_budget False."""
    tracker = UsageTracker(_SmartConfig(), budget_limit=0.5)
    tracker.record(UsageRecord("openai", "m1", tokens=0, cost=0.45, tier="best"))

    evaluation = tracker.check_budget(0.1)
    assert evaluation["within_budget"] is False
