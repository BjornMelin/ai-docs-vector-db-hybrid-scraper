"""Selection utilities for choosing embedding providers and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.embeddings.base import EmbeddingProvider

from .types import QualityTier


@dataclass
class TextAnalysis:
    """Captures text statistics for provider selection."""

    total_length: int
    avg_length: int
    complexity_score: float
    estimated_tokens: int
    text_type: str
    requires_high_quality: bool


@dataclass(frozen=True)
class RecommendationParams:
    """Parameters guiding provider recommendation."""

    quality_tier: QualityTier | None
    max_cost: float | None
    speed_priority: bool


class SelectionEngine:
    """Provides text analysis and smart provider recommendations."""

    def __init__(self, smart_config: Any | None, benchmarks: dict[str, dict[str, Any]]):
        self._smart_config = smart_config or {}
        self._benchmarks = benchmarks

    def analyze(self, texts: list[str]) -> TextAnalysis:
        """Analyze input texts to determine selection heuristics."""

        if not texts:
            return TextAnalysis(
                total_length=0,
                avg_length=0,
                complexity_score=0.0,
                estimated_tokens=0,
                text_type="empty",
                requires_high_quality=False,
            )

        valid_texts = self._sanitize_texts(texts)
        if not valid_texts:
            return TextAnalysis(
                total_length=0,
                avg_length=0,
                complexity_score=0.0,
                estimated_tokens=0,
                text_type="empty",
                requires_high_quality=False,
            )

        total_length, avg_length, estimated_tokens = self._length_metrics(valid_texts)
        complexity_score, code_ratio = self._complexity_metrics(texts, valid_texts)
        text_type = self._classify_text(avg_length, code_ratio)
        requires_high_quality = self._requires_high_quality(
            complexity_score, avg_length, code_ratio
        )

        return TextAnalysis(
            total_length=total_length,
            avg_length=avg_length,
            complexity_score=complexity_score,
            estimated_tokens=estimated_tokens,
            text_type=text_type,
            requires_high_quality=requires_high_quality,
        )

    def _sanitize_texts(self, texts: list[str]) -> list[str]:
        return [text for text in texts if text is not None]

    def _length_metrics(self, valid_texts: list[str]) -> tuple[int, int, int]:
        total_length = sum(len(text) for text in valid_texts)
        avg_length = total_length // len(valid_texts)
        chars_per_token = getattr(self._smart_config, "chars_per_token", 4) or 4
        estimated_tokens = int(total_length / chars_per_token)
        return total_length, avg_length, estimated_tokens

    def _complexity_metrics(
        self, texts: list[str], valid_texts: list[str]
    ) -> tuple[float, float]:
        all_words: set[str] = set()
        total_words = 0
        code_indicators = 0
        code_keywords = getattr(self._smart_config, "code_keywords", [])

        for text in valid_texts:
            lower_text = text.lower()
            words = lower_text.split()
            all_words.update(words)
            total_words += len(words)
            if any(keyword in lower_text for keyword in code_keywords):
                code_indicators += 1

        if total_words:
            diversity = len(all_words) / total_words
            complexity_score = min(diversity * 1.5, 1.0)
        else:
            complexity_score = 0.0

        code_ratio = (code_indicators / len(texts)) if texts else 0.0
        return complexity_score, code_ratio

    def _classify_text(self, avg_length: int, code_ratio: float) -> str:
        if code_ratio > 0.3:
            return "code"
        long_threshold = getattr(self._smart_config, "long_text_threshold", 1000)
        short_threshold = getattr(self._smart_config, "short_text_threshold", 100)
        if avg_length > long_threshold:
            return "long"
        if avg_length < short_threshold:
            return "short"
        return "docs"

    def _requires_high_quality(
        self, complexity_score: float, avg_length: int, code_ratio: float
    ) -> bool:
        if code_ratio > 0.3:
            return True
        quality_best_threshold = getattr(
            self._smart_config, "quality_best_threshold", 0.8
        )
        return complexity_score > quality_best_threshold or avg_length > 1500

    def recommend(
        self,
        providers: dict[str, EmbeddingProvider],
        text_analysis: TextAnalysis,
        params: RecommendationParams,
    ) -> dict[str, Any]:
        """Recommend best provider/model combination."""

        candidates: list[dict[str, Any]] = []
        for provider_name, provider in providers.items():
            for model in self._models_for_provider(provider_name, provider):
                candidate = self._build_candidate(
                    provider_name,
                    model,
                    text_analysis,
                    params,
                )
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            msg = (
                f"No models available for constraints: max_cost={params.max_cost}, "
                f"tokens={text_analysis.estimated_tokens}"
            )
            raise ValueError(msg)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        reasoning = self._generate_selection_reasoning(
            best,
            text_analysis,
            params.quality_tier,
            params.speed_priority,
        )

        return {
            "provider": best["provider"],
            "model": best["model"],
            "estimated_cost": best["estimated_cost"],
            "score": best["score"],
            "reasoning": reasoning,
            "alternatives": candidates[1:3],
        }

    def _build_candidate(
        self,
        provider_name: str,
        model: str,
        text_analysis: TextAnalysis,
        params: RecommendationParams,
    ) -> dict[str, Any] | None:
        benchmark = self._benchmarks.get(model)
        if not benchmark:
            return None

        cost_per_million = benchmark.get("cost_per_million_tokens", 0)
        cost = text_analysis.estimated_tokens * (cost_per_million / 1_000_000)
        if params.max_cost and cost > params.max_cost:
            return None

        score = self._calculate_model_score(
            benchmark,
            text_analysis,
            params.quality_tier,
            params.speed_priority,
        )
        return {
            "provider": provider_name,
            "model": model,
            "benchmark": benchmark,
            "estimated_cost": cost,
            "score": score,
        }

    def _models_for_provider(
        self, provider_name: str, provider: EmbeddingProvider
    ) -> list[str]:
        if provider_name == "openai":
            return ["text-embedding-3-small", "text-embedding-3-large"]
        return [provider.model_name]

    def _calculate_model_score(
        self,
        benchmark: dict[str, Any],
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> float:
        quality_score = benchmark.get("quality_score", 0.0)
        score = 0.0
        score += self._quality_component(quality_score)
        score += self._speed_component(benchmark, speed_priority)
        score += self._cost_component(benchmark, speed_priority)

        if quality_tier is not None:
            cost_per_million_tokens = benchmark.get("cost_per_million_tokens", 0)
            score = self._apply_quality_tier_bonus(
                score, quality_tier, cost_per_million_tokens, quality_score
            )

        score += self._content_bonus(benchmark, text_analysis)

        return float(min(score, 100))

    def _quality_component(self, quality_score: float) -> float:
        weight = getattr(self._smart_config, "quality_weight", 1.0)
        return quality_score * weight

    def _speed_component(
        self, benchmark: dict[str, Any], speed_priority: bool
    ) -> float:
        weight = (
            0.5 if speed_priority else getattr(self._smart_config, "speed_weight", 0.3)
        )
        avg_latency_ms = benchmark.get("avg_latency_ms", 100)
        threshold = getattr(self._smart_config, "speed_balanced_threshold", 200)
        speed_score = max(0, (threshold - avg_latency_ms) / threshold * 100)
        return speed_score * weight

    def _cost_component(self, benchmark: dict[str, Any], speed_priority: bool) -> float:
        weight = (
            0.1 if speed_priority else getattr(self._smart_config, "cost_weight", 0.2)
        )
        cost_per_million = benchmark.get("cost_per_million_tokens", 0)
        if cost_per_million == 0:
            cost_score = 100.0
        else:
            threshold = getattr(self._smart_config, "cost_expensive_threshold", 0.1)
            cost_score = max(0, (threshold - cost_per_million) / threshold * 100)
        return cost_score * weight

    def _content_bonus(
        self, benchmark: dict[str, Any], text_analysis: TextAnalysis
    ) -> float:
        quality_score = benchmark.get("quality_score", 0.0)
        best_threshold = getattr(self._smart_config, "quality_best_threshold", 0.8)
        if (text_analysis.text_type == "code" and quality_score > best_threshold) or (
            text_analysis.text_type == "short"
            and benchmark.get("avg_latency_ms", 100) < 60
        ):
            return 5.0
        return 0.0

    def _apply_quality_tier_bonus(
        self,
        score: float,
        quality_tier: QualityTier,
        cost_per_million_tokens: float,
        quality_score: float,
    ) -> float:
        bonus = 0.0
        if quality_tier == QualityTier.FAST and cost_per_million_tokens == 0:
            bonus = 25.0
        elif quality_tier == QualityTier.BEST:
            if quality_score > getattr(
                self._smart_config, "quality_best_threshold", 0.8
            ):
                bonus = 40.0
            elif quality_score > getattr(
                self._smart_config, "quality_balanced_threshold", 0.6
            ):
                bonus = 30.0
            else:
                bonus = -10.0
        elif quality_tier == QualityTier.BALANCED:
            if cost_per_million_tokens == 0:
                bonus = 10.0
            elif cost_per_million_tokens < getattr(
                self._smart_config, "cost_cheap_threshold", 0.05
            ):
                bonus = 15.0
        return score + bonus

    def _generate_selection_reasoning(
        self,
        selection: dict[str, Any],
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> str:
        provider = selection["provider"]
        model = selection["model"]
        score = selection["score"]

        reasons = [
            f"Selected {provider}/{model} with score {score:.1f}",
            f" - Text type: {text_analysis.text_type}",
            f" - Estimated tokens: {text_analysis.estimated_tokens}",
        ]

        if quality_tier == QualityTier.BEST:
            reasons.append(" - High quality embeddings preferred")
        elif quality_tier == QualityTier.FAST:
            reasons.append(" - Fast processing prioritized")

        if speed_priority:
            reasons.append(" - Speed prioritized over quality")
        else:
            reasons.append(" - Balanced quality and speed")

        if text_analysis.text_type == "code":
            reasons.append(" - Code content detected, optimizing for quality")
        elif text_analysis.text_type == "short":
            reasons.append(" - Short text detected, optimizing for speed")

        return "\n".join(reasons)
