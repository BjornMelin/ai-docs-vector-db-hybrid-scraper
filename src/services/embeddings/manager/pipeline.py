"""Embedding generation pipeline orchestrating providers, caching, and metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.services.embeddings.base import EmbeddingProvider
from src.services.errors import EmbeddingServiceError

from .providers import ProviderRegistry
from .selection import RecommendationParams, SelectionEngine, TextAnalysis
from .types import QualityTier
from .usage import UsageRecord, UsageTracker


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetricsContext:
    """Context for usage metric calculations."""

    provider: EmbeddingProvider
    provider_name: str | None
    selected_model: str
    texts: list[str]
    quality_tier: Any | None
    start_time: float


@dataclass(frozen=True)
class GenerationOptions:
    """Parameters controlling embedding generation."""

    quality_tier: QualityTier | None
    provider_name: str | None
    max_cost: float | None
    speed_priority: bool
    auto_select: bool
    generate_sparse: bool


@dataclass
class ProviderSelection:
    """Details about the chosen provider/model combination."""

    provider: EmbeddingProvider
    model: str
    estimated_cost: float
    reasoning: str


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """Protocol for providers supporting sparse embeddings."""

    async def generate_sparse_embeddings(
        self, texts: list[str]
    ) -> list[dict]:  # pragma: no cover - type hint
        """Generate sparse embeddings."""
        raise NotImplementedError


@dataclass
class PipelineContext:
    """Aggregated dependencies for the embedding pipeline."""

    config: Any
    usage: UsageTracker
    selection: SelectionEngine
    providers: ProviderRegistry
    cache_manager: Any | None
    smart_config: Any | None


class EmbeddingPipeline:
    """Coordinates embedding generation workflow."""

    def __init__(self, context: PipelineContext) -> None:
        self._config = context.config
        self._usage = context.usage
        self._selection = context.selection
        self._providers = context.providers
        self._cache_manager = context.cache_manager
        self._smart_config = context.smart_config

    def set_usage_tracker(self, usage: UsageTracker) -> None:
        """Update usage tracker reference after hot-swapping in manager."""
        self._usage = usage

    def set_selection_engine(self, selection: SelectionEngine) -> None:
        """Update selection engine reference."""
        self._selection = selection

    def set_smart_config(self, smart_config: Any | None) -> None:
        """Refresh smart selection config for budget/token estimations."""
        self._smart_config = smart_config

    async def generate(
        self,
        texts: list[str],
        options: GenerationOptions,
    ) -> dict[str, Any]:
        """Execute the embedding workflow and return enriched metadata."""
        start_time = time.time()
        text_analysis = self._selection.analyze(texts)

        cache_hit = await self._try_single_text_cache(
            texts=texts,
            text_analysis=text_analysis,
            options=options,
            start_time=start_time,
        )
        if cache_hit is not None:
            return cache_hit

        selection = self.select_provider(
            text_analysis=text_analysis,
            options=options,
        )

        budget_status = self._usage.check_budget(selection.estimated_cost)
        if not budget_status["within_budget"]:
            raise EmbeddingServiceError("Budget constraint violated")
        for warning in budget_status["warnings"]:
            logger.warning("Budget warning: %s", warning)

        embeddings = await selection.provider.generate_embeddings(texts, batch_size=32)
        sparse_embeddings = await self._generate_sparse_embeddings(
            selection.provider, texts, options.generate_sparse
        )

        metrics_context = EmbeddingMetricsContext(
            provider=selection.provider,
            provider_name=options.provider_name,
            selected_model=selection.model,
            texts=texts,
            quality_tier=options.quality_tier,
            start_time=start_time,
        )
        metrics = self._calculate_metrics(metrics_context)

        await self._cache_embedding_if_applicable(
            texts, embeddings, selection.model, metrics
        )

        return self._build_result(
            embeddings,
            sparse_embeddings,
            metrics,
            selection,
        )

    async def _try_single_text_cache(
        self,
        texts: list[str],
        text_analysis: TextAnalysis,
        options: GenerationOptions,
        start_time: float,
    ) -> dict[str, Any] | None:
        if not (self._cache_manager and len(texts) == 1):
            return None

        embedding_cache = self._get_embedding_cache()
        if embedding_cache is None:
            return None

        provider_key = options.provider_name or getattr(
            self._config.embedding_provider, "value", "fastembed"
        )
        model_name = (
            self._config.openai.model
            if provider_key == "openai"
            else self._config.fastembed.dense_model
        )
        cached_embedding = await embedding_cache.get_embedding(
            text=texts[0],
            provider=provider_key,
            model=model_name,
            dimensions=self._config.openai.dimensions,
        )
        if cached_embedding is None:
            return None

        return {
            "embeddings": [cached_embedding],
            "provider": provider_key,
            "model": model_name,
            "cost": 0.0,
            "latency_ms": (time.time() - start_time) * 1000,
            "tokens": 0,
            "reasoning": "Retrieved from cache",
            "quality_tier": (
                options.quality_tier.value if options.quality_tier else "default"
            ),
            "usage_stats": self._usage.report(),
            "cache_hit": True,
        }

    def select_provider(
        self,
        text_analysis: TextAnalysis,
        options: GenerationOptions,
    ) -> ProviderSelection:
        if options.provider_name:
            provider = self._providers.resolve(
                options.provider_name, options.quality_tier
            )
            selected_model = provider.model_name
            estimated_cost = (
                text_analysis.estimated_tokens * provider.cost_per_token
                if hasattr(provider, "cost_per_token")
                else 0.0
            )
            logger.info(
                "Manual selection: %s/%s ($%.4f)",
                options.provider_name,
                selected_model,
                estimated_cost,
            )
            return ProviderSelection(
                provider=provider,
                model=selected_model,
                estimated_cost=estimated_cost,
                reasoning="Manual provider override",
            )

        tier_for_recommendation = (
            options.quality_tier
            if options.auto_select
            else options.quality_tier or None
        )
        recommendation_params = RecommendationParams(
            quality_tier=tier_for_recommendation,
            max_cost=options.max_cost,
            speed_priority=options.speed_priority,
        )
        recommendation = self._selection.recommend(
            providers=self._providers.providers,
            text_analysis=text_analysis,
            params=recommendation_params,
        )
        recommended_provider = recommendation["provider"]
        selected_model = recommendation["model"]
        estimated_cost = recommendation["estimated_cost"]
        reasoning = (
            recommendation["reasoning"] if options.auto_select else "Default selection"
        )

        logger.info(
            "%s selection: %s/%s ($%.4f) - %s",
            "Smart" if options.auto_select else "Guided",
            recommended_provider,
            selected_model,
            estimated_cost,
            reasoning,
        )

        provider = self._providers.resolve(
            recommended_provider, tier_for_recommendation
        )
        if not selected_model:
            selected_model = provider.model_name

        return ProviderSelection(
            provider=provider,
            model=selected_model,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
        )

    def _calculate_metrics(self, context: EmbeddingMetricsContext) -> dict[str, Any]:
        end_time = time.time()
        latency_ms = (end_time - context.start_time) * 1000

        chars_per_token = getattr(self._smart_config, "chars_per_token", 4) or 4
        actual_tokens = sum(len(text) for text in context.texts) // chars_per_token
        actual_cost = actual_tokens * context.provider.cost_per_token

        tier_name = context.quality_tier.value if context.quality_tier else "default"
        provider_key = (
            context.provider_name
            or context.provider.__class__.__name__.lower().replace("provider", "")
        )

        record = UsageRecord(
            provider=provider_key,
            model=context.selected_model,
            tokens=actual_tokens,
            cost=actual_cost,
            tier=tier_name,
        )
        self._usage.record(record)

        return {
            "latency_ms": latency_ms,
            "tokens": actual_tokens,
            "cost": actual_cost,
            "tier_name": tier_name,
            "provider_key": provider_key,
        }

    async def _generate_sparse_embeddings(
        self,
        provider: EmbeddingProvider,
        texts: list[str],
        generate_sparse: bool,
    ) -> list[dict] | None:
        if not generate_sparse or not isinstance(provider, SparseEmbeddingProvider):
            return None

        sparse_embeddings = await provider.generate_sparse_embeddings(texts)
        logger.info("Generated %d sparse embeddings", len(sparse_embeddings))
        return sparse_embeddings

    async def _cache_embedding_if_applicable(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        selected_model: str,
        metrics: dict[str, Any],
    ) -> None:
        if not (self._cache_manager and len(texts) == 1 and len(embeddings) == 1):
            return

        embedding_cache = self._get_embedding_cache()
        if embedding_cache is None:
            return

        provider_key = metrics["provider_key"]
        await embedding_cache.set_embedding(
            text=texts[0],
            provider=provider_key,
            model=selected_model,
            dimensions=len(embeddings[0]),
            embedding=embeddings[0],
        )
        logger.info("Cached embedding for future use")

    def _build_result(
        self,
        embeddings: list[list[float]],
        sparse_embeddings: list[dict] | None,
        metrics: dict[str, Any],
        selection: ProviderSelection,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "embeddings": embeddings,
            "provider": metrics["provider_key"],
            "model": selection.model,
            "cost": metrics["cost"],
            "latency_ms": metrics["latency_ms"],
            "tokens": metrics["tokens"],
            "reasoning": selection.reasoning,
            "quality_tier": metrics["tier_name"],
            "usage_stats": self._usage.report(),
            "cache_hit": False,
        }
        if sparse_embeddings is not None:
            result["sparse_embeddings"] = sparse_embeddings
        return result

    def _get_embedding_cache(self) -> Any | None:
        if self._cache_manager is None:
            return None

        embedding_cache = getattr(self._cache_manager, "embedding_cache", None)
        if embedding_cache is not None:
            return embedding_cache

        has_get = hasattr(self._cache_manager, "get_embedding")
        has_set = hasattr(self._cache_manager, "set_embedding")
        if has_get and has_set:
            return self._cache_manager

        return None
