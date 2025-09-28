"""Embedding manager for multiple providers."""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any


try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

try:
    from src.services.cache import CacheManager
except ImportError:
    CacheManager = None

from src.config import Config
from src.services.errors import EmbeddingServiceError

from .base import EmbeddingProvider
from .fastembed_provider import FastEmbedProvider
from .openai_provider import OpenAIEmbeddingProvider


if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Quality tiers for provider selection."""

    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


@dataclass
class UsageStats:
    """Usage statistics for requests."""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_by_tier: dict[str, int] = None
    requests_by_provider: dict[str, int] = None
    daily_cost: float = 0.0
    last_reset_date: str = ""

    def __post_init__(self):
        if self.requests_by_tier is None:
            self.requests_by_tier = defaultdict(int)
        if self.requests_by_provider is None:
            self.requests_by_provider = defaultdict(int)


@dataclass
class TextAnalysis:
    """Text characteristics for selection."""

    total_length: int
    avg_length: int
    complexity_score: float  # 0-1 from vocabulary diversity
    estimated_tokens: int
    text_type: str  # "code", "docs", "short", "long"
    requires_high_quality: bool


@dataclass
class EmbeddingMetricsContext:
    """Context for metrics and stats update."""

    provider: EmbeddingProvider
    provider_name: str | None
    selected_model: str
    texts: list[str]
    quality_tier: QualityTier | None
    start_time: float


class EmbeddingManager:
    """Manages embedding providers, selection, generation, and tracking."""

    def __init__(
        self,
        config: Config,
        client_manager: "ClientManager",
        budget_limit: float | None = None,
        rate_limiter: object = None,
    ):
        """Initializes manager.

        Args:
            config: Configuration with provider settings.
            client_manager: Client manager for HTTP.
            budget_limit: Daily budget in USD.
            rate_limiter: Rate limiter for APIs.
        """
        self.config = config
        self.providers: dict[str, EmbeddingProvider] = {}
        self._initialized = False
        self.budget_limit = budget_limit
        self.usage_stats = UsageStats()
        self.rate_limiter = rate_limiter
        self._client_manager = client_manager

        # Cache initialization
        self.cache_manager: object | None = None
        if config.cache.enable_caching and CacheManager is not None:
            self.cache_manager = CacheManager(
                dragonfly_url=config.cache.dragonfly_url,
                enable_local_cache=config.cache.enable_local_cache,
                enable_distributed_cache=config.cache.enable_dragonfly_cache,
                local_max_size=config.cache.local_max_size,
                local_max_memory_mb=config.cache.local_max_memory_mb,
                distributed_ttl_seconds=config.cache.cache_ttl_seconds,
            )

        # Benchmarks and selection
        self._benchmarks: dict[str, dict[str, Any]] = getattr(
            config.embedding, "model_benchmarks", {}
        )
        self._smart_config = config.embedding.smart_selection

        # Tier mapping
        self._tier_providers = {
            QualityTier.FAST: "fastembed",
            QualityTier.BALANCED: "fastembed",
            QualityTier.BEST: "openai",
        }

        # Reranker
        self._reranker = None
        self._reranker_model = "BAAI/bge-reranker-v2-m3"
        if FlagReranker is not None:
            try:
                self._reranker = FlagReranker(self._reranker_model, use_fp16=True)
                logger.info("Reranker loaded: %s", self._reranker_model)
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Reranker load failed: %s", e)

    async def initialize(self) -> None:
        """Initializes providers.

        Raises:
            EmbeddingServiceError: No providers available.
        """
        if self._initialized:
            return

        # OpenAI
        if self.config.openai.api_key:
            try:
                provider = OpenAIEmbeddingProvider(
                    api_key=self.config.openai.api_key,
                    model_name=self.config.openai.model,
                    dimensions=self.config.openai.dimensions,
                    rate_limiter=self.rate_limiter,
                    client_manager=self._client_manager,
                )
                await provider.initialize()
                self.providers["openai"] = provider
                logger.info("OpenAI loaded: %s", self.config.openai.model)
            except (ImportError, ValueError, ConnectionError, RuntimeError) as e:
                logger.warning("OpenAI load failed: %s", e)

        # FastEmbed
        try:
            provider = FastEmbedProvider(model_name=self.config.fastembed.model)
            await provider.initialize()
            self.providers["fastembed"] = provider
            logger.info("FastEmbed loaded: %s", self.config.fastembed.model)
        except (ImportError, ValueError, RuntimeError, OSError) as e:
            logger.warning("FastEmbed load failed: %s", e)

        if not self.providers:
            raise EmbeddingServiceError("No providers. Check config.")

        self._initialized = True
        logger.info("Initialized with %d providers", len(self.providers))

    async def cleanup(self) -> None:
        """Cleans up resources."""
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info("Cleaned %s", name)
            except (RuntimeError, ConnectionError, TimeoutError) as e:
                logger.exception("Cleanup failed for %s: %s", name, e)

        self.providers.clear()
        self._initialized = False

        if self.cache_manager:
            await self.cache_manager.close()
            logger.info("Cache closed")

    def _select_provider_and_model(
        self,
        text_analysis: "TextAnalysis",
        quality_tier: QualityTier | None,
        provider_name: str | None,
        max_cost: float | None,
        speed_priority: bool,
        auto_select: bool,
    ) -> tuple[EmbeddingProvider, str, float, str]:
        """Selects provider and model.

        Args:
            text_analysis: Analysis results.
            quality_tier: Tier preference.
            provider_name: Specific provider.
            max_cost: Cost limit.
            speed_priority: Speed preference.
            auto_select: Use recommendation.

        Returns:
            Provider, model, cost, reasoning.
        """
        if auto_select and not provider_name:
            rec = self.get_smart_provider_recommendation(
                text_analysis, quality_tier, max_cost, speed_priority
            )
            provider_name = rec["provider"]
            selected_model = rec["model"]
            estimated_cost = rec["estimated_cost"]
            reasoning = rec["reasoning"]
            logger.info(
                "%s/%s ($%.4f): %s",
                provider_name,
                selected_model,
                estimated_cost,
                reasoning,
            )
        else:
            if not quality_tier:
                quality_tier = QualityTier.BALANCED
            rec = self.get_smart_provider_recommendation(
                text_analysis, quality_tier, max_cost, speed_priority
            )
            provider_name = rec["provider"]
            selected_model = rec["model"]
            estimated_cost = rec["estimated_cost"]
            reasoning = "Default"
            logger.info(
                "%s/%s ($%.4f): %s",
                provider_name,
                selected_model,
                estimated_cost,
                reasoning,
            )

        provider = self._get_provider_instance(provider_name, quality_tier)
        if not selected_model:
            selected_model = provider.model_name

        return provider, selected_model, estimated_cost, reasoning

    def _get_provider_instance(
        self, provider_name: str | None, quality_tier: QualityTier | None
    ) -> EmbeddingProvider:
        """Gets provider.

        Args:
            provider_name: Name.
            quality_tier: Tier.

        Returns:
            Provider instance.

        Raises:
            EmbeddingServiceError: Provider not found.
        """
        if provider_name:
            provider = self.providers.get(provider_name)
            if not provider:
                available = ", ".join(self.providers.keys())
                raise EmbeddingServiceError(
                    f"Provider '{provider_name}' unavailable. Available: {available}"
                )
        elif quality_tier:
            preferred = self._tier_providers.get(quality_tier)
            provider = self.providers.get(preferred)
            if not provider:
                provider = next(iter(self.providers.values()))
                logger.warning(
                    "Tier %s fallback to %s",
                    quality_tier.value,
                    provider.__class__.__name__,
                )
        else:
            provider = self.providers.get(self.config.embedding_provider.value)
            if not provider:
                provider = next(iter(self.providers.values()))

        return provider

    def _validate_budget_constraints(self, estimated_cost: float) -> None:
        """Validates budget.

        Args:
            estimated_cost: Cost.

        Raises:
            EmbeddingServiceError: Over budget.
        """
        check = self.check_budget_constraints(estimated_cost)
        if not check["within_budget"]:
            raise EmbeddingServiceError("Budget exceeded")
        for w in check["warnings"]:
            logger.warning(w)

    def _calculate_metrics_and_update_stats(
        self,
        context: EmbeddingMetricsContext,
    ) -> dict[str, object]:
        """Calculates metrics.

        Args:
            context: Context.

        Returns:
            Metrics dict.
        """
        end_time = time.time()
        latency = (end_time - context.start_time) * 1000

        tokens = sum(len(t) for t in context.texts) // int(
            self._smart_config.chars_per_token
        )
        cost = tokens * context.provider.cost_per_token

        tier = context.quality_tier.value if context.quality_tier else "default"
        p_key = (
            context.provider_name
            or context.provider.__class__.__name__.lower().replace("provider", "")
        )

        self.update_usage_stats(p_key, context.selected_model, tokens, cost, tier)

        return {
            "latency_ms": latency,
            "tokens": tokens,
            "cost": cost,
            "tier_name": tier,
            "provider_key": p_key,
        }

    async def generate_embeddings(
        self,
        texts: list[str],
        quality_tier: QualityTier | None = None,
        provider_name: str | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
        auto_select: bool = True,
        generate_sparse: bool = False,
    ) -> dict[str, object]:
        """Generates embeddings.

        Args:
            texts: Texts.
            quality_tier: Tier.
            provider_name: Provider.
            max_cost: Limit.
            speed_priority: Flag.
            auto_select: Flag.
            generate_sparse: Flag.

        Returns:
            Result dict.

        Raises:
            EmbeddingServiceError: Errors in process.
        """
        if not self._initialized:
            raise EmbeddingServiceError("Not initialized")

        if not texts:
            return {
                "embeddings": [],
                "provider": None,
                "model": None,
                "cost": 0.0,
                "reasoning": "No texts",
                "usage_stats": self.get_usage_report(),
            }

        start = time.time()
        analysis = self.analyze_text_characteristics(texts)

        # Cache single
        if self.cache_manager and len(texts) == 1:
            t = texts[0]
            if hasattr(self.cache_manager, "get_embedding"):
                p_key = provider_name or self.config.embedding_provider.value
                m_name = (
                    self.config.openai.model
                    if p_key == "openai"
                    else self.config.fastembed.model
                )
                cached = await self.cache_manager.get_embedding(
                    text=t,
                    provider=p_key,
                    model=m_name,
                    dimensions=self.config.openai.dimensions,
                )
                if cached is not None:
                    logger.info("Cache hit")
                    return {
                        "embeddings": [cached],
                        "provider": p_key,
                        "model": m_name,
                        "cost": 0.0,
                        "latency_ms": (time.time() - start) * 1000,
                        "tokens": 0,
                        "reasoning": "Cache",
                        "quality_tier": quality_tier.value
                        if quality_tier
                        else "default",
                        "usage_stats": self.get_usage_report(),
                        "cache_hit": True,
                    }

        # Select
        provider, model, est_cost, reason = self._select_provider_and_model(
            analysis, quality_tier, provider_name, max_cost, speed_priority, auto_select
        )

        self._validate_budget_constraints(est_cost)

        logger.info("Using %s for %d texts", provider.__class__.__name__, len(texts))

        # Dense
        try:
            embeddings = await self._generate_dense_embeddings(provider, texts)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.exception("Dense failed: %s", e)
            raise EmbeddingServiceError(f"Dense failed: {e}") from e

        # Sparse
        try:
            sparse = await self._generate_sparse_embeddings_if_needed(
                provider, texts, generate_sparse
            )
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning("Sparse failed: %s", e)
            sparse = None

        # Metrics
        try:
            ctx = EmbeddingMetricsContext(
                provider=provider,
                provider_name=provider_name,
                selected_model=model,
                texts=texts,
                quality_tier=quality_tier,
                start_time=start,
            )
            metrics = self._calculate_metrics_and_update_stats(ctx)
        except (ValueError, RuntimeError) as e:
            logger.exception("Metrics failed")
            raise EmbeddingServiceError(f"Metrics failed: {e}") from e

        # Cache
        try:
            await self._cache_embedding_if_applicable(texts, embeddings, model, metrics)
        except (AttributeError, RuntimeError, ConnectionError) as e:
            logger.warning("Cache failed: %s", e)

        # Result
        try:
            result = self._build_embedding_result(
                embeddings, sparse, metrics, model, reason
            )
        except (ValueError, RuntimeError) as e:
            logger.exception("Result failed")
            raise EmbeddingServiceError(f"Result failed: {e}") from e

        return result

    async def _generate_dense_embeddings(
        self, provider: EmbeddingProvider, texts: list[str]
    ) -> list[list[float]]:
        """Generates dense embeddings.

        Args:
            provider: Provider.
            texts: Texts.

        Returns:
            Vectors.
        """
        return await provider.generate_embeddings(texts, batch_size=32)

    async def _generate_sparse_embeddings_if_needed(
        self, provider: EmbeddingProvider, texts: list[str], generate_sparse: bool
    ) -> list[dict] | None:
        """Generates sparse if needed.

        Args:
            provider: Provider.
            texts: Texts.
            generate_sparse: Flag.

        Returns:
            Sparse or None.
        """
        if not generate_sparse or not hasattr(provider, "generate_sparse_embeddings"):
            return None

        sparse = await provider.generate_sparse_embeddings(texts)
        logger.info("%d sparse generated", len(sparse))
        return sparse

    async def _cache_embedding_if_applicable(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        selected_model: str,
        metrics: dict[str, object],
    ) -> None:
        """Caches single embedding.

        Args:
            texts: Texts.
            embeddings: Embeddings.
            selected_model: Model.
            metrics: Metrics.
        """
        if not (self.cache_manager and len(texts) == 1 and len(embeddings) == 1):
            return

        if hasattr(self.cache_manager, "set_embedding"):
            await self.cache_manager.set_embedding(
                text=texts[0],
                provider=metrics["provider_key"],
                model=selected_model,
                dimensions=len(embeddings[0]),
                embedding=embeddings[0],
            )
            logger.info("Cached")

    def _build_embedding_result(
        self,
        embeddings: list[list[float]],
        sparse_embeddings: list[dict] | None,
        metrics: dict[str, object],
        selected_model: str,
        reasoning: str,
    ) -> dict[str, object]:
        """Builds result.

        Args:
            embeddings: Dense.
            sparse_embeddings: Sparse.
            metrics: Metrics.
            selected_model: Model.
            reasoning: Reason.

        Returns:
            Result dict.
        """
        result = {
            "embeddings": embeddings,
            "provider": metrics["provider_key"],
            "model": selected_model,
            "cost": metrics["cost"],
            "latency_ms": metrics["latency_ms"],
            "tokens": metrics["tokens"],
            "reasoning": reasoning,
            "quality_tier": metrics["tier_name"],
            "usage_stats": self.get_usage_report(),
            "cache_hit": False,
        }

        if sparse_embeddings is not None:
            result["sparse_embeddings"] = sparse_embeddings

        return result

    async def rerank_results(
        self, query: str, results: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Reranks results.

        Args:
            query: Query.
            results: Results with 'content'.

        Returns:
            Reranked results.
        """
        if not self._reranker:
            logger.warning("No reranker")
            return results

        if len(results) <= 1:
            return results

        try:
            pairs = [[query, r.get("content", "")] for r in results]
            scores = self._reranker.compute_score(pairs, normalize=True)

            if isinstance(scores, (int, float)):
                scores = [scores]

            scored = list(zip(results, scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [r for r, _ in scored]
        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Rerank failed")
            return results
        else:
            logger.info("Reranked %d", len(results))
            return reranked

    def load_custom_benchmarks(self, benchmark_file: Path | str) -> None:
        """Loads benchmarks from file.

        Args:
            benchmark_file: JSON path.

        Raises:
            FileNotFoundError: File missing.
            json.JSONDecodeError: Invalid JSON.
            ValueError: Validation error.
        """
        path = Path(benchmark_file)
        with path.open() as f:
            data = json.load(f)

        config = Config(**data)

        self._benchmarks = getattr(config.embedding, "model_benchmarks", {})
        self._smart_config = getattr(config.embedding, "smart_selection", None)

        logger.info("Loaded %d from %s", len(self._benchmarks), path.name)

    def estimate_cost(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Estimates costs.

        Args:
            texts: Texts.
            provider_name: Provider or None.

        Returns:
            Costs dict.

        Raises:
            EmbeddingServiceError: Not initialized.
        """
        if not self._initialized:
            raise EmbeddingServiceError("Not initialized")

        chars = sum(len(t) for t in texts)
        tokens = chars / 4

        costs = {}
        providers = (
            {provider_name: self.providers.get(provider_name)}
            if provider_name
            else self.providers
        )

        for name, p in providers.items():
            if p:
                c = tokens * p.cost_per_token
                costs[name] = {
                    "estimated_tokens": tokens,
                    "cost_per_token": p.cost_per_token,
                    "total_cost": c,
                }

        return costs

    def get_provider_info(self) -> dict[str, dict[str, object]]:
        """Provider info.

        Returns:
            Info dict.
        """
        info = {}
        for name, p in self.providers.items():
            info[name] = {
                "model": p.model_name,
                "dimensions": p.dimensions,
                "cost_per_token": p.cost_per_token,
                "max_tokens": p.max_tokens_per_request,
            }
        return info

    async def get_optimal_provider(
        self,
        text_length: int,
        quality_required: bool = False,
        budget_limit: float | None = None,
    ) -> str:
        """Optimal provider.

        Args:
            text_length: Length.
            quality_required: Quality flag.
            budget_limit: Limit.

        Returns:
            Provider name.

        Raises:
            EmbeddingServiceError: No provider.
        """
        if not self._initialized:
            raise EmbeddingServiceError("Not initialized")

        tokens = text_length / 4

        candidates = []
        for name, p in self.providers.items():
            c = tokens * p.cost_per_token
            if budget_limit and c > budget_limit:
                continue
            candidates.append(
                {"name": name, "cost": c, "is_local": p.cost_per_token == 0}
            )

        if not candidates:
            raise EmbeddingServiceError(f"No provider for ${budget_limit}")

        if quality_required and "openai" in self.providers:
            return "openai"

        if text_length < 10000:
            for c in candidates:
                if c["is_local"]:
                    return c["name"]

        candidates.sort(key=lambda x: x["cost"])
        return candidates[0]["name"]

    def analyze_text_characteristics(self, texts: list[str]) -> TextAnalysis:
        """Analyzes text.

        Args:
            texts: Texts.

        Returns:
            Analysis.
        """
        if not texts:
            return TextAnalysis(0, 0, 0.0, 0, "empty", False)

        valid = [t for t in texts if t is not None]
        if not valid:
            return TextAnalysis(0, 0, 0.0, 0, "empty", False)

        length = sum(len(t) for t in valid)
        avg = length // len(valid)
        tokens = int(length / self._smart_config.chars_per_token)

        words_set = set()
        word_count = 0
        code_count = 0

        for t in valid:
            w = t.lower().split()
            words_set.update(w)
            word_count += len(w)
            if any(k in t.lower() for k in self._smart_config.code_keywords):
                code_count += 1

        complexity = 0.0
        if word_count > 0:
            complexity = min(len(words_set) / word_count * 1.5, 1.0)

        is_code = code_count / len(texts) > 0.3
        type_ = (
            "code"
            if is_code
            else (
                "long"
                if avg > self._smart_config.long_text_threshold
                else (
                    "short" if avg < self._smart_config.short_text_threshold else "docs"
                )
            )
        )

        high_quality = is_code or complexity > 0.7 or avg > 1500

        return TextAnalysis(length, avg, complexity, tokens, type_, high_quality)

    def get_smart_provider_recommendation(
        self,
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
    ) -> dict[str, object]:
        """Recommends provider/model.

        Args:
            text_analysis: Analysis.
            quality_tier: Tier.
            max_cost: Limit.
            speed_priority: Flag.

        Returns:
            Recommendation.

        Raises:
            EmbeddingServiceError: No model.
        """
        if not self._initialized:
            raise EmbeddingServiceError("Not initialized")

        candidates = []
        for p_name, p in self.providers.items():
            models = (
                ["text-embedding-3-small", "text-embedding-3-large"]
                if p_name == "openai"
                else [p.model_name]
            )
            for m in models:
                if m in self._benchmarks:
                    b = self._benchmarks[m]
                    c = text_analysis.estimated_tokens * (
                        b["cost_per_million_tokens"] / 1_000_000
                    )
                    if max_cost and c > max_cost:
                        continue
                    s = self._calculate_model_score(
                        b, text_analysis, quality_tier, speed_priority
                    )
                    candidates.append(
                        {
                            "provider": p_name,
                            "model": m,
                            "benchmark": b,
                            "estimated_cost": c,
                            "score": s,
                        }
                    )

        if not candidates:
            raise EmbeddingServiceError(
                f"No model for max_cost={max_cost}, tokens={text_analysis.estimated_tokens}"
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        reasoning = self._generate_selection_reasoning(
            best, text_analysis, quality_tier, speed_priority
        )

        return {
            "provider": best["provider"],
            "model": best["model"],
            "estimated_cost": best["estimated_cost"],
            "score": best["score"],
            "reasoning": reasoning,
            "alternatives": candidates[1:3],
        }

    def _calculate_model_score(
        self,
        benchmark: dict[str, Any],
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> float:
        """Model score.

        Args:
            benchmark: Benchmark.
            text_analysis: Analysis.
            quality_tier: Tier.
            speed_priority: Flag.

        Returns:
            Score.
        """
        score = 0.0

        score += benchmark["quality_score"] * self._smart_config.quality_weight

        s_weight = 0.5 if speed_priority else self._smart_config.speed_weight
        l = benchmark["avg_latency_ms"]
        s_score = max(
            0,
            (self._smart_config.speed_balanced_threshold - l)
            / self._smart_config.speed_balanced_threshold
            * 100,
        )
        score += s_score * s_weight

        c_weight = 0.1 if speed_priority else self._smart_config.cost_weight
        c_per_m = benchmark["cost_per_million_tokens"]
        if c_per_m == 0:
            c_score = 100
        else:
            c_score = max(
                0,
                (self._smart_config.cost_expensive_threshold - c_per_m)
                / self._smart_config.cost_expensive_threshold
                * 100,
            )
        score += c_score * c_weight

        if quality_tier == QualityTier.FAST and c_per_m == 0:
            score += 25
        elif quality_tier == QualityTier.BEST:
            q = benchmark["quality_score"]
            if q > self._smart_config.quality_best_threshold:
                score += 40
            elif q > self._smart_config.quality_balanced_threshold:
                score += 30
            else:
                score -= 10
        elif quality_tier == QualityTier.BALANCED:
            if c_per_m == 0:
                score += 10
            elif c_per_m < self._smart_config.cost_cheap_threshold:
                score += 15

        if (
            text_analysis.text_type == "code"
            and benchmark["quality_score"] > self._smart_config.quality_best_threshold
        ) or (text_analysis.text_type == "short" and benchmark["avg_latency_ms"] < 60):
            score += 5

        return min(score, 100.0)

    def _generate_selection_reasoning(
        self,
        selection: dict[str, object],
        text_analysis: TextAnalysis,
        _quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> str:
        """Reasoning string.

        Args:
            selection: Selection.
            text_analysis: Analysis.
            _quality_tier: Tier.
            speed_priority: Flag.

        Returns:
            String.
        """
        b = selection["benchmark"]
        reasons = []

        if speed_priority:
            reasons.append(f"Speed: {b['avg_latency_ms']}ms")
        elif b["cost_per_million_tokens"] == 0:
            reasons.append("Local zero cost")
        elif b["quality_score"] > 90:
            reasons.append(f"Quality {b['quality_score']}/100")

        if text_analysis.text_type == "code":
            reasons.append("Code high accuracy")
        elif text_analysis.requires_high_quality:
            reasons.append("Complex quality")

        est = selection["estimated_cost"]
        if est > 0:
            reasons.append(f"Cost ${est:.4f}")
        else:
            reasons.append("Zero cost")

        return "; ".join(reasons)

    def check_budget_constraints(self, estimated_cost: float) -> dict[str, object]:
        """Budget check.

        Args:
            estimated_cost: Cost.

        Returns:
            Check dict.
        """
        result = {
            "within_budget": True,
            "warnings": [],
            "daily_usage": self.usage_stats.daily_cost,
            "estimated_total": self.usage_stats.daily_cost + estimated_cost,
            "budget_limit": self.budget_limit,
        }

        if self.budget_limit:
            proj = self.usage_stats.daily_cost + estimated_cost
            if proj > self.budget_limit:
                result["within_budget"] = False
                result["warnings"].append(
                    f"Exceeds ${proj:.4f} > ${self.budget_limit:.2f}"
                )

            if self.budget_limit > 0:
                p = proj / self.budget_limit
                if p > self._smart_config.budget_critical_threshold:
                    result["warnings"].append(
                        f"Critical >{int(self._smart_config.budget_critical_threshold * 100)}%"
                    )
                elif p > self._smart_config.budget_warning_threshold:
                    result["warnings"].append(
                        f"Warning >{int(self._smart_config.budget_warning_threshold * 100)}%"
                    )

        return result

    def update_usage_stats(
        self, provider: str, _model: str, tokens: int, cost: float, tier: str
    ) -> None:
        """Updates stats.

        Args:
            provider: Provider.
            _model: Model.
            tokens: Tokens.
            cost: Cost.
            tier: Tier.
        """
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        self.usage_stats.daily_cost += cost
        self.usage_stats.requests_by_tier[tier] += 1
        self.usage_stats.requests_by_provider[provider] += 1

        today = time.strftime("%Y-%m-%d")
        if self.usage_stats.last_reset_date != today:
            self.usage_stats.daily_cost = cost
            self.usage_stats.last_reset_date = today

    def get_usage_report(self) -> dict[str, object]:
        """Usage report.

        Returns:
            Report dict.
        """
        req = self.usage_stats.total_requests
        cost = self.usage_stats.total_cost

        return {
            "summary": {
                "total_requests": req,
                "total_tokens": self.usage_stats.total_tokens,
                "total_cost": cost,
                "daily_cost": self.usage_stats.daily_cost,
                "avg_cost_per_request": cost / max(req, 1),
                "avg_tokens_per_request": self.usage_stats.total_tokens / max(req, 1),
            },
            "by_tier": dict(self.usage_stats.requests_by_tier),
            "by_provider": dict(self.usage_stats.requests_by_provider),
            "budget": {
                "daily_limit": self.budget_limit,
                "daily_usage": self.usage_stats.daily_cost,
                "remaining": self.budget_limit - self.usage_stats.daily_cost
                if self.budget_limit
                else None,
            },
        }
