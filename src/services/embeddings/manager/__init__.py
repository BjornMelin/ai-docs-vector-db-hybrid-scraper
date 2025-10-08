"""Embedding manager with provider selection."""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,no-else-return
# Global refactor is tracked separately; suppress aggregate warnings temporarily.

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast


try:
    from src.services.cache import CacheManager
except ImportError:
    CacheManager = None  # type: ignore[assignment]

from src.config.loader import Settings
from src.config.models import EmbeddingConfig as SettingsEmbeddingConfig
from src.services.errors import EmbeddingServiceError

from ..base import EmbeddingProvider
from ..fastembed_provider import FastEmbedProvider
from ..openai_provider import OpenAIEmbeddingProvider
from .pipeline import EmbeddingPipeline, GenerationOptions, PipelineContext
from .providers import ProviderFactories, ProviderRegistry
from .selection import RecommendationParams, SelectionEngine, TextAnalysis
from .types import QualityTier
from .usage import UsageRecord, UsageStats, UsageTracker


if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager
    from src.services.cache import CacheManager as CacheManagerType
    from src.services.cache.embedding_cache import EmbeddingCache
else:  # pragma: no cover - runtime fallback for optional cache dependency
    CacheManagerType = Any
    EmbeddingCache = Any

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding providers and selects based on text analysis, cost, quality.

    Supports OpenAI and FastEmbed providers. Selection uses text length, complexity,
    and type to balance quality, latency, and cost.

    Includes caching for single texts, reranking for search results, and usage tracking.
    """

    def __init__(
        self,
        config: Settings,
        client_manager: "ClientManager",
        budget_limit: float | None = None,
        rate_limiter: Any = None,
    ):
        """Initializes the embedding manager.

        Args:
            config: Configuration object with provider settings and selection rules.
            client_manager: Manages shared HTTP clients.
            budget_limit: Daily budget limit in USD, if set.
            rate_limiter: Optional rate limiter for API calls.
        """

        self.config = config
        self._initialized = False
        self.rate_limiter = rate_limiter
        self._client_manager = client_manager
        self._usage: UsageTracker | None = None
        self._budget_limit = budget_limit

        # Initialize cache manager if caching is enabled
        self.cache_manager: CacheManagerType | None = None
        if config.cache.enable_caching and CacheManager is not None:
            dragonfly_url_config = getattr(config.cache, "dragonfly_url", None)
            dragonfly_url = (
                str(dragonfly_url_config)
                if dragonfly_url_config
                else "redis://localhost:6379"
            )
            enable_local_cache = getattr(config.cache, "enable_local_cache", False)
            enable_distributed_cache = getattr(
                config.cache, "enable_dragonfly_cache", False
            )
            local_max_size = getattr(config.cache, "local_max_size", 1000)
            local_max_memory_mb = getattr(config.cache, "local_max_memory_mb", 512)
            cache_ttl_seconds = getattr(config.cache, "cache_ttl_seconds", {})
            cache_root = Path(getattr(config, "cache_dir", Path("cache")))
            memory_threshold = getattr(config.cache, "memory_pressure_threshold", None)
            cache_manager = CacheManager(
                dragonfly_url=dragonfly_url,
                enable_local_cache=enable_local_cache,
                enable_distributed_cache=enable_distributed_cache,
                local_max_size=local_max_size,
                local_max_memory_mb=local_max_memory_mb,
                distributed_ttl_seconds=cache_ttl_seconds,
                local_cache_path=cache_root / "embeddings",
                memory_pressure_threshold=memory_threshold,
            )
            self.cache_manager = cast(CacheManagerType, cache_manager)

        # Load model benchmarks and selection configuration from config
        self._benchmarks: dict[str, dict[str, Any]] = getattr(
            config.embedding, "model_benchmarks", {}
        )
        self._smart_config = getattr(config.embedding, "smart_selection", None)
        self._usage = UsageTracker(self._smart_config, budget_limit)
        self._selection = SelectionEngine(self._smart_config, self._benchmarks)
        self._provider_registry = ProviderRegistry(
            config=config,
            client_manager=client_manager,
            rate_limiter=rate_limiter,
            factories=None,
        )
        pipeline_context = PipelineContext(
            config=config,
            usage=self._usage,
            selection=self._selection,
            providers=self._provider_registry,
            cache_manager=self.cache_manager,
            smart_config=self._smart_config,
        )
        self._pipeline = EmbeddingPipeline(pipeline_context)

    @property
    def budget_limit(self) -> float | None:
        """Expose configured daily budget limit."""

        return self._budget_limit

    @budget_limit.setter
    def budget_limit(self, value: float | None) -> None:
        """Update budget limit and synchronize usage tracker."""

        self._budget_limit = value
        if self._usage is not None:
            self._usage.set_budget_limit(value)

    @property
    def usage_stats(self) -> UsageStats:
        """Return current usage statistics."""

        if self._usage is None:
            self._usage = UsageTracker(self._smart_config, self._budget_limit)
            self._pipeline.set_usage_tracker(self._usage)
        return self._usage.stats

    @property
    def providers(self) -> dict[str, EmbeddingProvider]:
        """Return the provider registry for compatibility with legacy tests."""

        return self._provider_registry.providers

    @providers.setter
    def providers(self, value: dict[str, EmbeddingProvider]) -> None:
        """Allow tests to replace provider map directly."""

        self._provider_registry.providers = value

    async def initialize(self) -> None:
        """Initializes available providers.

        Initializes OpenAI provider if API key is available and FastEmbed provider
        for local embeddings. At least one provider must initialize successfully.
        """
        if self._initialized:
            return

        self._provider_registry.set_factories(
            ProviderFactories(
                openai_cls=OpenAIEmbeddingProvider,
                fastembed_cls=FastEmbedProvider,
            )
        )

        providers = await self._provider_registry.initialize()
        self._initialized = True
        logger.info("Embedding manager initialized with %d providers", len(providers))

    async def cleanup(self) -> None:
        """Clean up all providers and cache.

        Shuts down all initialized providers and closes cache manager
        if initialized. Errors during cleanup are logged but not raised.
        """
        await self._provider_registry.cleanup()
        self._initialized = False

        # Close cache manager if initialized
        if self.cache_manager is not None:
            if hasattr(self.cache_manager, "close"):
                await self.cache_manager.close()  # type: ignore
            logger.info("Closed cache manager")

    def _select_provider_and_model(
        self,
        text_analysis: "TextAnalysis",
        quality_tier: QualityTier | None,
        provider_name: str | None,
        max_cost: float | None,
        speed_priority: bool,
        auto_select: bool,
    ) -> tuple[EmbeddingProvider, str, float, str]:
        """Select provider and model.

        Args:
            text_analysis: Analysis of text characteristics
            quality_tier: Optional quality tier preference
            provider_name: Optional specific provider name override
            max_cost: Optional maximum cost constraint in USD
            speed_priority: Whether to prioritize speed over quality
            auto_select: Whether to use auto-selection

        Returns:
            tuple: (provider, model_name, estimated_cost, reasoning)
        """
        options = GenerationOptions(
            quality_tier=quality_tier,
            provider_name=provider_name,
            max_cost=max_cost,
            speed_priority=speed_priority,
            auto_select=auto_select,
            generate_sparse=False,
        )
        selection = self._pipeline.select_provider(
            text_analysis=text_analysis,
            options=options,
        )
        return (
            selection.provider,
            selection.model,
            selection.estimated_cost,
            selection.reasoning,
        )

    def update_usage_stats(
        self,
        provider: str,
        model: str,
        tokens: int,
        cost: float,
        tier: str,
    ) -> None:
        """Update cumulative and daily usage statistics.

        Args:
            provider: Provider name (e.g., ``"openai"``).
            model: Model identifier used for embeddings.
            tokens: Token count consumed by the request.
            cost: Monetary cost incurred by the request.
            tier: Quality tier label associated with the request.
        """

        if self._usage is None:
            self._usage = UsageTracker(self._smart_config, self._budget_limit)
            self._pipeline.set_usage_tracker(self._usage)
        record = UsageRecord(
            provider=provider,
            model=model,
            tokens=tokens,
            cost=cost,
            tier=tier,
        )
        self._usage.record(record)

    def check_budget_constraints(self, estimated_cost: float) -> dict[str, Any]:
        """Evaluate projected spend against configured budget limits.

        Args:
            estimated_cost: Anticipated incremental cost for the request.

        Returns:
            dict[str, Any]: Budget evaluation summary containing flags, warnings,
            and projected totals.
        """
        if self._usage is None:
            self._usage = UsageTracker(self._smart_config, self._budget_limit)
        return self._usage.check_budget(estimated_cost)

    def get_usage_report(self) -> dict[str, Any]:
        """Build a snapshot of usage aggregates for monitoring and budgeting.

        Returns:
            dict[str, Any]: Aggregated counters grouped by tier, provider, and
            budget metadata for reporting.
        """

        if self._usage is None:
            self._usage = UsageTracker(self._smart_config, self._budget_limit)
        return self._usage.report()

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
        """Generate embeddings with provider selection and optimization.

        Workflow:
        1. Text analysis for optimal provider selection
        2. Cache lookup for single texts
        3. Budget validation
        4. Dense embedding generation
        5. Optional sparse embedding generation
        6. Metrics calculation and usage tracking
        7. Result caching

        Args:
            texts: Text strings to embed
            quality_tier: Quality tier (FAST, BALANCED, BEST) for auto-selection
            provider_name: Explicit provider name ("openai" or "fastembed")
            max_cost: Optional maximum cost constraint in USD
            speed_priority: Whether to prioritize speed over quality
            auto_select: Use automatic selection (True) or manual selection (False)
            generate_sparse: Whether to generate sparse embeddings for hybrid search

        Returns:
            dict: Result containing:
                - embeddings: List of embedding vectors
                - provider: Provider name used
                - model: Model name used
                - cost: Cost incurred in USD
                - latency_ms: Processing time
                - tokens: Token count processed
                - reasoning: Selection reasoning
                - quality_tier: Quality tier used
                - sparse_embeddings: Sparse embeddings if requested
                - cache_hit: Whether result came from cache
        """

        if not self._initialized:
            msg = "Manager not initialized"
            raise EmbeddingServiceError(msg)

        if not texts:
            return {
                "embeddings": [],
                "provider": None,
                "model": None,
                "cost": 0.0,
                "reasoning": "Empty input",
                "usage_stats": self.get_usage_report(),
            }

        options = GenerationOptions(
            quality_tier=quality_tier,
            provider_name=provider_name,
            max_cost=max_cost,
            speed_priority=speed_priority,
            auto_select=auto_select,
            generate_sparse=generate_sparse,
        )
        return await self._pipeline.generate(texts=texts, options=options)

    async def rerank_results(
        self, query: str, results: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Rerank search results using BGE reranker.

        Args:
            query: Search query to compare against
            results: List of search results with 'content' field

        Returns:
            Reranked results sorted by relevance score (highest first)
        """
        reranker = self._provider_registry.reranker
        if not reranker:
            logger.warning("Reranker not available, returning original results")
            return results

        if not results or len(results) <= 1:
            return results

        try:
            # Prepare query-result pairs for reranking
            pairs = [(query, str(result.get("content", ""))) for result in results]

            # Get reranking scores using BGE model
            raw_scores = reranker.compute_score(pairs, normalize=True, strict=False)

            score_sequence: list[float]
            if isinstance(raw_scores, (int, float)):
                score_sequence = [float(raw_scores)]
            elif hasattr(raw_scores, "tolist"):
                score_sequence = [float(score) for score in raw_scores.tolist()]  # type: ignore[assignment]
            else:
                score_sequence = [float(score) for score in raw_scores]  # type: ignore[assignment]

            if len(score_sequence) != len(results):
                logger.warning(
                    "Reranker returned %d scores for %d results",
                    len(score_sequence),
                    len(results),
                )
                return results

            # Combine results with scores and sort by relevance
            scored_results = list(zip(results, score_sequence, strict=False))
            scored_results.sort(key=lambda item: item[1], reverse=True)

            # Extract reordered results
            reranked = [result for result, _ in scored_results]
        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Reranking failed: ")
            # Return original results on failure
            return results
        else:
            logger.info(
                "Reranked %d results using %s",
                len(results),
                self._provider_registry.reranker_model,
            )
            return reranked

    def load_custom_benchmarks(self, benchmark_file: Path | str) -> None:
        """Load custom benchmark configuration from file.

        Args:
            benchmark_file: Path to benchmark configuration JSON file

        Raises:
            FileNotFoundError: If benchmark file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            pydantic.ValidationError: If data doesn't match expected schema
        """
        # Load and validate benchmark configuration
        benchmark_path = Path(benchmark_file)
        with benchmark_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        embedding_config = SettingsEmbeddingConfig.model_validate(
            benchmark_data.get("embedding", {})
        )

        # Update manager's benchmarks and selection configuration
        self._benchmarks = getattr(embedding_config, "model_benchmarks", {}) or {}
        self._smart_config = getattr(embedding_config, "smart_selection", None)
        self._selection = SelectionEngine(self._smart_config, self._benchmarks)
        if self._usage is None:
            self._usage = UsageTracker(self._smart_config, self._budget_limit)
        else:
            self._usage.set_smart_config(self._smart_config)
        self._pipeline.set_selection_engine(self._selection)
        self._pipeline.set_smart_config(self._smart_config)
        self._pipeline.set_usage_tracker(self._usage)

        logger.info(
            "Loaded custom benchmarks from %s with %d models",
            benchmark_path.name,
            len(self._benchmarks),
        )

    def estimate_cost(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Estimate embedding generation cost across providers.

        Uses character-to-token ratio (~4 chars per token) for estimation.
        For precise costs, actual tokenization should be used.

        Args:
            texts: List of texts to estimate cost for
            provider_name: Optional specific provider to estimate (est. all if None)

        Returns:
            dict[str, dict[str, float]]: Cost estimation per provider containing:
                - estimated_tokens: Token count estimate
                - cost_per_token: Cost per token in USD
                - total_cost: Total estimated cost in USD
        """
        if not self._initialized:
            msg = "Manager not initialized"
            raise EmbeddingServiceError(msg)

        # Simple token estimation using average character-to-token ratio
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars / 4

        costs = {}

        if provider_name:
            providers = {provider_name: self.providers.get(provider_name)}
        else:
            providers = self.providers

        for name, provider in providers.items():
            if provider:
                cost = estimated_tokens * provider.cost_per_token
                costs[name] = {
                    "estimated_tokens": estimated_tokens,
                    "cost_per_token": provider.cost_per_token,
                    "total_cost": cost,
                }

        return costs

    def get_provider_info(self) -> dict[str, dict[str, object]]:
        """Get information about available providers.

        Returns:
            dict[str, dict[str, object]]: Provider information
        """
        info = {}
        for name, provider in self.providers.items():
            info[name] = {
                "model": provider.model_name,
                "dimensions": provider.dimensions,
                "cost_per_token": provider.cost_per_token,
                "max_tokens": provider.max_tokens_per_request,
            }
        return info

    async def get_optimal_provider(
        self,
        text_length: int,
        quality_required: bool = False,
        budget_limit: float | None = None,
    ) -> str:
        """Select optimal provider using heuristics.

        Args:
            text_length: Total character count
            quality_required: Whether high quality is required
            budget_limit: Optional maximum cost in dollars

        Returns:
            Optimal provider name
        """
        if not self._initialized:
            msg = "Manager not initialized"
            raise EmbeddingServiceError(msg)

        # Simple heuristic-based provider selection
        estimated_tokens = text_length / 4

        candidates = []
        for name, provider in self.providers.items():
            cost = estimated_tokens * provider.cost_per_token

            # Apply budget constraint
            if budget_limit and cost > budget_limit:
                continue

            candidates.append(
                {
                    "name": name,
                    "cost": cost,
                    "is_local": provider.cost_per_token == 0,
                }
            )

        if not candidates:
            msg = f"No provider available within budget {budget_limit}"
            raise EmbeddingServiceError(msg)

        # Apply selection heuristics
        if quality_required and "openai" in self.providers:
            return "openai"

        # Prefer local processing for small texts
        if text_length < 10000:
            for candidate in candidates:
                if candidate["is_local"]:
                    return candidate["name"]

        # Return cheapest option for larger texts
        candidates.sort(key=lambda x: x["cost"])
        return candidates[0]["name"]

    def analyze_text_characteristics(self, texts: list[str]) -> TextAnalysis:
        """Analyze text characteristics for intelligent model selection.

        Args:
            texts: List of texts to analyze.

        Returns:
            TextAnalysis: Structured analysis summary.
        """

        return self._selection.analyze(texts)

    def get_smart_provider_recommendation(
        self,
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
    ) -> dict[str, Any]:
        """Get provider recommendation using scoring algorithm.

        Args:
            text_analysis: Text analysis results with complexity and type
            quality_tier: Optional quality tier override (FAST, BALANCED, BEST)
            max_cost: Optional maximum cost constraint in USD
            speed_priority: Whether to prioritize speed over quality

        Returns:
            dict: Recommendation containing provider/model/cost metadata.
        """
        if not self._initialized:
            msg = "Manager not initialized"
            raise EmbeddingServiceError(msg)

        params = RecommendationParams(
            quality_tier=quality_tier,
            max_cost=max_cost,
            speed_priority=speed_priority,
        )
        try:
            return self._selection.recommend(
                providers=self.providers,
                text_analysis=text_analysis,
                params=params,
            )
        except ValueError as exc:
            raise EmbeddingServiceError(str(exc)) from exc


__all__ = [
    "EmbeddingManager",
    "QualityTier",
    "TextAnalysis",
    "UsageStats",
    "FastEmbedProvider",
    "OpenAIEmbeddingProvider",
]
