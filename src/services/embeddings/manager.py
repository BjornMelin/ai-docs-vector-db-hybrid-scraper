"""Embedding manager with smart provider selection."""

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
    """Embedding quality tiers."""

    FAST = "fast"  # Local models, fastest
    BALANCED = "balanced"  # Balance of speed and quality
    BEST = "best"  # Highest quality, may be slower/costlier


@dataclass
class UsageStats:
    """Usage statistics tracking."""

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
    """Analysis of text characteristics for model selection."""

    total_length: int
    avg_length: int
    complexity_score: float  # 0-1 based on vocabulary diversity
    estimated_tokens: int
    text_type: str  # "code", "docs", "short", "long"
    requires_high_quality: bool


class EmbeddingManager:
    """Enterprise-grade embedding provider manager with intelligent selection.

    This manager orchestrates multiple embedding providers (OpenAI, FastEmbed) with
    smart selection algorithms, cost optimization, and performance tracking. It provides:

    - **Smart Provider Selection**: Automatic provider/model selection based on text
      characteristics, quality requirements, and cost constraints
    - **Budget Management**: Real-time cost tracking with configurable daily limits
    - **Performance Optimization**: Caching, batching, and reranking capabilities
    - **Quality Tiers**: FAST (local), BALANCED (cost-effective), BEST (high-quality)
    - **Multi-Modal Support**: Dense + sparse embeddings for hybrid search
    - **Enterprise Features**: Rate limiting, circuit breakers, usage analytics

    The manager analyzes text characteristics (length, complexity, type) to recommend
    optimal provider/model combinations, balancing quality, speed, and cost requirements.
    """

    def __init__(
        self,
        config: Config,
        client_manager: "ClientManager",
        budget_limit: float | None = None,
        rate_limiter: object = None,
    ):
        """Initialize embedding manager with smart selection capabilities.

        Args:
            config: Unified configuration containing provider settings and smart selection rules
            client_manager: ClientManager instance for dependency injection and resource management
            budget_limit: Optional daily budget limit in USD for automatic cost control
            rate_limiter: Optional RateLimitManager instance for API throttling

        """
        self.config = config
        self.providers: dict[str, EmbeddingProvider] = {}
        self._initialized = False
        self.budget_limit = budget_limit
        self.usage_stats = UsageStats()
        self.rate_limiter = rate_limiter
        self._client_manager = client_manager

        # Initialize cache manager if caching is enabled
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

        # Load model benchmarks and smart selection configuration from config
        # These drive the intelligent provider/model selection algorithms
        self._benchmarks: dict[str, dict[str, Any]] = getattr(
            config.embedding, "model_benchmarks", {}
        )
        self._smart_config = config.embedding.smart_selection

        # Quality tier to provider mappings - can be dynamically updated based on availability
        self._tier_providers = {
            QualityTier.FAST: "fastembed",  # Local processing for speed
            QualityTier.BALANCED: "fastembed",  # Cost-effective default
            QualityTier.BEST: "openai",  # Highest quality available
        }

        # Initialize reranker for search result optimization (optional component)
        self._reranker = None
        self._reranker_model = "BAAI/bge-reranker-v2-m3"
        if FlagReranker is not None:
            try:
                self._reranker = FlagReranker(self._reranker_model, use_fp16=True)
                logger.info("Initialized reranker")
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to initialize reranker: {e}")

    async def initialize(self) -> None:
        """Initialize available providers with connection validation.

        Initializes OpenAI provider if API key is available and FastEmbed provider
        for local embeddings. At least one provider must initialize successfully.

        Raises:
            EmbeddingServiceError: If no providers can be initialized

        """
        if self._initialized:
            return

        # Initialize OpenAI provider if API key is configured
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
                logger.info(
                    f"Initialized OpenAI provider with {self.config.openai.model}"
                )
            except (ImportError, ValueError, ConnectionError, RuntimeError) as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")

        # Initialize FastEmbed provider - always available for local embeddings
        try:
            provider = FastEmbedProvider(model_name=self.config.fastembed.model)
            await provider.initialize()
            self.providers["fastembed"] = provider
            logger.info(
                f"Initialized FastEmbed provider with {self.config.fastembed.model}"
            )
        except (ImportError, ValueError, RuntimeError, OSError) as e:
            logger.warning(f"Failed to initialize FastEmbed provider: {e}")

        # Validate that at least one provider is available
        if not self.providers:
            msg = (
                "No embedding providers available. "
                "Please configure OpenAI API key or enable local embeddings."
            )
            raise EmbeddingServiceError(msg)

        self._initialized = True
        logger.info(
            f"Embedding manager initialized with {len(self.providers)} providers"
        )

    async def cleanup(self) -> None:
        """Cleanup all providers and cache with graceful error handling.

        Gracefully shuts down all initialized providers and closes cache manager
        if it was initialized. Errors during cleanup are logged but not raised.
        """
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(
                    f"Cleaned up {name} provider"
                )  # TODO: Convert f-string to logging format
            except (RuntimeError, ConnectionError, TimeoutError):
                logger.exception("Error cleaning up  provider: {e}")

        self.providers.clear()
        self._initialized = False

        # Close cache manager if initialized
        if self.cache_manager:
            await self.cache_manager.close()
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
        """Select optimal provider and model using intelligent algorithms.

        This method implements the core smart selection logic, analyzing text characteristics
        and user preferences to choose the best provider/model combination. The selection
        considers quality requirements, cost constraints, and performance priorities.

        Args:
            text_analysis: Analysis of text characteristics (length, complexity, type)
            quality_tier: Optional quality tier preference (FAST, BALANCED, BEST)
            provider_name: Optional specific provider name override
            max_cost: Optional maximum cost constraint in USD
            speed_priority: Whether to prioritize speed over quality
            auto_select: Whether to use smart auto-selection algorithms

        Returns:
            tuple: Contains (provider, model_name, estimated_cost, reasoning)
                - provider: Selected EmbeddingProvider instance
                - model_name: Name of the selected model
                - estimated_cost: Estimated cost in USD
                - reasoning: Human-readable explanation of selection

        """
        if auto_select and not provider_name:
            # Use intelligent recommendation engine
            recommendation = self.get_smart_provider_recommendation(
                text_analysis, quality_tier, max_cost, speed_priority
            )

            provider_name = recommendation["provider"]
            selected_model = recommendation["model"]
            estimated_cost = recommendation["estimated_cost"]
            reasoning = recommendation["reasoning"]

            logger.info(
                f"Smart selection: {provider_name}/{selected_model} "
                f"(${estimated_cost:.4f}) - {reasoning}"
            )
        else:
            # Fallback to smart selection with balanced defaults
            # This modernizes the legacy behavior while maintaining compatibility
            if not quality_tier:
                quality_tier = QualityTier.BALANCED

            recommendation = self.get_smart_provider_recommendation(
                text_analysis, quality_tier, max_cost, speed_priority
            )

            provider_name = recommendation["provider"]
            selected_model = recommendation["model"]
            estimated_cost = recommendation["estimated_cost"]
            reasoning = "Default smart selection"

            logger.info(
                f"Default smart selection: {provider_name}/{selected_model} "
                f"(${estimated_cost:.4f}) - {reasoning}"
            )

        # Get the actual provider instance
        provider = self._get_provider_instance(provider_name, quality_tier)

        # Determine final model name
        if not selected_model:
            selected_model = provider.model_name

        return provider, selected_model, estimated_cost, reasoning

    def _get_provider_instance(
        self, provider_name: str | None, quality_tier: QualityTier | None
    ) -> EmbeddingProvider:
        """Get provider instance based on name or quality tier mapping.

        Args:
            provider_name: Specific provider name ("openai" or "fastembed")
            quality_tier: Quality tier to map to provider (FAST, BALANCED, BEST)

        Returns:
            EmbeddingProvider: The selected provider instance

        Raises:
            EmbeddingServiceError: If requested provider is not available

        """
        if provider_name:
            provider = self.providers.get(provider_name)
            if not provider:
                msg = f"Provider '{provider_name}' not available. Available"
                raise EmbeddingServiceError(msg)
        elif quality_tier:
            # Use quality tier mapping with fallback
            preferred = self._tier_providers.get(quality_tier)
            provider = self.providers.get(preferred)
            if not provider:
                # Fall back to any available provider
                provider = next(iter(self.providers.values()))
                logger.warning(
                    f"Preferred provider for {quality_tier.value} not available, "
                    f"using {provider.__class__.__name__}"
                )
        else:
            # Use default configured provider with fallback
            provider = self.providers.get(self.config.embedding_provider.value)
            if not provider:
                provider = next(iter(self.providers.values()))

        return provider

    def _validate_budget_constraints(self, estimated_cost: float) -> None:
        """Validate budget constraints and raise error if violated.

        Args:
            estimated_cost: Estimated cost for the embedding request in USD

        Raises:
            EmbeddingServiceError: If budget constraint would be violated

        """
        budget_check = self.check_budget_constraints(estimated_cost)
        if not budget_check["within_budget"]:
            msg = "Budget constraint violated"
            raise EmbeddingServiceError(msg)

        # Log warnings if any budget thresholds are exceeded
        for _warning in budget_check["warnings"]:
            logger.warning("Budget warning")

    def _calculate_metrics_and_update_stats(
        self,
        provider: EmbeddingProvider,
        provider_name: str | None,
        selected_model: str,
        texts: list[str],
        quality_tier: QualityTier | None,
        start_time: float,
    ) -> dict[str, object]:
        """Calculate comprehensive metrics and update usage statistics.

        This method handles all post-generation metric calculation including
        performance timing, token counting, cost estimation, and usage tracking.

        Args:
            provider: EmbeddingProvider instance used for generation
            provider_name: Optional provider name override
            selected_model: Model name that was used
            texts: List of texts that were embedded
            quality_tier: Quality tier used for selection
            start_time: Timestamp when embedding started

        Returns:
            dict[str, object]: Dictionary with calculated metrics:
                - latency_ms: Time taken in milliseconds
                - tokens: Number of tokens processed
                - cost: Actual cost in USD
                - tier_name: Name of quality tier used
                - provider_key: Normalized provider name

        """
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Estimate tokens using configured character-to-token ratio
        actual_tokens = sum(len(text) for text in texts) // int(
            self._smart_config.chars_per_token
        )
        actual_cost = actual_tokens * provider.cost_per_token

        # Normalize names for consistent tracking
        tier_name = quality_tier.value if quality_tier else "default"
        provider_key = provider_name or provider.__class__.__name__.lower().replace(
            "provider", ""
        )

        # Update service-level usage statistics
        self.update_usage_stats(
            provider_key,
            selected_model,
            actual_tokens,
            actual_cost,
            tier_name,
        )

        return {
            "latency_ms": latency_ms,
            "tokens": actual_tokens,
            "cost": actual_cost,
            "tier_name": tier_name,
            "provider_key": provider_key,
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
        """Generate embeddings with intelligent provider selection and optimization.

        This is the main entry point for embedding generation. It implements a sophisticated
        workflow including text analysis, caching, smart provider selection, budget validation,
        and comprehensive result tracking.

        Args:
            texts: Text strings to embed
            quality_tier: Quality tier (FAST, BALANCED, BEST) for auto-selection
            provider_name: Explicit provider ("openai" or "fastembed")
            max_cost: Optional maximum cost constraint
            speed_priority: Whether to prioritize speed over quality
            auto_select: Use smart selection (True) or legacy logic (False)
            generate_sparse: Whether to generate sparse embeddings for hybrid search

        Returns:
            Dictionary containing embeddings and comprehensive metadata

        Raises:
            EmbeddingServiceError: If manager not initialized or provider fails

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

        start_time = time.time()

        # Analyze text characteristics for intelligent selection
        text_analysis = self.analyze_text_characteristics(texts)

        # Check cache for single text embeddings (V2 will handle batches)
        if self.cache_manager and len(texts) == 1:
            text = texts[0]

            # Try to retrieve from cache using public API
            if hasattr(self.cache_manager, "get_embedding"):
                cached_embedding = await self.cache_manager.get_embedding(
                    text=text,
                    provider=provider_name or self.config.embedding_provider.value,
                    model=self.config.openai.model
                    if provider_name == "openai"
                    else self.config.fastembed.model,
                    dimensions=self.config.openai.dimensions,
                )

                if cached_embedding is not None:
                    logger.info("Cache hit for embedding")
                    return {
                        "embeddings": [cached_embedding],
                        "provider": provider_name
                        or self.config.embedding_provider.value,
                        "model": self.config.openai.model
                        if provider_name == "openai"
                        else self.config.fastembed.model,
                        "cost": 0.0,  # No cost for cached result
                        "latency_ms": (time.time() - start_time) * 1000,
                        "tokens": 0,
                        "reasoning": "Retrieved from cache",
                        "quality_tier": quality_tier.value
                        if quality_tier
                        else "default",
                        "usage_stats": self.get_usage_report(),
                        "cache_hit": True,
                    }

        # Execute smart provider and model selection
        provider, selected_model, estimated_cost, reasoning = (
            self._select_provider_and_model(
                text_analysis,
                quality_tier,
                provider_name,
                max_cost,
                speed_priority,
                auto_select,
            )
        )

        # Validate budget constraints before proceeding
        self._validate_budget_constraints(estimated_cost)

        logger.info(
            f"Using {provider.__class__.__name__} for {len(texts)} texts"
        )  # TODO: Convert f-string to logging format

        try:
            embeddings = await self._generate_dense_embeddings(provider, texts)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.exception("Dense embedding generation failed: ")
            msg = f"Failed to generate dense embeddings: {e}"
            raise EmbeddingServiceError(msg) from e

        try:
            sparse_embeddings = await self._generate_sparse_embeddings_if_needed(
                provider, texts, generate_sparse
            )
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to generate sparse embeddings: {e}")
            sparse_embeddings = None

        try:
            metrics = self._calculate_metrics_and_update_stats(
                provider, provider_name, selected_model, texts, quality_tier, start_time
            )
        except (ValueError, RuntimeError) as e:
            logger.exception("Failed to calculate metrics: ")
            msg = f"Failed to calculate embedding metrics: {e}"
            raise EmbeddingServiceError(msg) from e

        try:
            await self._cache_embedding_if_applicable(
                texts, embeddings, selected_model, metrics
            )
        except (AttributeError, RuntimeError, ConnectionError) as e:
            logger.warning(f"Failed to cache embedding: {e}")

        try:
            result = self._build_embedding_result(
                embeddings, sparse_embeddings, metrics, selected_model, reasoning
            )
        except (ValueError, RuntimeError) as e:
            logger.exception("Failed to build result: ")
            msg = f"Failed to build embedding result: {e}"
            raise EmbeddingServiceError(msg) from e

        return result

    async def _generate_dense_embeddings(
        self, provider, texts: list[str]
    ) -> list[list[float]]:
        """Generate dense embeddings using the provider."""
        return await provider.generate_embeddings(
            texts,
            batch_size=32,  # Default batch size for optimal performance
        )

    async def _generate_sparse_embeddings_if_needed(
        self, provider, texts: list[str], generate_sparse: bool
    ) -> list[dict] | None:
        """Generate sparse embeddings if requested and supported."""
        if not generate_sparse or not hasattr(provider, "generate_sparse_embeddings"):
            return None

        sparse_embeddings = await provider.generate_sparse_embeddings(texts)
        logger.info(
            f"Generated {len(sparse_embeddings)} sparse embeddings"
        )  # TODO: Convert f-string to logging format
        return sparse_embeddings

    async def _cache_embedding_if_applicable(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        selected_model: str,
        metrics: dict[str, object],
    ) -> None:
        """Cache embedding if conditions are met."""
        if not (self.cache_manager and len(texts) == 1 and len(embeddings) == 1):
            return

        if hasattr(self.cache_manager, "set_embedding"):
            await self.cache_manager.set_embedding(
                text=texts[0],
                model=selected_model,
                embedding=embeddings[0],
                provider=metrics["provider_key"],
                dimensions=len(embeddings[0]),
            )
            logger.info("Cached embedding for future use")

    def _build_embedding_result(
        self,
        embeddings: list[list[float]],
        sparse_embeddings: list[dict] | None,
        metrics: dict[str, object],
        selected_model: str,
        reasoning: str,
    ) -> dict[str, object]:
        """Build comprehensive embedding result with all metadata."""
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

        # Include sparse embeddings if generated
        if sparse_embeddings is not None:
            result["sparse_embeddings"] = sparse_embeddings

        return result

    async def rerank_results(
        self, query: str, results: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Rerank search results using BGE reranker for improved relevance.

        Uses the BGE reranker model to re-score and reorder search results based on
        their semantic relevance to the query. This is particularly effective for
        improving precision in retrieval systems.

        Args:
            query: Search query to compare against
            results: List of search results with 'content' field

        Returns:
            Reranked results sorted by relevance score (highest first)

        """
        if not self._reranker:
            logger.warning("Reranker not available, returning original results")
            return results

        if not results or len(results) <= 1:
            return results

        try:
            # Prepare query-result pairs for reranking
            pairs = [[query, result.get("content", "")] for result in results]

            # Get reranking scores using BGE model
            scores = self._reranker.compute_score(pairs, normalize=True)

            # Handle single result case where compute_score returns a scalar
            if isinstance(scores, int | float):
                scores = [scores]

            # Combine results with scores and sort by relevance
            scored_results = list(zip(results, scores, strict=False))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Extract reordered results
            reranked = [result for result, _ in scored_results]
        except (ValueError, RuntimeError, AttributeError):
            logger.exception("Reranking failed: ")
            # Return original results on failure
            return results
        else:
            logger.info(
                f"Reranked {len(results)} results using {self._reranker_model}"
            )  # TODO: Convert f-string to logging format
            return reranked

    def load_custom_benchmarks(self, benchmark_file: Path | str) -> None:
        """Load custom benchmark configuration from file for dynamic model selection.

        Dynamically loads benchmark configuration from JSON files like custom-benchmarks.json,
        replacing the current benchmarks and smart selection config. This allows for
        runtime adaptation of model selection algorithms based on updated performance data.

        Args:
            benchmark_file: Path to benchmark configuration JSON file

        Raises:
            FileNotFoundError: If benchmark file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            pydantic.ValidationError: If data doesn't match expected schema

        """
        # Load and validate benchmark configuration
        benchmark_path = Path(benchmark_file)
        with benchmark_path.open() as f:
            benchmark_data = json.load(f)

        benchmark_config = Config(**benchmark_data)

        # Update manager's benchmarks and smart selection configuration
        self._benchmarks = benchmark_config.embedding.model_benchmarks
        self._smart_config = benchmark_config.embedding.smart_selection

        logger.info(
            f"Loaded custom benchmarks from {benchmark_path.name} "
            f"with {len(self._benchmarks)} models"
        )

    def estimate_cost(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Estimate embedding generation cost across providers.

        Provides cost estimates to help users make informed decisions about
        provider selection and batch sizing.

        Args:
            texts: List of texts to estimate cost for
            provider_name: Optional specific provider to estimate

        Returns:
            dict[str, dict[str, float]]: Cost estimation per provider:
                - estimated_tokens: Number of tokens estimated
                - cost_per_token: Cost per token in USD
                - total_cost: Total estimated cost in USD

        Raises:
            EmbeddingServiceError: If manager not initialized

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
        """Get comprehensive information about available providers.

        Returns:
            dict[str, dict[str, object]]: Provider information with:
                - model: Model name
                - dimensions: Embedding dimensions
                - cost_per_token: Cost per token in USD
                - max_tokens: Maximum tokens per request

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
        """Select optimal provider using simple heuristics for text characteristics.

        This method provides a simplified provider selection for cases where
        the full smart selection algorithm isn't needed.

        Args:
            text_length: Total character count (estimates ~4 chars per token)
            quality_required: Whether high quality is required
            budget_limit: Optional maximum cost in dollars

        Returns:
            Optimal provider name ("openai" or "fastembed")

        Raises:
            EmbeddingServiceError: If no provider meets constraints

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

        This method performs comprehensive text analysis to inform smart provider
        selection. It evaluates length, complexity, vocabulary diversity, and
        content type to optimize model selection for the specific use case.

        Args:
            texts: List of texts to analyze for characteristics

        Returns:
            TextAnalysis: Analysis results containing:
                - total_length: Combined length of all texts
                - avg_length: Average text length
                - complexity_score: Vocabulary diversity score (0-1)
                - estimated_tokens: Estimated token count
                - text_type: Categorization ("code", "docs", "short", "long")
                - requires_high_quality: Whether high quality embedding is recommended

        """
        if not texts:
            return TextAnalysis(
                total_length=0,
                avg_length=0,
                complexity_score=0.0,
                estimated_tokens=0,
                text_type="empty",
                requires_high_quality=False,
            )

        # Filter out None values and handle edge cases
        valid_texts = [text for text in texts if text is not None]
        if not valid_texts:
            return TextAnalysis(
                total_length=0,
                avg_length=0,
                complexity_score=0.0,
                estimated_tokens=0,
                text_type="empty",
                requires_high_quality=False,
            )

        # Basic text metrics
        total_length = sum(len(text) for text in valid_texts)
        avg_length = total_length // len(valid_texts)
        estimated_tokens = int(total_length / self._smart_config.chars_per_token)

        # Analyze vocabulary complexity and content patterns
        all_words = set()
        total_words = 0
        code_indicators = 0

        for text in valid_texts:
            words = text.lower().split()
            all_words.update(words)
            total_words += len(words)

            # Detect code patterns using configured keywords
            if any(
                keyword in text.lower() for keyword in self._smart_config.code_keywords
            ):
                code_indicators += 1

        # Calculate complexity score based on vocabulary diversity
        if total_words > 0:
            complexity_score = len(all_words) / total_words
            # Normalize to reasonable range (typical diversity is 0.3-0.8)
            complexity_score = min(complexity_score * 1.5, 1.0)
        else:
            complexity_score = 0.0

        # Determine text type using configured thresholds
        is_code = code_indicators / len(texts) > 0.3
        if is_code:
            text_type = "code"
        elif avg_length > self._smart_config.long_text_threshold:
            text_type = "long"
        elif avg_length < self._smart_config.short_text_threshold:
            text_type = "short"
        else:
            text_type = "docs"

        # Determine if high quality embeddings are recommended
        requires_high_quality = is_code or complexity_score > 0.7 or avg_length > 1500

        return TextAnalysis(
            total_length=total_length,
            avg_length=avg_length,
            complexity_score=complexity_score,
            estimated_tokens=estimated_tokens,
            text_type=text_type,
            requires_high_quality=requires_high_quality,
        )

    def get_smart_provider_recommendation(
        self,
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
    ) -> dict[str, object]:
        """Get intelligent provider recommendation using comprehensive analysis.

        This is the core of the smart selection system. It evaluates all available
        models against the text characteristics and user constraints to find the
        optimal balance of quality, speed, and cost.

        Args:
            text_analysis: Analysis of text characteristics including length,
                complexity and text type
            quality_tier: Optional quality tier override (FAST, BALANCED, BEST)
            max_cost: Optional maximum cost constraint in USD
            speed_priority: Whether to prioritize speed over quality

        Returns:
            dict[str, object]: Recommendation with:
                - provider: Provider name ("openai" or "fastembed")
                - model: Specific model name
                - estimated_cost: Cost estimate in USD
                - score: Model score (0-100)
                - reasoning: Human-readable explanation
                - alternatives: Top 2 alternative options

        Raises:
            EmbeddingServiceError: If no models meet constraints or manager not initialized

        """
        if not self._initialized:
            msg = "Manager not initialized"
            raise EmbeddingServiceError(msg)

        # Evaluate all available model combinations
        candidates = []
        for provider_name, provider in self.providers.items():
            if provider_name == "openai":
                models = ["text-embedding-3-small", "text-embedding-3-large"]
            else:  # fastembed
                models = [provider.model_name]

            for model in models:
                if model in self._benchmarks:
                    benchmark = self._benchmarks[model]
                    cost = text_analysis.estimated_tokens * (
                        benchmark.cost_per_million_tokens / 1_000_000
                    )

                    # Apply cost constraint
                    if max_cost and cost > max_cost:
                        continue

                    # Calculate comprehensive score for this model
                    score = self._calculate_model_score(
                        benchmark, text_analysis, quality_tier, speed_priority
                    )

                    candidates.append(
                        {
                            "provider": provider_name,
                            "model": model,
                            "benchmark": benchmark,
                            "estimated_cost": cost,
                            "score": score,
                        }
                    )

        if not candidates:
            msg = (
                f"No models available for constraints: max_cost={max_cost}, "
                f"tokens={text_analysis.estimated_tokens}"
            )
            raise EmbeddingServiceError(msg)

        # Sort by score (higher is better) and select best option
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Generate human-readable reasoning for the selection
        reasoning = self._generate_selection_reasoning(
            best, text_analysis, quality_tier, speed_priority
        )

        return {
            "provider": best["provider"],
            "model": best["model"],
            "estimated_cost": best["estimated_cost"],
            "score": best["score"],
            "reasoning": reasoning,
            "alternatives": candidates[1:3],  # Show top 2 alternatives
        }

    def _calculate_model_score(
        self,
        benchmark: dict[str, Any],
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> float:
        """Calculate comprehensive score for model selection using weighted criteria.

        This scoring algorithm balances multiple factors to find the optimal model
        for the specific use case. It considers quality metrics, performance
        characteristics, cost efficiency, and user preferences.

        Args:
            benchmark: Model benchmark data with quality, speed, and cost metrics
            text_analysis: Text analysis results containing type and complexity
            quality_tier: Quality tier preference (FAST, BALANCED, BEST)
            speed_priority: Speed priority flag to weight speed higher

        Returns:
            float: Score between 0-100, where higher is better. Score combines:
                - Base quality score with configured weight
                - Speed score based on latency thresholds
                - Cost efficiency score (local models get max score)
                - Quality tier bonuses/penalties
                - Text type specific bonuses

        """
        score = 0.0

        # Quality scoring: Start with base quality metrics from benchmarks
        # Higher quality models (better on evaluation datasets) get higher base scores
        score += benchmark.quality_score * self._smart_config.quality_weight

        # Speed scoring: Favor models with lower latency for responsive applications
        # Dynamic weighting: speed becomes more important when explicitly prioritized
        speed_weight = 0.5 if speed_priority else self._smart_config.speed_weight
        speed_score = max(
            0,
            (self._smart_config.speed_balanced_threshold - benchmark.avg_latency_ms)
            / self._smart_config.speed_balanced_threshold
            * 100,
        )
        score += speed_score * speed_weight

        # Cost efficiency scoring: Balance quality with economic considerations
        # Local models get maximum cost score (zero cost), others scored inversely to cost
        cost_weight = 0.1 if speed_priority else self._smart_config.cost_weight
        if benchmark.cost_per_million_tokens == 0:  # Local model
            cost_score = 100
        else:
            # Inverse cost scoring: lower cost = higher score
            # Normalized against expensive threshold for relative comparison
            cost_score = max(
                0,
                (
                    self._smart_config.cost_expensive_threshold
                    - benchmark.cost_per_million_tokens
                )
                / self._smart_config.cost_expensive_threshold
                * 100,
            )
        score += cost_score * cost_weight

        # Quality tier bonuses: Align selection with user's explicit preferences
        # These bonuses strongly influence the final selection to match user intent
        if quality_tier == QualityTier.FAST and benchmark.cost_per_million_tokens == 0:
            # FAST tier prioritizes speed: big bonus for local models (instant processing)
            score += 25
        elif quality_tier == QualityTier.BEST:
            # BEST tier prioritizes quality: reward high-performance models heavily
            if benchmark.quality_score > self._smart_config.quality_best_threshold:
                score += 40  # Very strong bonus for top-tier models
            elif (
                benchmark.quality_score > self._smart_config.quality_balanced_threshold
            ):
                score += 30  # Good models still get significant bonus
            else:
                # Penalize low-quality models when user explicitly wants the best
                score -= 10
        elif quality_tier == QualityTier.BALANCED:
            # BALANCED tier seeks optimal cost/quality ratio
            if benchmark.cost_per_million_tokens == 0:
                score += 10  # Local models offer perfect cost efficiency
            elif (
                benchmark.cost_per_million_tokens
                < self._smart_config.cost_cheap_threshold
            ):
                score += 15  # Reward cost-effective cloud options

        # Content-specific optimizations: Adapt selection to text characteristics
        # Different content types have different embedding quality requirements
        if (
            text_analysis.text_type == "code"
            and benchmark.quality_score > self._smart_config.quality_best_threshold
        ):
            # Code embeddings require high precision for semantic code search
            # Syntactic nuances and technical terminology need quality models
            score += 5
        elif text_analysis.text_type == "short" and benchmark.avg_latency_ms < 60:
            # Short texts (queries, titles) benefit from immediate response
            # User experience is latency-sensitive for interactive applications
            score += 5

        return float(min(score, 100))

    def _generate_selection_reasoning(
        self,
        selection: dict[str, object],
        text_analysis: TextAnalysis,
        _quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> str:
        """Generate human-readable reasoning for model selection.

        Creates clear explanations for why a particular model was selected,
        helping users understand the decision-making process.

        Args:
            selection: Selected model info dictionary containing benchmark data
            text_analysis: Text analysis results with complexity and type
            quality_tier: Quality tier preference if specified
            speed_priority: Speed priority flag

        Returns:
            str: Human-readable reasoning explaining the selection

        """
        benchmark = selection["benchmark"]
        reasons = []

        # Primary selection reason
        if speed_priority:
            reasons.append(f"Speed prioritized: {benchmark.avg_latency_ms}ms latency")
        elif benchmark.cost_per_million_tokens == 0:
            reasons.append("Local model for cost efficiency")
        elif benchmark.quality_score > 90:
            reasons.append(f"High quality model: {benchmark.quality_score}/100 score")

        # Text-specific considerations
        if text_analysis.text_type == "code":
            reasons.append("Code detection: using high-accuracy model")
        elif text_analysis.requires_high_quality:
            reasons.append("Complex text detected: quality prioritized")

        # Cost information
        if selection["estimated_cost"] > 0:
            reasons.append(f"Cost: ${selection['estimated_cost']:.4f}")
        else:
            reasons.append("Zero cost (local processing)")

        return "; ".join(reasons)

    def check_budget_constraints(self, estimated_cost: float) -> dict[str, object]:
        """Check if request is within budget constraints with detailed reporting.

        Validates the estimated cost against daily budget limits and provides
        warnings when approaching thresholds. This helps prevent unexpected
        overspending while providing visibility into usage patterns.

        Args:
            estimated_cost: Estimated cost for the request in USD

        Returns:
            dict[str, object]: Budget check result with:
                - within_budget: Whether request is within budget
                - warnings: List of warning messages
                - daily_usage: Current daily cost
                - estimated_total: Daily usage plus estimated cost
                - budget_limit: Daily budget limit if set

        """
        result = {
            "within_budget": True,
            "warnings": [],
            "daily_usage": self.usage_stats.daily_cost,
            "estimated_total": self.usage_stats.daily_cost + estimated_cost,
            "budget_limit": self.budget_limit,
        }

        if self.budget_limit:
            # Check hard budget limit
            if self.usage_stats.daily_cost + estimated_cost > self.budget_limit:
                result["within_budget"] = False
                result["warnings"].append(
                    f"Request would exceed daily budget: "
                    f"${self.usage_stats.daily_cost + estimated_cost:.4f} > ${self.budget_limit:.2f}"
                )

            # Check warning thresholds
            usage_percent = (
                self.usage_stats.daily_cost + estimated_cost
            ) / self.budget_limit
            if usage_percent > self._smart_config.budget_critical_threshold:
                result["warnings"].append(
                    f"Budget usage > {int(self._smart_config.budget_critical_threshold * 100)}%"
                )
            elif usage_percent > self._smart_config.budget_warning_threshold:
                result["warnings"].append(
                    f"Budget usage > {int(self._smart_config.budget_warning_threshold * 100)}%"
                )

        return result

    def update_usage_stats(
        self, provider: str, _model: str, tokens: int, cost: float, tier: str
    ) -> None:
        """Update comprehensive usage statistics with automatic daily reset.

        Tracks cumulative usage across all dimensions for analytics and
        budget management. Daily statistics are automatically reset based
        on system date changes.

        Args:
            provider: Provider name ("openai" or "fastembed")
            model: Model name that was used
            tokens: Number of tokens processed
            cost: Cost incurred in USD
            tier: Quality tier used ("fast", "balanced", "best", or "default")

        Note:
            Daily cost is reset when a new day is detected based on the
            system date. All other statistics are cumulative.

        """
        # Update cumulative statistics
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        self.usage_stats.daily_cost += cost
        self.usage_stats.requests_by_tier[tier] += 1
        self.usage_stats.requests_by_provider[provider] += 1

        # Reset daily stats if new day detected
        today = time.strftime("%Y-%m-%d")
        if self.usage_stats.last_reset_date != today:
            self.usage_stats.daily_cost = cost
            self.usage_stats.last_reset_date = today

    def get_usage_report(self) -> dict[str, object]:
        """Get comprehensive usage report with analytics and budget information.

        Returns:
            dict[str, object]: Usage statistics with:
                - summary: Total requests, tokens, costs, and averages
                - by_tier: Request counts by quality tier
                - by_provider: Request counts by provider
                - budget: Daily limit, usage, and remaining budget

        """
        total_cost = self.usage_stats.total_cost
        total_requests = self.usage_stats.total_requests

        return {
            "summary": {
                "total_requests": total_requests,
                "total_tokens": self.usage_stats.total_tokens,
                "total_cost": total_cost,
                "daily_cost": self.usage_stats.daily_cost,
                "avg_cost_per_request": total_cost / max(total_requests, 1),
                "avg_tokens_per_request": self.usage_stats.total_tokens
                / max(total_requests, 1),
            },
            "by_tier": dict(self.usage_stats.requests_by_tier),
            "by_provider": dict(self.usage_stats.requests_by_provider),
            "budget": {
                "daily_limit": self.budget_limit,
                "daily_usage": self.usage_stats.daily_cost,
                "remaining": (
                    self.budget_limit - self.usage_stats.daily_cost
                    if self.budget_limit
                    else None
                ),
            },
        }
