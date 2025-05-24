"""Embedding manager with smart provider selection."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import APIConfig
from ..errors import EmbeddingServiceError
from .base import EmbeddingProvider
from .fastembed_provider import FastEmbedProvider
from .openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

# Smart Selection Constants
CHARS_PER_TOKEN = 4  # Approximate character-to-token ratio
CODE_KEYWORDS = {
    "def",
    "class",
    "import",
    "return",
    "if",
    "else",
    "for",
    "while",
    "try",
    "except",
    "function",
    "const",
    "let",
    "var",
    "public",
    "private",
}

# Scoring weights for smart selection
QUALITY_WEIGHT = 0.4
SPEED_WEIGHT = 0.3
COST_WEIGHT = 0.3

# Quality scoring parameters
QUALITY_SCORE_FAST_THRESHOLD = 60.0
QUALITY_SCORE_BALANCED_THRESHOLD = 75.0
QUALITY_SCORE_BEST_THRESHOLD = 85.0

# Speed scoring parameters
SPEED_FAST_THRESHOLD = 500.0  # tokens/second
SPEED_BALANCED_THRESHOLD = 200.0
SPEED_SLOW_THRESHOLD = 100.0

# Cost scoring parameters (per million tokens)
COST_CHEAP_THRESHOLD = 50.0
COST_MODERATE_THRESHOLD = 100.0
COST_EXPENSIVE_THRESHOLD = 200.0

# Budget management
BUDGET_WARNING_THRESHOLD = 0.8  # 80%
BUDGET_CRITICAL_THRESHOLD = 0.9  # 90%

# Text analysis parameters
COMPLEXITY_VOCAB_DIVERSITY_FACTOR = 20.0
COMPLEXITY_AVG_WORD_LENGTH_FACTOR = 20.0
SHORT_TEXT_THRESHOLD = 100  # characters
LONG_TEXT_THRESHOLD = 2000  # characters


class QualityTier(Enum):
    """Embedding quality tiers."""

    FAST = "fast"  # Local models, fastest
    BALANCED = "balanced"  # Balance of speed and quality
    BEST = "best"  # Highest quality, may be slower/costlier


@dataclass
class ModelBenchmark:
    """Performance benchmark for embedding models."""

    model_name: str
    provider: str
    avg_latency_ms: float
    quality_score: float  # 0-100 based on retrieval accuracy
    tokens_per_second: float
    cost_per_million_tokens: float
    max_context_length: int
    embedding_dimensions: int


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
    """Manager for smart embedding provider selection."""

    def __init__(self, config: APIConfig, budget_limit: float | None = None):
        """Initialize embedding manager.

        Args:
            config: API configuration
            budget_limit: Optional daily budget limit in USD
        """
        self.config = config
        self.providers: dict[str, EmbeddingProvider] = {}
        self._initialized = False
        self.budget_limit = budget_limit
        self.usage_stats = UsageStats()

        # Model benchmarks (initialized with research-backed values)
        self._benchmarks: dict[str, ModelBenchmark] = {
            "text-embedding-3-small": ModelBenchmark(
                model_name="text-embedding-3-small",
                provider="openai",
                avg_latency_ms=78,
                quality_score=85,
                tokens_per_second=12800,
                cost_per_million_tokens=20.0,
                max_context_length=8191,
                embedding_dimensions=1536,
            ),
            "text-embedding-3-large": ModelBenchmark(
                model_name="text-embedding-3-large",
                provider="openai",
                avg_latency_ms=120,
                quality_score=92,
                tokens_per_second=8300,
                cost_per_million_tokens=130.0,
                max_context_length=8191,
                embedding_dimensions=3072,
            ),
            "BAAI/bge-small-en-v1.5": ModelBenchmark(
                model_name="BAAI/bge-small-en-v1.5",
                provider="fastembed",
                avg_latency_ms=45,
                quality_score=78,
                tokens_per_second=22000,
                cost_per_million_tokens=0.0,
                max_context_length=512,
                embedding_dimensions=384,
            ),
            "BAAI/bge-large-en-v1.5": ModelBenchmark(
                model_name="BAAI/bge-large-en-v1.5",
                provider="fastembed",
                avg_latency_ms=89,
                quality_score=88,
                tokens_per_second=11000,
                cost_per_million_tokens=0.0,
                max_context_length=512,
                embedding_dimensions=1024,
            ),
        }

        # Dynamic quality tier mappings
        self._tier_providers = {
            QualityTier.FAST: "fastembed",
            QualityTier.BALANCED: "fastembed",  # Dynamic based on config
            QualityTier.BEST: "openai",
        }

    async def initialize(self) -> None:
        """Initialize available providers."""
        if self._initialized:
            return

        # Initialize OpenAI provider if API key available
        if self.config.openai_api_key:
            try:
                provider = OpenAIEmbeddingProvider(
                    api_key=self.config.openai_api_key,
                    model_name=self.config.openai_model,
                    dimensions=self.config.openai_dimensions,
                )
                await provider.initialize()
                self.providers["openai"] = provider
                logger.info(
                    f"Initialized OpenAI provider with {self.config.openai_model}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")

        # Initialize FastEmbed provider if enabled
        if self.config.enable_local_embeddings:
            try:
                provider = FastEmbedProvider(
                    model_name=self.config.local_embedding_model
                )
                await provider.initialize()
                self.providers["fastembed"] = provider
                logger.info(
                    f"Initialized FastEmbed provider with "
                    f"{self.config.local_embedding_model}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize FastEmbed provider: {e}")

        if not self.providers:
            raise EmbeddingServiceError(
                "No embedding providers available. "
                "Please configure OpenAI API key or enable local embeddings."
            )

        self._initialized = True
        logger.info(
            f"Embedding manager initialized with {len(self.providers)} providers"
        )

    async def cleanup(self) -> None:
        """Cleanup all providers."""
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(f"Cleaned up {name} provider")
            except Exception as e:
                logger.error(f"Error cleaning up {name} provider: {e}")

        self.providers.clear()
        self._initialized = False

    def _select_provider_and_model(
        self,
        text_analysis: "TextAnalysis",
        quality_tier: QualityTier | None,
        provider_name: str | None,
        max_cost: float | None,
        speed_priority: bool,
        auto_select: bool,
    ) -> tuple[EmbeddingProvider, str, float, str]:
        """Select provider and model based on parameters.

        Returns:
            Tuple of (provider, model_name, estimated_cost, reasoning)
        """
        if auto_select and not provider_name:
            # Get smart recommendation
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
            # Use legacy selection logic
            reasoning = "Legacy selection"
            estimated_cost = 0.0
            selected_model = None

            # If specific provider requested, estimate cost
            if provider_name and provider_name in self.providers:
                provider = self.providers[provider_name]
                estimated_cost = (
                    text_analysis.estimated_tokens * provider.cost_per_token
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
        """Get provider instance based on name or quality tier."""
        if provider_name:
            provider = self.providers.get(provider_name)
            if not provider:
                raise EmbeddingServiceError(
                    f"Provider '{provider_name}' not available. "
                    f"Available: {list(self.providers.keys())}"
                )
        elif quality_tier:
            preferred = self._tier_providers.get(quality_tier)
            provider = self.providers.get(preferred)
            if not provider:
                provider = next(iter(self.providers.values()))
                logger.warning(
                    f"Preferred provider for {quality_tier.value} not available, "
                    f"using {provider.__class__.__name__}"
                )
        else:
            provider = self.providers.get(self.config.preferred_embedding_provider)
            if not provider:
                provider = next(iter(self.providers.values()))

        return provider

    def _validate_budget_constraints(self, estimated_cost: float) -> None:
        """Check budget constraints and raise error if violated."""
        budget_check = self.check_budget_constraints(estimated_cost)
        if not budget_check["within_budget"]:
            raise EmbeddingServiceError(
                f"Budget constraint violated: {budget_check['warnings'][0]}"
            )

        # Log warnings if any
        for warning in budget_check["warnings"]:
            logger.warning(f"Budget warning: {warning}")

    def _calculate_metrics_and_update_stats(
        self,
        provider: EmbeddingProvider,
        provider_name: str | None,
        selected_model: str,
        texts: list[str],
        quality_tier: QualityTier | None,
        start_time: float,
    ) -> dict[str, Any]:
        """Calculate metrics and update usage statistics.

        Returns:
            Dictionary with calculated metrics
        """
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        actual_tokens = sum(len(text) for text in texts) // CHARS_PER_TOKEN
        actual_cost = actual_tokens * provider.cost_per_token

        # Update usage statistics
        tier_name = quality_tier.value if quality_tier else "default"
        provider_key = provider_name or provider.__class__.__name__.lower().replace(
            "provider", ""
        )

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
    ) -> dict[str, Any]:
        """Generate embeddings with smart provider selection.

        Args:
            texts: Text strings to embed
            quality_tier: Quality tier (FAST, BALANCED, BEST) for auto-selection
            provider_name: Explicit provider ("openai" or "fastembed")
            max_cost: Optional maximum cost constraint
            speed_priority: Whether to prioritize speed over quality
            auto_select: Use smart selection (True) or legacy logic (False)

        Returns:
            Dictionary containing embeddings and metadata

        Raises:
            EmbeddingServiceError: If manager not initialized or provider fails
        """
        if not self._initialized:
            raise EmbeddingServiceError("Manager not initialized")

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

        # Analyze text characteristics
        text_analysis = self.analyze_text_characteristics(texts)

        # Select provider and model
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

        # Validate budget constraints
        self._validate_budget_constraints(estimated_cost)

        logger.info(f"Using {provider.__class__.__name__} for {len(texts)} texts")

        try:
            # Generate embeddings
            embeddings = await provider.generate_embeddings(
                texts, batch_size=self.config.openai_batch_size
            )

            # Calculate metrics and update statistics
            metrics = self._calculate_metrics_and_update_stats(
                provider, provider_name, selected_model, texts, quality_tier, start_time
            )

            return {
                "embeddings": embeddings,
                "provider": metrics["provider_key"],
                "model": selected_model,
                "cost": metrics["cost"],
                "latency_ms": metrics["latency_ms"],
                "tokens": metrics["tokens"],
                "reasoning": reasoning,
                "quality_tier": metrics["tier_name"],
                "usage_stats": self.get_usage_report(),
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def estimate_cost(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> dict[str, float]:
        """Estimate embedding generation cost.

        Args:
            texts: List of texts
            provider_name: Optional specific provider

        Returns:
            Cost estimation details
        """
        if not self._initialized:
            raise EmbeddingServiceError("Manager not initialized")

        # Simple token estimation (avg 4 chars per token)
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

    def get_provider_info(self) -> dict[str, dict]:
        """Get information about available providers.

        Returns:
            Provider information
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
        """Select optimal provider based on text size and constraints.

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
            raise EmbeddingServiceError("Manager not initialized")

        # Simple heuristic for provider selection
        estimated_tokens = text_length / 4

        candidates = []
        for name, provider in self.providers.items():
            cost = estimated_tokens * provider.cost_per_token

            # Check budget constraint
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
            raise EmbeddingServiceError(
                f"No provider available within budget {budget_limit}"
            )

        # Sort by preference
        if quality_required and "openai" in self.providers:
            return "openai"

        # Prefer local for small texts
        if text_length < 10000:
            for candidate in candidates:
                if candidate["is_local"]:
                    return candidate["name"]

        # Return cheapest option
        candidates.sort(key=lambda x: x["cost"])
        return candidates[0]["name"]

    def analyze_text_characteristics(self, texts: list[str]) -> TextAnalysis:
        """Analyze text characteristics for smart model selection.

        Args:
            texts: List of texts to analyze

        Returns:
            TextAnalysis with characteristics
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

        total_length = sum(len(text) for text in texts)
        avg_length = total_length // len(texts)
        estimated_tokens = int(total_length / CHARS_PER_TOKEN)

        # Analyze complexity (vocabulary diversity)
        all_words = set()
        total_words = 0
        code_indicators = 0

        for text in texts:
            words = text.lower().split()
            all_words.update(words)
            total_words += len(words)

            # Check for code patterns using defined keywords
            if any(keyword in text.lower() for keyword in CODE_KEYWORDS):
                code_indicators += 1

        # Complexity score based on vocabulary diversity (cap at reasonable level)
        if total_words > 0:
            complexity_score = len(all_words) / total_words
            # Normalize to reasonable range (typical diversity is 0.3-0.8)
            complexity_score = min(complexity_score * 1.5, 1.0)
        else:
            complexity_score = 0.0

        # Determine text type
        is_code = code_indicators / len(texts) > 0.3
        if is_code:
            text_type = "code"
        elif avg_length > LONG_TEXT_THRESHOLD:
            text_type = "long"
        elif avg_length < SHORT_TEXT_THRESHOLD:
            text_type = "short"
        else:
            text_type = "docs"

        # High quality requirements for code or complex text
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
    ) -> dict[str, Any]:
        """Get smart provider recommendation based on text analysis.

        Args:
            text_analysis: Analysis of text characteristics
            quality_tier: Optional quality tier override
            max_cost: Optional maximum cost constraint
            speed_priority: Whether to prioritize speed over quality

        Returns:
            Recommendation with provider, model, reasoning
        """
        if not self._initialized:
            raise EmbeddingServiceError("Manager not initialized")

        # Get available models with their benchmarks
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

                    # Check constraints
                    if max_cost and cost > max_cost:
                        continue

                    # For context length, chunk if needed rather than exclude entirely
                    if text_analysis.estimated_tokens > benchmark.max_context_length:
                        # Add penalty for requiring chunking but don't exclude
                        pass

                    candidates.append(
                        {
                            "provider": provider_name,
                            "model": model,
                            "benchmark": benchmark,
                            "estimated_cost": cost,
                            "score": self._calculate_model_score(
                                benchmark, text_analysis, quality_tier, speed_priority
                            ),
                        }
                    )

        if not candidates:
            raise EmbeddingServiceError(
                f"No models available for constraints: max_cost={max_cost}, "
                f"tokens={text_analysis.estimated_tokens}"
            )

        # Sort by score (higher is better)
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
            "alternatives": candidates[1:3],  # Show top 2 alternatives
        }

    def _calculate_model_score(
        self,
        benchmark: ModelBenchmark,
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> float:
        """Calculate score for model selection.

        Args:
            benchmark: Model benchmark data
            text_analysis: Text analysis results
            quality_tier: Quality tier preference
            speed_priority: Speed priority flag

        Returns:
            Score (0-100, higher is better)
        """
        score = 0.0

        # Base quality score
        score += benchmark.quality_score * QUALITY_WEIGHT

        # Speed score (higher weight if speed priority)
        speed_weight = 0.5 if speed_priority else SPEED_WEIGHT
        speed_score = max(
            0,
            (SPEED_BALANCED_THRESHOLD - benchmark.avg_latency_ms)
            / SPEED_BALANCED_THRESHOLD
            * 100,
        )
        score += speed_score * speed_weight

        # Cost efficiency score (lower weight if speed priority)
        cost_weight = 0.1 if speed_priority else COST_WEIGHT
        if benchmark.cost_per_million_tokens == 0:  # Local model
            cost_score = 100
        else:
            # Lower cost = higher score
            cost_score = max(
                0,
                (COST_EXPENSIVE_THRESHOLD - benchmark.cost_per_million_tokens)
                / COST_EXPENSIVE_THRESHOLD
                * 100,
            )
        score += cost_score * cost_weight

        # Quality tier bonus (strong influence on selection)
        if quality_tier == QualityTier.FAST and benchmark.cost_per_million_tokens == 0:
            score += 25  # Strong bonus for local models in FAST tier
        elif quality_tier == QualityTier.BEST:
            if benchmark.quality_score > QUALITY_SCORE_BEST_THRESHOLD:
                score += 40  # Very strong bonus for high-quality models in BEST tier
            elif benchmark.quality_score > QUALITY_SCORE_BALANCED_THRESHOLD:
                score += 30  # Strong bonus for good quality models
            else:
                score -= 10  # Penalty for lower quality in BEST tier
        elif quality_tier == QualityTier.BALANCED:
            if benchmark.cost_per_million_tokens == 0:
                score += 10  # Bonus for local in balanced
            elif benchmark.cost_per_million_tokens < COST_CHEAP_THRESHOLD:
                score += 15  # Bonus for cost-effective options

        # Text type specific bonuses
        if (
            text_analysis.text_type == "code"
            and benchmark.quality_score > QUALITY_SCORE_BEST_THRESHOLD
        ):
            score += 5  # Code needs high quality
        elif text_analysis.text_type == "short" and benchmark.avg_latency_ms < 60:
            score += 5  # Short text benefits from speed

        return float(min(score, 100))

    def _generate_selection_reasoning(
        self,
        selection: dict[str, Any],
        text_analysis: TextAnalysis,
        quality_tier: QualityTier | None,
        speed_priority: bool,
    ) -> str:
        """Generate human-readable reasoning for selection.

        Args:
            selection: Selected model info
            text_analysis: Text analysis results
            quality_tier: Quality tier preference
            speed_priority: Speed priority flag

        Returns:
            Reasoning string
        """
        benchmark = selection["benchmark"]
        reasons = []

        # Primary reason
        if speed_priority:
            reasons.append(f"Speed prioritized: {benchmark.avg_latency_ms}ms latency")
        elif benchmark.cost_per_million_tokens == 0:
            reasons.append("Local model for cost efficiency")
        elif benchmark.quality_score > 90:
            reasons.append(f"High quality model: {benchmark.quality_score}/100 score")

        # Text-specific reasons
        if text_analysis.text_type == "code":
            reasons.append("Code detection: using high-accuracy model")
        elif text_analysis.requires_high_quality:
            reasons.append("Complex text detected: quality prioritized")

        # Cost consideration
        if selection["estimated_cost"] > 0:
            reasons.append(f"Cost: ${selection['estimated_cost']:.4f}")
        else:
            reasons.append("Zero cost (local processing)")

        return "; ".join(reasons)

    def check_budget_constraints(self, estimated_cost: float) -> dict[str, Any]:
        """Check if request is within budget constraints.

        Args:
            estimated_cost: Estimated cost for the request

        Returns:
            Budget check result with warnings
        """
        result = {
            "within_budget": True,
            "warnings": [],
            "daily_usage": self.usage_stats.daily_cost,
            "estimated_total": self.usage_stats.daily_cost + estimated_cost,
            "budget_limit": self.budget_limit,
        }

        if self.budget_limit:
            if self.usage_stats.daily_cost + estimated_cost > self.budget_limit:
                result["within_budget"] = False
                result["warnings"].append(
                    f"Request would exceed daily budget: "
                    f"${self.usage_stats.daily_cost + estimated_cost:.4f} > ${self.budget_limit:.2f}"
                )

            # Warnings at configured thresholds
            usage_percent = (
                self.usage_stats.daily_cost + estimated_cost
            ) / self.budget_limit
            if usage_percent > BUDGET_CRITICAL_THRESHOLD:
                result["warnings"].append(
                    f"Budget usage > {int(BUDGET_CRITICAL_THRESHOLD * 100)}%"
                )
            elif usage_percent > BUDGET_WARNING_THRESHOLD:
                result["warnings"].append(
                    f"Budget usage > {int(BUDGET_WARNING_THRESHOLD * 100)}%"
                )

        return result

    def update_usage_stats(
        self, provider: str, model: str, tokens: int, cost: float, tier: str
    ) -> None:
        """Update usage statistics.

        Args:
            provider: Provider name
            model: Model name
            tokens: Number of tokens processed
            cost: Cost incurred
            tier: Quality tier used
        """
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        self.usage_stats.daily_cost += cost
        self.usage_stats.requests_by_tier[tier] += 1
        self.usage_stats.requests_by_provider[provider] += 1

        # Reset daily stats if new day
        today = time.strftime("%Y-%m-%d")
        if self.usage_stats.last_reset_date != today:
            self.usage_stats.daily_cost = cost
            self.usage_stats.last_reset_date = today

    def get_usage_report(self) -> dict[str, Any]:
        """Get comprehensive usage report.

        Returns:
            Usage statistics and insights
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
