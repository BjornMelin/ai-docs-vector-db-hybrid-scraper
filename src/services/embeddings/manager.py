"""Embedding manager with smart provider selection."""

import logging
from enum import Enum

from ..config import APIConfig
from ..errors import EmbeddingServiceError
from .base import EmbeddingProvider
from .fastembed_provider import FastEmbedProvider
from .openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Embedding quality tiers."""

    FAST = "fast"  # Local models, fastest
    BALANCED = "balanced"  # Balance of speed and quality
    BEST = "best"  # Highest quality, may be slower/costlier


class EmbeddingManager:
    """Manager for smart embedding provider selection."""

    def __init__(self, config: APIConfig):
        """Initialize embedding manager.

        Args:
            config: API configuration
        """
        self.config = config
        self.providers: dict[str, EmbeddingProvider] = {}
        self._initialized = False

        # Quality tier mappings
        self._tier_providers = {
            QualityTier.FAST: "fastembed",
            QualityTier.BALANCED: "fastembed",  # Can be changed to OpenAI
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

    async def generate_embeddings(
        self,
        texts: list[str],
        quality_tier: QualityTier | None = None,
        provider_name: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with smart provider selection.

        Args:
            texts: List of texts to embed
            quality_tier: Optional quality tier for selection
            provider_name: Optional specific provider name

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise EmbeddingServiceError("Manager not initialized")

        # Select provider
        if provider_name:
            # Use specific provider
            provider = self.providers.get(provider_name)
            if not provider:
                raise EmbeddingServiceError(
                    f"Provider '{provider_name}' not available. "
                    f"Available: {list(self.providers.keys())}"
                )
        # Select based on quality tier or default
        elif quality_tier:
            preferred = self._tier_providers.get(quality_tier)
            provider = self.providers.get(preferred)

            # Fallback if preferred not available
            if not provider:
                provider = next(iter(self.providers.values()))
                logger.warning(
                    f"Preferred provider for {quality_tier.value} not available, "
                    f"using {provider.__class__.__name__}"
                )
        else:
            # Use configured preference
            provider = self.providers.get(self.config.preferred_embedding_provider)
            if not provider:
                provider = next(iter(self.providers.values()))

        logger.info(f"Using {provider.__class__.__name__} for {len(texts)} texts")

        # Generate embeddings
        return await provider.generate_embeddings(
            texts, batch_size=self.config.openai_batch_size
        )

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
        """Get optimal provider based on requirements.

        Args:
            text_length: Total text length
            quality_required: Whether high quality is required
            budget_limit: Optional budget limit

        Returns:
            Optimal provider name
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
