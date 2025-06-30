"""Embedding manager service coordinator."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from dependency_injector.wiring import Provide, inject

from src.infrastructure.container import ApplicationContainer
from src.services.errors import EmbeddingServiceError


if TYPE_CHECKING:
    from src.config import Config
    from src.infrastructure.client_manager import ClientManager
    from src.services.embeddings.manager import EmbeddingManager as CoreEmbeddingManager

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Focused manager for embedding provider coordination.

    Wraps and coordinates the core EmbeddingManager with
    smart provider selection and embedding generation.
    """

    def __init__(self):
        """Initialize embedding manager."""
        self._core_manager: CoreEmbeddingManager | None = None
        self._initialized = False

    @inject
    async def initialize(
        self,
        config: "Config" = Provide[ApplicationContainer.config],
        client_manager: "ClientManager" = Provide[ApplicationContainer.client_manager],
    ) -> None:
        """Initialize embedding manager using dependency injection.

        Args:
            config: Configuration from DI container
            client_manager: Client manager from DI container
        """
        if self._initialized:
            return

        try:
            from src.services.embeddings.manager import EmbeddingManager as CoreManager

            self._core_manager = CoreManager(
                config=config,
                client_manager=client_manager,
            )
            await self._core_manager.initialize()

            self._initialized = True
            logger.info("EmbeddingManager service initialized")

        except Exception as e:
            logger.exception(
                f"Failed to initialize EmbeddingManager: {e}"
            )  # TODO: Convert f-string to logging format
            msg = f"Failed to initialize embedding manager: {e}"
            raise EmbeddingServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup embedding manager resources."""
        if self._core_manager:
            await self._core_manager.cleanup()
            self._core_manager = None

        self._initialized = False
        logger.info("EmbeddingManager service cleaned up")

    async def generate_embeddings(
        self,
        texts: list[str],
        quality_tier: str | None = None,
        provider_name: str | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
        auto_select: bool = True,
        generate_sparse: bool = False,
    ) -> dict[str, Any]:
        """Generate embeddings with smart provider selection.

        Args:
            texts: Text strings to embed
            quality_tier: Quality tier (FAST, BALANCED, BEST)
            provider_name: Explicit provider (openai or fastembed)
            max_cost: Optional maximum cost constraint
            speed_priority: Whether to prioritize speed over quality
            auto_select: Use smart selection or legacy logic
            generate_sparse: Whether to generate sparse embeddings

        Returns:
            Dictionary containing embeddings and metadata

        Raises:
            EmbeddingServiceError: If manager not initialized or generation fails
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        try:
            # Convert quality tier string to enum if provided
            tier = None
            if quality_tier:
                from src.services.embeddings.manager import QualityTier

                tier_map = {
                    "FAST": QualityTier.FAST,
                    "BALANCED": QualityTier.BALANCED,
                    "BEST": QualityTier.BEST,
                }
                tier = tier_map.get(quality_tier.upper())

            return await self._core_manager.generate_embeddings(
                texts=texts,
                quality_tier=tier,
                provider_name=provider_name,
                max_cost=max_cost,
                speed_priority=speed_priority,
                auto_select=auto_select,
                generate_sparse=generate_sparse,
            )
        except Exception as e:
            logger.exception(
                f"Embedding generation failed: {e}"
            )  # TODO: Convert f-string to logging format
            msg = f"Embedding generation failed: {e}"
            raise EmbeddingServiceError(msg) from e

    async def rerank_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rerank search results using BGE reranker.

        Args:
            query: Search query
            results: List of search results with 'content' field

        Returns:
            Reranked results sorted by relevance

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        try:
            return await self._core_manager.rerank_results(query, results)
        except Exception as e:
            logger.exception(
                f"Result reranking failed: {e}"
            )  # TODO: Convert f-string to logging format
            # Return original results on failure
            return results

    def estimate_cost(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Estimate embedding generation cost.

        Args:
            texts: List of texts to estimate cost for
            provider_name: Optional specific provider to estimate

        Returns:
            Cost estimation per provider

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        return self._core_manager.estimate_cost(texts, provider_name)

    def get_provider_info(self) -> dict[str, dict[str, Any]]:
        """Get information about available providers.

        Returns:
            Provider information with models, dimensions, costs

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        return self._core_manager.get_provider_info()

    async def get_optimal_provider(
        self,
        text_length: int,
        quality_required: bool = False,
        budget_limit: float | None = None,
    ) -> str:
        """Select optimal provider based on constraints.

        Args:
            text_length: Total character count
            quality_required: Whether high quality is required
            budget_limit: Optional maximum cost in dollars

        Returns:
            Optimal provider name

        Raises:
            EmbeddingServiceError: If manager not initialized or no provider meets 
                constraints
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        return await self._core_manager.get_optimal_provider(
            text_length, quality_required, budget_limit
        )

    def analyze_text_characteristics(self, texts: list[str]) -> dict[str, Any]:
        """Analyze text characteristics for smart model selection.

        Args:
            texts: List of texts to analyze

        Returns:
            Analysis results with complexity, type, and quality requirements

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        analysis = self._core_manager.analyze_text_characteristics(texts)

        # Convert TextAnalysis to dict for service boundary
        return {
            "total_length": analysis.total_length,
            "avg_length": analysis.avg_length,
            "complexity_score": analysis.complexity_score,
            "estimated_tokens": analysis.estimated_tokens,
            "text_type": analysis.text_type,
            "requires_high_quality": analysis.requires_high_quality,
        }

    def get_smart_provider_recommendation(
        self,
        text_analysis: dict[str, Any],
        quality_tier: str | None = None,
        max_cost: float | None = None,
        speed_priority: bool = False,
    ) -> dict[str, Any]:
        """Get smart provider recommendation based on analysis.

        Args:
            text_analysis: Analysis of text characteristics
            quality_tier: Optional quality tier override
            max_cost: Optional maximum cost constraint
            speed_priority: Whether to prioritize speed over quality

        Returns:
            Recommendation with provider, model, cost, and reasoning

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        # Convert dict back to TextAnalysis for core manager
        from src.services.embeddings.manager import QualityTier, TextAnalysis

        analysis = TextAnalysis(
            total_length=text_analysis["total_length"],
            avg_length=text_analysis["avg_length"],
            complexity_score=text_analysis["complexity_score"],
            estimated_tokens=text_analysis["estimated_tokens"],
            text_type=text_analysis["text_type"],
            requires_high_quality=text_analysis["requires_high_quality"],
        )

        tier = None
        if quality_tier:
            tier_map = {
                "FAST": QualityTier.FAST,
                "BALANCED": QualityTier.BALANCED,
                "BEST": QualityTier.BEST,
            }
            tier = tier_map.get(quality_tier.upper())

        return self._core_manager.get_smart_provider_recommendation(
            analysis, tier, max_cost, speed_priority
        )

    def get_usage_report(self) -> dict[str, Any]:
        """Get comprehensive usage report.

        Returns:
            Usage statistics with costs, requests, and provider breakdown

        Raises:
            EmbeddingServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            msg = "Embedding manager not initialized"
            raise EmbeddingServiceError(msg)

        return self._core_manager.get_usage_report()

    async def get_status(self) -> dict[str, Any]:
        """Get embedding manager status.

        Returns:
            Status information for providers and usage
        """
        status = {
            "initialized": self._initialized,
            "providers": {},
            "usage": {},
        }

        if self._core_manager:
            try:
                status["providers"] = self.get_provider_info()
                status["usage"] = self.get_usage_report()
            except Exception as e:
                logger.warning(
                    f"Failed to get embedding status: {e}"
                )  # TODO: Convert f-string to logging format
                status["error"] = str(e)

        return status

    def get_core_manager(self) -> Optional["CoreEmbeddingManager"]:
        """Get core embedding manager instance.

        Returns:
            Core EmbeddingManager instance or None
        """
        return self._core_manager
