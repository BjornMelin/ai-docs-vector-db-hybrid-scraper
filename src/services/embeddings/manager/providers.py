"""Provider lifecycle management for embedding services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


@dataclass
class ProviderFactories:
    """Factories for provider implementations."""

    openai_cls: type[OpenAIEmbeddingProvider]
    fastembed_cls: type[FastEmbedProvider]


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import QualityTier


try:  # pragma: no cover - optional dependency
    from FlagEmbedding import FlagReranker
except ImportError:  # pragma: no cover - optional dependency
    FlagReranker = None


logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Manages embedding provider lifecycle and tier lookups."""

    _DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    _DEFAULT_TIER_MAP: dict[str, str] = {
        "fast": "fastembed",
        "balanced": "fastembed",
        "best": "openai",
    }

    def __init__(
        self,
        config: Any,
        factories: ProviderFactories | None = None,
    ) -> None:
        self._config = config
        self._providers: dict[str, EmbeddingProvider] = {}
        self._reranker = None
        self._reranker_model = self._DEFAULT_RERANKER_MODEL
        self._factories = factories or ProviderFactories(
            openai_cls=OpenAIEmbeddingProvider,
            fastembed_cls=FastEmbedProvider,
        )

    @property
    def providers(self) -> dict[str, EmbeddingProvider]:
        """Expose mutable provider dictionary for compatibility."""

        return self._providers

    @providers.setter
    def providers(self, value: dict[str, EmbeddingProvider]) -> None:
        self._providers = value

    @property
    def reranker(self) -> Any:
        """Return initialized reranker instance if available."""

        return self._reranker

    @property
    def reranker_model(self) -> str:
        """Expose reranker model identifier."""

        return self._reranker_model

    def set_factories(self, factories: ProviderFactories) -> None:
        """Update provider factories at runtime (useful for tests)."""

        self._factories = factories

    async def initialize(self) -> dict[str, EmbeddingProvider]:
        """Initialize available providers and optional reranker."""

        self._providers.clear()

        if getattr(self._config.openai, "api_key", None):
            await self._initialize_openai()

        await self._initialize_fastembed()

        if not self._providers:
            msg = (
                "No embedding providers available. "
                "Please configure OpenAI API key or enable local embeddings."
            )
            raise EmbeddingServiceError(msg)

        self._initialize_reranker()
        logger.info("Embedding providers initialized: %s", list(self._providers))
        return self._providers

    async def cleanup(self) -> None:
        """Clean up all registered providers."""

        for name, provider in list(self._providers.items()):
            try:
                await provider.cleanup()
                logger.info("Cleaned up %s provider", name)
            except (RuntimeError, ConnectionError, TimeoutError) as exc:
                logger.exception("Error cleaning up %s provider: %s", name, exc)
        self._providers.clear()
        self._reranker = None

    def resolve(
        self, provider_name: str | None, quality_tier: QualityTier | None
    ) -> EmbeddingProvider:
        """Resolve provider by explicit name or quality tier mapping."""

        if provider_name:
            provider = self._providers.get(provider_name)
            if not provider:
                available = ", ".join(sorted(self._providers)) or "none"
                msg = (
                    f"Provider '{provider_name}' not available. "
                    f"Available providers: {available}"
                )
                raise EmbeddingServiceError(msg)
            return provider

        if quality_tier is not None:
            provider_key = self._DEFAULT_TIER_MAP.get(quality_tier.value)
            provider = self._providers.get(provider_key) if provider_key else None
            if provider is not None:
                return provider
            if self._providers:
                fallback = next(iter(self._providers.values()))
                logger.warning(
                    "Preferred provider for %s not available, using %s",
                    quality_tier.value,
                    fallback.__class__.__name__,
                )
                return fallback
            msg = "No providers registered"
            raise EmbeddingServiceError(msg)

        default_name = getattr(
            getattr(self._config, "embedding_provider", None), "value", None
        )
        if default_name and default_name in self._providers:
            return self._providers[default_name]
        if self._providers:
            return next(iter(self._providers.values()))

        msg = "No providers registered"
        raise EmbeddingServiceError(msg)

    def _initialize_reranker(self) -> None:
        """Lazily initialize reranker if dependency present."""

        if FlagReranker is None:
            return
        try:
            self._reranker = FlagReranker(self._reranker_model, use_fp16=True)
            logger.info("Reranker initialized with model %s", self._reranker_model)
        except (
            ImportError,
            RuntimeError,
            OSError,
        ) as exc:  # pragma: no cover - optional
            logger.warning("Reranker initialization failed: %s", exc)
            self._reranker = None

    async def _initialize_openai(self) -> None:
        try:
            provider = self._factories.openai_cls(
                api_key=self._config.openai.api_key,
                model_name=self._config.openai.model,
                dimensions=self._config.openai.dimensions,
            )
            await provider.initialize()
        except Exception as exc:  # pragma: no cover - aligns with legacy behavior
            logger.warning("Failed to initialize OpenAI provider: %s", exc)
            return
        self._providers["openai"] = provider

    async def _initialize_fastembed(self) -> None:
        try:
            model = self._config.fastembed.dense_model
        except AttributeError as exc:  # pragma: no cover - defensive
            raise EmbeddingServiceError("FastEmbed configuration missing") from exc

        try:
            provider = self._factories.fastembed_cls(model_name=model)
            await provider.initialize()
        except Exception as exc:  # pragma: no cover - aligns with legacy behavior
            logger.warning("Failed to initialize FastEmbed provider: %s", exc)
            return
        self._providers["fastembed"] = provider
