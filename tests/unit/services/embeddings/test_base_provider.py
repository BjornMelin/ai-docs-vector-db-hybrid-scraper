"""Unit tests for the abstract embeddings provider contract."""
# pylint: abstract-class-instantiated

from __future__ import annotations

from abc import ABCMeta

import pytest

from src.services.embeddings.base import EmbeddingProvider


class DummyProvider(EmbeddingProvider):
    """Concrete provider used for exercising the abstract contract."""

    def __init__(self, *, normalize: bool = False):
        """Configure the dummy provider for testing."""
        super().__init__("dummy-model")
        self.dimensions = 4
        self._normalize_embeddings = normalize
        self.initialized = False
        self.cleaned_up = False

    async def initialize(self) -> None:
        """Mark provider as initialized."""
        self.initialized = True

    async def cleanup(self) -> None:
        """Mark provider as cleaned up."""
        self.cleaned_up = True

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Return fixed-length embeddings derived from text length."""
        if batch_size is not None:
            assert batch_size > 0
        return [[float(len(text))] * self.dimensions for text in texts]

    @property
    def cost_per_token(self) -> float:
        """Return nominal cost-per-token for the dummy provider."""
        return 0.0001

    @property
    def max_tokens_per_request(self) -> int:
        """Expose maximum tokens per request."""
        return 4096


def test_embedding_provider_is_abstract() -> None:
    """EmbeddingProvider must remain abstract to enforce the contract."""
    assert isinstance(EmbeddingProvider, ABCMeta)
    with pytest.raises(TypeError):
        EmbeddingProvider("invalid")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_concrete_provider_lifecycle() -> None:
    """Concrete providers must support initialize -> use -> cleanup lifecycle."""
    provider = DummyProvider()

    await provider.initialize()
    assert provider.initialized is True

    vectors = await provider.generate_embeddings(["alpha", "beta"])
    assert vectors == [[5.0] * 4, [4.0] * 4]
    assert provider.embedding_dimension == 4

    await provider.cleanup()
    assert provider.cleaned_up is True


def test_normalization_flag_defaults_to_false() -> None:
    """Providers default to non-normalized embeddings unless specified."""
    provider = DummyProvider()
    assert provider.normalize_embeddings is False


def test_normalization_flag_can_be_enabled() -> None:
    """Providers may opt-in to normalized embeddings."""
    provider = DummyProvider(normalize=True)
    assert provider.normalize_embeddings is True
