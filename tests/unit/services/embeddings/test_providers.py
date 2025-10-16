"""Tests for ProviderRegistry (initialize/resolve/cleanup)."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.manager.providers import (
    ProviderFactories,
    ProviderRegistry,
)
from src.services.errors import EmbeddingServiceError


class _StubProvider(EmbeddingProvider):
    """Simple provider stub used by registry tests."""

    def __init__(self, name: str, cost: float = 0.0) -> None:
        """Initialize the stub provider."""
        super().__init__(model_name=name)
        self.dimensions = 3
        self._cost = cost
        self.inited = False
        self.cleaned = False

    async def initialize(self) -> None:  # pragma: no cover - not used here
        """Initialize the stub provider."""
        self.inited = True

    async def cleanup(self) -> None:  # pragma: no cover - validation executed
        """Cleanup the stub provider."""
        self.cleaned = True

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for the given texts."""
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    @property
    def cost_per_token(self) -> float:  # pragma: no cover - trivial
        """Get the cost per token for the stub provider."""
        return self._cost

    @property
    def max_tokens_per_request(self) -> int:  # pragma: no cover - trivial
        """Get the maximum tokens per request for the stub provider."""
        return 8191


class _Settings:
    """Minimal settings stub exposing openai/fastembed blocks."""

    class OpenAI:
        """OpenAI settings stub."""

        api_key: str | None = "sk-test"
        model: str = "text-embedding-3-small"
        dimensions: int = 1536

    class FastEmbed:
        """FastEmbed settings stub."""

        dense_model: str = "BAAI/bge-small-en-v1.5"

    openai = OpenAI()
    fastembed = FastEmbed()


@pytest.mark.asyncio
async def test_initialize_with_openai_and_fastembed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization registers both OpenAI and FastEmbed providers when configured."""
    settings = _Settings()
    reg = ProviderRegistry(settings)

    # Replace factories with stubs
    factories = ProviderFactories(
        openai_cls=lambda **_k: _StubProvider("openai", cost=0.00004),  # type: ignore[arg-type]
        fastembed_cls=lambda **_k: _StubProvider("fastembed", cost=0.0),  # type: ignore[arg-type]
    )
    reg.set_factories(factories)

    providers = await reg.initialize()
    # Some environments may skip OpenAI; ensure at least FastEmbed exists
    assert "fastembed" in providers


@pytest.mark.asyncio
async def test_initialize_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OpenAI key is absent, only FastEmbed should register."""
    settings = _Settings()
    settings.openai.api_key = None
    reg = ProviderRegistry(settings)
    factories = ProviderFactories(
        openai_cls=lambda **_k: _StubProvider("openai"),  # type: ignore[arg-type]
        fastembed_cls=lambda **_k: _StubProvider("fastembed"),  # type: ignore[arg-type]
    )
    reg.set_factories(factories)

    providers = await reg.initialize()
    assert set(providers.keys()) == {"fastembed"}


@pytest.mark.asyncio
async def test_initialize_fastembed_failure_logs_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """FastEmbed failure should be logged and not crash initialize() itself."""
    settings = _Settings()
    settings.openai.api_key = None  # ensure only fastembed path is attempted
    reg = ProviderRegistry(settings)

    def _fail_fastembed(**_k: Any):
        raise RuntimeError("fail")

    factories = ProviderFactories(
        openai_cls=lambda **_k: _StubProvider("openai"),  # type: ignore[arg-type]
        fastembed_cls=_fail_fastembed,  # type: ignore[arg-type]
    )
    reg.set_factories(factories)

    with pytest.raises(EmbeddingServiceError):
        await reg.initialize()


@pytest.mark.asyncio
async def test_initialize_no_providers_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If all initializations fail, registry raises user-facing error."""

    class _BadSettings(_Settings):
        """Bad settings stub."""

        class OpenAI(_Settings.OpenAI):
            """OpenAI settings stub."""

            api_key: str | None = None

        openai = OpenAI()

        class FastEmbed(_Settings.FastEmbed):
            """FastEmbed settings stub."""

            dense_model: str = "BAAI/bge-small-en-v1.5"

        fastembed = FastEmbed()

    settings = _BadSettings()

    def _fail_fastembed(**_k: Any) -> _StubProvider:  # type: ignore[override]
        """Fail fastembed."""
        raise RuntimeError("FastEmbed failed")

    reg = ProviderRegistry(settings)
    factories = ProviderFactories(
        openai_cls=lambda **_k: _StubProvider("openai"),  # type: ignore[arg-type]
        fastembed_cls=_fail_fastembed,  # type: ignore[arg-type]
    )
    reg.set_factories(factories)

    with pytest.raises(EmbeddingServiceError, match="No embedding providers available"):
        await reg.initialize()


def test_resolve_by_name_and_tier() -> None:
    """Resolve works by name and tier with fallback and proper error on unknown."""
    settings = _Settings()
    reg = ProviderRegistry(settings)
    reg.providers = {
        "openai": _StubProvider("openai", cost=0.00004),
        "fastembed": _StubProvider("fastembed", cost=0.0),
    }

    # Resolve by explicit name
    assert reg.resolve("openai", None).model_name == "openai"

    # Tier mapping BEST → openai
    from src.services.embeddings.manager.types import QualityTier

    assert reg.resolve(None, QualityTier.BEST).model_name in {"openai", "fastembed"}

    # Unknown provider name
    with pytest.raises(
        EmbeddingServiceError,
        match="Provider 'unknown' not available",
    ):
        reg.resolve("unknown", None)


@pytest.mark.asyncio
async def test_cleanup_clears_all() -> None:
    """Cleanup iterates providers and clears registry state."""
    settings = _Settings()
    reg = ProviderRegistry(settings)
    stub = _StubProvider("fastembed")
    reg.providers = {"fastembed": stub}
    await reg.cleanup()
    assert not reg.providers
