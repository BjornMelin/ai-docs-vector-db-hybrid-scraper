"""Tests for the OpenAI embedding provider with direct SDK usage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


@pytest.fixture
def async_openai_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch AsyncOpenAI constructor to return an async mock client."""

    client = AsyncMock()
    client.close = AsyncMock()

    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    embedding_response.model = "text-embedding-3-small"
    embedding_response.usage = MagicMock(
        prompt_tokens=3,
        completion_tokens=0,
        total_tokens=3,
    )
    client.embeddings.create = AsyncMock(return_value=embedding_response)

    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.AsyncOpenAI",
        MagicMock(return_value=client),
    )
    return client


@pytest.fixture
def telemetry_tracker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    tracker = MagicMock()
    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.record_ai_operation", tracker
    )
    return tracker


class TestInitialization:
    """OpenAIEmbeddingProvider initialization behaviour."""

    def test_supported_model_defaults(self) -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model_name="text-embedding-3-small"
        )

        assert provider.api_key == "sk-test"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(api_key="sk-test", model_name="not-a-model")

    def test_custom_dimensions_within_bounds(self) -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model_name="text-embedding-3-large",
            dimensions=2048,
        )
        assert provider.dimensions == 2048

    def test_dimensions_above_limit(self) -> None:
        with pytest.raises(EmbeddingServiceError, match=r"Dimensions .* exceeds max"):
            OpenAIEmbeddingProvider(
                api_key="sk-test",
                model_name="text-embedding-3-small",
                dimensions=5000,
            )

    @pytest.mark.asyncio
    async def test_initialize_creates_client(
        self, async_openai_mock: AsyncMock
    ) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")

        await provider.initialize()

        assert provider._initialized is True
        assert provider._client is async_openai_mock

    @pytest.mark.asyncio
    async def test_initialize_without_key_fails(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="")

        with pytest.raises(
            EmbeddingServiceError, match="OpenAI API key not configured"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_handles_constructor_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.services.embeddings.openai_provider.AsyncOpenAI",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        provider = OpenAIEmbeddingProvider(api_key="sk-test")

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize OpenAI client"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_double_initialize_is_idempotent(
        self, async_openai_mock: AsyncMock
    ) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        await provider.initialize()
        await provider.initialize()

        assert provider._client is async_openai_mock

    @pytest.mark.asyncio
    async def test_cleanup_closes_client(self, async_openai_mock: AsyncMock) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        await provider.initialize()
        await provider.cleanup()

        async_openai_mock.close.assert_awaited_once()
        assert provider._client is None
        assert provider._initialized is False


class TestEmbeddingGeneration:
    """Embedding request flow."""

    @pytest.mark.asyncio
    async def test_generation_requires_initialization(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings(["hello"])

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(
        self,
        async_openai_mock: AsyncMock,
        telemetry_tracker: MagicMock,
    ) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        await provider.initialize()

        result = await provider.generate_embeddings([])

        assert result == []
        async_openai_mock.embeddings.create.assert_not_called()
        telemetry_tracker.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_embedding_generation(
        self,
        async_openai_mock: AsyncMock,
        telemetry_tracker: MagicMock,
    ) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        await provider.initialize()

        embeddings = await provider.generate_embeddings(["hello world"])

        async_openai_mock.embeddings.create.assert_awaited_once()
        assert embeddings == [[0.1, 0.2, 0.3]]
        telemetry_tracker.assert_called_once()
        kwargs = telemetry_tracker.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_api_error_raises_embedding_service_error(
        self,
        async_openai_mock: AsyncMock,
    ) -> None:
        async_openai_mock.embeddings.create.side_effect = RuntimeError("rate limit")

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError):
            await provider.generate_embeddings(["hello"])
