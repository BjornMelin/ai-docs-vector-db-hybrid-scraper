"""Tests for the OpenAI embedding provider with DI-managed clients."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Return a mocked AsyncOpenAI client."""

    client = AsyncMock()
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    embedding_response.usage = MagicMock(
        prompt_tokens=3,
        completion_tokens=0,
        total_tokens=3,
    )
    client.embeddings.create = AsyncMock(return_value=embedding_response)
    return client


@pytest.fixture
def record_tracker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch telemetry hook to capture recorded operations."""

    tracker = MagicMock()
    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.record_ai_operation", tracker
    )
    return tracker


class TestOpenAIProviderInitialization:
    """Lifecycle tests for the OpenAI embedding provider."""

    def test_provider_creation_valid_model(self, mock_openai_client: AsyncMock) -> None:
        """Provider should accept supported models and default dimensions."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            client=mock_openai_client,
        )

        assert provider.api_key == "test-key"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536
        assert not provider._initialized

    def test_provider_creation_invalid_model(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Unsupported model names should raise configuration errors."""

        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model_name="invalid-model",
                client=mock_openai_client,
            )

    def test_provider_creation_custom_dimensions(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Custom dimensions within bounds should be accepted."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-large",
            dimensions=2048,
            client=mock_openai_client,
        )

        assert provider.dimensions == 2048

    def test_provider_creation_dimensions_too_large(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Dimensions above model limits must raise validation errors."""

        with pytest.raises(EmbeddingServiceError, match=r"Dimensions .* exceeds max"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model_name="text-embedding-3-small",
                dimensions=2000,
                client=mock_openai_client,
            )

    @pytest.mark.asyncio
    async def test_initialization_with_client(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Initialization should reuse DI-supplied client instances."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        assert provider._initialized
        assert provider._client == mock_openai_client

    @pytest.mark.asyncio
    async def test_initialization_with_factory(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Client factories must be awaited exactly once."""

        client_factory = AsyncMock(return_value=mock_openai_client)
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_factory=client_factory
        )

        await provider.initialize()

        client_factory.assert_awaited_once()
        assert provider._client == mock_openai_client

    @pytest.mark.asyncio
    async def test_initialization_missing_api_key(self) -> None:
        """Initialization without API key or client should fail fast."""

        provider = OpenAIEmbeddingProvider(api_key="")
        with pytest.raises(
            EmbeddingServiceError, match="OpenAI API key not configured"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_failure(self) -> None:
        """Errors thrown by a client factory should surface as service errors."""

        client_factory = AsyncMock(side_effect=Exception("Connection failed"))
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_factory=client_factory
        )

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize OpenAI client"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_double_initialization_is_idempotent(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Calling initialize twice should be a no-op."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()
        await provider.initialize()

        assert provider._initialized

    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self, mock_openai_client: AsyncMock) -> None:
        """Cleanup should release the cached client."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()
        await provider.cleanup()

        assert not provider._initialized
        assert provider._client is None


class TestOpenAIProviderEmbeddings:
    """Embedding request behaviour."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Provider must enforce initialization before embedding generation."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_input(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
    ) -> None:
        """Empty input should short-circuit without invoking OpenAI."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        result = await provider.generate_embeddings([])

        assert result == []
        mock_openai_client.embeddings.create.assert_not_called()
        record_tracker.assert_called_once()
        op_kwargs = record_tracker.call_args.kwargs
        assert op_kwargs["tokens"] is None
        assert op_kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
    ) -> None:
        """Embeddings should be generated and telemetry recorded."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        embeddings = await provider.generate_embeddings(["hello", "world"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        mock_openai_client.embeddings.create.assert_awaited_once()
        assert record_tracker.call_count == 1
        op_kwargs = record_tracker.call_args.kwargs
        assert op_kwargs["success"] is True
        assert op_kwargs["tokens"] == 3

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_records_failure(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
    ) -> None:
        """Failures should propagate and mark telemetry as unsuccessful."""

        mock_openai_client.embeddings.create.side_effect = RuntimeError("API error")
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        with pytest.raises(
            EmbeddingServiceError, match="Failed to generate embeddings"
        ):
            await provider.generate_embeddings(["test"])

        assert record_tracker.call_count == 1
        assert record_tracker.call_args.kwargs["success"] is False


class TestOpenAIProviderBatchAPI:
    """Batch submission behaviour."""

    @pytest.mark.asyncio
    async def test_batch_api_not_initialized(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Provider must be initialized before submitting batch jobs."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings_batch_api(["text"])

    @pytest.mark.asyncio
    async def test_batch_api_custom_id_mismatch(
        self,
        mock_openai_client: AsyncMock,
    ) -> None:
        """Custom ID length must match text payload length."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="Custom ID list must match"):
            await provider.generate_embeddings_batch_api(
                ["text", "another"], custom_ids=["only-one"]
            )

    @pytest.mark.asyncio
    async def test_batch_api_success(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful batch submission returns batch ID and records telemetry."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        upload_mock = AsyncMock(return_value=SimpleNamespace(id="file-123"))
        create_batch_mock = AsyncMock(return_value=SimpleNamespace(id="batch-456"))

        monkeypatch.setattr(provider, "_upload_file", upload_mock)
        monkeypatch.setattr(provider, "_create_batch", create_batch_mock)
        monkeypatch.setattr(provider, "_count_tokens", lambda texts: 24)

        batch_id = await provider.generate_embeddings_batch_api(["foo", "bar"])

        assert batch_id == "batch-456"
        upload_mock.assert_awaited_once()
        create_batch_mock.assert_awaited_once()
        op_kwargs = record_tracker.call_args.kwargs
        assert op_kwargs["success"] is True
        assert op_kwargs["tokens"] == 24
        assert op_kwargs["attributes"]["gen_ai.request.batch_size"] == 2
        assert op_kwargs["attributes"]["gen_ai.request.custom_ids_provided"] is False

    @pytest.mark.asyncio
    async def test_batch_api_custom_ids(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batch submission should honour explicit custom identifiers."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        upload_mock = AsyncMock(return_value=SimpleNamespace(id="file-123"))
        create_batch_mock = AsyncMock(return_value=SimpleNamespace(id="batch-789"))
        monkeypatch.setattr(provider, "_upload_file", upload_mock)
        monkeypatch.setattr(provider, "_create_batch", create_batch_mock)
        monkeypatch.setattr(provider, "_count_tokens", lambda texts: 10)

        batch_id = await provider.generate_embeddings_batch_api(
            ["foo", "bar"], custom_ids=["id-1", "id-2"]
        )

        assert batch_id == "batch-789"
        op_kwargs = record_tracker.call_args.kwargs
        assert op_kwargs["attributes"]["gen_ai.request.custom_ids_provided"] is True

    @pytest.mark.asyncio
    async def test_batch_api_includes_dimensions(
        self,
        mock_openai_client: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """text-embedding-3 models should pass dimensions in payload."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            dimensions=256,
            client=mock_openai_client,
        )
        await provider.initialize()

        requests: list[dict[str, Any]] = []

        async def _upload_stub(file, purpose):  # type: ignore[override] # noqa: D401
            payload = file.read().decode("utf-8").strip().splitlines()
            for line in payload:
                requests.append(json.loads(line))
            return SimpleNamespace(id="file-123")

        async def _create_batch_stub(*args, **kwargs):  # noqa: D401
            return SimpleNamespace(id="batch-123")

        monkeypatch.setattr(provider, "_upload_file", _upload_stub)
        monkeypatch.setattr(provider, "_create_batch", _create_batch_stub)
        monkeypatch.setattr(provider, "_count_tokens", lambda texts: 6)

        await provider.generate_embeddings_batch_api(["foo"])

        assert requests
        assert requests[0]["body"]["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_batch_api_error_records_failure(
        self,
        mock_openai_client: AsyncMock,
        record_tracker: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Errors during submission should mark telemetry as unsuccessful."""

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client=mock_openai_client
        )
        await provider.initialize()

        upload_mock = AsyncMock(side_effect=RuntimeError("upload failure"))
        monkeypatch.setattr(provider, "_upload_file", upload_mock)
        monkeypatch.setattr(provider, "_count_tokens", lambda texts: 4)

        with pytest.raises(EmbeddingServiceError, match="Failed to create batch job"):
            await provider.generate_embeddings_batch_api(["foo"])

        assert record_tracker.call_args.kwargs["success"] is False
