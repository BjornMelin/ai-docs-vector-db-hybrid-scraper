"""Tests for OpenAI embedding provider with ClientManager integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


@pytest.fixture
def mock_client_manager():
    """Create mock ClientManager."""
    manager = AsyncMock()
    manager.get_openai_client = AsyncMock()
    return manager


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    client = AsyncMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    embedding_response.usage = MagicMock(
        prompt_tokens=3, completion_tokens=0, total_tokens=3
    )
    client.embeddings.create = AsyncMock(return_value=embedding_response)

    return client


@pytest.fixture
def record_tracker(monkeypatch):
    """Patch record_ai_operation to observe tracking metadata."""
    tracker = MagicMock()
    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.record_ai_operation", tracker
    )
    return tracker


class TestOpenAIProviderInitialization:
    """Test OpenAI provider initialization."""

    def test_provider_creation_valid_model(self, mock_client_manager):
        """Test creating provider with valid model."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            client_manager=mock_client_manager,
        )

        assert provider.api_key == "test-key"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536
        assert not provider._initialized

    def test_provider_creation_invalid_model(self, mock_client_manager):
        """Test creating provider with invalid model."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model_name="invalid-model",
                client_manager=mock_client_manager,
            )

    def test_provider_creation_custom_dimensions(self, mock_client_manager):
        """Test creating provider with custom dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-large",
            dimensions=2048,
            client_manager=mock_client_manager,
        )

        assert provider.dimensions == 2048

    def test_provider_creation_dimensions_too_large(self, mock_client_manager):
        """Test creating provider with dimensions exceeding limit."""
        with pytest.raises(EmbeddingServiceError, match=r"Dimensions .* exceeds max"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model_name="text-embedding-3-small",
                dimensions=2000,  # Exceeds 1536 limit
                client_manager=mock_client_manager,
            )

    @pytest.mark.asyncio
    async def test_initialization_with_client_manager(
        self, mock_client_manager, mock_openai_client
    ):
        """Test initialization using ClientManager."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        await provider.initialize()

        assert provider._initialized
        assert provider._client == mock_openai_client
        mock_client_manager.get_openai_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_client_manager_fallback(self, mock_client_manager):
        """Test init when ClientManager returns None (fallback to direct client)."""
        # ClientManager returns None, so provider
        # should fail since it requires ClientManager
        mock_client_manager.get_openai_client.return_value = None

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        with pytest.raises(
            EmbeddingServiceError, match="OpenAI API key not configured"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_no_api_key_with_client_manager(
        self, mock_client_manager
    ):
        """Test initialization when ClientManager returns None (no API key)."""
        mock_client_manager.get_openai_client.return_value = None

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        with pytest.raises(
            EmbeddingServiceError, match="OpenAI API key not configured"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_client_manager):
        """Test initialization failure handling."""
        mock_client_manager.get_openai_client.side_effect = Exception(
            "Connection failed"
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize OpenAI client"
        ):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_double_initialization(self, mock_client_manager, mock_openai_client):
        """Test that double initialization is safe."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        await provider.initialize()
        await provider.initialize()  # Should not raise error

        assert provider._initialized

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_client_manager, mock_openai_client):
        """Test provider cleanup."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        await provider.initialize()
        await provider.cleanup()

        assert not provider._initialized
        assert provider._client is None


class TestOpenAIProviderEmbeddingGeneration:
    """Test embedding generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self, mock_client_manager):
        """Test embedding generation when not initialized."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_input(
        self, mock_client_manager, mock_openai_client
    ):
        """Test embedding generation with empty input."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        result = await provider.generate_embeddings([])

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(
        self, mock_client_manager, mock_openai_client, record_tracker
    ):
        """Test embedding generation for single text."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        result = await provider.generate_embeddings(["test text"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.assert_called_once()
        assert record_tracker.call_count == 1
        kwargs = record_tracker.call_args.kwargs
        assert kwargs["operation_type"] == "embedding"
        assert kwargs["success"] is True
        assert kwargs["provider"] == "openai"
        assert kwargs["tokens"] == 3
        assert kwargs["prompt_tokens"] == 3
        assert kwargs["completion_tokens"] is None

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(
        self, mock_client_manager, mock_openai_client
    ):
        """Test embedding generation with batching."""
        # Mock multiple embeddings response
        embedding_response = MagicMock()
        embedding_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9]),
        ]
        embedding_response.usage = MagicMock(
            prompt_tokens=3, completion_tokens=0, total_tokens=3
        )
        mock_openai_client.embeddings.create.return_value = embedding_response
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        texts = ["text1", "text2", "text3"]
        result = await provider.generate_embeddings(texts, batch_size=3)

        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]

    @pytest.mark.asyncio
    async def test_generate_embeddings_usage_fallback(
        self,
        mock_client_manager,
        mock_openai_client,
        record_tracker,
        monkeypatch,
    ):
        """Validate token fallback when API usage metadata is missing."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        fallback_response = MagicMock()
        fallback_response.data = [MagicMock(embedding=[0.5, 0.6, 0.7])]
        fallback_response.usage = None
        mock_openai_client.embeddings.create.return_value = fallback_response

        monkeypatch.setattr(
            "src.services.embeddings.openai_provider.OpenAIEmbeddingProvider._count_tokens",
            lambda _self, _texts: 99,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        result = await provider.generate_embeddings(["fallback"])

        assert result == [[0.5, 0.6, 0.7]]
        kwargs = record_tracker.call_args.kwargs
        assert kwargs["tokens"] == 99
        assert kwargs["prompt_tokens"] == 99
        assert kwargs["completion_tokens"] is None

    @pytest.mark.asyncio
    async def test_generate_embeddings_large_batch(
        self, mock_client_manager, mock_openai_client
    ):
        """Test embedding generation with large batch requiring multiple API calls."""

        # Mock responses for each batch - must return the correct number of embeddings
        def mock_create_embeddings(*_args, **_kwargs):
            # Get the input text count from the request
            input_texts = _kwargs.get("input", [])
            batch_size = len(input_texts)

            embedding_response = MagicMock()
            embedding_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)
            ]
            # provide usage metadata for each call
            embedding_response.usage = MagicMock(
                prompt_tokens=batch_size,
                completion_tokens=0,
                total_tokens=batch_size,
            )
            return embedding_response

        mock_openai_client.embeddings.create.side_effect = mock_create_embeddings
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        # Create 150 texts to force multiple batches with default batch_size=100
        texts = [f"text{i}" for i in range(150)]
        result = await provider.generate_embeddings(texts)

        assert len(result) == 150
        # Should make 2 API calls (100 + 50)
        assert mock_openai_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_dimensions(
        self, mock_client_manager, mock_openai_client
    ):
        """Test embedding gen with custom dimensions for text-embedding-3 models."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            dimensions=512,
            client_manager=mock_client_manager,
        )
        await provider.initialize()

        await provider.generate_embeddings(["test"])

        # Verify dimensions parameter was passed
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args[1]["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_generate_embeddings_no_dimensions_for_old_model(
        self, mock_client_manager, mock_openai_client
    ):
        """Test that dimensions parameter is not passed for old models."""
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-ada-002",
            client_manager=mock_client_manager,
        )
        await provider.initialize()

        await provider.generate_embeddings(["test"])

        # Verify dimensions parameter was not passed
        call_args = mock_openai_client.embeddings.create.call_args
        assert "dimensions" not in call_args[1]

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_rate_limit(
        self, mock_client_manager, mock_openai_client, record_tracker
    ):
        """Test handling of rate limit API errors."""
        mock_openai_client.embeddings.create.side_effect = Exception(
            "rate_limit_exceeded"
        )
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="OpenAI rate limit exceeded"):
            await provider.generate_embeddings(["test"])
        assert record_tracker.call_count == 1
        assert record_tracker.call_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_quota(
        self, mock_client_manager, mock_openai_client
    ):
        """Test handling of quota exceeded API errors."""
        mock_openai_client.embeddings.create.side_effect = Exception(
            "insufficient_quota"
        )
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="OpenAI API quota exceeded"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_invalid_key(
        self, mock_client_manager, mock_openai_client
    ):
        """Test handling of invalid API key errors."""
        mock_openai_client.embeddings.create.side_effect = Exception("invalid_api_key")
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="Invalid OpenAI API key"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error_context_length(
        self, mock_client_manager, mock_openai_client
    ):
        """Test handling of context length exceeded errors."""
        mock_openai_client.embeddings.create.side_effect = Exception(
            "context_length_exceeded"
        )
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="Text too long for model"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_generic_error(
        self, mock_client_manager, mock_openai_client
    ):
        """Test handling of generic API errors."""
        mock_openai_client.embeddings.create.side_effect = Exception("Generic error")
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(
            EmbeddingServiceError, match="Failed to generate embeddings"
        ):
            await provider.generate_embeddings(["test"])


class TestOpenAIProviderProperties:
    """Test provider properties."""

    def test_cost_per_token_small_model(self, mock_client_manager):
        """Test cost per token for small model."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            client_manager=mock_client_manager,
        )

        expected_cost = 0.02 / 1_000_000  # $0.02 per 1M tokens
        assert provider.cost_per_token == expected_cost

    def test_cost_per_token_large_model(self, mock_client_manager):
        """Test cost per token for large model."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-large",
            client_manager=mock_client_manager,
        )

        expected_cost = 0.13 / 1_000_000  # $0.13 per 1M tokens
        assert provider.cost_per_token == expected_cost

    def test_max_tokens_per_request(self, mock_client_manager):
        """Test maximum tokens per request."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            client_manager=mock_client_manager,
        )

        assert provider.max_tokens_per_request == 8191


class TestOpenAIProviderBatchAPI:
    """Test batch API functionality."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_not_initialized(
        self, mock_client_manager
    ):
        """Test batch API when not initialized."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings_batch_api(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_success(
        self, mock_client_manager, mock_openai_client, record_tracker, monkeypatch
    ):
        """Test successful batch API submission."""
        # Mock file upload and batch creation
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-456"

        mock_openai_client.files.create = AsyncMock(return_value=mock_file_response)
        mock_openai_client.batches.create = AsyncMock(return_value=mock_batch_response)
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        monkeypatch.setattr(
            "src.services.embeddings.openai_provider.OpenAIEmbeddingProvider._count_tokens",
            lambda _self, _texts: 24,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        result = await provider.generate_embeddings_batch_api(["text1", "text2"])

        assert result == "batch-456"
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.batches.create.assert_called_once()
        assert record_tracker.call_count == 1
        kwargs = record_tracker.call_args.kwargs
        assert kwargs["operation_type"] == "embedding_batch"
        assert kwargs["tokens"] == 24
        assert kwargs["prompt_tokens"] == 24
        assert kwargs["attributes"]["gen_ai.request.batch_size"] == 2
        assert kwargs["attributes"]["gen_ai.request.custom_ids_provided"] is False
        assert kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_with_custom_ids(
        self, mock_client_manager, mock_openai_client, record_tracker, monkeypatch
    ):
        """Test batch API with custom IDs."""
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-456"

        mock_openai_client.files.create = AsyncMock(return_value=mock_file_response)
        mock_openai_client.batches.create = AsyncMock(return_value=mock_batch_response)
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        monkeypatch.setattr(
            "src.services.embeddings.openai_provider.OpenAIEmbeddingProvider._count_tokens",
            lambda _self, _texts: 18,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        custom_ids = ["id1", "id2"]
        result = await provider.generate_embeddings_batch_api(
            ["text1", "text2"], custom_ids=custom_ids
        )

        assert result == "batch-456"
        kwargs = record_tracker.call_args.kwargs
        assert kwargs["tokens"] == 18
        assert kwargs["attributes"]["gen_ai.request.custom_ids_provided"] is True

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_with_dimensions(
        self, mock_client_manager, mock_openai_client
    ):
        """Test batch API with dimensions for text-embedding-3 models."""
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-456"

        mock_openai_client.files.create = AsyncMock(return_value=mock_file_response)
        mock_openai_client.batches.create = AsyncMock(return_value=mock_batch_response)
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-small",
            dimensions=512,
            client_manager=mock_client_manager,
        )
        await provider.initialize()

        result = await provider.generate_embeddings_batch_api(["text1"])

        assert result == "batch-456"

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_error_handling(
        self, mock_client_manager, mock_openai_client, record_tracker
    ):
        """Test batch API error handling."""
        mock_openai_client.files.create = AsyncMock(
            side_effect=Exception("File upload failed")
        )
        mock_client_manager.get_openai_client.return_value = mock_openai_client

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", client_manager=mock_client_manager
        )
        await provider.initialize()

        with pytest.raises(EmbeddingServiceError, match="Failed to create batch job"):
            await provider.generate_embeddings_batch_api(["test"])
        assert record_tracker.call_args.kwargs["success"] is False
