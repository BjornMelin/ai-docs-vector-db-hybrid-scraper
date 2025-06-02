"""Tests for services/embeddings/openai_provider.py - OpenAI integration.

This module tests the OpenAI embedding provider that provides API client integration,
authentication, model selection, parameter handling, error handling, and retry logic.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestOpenAIEmbeddingProviderInitialization:
    """Test cases for OpenAIEmbeddingProvider initialization."""

    def test_provider_initialization_basic(self):
        """Test basic provider initialization."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small"
        )

        assert provider.api_key == "test-key"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536  # Default for text-embedding-3-small
        assert provider._client is None
        assert provider._initialized is False
        assert provider.rate_limiter is None

    def test_provider_initialization_with_dimensions(self):
        """Test provider initialization with custom dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small", dimensions=512
        )

        assert provider.dimensions == 512

    def test_provider_initialization_with_rate_limiter(self):
        """Test provider initialization with rate limiter."""
        rate_limiter = Mock()
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", rate_limiter=rate_limiter
        )

        assert provider.rate_limiter is rate_limiter

    def test_provider_initialization_unsupported_model(self):
        """Test provider initialization with unsupported model."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(api_key="test-key", model_name="unsupported-model")

    def test_provider_initialization_dimensions_too_large(self):
        """Test provider initialization with dimensions exceeding model limit."""
        with pytest.raises(EmbeddingServiceError, match="Dimensions .* exceeds max"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model_name="text-embedding-3-small",
                dimensions=2000,  # Exceeds 1536 limit
            )

    def test_provider_initialization_all_supported_models(self):
        """Test provider initialization with all supported models."""
        supported_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        for model in supported_models:
            provider = OpenAIEmbeddingProvider(api_key="test-key", model_name=model)
            assert provider.model_name == model
            assert provider.dimensions > 0

    def test_provider_model_configs_class_variable(self):
        """Test model configurations class variable."""
        configs = OpenAIEmbeddingProvider._model_configs

        assert "text-embedding-3-small" in configs
        assert "text-embedding-3-large" in configs
        assert "text-embedding-ada-002" in configs

        # Check structure
        small_config = configs["text-embedding-3-small"]
        assert "max_dimensions" in small_config
        assert "cost_per_million" in small_config
        assert "max_tokens" in small_config


class TestOpenAIProviderProperties:
    """Test cases for OpenAI provider properties."""

    def test_cost_per_token_property(self):
        """Test cost_per_token property calculation."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small"
        )

        expected_cost = 0.02 / 1_000_000  # $0.02 per 1M tokens
        assert provider.cost_per_token == expected_cost

    def test_max_tokens_per_request_property(self):
        """Test max_tokens_per_request property."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small"
        )

        assert provider.max_tokens_per_request == 8191

    def test_properties_different_models(self):
        """Test properties for different models."""
        small_provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small"
        )
        large_provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-large"
        )

        # Large model should be more expensive
        assert large_provider.cost_per_token > small_provider.cost_per_token
        # Large model should have more dimensions
        assert large_provider.dimensions > small_provider.dimensions


class TestOpenAIProviderInitialization:
    """Test cases for OpenAI provider initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful provider initialization."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            await provider.initialize()

            assert provider._client is mock_client
            assert provider._initialized is True
            mock_openai.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_openai:
            await provider.initialize()

            # Should not create new client
            mock_openai.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_error(self):
        """Test initialization error handling."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.side_effect = Exception("Connection failed")

            with pytest.raises(
                EmbeddingServiceError, match="Failed to initialize OpenAI client"
            ):
                await provider.initialize()

            assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        """Test successful provider cleanup."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        # Setup initialized state
        mock_client = AsyncMock()
        provider._client = mock_client
        provider._initialized = True

        await provider.cleanup()

        assert provider._client is None
        assert provider._initialized is False
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_client(self):
        """Test cleanup when no client exists."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        # Should not raise error
        await provider.cleanup()

        assert provider._client is None
        assert provider._initialized is False


class TestOpenAIEmbeddingGeneration:
    """Test cases for OpenAI embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self):
        """Test embedding generation when not initialized."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self):
        """Test embedding generation with empty text list."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        embeddings = await provider.generate_embeddings([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        # Mock OpenAI client and response
        mock_client = AsyncMock()
        provider._client = mock_client

        # Mock response structure
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]

        mock_client.embeddings.create.return_value = mock_response

        embeddings = await provider.generate_embeddings(["test text"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args[1]
        assert call_args["input"] == ["test text"]
        assert call_args["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_dimensions(self):
        """Test embedding generation with custom dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small", dimensions=512
        )
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_response

        await provider.generate_embeddings(["test"])

        # Verify dimensions parameter was included
        call_args = mock_client.embeddings.create.call_args[1]
        assert call_args["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_generate_embeddings_ada_model_no_dimensions(self):
        """Test embedding generation with ada model (no dimensions parameter)."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-ada-002"
        )
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_response

        await provider.generate_embeddings(["test"])

        # Verify no dimensions parameter for ada model
        call_args = mock_client.embeddings.create.call_args[1]
        assert "dimensions" not in call_args

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self):
        """Test embedding generation with batch processing."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        # Mock responses for 2 batches
        def mock_create_response(*args, **kwargs):
            batch_size = len(kwargs["input"])
            mock_data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)]
            mock_response = Mock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create_response

        # Test with 3 texts and batch size 2
        texts = ["text1", "text2", "text3"]
        embeddings = await provider.generate_embeddings(texts, batch_size=2)

        assert len(embeddings) == 3
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)

        # Should make 2 API calls (batches of 2 and 1)
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_rate_limiter(self):
        """Test embedding generation with rate limiter."""
        rate_limiter = AsyncMock()
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", rate_limiter=rate_limiter
        )
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_response

        await provider.generate_embeddings(["test"])

        # Verify rate limiter was called
        rate_limiter.acquire.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_generate_embeddings_without_rate_limiter(self):
        """Test embedding generation without rate limiter."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_response

        # Should not raise error
        await provider.generate_embeddings(["test"])


class TestOpenAIErrorHandling:
    """Test cases for OpenAI provider error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test handling of rate limit errors."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.embeddings.create.side_effect = Exception("rate_limit_exceeded")

        with pytest.raises(EmbeddingServiceError, match="OpenAI rate limit exceeded"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_quota_exceeded_error_handling(self):
        """Test handling of quota exceeded errors."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.embeddings.create.side_effect = Exception("insufficient_quota")

        with pytest.raises(EmbeddingServiceError, match="OpenAI API quota exceeded"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_invalid_api_key_error_handling(self):
        """Test handling of invalid API key errors."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.embeddings.create.side_effect = Exception("invalid_api_key")

        with pytest.raises(EmbeddingServiceError, match="Invalid OpenAI API key"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_context_length_exceeded_error_handling(self):
        """Test handling of context length exceeded errors."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.embeddings.create.side_effect = Exception("context_length_exceeded")

        with pytest.raises(EmbeddingServiceError, match="Text too long for model"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generic_error_handling(self):
        """Test handling of generic errors."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.embeddings.create.side_effect = Exception("Unknown error")

        with pytest.raises(
            EmbeddingServiceError, match="Failed to generate embeddings"
        ):
            await provider.generate_embeddings(["test"])


class TestOpenAIBatchAPI:
    """Test cases for OpenAI Batch API functionality."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_not_initialized(self):
        """Test batch API when provider not initialized."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings_batch_api(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_success(self):
        """Test successful batch API submission."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        # Mock file upload and batch creation responses
        mock_file_response = Mock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = Mock()
        mock_batch_response.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch_response

        texts = ["text1", "text2"]
        batch_id = await provider.generate_embeddings_batch_api(texts)

        assert batch_id == "batch-456"

        # Verify file upload and batch creation
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_with_custom_ids(self):
        """Test batch API with custom IDs."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_file_response = Mock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = Mock()
        mock_batch_response.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch_response

        texts = ["text1", "text2"]
        custom_ids = ["id1", "id2"]

        await provider.generate_embeddings_batch_api(texts, custom_ids)

        # File should be created successfully
        mock_client.files.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_with_dimensions(self):
        """Test batch API with custom dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small", dimensions=512
        )
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_file_response = Mock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = Mock()
        mock_batch_response.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch_response

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("builtins.open", create=True) as mock_open,
        ):
            # Mock temp file
            mock_file = Mock()
            mock_file.name = "/tmp/test.jsonl"
            mock_file.fileno.return_value = 1  # Mock file descriptor for fsync
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_temp_file.return_value = mock_file

            # Mock file read
            mock_open.return_value.__enter__.return_value = Mock()

            await provider.generate_embeddings_batch_api(["test"])

            # Verify dimensions were included in JSONL
            assert mock_file.write.called

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_with_rate_limiter(self):
        """Test batch API with rate limiter."""
        rate_limiter = AsyncMock()
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", rate_limiter=rate_limiter
        )
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        mock_file_response = Mock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = Mock()
        mock_batch_response.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch_response

        await provider.generate_embeddings_batch_api(["test"])

        # Rate limiter should be called for both file upload and batch creation
        assert rate_limiter.acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_error(self):
        """Test batch API error handling."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client
        mock_client.files.create.side_effect = Exception("Upload failed")

        with pytest.raises(EmbeddingServiceError, match="Failed to create batch job"):
            await provider.generate_embeddings_batch_api(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api_file_cleanup(self):
        """Test batch API temporary file cleanup."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        # Simulate file creation failure after temp file creation
        mock_client.files.create.side_effect = Exception("Upload failed")

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("os.path.exists", return_value=True),
            patch("os.unlink") as mock_unlink,
        ):
            mock_file = Mock()
            mock_file.name = "/tmp/test.jsonl"
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_temp_file.return_value = mock_file

            with pytest.raises(EmbeddingServiceError):
                await provider.generate_embeddings_batch_api(["test"])

            # Verify temp file was cleaned up
            mock_unlink.assert_called_once()


class TestOpenAIProviderIntegration:
    """Integration test cases for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_full_provider_lifecycle(self):
        """Test complete provider lifecycle."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model_name="text-embedding-3-small", dimensions=512
        )

        # Initial state
        assert not provider._initialized
        assert provider._client is None

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock embedding response
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1, 0.2, 0.3]
            mock_response = Mock()
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            # Initialize
            await provider.initialize()
            assert provider._initialized
            assert provider._client is mock_client

            # Generate embeddings
            embeddings = await provider.generate_embeddings(["test text"])
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]

            # Cleanup
            await provider.cleanup()
            assert not provider._initialized
            assert provider._client is None
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_embedding_requests(self):
        """Test multiple embedding generation requests."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        # Mock different responses for each call
        call_count = 0

        def mock_create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_data = [
                Mock(embedding=[0.1 * call_count, 0.2 * call_count, 0.3 * call_count])
            ]
            mock_response = Mock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create_response

        # Generate embeddings multiple times
        embeddings1 = await provider.generate_embeddings(["text1"])
        embeddings2 = await provider.generate_embeddings(["text2"])

        assert embeddings1[0] == [0.1, 0.2, 0.3]
        assert embeddings2[0] == [0.2, 0.4, 0.6]
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing of large text batches."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        def mock_create_response(*args, **kwargs):
            batch_size = len(kwargs["input"])
            mock_data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(batch_size)]
            mock_response = Mock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create_response

        # Test with 250 texts (should create 3 batches with default batch size 100)
        texts = [f"text{i}" for i in range(250)]
        embeddings = await provider.generate_embeddings(texts)

        assert len(embeddings) == 250
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)

        # Should make 3 API calls
        assert mock_client.embeddings.create.call_count == 3

    def test_model_configuration_consistency(self):
        """Test model configuration consistency across different models."""
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        for model_name in models_to_test:
            provider = OpenAIEmbeddingProvider(
                api_key="test-key", model_name=model_name
            )

            # Verify configuration is loaded correctly
            config = provider._model_configs[model_name]
            assert provider.dimensions == config["max_dimensions"]
            assert provider.cost_per_token == config["cost_per_million"] / 1_000_000
            assert provider.max_tokens_per_request == config["max_tokens"]

    @pytest.mark.asyncio
    async def test_provider_error_recovery(self):
        """Test provider error recovery scenarios."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        call_count = 0

        def mock_create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            # Success on second call
            mock_data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_response = Mock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create_response

        # First call should fail
        with pytest.raises(EmbeddingServiceError):
            await provider.generate_embeddings(["test"])

        # Second call should succeed
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self):
        """Test concurrent embedding generation."""
        import asyncio

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        provider._initialized = True

        mock_client = AsyncMock()
        provider._client = mock_client

        def mock_create_response(*args, **kwargs):
            mock_data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]]
            mock_response = Mock()
            mock_response.data = mock_data
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create_response

        # Generate embeddings concurrently
        tasks = [provider.generate_embeddings([f"text{i}"]) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(len(result) == 1 for result in results)
        assert mock_client.embeddings.create.call_count == 5
