"""Additional tests specifically for OpenAI provider coverage."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from openai import APIConnectionError
from openai import APIError
from openai import APITimeoutError
from openai import BadRequestError
from openai import InternalServerError
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestOpenAIProviderCoverage:
    """Tests to improve OpenAI provider coverage."""

    @pytest.mark.asyncio
    async def test_batch_processing_methods(self):
        """Test batch processing helper methods."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

        # Test _split_into_batches
        texts = [f"text{i}" for i in range(250)]
        batches = list(provider._split_into_batches(texts, batch_size=100))
        assert len(batches) == 3  # 250 / 100 = 2.5, so 3 batches
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

        # Test _split_into_token_batches
        # Create texts with varying token counts
        short_texts = ["short"] * 10  # ~1 token each
        medium_texts = ["medium length text here"] * 5  # ~4 tokens each
        long_texts = ["very long text " * 100] * 2  # ~300 tokens each

        all_texts = short_texts + medium_texts + long_texts
        token_batches = list(
            provider._split_into_token_batches(all_texts, max_tokens=100)
        )

        # Should split based on token count
        assert len(token_batches) > 1

        # Each batch should be under token limit
        for batch in token_batches:
            # Rough estimate: 4 chars per token
            estimated_tokens = sum(len(text) // 4 for text in batch)
            assert estimated_tokens <= 100 or len(batch) == 1  # Single text may exceed

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test comprehensive API error handling."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await provider.initialize()

            # Test APIConnectionError
            mock_instance.embeddings.create.side_effect = APIConnectionError(
                message="Connection failed"
            )
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "connection failed" in str(exc_info.value).lower()

            # Test APITimeoutError
            mock_instance.embeddings.create.side_effect = APITimeoutError(
                request=MagicMock()
            )
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "request timed out" in str(exc_info.value).lower()

            # Test InternalServerError
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_instance.embeddings.create.side_effect = InternalServerError(
                message="Server error",
                response=mock_response,
                body=None,
            )
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "server error" in str(exc_info.value).lower()

            # Test BadRequestError
            mock_response.status_code = 400
            mock_instance.embeddings.create.side_effect = BadRequestError(
                message="Invalid request",
                response=mock_response,
                body=None,
            )
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "invalid request" in str(exc_info.value).lower()

            # Test generic APIError
            mock_instance.embeddings.create.side_effect = APIError(
                message="Generic API error",
                request=MagicMock(),
                body=None,
            )
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "api error" in str(exc_info.value).lower()

            # Test generic Exception
            mock_instance.embeddings.create.side_effect = Exception("Unexpected error")
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])
            assert "unexpected error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_batch_embedding_with_rate_limiter(self):
        """Test batch embedding with rate limiting."""
        from src.services.utilities.rate_limiter import RateLimiter

        rate_limiter = RateLimiter(max_calls=10, time_window=1)
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            rate_limiter=rate_limiter,
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock successful responses
            def create_mock_response(num_embeddings):
                mock_data = []
                for i in range(num_embeddings):
                    mock_embedding = MagicMock()
                    mock_embedding.embedding = [0.1] * 1536
                    mock_data.append(mock_embedding)
                mock_response = MagicMock()
                mock_response.data = mock_data
                return mock_response

            mock_instance.embeddings.create.side_effect = (
                lambda input, **kwargs: create_mock_response(len(input))
            )

            await provider.initialize()

            # Test large batch that requires splitting and rate limiting
            large_batch = ["text"] * 300  # Will be split into multiple API calls

            embeddings = await provider.generate_embeddings(large_batch, batch_size=100)

            assert len(embeddings) == 300
            assert all(len(emb) == 1536 for emb in embeddings)

            # Should have made 3 calls (300 / 100)
            assert mock_instance.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_dimensions_parameter_handling(self):
        """Test how dimensions parameter is handled in API calls."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=512,  # Custom dimensions
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock response
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 512
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await provider.initialize()

            # Generate embeddings
            await provider.generate_embeddings(["test"])

            # Check that dimensions was passed to API
            mock_instance.embeddings.create.assert_called_once()
            call_args = mock_instance.embeddings.create.call_args
            assert call_args.kwargs.get("dimensions") == 512

    @pytest.mark.asyncio
    async def test_ada_002_no_dimensions(self):
        """Test that ada-002 doesn't pass dimensions parameter."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-ada-002",
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock response
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await provider.initialize()

            # Generate embeddings
            await provider.generate_embeddings(["test"])

            # Check that dimensions was NOT passed for ada-002
            mock_instance.embeddings.create.assert_called_once()
            call_args = mock_instance.embeddings.create.call_args
            assert "dimensions" not in call_args.kwargs

    @pytest.mark.asyncio
    async def test_model_validation(self):
        """Test model validation in constructor."""
        # Test invalid model
        with pytest.raises(EmbeddingServiceError) as exc_info:
            OpenAIEmbeddingProvider(
                api_key="sk-test-key",
                model_name="invalid-model",
            )
        assert "Unsupported model" in str(exc_info.value)

        # Test valid models
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        for model in valid_models:
            provider = OpenAIEmbeddingProvider(
                api_key="sk-test-key",
                model_name=model,
            )
            assert provider.model_name == model

    @pytest.mark.asyncio
    async def test_cleanup_without_client(self):
        """Test cleanup when client is not initialized."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        # Cleanup without initialization should not raise
        await provider.cleanup()
        assert provider._client is None
        assert not provider._initialized
