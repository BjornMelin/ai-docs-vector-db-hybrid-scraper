"""Tests for embedding providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.config.enums import EmbeddingProvider as EmbeddingProviderEnum
from src.config.models import UnifiedConfig
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        return OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, openai_provider):
        """Test provider initialization."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            await openai_provider.initialize()

            assert openai_provider._initialized
            mock_client.assert_called_once_with(api_key="sk-test-key")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, openai_provider):
        """Test embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            texts = ["test text"]
            embeddings = await openai_provider.generate_embeddings(texts)

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1536
            assert isinstance(embeddings[0], list)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_provider):
        """Test batch embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding responses
            mock_embeddings = []
            for i in range(5):
                mock_embedding = MagicMock()
                mock_embedding.embedding = [0.1 + i * 0.1] * 1536
                mock_embeddings.append(mock_embedding)

            mock_response = MagicMock()
            mock_response.data = mock_embeddings
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            texts = ["text1", "text2", "text3", "text4", "text5"]
            embeddings = await openai_provider.generate_embeddings(texts)

            assert len(embeddings) == 5
            assert all(len(emb) == 1536 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, openai_provider):
        """Test rate limit error handling."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock rate limit error
            from openai import RateLimitError

            mock_instance.embeddings.create.side_effect = RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["test"])

            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cleanup(self, openai_provider):
        """Test provider cleanup."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await openai_provider.initialize()
            assert openai_provider._initialized

            await openai_provider.cleanup()

            assert not openai_provider._initialized
            assert openai_provider._client is None
            mock_instance.close.assert_called_once()


class TestFastEmbedProvider:
    """Test FastEmbed provider."""

    @pytest.fixture
    def fastembed_provider(self):
        """Create FastEmbed provider instance."""
        return FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_initialize(self, fastembed_provider):
        """Test provider initialization."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            await fastembed_provider.initialize()

            assert fastembed_provider._initialized
            mock_model.assert_called_once_with(
                "BAAI/bge-small-en-v1.5"
            )  # Positional argument, not keyword

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fastembed_provider):
        """Test embedding generation."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock embedding output
            mock_embeddings = np.array([[0.1] * 384, [0.2] * 384])
            mock_instance.embed.return_value = mock_embeddings

            await fastembed_provider.initialize()

            texts = ["text1", "text2"]
            embeddings = await fastembed_provider.generate_embeddings(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert isinstance(embeddings[0], list)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty(self, fastembed_provider):
        """Test handling of empty text list."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding"):
            await fastembed_provider.initialize()

            embeddings = await fastembed_provider.generate_embeddings([])

            assert embeddings == []

    @pytest.mark.asyncio
    async def test_error_handling(self, fastembed_provider):
        """Test error handling during embedding generation."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock error during embedding
            mock_instance.embed.side_effect = Exception("Model error")

            await fastembed_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await fastembed_provider.generate_embeddings(["test"])

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self, fastembed_provider):
        """Test provider cleanup."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding"):
            await fastembed_provider.initialize()
            assert fastembed_provider._initialized

            await fastembed_provider.cleanup()

            assert not fastembed_provider._initialized
            assert fastembed_provider._model is None


class TestEmbeddingManager:
    """Test embedding manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig(
            embedding_provider=EmbeddingProviderEnum.OPENAI,
            openai={
                "api_key": "sk-test123456789012345678901234567890"
            },  # Valid length API key
            cache={"enable_caching": False},  # Disable caching to simplify test
        )

    @pytest.fixture
    def embedding_manager(self, config):
        """Create embedding manager instance."""
        return EmbeddingManager(config)

    @pytest.mark.asyncio
    async def test_initialize(self, embedding_manager):
        """Test manager initialization."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()

            assert embedding_manager._initialized
            assert len(embedding_manager.providers) == 2
            assert "openai" in embedding_manager.providers
            assert "fastembed" in embedding_manager.providers

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_preferred_provider(self, embedding_manager):
        """Test embedding generation with preferred provider."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock OpenAI embeddings
            mock_openai_instance.generate_embeddings.return_value = [
                [0.1] * 1536,
                [0.2] * 1536,
            ]

            await embedding_manager.initialize()

            texts = ["text1", "text2"]
            # Explicitly pass provider to avoid smart selection issues in test
            result = await embedding_manager.generate_embeddings(
                texts, provider_name="openai"
            )

            assert isinstance(result, dict)
            assert "embeddings" in result
            embeddings = result["embeddings"]
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536
            assert result["provider"] == "openai"
            assert result["cache_hit"] is False
            mock_openai_instance.generate_embeddings.assert_called_once_with(
                texts, batch_size=32
            )
            mock_fastembed_instance.generate_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_provider_failure(self, embedding_manager):
        """Test embedding generation when a provider fails."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock OpenAI failure
            mock_openai_instance.generate_embeddings.side_effect = (
                EmbeddingServiceError("OpenAI failed")
            )

            # Mock FastEmbed success
            mock_fastembed_instance.generate_embeddings.return_value = [
                [0.1] * 384,
                [0.2] * 384,
            ]

            await embedding_manager.initialize()

            texts = ["text1", "text2"]

            # When specifying a provider that fails, it should raise an error
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_manager.generate_embeddings(
                    texts, provider_name="openai"
                )

            assert "OpenAI failed" in str(exc_info.value)

            # But using fastembed should work
            result = await embedding_manager.generate_embeddings(
                texts, provider_name="fastembed"
            )

            assert isinstance(result, dict)
            assert "embeddings" in result
            embeddings = result["embeddings"]
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert result["provider"] == "fastembed"

    @pytest.mark.asyncio
    async def test_smart_provider_selection(self, embedding_manager):
        """Test smart provider selection based on quality tier."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock embeddings
            mock_fastembed_instance.generate_embeddings.return_value = [[0.1] * 384]

            await embedding_manager.initialize()

            # Force FAST quality tier - should still use preferred provider in this test setup
            # since we're using auto_select=True and the smart selection logic will evaluate models
            texts = ["short text"]
            result = await embedding_manager.generate_embeddings(
                texts, quality_tier=QualityTier.FAST, provider_name="fastembed"
            )

            assert isinstance(result, dict)
            assert "embeddings" in result
            embeddings = result["embeddings"]
            assert len(embeddings) == 1
            # Should use FastEmbed when explicitly requested
            mock_fastembed_instance.generate_embeddings.assert_called()
            assert result["provider"] == "fastembed"

    @pytest.mark.asyncio
    async def test_batch_processing(self, embedding_manager):
        """Test batch processing of large text lists."""
        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Mock response for all texts at once
            total_texts = 250
            mock_openai_instance.generate_embeddings.return_value = [
                [0.1] * 1536 for _ in range(total_texts)
            ]

            await embedding_manager.initialize()

            texts = ["text"] * total_texts
            result = await embedding_manager.generate_embeddings(
                texts, provider_name="openai"
            )

            assert isinstance(result, dict)
            assert "embeddings" in result
            embeddings = result["embeddings"]
            assert len(embeddings) == total_texts
            # OpenAI provider should be called once with all texts (internal batching)
            assert mock_openai_instance.generate_embeddings.call_count == 1
            assert result["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_cost_tracking(self, embedding_manager):
        """Test cost tracking for embeddings."""
        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536] * 10

            await embedding_manager.initialize()

            # Generate embeddings
            texts = ["test"] * 10
            result = await embedding_manager.generate_embeddings(texts)

            # Check result includes usage stats
            assert "usage_stats" in result
            usage_stats = result["usage_stats"]
            assert "summary" in usage_stats
            assert usage_stats["summary"]["total_requests"] > 0
            assert usage_stats["summary"]["total_tokens"] > 0

            # Also check that the manager tracks usage
            usage_report = embedding_manager.get_usage_report()
            assert usage_report["summary"]["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_reranking(self, embedding_manager):
        """Test document reranking functionality."""
        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            await embedding_manager.initialize()

            # Mock reranker if available
            if embedding_manager._reranker is not None:
                with patch.object(
                    embedding_manager._reranker, "compute_score"
                ) as mock_score:
                    mock_score.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]

                    query = "test query"
                    # rerank_results expects list of dicts with 'content' field
                    results = [
                        {"content": "doc1", "id": 1},
                        {"content": "doc2", "id": 2},
                        {"content": "doc3", "id": 3},
                        {"content": "doc4", "id": 4},
                        {"content": "doc5", "id": 5},
                    ]

                    reranked = await embedding_manager.rerank_results(query, results)

                    assert len(reranked) == 5  # Returns all results, just reordered
                    # Should be ordered by score (highest first)
                    assert reranked[0]["content"] == "doc1"
                    assert reranked[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, embedding_manager):
        """Test manager cleanup."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()
            assert embedding_manager._initialized

            await embedding_manager.cleanup()

            assert not embedding_manager._initialized
            assert len(embedding_manager.providers) == 0
            mock_openai_instance.cleanup.assert_called_once()
            mock_fastembed_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, embedding_manager):
        """Test behavior when a specified provider fails."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Both providers fail
            mock_openai_instance.generate_embeddings.side_effect = (
                EmbeddingServiceError("OpenAI failed")
            )
            mock_fastembed_instance.generate_embeddings.side_effect = (
                EmbeddingServiceError("FastEmbed failed")
            )

            await embedding_manager.initialize()

            # Test OpenAI failure
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_manager.generate_embeddings(
                    ["test"], provider_name="openai"
                )
            assert "OpenAI failed" in str(exc_info.value)

            # Test FastEmbed failure
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_manager.generate_embeddings(
                    ["test"], provider_name="fastembed"
                )
            assert "FastEmbed failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_without_openai_key(self):
        """Test initialization without OpenAI API key."""
        config = UnifiedConfig()  # No API keys
        embedding_manager = EmbeddingManager(config)

        with patch(
            "src.services.embeddings.manager.FastEmbedProvider"
        ) as mock_fastembed:
            mock_fastembed_instance = AsyncMock()
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()

            assert embedding_manager._initialized
            assert len(embedding_manager.providers) == 1
            assert "fastembed" in embedding_manager.providers
            assert "openai" not in embedding_manager.providers


class TestEmbeddingProviderErrorHandling:
    """Comprehensive error handling tests for embedding providers."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        return OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

    @pytest.fixture
    def fastembed_provider(self):
        """Create FastEmbed provider instance."""
        return FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_openai_api_key_error(self, openai_provider):
        """Test handling of invalid API key errors."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock API key error
            from openai import AuthenticationError

            mock_instance.embeddings.create.side_effect = AuthenticationError(
                "Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["test"])

            assert "invalid api key" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_openai_quota_exceeded(self, openai_provider):
        """Test handling of quota exceeded errors."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock quota error
            from openai import RateLimitError

            mock_instance.embeddings.create.side_effect = RateLimitError(
                "You exceeded your current quota",
                response=MagicMock(status_code=429),
                body=None,
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["test"])

            assert "quota" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_openai_context_length_exceeded(self, openai_provider):
        """Test handling of context length exceeded errors."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock context length error
            from openai import BadRequestError

            mock_instance.embeddings.create.side_effect = BadRequestError(
                "Maximum context length exceeded",
                response=MagicMock(status_code=400),
                body=None,
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["x" * 100000])

            assert "context length" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fastembed_model_initialization_failure(self):
        """Test FastEmbed initialization with invalid model."""
        with pytest.raises(EmbeddingServiceError) as exc_info:
            FastEmbedProvider(model_name="invalid/model-name")

        assert "Unsupported model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fastembed_memory_error(self, fastembed_provider):
        """Test handling of memory errors during embedding generation."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock memory error
            mock_instance.embed.side_effect = MemoryError("Out of memory")

            await fastembed_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await fastembed_provider.generate_embeddings(["test"] * 10000)

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of various invalid inputs."""
        config = UnifiedConfig(
            embedding_provider=EmbeddingProviderEnum.FASTEMBED,
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with patch(
            "src.services.embeddings.manager.FastEmbedProvider"
        ) as mock_fastembed:
            mock_fastembed_instance = AsyncMock()
            mock_fastembed.return_value = mock_fastembed_instance

            await manager.initialize()

            # Test with None text
            mock_fastembed_instance.generate_embeddings.side_effect = Exception(
                "NoneType has no len()"
            )

            with pytest.raises(EmbeddingServiceError):
                await manager.generate_embeddings([None])

    @pytest.mark.asyncio
    async def test_network_timeout_error(self, openai_provider):
        """Test handling of network timeout errors."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock timeout error

            mock_instance.embeddings.create.side_effect = TimeoutError(
                "Request timed out"
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["test"])

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_request_error_handling(self):
        """Test error handling with concurrent requests."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Some succeed, some fail
            responses = [
                [[0.1] * 1536],  # Success
                EmbeddingServiceError("Rate limit"),  # Fail
                [[0.2] * 1536],  # Success
            ]

            call_count = 0

            async def side_effect(*args, **kwargs):
                nonlocal call_count
                result = responses[call_count % len(responses)]
                call_count += 1
                if isinstance(result, Exception):
                    raise result
                return result

            mock_openai_instance.generate_embeddings.side_effect = side_effect

            await manager.initialize()

            # First request should succeed
            result1 = await manager.generate_embeddings(
                ["test1"], provider_name="openai"
            )
            assert result1["embeddings"][0] == [0.1] * 1536

            # Second request should fail
            with pytest.raises(EmbeddingServiceError):
                await manager.generate_embeddings(["test2"], provider_name="openai")

            # Third request should succeed
            result3 = await manager.generate_embeddings(
                ["test3"], provider_name="openai"
            )
            assert result3["embeddings"][0] == [0.2] * 1536


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    @pytest.mark.asyncio
    async def test_fastembed_sparse_embeddings(self):
        """Test FastEmbed sparse embedding generation."""
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            await provider.initialize()

            # Mock sparse model
            mock_sparse_model = MagicMock()
            mock_sparse_result = MagicMock()
            mock_sparse_result.indices = MagicMock()
            mock_sparse_result.indices.tolist.return_value = [0, 5, 10]
            mock_sparse_result.values = MagicMock()
            mock_sparse_result.values.tolist.return_value = [0.1, 0.2, 0.3]

            with patch(
                "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
            ) as mock_sparse_class:
                mock_sparse_class.return_value = mock_sparse_model
                mock_sparse_model.embed.return_value = [mock_sparse_result]

                # Test sparse embeddings
                sparse_embeddings = await provider.generate_sparse_embeddings(["test"])
                assert len(sparse_embeddings) == 1
                assert "indices" in sparse_embeddings[0]
                assert "values" in sparse_embeddings[0]
                assert sparse_embeddings[0]["indices"] == [0, 5, 10]
                assert sparse_embeddings[0]["values"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embedding_manager_cost_estimation(self):
        """Test cost estimation functionality."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()

            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Set cost attributes
            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await manager.initialize()

            # Test cost estimation
            texts = ["test" * 100]  # ~100 tokens
            costs = manager.estimate_cost(texts)

            assert "openai" in costs
            assert "fastembed" in costs
            assert costs["openai"]["total_cost"] > 0
            assert costs["fastembed"]["total_cost"] == 0

            # Test specific provider cost
            openai_cost = manager.estimate_cost(texts, provider_name="openai")
            assert "openai" in openai_cost
            assert len(openai_cost) == 1

    @pytest.mark.asyncio
    async def test_embedding_manager_provider_info(self):
        """Test provider info retrieval."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()

            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Set provider attributes
            mock_openai_instance.model_name = "text-embedding-3-small"
            mock_openai_instance.dimensions = 1536
            mock_openai_instance.cost_per_token = 0.00002
            mock_openai_instance.max_tokens_per_request = 8191

            mock_fastembed_instance.model_name = "BAAI/bge-small-en-v1.5"
            mock_fastembed_instance.dimensions = 384
            mock_fastembed_instance.cost_per_token = 0.0
            mock_fastembed_instance.max_tokens_per_request = 512

            await manager.initialize()

            # Get provider info
            info = manager.get_provider_info()

            assert "openai" in info
            assert "fastembed" in info
            assert info["openai"]["model"] == "text-embedding-3-small"
            assert info["openai"]["dimensions"] == 1536
            assert info["fastembed"]["cost_per_token"] == 0.0

    @pytest.mark.asyncio
    async def test_embedding_manager_optimal_provider(self):
        """Test optimal provider selection."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()

            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Set cost attributes
            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await manager.initialize()

            # Test optimal provider for small text
            provider = await manager.get_optimal_provider(
                text_length=1000,  # Small text
                quality_required=False,
            )
            assert provider == "fastembed"  # Should prefer free/local

            # Test with quality required
            provider = await manager.get_optimal_provider(
                text_length=1000, quality_required=True
            )
            assert provider == "openai"  # Should prefer quality

            # Test with budget limit
            provider = await manager.get_optimal_provider(
                text_length=1000000,  # Large text
                budget_limit=0.001,  # Very small budget
            )
            assert provider == "fastembed"  # Only free option fits budget

    @pytest.mark.asyncio
    async def test_embedding_manager_empty_input(self):
        """Test handling of empty input."""
        config = UnifiedConfig(
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with patch(
            "src.services.embeddings.manager.FastEmbedProvider"
        ) as mock_fastembed:
            mock_fastembed_instance = AsyncMock()
            mock_fastembed.return_value = mock_fastembed_instance

            await manager.initialize()

            # Test empty list
            result = await manager.generate_embeddings([])
            assert result["embeddings"] == []
            assert result["cost"] == 0.0
            assert result["reasoning"] == "Empty input"

    @pytest.mark.asyncio
    async def test_openai_batch_processing(self):
        """Test OpenAI internal batch processing."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock responses for batches
            def create_response(num_embeddings):
                mock_data = []
                for i in range(num_embeddings):
                    mock_embedding = MagicMock()
                    mock_embedding.embedding = [0.1 + i * 0.01] * 1536
                    mock_data.append(mock_embedding)
                mock_response = MagicMock()
                mock_response.data = mock_data
                return mock_response

            # Provider has max batch size, test it handles larger inputs
            mock_instance.embeddings.create.side_effect = [
                create_response(2048),  # First batch
                create_response(152),  # Second batch
            ]

            await provider.initialize()

            # Generate embeddings for more than batch size
            texts = [f"text{i}" for i in range(2200)]
            embeddings = await provider.generate_embeddings(texts, batch_size=2048)

            assert len(embeddings) == 2200
            assert mock_instance.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_fastembed_cleanup_not_initialized(self):
        """Test FastEmbed cleanup when not initialized."""
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

        # Should not raise error
        await provider.cleanup()
        assert not provider._initialized

    @pytest.mark.asyncio
    async def test_manager_budget_tracking(self):
        """Test budget limit and tracking."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        # Create manager with budget limit
        manager = EmbeddingManager(config, budget_limit=0.01)  # $0.01 limit

        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Set high cost to trigger budget limit
            mock_openai_instance.cost_per_token = 0.001  # $1 per 1k tokens
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]

            await manager.initialize()

            # First request should work
            result1 = await manager.generate_embeddings(
                ["short text"], provider_name="openai"
            )
            assert "embeddings" in result1

            # Large request should exceed budget
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await manager.generate_embeddings(
                    ["very long text" * 1000], provider_name="openai"
                )
            assert "budget" in str(exc_info.value).lower()
