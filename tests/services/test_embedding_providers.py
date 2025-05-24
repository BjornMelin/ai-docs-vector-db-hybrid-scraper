"""Tests for embedding providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.services.config import APIConfig
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
            mock_embedding.embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            embeddings = await openai_provider.generate_embeddings(
                ["test text"],
                batch_size=1,
            )

            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_provider):
        """Test batch embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding responses for batch processing
            # First batch: 2 embeddings
            mock_response1 = MagicMock()
            mock_response1.data = [
                MagicMock(embedding=[0.0, 0.0, 0.0]),
                MagicMock(embedding=[0.1, 0.2, 0.3]),
            ]

            # Second batch: 1 embedding
            mock_response2 = MagicMock()
            mock_response2.data = [
                MagicMock(embedding=[0.2, 0.4, 0.6]),
            ]

            # Configure mock to return different responses for each call
            mock_instance.embeddings.create.side_effect = [
                mock_response1,
                mock_response2,
            ]

            await openai_provider.initialize()

            texts = ["text1", "text2", "text3"]
            embeddings = await openai_provider.generate_embeddings(
                texts,
                batch_size=2,
            )

            assert len(embeddings) == 3
            assert embeddings[0] == [0.0, 0.0, 0.0]
            assert embeddings[1] == [0.1, 0.2, 0.3]
            assert embeddings[2] == [0.2, 0.4, 0.6]
            assert mock_instance.embeddings.create.call_count == 2  # 2 batches

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_api(self, openai_provider):
        """Test batch API for embeddings."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock file operations
            mock_file_response = MagicMock(id="file-123")
            mock_instance.files.create.return_value = mock_file_response

            # Mock batch response
            mock_batch_response = MagicMock(id="batch-456")
            mock_instance.batches.create.return_value = mock_batch_response

            await openai_provider.initialize()

            batch_id = await openai_provider.generate_embeddings_batch_api(
                ["text1", "text2"], custom_ids=["id1", "id2"]
            )

            assert batch_id == "batch-456"
            mock_instance.files.create.assert_called_once()
            mock_instance.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, openai_provider):
        """Test embedding generation error handling."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock error response
            mock_instance.embeddings.create.side_effect = Exception("API Error")

            await openai_provider.initialize()

            with pytest.raises(
                EmbeddingServiceError, match="Failed to generate embeddings"
            ):
                await openai_provider.generate_embeddings(["test"])

    def test_cost_per_token(self, openai_provider):
        """Test cost calculation."""
        cost = openai_provider.cost_per_token
        assert cost == 0.02 / 1_000_000  # $0.02 per 1M tokens

    def test_unsupported_model(self):
        """Test unsupported model error."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(
                api_key="sk-test",
                model_name="invalid-model",
            )

    @pytest.mark.asyncio
    async def test_batch_api_error(self, openai_provider):
        """Test batch API error handling."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock file creation to fail
            mock_instance.files.create.side_effect = Exception("File upload failed")

            await openai_provider.initialize()

            with pytest.raises(
                EmbeddingServiceError, match="Failed to create batch job"
            ):
                await openai_provider.generate_embeddings_batch_api(
                    ["text1", "text2"], custom_ids=["id1", "id2"]
                )

    @pytest.mark.asyncio
    async def test_batch_api_temp_file_cleanup_error(self, openai_provider):
        """Test batch API temp file cleanup with OSError."""
        with (
            patch("src.services.embeddings.openai_provider.AsyncOpenAI") as mock_client,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.unlink") as mock_unlink,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock temp file
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.jsonl"
            mock_temp.return_value.__enter__.return_value = mock_file

            # Mock file upload to fail
            mock_instance.files.create.side_effect = Exception("Upload failed")

            # Mock unlink to raise OSError
            mock_unlink.side_effect = OSError("Permission denied")

            await openai_provider.initialize()

            # This should not raise despite the cleanup error
            with pytest.raises(
                EmbeddingServiceError, match="Failed to create batch job"
            ):
                await openai_provider.generate_embeddings_batch_api(["text1"])

    @pytest.mark.asyncio
    async def test_cleanup(self, openai_provider):
        """Test provider cleanup."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.close = AsyncMock()  # Mock async close method
            mock_client.return_value = mock_instance

            await openai_provider.initialize()
            await openai_provider.cleanup()

            mock_instance.close.assert_called_once()
            assert not openai_provider._initialized

    def test_model_properties(self, openai_provider):
        """Test model properties."""
        assert openai_provider.model_name == "text-embedding-3-small"
        assert openai_provider.dimensions == 1536
        assert openai_provider.max_tokens_per_request == 8191
        assert openai_provider.cost_per_token > 0

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, openai_provider):
        """Test error when provider not initialized."""
        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await openai_provider.generate_embeddings(["test"])


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
            mock_model.assert_called_once_with("BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fastembed_provider):
        """Test embedding generation."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock embedding result
            mock_instance.embed.return_value = [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6]),
            ]

            await fastembed_provider.initialize()

            embeddings = await fastembed_provider.generate_embeddings(
                ["text1", "text2"]
            )

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_cost_per_token(self, fastembed_provider):
        """Test cost calculation (should be 0 for local)."""
        assert fastembed_provider.cost_per_token == 0.0

    def test_unsupported_model(self):
        """Test unsupported model error."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            FastEmbedProvider(model_name="invalid-model")

    def test_list_available_models(self):
        """Test listing available models."""
        models = FastEmbedProvider.list_available_models()
        assert "BAAI/bge-small-en-v1.5" in models
        assert len(models) > 5

    @pytest.mark.asyncio
    async def test_cleanup(self, fastembed_provider):
        """Test provider cleanup."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            await fastembed_provider.initialize()
            await fastembed_provider.cleanup()

            assert not fastembed_provider._initialized

    def test_model_properties(self, fastembed_provider):
        """Test model properties."""
        assert fastembed_provider.model_name == "BAAI/bge-small-en-v1.5"
        assert fastembed_provider.dimensions == 384
        assert fastembed_provider.max_tokens_per_request == 512
        assert fastembed_provider.cost_per_token == 0.0

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, fastembed_provider):
        """Test error when provider not initialized."""
        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await fastembed_provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, fastembed_provider):
        """Test embedding generation error."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock embed to raise exception
            mock_instance.embed.side_effect = Exception("Model error")

            await fastembed_provider.initialize()

            with pytest.raises(
                EmbeddingServiceError, match="Failed to generate embeddings"
            ):
                await fastembed_provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_initialize_error(self, fastembed_provider):
        """Test initialization error."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            # Mock to raise error on creation
            mock_model.side_effect = Exception("Model load failed")

            with pytest.raises(
                EmbeddingServiceError, match="Failed to initialize FastEmbed"
            ):
                await fastembed_provider.initialize()


class TestEmbeddingManager:
    """Test embedding manager."""

    @pytest.fixture
    def api_config(self):
        """Create test API config."""
        return APIConfig(
            openai_api_key="sk-test-key",
            enable_local_embeddings=True,
        )

    @pytest.fixture
    def embedding_manager(self, api_config):
        """Create embedding manager instance."""
        return EmbeddingManager(api_config)

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
    async def test_generate_embeddings_with_quality_tier(self, embedding_manager):
        """Test embedding generation with quality tier."""
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

            # Mock embeddings
            mock_openai_instance.generate_embeddings.return_value = [[0.1, 0.2]]
            mock_fastembed_instance.generate_embeddings.return_value = [[0.3, 0.4]]

            await embedding_manager.initialize()

            # Test FAST tier (should use fastembed)
            await embedding_manager.generate_embeddings(
                ["test"],
                quality_tier=QualityTier.FAST,
            )
            mock_fastembed_instance.generate_embeddings.assert_called_once()

            # Test BEST tier (should use openai)
            await embedding_manager.generate_embeddings(
                ["test"],
                quality_tier=QualityTier.BEST,
            )
            mock_openai_instance.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimate_cost(self, embedding_manager):
        """Test cost estimation."""
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

            # Mock cost properties
            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await embedding_manager.initialize()

            costs = embedding_manager.estimate_cost(["test" * 100])

            assert "openai" in costs
            assert "fastembed" in costs
            assert costs["openai"]["total_cost"] > 0
            assert costs["fastembed"]["total_cost"] == 0

    @pytest.mark.asyncio
    async def test_get_optimal_provider(self, embedding_manager):
        """Test optimal provider selection."""
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

            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await embedding_manager.initialize()

            # Small text should prefer local
            provider = await embedding_manager.get_optimal_provider(
                text_length=1000,
                quality_required=False,
            )
            assert provider == "fastembed"

            # Quality required should prefer OpenAI
            provider = await embedding_manager.get_optimal_provider(
                text_length=1000,
                quality_required=True,
            )
            assert provider == "openai"

    @pytest.mark.asyncio
    async def test_no_providers_available(self):
        """Test error when no providers available."""
        config = APIConfig(
            openai_api_key=None,
            enable_local_embeddings=False,
        )
        manager = EmbeddingManager(config)

        with pytest.raises(
            EmbeddingServiceError, match="No embedding providers available"
        ):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self, embedding_manager):
        """Test error when manager not initialized."""
        with pytest.raises(EmbeddingServiceError, match="Manager not initialized"):
            await embedding_manager.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_provider(self, embedding_manager):
        """Test embedding generation with specific provider."""
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

            # Mock embeddings
            mock_openai_instance.generate_embeddings.return_value = [[0.1, 0.2]]
            mock_fastembed_instance.generate_embeddings.return_value = [[0.3, 0.4]]

            await embedding_manager.initialize()

            # Test with specific provider via provider_name parameter
            embeddings = await embedding_manager.generate_embeddings(
                ["test"], provider_name="openai"
            )

            assert embeddings == [[0.1, 0.2]]
            mock_openai_instance.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_invalid_provider(self, embedding_manager):
        """Test error with invalid provider."""
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

            await embedding_manager.initialize()

            # When provider_name is invalid, it should raise an error
            with pytest.raises(
                EmbeddingServiceError, match="Provider 'invalid' not available"
            ):
                await embedding_manager.generate_embeddings(
                    ["test"], provider_name="invalid"
                )

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
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()
            await embedding_manager.cleanup()

            mock_openai_instance.cleanup.assert_called_once()
            mock_fastembed_instance.cleanup.assert_called_once()
            assert not embedding_manager._initialized

    @pytest.mark.asyncio
    async def test_manual_init_cleanup(self, embedding_manager):
        """Test manual initialization and cleanup."""
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

            # Initialize
            await embedding_manager.initialize()
            assert embedding_manager._initialized

            # Cleanup
            await embedding_manager.cleanup()
            mock_openai_instance.cleanup.assert_called_once()
            mock_fastembed_instance.cleanup.assert_called_once()
            assert not embedding_manager._initialized

    @pytest.mark.asyncio
    async def test_get_provider_info(self, embedding_manager):
        """Test getting provider information."""
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

            # Mock provider properties
            mock_openai_instance.model_name = "text-embedding-3-small"
            mock_openai_instance.dimensions = 1536
            mock_openai_instance.cost_per_token = 0.00002
            mock_openai_instance.max_tokens_per_request = 8191

            mock_fastembed_instance.model_name = "BAAI/bge-small-en-v1.5"
            mock_fastembed_instance.dimensions = 384
            mock_fastembed_instance.cost_per_token = 0.0
            mock_fastembed_instance.max_tokens_per_request = 512

            await embedding_manager.initialize()

            info = embedding_manager.get_provider_info()

            assert "openai" in info
            assert "fastembed" in info
            assert info["openai"]["model"] == "text-embedding-3-small"
            assert info["fastembed"]["cost_per_token"] == 0.0

    @pytest.mark.asyncio
    async def test_optimal_provider_budget_limit(self, embedding_manager):
        """Test optimal provider selection with budget limit."""
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

            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await embedding_manager.initialize()

            # With very low budget and quality required, it depends on the logic
            # Let's test without quality requirement to ensure local is preferred
            provider = await embedding_manager.get_optimal_provider(
                text_length=100000,  # Large text
                quality_required=False,  # No quality requirement
                budget_limit=0.0001,  # Very small budget
            )
            assert provider == "fastembed"  # Should prefer free local provider

    @pytest.mark.asyncio
    async def test_initialize_with_only_fastembed(self):
        """Test initialization with only FastEmbed provider."""
        config = APIConfig(
            openai_api_key=None,
            enable_local_embeddings=True,
        )
        manager = EmbeddingManager(config)

        with patch(
            "src.services.embeddings.manager.FastEmbedProvider"
        ) as mock_fastembed:
            mock_fastembed_instance = AsyncMock()
            mock_fastembed.return_value = mock_fastembed_instance

            await manager.initialize()

            assert manager._initialized
            assert "fastembed" in manager.providers
            assert "openai" not in manager.providers
