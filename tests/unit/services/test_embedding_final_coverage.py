"""Final tests to push embedding coverage above 90%."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.config.enums import EmbeddingProvider as EmbeddingProviderEnum
from src.config.models import ModelBenchmark
from src.config.models import UnifiedConfig
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestEmbeddingManagerInternals:
    """Test internal methods and edge cases for coverage."""

    @pytest.mark.asyncio
    async def test_manager_initialization_with_custom_benchmarks(self):
        """Test manager initialization with custom benchmarks."""
        custom_benchmarks = {
            "test-model": ModelBenchmark(
                model_name="test-model",
                provider="test",
                avg_latency_ms=50,
                quality_score=80,
                tokens_per_second=10000,
                cost_per_million_tokens=10.0,
                max_context_length=1000,
                embedding_dimensions=768,
            )
        }

        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            embedding={"model_benchmarks": custom_benchmarks},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        # Should use custom benchmarks
        assert "test-model" in manager._benchmarks
        assert manager._benchmarks["test-model"].quality_score == 80

    @pytest.mark.asyncio
    async def test_manager_smart_provider_selection(self):
        """Test smart provider selection logic."""
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

            # Test auto selection with short text and budget constraint
            mock_fastembed_instance.generate_embeddings.return_value = [[0.1] * 384]

            result = await manager.generate_embeddings(
                ["short text"],
                auto_select=True,
                quality_tier=QualityTier.FAST,
                max_cost=0.0001,  # Very low budget
            )

            # Should select a provider based on constraints
            assert "provider" in result
            assert "reasoning" in result
            assert result["provider"] in [
                "fastembed",
                "asyncmock",
            ]  # Either real or mocked

    @pytest.mark.asyncio
    async def test_manager_sparse_embeddings(self):
        """Test sparse embedding generation."""
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

            # Mock dense embeddings
            mock_fastembed_instance.generate_embeddings.return_value = [[0.1] * 384]

            # Mock sparse embeddings
            mock_fastembed_instance.generate_sparse_embeddings.return_value = [
                {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]}
            ]

            await manager.initialize()

            # Generate with sparse embeddings
            result = await manager.generate_embeddings(
                ["test text"],
                provider_name="fastembed",
                generate_sparse=True,
            )

            assert "sparse_embeddings" in result
            assert len(result["sparse_embeddings"]) == 1
            assert "indices" in result["sparse_embeddings"][0]

    @pytest.mark.asyncio
    async def test_manager_rerank_results(self):
        """Test reranking functionality."""
        config = UnifiedConfig(
            embedding={"enable_reranking": True},
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch("sentence_transformers.cross_encoder.CrossEncoder") as mock_ce,
        ):
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Mock CrossEncoder
            mock_ce_instance = MagicMock()
            mock_ce.return_value = mock_ce_instance
            mock_ce_instance.predict.return_value = np.array([0.9, 0.7, 0.5])

            await manager.initialize()

            # Test reranking
            query = "test query"
            results = [
                {"content": "result 1", "id": 1},
                {"content": "result 2", "id": 2},
                {"content": "result 3", "id": 3},
            ]

            reranked = await manager.rerank_results(query, results)

            # Should return all results in reranked order
            assert len(reranked) == 3
            # First result should be the one with highest score from predict
            assert reranked[0]["id"] == 1  # id 1 corresponds to score 0.9

    @pytest.mark.asyncio
    async def test_manager_quality_tier_selection(self):
        """Test quality tier based model selection."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        # Test quality tier mappings
        assert manager._tier_providers[QualityTier.FAST] == "fastembed"
        assert manager._tier_providers[QualityTier.BALANCED] == "fastembed"
        assert manager._tier_providers[QualityTier.BEST] == "openai"

        # Test model benchmarks are loaded
        assert len(manager._benchmarks) > 0
        assert "text-embedding-3-small" in manager._benchmarks

        # Test smart config is loaded
        assert manager._smart_config is not None
        assert hasattr(manager._smart_config, "quality_weight")
        assert hasattr(manager._smart_config, "speed_weight")
        assert hasattr(manager._smart_config, "cost_weight")

    @pytest.mark.asyncio
    async def test_openai_batch_api(self):
        """Test OpenAI batch API methods."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock file upload
            mock_file_response = MagicMock()
            mock_file_response.id = "file-123"
            mock_instance.files.create.return_value = mock_file_response

            # Mock batch creation
            mock_batch_response = MagicMock()
            mock_batch_response.id = "batch-456"
            mock_instance.batches.create.return_value = mock_batch_response

            await provider.initialize()

            # Test batch API submission
            batch_id = await provider.generate_embeddings_batch_api(
                ["text1", "text2", "text3"],
                custom_ids=["id1", "id2", "id3"],
            )

            assert batch_id == "batch-456"
            mock_instance.files.create.assert_called_once()
            mock_instance.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_fastembed_sparse_embeddings(self):
        """Test FastEmbed sparse embedding generation."""
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_text_embed:
            mock_instance = MagicMock()
            mock_text_embed.return_value = mock_instance

            # Mock dense embeddings
            mock_instance.embed.return_value = np.array([[0.1] * 384])

            await provider.initialize()

            # Test with sparse model
            with patch(
                "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
            ) as mock_sparse:
                mock_sparse_instance = MagicMock()
                mock_sparse.return_value = mock_sparse_instance

                # Mock sparse result
                mock_sparse_result = MagicMock()
                mock_sparse_result.indices = MagicMock()
                mock_sparse_result.indices.tolist.return_value = [1, 5, 10]
                mock_sparse_result.values = MagicMock()
                mock_sparse_result.values.tolist.return_value = [0.1, 0.2, 0.3]

                mock_sparse_instance.embed.return_value = [mock_sparse_result]

                # Generate sparse embeddings
                sparse = await provider.generate_sparse_embeddings(["test"])

                assert len(sparse) == 1
                assert sparse[0]["indices"] == [1, 5, 10]
                assert sparse[0]["values"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_openai_provider_initialization_error(self):
        """Test OpenAI provider initialization error handling."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            # Make client creation fail
            mock_client.side_effect = Exception("Failed to create client")

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.initialize()

            assert "Failed to initialize OpenAI client" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fastembed_supported_models(self):
        """Test FastEmbed supported models check."""
        # Test all supported models
        supported_models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "jinaai/jina-embeddings-v2-small-en",
        ]

        for model in supported_models:
            provider = FastEmbedProvider(model_name=model)
            assert provider.model_name == model

    @pytest.mark.asyncio
    async def test_manager_cleanup_with_cache_error(self):
        """Test manager cleanup when cache cleanup fails."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": True},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch("src.services.cache.manager.CacheManager") as mock_cache_manager,
        ):
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Mock cache manager that fails on cleanup
            mock_cache_instance = AsyncMock()
            mock_cache_manager.return_value = mock_cache_instance
            mock_cache_instance.cleanup.side_effect = Exception("Cache cleanup failed")

            await manager.initialize()

            # Cleanup should succeed even if cache cleanup fails
            await manager.cleanup()

            # Provider should still be cleaned up
            mock_openai_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_text_analysis_edge_cases(self):
        """Test text analysis with edge cases."""
        config = UnifiedConfig(
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        # Test with None values in list
        analysis = manager.analyze_text_characteristics([None, "test", None])
        assert analysis.total_length > 0
        assert analysis.text_type != "empty"

        # Test with code detection
        code_text = ["def function():\n    return True", "class MyClass:\n    pass"]
        analysis = manager.analyze_text_characteristics(code_text)
        assert analysis.text_type == "code"
        assert analysis.requires_high_quality is True

        # Test with mixed content
        mixed = ["Short", "This is a medium length text here", "def test(): pass"]
        analysis = manager.analyze_text_characteristics(mixed)
        assert analysis.complexity_score > 0

    @pytest.mark.asyncio
    async def test_openai_batch_api_cleanup(self):
        """Test OpenAI batch API with cleanup on error."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Make file upload fail
            mock_instance.files.create.side_effect = Exception("Upload failed")

            await provider.initialize()

            # Test batch API submission with failure
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings_batch_api(
                    ["text1", "text2"],
                )

            assert "Failed to create batch job" in str(exc_info.value)
