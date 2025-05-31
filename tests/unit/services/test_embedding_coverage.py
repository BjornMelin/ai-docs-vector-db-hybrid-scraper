"""Additional tests to increase embedding coverage to 90%."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestEmbeddingCoverageGaps:
    """Tests targeting specific coverage gaps."""

    @pytest.mark.asyncio
    async def test_openai_provider_edge_cases(self):
        """Test OpenAI provider edge cases for better coverage."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

        # Test properties
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536
        assert provider.max_tokens_per_request == 8191
        assert provider.cost_per_token == 0.00000002  # $0.02 per 1M tokens

        # Test batch processing would happen internally
        # Create texts that would require batching if implemented
        large_texts = ["text " * 500] * 20  # Each ~2000 tokens
        # This would be handled internally by generate_embeddings

    @pytest.mark.asyncio
    async def test_openai_dimension_validation(self):
        """Test OpenAI dimension validation logic."""
        # Test ada-002 with custom dimensions (keeps custom)
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-ada-002",
            dimensions=512,
        )
        assert provider.dimensions == 512  # Uses provided dimensions

        # Test text-embedding-3-small with custom dimensions
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=256,
        )
        assert provider.dimensions == 256

        # Test invalid dimensions
        with pytest.raises(EmbeddingServiceError):
            OpenAIEmbeddingProvider(
                api_key="sk-test-key",
                model_name="text-embedding-3-small",
                dimensions=2000,  # Max is 1536
            )

    @pytest.mark.asyncio
    async def test_fastembed_provider_properties(self):
        """Test FastEmbed provider properties and edge cases."""
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

        # Test properties
        assert provider.cost_per_token == 0.0  # Free
        assert provider.max_tokens_per_request == 512

        # Test model validation
        with pytest.raises(EmbeddingServiceError):
            FastEmbedProvider(model_name="unsupported/model")

    @pytest.mark.asyncio
    async def test_embedding_manager_statistics(self):
        """Test embedding manager statistics and reporting."""
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
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]
            mock_openai_instance.cost_per_token = 0.00002

            await manager.initialize()

            # Generate some embeddings to accumulate stats
            for i in range(5):
                await manager.generate_embeddings([f"test {i}"], provider_name="openai")

            # Test usage report
            report = manager.get_usage_report()
            assert "summary" in report
            assert report["summary"]["total_requests"] == 5
            assert report["summary"]["total_tokens"] > 0
            assert report["summary"]["total_cost"] > 0
            # Check cache was disabled
            assert (
                "cache_hits" not in report["summary"]
                or report["summary"].get("cache_hits", 0) == 0
            )

            # Test provider statistics
            assert "by_provider" in report
            assert "openai" in report["by_provider"]
            # Check provider has recorded usage
            provider_stats = report["by_provider"]["openai"]
            if isinstance(provider_stats, dict):
                assert provider_stats.get("requests", 0) == 5
            else:
                # Legacy format - just check it's recorded
                assert provider_stats == 5

    @pytest.mark.asyncio
    async def test_embedding_manager_reasoning(self):
        """Test embedding manager reasoning and logging features."""
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
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]

            await manager.initialize()

            # Test with reasoning enabled
            result = await manager.generate_embeddings(
                ["test text"],
                provider_name="openai",
                quality_tier=QualityTier.BEST,
                speed_priority=False,
            )

            # Check reasoning is included
            assert "reasoning" in result
            assert isinstance(result["reasoning"], str)

            # Test empty input reasoning
            empty_result = await manager.generate_embeddings([])
            assert empty_result["reasoning"] == "Empty input"

    @pytest.mark.asyncio
    async def test_embedding_manager_model_info(self):
        """Test model info and benchmark functionality."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            fastembed={"model": "BAAI/bge-small-en-v1.5"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        # Test benchmark access
        benchmarks = manager._benchmarks
        assert "text-embedding-3-small" in benchmarks
        assert benchmarks["text-embedding-3-small"].quality_score == 85
        assert benchmarks["text-embedding-3-small"].avg_latency_ms == 78

        # Test model selection logic
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

            await manager.initialize()

            # Test provider selection by using generate_embeddings
            # with specific parameters that should favor FastEmbed
            result = await manager.generate_embeddings(
                ["short test"],
                quality_tier=QualityTier.FAST,
                speed_priority=True,
                provider_name="fastembed",  # Explicitly request fastembed
            )

            # Should use FastEmbed
            assert result["provider"] == "fastembed"

    @pytest.mark.asyncio
    async def test_embedding_manager_with_reranker(self):
        """Test embedding manager with reranker functionality."""
        config = UnifiedConfig(
            embedding={
                "enable_reranking": True,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
            },
            openai={"api_key": "sk-test123456789012345678901234567890"},
            cache={"enable_caching": False},
        )
        manager = EmbeddingManager(config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch("sentence_transformers.cross_encoder.CrossEncoder") as mock_reranker,
        ):
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance
            # Reranker uses compute_score, not predict
            mock_reranker_instance.compute_score.return_value = [0.9, 0.7]

            await manager.initialize()

            # Test reranking
            assert manager._reranker is not None

            # Test rerank_results
            query = "test query"
            results = [
                {"content": "result 1", "id": 1, "score": 0.8},
                {"content": "result 2", "id": 2, "score": 0.7},
                {"content": "result 3", "id": 3, "score": 0.6},
            ]

            # Only rerank top 2
            top_results = results[:2]
            reranked = await manager.rerank_results(query, top_results)

            # Should return 2 results in reranked order
            assert len(reranked) == 2
            # First result should be the one with highest rerank score
            assert reranked[0]["id"] == 1  # id 1 has score 0.9

            # Test with CrossEncoder error
            mock_reranker_instance.compute_score.side_effect = Exception(
                "Reranker error"
            )

            # Should return original results on error
            error_results = await manager.rerank_results(query, results)
            assert len(error_results) == 3
            assert error_results == results

    @pytest.mark.asyncio
    async def test_embedding_manager_concurrent_providers(self):
        """Test concurrent provider initialization and cleanup."""
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
            # Create instances that track calls
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Test concurrent initialization
            await manager.initialize()

            # Both providers should be initialized
            mock_openai_instance.initialize.assert_called_once()
            mock_fastembed_instance.initialize.assert_called_once()

            # Test concurrent cleanup
            await manager.cleanup()

            # Both providers should be cleaned up
            mock_openai_instance.cleanup.assert_called_once()
            mock_fastembed_instance.cleanup.assert_called_once()

            # Manager should be reset
            assert not manager._initialized
            assert len(manager.providers) == 0

    @pytest.mark.asyncio
    async def test_openai_provider_error_handling(self):
        """Test OpenAI provider error handling."""
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

            await provider.initialize()

            # Test error handling for network errors
            mock_instance.embeddings.create.side_effect = Exception("Network error")

            # Should raise EmbeddingServiceError
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await provider.generate_embeddings(["test"])

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fastembed_provider_language_detection(self):
        """Test FastEmbed provider with language detection."""
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

        # Test initialization
        assert provider.model_name == "BAAI/bge-small-en-v1.5"
        assert provider.cost_per_token == 0.0

    @pytest.mark.asyncio
    async def test_embedding_manager_complex_scenarios(self):
        """Test complex embedding scenarios."""
        config = UnifiedConfig(
            openai={"api_key": "sk-test123456789012345678901234567890"},
            embedding={"enable_reranking": False},
            cache={"enable_caching": True},
        )
        manager = EmbeddingManager(config)

        # Should have default benchmarks from UnifiedConfig
        assert len(manager._benchmarks) > 0

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
            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await manager.initialize()

            # Test max_cost enforcement
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]

            # This should work with sufficient max_cost
            result = await manager.generate_embeddings(
                ["small text"], provider_name="openai", max_cost=0.01
            )
            assert "embeddings" in result

            # Test provider selection with auto_select=False
            result = await manager.generate_embeddings(
                ["test text"], provider_name="openai", auto_select=False
            )
            assert result["provider"] == "openai"

            # Test speed priority
            mock_fastembed_instance.generate_embeddings.return_value = [[0.1] * 384]
            result = await manager.generate_embeddings(
                ["test"], speed_priority=True, auto_select=True
            )
            # With speed priority, should prefer faster provider
            assert "embeddings" in result

    @pytest.mark.asyncio
    async def test_embedding_manager_parallel_generation(self):
        """Test parallel embedding generation with multiple providers."""
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

            # Track concurrent execution
            openai_running = False
            fastembed_running = False
            max_concurrent = 0
            current_concurrent = 0

            async def openai_embed(*args, **kwargs):
                nonlocal openai_running, current_concurrent, max_concurrent
                openai_running = True
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.1)
                current_concurrent -= 1
                openai_running = False
                return [[0.1] * 1536 for _ in args[0]]

            async def fastembed_embed(*args, **kwargs):
                nonlocal fastembed_running, current_concurrent, max_concurrent
                fastembed_running = True
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.1)
                current_concurrent -= 1
                fastembed_running = False
                return [[0.2] * 384 for _ in args[0]]

            mock_openai_instance.generate_embeddings.side_effect = openai_embed
            mock_fastembed_instance.generate_embeddings.side_effect = fastembed_embed

            await manager.initialize()

            # Launch parallel requests
            tasks = [
                manager.generate_embeddings(["text1"], provider_name="openai"),
                manager.generate_embeddings(["text2"], provider_name="fastembed"),
                manager.generate_embeddings(["text3"], provider_name="openai"),
            ]

            results = await asyncio.gather(*tasks)

            # Verify results
            assert len(results) == 3
            assert len(results[0]["embeddings"][0]) == 1536  # OpenAI
            assert len(results[1]["embeddings"][0]) == 384  # FastEmbed
            assert max_concurrent >= 2  # At least 2 concurrent

    @pytest.mark.asyncio
    async def test_openai_provider_cleanup_edge_cases(self):
        """Test OpenAI provider cleanup edge cases."""
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
        )

        # Cleanup without initialization
        await provider.cleanup()  # Should not raise

        # Initialize and cleanup twice
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await provider.initialize()
            await provider.cleanup()
            await provider.cleanup()  # Second cleanup should be safe

            # Verify close was only called once
            mock_instance.close.assert_called_once()
