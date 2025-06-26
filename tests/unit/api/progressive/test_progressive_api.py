"""Comprehensive tests for the progressive API system.

This test suite demonstrates sophisticated testing patterns
for the progressive API design, showcasing both behavior
validation and type safety verification.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.progressive import (
    AIDocSystem,
    AIDocSystemBuilder,
    EmbeddingOptions,
    FeatureDiscovery,
    QualityTier,
    SearchOptions,
    SearchStrategy,
    create_system,
)
from src.api.progressive.protocols import EmbeddingProtocol, SearchProtocol
from src.api.progressive.simple import SimpleSearchResult
from src.api.progressive.types import ProgressiveResponse


class TestProgressiveAPIDesign:
    """Test the progressive API design patterns.

    These tests validate that the API provides immediate value
    while progressively revealing sophisticated features.
    """

    def test_30_second_success_pattern(self):
        """Test that users can get immediate value with minimal setup."""
        # One-line setup should work immediately
        system = AIDocSystem.quick_start()

        # Should have sensible defaults
        assert system.embedding_provider == "fastembed"
        assert system.quality_tier == "balanced"
        assert system.enable_cache is True
        assert system.enable_monitoring is False  # Minimal for quick start

        # Should be ready to use
        assert not system._initialized  # Lazy initialization
        assert system.workspace_dir == Path.cwd()

    def test_progressive_builder_pattern(self):
        """Test that builder pattern reveals features progressively."""
        # Start with builder
        builder = AIDocSystem.builder()
        assert isinstance(builder, AIDocSystemBuilder)

        # Chain configuration progressively
        system = (
            builder.with_embedding_provider("openai")
            .with_quality_tier("best")
            .with_cache(enabled=True, ttl=3600)
            .with_monitoring(enabled=True, track_costs=True)
            .build()
        )

        # Verify configuration applied
        assert system.embedding_provider == "openai"
        assert system.quality_tier == "best"
        assert system.enable_cache is True
        assert system.enable_monitoring is True

    def test_builder_fluent_interface(self):
        """Test that builder provides fluent interface."""
        builder = AIDocSystem.builder()

        # Each method should return builder for chaining
        result1 = builder.with_embedding_provider("fastembed")
        assert result1 is builder

        result2 = builder.with_quality_tier("fast")
        assert result2 is builder

        result3 = builder.with_cache(enabled=True)
        assert result3 is builder

    def test_factory_pattern_convenience(self):
        """Test factory function provides convenient creation."""
        # Basic factory usage
        system1 = create_system()
        assert isinstance(system1, AIDocSystem)
        assert system1.embedding_provider == "fastembed"

        # Factory with options
        system2 = create_system(provider="openai", quality="best", enable_cache=True)
        assert system2.embedding_provider == "openai"
        assert system2.quality_tier == "best"
        assert system2.enable_cache is True

    @pytest.mark.asyncio
    async def test_async_context_manager_pattern(self):
        """Test that async context manager handles lifecycle correctly."""
        system = AIDocSystem.quick_start()

        # Should not be initialized initially
        assert not system._initialized

        # Context manager should initialize and cleanup
        async with system:
            assert system._initialized

        # Should be cleaned up after context
        assert not system._initialized

    def test_search_options_progressive_disclosure(self):
        """Test that SearchOptions support progressive feature discovery."""
        # Basic options (sensible defaults)
        basic_options = SearchOptions()
        assert basic_options.strategy == SearchStrategy.HYBRID
        assert basic_options.rerank is False
        assert basic_options.quality_tier == QualityTier.BALANCED

        # Progressive options
        progressive_options = SearchOptions(
            strategy=SearchStrategy.SEMANTIC,
            rerank=True,
            include_embeddings=True,
            similarity_threshold=0.7,
        )
        assert progressive_options.strategy == SearchStrategy.SEMANTIC
        assert progressive_options.rerank is True
        assert progressive_options.include_embeddings is True
        assert progressive_options.similarity_threshold == 0.7

        # Expert options
        expert_options = SearchOptions(
            custom_weights={"semantic": 0.7, "keyword": 0.3},
            fusion_algorithm="reciprocal_rank",
            rerank_model="custom-reranker",
        )
        assert expert_options.custom_weights is not None
        assert expert_options.fusion_algorithm == "reciprocal_rank"
        assert expert_options.rerank_model == "custom-reranker"


class TestSimpleSearchResult:
    """Test the SimpleSearchResult with progressive features."""

    def test_basic_result_properties(self):
        """Test basic search result properties."""
        result = SimpleSearchResult(
            content="Machine learning is a subset of AI",
            title="ML Introduction",
            url="https://example.com/ml",
            score=0.85,
            metadata={"category": "AI"},
        )

        assert result.content == "Machine learning is a subset of AI"
        assert result.title == "ML Introduction"
        assert result.url == "https://example.com/ml"
        assert result.score == 0.85
        assert result.metadata["category"] == "AI"

    def test_progressive_analysis_feature(self):
        """Test that analysis is computed lazily (progressive feature)."""
        result = SimpleSearchResult(
            content="def train_model(): return model", title="Code Example"
        )

        # Analysis should be computed on first access
        assert result._analysis is None

        analysis = result.get_analysis()
        assert analysis is not None
        assert analysis["type"] == "code"  # Should detect code content
        assert "length" in analysis
        assert "complexity" in analysis

        # Second access should return cached analysis
        analysis2 = result.get_analysis()
        assert analysis2 is analysis

    def test_progressive_suggestions_feature(self):
        """Test that suggestions are generated lazily."""
        result = SimpleSearchResult(
            content="Machine learning algorithms include decision trees and neural networks"
        )

        # Suggestions should be generated on first access
        assert result._suggestions is None

        suggestions = result.get_suggestions()
        assert suggestions is not None
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3  # Should limit to 3 suggestions

        # Should contain meaningful keywords
        suggestions_text = " ".join(suggestions).lower()
        assert any(
            word in suggestions_text for word in ["machine", "learning", "algorithms"]
        )

    def test_embedding_access(self):
        """Test embedding access (expert feature)."""
        embedding_vector = [0.1, 0.2, 0.3, 0.4]
        result = SimpleSearchResult(content="Test content", _embedding=embedding_vector)

        assert result.get_embedding() == embedding_vector

        # Result without embedding should return None
        result_no_embedding = SimpleSearchResult(content="Test content")
        assert result_no_embedding.get_embedding() is None


class TestFeatureDiscovery:
    """Test the feature discovery system."""

    def test_basic_feature_discovery(self):
        """Test basic feature discovery capabilities."""
        discovery = FeatureDiscovery()

        # Should provide basic features
        basic_features = discovery.get_basic_features()
        assert len(basic_features) > 0

        feature_names = [f.name for f in basic_features]
        assert "Quick Start" in feature_names
        assert "Simple Search" in feature_names

        # Each feature should have required attributes
        for feature in basic_features:
            assert feature.name
            assert feature.description
            assert feature.level == "basic"
            assert feature.example

    def test_progressive_feature_discovery(self):
        """Test progressive feature discovery."""
        discovery = FeatureDiscovery()

        progressive_features = discovery.get_progressive_features()
        assert len(progressive_features) > 0

        feature_names = [f.name for f in progressive_features]
        assert "Builder Pattern" in feature_names
        assert "Advanced Search" in feature_names

        # Progressive features should have higher sophistication
        for feature in progressive_features:
            assert feature.level == "progressive"

    def test_expert_feature_discovery(self):
        """Test expert feature discovery."""
        discovery = FeatureDiscovery()

        expert_features = discovery.get_expert_features()
        assert len(expert_features) > 0

        feature_names = [f.name for f in expert_features]
        assert "Custom Protocols" in feature_names

        # Expert features should have requirements
        for feature in expert_features:
            assert feature.level == "expert"

    def test_provider_discovery(self):
        """Test embedding provider discovery."""
        discovery = FeatureDiscovery()

        providers = discovery.discover_embedding_providers()
        assert len(providers) >= 2  # Should have at least fastembed and openai

        provider_names = [p.name for p in providers]
        assert "fastembed" in provider_names
        assert "openai" in provider_names

        # Each provider should have complete information
        for provider in providers:
            assert provider.name
            assert provider.description
            assert provider.capabilities
            assert provider.cost_model
            assert provider.performance_tier
            assert provider.example_usage

    def test_learning_path_generation(self):
        """Test learning path generation for different levels."""
        discovery = FeatureDiscovery()

        # Beginner path
        beginner_path = discovery.get_learning_path("beginner")
        assert len(beginner_path) > 0
        assert any("quick_start" in step for step in beginner_path)

        # Intermediate path
        intermediate_path = discovery.get_learning_path("intermediate")
        assert len(intermediate_path) > 0
        assert any("builder" in step for step in intermediate_path)

        # Advanced path
        advanced_path = discovery.get_learning_path("advanced")
        assert len(advanced_path) > 0
        assert any("protocol" in step.lower() for step in advanced_path)

    def test_provider_info_lookup(self):
        """Test specific provider information lookup."""
        discovery = FeatureDiscovery()

        # Should find existing provider
        openai_info = discovery.get_provider_info("openai")
        assert openai_info is not None
        assert openai_info.name == "openai"
        assert openai_info.performance_tier == QualityTier.BEST

        # Should return None for non-existent provider
        fake_info = discovery.get_provider_info("nonexistent")
        assert fake_info is None


class TestAdvancedConfigurationPatterns:
    """Test advanced configuration and expert patterns."""

    def test_embedding_config_builder(self):
        """Test sophisticated embedding configuration."""
        from src.api.progressive.builders import EmbeddingConfigBuilder

        config = (
            EmbeddingConfigBuilder()
            .with_provider("openai")
            .with_model("text-embedding-3-large")
            .with_batch_size(64)
            .with_normalization(True)
            .with_custom_preprocessing({"clean_html": True})
            .build()
        )

        assert config.provider == "openai"
        assert config.model_name == "text-embedding-3-large"
        assert config.batch_size == 64
        assert config.normalize is True
        assert config.custom_preprocessing["clean_html"] is True

    def test_search_config_builder(self):
        """Test sophisticated search configuration."""
        from src.api.progressive.builders import SearchConfigBuilder

        config = (
            SearchConfigBuilder()
            .with_strategy("semantic")
            .with_reranking(enabled=True)
            .with_similarity_threshold(0.8)
            .with_custom_weights({"semantic": 0.7, "keyword": 0.3})
            .with_fusion_algorithm("reciprocal_rank")
            .build()
        )

        assert config.strategy == SearchStrategy.SEMANTIC
        assert config.rerank is True
        assert config.similarity_threshold == 0.8
        assert config.custom_weights["semantic"] == 0.7
        assert config.fusion_algorithm == "reciprocal_rank"

    def test_advanced_config_builder_composition(self):
        """Test composition of advanced configuration builders."""
        from src.api.progressive.builders import AdvancedConfigBuilder

        config = (
            AdvancedConfigBuilder()
            .with_embedding_provider("openai")
            .with_quality_tier("best")
            .with_embedding_config(
                lambda emb: emb.with_batch_size(128).with_normalization(True)
            )
            .with_search_config(
                lambda search: search.with_strategy("semantic").with_reranking(True)
            )
            .with_experimental_features({"neural_reranking": True})
            .with_debug_mode(True)
            .build()
        )

        assert config.embedding_provider == "openai"
        assert config.quality_tier == QualityTier.BEST
        assert config.embedding_options.batch_size == 128
        assert config.search_options.strategy == SearchStrategy.SEMANTIC
        assert config.experimental_features["neural_reranking"] is True
        assert config.debug_mode is True


class TestTypeSystemSophistication:
    """Test the sophisticated type system and protocols."""

    def test_search_protocol_typing(self):
        """Test that SearchProtocol provides proper typing."""

        # Protocol should be runtime checkable
        assert hasattr(SearchProtocol, "__runtime_checkable__")

        # Should have required methods
        assert hasattr(SearchProtocol, "search")
        assert hasattr(SearchProtocol, "add_document")
        assert hasattr(SearchProtocol, "get_stats")

    def test_embedding_protocol_typing(self):
        """Test that EmbeddingProtocol provides proper typing."""

        # Protocol should be runtime checkable
        assert hasattr(EmbeddingProtocol, "__runtime_checkable__")

        # Should have required methods
        assert hasattr(EmbeddingProtocol, "generate_embedding")
        assert hasattr(EmbeddingProtocol, "generate_embeddings")
        assert hasattr(EmbeddingProtocol, "get_embedding_dimension")

    def test_progressive_response_wrapper(self):
        """Test the ProgressiveResponse wrapper pattern."""
        # Basic response
        data = {"test": "value"}
        response = ProgressiveResponse(data=data)

        assert response.data == data
        assert response.success is True
        assert response.message == ""
        assert response.metadata is None

        # Progressive enhancement
        enhanced = (
            response.with_metadata({"source": "test"})
            .with_metrics({"latency": 0.1})
            .with_suggestions(["try this", "or that"])
        )

        assert enhanced.metadata["source"] == "test"
        assert enhanced.metrics["latency"] == 0.1
        assert "try this" in enhanced.suggestions

        # Expert enhancement
        expert = enhanced.with_debug_info({"trace": "detailed"})
        assert expert.debug_info["trace"] == "detailed"

    def test_enum_type_safety(self):
        """Test that enums provide type safety."""
        # Quality tier enum
        assert QualityTier.FAST.value == "fast"
        assert QualityTier.BALANCED.value == "balanced"
        assert QualityTier.BEST.value == "best"

        # Search strategy enum
        assert SearchStrategy.VECTOR.value == "vector"
        assert SearchStrategy.HYBRID.value == "hybrid"
        assert SearchStrategy.SEMANTIC.value == "semantic"
        assert SearchStrategy.ADAPTIVE.value == "adaptive"


class TestErrorHandlingPatterns:
    """Test sophisticated error handling patterns."""

    def test_initialization_error_handling(self):
        """Test that proper errors are raised for uninitialized systems."""
        system = AIDocSystem.quick_start()

        # Should raise helpful error when not initialized
        with pytest.raises(RuntimeError) as exc_info:
            system._ensure_initialized()

        error_message = str(exc_info.value)
        assert "not initialized" in error_message
        assert "async with system:" in error_message or "initialize()" in error_message

    def test_configuration_validation(self):
        """Test configuration validation with helpful messages."""
        from src.api.progressive.factory import validate_configuration
        from src.api.progressive.types import CacheOptions, SystemConfiguration

        # Valid configuration should pass
        valid_config = SystemConfiguration()
        issues = validate_configuration(valid_config)
        assert len(issues) == 0

        # Invalid configuration should provide helpful feedback
        invalid_config = SystemConfiguration(
            embedding_provider="invalid_provider",
            cache_options=CacheOptions(ttl_seconds=30, max_size=5),
        )
        issues = validate_configuration(invalid_config)
        assert len(issues) > 0
        assert any("Unknown embedding provider" in issue for issue in issues)
        assert any("Cache TTL should be at least 60" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test that the system degrades gracefully on errors."""
        system = AIDocSystem.quick_start()

        # Mock a failing component
        with patch.object(system, "_search_service", new=None):
            async with system:
                # System should still initialize other components
                assert system._initialized
                # But search would fail gracefully
                with pytest.raises(AttributeError):
                    await system.search("test")


@pytest.mark.integration
class TestIntegrationPatterns:
    """Integration tests demonstrating real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_end_to_end_basic_workflow(self):
        """Test complete basic workflow with mocked services."""
        # Mock the underlying services
        with (
            patch("src.infrastructure.client_manager.ClientManager"),
            patch("src.services.embeddings.manager.EmbeddingManager") as mock_embedding,
            patch("src.services.vector_db.search.VectorSearchService") as mock_search,
            patch("src.services.crawling.manager.CrawlingManager") as mock_crawling,
        ):
            # Setup mocks
            mock_embedding_instance = AsyncMock()
            mock_embedding.return_value = mock_embedding_instance

            mock_search_instance = AsyncMock()
            mock_search_instance.search.return_value = [
                {
                    "content": "Test content about machine learning",
                    "title": "ML Guide",
                    "score": 0.9,
                    "metadata": {"category": "AI"},
                }
            ]
            mock_search.return_value = mock_search_instance

            mock_crawling_instance = AsyncMock()
            mock_crawling.return_value = mock_crawling_instance

            # Test the workflow
            system = AIDocSystem.quick_start()

            async with system:
                # Should initialize successfully
                assert system._initialized

                # Search should work
                results = await system.search("machine learning")
                assert len(results) == 1
                assert isinstance(results[0], SimpleSearchResult)
                assert results[0].title == "ML Guide"
                assert results[0].score == 0.9

                # Progressive features should work
                analysis = results[0].get_analysis()
                assert "length" in analysis
                assert "complexity" in analysis

    @pytest.mark.asyncio
    async def test_builder_pattern_integration(self):
        """Test builder pattern integration with mocked services."""
        with (
            patch("src.infrastructure.client_manager.ClientManager"),
            patch("src.services.embeddings.manager.EmbeddingManager") as mock_embedding,
            patch("src.services.vector_db.search.VectorSearchService") as mock_search,
            patch("src.services.crawling.manager.CrawlingManager") as mock_crawling,
            patch("src.services.cache.manager.CacheManager") as mock_cache,
        ):
            # Setup mocks
            for mock_service in [
                mock_embedding,
                mock_search,
                mock_crawling,
                mock_cache,
            ]:
                mock_instance = AsyncMock()
                mock_service.return_value = mock_instance

            # Test sophisticated builder configuration
            system = (
                AIDocSystemBuilder()
                .with_embedding_provider("openai")
                .with_quality_tier("best")
                .with_cache(enabled=True, ttl=7200)
                .with_monitoring(enabled=True, track_costs=True)
                .build()
            )

            async with system:
                # Verify configuration was applied
                assert system.embedding_provider == "openai"
                assert system.quality_tier == "best"
                assert system.enable_cache is True
                assert system.enable_monitoring is True


# Property-based testing for robust validation
@pytest.mark.property
class TestPropertyBasedValidation:
    """Property-based tests using hypothesis for robust validation."""

    def test_search_result_properties(self):
        """Test SearchResult properties with various inputs."""
        from hypothesis import given, strategies as st

        @given(
            content=st.text(min_size=1, max_size=1000),
            title=st.text(max_size=100),
            score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
        def test_result_creation(content, title, score):
            result = SimpleSearchResult(content=content, title=title, score=score)

            # Properties should be preserved
            assert result.content == content
            assert result.title == title
            assert result.score == score

            # Analysis should always work
            analysis = result.get_analysis()
            assert isinstance(analysis, dict)
            assert "length" in analysis
            assert analysis["length"] == len(content)

        # Run the property test
        test_result_creation()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
