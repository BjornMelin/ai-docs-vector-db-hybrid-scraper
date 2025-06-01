"""Test EmbeddingConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.enums import EmbeddingModel
from config.enums import EmbeddingProvider
from config.enums import SearchStrategy
from config.models import EmbeddingConfig
from config.models import ModelBenchmark
from config.models import SmartSelectionConfig


class TestModelBenchmark:
    """Test ModelBenchmark model validation and behavior."""

    def test_valid_benchmark(self):
        """Test creating a valid model benchmark."""
        benchmark = ModelBenchmark(
            model_name="text-embedding-3-small",
            provider="openai",
            avg_latency_ms=78.5,
            quality_score=85.0,
            tokens_per_second=12800.0,
            cost_per_million_tokens=20.0,
            max_context_length=8191,
            embedding_dimensions=1536,
        )

        assert benchmark.model_name == "text-embedding-3-small"
        assert benchmark.provider == "openai"
        assert benchmark.avg_latency_ms == 78.5
        assert benchmark.quality_score == 85.0
        assert benchmark.tokens_per_second == 12800.0
        assert benchmark.cost_per_million_tokens == 20.0
        assert benchmark.max_context_length == 8191
        assert benchmark.embedding_dimensions == 1536

    def test_benchmark_constraints(self):
        """Test ModelBenchmark field constraints."""
        # Invalid: negative latency
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test",
                provider="test",
                avg_latency_ms=-10,
                quality_score=80,
                tokens_per_second=1000,
                cost_per_million_tokens=10,
                max_context_length=512,
                embedding_dimensions=384,
            )
        errors = exc_info.value.errors()
        assert any("avg_latency_ms" in str(e["loc"]) for e in errors)

        # Invalid: quality score > 100
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test",
                provider="test",
                avg_latency_ms=50,
                quality_score=101,
                tokens_per_second=1000,
                cost_per_million_tokens=10,
                max_context_length=512,
                embedding_dimensions=384,
            )
        errors = exc_info.value.errors()
        assert any("quality_score" in str(e["loc"]) for e in errors)

        # Invalid: negative cost
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test",
                provider="test",
                avg_latency_ms=50,
                quality_score=80,
                tokens_per_second=1000,
                cost_per_million_tokens=-10,
                max_context_length=512,
                embedding_dimensions=384,
            )
        errors = exc_info.value.errors()
        assert any("cost_per_million_tokens" in str(e["loc"]) for e in errors)

    def test_local_model_zero_cost(self):
        """Test that local models can have zero cost."""
        benchmark = ModelBenchmark(
            model_name="BAAI/bge-small-en-v1.5",
            provider="fastembed",
            avg_latency_ms=45.0,
            quality_score=78.0,
            tokens_per_second=22000.0,
            cost_per_million_tokens=0.0,  # Free local model
            max_context_length=512,
            embedding_dimensions=384,
        )

        assert benchmark.cost_per_million_tokens == 0.0


class TestEmbeddingConfig:
    """Test EmbeddingConfig model validation and behavior."""

    def test_default_values(self):
        """Test EmbeddingConfig with default values."""
        config = EmbeddingConfig()

        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.sparse_model is None
        assert config.search_strategy == SearchStrategy.DENSE
        assert config.enable_quantization is True
        assert config.matryoshka_dimensions == [1536, 1024, 512, 256]

        # Reranking
        assert config.enable_reranking is False
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert config.rerank_top_k == 20

        # Model benchmarks
        assert isinstance(config.model_benchmarks, dict)
        assert "text-embedding-3-small" in config.model_benchmarks
        assert isinstance(
            config.model_benchmarks["text-embedding-3-small"], ModelBenchmark
        )

        # Smart selection
        assert isinstance(config.smart_selection, SmartSelectionConfig)

    def test_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.FASTEMBED,
            dense_model=EmbeddingModel.BGE_LARGE_EN_V15,
            sparse_model=EmbeddingModel.SPLADE_PP_EN_V1,
            search_strategy=SearchStrategy.HYBRID,
            enable_quantization=False,
            matryoshka_dimensions=[768, 384, 192],
            enable_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            rerank_top_k=50,
        )

        assert config.provider == EmbeddingProvider.FASTEMBED
        assert config.dense_model == EmbeddingModel.BGE_LARGE_EN_V15
        assert config.sparse_model == EmbeddingModel.SPLADE_PP_EN_V1
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.enable_quantization is False
        assert config.matryoshka_dimensions == [768, 384, 192]
        assert config.enable_reranking is True
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert config.rerank_top_k == 50

    def test_enum_validation(self):
        """Test enum field validation."""
        # Valid provider
        for provider in EmbeddingProvider:
            config = EmbeddingConfig(provider=provider)
            assert config.provider == provider

        # Valid dense model
        for model in [
            EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
            EmbeddingModel.BGE_SMALL_EN_V15,
            EmbeddingModel.NV_EMBED_V2,
        ]:
            config = EmbeddingConfig(dense_model=model)
            assert config.dense_model == model

        # Valid search strategy
        for strategy in SearchStrategy:
            config = EmbeddingConfig(search_strategy=strategy)
            assert config.search_strategy == strategy

    def test_matryoshka_dimensions_list(self):
        """Test matryoshka dimensions list field."""
        # Custom dimensions in descending order
        config1 = EmbeddingConfig(matryoshka_dimensions=[2048, 1024, 512, 256, 128])
        assert len(config1.matryoshka_dimensions) == 5
        assert config1.matryoshka_dimensions[0] == 2048
        assert config1.matryoshka_dimensions[-1] == 128

        # Empty list is valid
        config2 = EmbeddingConfig(matryoshka_dimensions=[])
        assert config2.matryoshka_dimensions == []

        # Single dimension
        config3 = EmbeddingConfig(matryoshka_dimensions=[768])
        assert config3.matryoshka_dimensions == [768]

    def test_custom_model_benchmarks(self):
        """Test custom model benchmarks."""
        custom_benchmarks = {
            "custom-model-v1": ModelBenchmark(
                model_name="custom-model-v1",
                provider="custom",
                avg_latency_ms=100.0,
                quality_score=90.0,
                tokens_per_second=10000.0,
                cost_per_million_tokens=15.0,
                max_context_length=4096,
                embedding_dimensions=768,
            )
        }

        config = EmbeddingConfig(model_benchmarks=custom_benchmarks)
        assert "custom-model-v1" in config.model_benchmarks
        assert config.model_benchmarks["custom-model-v1"].quality_score == 90.0

    def test_validate_benchmark_keys(self):
        """Test that benchmark dict keys must match model names."""
        # Invalid: key doesn't match model name
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(
                model_benchmarks={
                    "wrong-key": ModelBenchmark(
                        model_name="correct-name",
                        provider="test",
                        avg_latency_ms=50,
                        quality_score=80,
                        tokens_per_second=1000,
                        cost_per_million_tokens=10,
                        max_context_length=512,
                        embedding_dimensions=384,
                    )
                }
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Dictionary key 'wrong-key' does not match" in str(errors[0]["msg"])

    def test_custom_smart_selection_config(self):
        """Test custom smart selection configuration."""
        custom_smart_selection = SmartSelectionConfig(
            quality_weight=0.5,
            speed_weight=0.2,
            cost_weight=0.3,
            quality_fast_threshold=70.0,
        )

        config = EmbeddingConfig(smart_selection=custom_smart_selection)
        assert config.smart_selection.quality_weight == 0.5
        assert config.smart_selection.speed_weight == 0.2
        assert config.smart_selection.cost_weight == 0.3
        assert config.smart_selection.quality_fast_threshold == 70.0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(provider=EmbeddingProvider.OPENAI, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.FASTEMBED,
            dense_model=EmbeddingModel.BGE_SMALL_EN_V15,
            search_strategy=SearchStrategy.SPARSE,
            enable_reranking=True,
            rerank_top_k=30,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["provider"] == EmbeddingProvider.FASTEMBED
        assert data["dense_model"] == EmbeddingModel.BGE_SMALL_EN_V15
        assert data["search_strategy"] == SearchStrategy.SPARSE
        assert data["enable_reranking"] is True
        assert data["rerank_top_k"] == 30

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"provider":"fastembed"' in json_str
        assert '"enable_reranking":true' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI, enable_reranking=False
        )

        updated = original.model_copy(
            update={
                "provider": EmbeddingProvider.FASTEMBED,
                "enable_reranking": True,
                "rerank_top_k": 40,
            }
        )

        assert original.provider == EmbeddingProvider.OPENAI
        assert original.enable_reranking is False
        assert updated.provider == EmbeddingProvider.FASTEMBED
        assert updated.enable_reranking is True
        assert updated.rerank_top_k == 40

    def test_default_model_benchmarks_structure(self):
        """Test the structure of default model benchmarks."""
        config = EmbeddingConfig()

        # Check that default benchmarks are populated
        assert len(config.model_benchmarks) >= 4

        # Check specific models exist
        expected_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ]

        for model in expected_models:
            assert model in config.model_benchmarks
            benchmark = config.model_benchmarks[model]
            assert isinstance(benchmark, ModelBenchmark)
            assert benchmark.model_name == model

        # Verify OpenAI models have costs
        assert (
            config.model_benchmarks["text-embedding-3-small"].cost_per_million_tokens
            > 0
        )
        assert (
            config.model_benchmarks["text-embedding-3-large"].cost_per_million_tokens
            > 0
        )

        # Verify local models are free
        assert (
            config.model_benchmarks["BAAI/bge-small-en-v1.5"].cost_per_million_tokens
            == 0.0
        )
        assert (
            config.model_benchmarks["BAAI/bge-large-en-v1.5"].cost_per_million_tokens
            == 0.0
        )
