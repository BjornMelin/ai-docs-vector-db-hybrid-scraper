"""Tests for benchmark configuration models."""

import pytest
from pydantic import ValidationError
from src.config.benchmark_models import BenchmarkConfiguration
from src.config.benchmark_models import EmbeddingBenchmarkSet
from src.config.models import ModelBenchmark
from src.config.models import SmartSelectionConfig


class TestModelBenchmark:
    """Test ModelBenchmark model validation."""

    def test_valid_model_benchmark(self):
        """Test creating a valid model benchmark."""
        benchmark = ModelBenchmark(
            model_name="text-embedding-3-small",
            provider="openai",
            avg_latency_ms=78.5,
            quality_score=85,
            tokens_per_second=12800,
            cost_per_million_tokens=20.0,
            max_context_length=8191,
            embedding_dimensions=1536,
        )

        assert benchmark.model_name == "text-embedding-3-small"
        assert benchmark.provider == "openai"
        assert benchmark.avg_latency_ms == 78.5
        assert benchmark.quality_score == 85
        assert benchmark.tokens_per_second == 12800
        assert benchmark.cost_per_million_tokens == 20.0
        assert benchmark.max_context_length == 8191
        assert benchmark.embedding_dimensions == 1536

    def test_invalid_quality_score(self):
        """Test validation of quality score bounds."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test-model",
                provider="test",
                avg_latency_ms=50,
                quality_score=105,  # Invalid: > 100
                tokens_per_second=10000,
                cost_per_million_tokens=0,
                max_context_length=512,
                embedding_dimensions=384,
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_negative_latency(self):
        """Test validation of positive latency."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test-model",
                provider="test",
                avg_latency_ms=-10,  # Invalid: negative
                quality_score=80,
                tokens_per_second=10000,
                cost_per_million_tokens=0,
                max_context_length=512,
                embedding_dimensions=384,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_zero_embedding_dimensions(self):
        """Test validation of positive embedding dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBenchmark(
                model_name="test-model",
                provider="test",
                avg_latency_ms=50,
                quality_score=80,
                tokens_per_second=10000,
                cost_per_million_tokens=0,
                max_context_length=512,
                embedding_dimensions=0,  # Invalid: must be > 0
            )
        assert "greater than 0" in str(exc_info.value)

    def test_free_model_zero_cost(self):
        """Test that free models can have zero cost."""
        benchmark = ModelBenchmark(
            model_name="BAAI/bge-small-en-v1.5",
            provider="fastembed",
            avg_latency_ms=45,
            quality_score=78,
            tokens_per_second=22000,
            cost_per_million_tokens=0.0,  # Valid: free model
            max_context_length=512,
            embedding_dimensions=384,
        )
        assert benchmark.cost_per_million_tokens == 0.0


class TestSmartSelectionConfig:
    """Test SmartSelectionConfig model validation."""

    def test_valid_smart_selection_config(self):
        """Test creating a valid smart selection config."""
        config = SmartSelectionConfig(
            quality_weight=0.4,
            speed_weight=0.3,
            cost_weight=0.3,
            quality_best_threshold=85.0,
            budget_warning_threshold=0.8,
            short_text_threshold=100,
            long_text_threshold=2000,
        )

        assert config.quality_weight == 0.4
        assert config.speed_weight == 0.3
        assert config.cost_weight == 0.3
        assert config.quality_best_threshold == 85.0
        assert config.budget_warning_threshold == 0.8
        assert config.short_text_threshold == 100
        assert config.long_text_threshold == 2000

    def test_weights_sum_to_one(self):
        """Test validation that weights sum to 1.0."""
        # Valid: weights sum to 1.0
        config = SmartSelectionConfig(
            quality_weight=0.5,
            speed_weight=0.3,
            cost_weight=0.2,
        )
        assert config.quality_weight + config.speed_weight + config.cost_weight == 1.0

    def test_weights_do_not_sum_to_one(self):
        """Test validation fails when weights don't sum to 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(
                quality_weight=0.5,
                speed_weight=0.3,
                cost_weight=0.3,  # Sum = 1.1, invalid
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_negative_weight(self):
        """Test validation of non-negative weights."""
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(
                quality_weight=-0.1,  # Invalid: negative
                speed_weight=0.6,
                cost_weight=0.5,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_weight_exceeds_one(self):
        """Test validation of weights not exceeding 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            SmartSelectionConfig(
                quality_weight=1.1,  # Invalid: > 1.0
                speed_weight=0.0,
                cost_weight=0.0,
            )
        assert "less than or equal to 1" in str(exc_info.value)


class TestEmbeddingBenchmarkSet:
    """Test EmbeddingBenchmarkSet model validation."""

    def test_valid_embedding_benchmark_set(self):
        """Test creating a valid embedding benchmark set."""
        benchmark_set = EmbeddingBenchmarkSet(
            smart_selection=SmartSelectionConfig(
                quality_weight=0.4,
                speed_weight=0.3,
                cost_weight=0.3,
            ),
            model_benchmarks={
                "text-embedding-3-small": ModelBenchmark(
                    model_name="text-embedding-3-small",
                    provider="openai",
                    avg_latency_ms=78,
                    quality_score=85,
                    tokens_per_second=12800,
                    cost_per_million_tokens=20.0,
                    max_context_length=8191,
                    embedding_dimensions=1536,
                ),
                "BAAI/bge-small-en-v1.5": ModelBenchmark(
                    model_name="BAAI/bge-small-en-v1.5",
                    provider="fastembed",
                    avg_latency_ms=45,
                    quality_score=78,
                    tokens_per_second=22000,
                    cost_per_million_tokens=0.0,
                    max_context_length=512,
                    embedding_dimensions=384,
                ),
            },
        )

        assert len(benchmark_set.model_benchmarks) == 2
        assert "text-embedding-3-small" in benchmark_set.model_benchmarks
        assert "BAAI/bge-small-en-v1.5" in benchmark_set.model_benchmarks
        assert benchmark_set.smart_selection.quality_weight == 0.4

    def test_empty_model_benchmarks(self):
        """Test that empty model benchmarks dict is valid."""
        benchmark_set = EmbeddingBenchmarkSet(
            smart_selection=SmartSelectionConfig(),
            model_benchmarks={},  # Empty is valid
        )
        assert len(benchmark_set.model_benchmarks) == 0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingBenchmarkSet(
                smart_selection=SmartSelectionConfig(),
                model_benchmarks={},
                extra_field="not allowed",  # Should fail
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestBenchmarkConfiguration:
    """Test BenchmarkConfiguration model validation."""

    def test_valid_benchmark_configuration(self):
        """Test creating a valid benchmark configuration."""
        config = BenchmarkConfiguration(
            embedding=EmbeddingBenchmarkSet(
                smart_selection=SmartSelectionConfig(
                    quality_weight=0.4,
                    speed_weight=0.3,
                    cost_weight=0.3,
                ),
                model_benchmarks={
                    "test-model": ModelBenchmark(
                        model_name="test-model",
                        provider="test",
                        avg_latency_ms=50,
                        quality_score=80,
                        tokens_per_second=10000,
                        cost_per_million_tokens=10.0,
                        max_context_length=512,
                        embedding_dimensions=384,
                    ),
                },
            )
        )

        assert config.embedding.smart_selection.quality_weight == 0.4
        assert len(config.embedding.model_benchmarks) == 1
        assert "test-model" in config.embedding.model_benchmarks

    def test_from_custom_benchmarks_json_structure(self):
        """Test loading from custom-benchmarks.json structure."""
        # This mimics the structure of custom-benchmarks.json
        data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.4,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,
                    "quality_best_threshold": 85.0,
                    "budget_warning_threshold": 0.8,
                    "short_text_threshold": 100,
                    "long_text_threshold": 2000,
                },
                "model_benchmarks": {
                    "text-embedding-3-small": {
                        "model_name": "text-embedding-3-small",
                        "provider": "openai",
                        "avg_latency_ms": 78,
                        "quality_score": 85,
                        "tokens_per_second": 12800,
                        "cost_per_million_tokens": 20.0,
                        "max_context_length": 8191,
                        "embedding_dimensions": 1536,
                    },
                    "custom-local-model": {
                        "model_name": "custom-local-model",
                        "provider": "fastembed",
                        "avg_latency_ms": 35,
                        "quality_score": 82,
                        "tokens_per_second": 25000,
                        "cost_per_million_tokens": 0.0,
                        "max_context_length": 1024,
                        "embedding_dimensions": 768,
                    },
                },
            }
        }

        config = BenchmarkConfiguration(**data)

        # Verify structure was parsed correctly
        assert config.embedding.smart_selection.quality_weight == 0.4
        assert config.embedding.smart_selection.speed_weight == 0.3
        assert config.embedding.smart_selection.cost_weight == 0.3
        assert len(config.embedding.model_benchmarks) == 2

        # Verify specific models
        small_model = config.embedding.model_benchmarks["text-embedding-3-small"]
        assert small_model.model_name == "text-embedding-3-small"
        assert small_model.provider == "openai"
        assert small_model.quality_score == 85

        custom_model = config.embedding.model_benchmarks["custom-local-model"]
        assert custom_model.model_name == "custom-local-model"
        assert custom_model.provider == "fastembed"
        assert custom_model.cost_per_million_tokens == 0.0

    def test_nested_validation_errors_propagate(self):
        """Test that nested validation errors propagate correctly."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfiguration(
                embedding=EmbeddingBenchmarkSet(
                    smart_selection=SmartSelectionConfig(
                        quality_weight=0.6,
                        speed_weight=0.3,
                        cost_weight=0.3,  # Sum = 1.2, invalid
                    ),
                    model_benchmarks={},
                )
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_extra_fields_at_root_forbidden(self):
        """Test that extra fields at root level are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfiguration(
                embedding=EmbeddingBenchmarkSet(
                    smart_selection=SmartSelectionConfig(),
                    model_benchmarks={},
                ),
                extra_field="not allowed",  # Should fail
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)
