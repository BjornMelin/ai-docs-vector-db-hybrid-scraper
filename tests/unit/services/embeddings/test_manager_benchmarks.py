"""Tests for EmbeddingManager benchmark loading functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.services.embeddings.manager import EmbeddingManager, TextAnalysis


class TestEmbeddingManagerBenchmarks:
    """Test EmbeddingManager benchmark-related methods."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""

        config = MagicMock(spec=Settings)

        # Mock cache config
        mock_cache = MagicMock()
        mock_cache.enable_caching = False
        config.cache = mock_cache

        # Mock openai config
        mock_openai = MagicMock()
        mock_openai.api_key = None
        config.openai = mock_openai

        # Mock fastembed config
        mock_fastembed = MagicMock()
        mock_fastembed.model = "BAAI/bge-small-en-v1.5"
        config.fastembed = mock_fastembed

        # Mock embedding config
        mock_embedding = MagicMock()
        mock_embedding.model_benchmarks = {
            "default-model": MagicMock(
                model_name="default-model",
                provider="test",
                avg_latency_ms=50,
                quality_score=80,
            )
        }
        mock_embedding.smart_selection = MagicMock(
            quality_weight=0.4,
            speed_weight=0.3,
            cost_weight=0.3,
        )
        config.embedding = mock_embedding

        # Mock embedding provider enum
        mock_provider = MagicMock()
        mock_provider.value = "fastembed"
        config.embedding_provider = mock_provider

        return config

    def test_load_custom_benchmarks_success(self, mock_config, tmp_path):
        """Test successful loading of custom benchmarks."""
        # Create embedding manager
        manager = EmbeddingManager(
            config=mock_config,
        )

        # Verify initial benchmarks
        assert "default-model" in manager._benchmarks
        assert manager._smart_config is not None
        assert manager._smart_config.quality_weight == 0.4

        # Create custom benchmark file
        custom_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.5,
                    "speed_weight": 0.2,
                    "cost_weight": 0.3,
                    "quality_best_threshold": 90.0,
                    "budget_warning_threshold": 0.7,
                    "short_text_threshold": 50,
                    "long_text_threshold": 1500,
                },
                "model_benchmarks": {
                    "custom-model-1": {
                        "model_name": "custom-model-1",
                        "provider": "custom",
                        "avg_latency_ms": 25,
                        "quality_score": 95,
                        "tokens_per_second": 30000,
                        "cost_per_million_tokens": 5.0,
                        "max_context_length": 2048,
                        "embedding_dimensions": 768,
                    },
                    "custom-model-2": {
                        "model_name": "custom-model-2",
                        "provider": "custom",
                        "avg_latency_ms": 100,
                        "quality_score": 70,
                        "tokens_per_second": 5000,
                        "cost_per_million_tokens": 1.0,
                        "max_context_length": 512,
                        "embedding_dimensions": 256,
                    },
                },
            }
        }

        benchmark_file = tmp_path / "custom_benchmarks.json"
        with benchmark_file.open("w") as f:
            json.dump(custom_data, f)

        # Load custom benchmarks
        manager.load_custom_benchmarks(benchmark_file)

        # Verify benchmarks were replaced
        assert "default-model" not in manager._benchmarks
        assert "custom-model-1" in manager._benchmarks
        assert "custom-model-2" in manager._benchmarks
        assert len(manager._benchmarks) == 2

        # Verify smart selection config was updated
        assert manager._smart_config is not None
        assert manager._smart_config.quality_weight == 0.5
        assert manager._smart_config.speed_weight == 0.2
        assert manager._smart_config.cost_weight == 0.3
        assert manager._smart_config.quality_best_threshold == 90.0

        # Verify specific model details
        model1 = manager._benchmarks["custom-model-1"]
        assert model1["model_name"] == "custom-model-1"
        assert model1["quality_score"] == 95
        assert model1["avg_latency_ms"] == 25

    def test_load_custom_benchmarks_file_not_found(self, mock_config):
        """Test loading custom benchmarks when file doesn't exist."""
        manager = EmbeddingManager(config=mock_config)

        with pytest.raises(FileNotFoundError):
            manager.load_custom_benchmarks("/non/existent/file.json")

    def test_load_custom_benchmarks_invalid_json(self, mock_config, tmp_path):
        """Test loading custom benchmarks with invalid JSON."""
        manager = EmbeddingManager(config=mock_config)

        # Create file with invalid JSON
        benchmark_file = tmp_path / "invalid.json"
        with benchmark_file.open("w") as f:
            f.write("{ invalid json")

        with pytest.raises(json.JSONDecodeError):
            manager.load_custom_benchmarks(benchmark_file)

    def test_load_custom_benchmarks_preserves_previous_on_error(
        self, mock_config, tmp_path
    ):
        """Test that previous benchmarks are preserved if loading fails."""
        manager = EmbeddingManager(config=mock_config)

        # Store original benchmarks
        original_benchmarks = manager._benchmarks.copy()
        assert manager._smart_config is not None
        original_smart_config = manager._smart_config

        # Create file with invalid schema
        benchmark_file = tmp_path / "invalid_schema.json"
        with benchmark_file.open("w") as f:
            json.dump({"invalid": "schema"}, f)

        # Try to load invalid benchmarks
        with pytest.raises(ValueError):  # Will raise validation error
            manager.load_custom_benchmarks(benchmark_file)

        # Verify original benchmarks are preserved
        assert manager._benchmarks == original_benchmarks
        assert manager._smart_config == original_smart_config

    @patch("src.services.embeddings.manager.logger")
    def test_load_custom_benchmarks_logging(self, mock_logger, mock_config, tmp_path):
        """Test that loading custom benchmarks logs appropriate messages."""
        manager = EmbeddingManager(config=mock_config)

        # Create minimal valid benchmark file
        custom_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.4,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,
                },
                "model_benchmarks": {
                    "model1": {
                        "model_name": "model1",
                        "provider": "test",
                        "avg_latency_ms": 50,
                        "quality_score": 80,
                        "tokens_per_second": 10000,
                        "cost_per_million_tokens": 10.0,
                        "max_context_length": 512,
                        "embedding_dimensions": 384,
                    },
                    "model2": {
                        "model_name": "model2",
                        "provider": "test",
                        "avg_latency_ms": 100,
                        "quality_score": 90,
                        "tokens_per_second": 5000,
                        "cost_per_million_tokens": 20.0,
                        "max_context_length": 1024,
                        "embedding_dimensions": 768,
                    },
                },
            }
        }

        benchmark_file = tmp_path / "benchmarks.json"
        with benchmark_file.open("w") as f:
            json.dump(custom_data, f)

        # Load benchmarks
        manager.load_custom_benchmarks(benchmark_file)

        # Verify logging
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "Loaded custom benchmarks" in log_message
        assert "benchmarks.json" in log_message
        assert "2 models" in log_message

    def test_load_custom_benchmarks_integration_with_smart_selection(
        self, mock_config, tmp_path
    ):
        """Test that loaded benchmarks integrate properly with  selection."""
        manager = EmbeddingManager(config=mock_config)

        # Create custom benchmark with specific models
        custom_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.6,
                    "speed_weight": 0.2,
                    "cost_weight": 0.2,
                    "quality_best_threshold": 85.0,
                    "short_text_threshold": 100,
                    "long_text_threshold": 2000,
                },
                "model_benchmarks": {
                    "high-quality-model": {
                        "model_name": "high-quality-model",
                        "provider": "premium",
                        "avg_latency_ms": 150,
                        "quality_score": 98,
                        "tokens_per_second": 3000,
                        "cost_per_million_tokens": 100.0,
                        "max_context_length": 4096,
                        "embedding_dimensions": 2048,
                    },
                    "fast-cheap-model": {
                        "model_name": "fast-cheap-model",
                        "provider": "local",
                        "avg_latency_ms": 20,
                        "quality_score": 75,
                        "tokens_per_second": 50000,
                        "cost_per_million_tokens": 0.0,
                        "max_context_length": 512,
                        "embedding_dimensions": 256,
                    },
                },
            }
        }

        benchmark_file = tmp_path / "custom.json"
        with benchmark_file.open("w") as f:
            json.dump(custom_data, f)

        # Load benchmarks
        manager.load_custom_benchmarks(benchmark_file)

        # Create mock text analysis

        text_analysis = TextAnalysis(
            total_length=500,
            avg_length=100,
            complexity_score=0.5,
            estimated_tokens=125,
            text_type="docs",
            requires_high_quality=True,
        )

        # Mock providers (matching provider names in benchmark data)
        # Mark manager as initialized and mock providers
        manager._initialized = True
        mock_openai_provider = MagicMock()
        mock_openai_provider.model_name = "high-quality-model"
        mock_local_provider = MagicMock()
        mock_local_provider.model_name = "fast-cheap-model"
        manager.providers = {
            "openai": mock_openai_provider,
            "fastembed": mock_local_provider,
        }

        # Get recommendation - should return a valid recommendation
        recommendation = manager.get_smart_provider_recommendation(text_analysis)

        # Verify we get a valid recommendation
        assert recommendation["model"] in ["high-quality-model", "fast-cheap-model"]
        assert "provider" in recommendation
        assert "estimated_cost" in recommendation
        assert "reasoning" in recommendation
        assert recommendation["score"] >= 0
