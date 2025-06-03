"""Tests for ConfigLoader benchmark loading functionality."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config.benchmark_models import BenchmarkConfiguration
from src.config.loader import ConfigLoader


class TestConfigLoaderBenchmarks:
    """Test ConfigLoader benchmark-related methods."""

    def test_load_benchmark_config_success(self, tmp_path):
        """Test successful loading of benchmark configuration."""
        # Create a valid benchmark config file
        benchmark_data = {
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
                    "BAAI/bge-small-en-v1.5": {
                        "model_name": "BAAI/bge-small-en-v1.5",
                        "provider": "fastembed",
                        "avg_latency_ms": 45,
                        "quality_score": 78,
                        "tokens_per_second": 22000,
                        "cost_per_million_tokens": 0.0,
                        "max_context_length": 512,
                        "embedding_dimensions": 384,
                    },
                },
            }
        }

        # Write to temporary file
        benchmark_file = tmp_path / "custom-benchmarks.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f, indent=2)

        # Load using ConfigLoader
        result = ConfigLoader.load_benchmark_config(benchmark_file)

        # Verify result
        assert isinstance(result, BenchmarkConfiguration)
        assert result.embedding.smart_selection.quality_weight == 0.4
        assert len(result.embedding.model_benchmarks) == 2
        assert "text-embedding-3-small" in result.embedding.model_benchmarks
        assert "BAAI/bge-small-en-v1.5" in result.embedding.model_benchmarks

    def test_load_benchmark_config_file_not_found(self):
        """Test loading benchmark config when file doesn't exist."""
        non_existent_file = Path("/tmp/non_existent_benchmarks.json")

        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader.load_benchmark_config(non_existent_file)

        assert "Benchmark file not found" in str(exc_info.value)
        assert str(non_existent_file) in str(exc_info.value)

    def test_load_benchmark_config_invalid_json(self, tmp_path):
        """Test loading benchmark config with invalid JSON."""
        # Create file with invalid JSON
        benchmark_file = tmp_path / "invalid.json"
        with open(benchmark_file, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            ConfigLoader.load_benchmark_config(benchmark_file)

    def test_load_benchmark_config_invalid_schema(self, tmp_path):
        """Test loading benchmark config with invalid schema."""
        # Create file with valid JSON but invalid schema
        benchmark_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.6,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,  # Sum = 1.2, invalid
                },
                "model_benchmarks": {},
            }
        }

        benchmark_file = tmp_path / "invalid-schema.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f)

        with pytest.raises(ValidationError) as exc_info:
            ConfigLoader.load_benchmark_config(benchmark_file)

        assert "must sum to 1.0" in str(exc_info.value)

    def test_load_benchmark_config_missing_required_fields(self, tmp_path):
        """Test loading benchmark config with missing required fields."""
        # Create file with missing required fields
        benchmark_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.4,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,
                },
                # Missing model_benchmarks field
            }
        }

        benchmark_file = tmp_path / "missing-fields.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f)

        with pytest.raises(ValidationError) as exc_info:
            ConfigLoader.load_benchmark_config(benchmark_file)

        assert "model_benchmarks" in str(exc_info.value)

    def test_load_benchmark_config_extra_fields(self, tmp_path):
        """Test that extra fields are rejected."""
        # Create file with extra fields
        benchmark_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.4,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,
                    "extra_field": "should fail",  # Extra field
                },
                "model_benchmarks": {},
            }
        }

        benchmark_file = tmp_path / "extra-fields.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f)

        with pytest.raises(ValidationError) as exc_info:
            ConfigLoader.load_benchmark_config(benchmark_file)

        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_load_benchmark_config_path_types(self, tmp_path):
        """Test loading benchmark config with different path types."""
        # Create a valid benchmark config file
        benchmark_data = {
            "embedding": {
                "smart_selection": {
                    "quality_weight": 0.4,
                    "speed_weight": 0.3,
                    "cost_weight": 0.3,
                },
                "model_benchmarks": {},
            }
        }

        benchmark_file = tmp_path / "benchmarks.json"
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f)

        # Test with Path object
        result1 = ConfigLoader.load_benchmark_config(benchmark_file)
        assert isinstance(result1, BenchmarkConfiguration)

        # Test with string path
        result2 = ConfigLoader.load_benchmark_config(str(benchmark_file))
        assert isinstance(result2, BenchmarkConfiguration)

        # Both should produce same result
        assert result1.embedding.smart_selection.quality_weight == result2.embedding.smart_selection.quality_weight

    def test_load_actual_custom_benchmarks_file(self):
        """Test loading the actual custom-benchmarks.json from the project."""
        # Try to load the actual file if it exists
        project_root = Path(__file__).parent.parent.parent.parent
        custom_benchmarks_path = project_root / "config" / "templates" / "custom-benchmarks.json"

        if custom_benchmarks_path.exists():
            result = ConfigLoader.load_benchmark_config(custom_benchmarks_path)

            # Verify it loads successfully
            assert isinstance(result, BenchmarkConfiguration)
            assert result.embedding.smart_selection is not None
            assert len(result.embedding.model_benchmarks) > 0

            # Verify some expected models
            assert "text-embedding-3-small" in result.embedding.model_benchmarks
            assert "BAAI/bge-small-en-v1.5" in result.embedding.model_benchmarks

            # Verify weights sum to 1.0
            selection = result.embedding.smart_selection
            weight_sum = selection.quality_weight + selection.speed_weight + selection.cost_weight
            assert abs(weight_sum - 1.0) < 0.001  # Allow for floating point precision
