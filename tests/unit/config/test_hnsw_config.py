"""Test HNSWConfig and CollectionHNSWConfigs Pydantic models."""

import pytest
from pydantic import ValidationError

from config.models import CollectionHNSWConfigs
from config.models import HNSWConfig


class TestHNSWConfig:
    """Test HNSWConfig model validation and behavior."""

    def test_default_values(self):
        """Test HNSWConfig with default values."""
        config = HNSWConfig()

        # Core parameters
        assert config.m == 16
        assert config.ef_construct == 200
        assert config.full_scan_threshold == 10000
        assert config.max_indexing_threads == 0

        # Runtime ef recommendations
        assert config.min_ef == 50
        assert config.balanced_ef == 100
        assert config.max_ef == 200

        # Adaptive ef settings
        assert config.enable_adaptive_ef is True
        assert config.default_time_budget_ms == 100

    def test_m_parameter_constraints(self):
        """Test m parameter constraints (0 < m <= 64)."""
        # Valid values
        config1 = HNSWConfig(m=1)
        assert config1.m == 1

        config2 = HNSWConfig(m=64)
        assert config2.m == 64

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(m=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("m",)
        assert "greater than 0" in str(errors[0]["msg"])

        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(m=65)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("m",)
        assert "less than or equal to 64" in str(errors[0]["msg"])

    def test_ef_construct_constraints(self):
        """Test ef_construct constraints (0 < ef_construct <= 1000)."""
        # Valid values
        config1 = HNSWConfig(ef_construct=1)
        assert config1.ef_construct == 1

        config2 = HNSWConfig(ef_construct=1000)
        assert config2.ef_construct == 1000

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(ef_construct=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("ef_construct",)

        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(ef_construct=1001)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("ef_construct",)

    def test_positive_constraints(self):
        """Test positive value constraints."""
        # Test full_scan_threshold
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(full_scan_threshold=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("full_scan_threshold",)

        # Test min_ef
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(min_ef=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("min_ef",)

        # Test default_time_budget_ms
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(default_time_budget_ms=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("default_time_budget_ms",)

    def test_max_indexing_threads_non_negative(self):
        """Test max_indexing_threads must be non-negative."""
        config1 = HNSWConfig(max_indexing_threads=0)  # Auto
        assert config1.max_indexing_threads == 0

        config2 = HNSWConfig(max_indexing_threads=8)
        assert config2.max_indexing_threads == 8

        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(max_indexing_threads=-1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_indexing_threads",)

    def test_custom_configurations(self):
        """Test custom HNSW configurations."""
        config = HNSWConfig(
            m=24,
            ef_construct=400,
            full_scan_threshold=5000,
            max_indexing_threads=4,
            min_ef=100,
            balanced_ef=150,
            max_ef=250,
            enable_adaptive_ef=False,
            default_time_budget_ms=200,
        )

        assert config.m == 24
        assert config.ef_construct == 400
        assert config.full_scan_threshold == 5000
        assert config.max_indexing_threads == 4
        assert config.min_ef == 100
        assert config.balanced_ef == 150
        assert config.max_ef == 250
        assert config.enable_adaptive_ef is False
        assert config.default_time_budget_ms == 200

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            HNSWConfig(m=16, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"


class TestCollectionHNSWConfigs:
    """Test CollectionHNSWConfigs model validation and behavior."""

    def test_default_configurations(self):
        """Test default collection-specific configurations."""
        configs = CollectionHNSWConfigs()

        # API Reference - High accuracy
        assert configs.api_reference.m == 20
        assert configs.api_reference.ef_construct == 300
        assert configs.api_reference.full_scan_threshold == 5000
        assert configs.api_reference.min_ef == 100
        assert configs.api_reference.balanced_ef == 150
        assert configs.api_reference.max_ef == 200

        # Tutorials - Balanced
        assert configs.tutorials.m == 16
        assert configs.tutorials.ef_construct == 200
        assert configs.tutorials.full_scan_threshold == 10000
        assert configs.tutorials.min_ef == 75
        assert configs.tutorials.balanced_ef == 100
        assert configs.tutorials.max_ef == 150

        # Blog Posts - Fast
        assert configs.blog_posts.m == 12
        assert configs.blog_posts.ef_construct == 150
        assert configs.blog_posts.full_scan_threshold == 20000
        assert configs.blog_posts.min_ef == 50
        assert configs.blog_posts.balanced_ef == 75
        assert configs.blog_posts.max_ef == 100

        # Code Examples - Code-specific
        assert configs.code_examples.m == 18
        assert configs.code_examples.ef_construct == 250
        assert configs.code_examples.full_scan_threshold == 8000
        assert configs.code_examples.min_ef == 100
        assert configs.code_examples.balanced_ef == 125
        assert configs.code_examples.max_ef == 175

        # General - Default balanced
        assert configs.general.m == 16
        assert configs.general.ef_construct == 200
        assert configs.general.full_scan_threshold == 10000

    def test_custom_configurations(self):
        """Test custom collection configurations."""
        custom_api = HNSWConfig(m=32, ef_construct=500)
        custom_tutorials = HNSWConfig(m=14, ef_construct=180)

        configs = CollectionHNSWConfigs(
            api_reference=custom_api, tutorials=custom_tutorials
        )

        assert configs.api_reference.m == 32
        assert configs.api_reference.ef_construct == 500
        assert configs.tutorials.m == 14
        assert configs.tutorials.ef_construct == 180

        # Other collections should still have defaults
        assert configs.blog_posts.m == 12
        assert configs.code_examples.m == 18
        assert configs.general.m == 16

    def test_partial_custom_configurations(self):
        """Test that unspecified collections get defaults."""
        configs = CollectionHNSWConfigs(api_reference=HNSWConfig(m=25))

        assert configs.api_reference.m == 25
        assert configs.api_reference.ef_construct == 200  # Default
        assert configs.tutorials.m == 16  # Default factory
        assert configs.blog_posts.m == 12  # Default factory

    def test_model_serialization(self):
        """Test model serialization."""
        configs = CollectionHNSWConfigs()

        data = configs.model_dump()
        assert isinstance(data, dict)
        assert "api_reference" in data
        assert "tutorials" in data
        assert "blog_posts" in data
        assert "code_examples" in data
        assert "general" in data

        # Check nested structure
        assert data["api_reference"]["m"] == 20
        assert data["tutorials"]["m"] == 16

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            CollectionHNSWConfigs(
                api_reference=HNSWConfig(), unknown_collection=HNSWConfig()
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"
