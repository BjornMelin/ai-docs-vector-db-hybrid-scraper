"""Test FastEmbedConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import FastEmbedConfig


class TestFastEmbedConfig:
    """Test FastEmbedConfig model validation and behavior."""

    def test_default_values(self):
        """Test FastEmbedConfig with default values."""
        config = FastEmbedConfig()

        assert config.model == "BAAI/bge-small-en-v1.5"
        assert config.cache_dir is None
        assert config.max_length == 512
        assert config.batch_size == 32

    def test_custom_values(self):
        """Test FastEmbedConfig with custom values."""
        config = FastEmbedConfig(
            model="BAAI/bge-large-en-v1.5",
            cache_dir="/custom/cache/path",
            max_length=1024,
            batch_size=64,
        )

        assert config.model == "BAAI/bge-large-en-v1.5"
        assert config.cache_dir == "/custom/cache/path"
        assert config.max_length == 1024
        assert config.batch_size == 64

    def test_model_names(self):
        """Test various model names."""
        # Common FastEmbed models
        models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
        ]

        for model_name in models:
            config = FastEmbedConfig(model=model_name)
            assert config.model == model_name

    def test_cache_dir_optional(self):
        """Test cache_dir is optional."""
        # None by default
        config1 = FastEmbedConfig()
        assert config1.cache_dir is None

        # Can be set to a path
        config2 = FastEmbedConfig(cache_dir="./models/cache")
        assert config2.cache_dir == "./models/cache"

        # Can be absolute path
        config3 = FastEmbedConfig(cache_dir="/opt/models/fastembed")
        assert config3.cache_dir == "/opt/models/fastembed"

    def test_max_length_constraints(self):
        """Test max_length must be positive."""
        # Valid lengths
        config1 = FastEmbedConfig(max_length=128)
        assert config1.max_length == 128

        config2 = FastEmbedConfig(max_length=2048)
        assert config2.max_length == 2048

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(max_length=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_length",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: negative
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(max_length=-100)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_length",)

    def test_batch_size_constraints(self):
        """Test batch_size must be positive."""
        # Valid batch sizes
        config1 = FastEmbedConfig(batch_size=1)
        assert config1.batch_size == 1

        config2 = FastEmbedConfig(batch_size=256)
        assert config2.batch_size == 256

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(batch_size=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(model="BAAI/bge-small-en-v1.5", unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = FastEmbedConfig(
            model="intfloat/e5-large-v2",
            cache_dir="/mnt/models",
            max_length=768,
            batch_size=48,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["model"] == "intfloat/e5-large-v2"
        assert data["cache_dir"] == "/mnt/models"
        assert data["max_length"] == 768
        assert data["batch_size"] == 48

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"model":"intfloat/e5-large-v2"' in json_str
        assert '"cache_dir":"/mnt/models"' in json_str
        assert '"max_length":768' in json_str
        assert '"batch_size":48' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = FastEmbedConfig(model="BAAI/bge-small-en-v1.5", batch_size=32)

        updated = original.model_copy(
            update={
                "model": "BAAI/bge-large-en-v1.5",
                "batch_size": 64,
                "max_length": 1024,
            }
        )

        assert original.model == "BAAI/bge-small-en-v1.5"
        assert original.batch_size == 32
        assert original.max_length == 512  # Default
        assert updated.model == "BAAI/bge-large-en-v1.5"
        assert updated.batch_size == 64
        assert updated.max_length == 1024

    def test_type_validation(self):
        """Test type validation for fields."""
        # Test string field with wrong type
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(model=123)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("model",)

        # Test int field with wrong type (string that can't convert)
        with pytest.raises(ValidationError) as exc_info:
            FastEmbedConfig(batch_size="thirty-two")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)

    def test_performance_configuration_scenarios(self):
        """Test different performance configuration scenarios."""
        # Low memory configuration
        low_mem_config = FastEmbedConfig(
            model="BAAI/bge-small-en-v1.5",  # Smaller model
            max_length=256,  # Shorter sequences
            batch_size=16,  # Smaller batches
        )
        assert low_mem_config.max_length == 256
        assert low_mem_config.batch_size == 16

        # High throughput configuration
        high_throughput_config = FastEmbedConfig(
            model="BAAI/bge-base-en-v1.5",  # Balanced model
            max_length=512,
            batch_size=128,  # Larger batches
        )
        assert high_throughput_config.batch_size == 128

        # High quality configuration
        high_quality_config = FastEmbedConfig(
            model="BAAI/bge-large-en-v1.5",  # Larger model
            max_length=1024,  # Longer sequences
            batch_size=32,  # Moderate batches
        )
        assert high_quality_config.model == "BAAI/bge-large-en-v1.5"
        assert high_quality_config.max_length == 1024

    def test_cache_dir_with_none_serialization(self):
        """Test that None cache_dir serializes properly."""
        config = FastEmbedConfig(cache_dir=None)

        data = config.model_dump()
        assert data["cache_dir"] is None

        # JSON serialization should handle None
        json_str = config.model_dump_json()
        assert '"cache_dir":null' in json_str

    def test_multilingual_model_names(self):
        """Test multilingual model configurations."""
        multilingual_models = [
            "BAAI/bge-m3",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-large",
        ]

        for model_name in multilingual_models:
            config = FastEmbedConfig(model=model_name)
            assert config.model == model_name
