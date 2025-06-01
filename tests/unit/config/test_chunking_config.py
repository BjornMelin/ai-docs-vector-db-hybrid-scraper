"""Test ChunkingConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.enums import ChunkingStrategy
from config.models import ChunkingConfig


class TestChunkingConfig:
    """Test ChunkingConfig model validation and behavior."""

    def test_default_values(self):
        """Test ChunkingConfig with default values."""
        config = ChunkingConfig()

        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 200

        # Code-aware chunking
        assert config.enable_ast_chunking is True
        assert config.preserve_function_boundaries is True
        assert config.preserve_code_blocks is True
        assert config.supported_languages == ["python", "javascript", "typescript"]

        # Advanced options
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000
        assert config.detect_language is True

    def test_custom_values(self):
        """Test ChunkingConfig with custom values."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC,
            chunk_size=2000,
            chunk_overlap=300,
            enable_ast_chunking=False,
            preserve_function_boundaries=False,
            preserve_code_blocks=False,
            supported_languages=["python", "go", "rust"],
            min_chunk_size=200,
            max_chunk_size=4000,
            detect_language=False,
        )

        assert config.strategy == ChunkingStrategy.BASIC
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 300
        assert config.enable_ast_chunking is False
        assert config.supported_languages == ["python", "go", "rust"]
        assert config.max_chunk_size == 4000

    def test_chunk_size_constraints(self):
        """Test chunk size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("chunk_size",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_chunk_overlap_non_negative(self):
        """Test chunk overlap must be non-negative."""
        # Valid: 0 overlap
        config = ChunkingConfig(chunk_overlap=0)
        assert config.chunk_overlap == 0

        # Invalid: negative overlap
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_overlap=-10)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("chunk_overlap",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

    def test_min_max_chunk_size_constraints(self):
        """Test min and max chunk size constraints."""
        # Valid values
        config = ChunkingConfig(min_chunk_size=50, max_chunk_size=5000)
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 5000

        # Invalid: min_chunk_size <= 0
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(min_chunk_size=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("min_chunk_size",)

        # Invalid: max_chunk_size <= 0
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(max_chunk_size=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_chunk_size",)

    def test_chunk_overlap_validation(self):
        """Test that chunk_overlap must be less than chunk_size."""
        # Invalid: overlap >= chunk_size
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "chunk_overlap must be less than chunk_size" in str(errors[0]["msg"])

        # Invalid: overlap > chunk_size
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=1000, chunk_overlap=1500)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "chunk_overlap must be less than chunk_size" in str(errors[0]["msg"])

    def test_min_max_chunk_relationship(self):
        """Test that min_chunk_size must be less than max_chunk_size."""
        # Invalid: min >= max
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(min_chunk_size=2000, max_chunk_size=1000)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "min_chunk_size must be less than max_chunk_size" in str(
            errors[0]["msg"]
        )

        # Invalid: min == max
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(min_chunk_size=1000, max_chunk_size=1000)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "min_chunk_size must be less than max_chunk_size" in str(
            errors[0]["msg"]
        )

    def test_chunk_size_max_chunk_relationship(self):
        """Test that chunk_size cannot exceed max_chunk_size."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=5000, max_chunk_size=3000)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "chunk_size cannot exceed max_chunk_size" in str(errors[0]["msg"])

    def test_complex_validation_scenario(self):
        """Test complex validation with multiple constraints."""
        # Valid configuration with all constraints satisfied
        config = ChunkingConfig(
            chunk_size=1500,  # Between min and max
            chunk_overlap=150,  # Less than chunk_size
            min_chunk_size=100,  # Less than max
            max_chunk_size=2000,  # Greater than chunk_size
        )

        assert config.chunk_size == 1500
        assert config.chunk_overlap == 150
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000

    def test_strategy_enum_validation(self):
        """Test ChunkingStrategy enum validation."""
        # Valid strategies
        for strategy in ChunkingStrategy:
            config = ChunkingConfig(strategy=strategy)
            assert config.strategy == strategy

        # Invalid strategy (not an enum value)
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(strategy="invalid_strategy")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("strategy",)

    def test_supported_languages_list(self):
        """Test supported languages list field."""
        # Empty list is valid
        config1 = ChunkingConfig(supported_languages=[])
        assert config1.supported_languages == []

        # Custom languages
        config2 = ChunkingConfig(supported_languages=["python", "java", "c++", "ruby"])
        assert len(config2.supported_languages) == 4
        assert "java" in config2.supported_languages

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=1600, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.AST,
            chunk_size=2000,
            chunk_overlap=250,
            supported_languages=["python", "rust"],
        )

        # Test model_dump
        data = config.model_dump()
        assert data["strategy"] == ChunkingStrategy.AST
        assert data["chunk_size"] == 2000
        assert data["chunk_overlap"] == 250
        assert data["supported_languages"] == ["python", "rust"]

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"chunk_size":2000' in json_str
        assert '"chunk_overlap":250' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = ChunkingConfig(chunk_size=1600, chunk_overlap=200)

        updated = original.model_copy(
            update={"chunk_size": 2000, "enable_ast_chunking": False}
        )

        assert original.chunk_size == 1600
        assert original.enable_ast_chunking is True
        assert updated.chunk_size == 2000
        assert updated.enable_ast_chunking is False
        assert updated.chunk_overlap == 200  # Preserved
