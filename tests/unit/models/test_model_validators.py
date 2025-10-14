"""Unit tests for :mod:`src.models.validators`."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import TypeAdapter

from src.models.validators import (
    collection_name_field,
    firecrawl_api_key_validator,
    non_negative_int,
    openai_api_key_validator,
    percentage,
    port_number,
    positive_int,
    url_validator,
    validate_api_key_common,
    validate_cache_ttl,
    validate_chunk_sizes,
    validate_collection_name,
    validate_embedding_model_name,
    validate_model_benchmark_consistency,
    validate_percentage,
    validate_positive_int,
    validate_rate_limit_config,
    validate_scoring_weights,
    validate_url_format,
    validate_vector_dimensions,
)


class TestApiKeyValidation:
    """Shared API key validation helpers."""

    def test_none_is_returned(self) -> None:
        """Test that None passes through unchanged."""
        assert validate_api_key_common(None, "sk-", "OpenAI") is None

    def test_empty_or_whitespace_collapses_to_none(self) -> None:
        """Test that empty or whitespace-only strings normalize to None."""
        assert validate_api_key_common("", "sk-", "OpenAI") is None
        assert validate_api_key_common("   ", "sk-", "OpenAI") is None

    def test_valid_key_round_trip(self) -> None:
        """Test that valid API key returns unchanged."""
        key = "sk-1234567890abcdef"
        assert validate_api_key_common(key, "sk-", "OpenAI") == key

    def test_non_ascii_rejected(self) -> None:
        """Test that keys containing non-ASCII characters raise ValueError."""
        with pytest.raises(ValueError, match="contains non-ASCII characters"):
            validate_api_key_common("sk-test™", "sk-", "OpenAI")

    def test_wrong_prefix_rejected(self) -> None:
        """Test that keys with incorrect prefix raise ValueError."""
        with pytest.raises(ValueError, match="must start with 'sk-'"):
            validate_api_key_common("fc-123456", "sk-", "OpenAI")

    def test_too_short_rejected_when_not_test_key(self) -> None:
        """Test that short keys without test prefix raise ValueError."""
        with pytest.raises(ValueError, match="appears to be too short"):
            validate_api_key_common("sk-123", "sk-", "OpenAI")

    def test_test_keys_skip_length_enforcement(self) -> None:
        """Test that keys with test prefix bypass length validation."""
        assert validate_api_key_common("sk-test-1", "sk-", "OpenAI") == "sk-test-1"

    def test_too_long_rejected(self) -> None:
        """Test that excessively long keys raise ValueError."""
        long_value = "sk-" + "x" * 300
        with pytest.raises(ValueError, match="appears to be too long"):
            validate_api_key_common(long_value, "sk-", "OpenAI")

    def test_invalid_characters_rejected(self) -> None:
        """Test that keys with disallowed characters raise ValueError."""
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key_common("sk-not_allowed!", "sk-", "OpenAI")


class TestDecoratorValidators:
    """High-level decorator helpers."""

    def test_openai_api_key_validator(self) -> None:
        """Test that OpenAI validator enforces sk- prefix."""
        key = "sk-1234567890abcdefghijklmnop"
        assert openai_api_key_validator(key) == key
        with pytest.raises(ValueError, match="OpenAI API key must start"):
            openai_api_key_validator("fc-123")
        assert openai_api_key_validator(None) is None

    def test_firecrawl_api_key_validator(self) -> None:
        """Test that Firecrawl validator enforces fc- prefix."""
        key = "fc-abc_123-xyz"
        assert firecrawl_api_key_validator(key) == key
        with pytest.raises(ValueError, match="Firecrawl API key must start"):
            firecrawl_api_key_validator("sk-123")

    def test_url_validator_delegates(self) -> None:
        """Test that url_validator enforces http/https schemes."""
        assert url_validator("https://example.com") == "https://example.com"
        with pytest.raises(ValueError, match="must start with http:// or https://"):
            url_validator("ftp://example.com")


class TestSimpleValidators:
    """Scalar validators."""

    def test_validate_url_format_trims_slashes(self) -> None:
        """Test that trailing slashes are stripped from URLs."""
        assert validate_url_format("https://example.com/") == "https://example.com"

    def test_validate_positive_int(self) -> None:
        """Test that zero or negative integers raise ValueError."""
        assert validate_positive_int(5, field_name="chunks") == 5
        with pytest.raises(ValueError, match="chunks must be positive"):
            validate_positive_int(0, field_name="chunks")

    def test_validate_percentage(self) -> None:
        """Test that values outside 0.0-1.0 range raise ValueError."""
        assert validate_percentage(0.25, field_name="confidence") == 0.25
        with pytest.raises(ValueError, match=r"confidence must be between 0.0 and 1.0"):
            validate_percentage(1.5, field_name="confidence")

    def test_validate_cache_ttl_bounds(self) -> None:
        """Test that cache TTL enforces 60-86400 second bounds."""
        assert validate_cache_ttl(600) == 600
        with pytest.raises(ValueError, match="at least 60 seconds"):
            validate_cache_ttl(10)
        with pytest.raises(ValueError, match="cannot exceed 86400 seconds"):
            validate_cache_ttl(100_000)


class TestRateLimitConfig:
    """Rate limit configuration validation."""

    def test_valid_config_passes(self) -> None:
        """Test that well-formed rate limit config validates successfully."""
        payload = {"openai": {"max_calls": 100, "time_window": 60}}
        assert validate_rate_limit_config(payload) == payload

    def test_limits_must_be_mapping(self) -> None:
        """Test that provider limits must be dictionary not scalar."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            validate_rate_limit_config({"openai": "invalid"})  # type: ignore[arg-type]

    def test_required_keys_present(self) -> None:
        """Test that missing required keys raise ValueError."""
        with pytest.raises(ValueError, match="must contain"):
            validate_rate_limit_config({"openai": {"max_calls": 1}})

    def test_positive_numbers_enforced(self) -> None:
        """Test that zero or negative limit values raise ValueError."""
        with pytest.raises(
            ValueError, match="max_calls for provider 'openai' must be positive"
        ):
            validate_rate_limit_config({"openai": {"max_calls": 0, "time_window": 60}})
        with pytest.raises(
            ValueError, match="time_window for provider 'openai' must be positive"
        ):
            validate_rate_limit_config({"openai": {"max_calls": 1, "time_window": 0}})


class TestChunkAndWeightValidators:
    """Chunk sizing and scoring weight checks."""

    def test_validate_chunk_sizes(self) -> None:
        """Test that chunk_size must be positive and overlap must be smaller."""
        validate_chunk_sizes(500, 100)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_chunk_sizes(0, 10)
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            validate_chunk_sizes(200, -1)
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            validate_chunk_sizes(100, 100)

    def test_validate_scoring_weights(self) -> None:
        """Test that scoring weights must sum to 1.0 within tolerance."""
        validate_scoring_weights(0.6, 0.3, 0.1)
        with pytest.raises(
            ValueError, match=r"Scoring weights must sum to 1.0, got 1.5"
        ):
            validate_scoring_weights(0.5, 0.5, 0.5)
        with pytest.raises(ValueError, match=r"Scoring weights must sum to 1.0"):
            validate_scoring_weights(0.2, 0.2, 0.2)


class TestVectorAndEmbeddingValidators:
    """Vector dimension and embedding model checks."""

    @pytest.mark.parametrize("dimension", [128, 384, 4096])
    def test_common_dimensions_allowed(self, dimension: int) -> None:
        """Test that common vector dimensions pass validation."""
        assert validate_vector_dimensions(dimension) == dimension

    def test_dimension_bounds(self) -> None:
        """Test that vector dimensions enforce 1-10000 range."""
        with pytest.raises(ValueError, match="Vector dimensions must be positive"):
            validate_vector_dimensions(0)
        with pytest.raises(ValueError, match="Vector dimensions too large: 10001"):
            validate_vector_dimensions(10_001)

    def test_model_benchmark_consistency(self) -> None:
        """Test that model name must match benchmark model."""
        assert (
            validate_model_benchmark_consistency(
                "text-embedding-ada-002", "text-embedding-ada-002"
            )
            == "text-embedding-ada-002"
        )
        with pytest.raises(ValueError, match="does not match"):
            validate_model_benchmark_consistency("ada-002", "text-embedding-ada-002")

    def test_collection_name_rules(self) -> None:
        """Test that collection names enforce limits.

        Limits include alphanumeric, hyphens, underscores, and length limits.
        """
        assert validate_collection_name("docs-01") == "docs-01"
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_collection_name("")
        with pytest.raises(ValueError, match="can only contain alphanumeric"):
            validate_collection_name("bad name")
        with pytest.raises(ValueError, match="must be at least 2 characters"):
            validate_collection_name("a")
        with pytest.raises(ValueError, match="cannot exceed 64 characters"):
            validate_collection_name("x" * 65)

    def test_embedding_model_name_rules(self) -> None:
        """Test that embedding model names enforce char whitelist and 200 char limit."""
        assert validate_embedding_model_name("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            validate_embedding_model_name("")
        with pytest.raises(ValueError, match="Model name is too long"):
            validate_embedding_model_name("x" * 201)
        with pytest.raises(ValueError, match="Model name contains invalid characters"):
            validate_embedding_model_name("bad|name")


class TestFieldFactories:
    """Ensure field helpers enforce constraints at runtime."""

    @pytest.mark.parametrize(
        ("annotation", "valid", "invalid"),
        [
            (Annotated[int, positive_int()], 5, 0),
            (Annotated[int, non_negative_int()], 0, -1),
            (Annotated[int, port_number()], 8080, 0),
            (Annotated[float, percentage()], 0.5, 1.5),
            (Annotated[str, collection_name_field()], "docs-01", "bad name"),
        ],
    )
    def test_field_factories_validate_values(
        self, annotation: Any, valid: Any, invalid: Any
    ) -> None:
        """Test that field factory annotations enforce constraints at runtime."""
        adapter = TypeAdapter(annotation)
        assert adapter.validate_python(valid) == valid
        with pytest.raises(ValueError):
            adapter.validate_python(invalid)
