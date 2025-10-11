"""Unit tests for validators module."""

import pytest

from src.models.validators import (
    firecrawl_api_key_validator,
    openai_api_key_validator,
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


class TestValidateApiKeyCommon:
    """Test validate_api_key_common function."""

    def test_none_value(self):
        """Test that None is returned as-is."""
        result = validate_api_key_common(None, "sk-", "OpenAI")
        assert result is None

    def test_empty_string(self):
        """Test that empty string returns None."""
        result = validate_api_key_common("", "sk-", "OpenAI")
        assert result is None

        result = validate_api_key_common("   ", "sk-", "OpenAI")
        assert result is None

    def test_valid_api_key(self):
        """Test valid API key validation."""
        key = "sk-1234567890abcdef"
        result = validate_api_key_common(key, "sk-", "OpenAI")
        assert result == key

    def test_non_ascii_characters(self):
        """Test that non-ASCII characters are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_api_key_common("sk-test™", "sk-", "OpenAI")
        assert "non-ASCII characters" in str(exc_info.value)

    def test_wrong_prefix(self):
        """Test that wrong prefix is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_api_key_common("fc-1234567890", "sk-", "OpenAI")
        assert "must start with 'sk-'" in str(exc_info.value)

    def test_too_short(self):
        """Test that too short keys are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_api_key_common("sk-123", "sk-", "OpenAI", min_length=10)
        assert "too short" in str(exc_info.value)

    def test_too_long(self):
        """Test that too long keys are rejected."""
        long_key = "sk-" + "a" * 200
        with pytest.raises(ValueError) as exc_info:
            validate_api_key_common(long_key, "sk-", "OpenAI", max_length=100)
        assert "too long" in str(exc_info.value)

    def test_invalid_characters(self):
        """Test that invalid characters are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_api_key_common("sk-test!@#", "sk-", "OpenAI")
        assert "invalid characters" in str(exc_info.value)

    def test_custom_allowed_chars(self):
        """Test custom allowed characters pattern."""
        # Default pattern allows alphanumeric and hyphen
        key1 = "sk-abc123-def456"
        result = validate_api_key_common(key1, "sk-", "OpenAI")
        assert result == key1

        # Custom pattern allows underscores
        key2 = "fc-abc_123_def"
        result = validate_api_key_common(
            key2, "fc-", "Firecrawl", allowed_chars=r"[A-Za-z0-9_-]+"
        )
        assert result == key2


class TestValidateUrlFormat:
    """Test validate_url_format function."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        url = "http://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        url = "https://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_trailing_slash_removed(self):
        """Test that trailing slash is removed."""
        result = validate_url_format("https://example.com/")
        assert result == "https://example.com"

        result = validate_url_format("https://example.com//")
        assert result == "https://example.com"

    def test_invalid_protocol(self):
        """Test that non-HTTP(S) protocols are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_url_format("ftp://example.com")
        assert "must start with http:// or https://" in str(exc_info.value)

        with pytest.raises(ValueError):
            validate_url_format("example.com")


class TestValidatePositiveInt:
    """Test validate_positive_int function."""

    def test_positive_values(self):
        """Test positive integer values."""
        assert validate_positive_int(1) == 1
        assert validate_positive_int(100) == 100
        assert validate_positive_int(999999) == 999999

    def test_zero_rejected(self):
        """Test that zero is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_positive_int(0)
        assert "must be positive" in str(exc_info.value)

    def test_negative_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            validate_positive_int(-1)
        with pytest.raises(ValueError):
            validate_positive_int(-100)

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_positive_int(0, field_name="chunk_size")
        assert "chunk_size must be positive" in str(exc_info.value)


class TestValidatePercentage:
    """Test validate_percentage function."""

    def test_valid_percentages(self):
        """Test valid percentage values."""
        assert validate_percentage(0.0) == 0.0
        assert validate_percentage(0.5) == 0.5
        assert validate_percentage(1.0) == 1.0
        assert validate_percentage(0.75) == 0.75

    def test_below_zero_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_percentage(-0.1)
        assert "must be between 0.0 and 1.0" in str(exc_info.value)

    def test_above_one_rejected(self):
        """Test that values above 1.0 are rejected."""
        with pytest.raises(ValueError):
            validate_percentage(1.1)
        with pytest.raises(ValueError):
            validate_percentage(2.0)

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_percentage(1.5, field_name="confidence_score")
        assert "confidence_score must be between 0.0 and 1.0" in str(exc_info.value)


class TestValidateRateLimitConfig:
    """Test validate_rate_limit_config function."""

    def test_valid_config(self):
        """Test valid rate limit configuration."""
        config = {
            "openai": {"max_calls": 100, "time_window": 60},
            "firecrawl": {"max_calls": 50, "time_window": 30},
        }
        result = validate_rate_limit_config(config)
        assert result == config

    def test_missing_required_keys(self):
        """Test that missing required keys are rejected."""
        config = {
            "openai": {"max_calls": 100},  # Missing time_window
        }
        with pytest.raises(ValueError) as exc_info:
            validate_rate_limit_config(config)
        assert "must contain keys" in str(exc_info.value)

    def test_non_dict_limits(self):
        """Test that non-dict limits are rejected."""
        config = {
            "openai": "invalid",  # Should be dict
        }
        with pytest.raises(TypeError) as exc_info:
            validate_rate_limit_config(config)  # type: ignore[arg-type]
        assert "must be a dictionary" in str(exc_info.value)

    def test_non_positive_max_calls(self):
        """Test that non-positive max_calls is rejected."""
        config = {
            "openai": {"max_calls": 0, "time_window": 60},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_rate_limit_config(config)
        assert "max_calls for provider 'openai' must be positive" in str(exc_info.value)

    def test_non_positive_time_window(self):
        """Test that non-positive time_window is rejected."""
        config = {
            "openai": {"max_calls": 100, "time_window": 0},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_rate_limit_config(config)
        assert "time_window for provider 'openai' must be positive" in str(
            exc_info.value
        )


class TestValidateChunkSizes:
    """Test validate_chunk_sizes function."""

    def test_valid_sizes(self):
        """Test valid chunk size configurations."""
        # Valid configuration
        validate_chunk_sizes(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
            max_chunk_size=2000,
        )
        # No exception should be raised

    def test_overlap_too_large(self):
        """Test that overlap >= chunk_size is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_chunk_sizes(
                chunk_size=1000,
                chunk_overlap=1000,  # Equal to chunk_size
                min_chunk_size=100,
                max_chunk_size=2000,
            )
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

    def test_min_greater_than_max(self):
        """Test that min >= max is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_chunk_sizes(
                chunk_size=1000,
                chunk_overlap=200,
                min_chunk_size=2000,  # Greater than max
                max_chunk_size=1500,
            )
        assert "min_chunk_size must be less than max_chunk_size" in str(exc_info.value)

    def test_chunk_size_exceeds_max(self):
        """Test that chunk_size > max_chunk_size is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_chunk_sizes(
                chunk_size=2000,
                chunk_overlap=200,
                min_chunk_size=100,
                max_chunk_size=1500,  # Less than chunk_size
            )
        assert "chunk_size cannot exceed max_chunk_size" in str(exc_info.value)


class TestValidateScoringWeights:
    """Test validate_scoring_weights function."""

    def test_valid_weights(self):
        """Test valid weight combinations."""
        # Exact 1.0
        validate_scoring_weights(0.5, 0.3, 0.2)
        validate_scoring_weights(0.33, 0.33, 0.34)
        validate_scoring_weights(1.0, 0.0, 0.0)

    def test_small_floating_point_error_allowed(self):
        """Test that small floating point errors are allowed."""
        # Total = 0.999
        validate_scoring_weights(0.333, 0.333, 0.333)
        # Total = 1.001
        validate_scoring_weights(0.334, 0.333, 0.334)

    def test_weights_not_summing_to_one(self):
        """Test that weights not summing to 1.0 are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_scoring_weights(0.5, 0.5, 0.5)  # Sum = 1.5
        assert "must sum to 1.0" in str(exc_info.value)
        assert "got 1.5" in str(exc_info.value)

        with pytest.raises(ValueError):
            validate_scoring_weights(0.2, 0.2, 0.2)  # Sum = 0.6


class TestValidateVectorDimensions:
    """Test validate_vector_dimensions function."""

    def test_common_dimensions(self):
        """Test common vector dimensions."""
        for dim in [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]:
            assert validate_vector_dimensions(dim) == dim

    def test_uncommon_dimensions_allowed(self):
        """Test that uncommon dimensions are still allowed."""
        assert validate_vector_dimensions(100) == 100
        assert validate_vector_dimensions(500) == 500
        assert validate_vector_dimensions(1500) == 1500

    def test_non_positive_rejected(self):
        """Test that non-positive dimensions are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_vector_dimensions(0)
        assert "must be positive" in str(exc_info.value)

        with pytest.raises(ValueError):
            validate_vector_dimensions(-1)

    def test_too_large_rejected(self):
        """Test that very large dimensions are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_vector_dimensions(10001)
        assert "too large" in str(exc_info.value)

    def test_with_model_name(self):
        """Test validation with model name context."""
        assert validate_vector_dimensions(384, "all-MiniLM-L6-v2") == 384


class TestValidateModelBenchmarkConsistency:
    """Test validate_model_benchmark_consistency function."""

    def test_matching_key_and_name(self):
        """Test that matching key and name pass validation."""
        key = "text-embedding-ada-002"
        model_name = "text-embedding-ada-002"
        result = validate_model_benchmark_consistency(key, model_name)
        assert result == key

    def test_mismatched_key_and_name(self):
        """Test that mismatched key and name are rejected."""
        key = "ada-002"
        model_name = "text-embedding-ada-002"
        with pytest.raises(ValueError) as exc_info:
            validate_model_benchmark_consistency(key, model_name)
        assert "does not match" in str(exc_info.value)
        assert key in str(exc_info.value)
        assert model_name in str(exc_info.value)


class TestValidateCollectionNameField:
    """Test validate_collection_name function."""

    def test_valid_names(self):
        """Test valid collection names."""
        assert validate_collection_name("documents") == "documents"
        assert validate_collection_name("test_collection") == "test_collection"
        assert validate_collection_name("my-collection-123") == "my-collection-123"
        assert validate_collection_name("AB") == "AB"  # Minimum length

    def test_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_collection_name("")
        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_characters(self):
        """Test that invalid characters are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_collection_name("my collection")  # Space
        assert "can only contain alphanumeric" in str(exc_info.value)

        with pytest.raises(ValueError):
            validate_collection_name("my.collection")  # Dot
        with pytest.raises(ValueError):
            validate_collection_name("my@collection")  # At sign

    def test_too_short(self):
        """Test that too short names are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_collection_name("a")
        assert "at least 2 characters" in str(exc_info.value)

    def test_too_long(self):
        """Test that too long names are rejected."""
        long_name = "a" * 65
        with pytest.raises(ValueError) as exc_info:
            validate_collection_name(long_name)
        assert "cannot exceed 64 characters" in str(exc_info.value)


class TestValidateEmbeddingModelName:
    """Test validate_embedding_model_name function."""

    def test_valid_names(self):
        """Test valid model names."""
        assert (
            validate_embedding_model_name("text-embedding-ada-002")
            == "text-embedding-ada-002"
        )
        assert validate_embedding_model_name("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
        assert (
            validate_embedding_model_name("model/name:version") == "model/name:version"
        )

    def test_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_embedding_model_name("")
        assert "cannot be empty" in str(exc_info.value)

    def test_too_long(self):
        """Test that too long names are rejected."""
        long_name = "a" * 201
        with pytest.raises(ValueError) as exc_info:
            validate_embedding_model_name(long_name)
        assert "too long" in str(exc_info.value)

    def test_invalid_characters(self):
        """Test that potentially dangerous characters are rejected."""
        for char in ["<", ">", "|", "&", ";"]:
            with pytest.raises(ValueError) as exc_info:
                validate_embedding_model_name(f"model{char}name")
            assert "invalid characters" in str(exc_info.value)

    def test_with_provider_context(self):
        """Test validation with provider context."""
        result = validate_embedding_model_name("ada-002", provider="openai")
        assert result == "ada-002"


class TestValidateCacheTtl:
    """Test validate_cache_ttl function."""

    def test_valid_ttl(self):
        """Test valid TTL values."""
        assert validate_cache_ttl(300) == 300
        assert validate_cache_ttl(3600) == 3600
        assert validate_cache_ttl(86400) == 86400

    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        assert validate_cache_ttl(10, min_ttl=10, max_ttl=100) == 10
        assert validate_cache_ttl(100, min_ttl=10, max_ttl=100) == 100

    def test_below_minimum(self):
        """Test that values below minimum are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_cache_ttl(30)  # Default min is 60
        assert "at least 60 seconds" in str(exc_info.value)

    def test_above_max(self):
        """Test that values above maximum are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_cache_ttl(100000)  # Default max is 86400
        assert "cannot exceed 86400 seconds" in str(exc_info.value)


class TestValidatorDecorators:
    """Test validator decorator functions."""

    def test_openai_api_key_validator(self):
        """Test OpenAI API key validator."""
        # Valid key
        key = "sk-1234567890abcdefghijklmnop"
        result = openai_api_key_validator(key)
        assert result == key

        # Invalid prefix
        with pytest.raises(ValueError) as exc_info:
            openai_api_key_validator("fc-1234567890")
        assert "OpenAI" in str(exc_info.value)

        # None value
        assert openai_api_key_validator(None) is None

    def test_firecrawl_api_key_validator(self):
        """Test Firecrawl API key validator."""
        # Valid key
        key = "fc-abc123_def-456"
        result = firecrawl_api_key_validator(key)
        assert result == key

        # Invalid prefix
        with pytest.raises(ValueError) as exc_info:
            firecrawl_api_key_validator("sk-1234567890")
        assert "Firecrawl" in str(exc_info.value)

    def test_url_validator_decorator(self):
        """Test URL validator decorator."""
        url = "https://example.com"
        result = url_validator(url)
        assert result == url

        with pytest.raises(ValueError):
            url_validator("not-a-url")
