"""Shared validators and custom types for Pydantic models.

This module contains reusable validators, custom field types, and validation
utilities used across all models in the application.
"""

import re

from pydantic import Field


def validate_api_key_common(
    value: str | None,
    prefix: str,
    service_name: str,
    min_length: int = 10,
    max_length: int = 200,
    allowed_chars: str = r"[A-Za-z0-9-]+",
) -> str | None:
    """Common API key validation logic for all services.

    Args:
        value: The API key value to validate
        prefix: Required prefix for the API key
        service_name: Name of the service for error messages
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allowed_chars: Regex pattern for allowed characters

    Returns:
        Validated API key or None

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return value

    value = value.strip()
    if not value:
        return None

    # Check for ASCII-only characters (security requirement)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as err:
        raise ValueError(
            f"{service_name} API key contains non-ASCII characters"
        ) from err

    # Check required prefix
    if not value.startswith(prefix):
        raise ValueError(f"{service_name} API key must start with '{prefix}'")

    # Length validation with DoS protection
    if len(value) < min_length:
        raise ValueError(f"{service_name} API key appears to be too short")

    if len(value) > max_length:
        raise ValueError(f"{service_name} API key appears to be too long")

    # Character validation
    if not re.match(f"^{re.escape(prefix)}{allowed_chars}$", value):
        raise ValueError(f"{service_name} API key contains invalid characters")

    return value


def validate_url_format(value: str) -> str:
    """Validate URL format for various services.

    Args:
        value: URL string to validate

    Returns:
        Validated and normalized URL

    Raises:
        ValueError: If URL format is invalid
    """
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value.rstrip("/")


def validate_positive_int(value: int, field_name: str = "value") -> int:
    """Validate that an integer is positive.

    Args:
        value: Integer to validate
        field_name: Name of the field for error messages

    Returns:
        Validated integer

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def validate_percentage(value: float, field_name: str = "value") -> float:
    """Validate that a float is a valid percentage (0.0 to 1.0).

    Args:
        value: Float to validate
        field_name: Name of the field for error messages

    Returns:
        Validated percentage

    Raises:
        ValueError: If value is not a valid percentage
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


def validate_rate_limit_config(
    value: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Validate rate limit configuration structure.

    Args:
        value: Rate limit configuration dictionary

    Returns:
        Validated configuration

    Raises:
        ValueError: If configuration structure is invalid
    """
    for provider, limits in value.items():
        if not isinstance(limits, dict):
            raise ValueError(
                f"Rate limits for provider '{provider}' must be a dictionary"
            )

        required_keys = {"max_calls", "time_window"}
        if not required_keys.issubset(limits.keys()):
            raise ValueError(
                f"Rate limits for provider '{provider}' must contain "
                f"keys: {required_keys}, got: {set(limits.keys())}"
            )

        if limits["max_calls"] <= 0:
            raise ValueError(f"max_calls for provider '{provider}' must be positive")

        if limits["time_window"] <= 0:
            raise ValueError(f"time_window for provider '{provider}' must be positive")

    return value


def validate_chunk_sizes(
    chunk_size: int, chunk_overlap: int, min_chunk_size: int, max_chunk_size: int
) -> None:
    """Validate chunk size relationships.

    Args:
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum allowed chunk size
        max_chunk_size: Maximum allowed chunk size

    Raises:
        ValueError: If size relationships are invalid
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if min_chunk_size >= max_chunk_size:
        raise ValueError("min_chunk_size must be less than max_chunk_size")
    if chunk_size > max_chunk_size:
        raise ValueError("chunk_size cannot exceed max_chunk_size")


def validate_scoring_weights(
    quality_weight: float, speed_weight: float, cost_weight: float
) -> None:
    """Validate that scoring weights sum to approximately 1.0.

    Args:
        quality_weight: Weight for quality scoring
        speed_weight: Weight for speed scoring
        cost_weight: Weight for cost scoring

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    total = quality_weight + speed_weight + cost_weight
    if abs(total - 1.0) > 0.01:  # Allow small floating point errors
        raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


def validate_vector_dimensions(value: int, model_name: str = "") -> int:
    """Validate vector dimensions are within reasonable bounds.

    Args:
        value: Vector dimensions to validate
        model_name: Model name for context in error messages

    Returns:
        Validated dimensions

    Raises:
        ValueError: If dimensions are invalid
    """
    if value <= 0:
        raise ValueError("Vector dimensions must be positive")
    if value > 10000:  # Reasonable upper bound
        raise ValueError(f"Vector dimensions too large: {value}")

    # Check for common dimension sizes
    common_dims = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    if value not in common_dims:
        # Log warning but don't fail validation
        pass

    return value


def validate_model_benchmark_consistency(key: str, model_name: str) -> str:
    """Validate that dictionary key matches ModelBenchmark.model_name.

    Args:
        key: Dictionary key
        model_name: Model name from the benchmark

    Returns:
        Validated key

    Raises:
        ValueError: If key doesn't match model name
    """
    if key != model_name:
        raise ValueError(
            f"Dictionary key '{key}' does not match "
            f"ModelBenchmark.model_name '{model_name}'. "
            f"Keys must be consistent for proper model identification."
        )
    return key


def validate_collection_name(value: str) -> str:
    """Validate collection name format.

    Args:
        value: Collection name to validate

    Returns:
        Validated collection name

    Raises:
        ValueError: If name format is invalid
    """
    if not value:
        raise ValueError("Collection name cannot be empty")

    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise ValueError(
            "Collection name can only contain alphanumeric characters, "
            "underscores, and hyphens"
        )

    # Check length
    if len(value) < 2:
        raise ValueError("Collection name must be at least 2 characters")
    if len(value) > 64:
        raise ValueError("Collection name cannot exceed 64 characters")

    return value


def validate_embedding_model_name(value: str, provider: str = "") -> str:
    """Validate embedding model name format.

    Args:
        value: Model name to validate
        provider: Provider name for context

    Returns:
        Validated model name

    Raises:
        ValueError: If model name is invalid
    """
    if not value:
        raise ValueError("Model name cannot be empty")

    # Check for reasonable length
    if len(value) > 200:
        raise ValueError("Model name is too long")

    # Check for invalid characters that could cause issues
    if any(char in value for char in ["<", ">", "|", "&", ";"]):
        raise ValueError("Model name contains invalid characters")

    return value


def validate_cache_ttl(value: int, min_ttl: int = 60, max_ttl: int = 86400) -> int:
    """Validate cache TTL values.

    Args:
        value: TTL value in seconds
        min_ttl: Minimum allowed TTL
        max_ttl: Maximum allowed TTL

    Returns:
        Validated TTL

    Raises:
        ValueError: If TTL is out of range
    """
    if value < min_ttl:
        raise ValueError(f"Cache TTL must be at least {min_ttl} seconds")
    if value > max_ttl:
        raise ValueError(f"Cache TTL cannot exceed {max_ttl} seconds")
    return value


# Custom field types with built-in validation
def PositiveInt(description: str = "Positive integer") -> int:
    """Create a positive integer field."""
    return Field(gt=0, description=description)


def NonNegativeInt(description: str = "Non-negative integer") -> int:
    """Create a non-negative integer field."""
    return Field(ge=0, description=description)


def Percentage(description: str = "Percentage (0.0 to 1.0)") -> float:
    """Create a percentage field."""
    return Field(ge=0.0, le=1.0, description=description)


def PortNumber(description: str = "Port number") -> int:
    """Create a port number field."""
    return Field(ge=1, le=65535, description=description)


def CollectionName(description: str = "Collection name") -> str:
    """Create a collection name field."""
    return Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description=description,
    )


# Validation decorators for common patterns
def openai_api_key_validator(cls, v: str | None) -> str | None:
    """Validator for OpenAI API keys."""
    return validate_api_key_common(
        v, prefix="sk-", service_name="OpenAI", min_length=20, max_length=200
    )


def firecrawl_api_key_validator(cls, v: str | None) -> str | None:
    """Validator for Firecrawl API keys."""
    return validate_api_key_common(
        v,
        prefix="fc-",
        service_name="Firecrawl",
        min_length=10,
        max_length=200,
        allowed_chars=r"[A-Za-z0-9_-]+",
    )


def url_validator(cls, v: str) -> str:
    """Validator for URL fields."""
    return validate_url_format(v)


# Export all validators and utilities
__all__ = [
    "CollectionName",
    "NonNegativeInt",
    "Percentage",
    "PortNumber",
    "PositiveInt",
    "firecrawl_api_key_validator",
    "openai_api_key_validator",
    "url_validator",
    "validate_api_key_common",
    "validate_cache_ttl",
    "validate_chunk_sizes",
    "validate_collection_name",
    "validate_embedding_model_name",
    "validate_model_benchmark_consistency",
    "validate_percentage",
    "validate_positive_int",
    "validate_rate_limit_config",
    "validate_scoring_weights",
    "validate_url_format",
    "validate_vector_dimensions",
]
