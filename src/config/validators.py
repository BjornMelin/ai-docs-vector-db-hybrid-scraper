"""Configuration validators for the unified config system.

This module provides validation functions for configuration models,
avoiding circular import dependencies by keeping validators close to the models.
"""

import re


def validate_api_key_common(
    value: str | None,
    prefix: str,
    service_name: str,
    min_length: int = 10,
    max_length: int = 200,
    allowed_chars: str = r"[A-Za-z0-9-]+",
) -> str | None:
    """Validate API key format and structure.

    Args:
        value: API key value to validate
        prefix: Expected prefix (e.g., "sk-", "fc-")
        service_name: Service name for error messages
        min_length: Minimum key length
        max_length: Maximum key length
        allowed_chars: Regex pattern for allowed characters

    Returns:
        Validated API key or None

    Raises:
        ValueError: If API key format is invalid
    """
    if value is None:
        return value
    value = value.strip()
    if not value:
        return None
    try:
        value.encode("ascii")
    except UnicodeEncodeError as err:
        raise ValueError(
            f"{service_name} API key contains non-ASCII characters"
        ) from err
    if not value.startswith(prefix):
        raise ValueError(f"{service_name} API key must start with '{prefix}'")

    # Special handling for OpenAI test keys - check characters first, then length
    if service_name == "OpenAI" and value.startswith("sk-test"):
        # Test keys still need to follow character rules
        if not re.match(r"^sk-test[A-Za-z0-9-]+$", value):
            raise ValueError(f"{service_name} test API key contains invalid characters")
        # Relaxed length requirements for test keys
        if len(value) < 8:  # Minimum reasonable test key length
            raise ValueError(f"{service_name} test API key appears to be too short")
        return value

    if len(value) < min_length:
        raise ValueError(f"{service_name} API key appears to be too short")
    if len(value) > max_length:
        raise ValueError(f"{service_name} API key appears to be too long")

    if not re.match(f"^{re.escape(prefix)}{allowed_chars}$", value):
        raise ValueError(f"{service_name} API key contains invalid characters")
    return value


def validate_url_format(value: str) -> str:
    """Validate URL format.

    Args:
        value: URL to validate

    Returns:
        Validated URL with trailing slash removed

    Raises:
        ValueError: If URL format is invalid
    """
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value.rstrip("/")


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
        ValueError: If chunk size relationships are invalid
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if min_chunk_size >= max_chunk_size:
        raise ValueError("min_chunk_size must be less than max_chunk_size")
    if chunk_size > max_chunk_size:
        raise ValueError("chunk_size cannot exceed max_chunk_size")


def validate_rate_limit_config(
    value: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Validate rate limit configuration structure.

    Args:
        value: Rate limit configuration dictionary

    Returns:
        Validated rate limit configuration

    Raises:
        ValueError: If rate limit structure is invalid
    """
    for provider, limits in value.items():
        if not isinstance(limits, dict):
            raise ValueError(
                f"Rate limits for provider '{provider}' must be a dictionary"
            )
        required_keys = {"max_calls", "time_window"}
        if not required_keys.issubset(limits.keys()):
            raise ValueError(
                f"Rate limits for provider '{provider}' must contain keys: {required_keys}, got: {set(limits.keys())}"
            )
        if limits["max_calls"] <= 0:
            raise ValueError(f"max_calls for provider '{provider}' must be positive")
        if limits["time_window"] <= 0:
            raise ValueError(f"time_window for provider '{provider}' must be positive")
    return value


def validate_scoring_weights(
    quality_weight: float, speed_weight: float, cost_weight: float
) -> None:
    """Validate that scoring weights sum to approximately 1.0.

    Args:
        quality_weight: Quality scoring weight
        speed_weight: Speed scoring weight
        cost_weight: Cost scoring weight

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    total = quality_weight + speed_weight + cost_weight
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


def validate_model_benchmark_consistency(key: str, model_name: str) -> str:
    """Validate model benchmark key consistency.

    Args:
        key: Dictionary key
        model_name: Model name from benchmark

    Returns:
        Validated key

    Raises:
        ValueError: If key doesn't match model name
    """
    if key != model_name:
        raise ValueError(
            f"Dictionary key '{key}' does not match ModelBenchmark.model_name '{model_name}'. Keys must be consistent for proper model identification."
        )
    return key
