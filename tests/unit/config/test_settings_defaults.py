"""Regression tests for Settings defaults formerly defined as core constants."""

# pylint: disable=no-member  # Pydantic models populate attributes dynamically at runtime

import pytest
from pydantic import ValidationError  # pyright: ignore[reportMissingImports]

from src.config.loader import Settings  # pyright: ignore[reportMissingImports]


def test_cache_ttl_defaults() -> None:
    """Cache defaults should retain the canonical TTL from the legacy constants."""

    settings = Settings()

    assert settings.cache.ttl_search_results == 3600


def test_cache_ttl_search_results_negative_value() -> None:
    """Negative TTL values for search results should be rejected."""

    with pytest.raises(ValidationError):
        Settings(cache={"ttl_search_results": -100})


def test_chunking_defaults() -> None:
    """Chunking defaults should expose the canonical chunk size and overlap."""

    settings = Settings()

    assert settings.chunking.chunk_size == 1600
    assert settings.chunking.chunk_overlap == 320


def test_circuit_breaker_defaults() -> None:
    """Circuit breaker defaults should match the historical thresholds."""

    settings = Settings()

    assert settings.performance.request_timeout == 30.0
    assert settings.performance.max_retries == 3
    assert settings.performance.retry_base_delay == 1.0
    assert settings.circuit_breaker.failure_threshold == 5
    assert settings.circuit_breaker.recovery_timeout == 60.0
    assert settings.circuit_breaker.service_overrides == {
        "openai": {"failure_threshold": 3, "recovery_timeout": 30.0},
        "firecrawl": {"failure_threshold": 5, "recovery_timeout": 60.0},
        "qdrant": {"failure_threshold": 3, "recovery_timeout": 15.0},
        "redis": {"failure_threshold": 2, "recovery_timeout": 10.0},
    }
