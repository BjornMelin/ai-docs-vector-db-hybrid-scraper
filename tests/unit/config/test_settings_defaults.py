"""Regression tests for Settings defaults formerly defined as core constants."""

from src.config.loader import Settings


def test_cache_ttl_defaults() -> None:
    """Cache defaults should retain the canonical TTL from the legacy constants."""

    settings = Settings()

    assert settings.cache.ttl_search_results == 3600


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
