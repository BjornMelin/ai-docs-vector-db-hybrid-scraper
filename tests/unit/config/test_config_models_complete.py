"""Deterministic coverage for the unified configuration models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.config import (
    BrowserUseConfig,
    Crawl4AIConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingProvider,
    FastEmbedConfig,
    HyDEConfig,
    ObservabilityConfig,
    PerformanceConfig,
    PlaywrightConfig,
    SearchStrategy,
    SecurityConfig,
)


TEST_REDIS_PASSWORD = "test_redis_secret"  # noqa: S105

DEFAULT_EXPECTATIONS = [
    pytest.param(
        FastEmbedConfig,
        {
            "model": "BAAI/bge-small-en-v1.5",
            "cache_dir": None,
            "max_length": 512,
            "batch_size": 32,
        },
        id="fastembed",
    ),
    pytest.param(
        Crawl4AIConfig,
        {
            "browser_type": "chromium",
            "headless": True,
            "max_concurrent_crawls": 10,
            "page_timeout": 30.0,
            "remove_scripts": True,
            "remove_styles": True,
        },
        id="crawl4ai",
    ),
    pytest.param(
        PlaywrightConfig,
        {"browser": "chromium", "headless": True, "timeout": 30000},
        id="playwright",
    ),
    pytest.param(
        BrowserUseConfig,
        {
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "headless": True,
            "timeout": 30000,
            "max_retries": 3,
            "max_steps": 20,
            "disable_security": False,
            "generate_gif": False,
        },
        id="browseruse",
    ),
    pytest.param(
        HyDEConfig,
        {
            "enable_hyde": True,
            "model": "gpt-3.5-turbo",
            "num_generations": 5,
            "generation_temperature": 0.7,
            "max_tokens": 150,
            "cache_ttl": 3600,
            "query_weight": 0.3,
        },
        id="hyde",
    ),
    pytest.param(
        SecurityConfig,
        {
            "enabled": True,
            "allowed_domains": ["*"],
            "blocked_domains": [],
            "require_api_keys": True,
            "api_key_header": "X-API-Key",
            "enable_rate_limiting": True,
            "rate_limit_requests": 100,
            "rate_limit_requests_per_minute": 60,
            "default_rate_limit": 100,
            "rate_limit_window": 60,
            "x_frame_options": "DENY",
            "x_content_type_options": "nosniff",
            "x_xss_protection": "1; mode=block",
            "strict_transport_security": "max-age=31536000; includeSubDomains",
            "content_security_policy": None,
            "max_query_length": 1000,
            "max_url_length": 2048,
        },
        id="security",
    ),
    pytest.param(
        DatabaseConfig,
        {
            "database_url": "sqlite+aiosqlite:///data/app.db",
            "echo_queries": False,
            "pool_size": 20,
            "max_overflow": 10,
            "pool_timeout": 30.0,
        },
        id="database",
    ),
    pytest.param(
        PerformanceConfig,
        {
            "max_concurrent_requests": 10,
            "max_concurrent_crawls": 10,
            "max_concurrent_embeddings": 32,
            "request_timeout": 30.0,
            "max_retries": 3,
            "retry_base_delay": 1.0,
            "max_memory_usage_mb": 1000.0,
            "batch_embedding_size": 100,
            "batch_crawl_size": 50,
        },
        id="performance",
    ),
    pytest.param(
        ObservabilityConfig,
        {
            "enabled": False,
            "service_name": "ai-docs-vector-db",
            "service_version": "1.0.0",
            "service_namespace": "ai-docs",
            "otlp_endpoint": "http://localhost:4317",
            "otlp_headers": {},
            "otlp_insecure": True,
            "trace_sample_rate": 1.0,
            "track_ai_operations": True,
            "track_costs": True,
            "instrument_fastapi": True,
            "instrument_httpx": True,
            "instrument_redis": True,
            "instrument_sqlalchemy": True,
            "console_exporter": False,
        },
        id="observability",
    ),
    pytest.param(
        EmbeddingConfig,
        {
            "provider": "fastembed",
            "dense_model": "text-embedding-3-small",
            "search_strategy": "dense",
            "enable_quantization": True,
        },
        id="embedding",
    ),
]

CUSTOM_CONFIG_CASES = [
    pytest.param(
        FastEmbedConfig,
        {"model": "BAAI/bge-large-en-v1.5", "batch_size": 64, "max_length": 1024},
        {
            "model": "BAAI/bge-large-en-v1.5",
            "cache_dir": None,
            "max_length": 1024,
            "batch_size": 64,
        },
        id="fastembed",
    ),
    pytest.param(
        Crawl4AIConfig,
        {
            "browser_type": "firefox",
            "headless": False,
            "max_concurrent_crawls": 25,
            "page_timeout": 45.5,
        },
        {
            "browser_type": "firefox",
            "headless": False,
            "max_concurrent_crawls": 25,
            "page_timeout": 45.5,
            "remove_scripts": True,
            "remove_styles": True,
        },
        id="crawl4ai",
    ),
    pytest.param(
        BrowserUseConfig,
        {
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet",
            "disable_security": True,
            "generate_gif": True,
        },
        {
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet",
            "headless": True,
            "timeout": 30000,
            "max_retries": 3,
            "max_steps": 20,
            "disable_security": True,
            "generate_gif": True,
        },
        id="browseruse",
    ),
    pytest.param(
        HyDEConfig,
        {
            "enabled": False,
            "enable_hyde": False,
            "num_generations": 3,
            "generation_temperature": 0.4,
            "max_tokens": 256,
        },
        {
            "enable_hyde": False,
            "enabled": False,
            "model": "gpt-3.5-turbo",
            "num_generations": 3,
            "generation_temperature": 0.4,
            "max_tokens": 256,
            "temperature": 0.7,
            "cache_ttl": 3600,
            "query_weight": 0.3,
        },
        id="hyde",
    ),
    pytest.param(
        SecurityConfig,
        {
            "allowed_domains": ["example.com"],
            "blocked_domains": ["bad.com"],
            "rate_limit_requests": 250,
            "content_security_policy": "default-src 'self'",
        },
        {
            "enabled": True,
            "allowed_domains": ["example.com"],
            "blocked_domains": ["bad.com"],
            "require_api_keys": True,
            "api_key_header": "X-API-Key",
            "enable_rate_limiting": True,
            "rate_limit_requests": 250,
            "rate_limit_requests_per_minute": 60,
            "default_rate_limit": 100,
            "rate_limit_window": 60,
            "x_frame_options": "DENY",
            "x_content_type_options": "nosniff",
            "x_xss_protection": "1; mode=block",
            "strict_transport_security": "max-age=31536000; includeSubDomains",
            "content_security_policy": "default-src 'self'",
            "max_query_length": 1000,
            "max_url_length": 2048,
        },
        id="security",
    ),
    pytest.param(
        DatabaseConfig,
        {
            "database_url": "postgresql+asyncpg://user:pass@localhost/db",
            "pool_size": 50,
        },
        {
            "database_url": "postgresql+asyncpg://user:pass@localhost/db",
            "echo_queries": False,
            "pool_size": 50,
            "max_overflow": 10,
            "pool_timeout": 30.0,
        },
        id="database",
    ),
    pytest.param(
        PerformanceConfig,
        {
            "max_concurrent_requests": 40,
            "max_retries": 1,
            "batch_crawl_size": 10,
        },
        {
            "max_concurrent_requests": 40,
            "max_concurrent_crawls": 10,
            "max_concurrent_embeddings": 32,
            "request_timeout": 30.0,
            "max_retries": 1,
            "retry_base_delay": 1.0,
            "max_memory_usage_mb": 1000.0,
            "batch_embedding_size": 100,
            "batch_crawl_size": 10,
        },
        id="performance",
    ),
    pytest.param(
        ObservabilityConfig,
        {
            "enabled": True,
            "otlp_endpoint": "https://otel.example.com",
            "trace_sample_rate": 0.25,
            "console_exporter": True,
        },
        {
            "enabled": True,
            "service_name": "ai-docs-vector-db",
            "service_version": "1.0.0",
            "service_namespace": "ai-docs",
            "otlp_endpoint": "https://otel.example.com",
            "otlp_headers": {},
            "otlp_insecure": True,
            "trace_sample_rate": 0.25,
            "track_ai_operations": True,
            "track_costs": True,
            "instrument_fastapi": True,
            "instrument_httpx": True,
            "instrument_redis": True,
            "instrument_sqlalchemy": True,
            "console_exporter": True,
        },
        id="observability",
    ),
    pytest.param(
        EmbeddingConfig,
        {
            "provider": EmbeddingProvider.OPENAI,
            "dense_model": EmbeddingModel.TEXT_EMBEDDING_3_LARGE,
            "search_strategy": SearchStrategy.HYBRID,
            "enable_quantization": False,
        },
        {
            "provider": "openai",
            "dense_model": "text-embedding-3-large",
            "search_strategy": "hybrid",
            "enable_quantization": False,
        },
        id="embedding",
    ),
]

INVALID_CONFIG_CASES = [
    pytest.param(FastEmbedConfig, {"max_length": 0}, id="fastembed-length"),
    pytest.param(FastEmbedConfig, {"batch_size": 0}, id="fastembed-batch"),
    pytest.param(Crawl4AIConfig, {"max_concurrent_crawls": 0}, id="crawl4ai-min"),
    pytest.param(Crawl4AIConfig, {"max_concurrent_crawls": 51}, id="crawl4ai-max"),
    pytest.param(PlaywrightConfig, {"timeout": 0}, id="playwright-timeout"),
    pytest.param(BrowserUseConfig, {"max_retries": 0}, id="browseruse-retries"),
    pytest.param(HyDEConfig, {"num_generations": 0}, id="hyde-generations"),
    pytest.param(SecurityConfig, {"rate_limit_requests": 0}, id="security-rate"),
    pytest.param(DatabaseConfig, {"pool_size": 0}, id="database-pool"),
    pytest.param(
        PerformanceConfig, {"max_concurrent_requests": 0}, id="performance-requests"
    ),
    pytest.param(
        ObservabilityConfig, {"trace_sample_rate": 1.1}, id="observability-sample"
    ),
    pytest.param(EmbeddingConfig, {"provider": "invalid"}, id="embedding-provider"),
]


@pytest.mark.parametrize(("config_cls", "expected_dump"), DEFAULT_EXPECTATIONS)
def test_config_defaults_align_with_json_dump(
    config_cls: type[BaseModel], expected_dump: dict[str, Any]
) -> None:
    """Ensure each config exposes the documented defaults."""

    config = config_cls()
    assert config.model_dump(mode="json") == expected_dump


@pytest.mark.parametrize(
    ("config_cls", "payload", "expected_dump"), CUSTOM_CONFIG_CASES
)
def test_config_accepts_custom_payload(
    config_cls: type[BaseModel], payload: dict[str, Any], expected_dump: dict[str, Any]
) -> None:
    """Validate that representative overrides survive validation and serialization."""

    config = config_cls.model_validate(payload)
    assert config.model_dump(mode="json") == expected_dump


@pytest.mark.parametrize(("config_cls", "payload"), INVALID_CONFIG_CASES)
def test_config_rejects_invalid_payload(
    config_cls: type[BaseModel], payload: dict[str, Any]
) -> None:
    """Confirm guardrails stay in place for invalid input values."""

    with pytest.raises(ValidationError):
        config_cls.model_validate(payload)
