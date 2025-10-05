"""Unit tests for the configuration loader.

The suite exercises the pydantic-settings integration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError

from src.config import (
    ApplicationMode,
    AutoDetectedServices,
    ChunkingStrategy,
    Config,
    CrawlProvider,
    DetectedEnvironment,
    DetectedService,
    EmbeddingProvider,
    Environment,
    SearchStrategy,
    get_config,
    get_config_with_auto_detection,
    reset_config,
    set_config,
)


@pytest.fixture(autouse=True)
def _clean_config_cache() -> Iterator[None]:
    """Ensure global config cache does not leak between tests."""

    reset_config()
    yield
    reset_config()


class PathOverrides(TypedDict):
    """Typed mapping of configuration directory overrides."""

    data_dir: Path
    cache_dir: Path
    logs_dir: Path


def _path_overrides(tmp_path: Path) -> PathOverrides:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    logs_dir = tmp_path / "logs"
    return {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "logs_dir": logs_dir,
    }


def test_config_loads_env_values_and_nested_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Environment variables, including nested keys, populate the Config model."""

    overrides = _path_overrides(tmp_path)
    monkeypatch.setenv("AI_DOCS_APP_NAME", "env-app")
    monkeypatch.setenv("AI_DOCS_ENVIRONMENT", "staging")
    monkeypatch.setenv("AI_DOCS_CACHE__TTL_EMBEDDINGS", "123")
    monkeypatch.setenv(
        "AI_DOCS_OPENAI_API_KEY", ""
    )  # ignored thanks to env_ignore_empty
    monkeypatch.setenv("AI_DOCS_OPENAI__API_KEY", "sk-test")
    monkeypatch.setenv("AI_DOCS_CACHE__ENABLE_CACHING", "false")
    monkeypatch.setenv("AI_DOCS_DATA_DIR", str(overrides["data_dir"]))
    monkeypatch.setenv("AI_DOCS_CACHE_DIR", str(overrides["cache_dir"]))
    monkeypatch.setenv("AI_DOCS_LOGS_DIR", str(overrides["logs_dir"]))

    config = Config()
    snapshot = config.model_dump()

    assert config.app_name == "env-app"
    assert config.environment == Environment.STAGING
    assert snapshot["cache"]["ttl_embeddings"] == 123
    assert snapshot["openai"]["api_key"] == "sk-test"
    assert config.openai_api_key is None  # empty string treated as missing
    assert snapshot["cache"]["enable_caching"] is False
    assert overrides["data_dir"].exists()
    assert overrides["cache_dir"].exists()
    assert overrides["logs_dir"].exists()


@pytest.mark.parametrize(
    ("provider_kwargs", "expected_message"),
    [
        (
            {"embedding_provider": EmbeddingProvider.OPENAI},
            "OpenAI API key required",
        ),
        (
            {"crawl_provider": CrawlProvider.FIRECRAWL},
            "Firecrawl API key required",
        ),
    ],
)
def test_config_requires_provider_keys_outside_testing(
    provider_kwargs: dict[str, object], expected_message: str
) -> None:
    """Production/staging configs should enforce provider-specific API keys."""

    payload: dict[str, object] = {"environment": Environment.PRODUCTION}
    payload.update(provider_kwargs)

    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(payload)

    assert expected_message in str(exc_info.value)


def test_config_allows_missing_provider_keys_in_testing_environment() -> None:
    """Testing configs bypass provider key validation for ease of local development."""

    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "embedding_provider": EmbeddingProvider.OPENAI,
            "crawl_provider": CrawlProvider.FIRECRAWL,
        }
    )
    snapshot = config.model_dump()

    assert config.embedding_provider is EmbeddingProvider.OPENAI
    assert config.crawl_provider is CrawlProvider.FIRECRAWL
    assert snapshot["openai"]["api_key"] is None
    assert snapshot["firecrawl"]["api_key"] is None


def test_config_synchronizes_service_credentials_and_urls(tmp_path: Path) -> None:
    """Ensure top-level credentials and URLs cascade into nested config sections."""

    overrides = _path_overrides(tmp_path)
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "openai_api_key": "sk-live",
            "firecrawl_api_key": "fc-live",
            "qdrant_api_key": "qd-live",
            "qdrant_url": "http://qdrant:6333",
            "redis_url": "redis://redis:6379/1",
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    snapshot = config.model_dump()

    assert snapshot["openai"]["api_key"] == "sk-live"
    assert snapshot["firecrawl"]["api_key"] == "fc-live"
    assert snapshot["qdrant"]["api_key"] == "qd-live"
    assert snapshot["qdrant"]["url"] == "http://qdrant:6333"
    assert snapshot["cache"]["redis_url"] == "redis://redis:6379/1"
    assert snapshot["task_queue"]["redis_url"] == "redis://redis:6379/1"


def test_config_simple_mode_adjustments_and_helpers(tmp_path: Path) -> None:
    """Simple mode should clamp aggressive settings and expose simplified helpers."""

    overrides = _path_overrides(tmp_path)
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "mode": ApplicationMode.SIMPLE,
            "performance": {"max_concurrent_crawls": 25},
            "cache": {"local_max_memory_mb": 512},
            "reranking": {"enabled": True},
            "observability": {"enabled": True},
            "chunking": {"strategy": ChunkingStrategy.ENHANCED},
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    snapshot = config.model_dump()

    assert snapshot["performance"]["max_concurrent_crawls"] == 10
    assert snapshot["cache"]["local_max_memory_mb"] == 200
    assert snapshot["reranking"]["enabled"] is False
    assert snapshot["observability"]["enabled"] is False
    assert config.get_effective_chunking_strategy() is ChunkingStrategy.BASIC
    assert config.get_effective_search_strategy() is SearchStrategy.DENSE


def test_config_enterprise_mode_adjustments(tmp_path: Path) -> None:
    """Enterprise mode retains advanced features while keeping concurrency sensible."""

    overrides = _path_overrides(tmp_path)
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "mode": ApplicationMode.ENTERPRISE,
            "performance": {"max_concurrent_crawls": 45},
            "chunking": {"strategy": ChunkingStrategy.AST_AWARE},
            "reranking": {"enabled": True},
            "observability": {"enabled": True},
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    snapshot = config.model_dump()

    assert snapshot["performance"]["max_concurrent_crawls"] == 45
    assert snapshot["reranking"]["enabled"] is True
    assert snapshot["observability"]["enabled"] is True
    assert config.get_effective_chunking_strategy() is ChunkingStrategy.AST_AWARE
    assert config.get_effective_search_strategy() is SearchStrategy.HYBRID


def test_config_auto_detection_storage_round_trip() -> None:
    """Private auto-detected services storage should retain assignments."""

    config = Config(environment=Environment.TESTING)
    detected = AutoDetectedServices(
        environment=DetectedEnvironment(environment_type=Environment.STAGING),
        services=[
            DetectedService(name="qdrant", url="http://qdrant:6333", healthy=True),
        ],
        errors=["timeout"],
    )
    config.set_auto_detected_services(detected)

    assert config.get_auto_detected_services() is detected


def test_global_config_cache_and_reset(tmp_path: Path) -> None:
    """Global helpers should manage cached configuration instances predictably."""

    overrides = _path_overrides(tmp_path)
    reset_config()

    config_a = get_config(force_reload=True)
    config_b = get_config()
    assert config_a is config_b

    replacement = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    set_config(replacement)
    assert get_config() is replacement

    reset_config()
    config_c = get_config()
    assert config_c is not replacement


@pytest.mark.asyncio
async def test_get_config_with_auto_detection_enabled(tmp_path: Path) -> None:
    """Auto detection should populate services when the feature is enabled."""

    overrides = _path_overrides(tmp_path)
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "auto_detection": {"enabled": True},
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    config.set_auto_detected_services(None)
    set_config(config)

    try:
        result = await get_config_with_auto_detection()
    finally:
        reset_config()

    services = result.get_auto_detected_services()
    assert services is not None
    assert services.environment.environment_type is Environment.DEVELOPMENT


@pytest.mark.asyncio
async def test_get_config_with_auto_detection_disabled(tmp_path: Path) -> None:
    """Auto detection should clear stored services when the feature is disabled."""

    overrides = _path_overrides(tmp_path)
    preexisting = AutoDetectedServices(
        environment=DetectedEnvironment(environment_type=Environment.STAGING),
    )
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "auto_detection": {"enabled": False},
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    config.set_auto_detected_services(preexisting)
    set_config(config)

    try:
        result = await get_config_with_auto_detection()
    finally:
        reset_config()

    assert result.get_auto_detected_services() is None
