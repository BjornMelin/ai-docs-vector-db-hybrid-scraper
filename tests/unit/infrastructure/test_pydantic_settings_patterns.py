"""Unit tests for the configuration loader.

The suite exercises the pydantic-settings integration."""

# pylint: disable=no-member

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings, refresh_settings
from src.config.models import (
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    SearchStrategy,
)


@pytest.fixture(autouse=True)
def _clean_config_cache() -> Iterator[None]:
    """Ensure global config cache does not leak between tests."""

    refresh_settings()
    yield
    refresh_settings()


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
    """Environment variables, including nested keys, populate the Settings model."""

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

    config = Settings()
    snapshot = config.model_dump()

    assert config.app_name == "env-app"
    assert config.environment == Environment.STAGING
    assert snapshot["cache"]["ttl_embeddings"] == 123
    assert snapshot["openai"]["api_key"] == "sk-test"
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
        Settings.model_validate(payload)

    assert expected_message in str(exc_info.value)


def test_config_allows_missing_provider_keys_in_testing_environment() -> None:
    """Testing configs bypass provider key validation for ease of local development."""

    config = Settings.model_validate(
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


def test_config_nested_credentials_and_urls(tmp_path: Path) -> None:
    """Nested credentials and URLs should be preserved without side effects."""

    overrides = _path_overrides(tmp_path)
    config = Settings.model_validate(
        {
            "environment": Environment.TESTING,
            "openai": {"api_key": "sk-live"},
            "firecrawl": {"api_key": "fc-live"},
            "qdrant": {"api_key": "qd-live", "url": "http://qdrant:6333"},
            "cache": {"redis_url": "redis://redis:6379/1"},
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


def test_config_feature_flags_and_helpers(tmp_path: Path) -> None:
    """Feature flags control optional capabilities without mode profiles."""

    overrides = _path_overrides(tmp_path)
    config = Settings.model_validate(
        {
            "environment": Environment.TESTING,
            "enable_advanced_monitoring": False,
            "enable_deployment_features": False,
            "enable_ab_testing": True,
            "observability": {"enabled": True},
            "chunking": {"strategy": ChunkingStrategy.AST_AWARE},
            "embedding": {"search_strategy": SearchStrategy.HYBRID},
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )

    feature_flags = config.get_feature_flags()

    assert feature_flags == {
        "advanced_monitoring": False,
        "deployment_features": False,
        "a_b_testing": True,
        "comprehensive_observability": True,
    }
    assert config.get_effective_chunking_strategy() is ChunkingStrategy.AST_AWARE
    assert config.get_effective_search_strategy() is SearchStrategy.HYBRID


def test_global_config_cache_and_reset(tmp_path: Path) -> None:
    """Global helpers should manage cached configuration instances predictably."""

    overrides = _path_overrides(tmp_path)
    refresh_settings()

    config_a = get_settings()
    config_b = get_settings()
    assert config_a is config_b

    replacement = Settings.model_validate(
        {
            "environment": Environment.TESTING,
            "data_dir": overrides["data_dir"],
            "cache_dir": overrides["cache_dir"],
            "logs_dir": overrides["logs_dir"],
        }
    )
    refresh_settings(settings=replacement)
    assert get_settings() is replacement

    refresh_settings()
    config_c = get_settings()
    assert config_c is not replacement
