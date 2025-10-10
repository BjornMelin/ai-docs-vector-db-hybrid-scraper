"""Tests for configuration validation in the unified MCP server."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from src import unified_mcp_server
from src.config import Settings, refresh_settings
from src.config.models import CrawlProvider, EmbeddingProvider, Environment


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure the cached settings object is reset around each test."""

    refresh_settings()
    yield
    refresh_settings()


def _settings_with(
    *,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.FASTEMBED,
    openai_key: str | None = "sk-demo",
    crawl_provider: CrawlProvider = CrawlProvider.CRAWL4AI,
    firecrawl_key: str | None = "fc-demo",
    qdrant_url: str | None = "http://localhost:6333",
    environment: Environment = Environment.DEVELOPMENT,
) -> Settings:
    payload = {
        "embedding": {"provider": embedding_provider.value},
        "openai": {"api_key": openai_key},
        "crawl_provider": crawl_provider.value,
        "firecrawl": {"api_key": firecrawl_key},
        "qdrant": {"url": qdrant_url},
        "environment": environment.value,
    }
    return Settings.model_validate(payload)


def test_validate_configuration_requires_openai_key() -> None:
    """An OpenAI embedding provider must supply an API key."""

    config = _settings_with(
        embedding_provider=EmbeddingProvider.OPENAI,
        openai_key=None,
    )

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        pytest.raises(ValueError) as exc_info,
    ):
        unified_mcp_server.validate_configuration()

    assert "OpenAI API key is required" in str(exc_info.value)


def test_validate_configuration_requires_qdrant_url() -> None:
    """Qdrant connectivity is mandatory for the MCP server."""

    config = _settings_with(qdrant_url="")

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        pytest.raises(ValueError) as exc_info,
    ):
        unified_mcp_server.validate_configuration()

    assert "Qdrant URL is required" in str(exc_info.value)


def test_validate_configuration_warns_on_missing_firecrawl(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Firecrawl key absence should be logged as a warning when optional."""

    config = _settings_with(
        crawl_provider=CrawlProvider.FIRECRAWL,
        firecrawl_key=None,
        environment=Environment.TESTING,
    )

    with patch("src.unified_mcp_server.get_settings", return_value=config):
        unified_mcp_server.validate_configuration()

    assert "Firecrawl API key not set" in caplog.text


def test_validate_configuration_accepts_valid_payload() -> None:
    """Valid configurations should pass downstream streaming validation."""

    config = _settings_with()

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config") as mock_validate,
    ):
        unified_mcp_server.validate_configuration()

    mock_validate.assert_called_once()


@pytest.mark.parametrize(
    "env_vars,expected_errors,expected_warnings",
    [
        (
            {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_PORT": "not-a-number"},
            ["Invalid port value"],
            [],
        ),
        (
            {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_BUFFER_SIZE": "-1024"},
            [],
            ["Buffer size"],
        ),
        (
            {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_BUFFER_SIZE": "abc"},
            ["Invalid buffer size"],
            [],
        ),
        (
            {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_MAX_RESPONSE_SIZE": "-1"},
            ["Max response size"],
            [],
        ),
    ],
)
def test_validate_streaming_config_records_issues(
    env_vars: dict[str, str],
    expected_errors: list[str],
    expected_warnings: list[str],
) -> None:
    """Streaming misconfiguration should surface as errors or warnings."""

    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(os.environ, env_vars, clear=True):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    for substring in expected_errors:
        assert any(substring in error for error in errors)
    for substring in expected_warnings:
        assert any(substring in warning for warning in warnings)

    if not expected_errors:
        assert not errors
    if not expected_warnings:
        assert not warnings


def test_validate_streaming_config_skips_non_streamable_transport() -> None:
    """Alternative transports bypass streamable-http validation."""

    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "stdio"}, clear=True):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert not errors
    assert not warnings
