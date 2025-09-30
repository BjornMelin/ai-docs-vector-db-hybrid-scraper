"""Tests for configuration validation in the unified MCP server."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import unified_mcp_server


def _build_config(
    *,
    openai_key: str | None = "sk-123",
    firecrawl_key: str | None = "fc-123",
    qdrant_url: str | None = "http://localhost:6333",
    providers: list[str] | None = None,
    crawling_providers: list[str] | None = None,
):
    providers = providers or ["openai"]
    crawling_providers = crawling_providers or ["firecrawl"]
    return SimpleNamespace(
        get_active_providers=lambda: providers,
        openai=SimpleNamespace(api_key=openai_key),
        crawling=SimpleNamespace(providers=crawling_providers),
        firecrawl=SimpleNamespace(api_key=firecrawl_key),
        qdrant=SimpleNamespace(url=qdrant_url),
        cache=SimpleNamespace(
            enable_dragonfly_cache=False,
            enable_local_cache=False,
            dragonfly_url=None,
        ),
        monitoring=SimpleNamespace(
            enabled=False,
            include_system_metrics=False,
            system_metrics_interval=60,
        ),
    )


def test_validate_configuration_requires_openai_key():
    config = _build_config(openai_key=None)

    with (
        patch("src.unified_mcp_server.get_config", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
        pytest.raises(ValueError, match="OpenAI API key is required"),
    ):
        unified_mcp_server.validate_configuration()


def test_validate_configuration_requires_qdrant_url():
    config = _build_config(
        qdrant_url=None, providers=["fastembed"], crawling_providers=[]
    )

    with (
        patch("src.unified_mcp_server.get_config", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
        pytest.raises(ValueError, match="Qdrant URL is required"),
    ):
        unified_mcp_server.validate_configuration()


def test_validate_configuration_warns_on_missing_firecrawl(caplog):
    config = _build_config(
        firecrawl_key=None, providers=["fastembed"], crawling_providers=["firecrawl"]
    )

    with (
        patch("src.unified_mcp_server.get_config", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
    ):
        unified_mcp_server.validate_configuration()

    assert "Firecrawl API key not set" in caplog.text


def test_validate_configuration_passes_for_valid_config():
    config = _build_config(
        firecrawl_key="", providers=["fastembed"], crawling_providers=["other"]
    )

    with (
        patch("src.unified_mcp_server.get_config", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config") as mock_validate,
    ):
        unified_mcp_server.validate_configuration()

    mock_validate.assert_called_once()


def test_validate_streaming_config_handles_invalid_port():
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_PORT": "not-a-number"},
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert any("Invalid port value" in error for error in errors)


def test_validate_streaming_config_skips_non_streamable_transport():
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {"FASTMCP_TRANSPORT": "stdio"},
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert not errors
    assert not warnings
