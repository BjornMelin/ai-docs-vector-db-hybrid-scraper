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
    """Constructs a minimal namespace that mimics the production config.

    Args:
        openai_key: API key used for OpenAI-backed tools.
        firecrawl_key: API key for the Firecrawl crawling provider.
        qdrant_url: Qdrant endpoint URL required for vector storage.
        providers: Enabled provider list for model-backed tools.
        crawling_providers: Enabled crawling providers.

    Returns:
        SimpleNamespace: Minimal configuration namespace for tests.
    """
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
    """validate_configuration should raise when the OpenAI key is absent."""
    config = _build_config(openai_key=None)

    with (
        patch("src.unified_mcp_server.get_config", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
        pytest.raises(ValueError, match="OpenAI API key is required"),
    ):
        unified_mcp_server.validate_configuration()


def test_validate_configuration_requires_qdrant_url():
    """validate_configuration should reject missing Qdrant endpoints."""
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
    """validate_configuration should only warn when Firecrawl is optional."""
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
    """validate_configuration should pass configs with optional Firecrawl."""
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
    """_validate_streaming_config should record invalid port values as errors."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {"FASTMCP_TRANSPORT": "streamable-http", "FASTMCP_PORT": "not-a-number"},
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert any("Invalid port value" in error for error in errors)


def test_validate_streaming_config_handles_negative_buffer_size():
    """Negative buffer sizes should emit a performance warning."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_BUFFER_SIZE": "-1024",
        },
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert not errors
    assert any("Buffer size" in warning for warning in warnings)


def test_validate_streaming_config_handles_non_integer_buffer_size():
    """Non-integer buffer sizes should be captured as configuration errors."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_BUFFER_SIZE": "not-an-int",
        },
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert any("Invalid buffer size" in error for error in errors)
    assert not warnings


def test_validate_streaming_config_handles_invalid_max_response_size():
    """Non-positive max response sizes should be recorded as errors."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_MAX_RESPONSE_SIZE": "-2048",
        },
        clear=True,
    ):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert "Max response size must be positive" in errors


def test_validate_streaming_config_skips_non_streamable_transport():
    """Alternate transports should bypass streaming validation entirely."""
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
