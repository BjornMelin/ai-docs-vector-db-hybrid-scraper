"""Tests for configuration validation in the unified MCP server."""

from __future__ import annotations

import os
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import unified_mcp_server


def test_validate_configuration_requires_openai_key(
    build_unified_mcp_config: Callable[..., SimpleNamespace],
) -> None:
    """`validate_configuration` should raise when the OpenAI key is absent."""
    config = build_unified_mcp_config(openai_key=None)
    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
        pytest.raises(ValueError, match="OpenAI API key is required"),
    ):
        unified_mcp_server.validate_configuration()


def test_validate_configuration_requires_qdrant_url(
    build_unified_mcp_config: Callable[..., SimpleNamespace],
) -> None:
    """`validate_configuration` should reject missing Qdrant endpoints."""
    config = build_unified_mcp_config(
        qdrant_url=None,
        providers=["fastembed"],
        crawling_providers=[],
    )

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
        pytest.raises(ValueError, match="Qdrant URL is required"),
    ):
        unified_mcp_server.validate_configuration()


def test_validate_configuration_warns_on_missing_firecrawl(
    build_unified_mcp_config: Callable[..., SimpleNamespace],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`validate_configuration` should only warn when Firecrawl is optional."""
    config = build_unified_mcp_config(
        firecrawl_key=None,
        providers=["fastembed"],
        crawling_providers=["firecrawl"],
    )

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config"),
    ):
        unified_mcp_server.validate_configuration()

    assert "Firecrawl API key not set" in caplog.text


def test_validate_configuration_passes_for_valid_config(
    build_unified_mcp_config: Callable[..., SimpleNamespace],
) -> None:
    """`validate_configuration` should pass configs with optional Firecrawl."""
    config = build_unified_mcp_config(
        firecrawl_key="",
        providers=["fastembed"],
        crawling_providers=["other"],
    )

    with (
        patch("src.unified_mcp_server.get_settings", return_value=config),
        patch("src.unified_mcp_server._validate_streaming_config") as mock_validate,
    ):
        unified_mcp_server.validate_configuration()

    mock_validate.assert_called_once()


@pytest.mark.parametrize(
    "env_vars,expected_error_substrings,expected_warning_substrings",
    [
        pytest.param(
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_PORT": "not-a-number",
            },
            ["Invalid port value"],
            [],
            id="invalid-port",
        ),
        pytest.param(
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_BUFFER_SIZE": "-1024",
            },
            [],
            ["Buffer size"],
            id="negative-buffer",
        ),
        pytest.param(
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_BUFFER_SIZE": "not-an-int",
            },
            ["Invalid buffer size"],
            [],
            id="non-integer-buffer",
        ),
        pytest.param(
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_MAX_RESPONSE_SIZE": "-2048",
            },
            ["Max response size must be positive"],
            [],
            id="invalid-max-response",
        ),
    ],
)
def test_validate_streaming_config_records_issues(
    env_vars: dict[str, str],
    expected_error_substrings: list[str],
    expected_warning_substrings: list[str],
) -> None:
    """Streaming configuration issues should be recorded as errors or warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(os.environ, env_vars, clear=True):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    for substring in expected_error_substrings:
        assert any(substring in error for error in errors)
    for substring in expected_warning_substrings:
        assert any(substring in warning for warning in warnings)

    if not expected_error_substrings:
        assert not errors
    if not expected_warning_substrings:
        assert not warnings


def test_validate_streaming_config_skips_non_streamable_transport() -> None:
    """Alternate transports should bypass streaming validation entirely."""
    errors: list[str] = []
    warnings: list[str] = []

    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "stdio"}, clear=True):
        unified_mcp_server._validate_streaming_config(errors, warnings)

    assert not errors
    assert not warnings
