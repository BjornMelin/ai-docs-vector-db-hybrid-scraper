"""Tests for configuration validation in the unified MCP server."""

from __future__ import annotations

import asyncio
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


class TestInitializeMonitoringSystem:
    """Tests for initialize_monitoring_system function."""

    def test_returns_none_when_monitoring_disabled(self) -> None:
        """Should return None when monitoring.enabled is False."""
        settings = SimpleNamespace(
            monitoring=SimpleNamespace(enabled=False),
        )

        result = unified_mcp_server.initialize_monitoring_system(
            settings, qdrant_client=None, dragonfly_url=None
        )

        assert result is None

    def test_returns_manager_when_health_checks_disabled(self) -> None:
        """Should return base HealthCheckManager when health_checks disabled."""
        settings = SimpleNamespace(
            monitoring=SimpleNamespace(
                enabled=True,
                enable_health_checks=False,
            ),
        )

        with (
            patch.object(
                unified_mcp_server.HealthCheckConfig,
                "from_unified_config",
                return_value=SimpleNamespace(),
            ),
            patch.object(unified_mcp_server, "HealthCheckManager") as mock_manager_cls,
        ):
            mock_manager = object()
            mock_manager_cls.return_value = mock_manager

            result = unified_mcp_server.initialize_monitoring_system(
                settings, qdrant_client=None, dragonfly_url=None
            )

            assert result is mock_manager
            mock_manager_cls.assert_called_once()

    def test_calls_build_health_manager_when_fully_enabled(self) -> None:
        """Should call build_health_manager when both monitoring and checks enabled."""
        settings = SimpleNamespace(
            monitoring=SimpleNamespace(
                enabled=True,
                enable_health_checks=True,
            ),
        )
        mock_qdrant = object()
        mock_url = "redis://localhost:6379"

        with (
            patch.object(
                unified_mcp_server.HealthCheckConfig,
                "from_unified_config",
                return_value=SimpleNamespace(enabled=True),
            ),
            patch.object(
                unified_mcp_server, "build_health_manager", return_value=object()
            ) as mock_build,
        ):
            unified_mcp_server.initialize_monitoring_system(
                settings, qdrant_client=mock_qdrant, dragonfly_url=mock_url
            )

            mock_build.assert_called_once_with(
                settings,
                qdrant_client=mock_qdrant,
                dragonfly_url=mock_url,
            )


class TestNormalizeTransport:
    """Tests for _normalize_transport function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("stdio", "stdio"),
            ("http", "http"),
            ("sse", "sse"),
            ("streamable-http", "streamable-http"),
        ],
    )
    def test_returns_valid_transport_unchanged(
        self, input_value: str, expected: str
    ) -> None:
        """Should return valid transport values unchanged."""
        result = unified_mcp_server._normalize_transport(input_value)
        assert result == expected

    def test_returns_stdio_for_invalid_transport(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return stdio and log warning for invalid transport."""
        result = unified_mcp_server._normalize_transport("invalid-transport")

        assert result == "stdio"
        assert "Unsupported FASTMCP_TRANSPORT" in caplog.text


class TestGetIntEnv:
    """Tests for _get_int_env function."""

    def test_returns_default_when_env_not_set(self) -> None:
        """Should return default when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = unified_mcp_server._get_int_env("NONEXISTENT_VAR", 42)

        assert result == 42

    def test_returns_parsed_int_from_env(self) -> None:
        """Should parse and return integer from environment variable."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "123"}, clear=True):
            result = unified_mcp_server._get_int_env("TEST_INT_VAR", 0)

        assert result == 123

    def test_returns_default_on_invalid_int(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return default and log warning for non-integer values."""
        with patch.dict(os.environ, {"BAD_INT": "not-a-number"}, clear=True):
            result = unified_mcp_server._get_int_env("BAD_INT", 99)

        assert result == 99
        assert "Invalid integer value" in caplog.text


class TestSetupFastmcpMonitoring:
    """Tests for setup_fastmcp_monitoring function."""

    def test_skips_when_system_metrics_disabled(self) -> None:
        """Should not register manager when include_system_metrics is False."""
        server = object()
        config = SimpleNamespace(
            monitoring=SimpleNamespace(include_system_metrics=False)
        )
        health_manager = object()

        unified_mcp_server.setup_fastmcp_monitoring(server, config, health_manager)

        assert id(server) not in unified_mcp_server._MONITORING_STATE

    def test_registers_manager_when_system_metrics_enabled(self) -> None:
        """Should register health manager when include_system_metrics is True."""
        server = object()
        config = SimpleNamespace(
            monitoring=SimpleNamespace(include_system_metrics=True)
        )
        health_manager = object()

        try:
            unified_mcp_server.setup_fastmcp_monitoring(server, config, health_manager)

            assert unified_mcp_server._MONITORING_STATE[id(server)] is health_manager
        finally:
            # Clean up
            unified_mcp_server._MONITORING_STATE.pop(id(server), None)


class TestRunPeriodicHealthChecks:
    """Tests for run_periodic_health_checks function."""

    @pytest.mark.asyncio
    async def test_calls_check_all_repeatedly(self) -> None:
        """Should call health_manager.check_all in a loop."""
        from unittest.mock import AsyncMock

        mock_manager = AsyncMock()
        mock_manager.check_all = AsyncMock()

        call_count = 0

        async def counting_check_all():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError()

        mock_manager.check_all = counting_check_all

        with pytest.raises(asyncio.CancelledError):
            await unified_mcp_server.run_periodic_health_checks(
                mock_manager, interval_seconds=0.001
            )

        assert call_count == 3
