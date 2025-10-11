"""Tests for MCP configuration management tools."""

# pylint: disable=duplicate-code

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.mcp_tools.tools import configuration as configuration_module


class DummyConfig:
    """Lightweight configuration stub compatible with the MCP tools."""

    def __init__(self) -> None:
        self.embedding_provider = SimpleNamespace(name="OPENAI")
        self.openai = SimpleNamespace(api_key=None)
        self.firecrawl = SimpleNamespace(api_key=None)
        self.crawl_provider = SimpleNamespace(name="FIRECRAWL")
        self.qdrant = SimpleNamespace(url="", enable_grouping=True)
        self.misc_setting = "value"

    def model_dump(self) -> dict[str, Any]:
        return {
            "openai": {"api_key": "secret-openai"},
            "qdrant": {"api_key": "secret-qdrant"},
            "misc_setting": self.misc_setting,
        }


@pytest.fixture
def mock_mcp():
    """Return a mock MCP instance that records registered tools."""

    api = MagicMock()
    registry: dict[str, Any] = {}

    def capture(func):
        registry[func.__name__] = func
        return func

    api.tool.return_value = capture
    api._registered = registry  # type: ignore[attr-defined]
    return api


@pytest.fixture
def dummy_config():
    """Return a dummy configuration object."""

    return DummyConfig()


@pytest.fixture
def patched_get_config(monkeypatch: pytest.MonkeyPatch, dummy_config: DummyConfig):
    """Patch ``get_settings`` to return the dummy configuration."""

    monkeypatch.setattr(
        configuration_module, "load_unified_config", lambda: dummy_config
    )


@pytest.fixture
def patched_grouping_probe(monkeypatch: pytest.MonkeyPatch):
    """Patch grouping probe helper to control support flag."""

    async def fake_probe(_client_manager):
        return False

    monkeypatch.setattr(configuration_module, "_probe_grouping_support", fake_probe)


class TestConfigurationTools:
    """Unit tests for configuration MCP tools."""

    @pytest.mark.asyncio
    async def test_get_config_returns_masked_values(
        self,
        mock_mcp: MagicMock,
        patched_get_config,
    ) -> None:
        """Global configuration export should mask sensitive keys."""

        configuration_module.register_tools(mock_mcp, vector_service=MagicMock())
        get_config_tool = mock_mcp._registered["get_settings"]

        result = await get_config_tool(ctx=None)

        config_payload = result["config"]
        assert config_payload["openai"]["api_key"] == "<redacted>"
        assert config_payload["qdrant"]["api_key"] == "<redacted>"
        assert config_payload["misc_setting"] == "value"

    @pytest.mark.asyncio
    async def test_get_config_specific_key(
        self,
        mock_mcp: MagicMock,
        patched_get_config,
        dummy_config: DummyConfig,
    ) -> None:
        """Requesting an explicit key should return sanitized data."""

        configuration_module.register_tools(mock_mcp, vector_service=MagicMock())
        get_config_tool = mock_mcp._registered["get_settings"]

        response = await get_config_tool(key="misc_setting", ctx=None)

        assert response == {"key": "misc_setting", "found": True, "value": "value"}

        missing = await get_config_tool(key="unknown", ctx=None)
        assert missing == {"key": "unknown", "found": False, "value": None}

    @pytest.mark.asyncio
    async def test_validate_config_flags_warnings(
        self,
        mock_mcp: MagicMock,
        patched_get_config,
        patched_grouping_probe,
    ) -> None:
        """Validation should surface provider and grouping warnings."""

        client_manager = MagicMock()
        configuration_module.register_tools(
            mock_mcp,
            vector_service=client_manager,
        )
        validate_tool = mock_mcp._registered["validate_config"]

        result = await validate_tool(ctx=None)

        assert result["valid"] is False
        assert "Qdrant URL is not configured" in result["issues"]
        assert any("OpenAI embeddings" in warning for warning in result["warnings"])
        assert any("Firecrawl provider" in warning for warning in result["warnings"])
        assert any("QueryPointGroups" in warning for warning in result["warnings"])
        assert result["grouping_supported"] is False
        assert result["populated_fields"] >= 1

    def test_tools_registered(self, mock_mcp: MagicMock, patched_get_config) -> None:
        """Registering configuration tools should expose two MCP endpoints."""

        configuration_module.register_tools(mock_mcp, vector_service=MagicMock())
        assert set(mock_mcp._registered.keys()) == {"get_settings", "validate_config"}
