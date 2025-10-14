"""Config schema tests covering browser automation aggregation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.browser import BrowserAutomationConfig


def test_browser_automation_defaults() -> None:
    """Validate default browser automation configuration values."""
    config = BrowserAutomationConfig()

    config_dump = config.model_dump()

    assert config_dump["lightweight"]["timeout_seconds"] == 8.0
    assert config_dump["crawl4ai"]["headless"] is True
    assert config_dump["playwright"]["browser"] == "chromium"
    assert config_dump["browser_use"]["llm_provider"] == "openai"
    assert list(config_dump["firecrawl"]["default_formats"]) == ["markdown", "html"]
    assert config_dump["router"]["per_attempt_cap_ms"] == 20000


def test_browser_automation_custom_payload() -> None:
    """Ensure overrides apply when loading custom payloads."""
    payload = {
        "lightweight": {"timeout_seconds": 4.0, "allow_redirects": False},
        "crawl4ai": {"headless": False, "browser_type": "firefox"},
        "browser_use": {"llm_provider": "gemini", "model": "gemini-1.5"},
        "firecrawl": {"api_key": "fc-test-key", "default_formats": ["markdown"]},
        "router": {
            "rate_limits": {
                "lightweight": {"max_requests": 5, "period_seconds": 2.0},
                "firecrawl": {"max_requests": 2, "period_seconds": 1.5},
            }
        },
    }

    config = BrowserAutomationConfig.model_validate(payload)

    config_dump = config.model_dump()

    assert config_dump["lightweight"]["timeout_seconds"] == 4.0
    assert config_dump["crawl4ai"]["browser_type"] == "firefox"
    assert config_dump["browser_use"]["llm_provider"] == "gemini"
    assert config_dump["firecrawl"]["api_key"] == "fc-test-key"
    assert config_dump["router"]["rate_limits"]["firecrawl"]["max_requests"] == 2


@pytest.mark.parametrize(
    "browser_use",
    [
        {"llm_provider": "unknown"},
        {"llm_provider": "openai", "timeout_ms": 0},
    ],
)
def test_browser_automation_invalid(browser_use: dict[str, str]) -> None:
    """Reject invalid browser-use configurations."""
    payload = {"browser_use": browser_use}
    with pytest.raises(ValidationError):
        BrowserAutomationConfig.model_validate(payload)
