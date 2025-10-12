"""Config schema tests covering browser automation aggregation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.browser.config import BrowserAutomationConfig


def test_browser_automation_defaults() -> None:
    config = BrowserAutomationConfig()

    assert config.lightweight.timeout_seconds == 8.0
    assert config.crawl4ai.headless is True
    assert config.playwright.browser == "chromium"
    assert config.browser_use.llm_provider == "openai"
    assert list(config.firecrawl.default_formats) == ["markdown", "html"]
    assert config.router.per_attempt_cap_ms == 20000


def test_browser_automation_custom_payload() -> None:
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

    assert config.lightweight.timeout_seconds == 4.0
    assert config.crawl4ai.browser_type == "firefox"
    assert config.browser_use.llm_provider == "gemini"
    assert config.firecrawl.api_key == "fc-test-key"
    assert config.router.rate_limits["firecrawl"].max_requests == 2


@pytest.mark.parametrize(
    "browser_use",
    [
        {"llm_provider": "unknown"},
        {"llm_provider": "openai", "model": ""},
    ],
)
def test_browser_automation_invalid(browser_use: dict[str, str]) -> None:
    payload = {"browser_use": browser_use}
    with pytest.raises(ValidationError):
        BrowserAutomationConfig.model_validate(payload)
