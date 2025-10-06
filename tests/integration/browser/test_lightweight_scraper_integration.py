"""Browser integration tests using the lightweight scraper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import Config
from src.services.browser.lightweight_scraper import LightweightScraper


@pytest.mark.asyncio
@pytest.mark.browser
async def test_lightweight_scraper_handles_static_page(integration_server: str) -> None:
    """Lightweight scraper should fetch simple HTML from the integration server."""

    config = Config()
    config_stub = SimpleNamespace(
        content_threshold=20,
        lightweight_timeout=5.0,
        max_retries=0,
    )
    object.__setattr__(config, "browser_automation", config_stub)

    scraper = LightweightScraper(config)
    await scraper.initialize()
    try:
        result = await scraper.scrape(f"{integration_server}/static")
    finally:
        await scraper.cleanup()

    assert result is not None
    assert result.success is True
    assert "Integration Static" in result.text
    assert result.tier == 0
    assert result.url.endswith("/static")


@pytest.mark.asyncio
@pytest.mark.browser
async def test_lightweight_scraper_escalates_on_forbidden(
    integration_server: str,
) -> None:
    """403 responses should trigger escalation to higher tiers."""

    config = Config()
    config_stub = SimpleNamespace(
        content_threshold=20,
        lightweight_timeout=5.0,
        max_retries=0,
    )
    object.__setattr__(config, "browser_automation", config_stub)

    scraper = LightweightScraper(config)
    await scraper.initialize()
    try:
        result = await scraper.scrape(f"{integration_server}/forbidden")
    finally:
        await scraper.cleanup()

    assert result is None
