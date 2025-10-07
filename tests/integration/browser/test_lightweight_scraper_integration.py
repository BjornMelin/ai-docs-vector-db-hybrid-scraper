"""Browser integration tests using the lightweight scraper."""

# pylint: disable=duplicate-code

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config, Environment
from src.services.browser.lightweight_scraper import LightweightScraper


@pytest.mark.asyncio
@pytest.mark.browser
async def test_lightweight_scraper_handles_static_page(
    integration_server: str,
    tmp_path: Path,
) -> None:
    """Lightweight scraper should fetch simple HTML from the integration server."""

    base = tmp_path / "browser_cfg"
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "data_dir": base / "data",
            "cache_dir": base / "cache",
            "logs_dir": base / "logs",
        }
    )

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
    tmp_path: Path,
) -> None:
    """403 responses should trigger escalation to higher tiers."""

    base = tmp_path / "browser_cfg_forbidden"
    config = Config.model_validate(
        {
            "environment": Environment.TESTING,
            "data_dir": base / "data",
            "cache_dir": base / "cache",
            "logs_dir": base / "logs",
        }
    )

    scraper = LightweightScraper(config)
    await scraper.initialize()
    try:
        result = await scraper.scrape(f"{integration_server}/forbidden")
    finally:
        await scraper.cleanup()

    assert result is None
