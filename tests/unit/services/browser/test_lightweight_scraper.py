"""Unit tests for the browser lightweight scraper."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from src.config import Settings
from src.config.models import Environment
from src.services.browser.lightweight_scraper import LightweightScraper


@pytest.mark.asyncio
async def test_lightweight_scraper_returns_none_on_client_error(
    tmp_path: Path,
) -> None:
    """Ensure client errors escalate to higher tiers without raising."""

    base = tmp_path / "lightweight_scraper_client_error"
    config = Settings.model_validate(
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
        request = httpx.Request("GET", "https://example.com/forbidden")
        with respx.mock(assert_all_mocked=False) as mock:
            mock.get("https://example.com/forbidden").mock(
                return_value=httpx.Response(403, request=request)
            )

            result = await scraper.scrape("https://example.com/forbidden")
    finally:
        await scraper.cleanup()

    assert result is None
