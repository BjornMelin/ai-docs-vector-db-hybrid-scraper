"""Browser integration tests using the lightweight scraper."""

# pylint: disable=duplicate-code

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest


if "src.services.crawling" not in sys.modules:
    crawling_module = types.ModuleType("src.services.crawling")

    async def crawl_page(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    crawling_module.crawl_page = crawl_page  # type: ignore[attr-defined]
    sys.modules["src.services.crawling"] = crawling_module

if "src.services.crawling.c4a_presets" not in sys.modules:
    presets_module = types.ModuleType("src.services.crawling.c4a_presets")

    def _noop(*_args: Any, **_kwargs: Any) -> Any:
        return types.SimpleNamespace(clone=lambda **__: types.SimpleNamespace())

    presets_module.BrowserOptions = object  # type: ignore[attr-defined]
    presets_module.base_run_config = _noop  # type: ignore[attr-defined]
    presets_module.memory_dispatcher = _noop  # type: ignore[attr-defined]
    presets_module.preset_browser_config = _noop  # type: ignore[attr-defined]
    sys.modules["src.services.crawling.c4a_presets"] = presets_module

from src.config import Settings
from src.config.models import Environment
from src.services.browser.lightweight_scraper import LightweightScraper


@pytest.mark.asyncio
@pytest.mark.browser
async def test_lightweight_scraper_handles_static_page(
    integration_server: str,
    tmp_path: Path,
) -> None:
    """Lightweight scraper should fetch simple HTML from the integration server."""

    base = tmp_path / "browser_cfg"
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
        result = await scraper.scrape(f"{integration_server}/forbidden")
    finally:
        await scraper.cleanup()

    assert result is None
