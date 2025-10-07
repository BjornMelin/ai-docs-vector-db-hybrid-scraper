"""Browser automation services with a single router and unified manager."""

from .browser_use_adapter import BrowserUseAdapter
from .crawl4ai_adapter import Crawl4AIAdapter
from .firecrawl_adapter import FirecrawlAdapter, FirecrawlAdapterConfig
from .lightweight_scraper import LightweightScraper
from .playwright_adapter import PlaywrightAdapter
from .router import AutomationRouter
from .unified_manager import UnifiedBrowserManager, UnifiedScrapingRequest


__all__ = [
    "AutomationRouter",
    "UnifiedBrowserManager",
    "UnifiedScrapingRequest",
    "LightweightScraper",
    "Crawl4AIAdapter",
    "PlaywrightAdapter",
    "BrowserUseAdapter",
    "FirecrawlAdapter",
    "FirecrawlAdapterConfig",
]
