import typing


"""Browser automation services with intelligent routing."""

from .automation_router import AutomationRouter
from .browser_router import EnhancedAutomationRouter
from .browser_use_adapter import BrowserUseAdapter
from .crawl4ai_adapter import Crawl4AIAdapter
from .lightweight_scraper import LightweightScraper
from .playwright_adapter import PlaywrightAdapter
from .unified_manager import UnifiedBrowserManager


__all__ = [
    "AutomationRouter",
    "BrowserUseAdapter",
    "Crawl4AIAdapter",
    "EnhancedAutomationRouter",
    "LightweightScraper",
    "PlaywrightAdapter",
    "UnifiedBrowserManager",
]
