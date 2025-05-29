"""Browser automation services with intelligent routing."""

from .automation_router import AutomationRouter
from .browser_use_adapter import BrowserUseAdapter
from .crawl4ai_adapter import Crawl4AIAdapter
from .playwright_adapter import PlaywrightAdapter

__all__ = [
    "AutomationRouter",
    "BrowserUseAdapter",
    "Crawl4AIAdapter",
    "PlaywrightAdapter",
]
