"""Browser automation services with optional adapters."""

from .browser_use_adapter import BrowserUseAdapter
from .lightweight_scraper import LightweightScraper
from .playwright_adapter import PlaywrightAdapter


__all__: list[str] = [
    "LightweightScraper",
    "BrowserUseAdapter",
    "PlaywrightAdapter",
]

try:
    from .router import AutomationRouter
    from .unified_manager import UnifiedBrowserManager, UnifiedScrapingRequest

    __all__ += ["AutomationRouter", "UnifiedBrowserManager", "UnifiedScrapingRequest"]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AutomationRouter = None  # type: ignore
    UnifiedBrowserManager = None  # type: ignore
    UnifiedScrapingRequest = None  # type: ignore


try:  # Optional adapters
    from .crawl4ai_adapter import Crawl4AIAdapter

    __all__.append("Crawl4AIAdapter")
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    Crawl4AIAdapter = None  # type: ignore

try:
    from .firecrawl_adapter import FirecrawlAdapter, FirecrawlAdapterConfig

    __all__ += ["FirecrawlAdapter", "FirecrawlAdapterConfig"]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    FirecrawlAdapter = None  # type: ignore
    FirecrawlAdapterConfig = None  # type: ignore
