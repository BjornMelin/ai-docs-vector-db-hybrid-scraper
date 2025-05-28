"""Browser automation services with intelligent routing."""

from .automation_router import AutomationRouter
from .crawl4ai_adapter import Crawl4AIAdapter
from .playwright_adapter import PlaywrightAdapter
from .stagehand_adapter import StagehandAdapter

__all__ = [
    "AutomationRouter",
    "Crawl4AIAdapter",
    "PlaywrightAdapter",
    "StagehandAdapter",
]
