"""Type hints for the `crawl4ai.services.crawling.c4a_presets` module."""

from __future__ import annotations

from typing import Any

class BrowserOptions:
    browser_type: str
    headless: bool

    def __init__(self, browser_type: str, headless: bool) -> None: ...

def preset_browser_config(options: BrowserOptions) -> Any: ...
def base_run_config(
    *,
    cache_mode: Any,
    page_timeout: int | float,
    strip_scripts: bool,
    strip_styles: bool,
) -> Any: ...
def memory_dispatcher(*, max_session_permit: int) -> Any: ...
