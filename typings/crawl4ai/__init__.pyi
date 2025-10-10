"""Type hints for the `crawl4ai` package."""

from __future__ import annotations

from typing import Any

class CacheMode:
    BYPASS: str

class AsyncWebCrawler:
    def __init__(self, *, config: Any) -> None: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...
