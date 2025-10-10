"""Type hints for the `crawl4ai.services.crawling` module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

async def crawl_page(
    target: str | Sequence[str],
    run_config: Any,
    browser_config: Any,
    *,
    crawler: Any | None = ...,
    dispatcher: Any | None = ...,
) -> Any: ...
