"""Type hints for the `crawl4ai.models` module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

class MarkdownPayload:
    raw_markdown: str | None
    fit_markdown: str | None

class CrawlResult:
    success: bool
    url: str
    html: str | None
    markdown: MarkdownPayload | None
    metadata: Mapping[str, Any] | None
    extracted_content: Any
    links: Mapping[str, Any] | None
    media: Mapping[str, Any] | None
