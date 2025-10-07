"""
Tier 0 lightweight scraper.

Fast HTTP fetch with httpx and robust text extraction via Trafilatura.
Link parsing with selectolax for minimal overhead.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

try:
    import trafilatura
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("Install trafilatura to use LightweightScraper.") from exc

try:
    from selectolax.parser import HTMLParser
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("Install selectolax to use LightweightScraper.") from exc


class ScrapedContent(BaseModel):
    """Normalized content structure."""

    url: str
    title: str = ""
    text: str
    links: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_time_ms: float = 0.0
    tier: int = 0
    success: bool = True


class LightweightScraper:
    """High-performance static scraper."""

    def __init__(self, config: Any) -> None:
        self._timeout = getattr(getattr(config, "http", object()), "timeout", 15)

    async def scrape(self, url: str, *, timeout_ms: int = 10000) -> dict[str, Any]:
        """Fetch and extract content from static pages."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=min(timeout_ms / 1000, self._timeout)
            ) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text

            text = trafilatura.extract(html, include_comments=False) or ""
            title = self._extract_title(html)
            links = self._extract_links(url, html)
            elapsed = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "url": str(resp.url),
                "title": title,
                "content": text,
                "html": html,
                "metadata": {"links": links, "status": resp.status_code},
                "provider": "lightweight",
                "extraction_time_ms": elapsed,
            }
        except httpx.HTTPError as err:
            logger.debug("HTTP error for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "lightweight",
            }

    @staticmethod
    def _extract_title(html: str) -> str:
        parser = HTMLParser(html)
        node = parser.css_first("title")
        return node.text(strip=True) if node else ""

    @staticmethod
    def _extract_links(base_url: str, html: str) -> list[dict[str, str]]:
        parser = HTMLParser(html)
        out: list[dict[str, str]] = []
        for a in parser.css("a"):
            href = a.attributes.get("href")
            if not href:
                continue
            href = href.strip()
            if href.startswith("#") or href.startswith("javascript:"):
                continue
            abs_url = urljoin(base_url, href)
            text = (a.text() or "").strip()
            out.append({"url": abs_url, "text": text})
        return out
