"""Lightweight scraping utilities for tier-0 retrieval."""

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
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError("Install trafilatura to use LightweightScraper.") from exc

try:
    from selectolax.parser import HTMLParser
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError("Install selectolax to use LightweightScraper.") from exc


class ScrapedContent(BaseModel):
    """Normalized content structure returned by the lightweight scraper."""

    url: str
    title: str = ""
    text: str
    links: list[dict[str, str]] = Field(default_factory=list)
    headings: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_time_ms: float = 0.0
    tier: int = 0
    success: bool = True


class ContentAnalysis(BaseModel):
    """Result describing whether the lightweight tier can process a URL."""

    can_handle: bool
    size_estimate: int = 0
    reasons: list[str] = Field(default_factory=list)


class LightweightScraper:
    """Tier-0 scraper leveraging httpx and selectolax."""

    def __init__(self, config: Any) -> None:
        browser_config = getattr(config, "browser_use", None)
        default_timeout = getattr(browser_config, "timeout", 15000)
        timeout_seconds = (
            default_timeout / 1000 if isinstance(default_timeout, (int, float)) else 15
        )
        self._timeout = max(timeout_seconds, 1)
        self._max_retries = getattr(browser_config, "max_retries", 3)
        self._content_threshold = getattr(browser_config, "min_content_length", 20)

    async def initialize(self) -> None:
        """Placeholder initialization to mirror richer scrapers."""

    async def cleanup(self) -> None:
        """Placeholder cleanup hook."""

    async def can_handle(self, url: str) -> ContentAnalysis:
        """Perform a cheap HEAD request to see if lightweight scraping applies."""

        reasons: list[str] = []
        size_estimate = 0
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.head(url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            reasons.append(f"HEAD request failed ({exc})")
            return ContentAnalysis(can_handle=False, reasons=reasons)

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            reasons.append("HTML content type detected")
        else:
            reasons.append(f"Unsupported content type: {content_type or 'unknown'}")

        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit():
            size_estimate = int(content_length)
            if size_estimate > 1_000_000:
                reasons.append("Large content size may require heavier automation")
        else:
            reasons.append("Content length unavailable; assuming lightweight tier")

        can_handle = "text/html" in content_type and size_estimate <= 1_000_000
        return ContentAnalysis(
            can_handle=can_handle,
            size_estimate=size_estimate,
            reasons=reasons,
        )

    async def scrape(
        self, url: str, *, timeout_ms: int = 10_000
    ) -> ScrapedContent | None:
        """Fetch and extract content from static pages."""

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=min(timeout_ms / 1000, self._timeout)
            ) as client:
                response = await client.get(url, follow_redirects=True)
        except httpx.TimeoutException:
            logger.debug("Timeout fetching %s", url)
            return None
        except httpx.HTTPError:  # pragma: no cover - network failure
            logger.debug("HTTP error fetching %s", url, exc_info=True)
            return None

        status_code = response.status_code
        if status_code >= 500:
            logger.warning(
                "Received HTTP %s while lightweight scraping %s; "
                "escalating to higher tier",
                status_code,
                url,
            )
            return None
        if status_code >= 400:
            logger.info(
                "Received HTTP %s while lightweight scraping %s; "
                "escalating to higher tier",
                status_code,
                url,
            )
            return None

        html = response.text

        text = self._extract_text(html, url)
        if len(text.strip()) < self._content_threshold:
            return None

        title = self._extract_title(html)
        links = self._extract_links(url, html)
        headings = self._extract_headings(html)
        elapsed = (time.perf_counter() - start) * 1000

        return ScrapedContent(
            success=True,
            url=str(response.url),
            title=title,
            text=text,
            links=links,
            headings=headings,
            metadata={"links": links, "status": response.status_code},
            extraction_time_ms=elapsed,
        )

    @staticmethod
    def _extract_title(html: str) -> str:
        parser = HTMLParser(html)
        node = parser.css_first("title")
        return node.text(strip=True) if node else ""

    @staticmethod
    def _extract_links(base_url: str, html: str) -> list[dict[str, str]]:
        parser = HTMLParser(html)
        links: list[dict[str, str]] = []
        for anchor in parser.css("a"):
            href = anchor.attributes.get("href")
            if not href:
                continue
            href = href.strip()
            if href.startswith("#") or href.startswith("javascript:"):
                continue
            abs_url = urljoin(base_url, href)
            text = (anchor.text() or "").strip()
            links.append({"url": abs_url, "text": text})
        return links

    @staticmethod
    def _extract_headings(html: str) -> list[dict[str, str]]:
        parser = HTMLParser(html)
        headings: list[dict[str, str]] = []
        for level in ("h1", "h2", "h3"):
            for heading in parser.css(level):
                text = heading.text(deep=True).strip()
                if text:
                    headings.append({"level": level, "text": text})
        return headings

    def _extract_text(self, html: str, url: str) -> str:
        """Extract readable text with a fallback when trafilatura fails."""

        try:
            text = trafilatura.extract(html, include_comments=False) or ""
        except Exception as exc:  # pragma: no cover - library quirks
            logger.warning(
                "Trafilatura extraction failed for %s: %s", url, exc, exc_info=True
            )
            return self._fallback_text(html)

        if text.strip():
            return text

        logger.debug("Trafilatura returned empty text for %s; using fallback", url)
        return self._fallback_text(html)

    @staticmethod
    def _fallback_text(html: str) -> str:
        parser = HTMLParser(html)
        body = parser.body
        if body is None:
            return ""
        return body.text(separator=" ", strip=True)
