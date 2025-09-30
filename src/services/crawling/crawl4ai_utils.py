"""Shared utilities for the Crawl4AI provider."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse


DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_MEMORY_THRESHOLD = 70.0
DEFAULT_DISPATCHER_INTERVAL = 1.0
DEFAULT_MAX_SESSION_PERMIT = 10
DEFAULT_STREAMING_ENABLED = True
DEFAULT_RATE_LIMIT_BASE_DELAY = (1.0, 2.0)
DEFAULT_RATE_LIMIT_MAX_DELAY = 30.0
DEFAULT_RATE_LIMIT_RETRIES = 2


@dataclass(slots=True)
class Crawl4AIScrapeOptions:
    """Options accepted by the Crawl4AI provider scrape entrypoint."""

    extraction_type: str = "markdown"
    wait_for: str | None = None
    js_code: str | None = None
    stream: bool | None = None


@dataclass(slots=True)
class CrawlQueueState:
    """Mutable crawl state used for site crawling."""

    base_domain: str
    max_pages: int
    max_visited: int
    pending: deque[str]
    visited_order: deque[str]
    visited_lookup: set[str]
    pages: list[dict[str, Any]]

    def take_batch(self, limit: int) -> list[str]:
        """Return the next batch of URLs to crawl."""

        batch: list[str] = []
        while self.pending and len(batch) < limit:
            candidate = self.pending.popleft()
            if candidate in self.visited_lookup:
                continue
            self.visited_lookup.add(candidate)
            self.visited_order.append(candidate)
            batch.append(candidate)
            self._trim_visited()
        return batch

    def queue_link(self, link_url: str) -> None:
        """Queue a new candidate URL if it is not already tracked."""

        if link_url not in self.visited_lookup and link_url not in self.pending:
            self.pending.append(link_url)

    def _trim_visited(self) -> None:
        """Keep the visited cache bounded for long crawls."""

        if len(self.visited_order) <= self.max_visited:
            return

        keep_count = int(self.max_visited * 0.8)
        while len(self.visited_order) > keep_count:
            removed = self.visited_order.popleft()
            self.visited_lookup.discard(removed)


def config_value(config: Any, attribute: str, default: Any) -> Any:
    """Fetch an attribute from the config object with a fallback."""

    return cast(Any, getattr(config, attribute, default))


def resolve_stream_flag(override: bool | None, default: bool) -> bool:
    """Resolve stream override against the configured default."""

    return override if override is not None else default


async def normalize_results(results: object) -> list[object]:
    """Normalize async crawl responses into a list."""

    if isinstance(results, AsyncGenerator):
        return [item async for item in results]
    if hasattr(results, "__iter__") and not isinstance(results, list):
        return list(cast(Iterable[object], results))
    return [results]


def create_queue_state(start_url: str, max_pages: int) -> CrawlQueueState:
    """Create crawl state for a site crawl."""

    return CrawlQueueState(
        base_domain=urlparse(start_url).netloc,
        max_pages=max_pages,
        max_visited=max(max_pages * 3, 1000),
        pending=deque([start_url]),
        visited_order=deque(),
        visited_lookup=set(),
        pages=[],
    )


def should_enqueue_link(state: CrawlQueueState, link_url: str) -> bool:
    """Return True when a link is eligible to be crawled."""

    if not link_url or not link_url.startswith("http"):
        return False
    return urlparse(link_url).netloc == state.base_domain


def process_crawl_results(
    state: CrawlQueueState,
    batch_results: Iterable[dict[str, Any]],
) -> None:
    """Apply crawl results to the queue state."""

    for result in batch_results:
        if not result.get("success"):
            continue

        state.pages.append(
            {
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "html": result.get("html", ""),
                "metadata": result.get("metadata", {}),
                "title": result.get("title", ""),
            }
        )

        for link in cast(Iterable[dict[str, Any]], result.get("links") or []):
            link_url = link.get("href", "")
            if should_enqueue_link(state, link_url):
                state.queue_link(link_url)


def build_crawl_error_context(
    state: CrawlQueueState, starting_url: str
) -> dict[str, object]:
    """Construct context metadata for crawl failures."""

    return {
        "starting_url": starting_url,
        "pages_crawled": len(state.pages),
        "urls_visited": len(state.visited_lookup),
        "urls_remaining": len(state.pending),
        "max_pages_target": state.max_pages,
    }


def classify_scrape_failure(url: str, message: str, logger: logging.Logger) -> str:
    """Return user-friendly error messages for scraping failures."""

    lower_message = message.lower()
    if "rate limit" in lower_message:
        logger.warning("Crawl4AI rate limit hit for %s", url)
        return "Rate limit exceeded. Please try again later."
    if "invalid api key" in lower_message or "unauthorized" in lower_message:
        logger.error("Invalid Crawl4AI API key for %s", url)
        return "Invalid API key. Please check your Crawl4AI configuration."
    if "timeout" in lower_message:
        logger.warning("Timeout while scraping %s", url)
        return "Request timed out. The page may be too large or slow to load."
    if "not found" in lower_message or "404" in lower_message:
        logger.info("Page not found for %s", url)
        return "Page not found (404)."
    return "Scraping failed"
