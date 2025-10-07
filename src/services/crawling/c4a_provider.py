"""Thin async wrappers around Crawl4AI primitives."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, cast

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.async_dispatcher import SemaphoreDispatcher
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
from crawl4ai.models import CrawlResult

from .c4a_presets import best_first_run_config, bfs_run_config


DispatcherType = MemoryAdaptiveDispatcher | SemaphoreDispatcher


async def crawl_page(
    url: str | Sequence[str],
    run_cfg: CrawlerRunConfig,
    browser_cfg: BrowserConfig,
    *,
    dispatcher: DispatcherType | None = None,
    crawler: AsyncWebCrawler | None = None,
) -> CrawlResult | list[CrawlResult]:
    """Crawl one or many URLs using Crawl4AI.

    Args:
        url: Target URL or sequence of URLs to crawl.
        run_cfg: Run configuration controlling extraction and crawl options.
        browser_cfg: Browser configuration used when creating a crawler.
        dispatcher: Optional dispatcher when coordinating multiple URLs.
        crawler: Optional crawler instance to reuse instead of owning one.

    Returns:
        A single :class:`~crawl4ai.models.CrawlResult` when ``url`` is a
        string; otherwise a list of results in the order produced by Crawl4AI.

    Raises:
        ValueError: If no URLs are provided.
    """

    urls = [url] if isinstance(url, str) else list(url)
    if not urls:
        raise ValueError("No URLs provided for crawl_page")

    async def _execute(active: AsyncWebCrawler) -> list[CrawlResult]:
        if dispatcher is not None:
            raw_many = await active.arun_many(
                urls=urls,
                config=run_cfg,
                dispatcher=dispatcher,
            )
            return await _coerce_results(raw_many)

        if len(urls) == 1:
            raw_single = await active.arun(url=urls[0], config=run_cfg)
            return await _coerce_results(raw_single)

        raw_many = await active.arun_many(urls=urls, config=run_cfg)
        return await _coerce_results(raw_many)

    if crawler is not None:
        results = await _execute(crawler)
    else:
        async with AsyncWebCrawler(config=browser_cfg) as owned:
            results = await _execute(owned)

    if isinstance(url, str):
        return results[0]
    return results


# The signature mirrors Crawl4AI BFS configuration knobs for parity with the SDK.
async def crawl_deep_bfs(  # pylint: disable=too-many-arguments
    url: str,
    depth: int,
    run_cfg: CrawlerRunConfig,
    browser_cfg: BrowserConfig,
    *,
    dispatcher: DispatcherType | None = None,
    crawler: AsyncWebCrawler | None = None,
) -> list[CrawlResult]:
    """Run a breadth-first deep crawl using Crawl4AI presets.

    Args:
        url: Seed URL for the crawl.
        depth: Maximum crawl depth.
        run_cfg: Base crawler run configuration to clone or reuse.
        browser_cfg: Browser configuration to employ when owning the crawler.
        dispatcher: Optional dispatcher for throttling and concurrency control.
        crawler: Optional crawler to reuse in cooperative contexts.

    Returns:
        A list of crawl results captured from the breadth-first traversal.
    """

    if not isinstance(run_cfg.deep_crawl_strategy, BFSDeepCrawlStrategy):
        cfg = bfs_run_config(depth, base_config=run_cfg)
    else:
        cfg = run_cfg.clone()

    results = await crawl_page(
        url,
        cfg,
        browser_cfg,
        dispatcher=dispatcher,
        crawler=crawler,
    )
    return results if isinstance(results, list) else [results]


async def crawl_best_first(
    url: str,
    keywords: Sequence[str],
    run_cfg: CrawlerRunConfig,
    browser_cfg: BrowserConfig,
    *,
    crawler: AsyncWebCrawler | None = None,
) -> list[CrawlResult] | AsyncIterator[CrawlResult]:
    """Run a best-first deep crawl prioritised by keyword relevance.

    Args:
        url: Seed URL to explore.
        keywords: Keywords powering the relevance scorer.
        run_cfg: Base crawler run configuration to clone or reuse.
        browser_cfg: Browser configuration to use when owning the crawler.
        crawler: Optional crawler to reuse; required for cooperative streaming.

    Returns:
        An async iterator of crawl results when the configuration enables
        streaming, otherwise a list of crawl results ordered by score.
    """

    if not isinstance(run_cfg.deep_crawl_strategy, BestFirstCrawlingStrategy):
        cfg = best_first_run_config(keywords, base_config=run_cfg)
    else:
        cfg = run_cfg.clone()

    if cfg.stream:
        if crawler is not None:
            stream = await crawler.arun(url=url, config=cfg)
            return cast(AsyncIterator[CrawlResult], stream)

        async def _stream() -> AsyncIterator[CrawlResult]:
            async with AsyncWebCrawler(config=browser_cfg) as owned:
                raw_stream = await owned.arun(url=url, config=cfg)
                stream = _ensure_async_iter(raw_stream)
                async for item in stream:
                    yield item

        return _stream()

    results = await crawl_page(url, cfg, browser_cfg, crawler=crawler)
    return results if isinstance(results, list) else [results]


async def _coerce_results(raw: Any) -> list[CrawlResult]:
    """Normalise any Crawl4AI return type into a list of results.

    Args:
        raw: Raw value returned from ``arun`` or ``arun_many``.

    Returns:
        A list of crawl results, preserving the original production order.
    """

    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, AsyncIterator) or hasattr(raw, "__aiter__"):
        iterator = _ensure_async_iter(raw)
        return [item async for item in iterator]
    if isinstance(raw, str | bytes):
        msg = "Unexpected Crawl4AI result type: string-like payload"
        raise TypeError(msg)
    if isinstance(raw, Iterable):
        return list(raw)
    return [raw]


def _ensure_async_iter(raw: Any) -> AsyncIterator[CrawlResult]:
    """Guarantee an asynchronous iterator for streaming responses.

    Args:
        raw: Value returned from a streaming Crawl4AI call.

    Returns:
        An async iterator yielding :class:`~crawl4ai.models.CrawlResult`.
    """

    if isinstance(raw, AsyncIterator):
        return raw
    if hasattr(raw, "__aiter__"):
        return cast(AsyncIterator[CrawlResult], raw)

    async def _generator() -> AsyncIterator[CrawlResult]:
        if isinstance(raw, list):
            for item in raw:
                yield cast(CrawlResult, item)  # pragma: no cover - defensive path
        else:
            yield cast(CrawlResult, raw)

    return _generator()


__all__ = [
    "crawl_page",
    "crawl_deep_bfs",
    "crawl_best_first",
]
