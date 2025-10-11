"""Crawl4AI adapter providing service-layer entry points."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.models import CrawlResult


try:  # pragma: no cover - optional dependency
    from playwright.async_api import Error as PlaywrightError
except ImportError:  # pragma: no cover - Playwright not installed in tests
    PlaywrightError = RuntimeError

from src.config.models import Crawl4AIConfig
from src.services.base import BaseService
from src.services.crawling import crawl_page
from src.services.crawling.c4a_presets import (  # type: ignore[import]
    BrowserOptions,
    base_run_config,
    memory_dispatcher,
    preset_browser_config,
)
from src.services.errors import CrawlServiceError


PlaywrightException = cast(type[Exception], PlaywrightError)


logger = logging.getLogger(__name__)

RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TimeoutError,
    OSError,
    PlaywrightException,
    CrawlServiceError,
)


@dataclass(slots=True)
class Metrics:
    """Aggregated adapter metrics tracked during crawls."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return metrics as a JSON-serialisable mapping."""

        avg_time = (
            self.total_time_ms / self.total_requests if self.total_requests else 0.0
        )
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time_ms": avg_time,
            "success_rate": success_rate,
        }


class Crawl4AIAdapter(BaseService):
    """Thin adapter around the Crawl4AI provider."""

    def __init__(self, config: Crawl4AIConfig):
        super().__init__(None)
        self.config = config
        browser_options = BrowserOptions(
            browser_type=config.browser_type,
            headless=config.headless,
        )
        self._browser_config = preset_browser_config(browser_options)
        self._run_config_template = base_run_config(
            cache_mode=CacheMode.BYPASS,
            page_timeout=config.page_timeout,
            strip_scripts=config.remove_scripts,
            strip_styles=config.remove_styles,
        )
        self._dispatcher = memory_dispatcher(
            max_session_permit=config.max_concurrent_crawls,
        )
        self._metrics = Metrics()
        self._crawler: AsyncWebCrawler | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Start a shared Crawl4AI crawler instance.

        Raises:
            CrawlServiceError: If the crawler cannot be started.
        """

        if self._initialized:
            return
        try:
            self._crawler = AsyncWebCrawler(config=self._browser_config)
            await self._crawler.start()
            self._mark_initialized()
            logger.info("Crawl4AI adapter initialized with shared crawler")
        # pylint: disable=catching-non-exception
        except RECOVERABLE_ERRORS as exc:  # pragma: no cover - defensive path
            self._crawler = None
            msg = "Failed to initialize Crawl4AI adapter"
            raise CrawlServiceError(msg) from exc

    async def cleanup(self) -> None:
        """Shutdown crawler resources and reset adapter state."""

        if self._crawler:
            try:
                await self._crawler.close()
            except (OSError, AttributeError, ConnectionError, ImportError):
                logger.exception("Error cleaning up Crawl4AI adapter")
            finally:
                self._crawler = None
        self._mark_uninitialized()

    async def scrape(
        self,
        url: str,
        wait_for_selector: str | None = None,
        js_code: str | None = None,
        _timeout: int = 30000,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Scrape a single URL using the shared crawler.

        Args:
            url: Target URL to fetch.
            wait_for_selector: Optional selector to await before extraction.
            js_code: Optional JavaScript to execute prior to extraction.
            _timeout: Ignored legacy parameter kept for interface parity.

        Returns:
            A payload containing crawl outcome, markdown, and metadata.

        Raises:
            CrawlServiceError: If the adapter is not initialized.
        """

        if not self._initialized or not self._crawler:
            msg = "Adapter not initialized"
            raise CrawlServiceError(msg)

        started = perf_counter()
        run_cfg = self._run_config_template.clone(
            wait_for=wait_for_selector,
            js_code=js_code,
            stream=False,
        )

        try:
            raw_result = await crawl_page(
                url,
                run_cfg,
                self._browser_config,
                crawler=self._crawler,
            )
            if isinstance(raw_result, list):
                if not raw_result:
                    msg = "Crawl returned no results"
                    raise CrawlServiceError(msg)
                crawl_result = raw_result[0]
            else:
                crawl_result = raw_result
            payload = self._build_success_payload(crawl_result)
            if payload.get("success", False):
                self._metrics.successful_requests += 1
            else:
                self._metrics.failed_requests += 1
            return payload
        # pylint: disable=catching-non-exception
        except RECOVERABLE_ERRORS as exc:  # pragma: no cover - defensive path
            logger.exception("Crawl4AI adapter error for %s", url)
            self._metrics.failed_requests += 1
            return {
                "success": False,
                "url": url,
                "error": str(exc),
                "content": "",
                "metadata": {
                    "extraction_type": "crawl4ai",
                    "processing_time_ms": (perf_counter() - started) * 1000,
                },
            }
        finally:
            duration_ms = (perf_counter() - started) * 1000
            self._metrics.total_requests += 1
            self._metrics.total_time_ms += duration_ms

    async def crawl_bulk(
        self,
        urls: list[str],
        extraction_type: str = "crawl4ai_bulk",
    ) -> list[dict[str, Any]]:
        """Crawl multiple URLs concurrently using the shared dispatcher.

        Args:
            urls: URLs to fetch.
            extraction_type: Marker injected into metadata for traceability.

        Returns:
            A list of payloads mirroring :meth:`scrape` outputs.

        Raises:
            CrawlServiceError: If the adapter is not initialized.
        """

        if not self._initialized or not self._crawler:
            msg = "Adapter not initialized"
            raise CrawlServiceError(msg)

        if not urls:
            return []

        run_cfg = self._run_config_template.clone(stream=False)
        try:
            raw_results = await crawl_page(
                urls,
                run_cfg,
                self._browser_config,
                dispatcher=self._dispatcher,
                crawler=self._crawler,
            )
        # pylint: disable=catching-non-exception
        except RECOVERABLE_ERRORS as exc:  # pragma: no cover - defensive path
            logger.exception("Bulk crawl failed for %d URLs", len(urls))
            failure_payload = {
                "success": False,
                "error": str(exc),
                "content": "",
                "metadata": {"extraction_type": extraction_type},
            }
            self._metrics.failed_requests += len(urls)
            self._metrics.total_requests += len(urls)
            return [
                {
                    **failure_payload,
                    "url": url,
                }
                for url in urls
            ]

        results = raw_results if isinstance(raw_results, list) else [raw_results]
        normalized: list[dict[str, Any]] = []
        for item in results:
            if getattr(item, "success", False):
                payload = self._build_success_payload(
                    item, extraction_type=extraction_type
                )
                normalized.append(payload)
                self._metrics.successful_requests += 1
            else:
                normalized.append(
                    {
                        "success": False,
                        "url": getattr(item, "url", ""),
                        "error": getattr(item, "error_message", "Unknown error"),
                        "content": "",
                        "metadata": {"extraction_type": extraction_type},
                    }
                )
                self._metrics.failed_requests += 1

        self._metrics.total_requests += len(results)
        return normalized

    def get_capabilities(self) -> dict[str, Any]:
        """Describe adapter capabilities for discovery and UI surfaces."""

        return {
            "name": "crawl4ai",
            "description": (
                "Async Playwright crawler with Fit Markdown, BFS and best-first "
                "strategies."
            ),
            "advantages": [
                "Generates raw + Fit Markdown",
                "Supports BFS deep crawl presets",
                "Keyword best-first streaming",
                "Memory adaptive dispatcher for batches",
                "Session-aware JavaScript execution",
            ],
            "limitations": [
                "Limited AI-driven interactions",
                "Requires Playwright runtime",
            ],
            "best_for": [
                "Documentation sites",
                "Blog posts",
                "API references",
                "Static knowledge bases",
            ],
            "performance": {
                "concurrency": f"{self.config.max_concurrent_crawls} sessions",
                "cache_mode": "BYPASS by default",
            },
            "javascript_support": "custom scripts + wait_for",
            "dynamic_content": "moderate",
            "authentication": False,
            "cost": 0,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform a lightweight scrape to validate the adapter."""

        if not self._initialized:
            return {
                "healthy": False,
                "status": "not_initialized",
                "message": "Adapter not initialized",
            }
        try:
            test_url = "https://httpbin.org/html"
            started = perf_counter()
            result = await asyncio.wait_for(self.scrape(test_url), timeout=10.0)
            response_time = (perf_counter() - started) * 1000
            return {
                "healthy": result.get("success", False),
                "status": "operational" if result.get("success") else "degraded",
                "message": result.get("error", "Health check passed"),
                "response_time_ms": response_time,
                "test_url": test_url,
                "capabilities": self.get_capabilities(),
            }
        except TimeoutError:
            return {
                "healthy": False,
                "status": "timeout",
                "message": "Health check timed out",
                "response_time_ms": 10000,
            }
        # pylint: disable=catching-non-exception
        except RECOVERABLE_ERRORS as exc:  # pragma: no cover - defensive path
            return {
                "healthy": False,
                "status": "error",
                "message": str(exc),
            }

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Expose adapter performance counters for observability surfaces."""

        return self._metrics.as_dict()

    def _build_success_payload(
        self,
        result: CrawlResult,
        *,
        extraction_type: str = "crawl4ai",
    ) -> dict[str, Any]:
        """Create a consistent success payload from a Crawl4AI result.

        Args:
            result: Crawl result returned by Crawl4AI.
            extraction_type: Marker recorded in metadata for traceability.

        Returns:
            Normalised dictionary used by upstream services.
        """

        markdown_raw = ""
        markdown_fit = ""
        if getattr(result, "markdown", None):
            markdown_raw = getattr(result.markdown, "raw_markdown", "") or ""
            markdown_fit = getattr(result.markdown, "fit_markdown", "") or ""
        content = markdown_fit or markdown_raw or result.html or ""
        metadata = {
            **(result.metadata or {}),
            "extraction_type": extraction_type,
            "word_count": len(content.split()),
            "fit_markdown": markdown_fit,
            "has_structured_data": bool(result.extracted_content),
        }
        return {
            "success": result.success,
            "url": result.url,
            "title": metadata.get("title", ""),
            "content": content,
            "html": result.html or "",
            "metadata": metadata,
            "links": result.links or {},
            "media": result.media or {},
            "structured_data": result.extracted_content,
            "provider": "crawl4ai",
        }


__all__ = ["Crawl4AIAdapter"]
