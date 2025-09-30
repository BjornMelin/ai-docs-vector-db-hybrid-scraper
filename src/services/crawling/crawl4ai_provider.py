"""Enhanced Crawl4AI provider with concurrency dispatchers."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig as BrowserConfigType,
    CrawlerRunConfig as CrawlerRunConfigType,
    SemaphoreDispatcher,
)
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)


try:
    from crawl4ai import (
        CrawlerMonitor,
        LXMLWebScrapingStrategy,
        MemoryAdaptiveDispatcher,
        RateLimiter as Crawl4AIRateLimiter,
    )

    MEMORY_ADAPTIVE_AVAILABLE = True
except ImportError:
    MEMORY_ADAPTIVE_AVAILABLE = False
    MemoryAdaptiveDispatcher = None  # type: ignore[assignment]
    CrawlerMonitor = None  # type: ignore[assignment]
    LXMLWebScrapingStrategy = None  # type: ignore[assignment]
    Crawl4AIRateLimiter = None  # type: ignore[assignment]

from crawl4ai.models import CrawlerTaskResult

from src.config import Crawl4AIConfig
from src.services.base import BaseService
from src.services.errors import CrawlServiceError

from .base import CrawlProvider
from .crawl4ai_utils import (
    DEFAULT_DISPATCHER_INTERVAL,
    DEFAULT_MAX_SESSION_PERMIT,
    DEFAULT_MEMORY_THRESHOLD,
    DEFAULT_RATE_LIMIT_BASE_DELAY,
    DEFAULT_RATE_LIMIT_MAX_DELAY,
    DEFAULT_RATE_LIMIT_RETRIES,
    DEFAULT_STREAMING_ENABLED,
    DEFAULT_VIEWPORT,
    Crawl4AIScrapeOptions,
    CrawlQueueState,
    build_crawl_error_context,
    classify_scrape_failure,
    config_value,
    create_queue_state,
    normalize_results,
    process_crawl_results,
    resolve_stream_flag,
)
from .extractors import DocumentationExtractor, JavaScriptExecutor


CrawlerRunConfig = CrawlerRunConfigType


logger = logging.getLogger(__name__)


class Crawl4AIProvider(BaseService, CrawlProvider):
    """High-performance web crawling with
    Memory-Adaptive Dispatcher for intelligent concurrency control."""

    def __init__(self, config: Crawl4AIConfig, rate_limiter: object = None):
        """Initialize Crawl4AI provider with configured dispatcher."""

        super().__init__()
        self.config = cast(Any, config)
        _ = rate_limiter  # Legacy parameter retained for compatibility

        self.dispatcher = None
        self.use_memory_dispatcher = False
        self.dispatcher = self._create_dispatcher()

        # Initialize helpers
        self.js_executor = JavaScriptExecutor()
        self.doc_extractor = DocumentationExtractor()

        self._crawler: AsyncWebCrawler | None = None
        self._initialized = False

    def _create_dispatcher(self) -> object | None:
        """Instantiate the appropriate dispatcher for concurrency control."""

        self.use_memory_dispatcher = False
        memory_enabled = bool(
            getattr(self.config, "enable_memory_adaptive_dispatcher", True)
        )

        if MEMORY_ADAPTIVE_AVAILABLE and memory_enabled:
            dispatcher = self._create_memory_dispatcher()
            logger.info(
                "Using MemoryAdaptiveDispatcher for intelligent concurrency control"
            )
            self.use_memory_dispatcher = True
            return dispatcher

        if memory_enabled and not MEMORY_ADAPTIVE_AVAILABLE:
            logger.warning(
                "Memory-Adaptive Dispatcher requested but not available, "
                "falling back to SemaphoreDispatcher",
            )

        if self.config.max_concurrent_crawls > 0:
            logger.info(
                "Using SemaphoreDispatcher for concurrency control (max: %s)",
                self.config.max_concurrent_crawls,
            )
            return SemaphoreDispatcher(
                semaphore_count=self.config.max_concurrent_crawls
            )

        logger.info("Running without dispatcher limits (max_concurrent_crawls <= 0)")
        return None

    def _create_memory_dispatcher(self) -> Any:
        """Create Memory-Adaptive Dispatcher with configuration."""

        if not (
            MEMORY_ADAPTIVE_AVAILABLE
            and MemoryAdaptiveDispatcher is not None
            and Crawl4AIRateLimiter is not None
            and CrawlerMonitor is not None
        ):
            msg = "Memory-Adaptive Dispatcher unavailable in this environment"
            raise CrawlServiceError(msg)

        base_delay = config_value(
            self.config, "rate_limit_base_delay_min", DEFAULT_RATE_LIMIT_BASE_DELAY[0]
        )
        max_delay_base = config_value(
            self.config, "rate_limit_base_delay_max", DEFAULT_RATE_LIMIT_BASE_DELAY[1]
        )
        crawl4ai_rate_limiter = Crawl4AIRateLimiter(
            base_delay=(base_delay, max_delay_base),
            max_delay=config_value(
                self.config, "rate_limit_max_delay", DEFAULT_RATE_LIMIT_MAX_DELAY
            ),
            max_retries=config_value(
                self.config, "rate_limit_max_retries", DEFAULT_RATE_LIMIT_RETRIES
            ),
        )

        monitor = CrawlerMonitor(refresh_rate=1.0, enable_ui=False)

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=config_value(
                self.config, "memory_threshold_percent", DEFAULT_MEMORY_THRESHOLD
            ),
            check_interval=config_value(
                self.config, "dispatcher_check_interval", DEFAULT_DISPATCHER_INTERVAL
            ),
            max_session_permit=config_value(
                self.config, "max_session_permit", DEFAULT_MAX_SESSION_PERMIT
            ),
            rate_limiter=crawl4ai_rate_limiter,
            monitor=monitor,
        )

        logger.info(
            "Created MemoryAdaptiveDispatcher: memory_threshold=%s%%, max_sessions=%s, "
            "check_interval=%ss",
            config_value(
                self.config, "memory_threshold_percent", DEFAULT_MEMORY_THRESHOLD
            ),
            config_value(self.config, "max_session_permit", DEFAULT_MAX_SESSION_PERMIT),
            config_value(
                self.config, "dispatcher_check_interval", DEFAULT_DISPATCHER_INTERVAL
            ),
        )

        return dispatcher

    async def initialize(self) -> None:
        """Initialize Crawl4AI crawler with Memory-Adaptive Dispatcher."""

        if self._initialized:
            return

        try:
            await self._initialize_memory_dispatcher()
            await self._initialize_crawler_with_config()

        except Exception as e:
            msg = "Failed to initialize Crawl4AI"
            raise CrawlServiceError(msg) from e

    async def _run_with_dispatcher(
        self,
        urls: list[str],
        configs: list[CrawlerRunConfig],
    ) -> list[CrawlerTaskResult]:
        """Run crawl tasks using the configured dispatcher or directly."""

        if not self._crawler:
            msg = "Crawler not initialized"
            raise CrawlServiceError(msg)

        dispatcher = self.dispatcher
        payload_config: CrawlerRunConfig | list[CrawlerRunConfig]
        payload_config = configs if len(configs) > 1 else configs[0]

        if dispatcher:
            if self.use_memory_dispatcher:
                dispatcher_any = cast(Any, dispatcher)
                return await dispatcher_any.run_urls(
                    urls=urls, crawler=self._crawler, config=payload_config
                )
            if isinstance(dispatcher, SemaphoreDispatcher):
                return await dispatcher.run_urls(self._crawler, urls, payload_config)

        results = await self._crawler.arun_many(
            urls=urls,
            config=payload_config,
        )

        extracted_results = await normalize_results(results)
        task_results: list[CrawlerTaskResult] = []
        for index, (url, crawl_result) in enumerate(
            zip(urls, extracted_results, strict=False)
        ):
            task_results.append(
                CrawlerTaskResult(
                    task_id=f"direct-{index}",
                    url=url,
                    result=cast(Any, crawl_result),
                    memory_usage=0.0,
                    peak_memory=0.0,
                    start_time=0.0,
                    end_time=0.0,
                )
            )

        return task_results

    def _dispatcher_label(self) -> str:
        """Describe the current dispatcher type."""

        if (
            MEMORY_ADAPTIVE_AVAILABLE
            and MemoryAdaptiveDispatcher is not None
            and isinstance(self.dispatcher, MemoryAdaptiveDispatcher)
        ):
            return "memory_adaptive"
        if isinstance(self.dispatcher, SemaphoreDispatcher):
            return "semaphore"
        return "direct"

    async def _run_single_task(
        self, url: str, run_config: CrawlerRunConfig
    ) -> CrawlerTaskResult | None:
        """Execute a single crawl and return the resulting task."""

        tasks = await self._run_with_dispatcher([url], [run_config])
        if not tasks:
            return None
        return tasks[0]

    def _build_browser_config(self) -> Any:
        """Construct browser configuration for Crawl4AI."""

        viewport = cast(
            dict[str, int], getattr(self.config, "viewport", DEFAULT_VIEWPORT)
        )
        return BrowserConfigType(
            browser_type=self.config.browser_type,
            headless=self.config.headless,
            viewport_width=viewport.get("width", DEFAULT_VIEWPORT["width"]),
            viewport_height=viewport.get("height", DEFAULT_VIEWPORT["height"]),
            user_agent="Mozilla/5.0 (compatible; AIDocs/1.0; +https://github.com/ai-docs)",
        )

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources and dispatcher state."""

        if self.use_memory_dispatcher and self.dispatcher is not None:
            try:
                self.dispatcher = None
                logger.info("MemoryAdaptiveDispatcher reference cleared")
            except (ConnectionError, OSError, PermissionError):
                logger.exception("Error cleaning up MemoryAdaptiveDispatcher")
            finally:
                self.use_memory_dispatcher = False

        if self._crawler:
            try:
                await self._crawler.close()
            except (OSError, AttributeError, ConnectionError, ImportError):
                logger.exception("Error closing crawler")
            finally:
                self._crawler = None
                self._initialized = False
                logger.info(
                    "Crawl4AI resources cleaned up with %s dispatcher",
                    self._dispatcher_label(),
                )

    def _create_extraction_strategy(self, extraction_type: str) -> object | None:
        """Create extraction strategy based on type.

        Args:
            extraction_type: Type of extraction ("structured", "llm", or "markdown")

        Returns:
            Extraction strategy instance or None for markdown extraction
        """

        if extraction_type == "structured":
            return JsonCssExtractionStrategy(
                schema=self.doc_extractor.create_extraction_schema()
            )
        if extraction_type == "llm":
            llm_config = LLMConfig(provider="ollama/llama2")
            return LLMExtractionStrategy(
                llm_config=llm_config,
                instruction="Extract technical documentation with code examples",
            )
        return None

    def _create_run_config(
        self,
        wait_for: Any,
        js_code: Any,
        extraction_strategy: Any,
    ) -> CrawlerRunConfig:
        """Create crawler run configuration.

        Args:
            wait_for: CSS selector to wait for
            js_code: JavaScript code to execute
            extraction_strategy: Extraction strategy instance

        Returns:
            CrawlerRunConfig: Configured crawler run settings
        """

        return cast(
            CrawlerRunConfig,
            CrawlerRunConfigType(
                word_count_threshold=10,
                css_selector=", ".join(self.doc_extractor.selectors["content"]),
                excluded_tags=[
                    "nav",
                    "footer",
                    "header",
                    "aside",
                    "script",
                    "style",
                ],
                wait_for=wait_for,
                js_code=js_code,
                extraction_strategy=extraction_strategy,
                cache_mode=cast(Any, "enabled"),
                page_timeout=int(
                    config_value(self.config, "page_timeout", 30.0) * 1000
                ),
                wait_until="networkidle",
            ),
        )

    def _build_success_result(
        self,
        url: str,
        result: Any,
        extraction_type: str,
    ) -> dict[str, object]:
        """Build success result dictionary.

        Args:
            url: The scraped URL
            result: Crawl result object
            extraction_type: Type of extraction used

        Returns:
            dict[str, object]: Formatted success result
        """

        structured_data = {}
        if result.extracted_content:
            structured_data = result.extracted_content

        return {
            "success": True,
            "url": url,
            "title": result.metadata.get("title", ""),
            "content": result.markdown or "",
            "html": result.html or "",
            "metadata": {
                **result.metadata,
                "extraction_type": extraction_type,
                "word_count": len((result.markdown or "").split()),
                "has_structured_data": bool(structured_data),
            },
            "structured_data": structured_data,
            "links": result.links or [],
            "media": result.media or {},
            "provider": "crawl4ai",
        }

    def _build_error_result(
        self,
        url: str,
        error: str | Exception,
        extraction_type: str | None = None,
    ) -> dict[str, object]:
        """Build error result dictionary.

        Args:
            url: The URL that failed
            error: Error message or exception
            extraction_type: Type of extraction attempted

        Returns:
            dict[str, object]: Formatted error result
        """

        error_context = {
            "url": url,
            "extraction_type": extraction_type,
            "dispatcher": self._dispatcher_label(),
        }

        logger.error("Failed to scrape %s: %s | Context", url, error)

        return {
            "success": False,
            "error": str(error),
            "error_context": error_context,
            "content": "",
            "metadata": {},
            "url": url,
            "provider": "crawl4ai",
        }

    def _format_scrape_response(
        self,
        url: str,
        task: CrawlerTaskResult | None,
        scrape_options: Crawl4AIScrapeOptions,
        stream_enabled: bool,
    ) -> dict[str, object]:
        """Format crawl task results into the public response contract."""

        if not task:
            return self._build_error_result(
                url,
                "Crawl returned no results",
                scrape_options.extraction_type,
            )

        crawl_result = task.result
        if crawl_result.success:
            payload = self._build_success_result(
                url, crawl_result, scrape_options.extraction_type
            )
            metadata = cast(dict[str, object], payload["metadata"])
            if stats := self._build_dispatcher_metadata(task):
                metadata["dispatcher_stats"] = stats
            if stream_enabled:
                metadata["streaming"] = {
                    "enabled": True,
                    "chunks_received": 1,
                    "real_time_processing": False,
                }
            return payload

        error_msg = getattr(crawl_result, "error_message", "Crawl failed")
        return self._build_error_result(url, error_msg, scrape_options.extraction_type)

    async def scrape_url(
        self,
        url: str,
        formats: list[str] | None = None,
        **options: object,
    ) -> dict[str, object]:
        """Scrape single URL with Memory-Adaptive Dispatcher and optional streaming.

        Args:
            url: URL to scrape
            formats: Output formats (ignored, always returns markdown + html)
            **options: Optional Crawl4AI-specific parameters (`extraction_type`,
                `wait_for`, `js_code`, `stream`).

        Returns:
            dict[str, object]: Scrape result with:
                - success: Whether scraping succeeded
                - content: Extracted content in markdown format
                - html: Raw HTML content
                - metadata: Additional information
                - structured_data: Structured extraction results
                - error: Error message if failed
                - stream_data: Streaming iterator if streaming enabled
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        _ = formats
        try:
            scrape_options = Crawl4AIScrapeOptions(**cast(dict[str, Any], options))
        except TypeError as exc:
            msg = f"Invalid scrape options: {exc}"
            raise CrawlServiceError(msg) from exc
        try:
            stream_enabled = resolve_stream_flag(
                scrape_options.stream,
                config_value(
                    self.config, "enable_streaming", DEFAULT_STREAMING_ENABLED
                ),
            )
            task = await self._run_single_task(
                url,
                await self._prepare_run_config(
                    scrape_options.extraction_type,
                    scrape_options.wait_for,
                    await self._prepare_javascript_code(url, scrape_options.js_code),
                    stream_enabled,
                ),
            )

            return self._format_scrape_response(
                url, task, scrape_options, stream_enabled
            )

        except Exception as e:
            logger.exception("Failed to scrape %s", url)
            detail = classify_scrape_failure(url, str(e), logger)
            return self._build_error_result(
                url,
                detail,
                scrape_options.extraction_type,
            )

    def _build_dispatcher_metadata(
        self, task_result: CrawlerTaskResult | None
    ) -> dict[str, object] | None:
        """Collect dispatcher metadata for result payloads."""

        if self.use_memory_dispatcher and self.dispatcher is not None:
            stats: dict[str, object] = {
                "dispatcher_type": "memory_adaptive",
                "memory_threshold_percent": config_value(
                    self.config, "memory_threshold_percent", DEFAULT_MEMORY_THRESHOLD
                ),
                "max_session_permit": config_value(
                    self.config, "max_session_permit", DEFAULT_MAX_SESSION_PERMIT
                ),
                "check_interval": config_value(
                    self.config,
                    "dispatcher_check_interval",
                    DEFAULT_DISPATCHER_INTERVAL,
                ),
            }

            if task_result:
                stats.update(
                    {
                        "memory_usage": task_result.memory_usage,
                        "peak_memory": task_result.peak_memory,
                        "wait_time": task_result.wait_time,
                    }
                )

            dispatcher_any = cast(Any, self.dispatcher)
            if hasattr(dispatcher_any, "get_stats"):
                try:
                    stats.update(dispatcher_any.get_stats())
                except (asyncio.CancelledError, TimeoutError, RuntimeError) as exc:
                    logger.warning("Could not get dispatcher stats")
                    stats["stats_error"] = str(exc)

            return stats

        if isinstance(self.dispatcher, SemaphoreDispatcher):
            stats = {
                "dispatcher_type": "semaphore",
                "semaphore_count": self.config.max_concurrent_crawls,
            }
            if task_result:
                stats["wait_time"] = task_result.wait_time
            return stats

        return None

    async def scrape_url_stream(
        self,
        url: str,
        extraction_type: str = "markdown",
        wait_for: str | None = None,
        js_code: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream scrape results in real-time using Memory-Adaptive Dispatcher.

        Args:
            url: URL to scrape
            extraction_type: Type of extraction to use
            wait_for: CSS selector to wait for
            js_code: Custom JavaScript to execute

        Yields:
            dict[str, object]: Streaming chunks of scrape results
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        if not (self.use_memory_dispatcher and self.dispatcher is not None):
            msg = "Streaming requires Memory-Adaptive Dispatcher"
            raise CrawlServiceError(msg)

        try:
            prepared_js = await self._prepare_javascript_code(url, js_code)
            extraction_strategy = self._create_extraction_strategy(extraction_type)
            run_config = self._create_run_config(
                wait_for, prepared_js, extraction_strategy
            )
            run_config.stream = True

            task_results = await self._run_with_dispatcher([url], [run_config])
            loop_time = asyncio.get_running_loop().time()

            for task in task_results:
                crawl_result = task.result
                if crawl_result.success:
                    yield {
                        "url": url,
                        "chunk": crawl_result,
                        "timestamp": loop_time,
                        "extraction_type": extraction_type,
                        "provider": "crawl4ai",
                        "streaming": True,
                        "dispatcher_stats": self._build_dispatcher_metadata(task),
                    }
                else:
                    yield self._build_error_result(
                        url,
                        getattr(crawl_result, "error_message", "Crawl failed"),
                        extraction_type,
                    )

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as exc:
            yield self._build_error_result(url, str(exc), extraction_type)

    async def crawl_bulk(
        self, urls: list[str], extraction_type: str = "markdown"
    ) -> list[dict[str, object]]:
        """Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            extraction_type: Type of extraction to use

        Returns:
            List of crawl results
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        run_configs: list[CrawlerRunConfig] = []
        for target_url in urls:
            prepared_js = await self._prepare_javascript_code(target_url, None)
            run_configs.append(
                await self._prepare_run_config(
                    extraction_type, None, prepared_js, False
                )
            )
        task_results = await self._run_with_dispatcher(urls, run_configs)

        successful: list[dict[str, object]] = []
        failed_count = 0

        for url, task in zip(urls, task_results, strict=False):
            payload = self._build_bulk_payload(url, task, extraction_type)
            if payload is not None:
                successful.append(payload)
            else:
                failed_count += 1
                logger.error("Failed to crawl %s", url)

        if failed_count:
            logger.warning("Failed to crawl %s URLs out of %s", failed_count, len(urls))

        return successful

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, object]:
        """Crawl entire site using recursive URL discovery.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats (ignored)

        Returns:
            Crawl result with all pages
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        _ = formats
        state = create_queue_state(url, max_pages)

        try:
            while await self._crawl_site_step(state, url):
                continue

            return {
                "success": True,
                "pages": state.pages,
                "total": len(state.pages),
                "provider": "crawl4ai",
            }

        except Exception as e:
            logger.exception("Failed to crawl site %s | Context", url)
            return {
                "success": False,
                "error": str(e),
                "error_context": build_crawl_error_context(state, url),
                "pages": state.pages,
                "total": len(state.pages),
                "provider": "crawl4ai",
            }

    async def _prepare_javascript_code(
        self, url: str, js_code: str | None
    ) -> str | None:
        """Prepare JavaScript code for crawling."""

        if not js_code:
            return self.js_executor.get_js_for_site(url)
        return js_code

    async def _prepare_run_config(
        self,
        extraction_type: str,
        wait_for: str | None,
        js_code: str | None,
        enable_streaming: bool,
    ) -> CrawlerRunConfig:
        """Prepare crawler run configuration."""

        extraction_strategy = self._create_extraction_strategy(extraction_type)
        run_config = self._create_run_config(
            cast(Any, wait_for),
            cast(Any, js_code),
            cast(Any, extraction_strategy),
        )

        if enable_streaming:
            run_config.stream = True

        return run_config

    async def _crawl_site_step(self, state: CrawlQueueState, starting_url: str) -> bool:
        """Execute one crawl iteration; return False when complete."""

        if not state.pending or len(state.pages) >= state.max_pages:
            return False

        batch_urls = state.take_batch(min(10, state.max_pages - len(state.pages)))
        if not batch_urls:
            return False

        process_crawl_results(state, await self.crawl_bulk(batch_urls))

        logger.info(
            "Crawled %s/%s pages from %s",
            len(state.pages),
            state.max_pages,
            starting_url,
        )
        return True

    def _build_bulk_payload(
        self, url: str, task: CrawlerTaskResult, extraction_type: str
    ) -> dict[str, object] | None:
        """Convert a bulk crawl task into a standardized payload."""

        crawl_result = task.result
        if not crawl_result.success:
            return None

        payload = self._build_success_result(url, crawl_result, extraction_type)
        metadata = cast(dict[str, object], payload["metadata"])
        if stats := self._build_dispatcher_metadata(task):
            metadata["dispatcher_stats"] = stats
        return payload

    async def _initialize_memory_dispatcher(self) -> None:
        """Initialize Memory-Adaptive Dispatcher if enabled."""

        if not (self.use_memory_dispatcher and self.dispatcher is not None):
            return

        dispatcher_any = cast(Any, self.dispatcher)
        if hasattr(dispatcher_any, "initialize"):
            await dispatcher_any.initialize()
            logger.info("MemoryAdaptiveDispatcher initialized successfully")
        else:
            logger.info("MemoryAdaptiveDispatcher ready (no initialize method)")

    async def _initialize_crawler_with_config(self) -> None:
        """Initialize crawler with enhanced configuration."""

        crawler_config = self._build_browser_config()

        # Add LXMLWebScrapingStrategy for performance improvement if available
        await self._setup_web_scraping_strategy(crawler_config)

        self._crawler = AsyncWebCrawler(config=crawler_config)
        await self._crawler.start()
        self._initialized = True

        strategy_info = (
            "with LXMLWebScrapingStrategy"
            if MEMORY_ADAPTIVE_AVAILABLE
            else "with default strategy"
        )
        logger.info(
            "Crawl4AI crawler initialized %s and %s dispatcher",
            strategy_info,
            self._dispatcher_label(),
        )

    async def _setup_web_scraping_strategy(self, crawler_config) -> None:
        """Setup web scraping strategy if available."""

        if not MEMORY_ADAPTIVE_AVAILABLE or LXMLWebScrapingStrategy is None:
            return

        try:
            crawler_config.web_scraping_strategy = LXMLWebScrapingStrategy()
            logger.info("Using LXMLWebScrapingStrategy for enhanced performance")
        except (
            AttributeError,
            ConnectionError,
            ImportError,
            RuntimeError,
        ):
            logger.warning("Could not set LXMLWebScrapingStrategy")
