"""Crawl4AI configuration presets for browser and crawler runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from crawl4ai import (
    BrowserConfig,
    CacheMode,
    CrawlerMonitor,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    LinkPreviewConfig,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    SEOFilter,
    URLFilter,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


DEFAULT_VIEWPORT = (1280, 720)
DEFAULT_BM25_THRESHOLD = 1.2
DEFAULT_PRUNING_THRESHOLD = 0.48
DEFAULT_MIN_WORDS = 40
DEFAULT_MAX_PAGES = 50
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_MEMORY_THRESHOLD = 70.0
DEFAULT_CHECK_INTERVAL = 1.0
DEFAULT_MAX_SESSION_PERMIT = 10
DEFAULT_LINK_TIMEOUT = 10
DEFAULT_LINK_CONCURRENCY = 5
DEFAULT_LINK_THRESHOLD = 0.3


@dataclass(slots=True)
class BrowserOptions:
    """Typed bundle for :class:`BrowserConfig` presets.

    Attributes:
        browser_type: Browser family to launch (e.g. ``"chromium"``).
        headless: Whether to run the browser without a window.
        viewport: Viewport ``(width, height)`` tuple.
        enable_stealth: Enable stealth/undetected mode for evasive sites.
        user_agent: Optional user agent override.
        proxy: Optional proxy URL.
        verbose: Emit verbose Crawl4AI logs when ``True``.
    """

    browser_type: str = "chromium"
    headless: bool = True
    viewport: tuple[int, int] = DEFAULT_VIEWPORT
    enable_stealth: bool = False
    user_agent: str | None = None
    proxy: str | None = None
    verbose: bool = False


def preset_browser_config(options: BrowserOptions | None = None) -> BrowserConfig:
    """Create a reusable :class:`BrowserConfig` with safe defaults.

    Args:
        options: Optional overrides for browser behaviour.

    Returns:
        A fully-populated :class:`BrowserConfig` instance ready for reuse.
    """

    opts = options or BrowserOptions()
    config_kwargs: dict[str, Any] = {
        "browser_type": opts.browser_type,
        "headless": opts.headless,
        "viewport_width": opts.viewport[0],
        "viewport_height": opts.viewport[1],
        "viewport": {"width": opts.viewport[0], "height": opts.viewport[1]},
        "verbose": opts.verbose,
        "enable_stealth": opts.enable_stealth,
        "ignore_https_errors": True,
        "java_script_enabled": True,
    }
    if opts.proxy is not None:
        config_kwargs["proxy"] = opts.proxy
    if opts.user_agent is not None:
        config_kwargs["user_agent"] = opts.user_agent
    return BrowserConfig(**config_kwargs)


def build_markdown_generator(
    *,
    query: str | None = None,
    pruning_threshold: float = DEFAULT_PRUNING_THRESHOLD,
    min_word_threshold: int = DEFAULT_MIN_WORDS,
    bm25_threshold: float = DEFAULT_BM25_THRESHOLD,
) -> DefaultMarkdownGenerator:
    """Configure markdown generation with Fit Markdown heuristics.

    Args:
        query: Optional relevance query activating BM25 filtering.
        pruning_threshold: Threshold for pruning boilerplate content.
        min_word_threshold: Minimum word count retained by pruning filter.
        bm25_threshold: Similarity threshold for BM25 filtering.

    Returns:
        A markdown generator tuned for compact, high-signal content.
    """

    if query:
        content_filter = BM25ContentFilter(
            user_query=query, bm25_threshold=bm25_threshold
        )
    else:
        content_filter = PruningContentFilter(
            threshold=pruning_threshold,
            threshold_type="fixed",
            min_word_threshold=min_word_threshold,
        )
    return DefaultMarkdownGenerator(content_filter=content_filter)


# Exposes Crawl4AI filter composition knobs directly for adapter reuse.
def build_filter_chain(  # pylint: disable=too-many-arguments
    *,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    url_patterns: Sequence[str] | None = None,
    content_types: Sequence[str] | None = None,
    relevance_query: str | None = None,
    relevance_threshold: float = 0.4,
    seo_keywords: Sequence[str] | None = None,
    seo_threshold: float = 0.5,
) -> FilterChain | None:
    """Compose a :class:`FilterChain` for deep crawl targeting.

    Args:
        include_domains: Domain whitelist to retain.
        exclude_domains: Domain blacklist to drop.
        url_patterns: Glob or regex patterns to permit.
        content_types: MIME types to retain during traversal.
        relevance_query: Optional query for content relevance filtering.
        relevance_threshold: Threshold applied to the relevance filter.
        seo_keywords: Keywords influencing SEO filter decisions.
        seo_threshold: Confidence threshold for the SEO filter.

    Returns:
        A composed :class:`FilterChain` or ``None`` when no filters apply.
    """

    filters: list[URLFilter] = []
    if url_patterns:
        filters.append(URLPatternFilter(patterns=list(url_patterns)))
    if include_domains or exclude_domains:
        domain_kwargs: dict[str, str | list[str]] = {}
        if include_domains:
            domain_kwargs["allowed_domains"] = list(include_domains)
        if exclude_domains:
            domain_kwargs["blocked_domains"] = list(exclude_domains)
        filters.append(DomainFilter(**domain_kwargs))
    if content_types:
        filters.append(ContentTypeFilter(allowed_types=list(content_types)))
    if relevance_query:
        filters.append(
            ContentRelevanceFilter(query=relevance_query, threshold=relevance_threshold)
        )
    if seo_keywords:
        filters.append(SEOFilter(threshold=seo_threshold, keywords=list(seo_keywords)))
    return FilterChain(filters) if filters else None


# Mirrors Crawl4AI link preview configuration to avoid bespoke wrappers.
def link_preview_config(  # pylint: disable=too-many-arguments
    query: str,
    *,
    max_links: int = 10,
    concurrency: int = DEFAULT_LINK_CONCURRENCY,
    timeout: int = DEFAULT_LINK_TIMEOUT,
    include_patterns: Sequence[str] | None = None,
    score_threshold: float = DEFAULT_LINK_THRESHOLD,
    include_internal: bool = True,
    include_external: bool = False,
) -> LinkPreviewConfig:
    """Preset for intelligent link previews with scoring.

    Args:
        query: Query string used to rank link previews.
        max_links: Maximum number of links to return.
        concurrency: Concurrent link fetches permitted.
        timeout: Timeout in seconds for preview fetching.
        include_patterns: Optional allowlist of URL patterns.
        score_threshold: Minimum score to keep a link preview.
        include_internal: Whether to include internal links.
        include_external: Whether to include external links.

    Returns:
        Configured :class:`LinkPreviewConfig` for pre-ranking links.
    """

    link_kwargs: dict[str, Any] = {
        "query": query,
        "max_links": max_links,
        "concurrency": concurrency,
        "timeout": timeout,
        "score_threshold": score_threshold,
        "include_internal": include_internal,
        "include_external": include_external,
        "verbose": False,
    }
    if include_patterns:
        link_kwargs["include_patterns"] = list(include_patterns)
    return LinkPreviewConfig(**link_kwargs)


# Mirrors Crawl4AI run configuration surface to keep presets declarative.
def base_run_config(  # pylint: disable=too-many-arguments
    *,
    markdown_generator: DefaultMarkdownGenerator | None = None,
    cache_mode: CacheMode = CacheMode.BYPASS,
    page_timeout: float = 30.0,
    session_id: str | None = None,
    js_code: str | Sequence[str] | None = None,
    wait_for: str | None = None,
    stream: bool = False,
    link_preview: LinkPreviewConfig | None = None,
    strip_scripts: bool = True,
    strip_styles: bool = True,
    verbose: bool = False,
) -> CrawlerRunConfig:
    """Construct a :class:`CrawlerRunConfig` baseline used across runs.

    Args:
        markdown_generator: Optional markdown generator override.
        cache_mode: Cache strategy toggle.
        page_timeout: Page timeout in seconds.
        session_id: Session identifier for stateful crawls.
        js_code: JavaScript snippets injected pre-screenshot.
        wait_for: CSS selector or condition required before extraction.
        stream: Enable streaming semantics for Crawl4AI.
        link_preview: Optional link preview configuration.
        strip_scripts: Remove ``<script>`` tags from output when ``True``.
        strip_styles: Remove ``<style>`` tags from output when ``True``.
        verbose: Enable verbose Crawl4AI logging when ``True``.

    Returns:
        A :class:`CrawlerRunConfig` basis reused by higher-level helpers.
    """

    excluded_tags: list[str] | None = None
    if strip_scripts or strip_styles:
        excluded_tags = []
        if strip_scripts:
            excluded_tags.append("script")
        if strip_styles:
            excluded_tags.append("style")

    config_kwargs: dict[str, Any] = {
        "cache_mode": cache_mode,
        "markdown_generator": markdown_generator or build_markdown_generator(),
        "page_timeout": int(page_timeout * 1000),
        "score_links": link_preview is not None,
        "stream": stream,
        "wait_until": "domcontentloaded",
        "verbose": verbose,
        "remove_overlay_elements": True,
        "ignore_body_visibility": True,
    }
    if session_id is not None:
        config_kwargs["session_id"] = session_id
    if js_code is not None:
        config_kwargs["js_code"] = (
            js_code if isinstance(js_code, str) else list(js_code)
        )
    if wait_for is not None:
        config_kwargs["wait_for"] = wait_for
    if excluded_tags:
        config_kwargs["excluded_tags"] = excluded_tags
    if link_preview is not None:
        config_kwargs["link_preview_config"] = link_preview
    return CrawlerRunConfig(**config_kwargs)


# Keeps Crawl4AI BFS strategy parameters explicit for flexibility.
def bfs_run_config(  # pylint: disable=too-many-arguments
    depth: int,
    *,
    include_external: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filter_chain: FilterChain | None = None,
    base_config: CrawlerRunConfig | None = None,
) -> CrawlerRunConfig:
    """Preset :class:`CrawlerRunConfig` for breadth-first deep crawling.

    Args:
        depth: Maximum crawl depth allowed.
        include_external: Whether to pursue off-domain links.
        max_pages: Hard page-count guardrail.
        score_threshold: Minimum score to retain a page.
        filter_chain: Optional filter chain to constrain traversal.
        base_config: Base configuration to clone and augment.

    Returns:
        A breadth-first :class:`CrawlerRunConfig` preset.
    """

    cfg = (base_config or base_run_config()).clone()
    strategy_kwargs: dict[str, Any] = {
        "max_depth": depth,
        "include_external": include_external,
        "max_pages": max_pages,
        "score_threshold": score_threshold,
    }
    if filter_chain is not None:
        strategy_kwargs["filter_chain"] = filter_chain
    cfg.deep_crawl_strategy = BFSDeepCrawlStrategy(**strategy_kwargs)
    cfg.stream = False
    return cfg


# Retains Crawl4AI best-first strategy parameters for relevance tuning.
def best_first_run_config(  # pylint: disable=too-many-arguments
    keywords: Sequence[str],
    *,
    depth: int = 2,
    max_pages: int = 25,
    include_external: bool = False,
    filter_chain: FilterChain | None = None,
    base_config: CrawlerRunConfig | None = None,
    weight: float = 0.7,
    stream: bool = True,
) -> CrawlerRunConfig:
    """Preset :class:`CrawlerRunConfig` for relevance-first deep crawling.

    Args:
        keywords: Keywords driving the relevance scorer.
        depth: Maximum crawl depth.
        max_pages: Page-count guardrail.
        include_external: Whether to crawl external links.
        filter_chain: Optional filter chain for traversal targeting.
        base_config: Base configuration to clone and augment.
        weight: Weight applied to keyword relevance scoring.
        stream: When ``True`` enable low-latency streaming results.

    Returns:
        A best-first :class:`CrawlerRunConfig` preset tuned for relevance.
    """

    cfg = (base_config or base_run_config(stream=stream)).clone(stream=stream)
    best_first_kwargs: dict[str, Any] = {
        "max_depth": depth,
        "include_external": include_external,
        "max_pages": max_pages,
        "url_scorer": KeywordRelevanceScorer(keywords=list(keywords), weight=weight),
    }
    if filter_chain is not None:
        best_first_kwargs["filter_chain"] = filter_chain
    cfg.deep_crawl_strategy = BestFirstCrawlingStrategy(**best_first_kwargs)
    return cfg


# Mirrors dispatcher tuning surface so callers can tweak without wrappers.
def memory_dispatcher(  # pylint: disable=too-many-arguments
    *,
    memory_threshold_percent: float = DEFAULT_MEMORY_THRESHOLD,
    check_interval: float = DEFAULT_CHECK_INTERVAL,
    max_session_permit: int = DEFAULT_MAX_SESSION_PERMIT,
    rate_limit_base_delay: tuple[float, float] | None = None,
    rate_limit_max_delay: float = 30.0,
    rate_limit_retries: int = 2,
    monitor_refresh_rate: float = 1.0,
) -> MemoryAdaptiveDispatcher:
    """Create a tuned :class:`MemoryAdaptiveDispatcher` for batch crawls.

    Args:
        memory_threshold_percent: Memory utilisation trigger for throttling.
        check_interval: Interval in seconds between memory checks.
        max_session_permit: Maximum concurrent browser sessions.
        rate_limit_base_delay: Optional jitter range for rate limiting.
        rate_limit_max_delay: Cap for exponential backoff delay.
        rate_limit_retries: Maximum retries before giving up.
        monitor_refresh_rate: Refresh interval for the dispatcher monitor.

    Returns:
        Configured :class:`MemoryAdaptiveDispatcher` instance.
    """

    rate_limiter = None
    if rate_limit_base_delay is not None:
        rate_limiter = RateLimiter(
            base_delay=rate_limit_base_delay,
            max_delay=rate_limit_max_delay,
            max_retries=rate_limit_retries,
        )
    monitor = CrawlerMonitor(refresh_rate=monitor_refresh_rate, enable_ui=False)
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=memory_threshold_percent,
        check_interval=check_interval,
        max_session_permit=max_session_permit,
        rate_limiter=rate_limiter,
        monitor=monitor,
    )


__all__ = [
    "BrowserOptions",
    "preset_browser_config",
    "build_markdown_generator",
    "build_filter_chain",
    "link_preview_config",
    "base_run_config",
    "bfs_run_config",
    "best_first_run_config",
    "memory_dispatcher",
]
