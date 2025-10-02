"""Unit tests for Crawl4AI preset helpers."""

from crawl4ai import CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy

from src.services.crawling.c4a_presets import (
    BrowserOptions,
    base_run_config,
    best_first_run_config,
    bfs_run_config,
    build_filter_chain,
    build_markdown_generator,
    memory_dispatcher,
    preset_browser_config,
)


def test_preset_browser_config_defaults() -> None:
    """Create a default browser configuration."""

    browser_cfg = preset_browser_config()
    assert browser_cfg.browser_type == "chromium"
    assert browser_cfg.headless is True
    assert browser_cfg.viewport_width == 1280
    assert browser_cfg.viewport_height == 720


def test_preset_browser_config_custom_options() -> None:
    """Create a browser configuration with custom options."""

    opts = BrowserOptions(browser_type="firefox", headless=False, viewport=(1920, 1080))
    browser_cfg = preset_browser_config(opts)
    assert browser_cfg.browser_type == "firefox"
    assert browser_cfg.headless is False
    assert browser_cfg.viewport_width == 1920
    assert browser_cfg.viewport_height == 1080


def test_markdown_generator_uses_pruning_by_default() -> None:
    """Test that the markdown generator uses pruning by default."""

    generator = build_markdown_generator()
    assert isinstance(generator.content_filter, PruningContentFilter)


def test_markdown_generator_switches_to_bm25() -> None:
    """Test that the markdown generator switches to BM25 when a query is provided."""

    generator = build_markdown_generator(query="python crawl tutorial")
    assert isinstance(generator.content_filter, BM25ContentFilter)


def test_build_filter_chain_combines_filters() -> None:
    """Test that the filter chain combines multiple filters."""

    chain = build_filter_chain(
        include_domains=["docs.crawl4ai.com"],
        url_patterns=["*core*"],
        content_types=["text/html"],
        relevance_query="crawler",
        seo_keywords=["guide"],
    )
    assert chain is not None
    assert len(chain.filters) == 5


def test_base_run_config_defaults() -> None:
    """Create a base run configuration with default settings."""

    run_cfg = base_run_config()
    assert run_cfg.cache_mode == CacheMode.BYPASS
    assert set(run_cfg.excluded_tags or []) == {"script", "style"}


def test_bfs_run_config_assigns_strategy() -> None:
    """Create a BFS deep crawl configuration."""

    cfg = bfs_run_config(2)
    assert isinstance(cfg.deep_crawl_strategy, BFSDeepCrawlStrategy)
    assert cfg.deep_crawl_strategy.max_depth == 2
    assert cfg.stream is False


def test_best_first_run_config_assigns_strategy() -> None:
    """Create a Best-First deep crawl configuration."""

    cfg = best_first_run_config(["crawler", "markdown"], stream=True)
    assert isinstance(cfg.deep_crawl_strategy, BestFirstCrawlingStrategy)
    assert cfg.stream is True


def test_memory_dispatcher_configuration() -> None:
    """Create a memory-based dispatcher with custom settings."""

    dispatcher = memory_dispatcher(memory_threshold_percent=65.0, max_session_permit=5)
    assert dispatcher.memory_threshold_percent == 65.0
    assert dispatcher.max_session_permit == 5
