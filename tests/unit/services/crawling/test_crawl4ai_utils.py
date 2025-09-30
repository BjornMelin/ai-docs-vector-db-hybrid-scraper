"""Tests for Crawl4AI utility functions."""

from collections import deque
from unittest.mock import Mock

import pytest

from src.services.crawling.crawl4ai_utils import (
    Crawl4AIScrapeOptions,
    CrawlQueueState,
    config_value,
    create_queue_state,
    normalize_results,
    resolve_stream_flag,
    should_enqueue_link,
)


class TestCrawl4AIScrapeOptions:
    """Tests for Crawl4AIScrapeOptions dataclass."""

    def test_default_options(self):
        """Test default scrape options."""

        options = Crawl4AIScrapeOptions()

        assert options.extraction_type == "markdown"
        assert options.wait_for is None
        assert options.js_code is None
        assert options.stream is None

    def test_custom_options(self):
        """Test custom scrape options."""

        options = Crawl4AIScrapeOptions(
            extraction_type="structured",
            wait_for=".content-loaded",
            js_code="console.log('test')",
            stream=True,
        )

        assert options.extraction_type == "structured"
        assert options.wait_for == ".content-loaded"
        assert options.js_code == "console.log('test')"
        assert options.stream is True


class TestCrawlQueueState:
    """Tests for CrawlQueueState dataclass and methods."""

    def test_queue_state_initialization(self):
        """Test queue state initialization."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=50,
            max_visited=150,
            pending=deque(["https://example.com"]),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        assert state.base_domain == "example.com"
        assert state.max_pages == 50
        assert state.max_visited == 150
        assert len(state.pending) == 1
        assert len(state.visited_order) == 0
        assert len(state.visited_lookup) == 0
        assert len(state.pages) == 0

    def test_take_batch(self):
        """Test taking a batch of URLs from the queue."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(
                [
                    "https://example.com/page1",
                    "https://example.com/page2",
                    "https://example.com/page3",
                ]
            ),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        batch = state.take_batch(2)

        assert len(batch) == 2
        assert "https://example.com/page1" in batch
        assert "https://example.com/page2" in batch
        assert len(state.pending) == 1
        assert len(state.visited_lookup) == 2

    def test_take_batch_skips_visited(self):
        """Test that take_batch skips already-visited URLs."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(
                [
                    "https://example.com/page1",
                    "https://example.com/page2",
                ]
            ),
            visited_order=deque(),
            visited_lookup={"https://example.com/page1"},
            pages=[],
        )

        batch = state.take_batch(2)

        assert len(batch) == 1
        assert "https://example.com/page2" in batch
        assert "https://example.com/page1" not in batch

    def test_queue_link(self):
        """Test queuing a new link."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        state.queue_link("https://example.com/new-page")

        assert len(state.pending) == 1
        assert "https://example.com/new-page" in state.pending

    def test_queue_link_skips_duplicates(self):
        """Test that queue_link skips already-queued URLs."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(["https://example.com/page1"]),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        state.queue_link("https://example.com/page1")

        # Should not add duplicate
        assert len(state.pending) == 1

    def test_queue_link_skips_visited(self):
        """Test that queue_link skips already-visited URLs."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup={"https://example.com/visited"},
            pages=[],
        )

        state.queue_link("https://example.com/visited")

        # Should not queue visited URL
        assert len(state.pending) == 0

    def test_trim_visited(self):
        """Test that _trim_visited keeps cache bounded."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=5,
            pending=deque(),
            visited_order=deque(
                [
                    "https://example.com/page1",
                    "https://example.com/page2",
                    "https://example.com/page3",
                    "https://example.com/page4",
                    "https://example.com/page5",
                    "https://example.com/page6",
                ]
            ),
            visited_lookup={
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
                "https://example.com/page4",
                "https://example.com/page5",
                "https://example.com/page6",
            },
            pages=[],
        )

        state._trim_visited()

        # Should reduce to 80% of max_visited (4 items)
        assert len(state.visited_order) == 4
        assert len(state.visited_lookup) == 4


class TestConfigValue:
    """Tests for config_value utility function."""

    def test_config_value_returns_attribute(self):
        """Test that config_value returns existing attribute."""

        config = Mock()
        config.test_attr = "test_value"

        result = config_value(config, "test_attr", "default")

        assert result == "test_value"

    def test_config_value_returns_default(self):
        """Test that config_value returns default for missing attribute."""

        config = Mock(spec=[])

        result = config_value(config, "missing_attr", "default_value")

        assert result == "default_value"

    def test_config_value_handles_none(self):
        """Test that config_value can return None."""

        config = Mock()
        config.nullable_attr = None

        result = config_value(config, "nullable_attr", "default")

        assert result is None


class TestResolveStreamFlag:
    """Tests for resolve_stream_flag utility function."""

    def test_resolve_stream_flag_uses_override_true(self):
        """Test that explicit True override is used."""

        result = resolve_stream_flag(True, False)

        assert result is True

    def test_resolve_stream_flag_uses_override_false(self):
        """Test that explicit False override is used."""

        result = resolve_stream_flag(False, True)

        assert result is False

    def test_resolve_stream_flag_uses_default_when_none(self):
        """Test that default is used when override is None."""

        result = resolve_stream_flag(None, True)

        assert result is True

        result = resolve_stream_flag(None, False)

        assert result is False


class TestNormalizeResults:
    """Tests for normalize_results utility function."""

    @pytest.mark.asyncio
    async def test_normalize_async_generator(self):
        """Test normalizing async generator to list."""

        async def mock_gen():
            """Mock async generator."""
            yield 1
            yield 2
            yield 3

        result = await normalize_results(mock_gen())

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_normalize_iterable(self):
        """Test normalizing regular iterable to list."""

        result = await normalize_results((1, 2, 3))

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_normalize_single_object(self):
        """Test normalizing single object to list."""

        mock_obj = Mock()
        result = await normalize_results(mock_obj)

        assert isinstance(result, list)
        assert result == [mock_obj]

    @pytest.mark.asyncio
    async def test_normalize_string_to_char_list(self):
        """Test strings normalized to list of characters (iterable behavior)."""

        result = await normalize_results("abc")

        assert isinstance(result, list)
        # Strings are iterable, so they become list of chars
        assert result == ["a", "b", "c"]


class TestCreateQueueState:
    """Tests for create_queue_state utility function."""

    def test_create_queue_state_basic(self):
        """Test basic queue state creation."""

        state = create_queue_state("https://example.com", 50)

        assert state.base_domain == "example.com"
        assert state.max_pages == 50
        assert len(state.pending) == 1
        assert "https://example.com" in state.pending
        assert len(state.visited_order) == 0
        assert len(state.visited_lookup) == 0
        assert len(state.pages) == 0

    def test_create_queue_state_calculates_max_visited(self):
        """Test that max_visited is calculated correctly."""

        state = create_queue_state("https://example.com", 100)

        # max_visited should be max(max_pages * 3, 1000)
        assert state.max_visited == max(100 * 3, 1000)

    def test_create_queue_state_with_subdomain(self):
        """Test queue state creation with subdomain."""

        state = create_queue_state("https://blog.example.com/page", 25)

        assert state.base_domain == "blog.example.com"
        assert "https://blog.example.com/page" in state.pending

    def test_create_queue_state_with_path(self):
        """Test queue state creation with path."""

        state = create_queue_state("https://example.com/docs/index", 10)

        assert state.base_domain == "example.com"
        assert "https://example.com/docs/index" in state.pending


class TestShouldEnqueueLink:
    """Tests for should_enqueue_link utility function."""

    def test_should_enqueue_same_domain(self):
        """Test that links from same domain should be enqueued."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        assert should_enqueue_link(state, "https://example.com/page1") is True

    def test_should_not_enqueue_different_domain(self):
        """Test that links from different domain should not be enqueued."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        assert should_enqueue_link(state, "https://other.com/page1") is False

    def test_should_not_enqueue_empty_url(self):
        """Test that empty URLs should not be enqueued."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        assert should_enqueue_link(state, "") is False

    def test_should_not_enqueue_non_http_url(self):
        """Test that non-HTTP URLs should not be enqueued."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        assert should_enqueue_link(state, "mailto:test@example.com") is False
        assert should_enqueue_link(state, "javascript:void(0)") is False
        assert should_enqueue_link(state, "#anchor") is False

    def test_should_enqueue_subdomain(self):
        """Test handling of subdomains."""

        state = CrawlQueueState(
            base_domain="example.com",
            max_pages=10,
            max_visited=30,
            pending=deque(),
            visited_order=deque(),
            visited_lookup=set(),
            pages=[],
        )

        # Different subdomain should not be enqueued
        assert should_enqueue_link(state, "https://blog.example.com/page1") is False
