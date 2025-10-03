"""Tests for the multi-tier Playwright adapter."""

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.config import PlaywrightConfig, PlaywrightTierConfig


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(name="adapter_cls")
def fixture_adapter_cls():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from src.services.browser.playwright_adapter import PlaywrightAdapter

    return PlaywrightAdapter


@pytest.fixture(name="crawl_error")
def fixture_crawl_error():
    from src.services.errors import CrawlServiceError

    return CrawlServiceError


@pytest.fixture
def basic_config() -> PlaywrightConfig:
    """Return a minimal Playwright configuration for tests."""
    return PlaywrightConfig(
        browser="chromium",
        headless=True,
        viewport={"width": 1280, "height": 720},
        user_agent="Mozilla/5.0 (compatible; Test/1.0)",
    )


class TestInitialization:
    """Verify adapter initialization behaviour."""

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    def test_adapter_marks_unavailable_when_playwright_missing(
        self, basic_config, adapter_cls
    ) -> None:
        """Initialization should mark adapter as unavailable when dependency missing."""
        adapter = adapter_cls(basic_config)
        assert adapter._available is False

    @pytest.mark.asyncio
    async def test_initialize_launches_baseline_runtime(
        self, basic_config, adapter_cls
    ) -> None:
        """Adapter should start Playwright and launch the configured browser."""
        mock_playwright = AsyncMock()
        mock_launcher = AsyncMock()
        mock_browser = AsyncMock()
        mock_launcher.launch.return_value = mock_browser
        mock_playwright.chromium = mock_launcher

        async_playwright_patch = patch(
            "src.services.browser.playwright_adapter.async_playwright",
            return_value=AsyncMock(start=AsyncMock(return_value=mock_playwright)),
        )

        with async_playwright_patch:
            adapter = adapter_cls(basic_config)
            await adapter.initialize()

        assert adapter._initialized is True
        assert "baseline" in adapter._runtimes
        mock_launcher.launch.assert_called_once_with(headless=True)

    @pytest.mark.asyncio
    async def test_initialize_requires_rebrowser_when_tier_requests_it(
        self, basic_config, adapter_cls, crawl_error
    ) -> None:
        """Undetected tiers should fail if rebrowser-playwright is missing."""
        config = basic_config.model_copy(
            update={
                "tiers": [
                    PlaywrightTierConfig(
                        name="undetected",
                        use_undetected_browser=True,
                        enable_stealth=True,
                    )
                ]
            }
        )

        with (
            patch(
                "src.services.browser.playwright_adapter.async_playwright",
                return_value=AsyncMock(
                    start=AsyncMock(return_value=AsyncMock(chromium=AsyncMock()))
                ),
            ),
            patch(
                "src.services.browser.playwright_adapter.REBROWSER_AVAILABLE",
                False,
            ),
        ):
            adapter = adapter_cls(config)
            with pytest.raises(crawl_error):
                await adapter.initialize()


class TestScrape:
    """Validate scraping workflow."""

    @pytest.fixture
    def mock_runtime(self):
        """Create a fully mocked Playwright runtime."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.url = "https://example.org/"
        mock_page.inner_text.return_value = "Example body"
        mock_page.content.return_value = "<html>...</html>"
        mock_page.title.return_value = "Example"
        mock_page.goto.return_value = SimpleNamespace(status=200)

        return mock_browser, mock_context, mock_page

    @pytest.mark.asyncio
    async def test_scrape_success(
        self, basic_config, adapter_cls, mock_runtime, monkeypatch
    ) -> None:
        """Scrape should navigate, execute actions, and return structured metadata."""
        mock_browser, mock_context, mock_page = mock_runtime
        adapter = adapter_cls(basic_config)
        adapter._initialized = True
        adapter._runtimes["baseline"] = SimpleNamespace(
            name="baseline",
            browser=mock_browser,
            use_undetected=False,
        )

        monkeypatch.setattr(
            "src.services.browser.playwright_adapter.stealth_async",
            AsyncMock(),
        )

        actions = [{"type": "click", "selector": "button.login"}]
        result = await adapter.scrape("https://example.org", actions)

        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()
        mock_page.goto.assert_called_once_with(
            "https://example.org", wait_until="networkidle", timeout=30000
        )

        assert result["success"] is True
        assert result["url"] == "https://example.org/"
        metadata = result["metadata"]
        assert metadata["tier"] == "baseline"
        assert metadata["runtime"] == "baseline"
        assert metadata["actions_executed"] == 1
        assert metadata["successful_actions"] == 1
        assert metadata["challenge_detected"] is False
        assert metadata["challenge_outcome"] == "none"

    @pytest.mark.asyncio
    async def test_scrape_marks_challenge_on_block_status(
        self, basic_config, adapter_cls, mock_runtime, monkeypatch
    ) -> None:
        """HTTP status codes configured as challenges should toggle escalation."""
        mock_browser, mock_context, mock_page = mock_runtime
        mock_page.goto.return_value = SimpleNamespace(status=429)

        adapter = adapter_cls(basic_config)
        adapter._initialized = True
        adapter._runtimes["baseline"] = SimpleNamespace(
            name="baseline",
            browser=mock_browser,
            use_undetected=False,
        )

        monkeypatch.setattr(
            "src.services.browser.playwright_adapter.stealth_async",
            AsyncMock(),
        )

        result = await adapter.scrape("https://example.org")
        assert result["success"] is False
        assert "challenge" in result["error"].lower()
        assert result["metadata"]["challenge_detected"] is True
        assert result["metadata"]["challenge_outcome"] == "detected"

    @pytest.mark.asyncio
    async def test_scrape_requires_initialization(
        self, basic_config, adapter_cls, crawl_error
    ) -> None:
        """Calling scrape before initialize should raise a CrawlServiceError."""
        adapter = adapter_cls(basic_config)
        with pytest.raises(crawl_error, match="Adapter not initialized"):
            await adapter.scrape("https://example.org")


class TestCleanup:
    """Ensure resources are disposed safely."""

    @pytest.mark.asyncio
    async def test_cleanup_closes_resources(self, basic_config, adapter_cls) -> None:
        """Cleanup should close browser and stop the runtime."""
        adapter = adapter_cls(basic_config)
        adapter._initialized = True
        browser_mock = AsyncMock()
        playwright_mock = AsyncMock()
        adapter._runtimes["baseline"] = SimpleNamespace(
            name="baseline",
            browser=browser_mock,
            use_undetected=False,
        )
        adapter._playwright_handles.append(playwright_mock)

        await adapter.cleanup()

        browser_mock.close.assert_awaited()
        playwright_mock.stop.assert_awaited()
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_tolerates_errors(
        self, basic_config, adapter_cls, caplog
    ) -> None:
        """Cleanup must swallow exceptions and continue closing remaining resources."""
        adapter = adapter_cls(basic_config)
        adapter._initialized = True
        failing_browser = AsyncMock()
        failing_browser.close.side_effect = RuntimeError("boom")
        playwright_mock = AsyncMock()
        adapter._runtimes["baseline"] = SimpleNamespace(
            name="baseline",
            browser=failing_browser,
            use_undetected=False,
        )
        adapter._playwright_handles.append(playwright_mock)

        with caplog.at_level("WARNING"):
            await adapter.cleanup()

        playwright_mock.stop.assert_awaited()
        assert adapter._initialized is False
