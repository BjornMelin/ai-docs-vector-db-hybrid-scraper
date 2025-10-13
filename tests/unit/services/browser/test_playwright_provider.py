"""Unit tests for the Playwright provider."""

from __future__ import annotations

from typing import Any

import pytest

from src.config.browser import PlaywrightSettings, StealthMode
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import playwright as playwright_module
from src.services.browser.providers.base import ProviderContext
from src.services.browser.providers.playwright import PlaywrightProvider


class _StubPage:
    """Minimal Playwright page test double."""

    def __init__(self, html: str, title: str = "Example", text: str = "body") -> None:
        self._html = html
        self._title = title
        self._text = text
        self.url = "about:blank"
        self.goto_calls: list[tuple[str, int]] = []

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.url = url
        self.goto_calls.append((wait_until, timeout))

    async def content(self) -> str:
        return self._html

    async def inner_text(self, selector: str) -> str:
        if selector == "body":
            return self._text
        msg = f"unexpected selector {selector}"
        raise playwright_module.PlaywrightError(msg)

    async def title(self) -> str:
        return self._title


class _StubContext:
    """Async context manager returning a stub page."""

    def __init__(self, page: _StubPage) -> None:
        self._page = page

    async def __aenter__(self) -> _StubContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def new_page(self) -> _StubPage:
        return self._page


class _StubBrowser:
    """Playwright browser double producing stub context."""

    def __init__(self, page: _StubPage) -> None:
        self._context = _StubContext(page)

    def new_context(self) -> _StubContext:
        return self._context


async def _passthrough_retry(**kwargs: Any) -> Any:
    """Execute the wrapped coroutine without retries for deterministic tests."""

    func = kwargs["func"]
    return await func()


@pytest.mark.asyncio
async def test_playwright_scrape_returns_success_when_no_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scrape should succeed when the page HTML lacks challenge markers."""

    monkeypatch.setattr(
        playwright_module, "execute_with_retry", _passthrough_retry, raising=True
    )

    settings = PlaywrightSettings(
        stealth=StealthMode.DISABLED,
        challenges=("captcha", "verify you are human"),
    )
    provider = PlaywrightProvider(ProviderContext(ProviderKind.PLAYWRIGHT), settings)
    provider._browser = _StubBrowser(_StubPage("<html><body>ok</body></html>"))  # type: ignore[attr-defined]

    result = await provider.scrape(ScrapeRequest(url="https://example.com"))
    assert isinstance(result, BrowserResult)
    assert result.success is True
    assert result.metadata["challenge_detected"] is False
    assert result.url == "https://example.com"


@pytest.mark.asyncio
async def test_playwright_scrape_raises_on_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scrape should raise a provider error when challenge keywords are present."""

    monkeypatch.setattr(
        playwright_module, "execute_with_retry", _passthrough_retry, raising=True
    )

    settings = PlaywrightSettings(
        stealth=StealthMode.DISABLED,
        challenges=("captcha",),
    )
    provider = PlaywrightProvider(ProviderContext(ProviderKind.PLAYWRIGHT), settings)
    provider._browser = _StubBrowser(_StubPage("<html>captcha verification</html>"))  # type: ignore[attr-defined]

    with pytest.raises(BrowserProviderError):
        await provider.scrape(ScrapeRequest(url="https://example.com"))
