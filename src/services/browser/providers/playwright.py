"""Playwright-based provider."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import playwright_stealth  # type: ignore[import]  # pyright: ignore[reportMissingTypeStubs]
from playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from src.config.browser import CaptchaProvider, PlaywrightSettings, StealthMode
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.runtime import execute_with_retry

from .base import BrowserProvider, ProviderContext


ApplyStealthFn = Callable[[Page], Awaitable[None]]

maybe_apply = getattr(playwright_stealth, "stealth_async", None)
APPLY_STEALTH: ApplyStealthFn = cast(ApplyStealthFn, maybe_apply)


class PlaywrightProvider(BrowserProvider):
    """Launches headless browser sessions via Playwright."""

    kind = ProviderKind.PLAYWRIGHT

    def __init__(self, context: ProviderContext, settings: PlaywrightSettings) -> None:
        """Init Playwright provider with browser runtime and optional stealth config."""
        super().__init__(context)
        self._settings = settings
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def initialize(self) -> None:
        """Start Playwright and launch configured browser."""
        if self._settings.stealth is StealthMode.REBROWSER:
            raise BrowserProviderError(
                "rebrowser stealth mode is not available in this deployment",
                provider=self.kind.value,
            )
        if self._settings.captcha_provider is not CaptchaProvider.NONE:
            raise BrowserProviderError(
                "Captcha solving is not implemented for Playwright provider",
                provider=self.kind.value,
            )

        self._playwright = await async_playwright().start()
        browser_factory = getattr(self._playwright, self._settings.browser, None)
        if browser_factory is None:
            raise BrowserProviderError(
                f"Unsupported browser type: {self._settings.browser}",
                provider=self.kind.value,
            )
        self._browser = await browser_factory.launch(headless=self._settings.headless)

    async def close(self) -> None:
        """Dispose browser and Playwright runtime."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def _execute_actions(
        self, page: Page, actions: Sequence[Mapping[str, Any]]
    ) -> None:
        """Execute a series of actions on the given page."""
        for action in actions:
            action_type = action.get("type")
            if action_type == "click" and "selector" in action:
                await page.click(action["selector"])
            elif action_type == "fill" and {"selector", "text"} <= action.keys():
                await page.fill(action["selector"], action["text"])
            elif action_type == "wait_for_selector" and "selector" in action:
                await page.wait_for_selector(action["selector"])
            elif action_type == "evaluate" and "script" in action:
                await page.evaluate(action["script"])

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Run a Playwright session."""
        if self._browser is None:  # pragma: no cover - lifecycle guard
            raise RuntimeError("Provider not initialized")

        timeout = request.timeout_ms or self._settings.timeout_ms

        async def _run_session() -> BrowserResult:
            assert self._browser is not None
            async with self._browser.new_context() as context:  # pyright: ignore[reportGeneralTypeIssues]
                page = await context.new_page()
                if self._settings.stealth is StealthMode.PLAYWRIGHT_STEALTH:
                    await APPLY_STEALTH(page)  # pylint: disable=not-callable

                await page.goto(request.url, wait_until="networkidle", timeout=timeout)
                if request.actions:
                    await self._execute_actions(page, request.actions)

                html = await page.content()
                try:
                    text_content = await page.inner_text("body")
                except PlaywrightError:
                    text_content = ""
                metadata: dict[str, Any] = {"url": page.url}
                challenge_detected = any(
                    keyword.lower() in html.lower()
                    for keyword in self._settings.challenges
                )
                metadata["challenge_detected"] = challenge_detected

                if challenge_detected:
                    # escalate as provider error to trigger router fallback
                    raise BrowserProviderError(
                        "anti-bot challenge detected",
                        provider=self.kind.value,
                        context={"url": request.url},
                    )

                return BrowserResult(
                    success=not challenge_detected,
                    url=page.url,
                    title=await page.title(),
                    content=text_content,
                    html=html,
                    metadata=metadata,
                    provider=self.kind,
                    links=None,
                    assets=None,
                    elapsed_ms=None,
                )

        # Respect caller-provided timeout by avoiding extra retry backoff.
        execute_kwargs: dict[str, Any] = {
            "provider": self.kind,
            "operation": "session",
            "func": _run_session,
            "retry_on": (PlaywrightTimeoutError, TimeoutError),
        }
        if request.timeout_ms is not None:
            execute_kwargs["attempts"] = 1

        return await execute_with_retry(**execute_kwargs)
