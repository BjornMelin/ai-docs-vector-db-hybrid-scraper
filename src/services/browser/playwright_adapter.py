"""Playwright adapter for browser automation with multi-tier anti-bot stack."""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ValidationError

from src.config import (
    PlaywrightCaptchaSettings,
    PlaywrightConfig,
    PlaywrightTierConfig,
)
from src.services.base import BaseService
from src.services.errors import CrawlServiceError

from .action_schemas import validate_actions


logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from playwright.async_api import Browser, BrowserContext, Page, Response
else:  # pragma: no cover - runtime fallbacks when playwright missing
    Browser = BrowserContext = Page = Response = Any

try:  # pragma: no cover - import guard
    from playwright.async_api import (  # type: ignore[assignment]
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when dependency missing
    async_playwright = None
    PlaywrightTimeoutError = builtins.TimeoutError
    PLAYWRIGHT_AVAILABLE = False

try:  # pragma: no cover - import guard for undetected runtime
    from rebrowser_playwright.async_api import (
        async_playwright as undetected_playwright,  # type: ignore[assignment]
    )

    REBROWSER_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when dependency missing
    undetected_playwright = None
    REBROWSER_AVAILABLE = False

try:  # pragma: no cover - import guard
    import warnings

    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="fake_http_header"
    )
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="importlib.resources._legacy"
    )
    from playwright_stealth import stealth_async

    STEALTH_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when dependency missing
    stealth_async = None
    STEALTH_AVAILABLE = False


@dataclass
class _RuntimeBundle:
    """Bundle storing a Playwright runtime and launched browser."""

    name: str
    playwright: Any
    browser: Browser
    use_undetected: bool


class CapMonsterSolver:
    """Minimal CapMonster Cloud client implemented with httpx."""

    _TASK_TYPES = {
        "recaptcha_v2": "NoCaptchaTaskProxyless",
        "hcaptcha": "HCaptchaTaskProxyless",
        "turnstile": "TurnstileTaskProxyless",
    }

    def __init__(self, settings: PlaywrightCaptchaSettings) -> None:
        self._settings = settings
        if settings.captcha_type not in self._TASK_TYPES:
            msg = f"Unsupported captcha type: {settings.captcha_type}"
            raise CrawlServiceError(msg)

    async def solve(self, url: str, site_key: str) -> str:
        """Solve captcha and return the response token."""
        task_payload = {
            "clientKey": self._settings.api_key,
            "task": {
                "type": self._TASK_TYPES[self._settings.captcha_type],
                "websiteURL": url,
                "websiteKey": site_key,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            create_resp = await client.post(
                "https://api.capmonster.cloud/createTask", json=task_payload
            )
            create_resp.raise_for_status()
            data = create_resp.json()
            if data.get("errorId") != 0:
                msg = data.get("errorDescription", "Unknown CapMonster error")
                raise CrawlServiceError(msg)
            task_id = data.get("taskId")
            if not task_id:
                raise CrawlServiceError("CapMonster did not return a taskId")

            poll_payload = {"clientKey": self._settings.api_key, "taskId": task_id}
            for _ in range(30):  # ~30 seconds max
                await asyncio.sleep(1.0)
                poll_resp = await client.post(
                    "https://api.capmonster.cloud/getTaskResult", json=poll_payload
                )
                poll_resp.raise_for_status()
                result = poll_resp.json()
                if result.get("errorId") not in (None, 0):
                    msg = result.get("errorDescription", "CapMonster polling error")
                    raise CrawlServiceError(msg)
                if result.get("status") == "ready":
                    solution = result.get("solution", {})
                    token = (
                        solution.get("gRecaptchaResponse")
                        or solution.get("token")
                        or solution.get("captcha_token")
                    )
                    if token:
                        return str(token)
                    raise CrawlServiceError("CapMonster returned an empty solution")
            raise CrawlServiceError("CapMonster solving timed out")


class PlaywrightAdapter(BaseService):
    """Adapter that exposes high-level Playwright automation primitives."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, config: PlaywrightConfig, enable_stealth: bool | None = None
    ) -> None:
        """Initialize the adapter.

        Args:
            config: Playwright configuration values.
            enable_stealth: Optional override for applying tf_playwright_stealth.
        """
        super().__init__(None)
        self.config = config
        self.enable_stealth = (
            config.enable_stealth if enable_stealth is None else enable_stealth
        )
        self._available = PLAYWRIGHT_AVAILABLE
        self._initialized = False
        self._stealth_warned = False
        self._runtimes: dict[str, _RuntimeBundle] = {}
        self._playwright_handles: list[Any] = []
        self._captcha_solvers: dict[str, CapMonsterSolver] = {}

        if not self._available:
            logger.warning("Playwright is not installed; adapter will remain disabled")

    async def initialize(self) -> None:
        """Start the Playwright runtime(s) required by configured tiers."""
        if not self._available:
            msg = "Playwright not available - install playwright package"
            raise CrawlServiceError(msg)
        if self._initialized:
            return

        # Baseline runtime always available
        await self._ensure_runtime(use_undetected=False)

        # Start undetected runtime only if requested in tiers
        if any(tier.use_undetected_browser for tier in self.config.tiers):
            await self._ensure_runtime(use_undetected=True)

        self._initialized = True
        logger.info("Playwright adapter initialized (%s runtimes)", len(self._runtimes))

    async def cleanup(self) -> None:
        """Dispose all Playwright browsers and runtimes."""
        for runtime in list(self._runtimes.values()):
            with suppress(Exception):  # pragma: no cover - cleanup path
                await runtime.browser.close()
        self._runtimes.clear()

        for handle in list(self._playwright_handles):
            with suppress(Exception):  # pragma: no cover - cleanup path
                await handle.stop()
        self._playwright_handles.clear()

        if self._initialized:
            self._initialized = False
            logger.info("Playwright adapter cleaned up")

    async def scrape(
        self,
        url: str,
        actions: list[dict[str, Any]] | None = None,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Navigate to ``url`` and execute optional actions with tier escalation."""
        if not self._available:
            msg = "Playwright not available"
            raise CrawlServiceError(msg)
        if not self._initialized:
            msg = "Adapter not initialized"
            raise CrawlServiceError(msg)

        attempts: list[dict[str, Any]] = []
        for tier in self.config.tiers:
            runtime = await self._ensure_runtime(tier.use_undetected_browser)
            result = await self._scrape_with_runtime(
                runtime=runtime,
                tier=tier,
                url=url,
                actions=actions or [],
                timeout=timeout or self.config.timeout,
            )
            attempts.append(result)
            if result.get("success") and not result["metadata"].get(
                "challenge_detected", False
            ):
                return result

        # Return the most recent attempt when all tiers failed or were challenged.
        return attempts[-1]

    async def _scrape_with_runtime(
        self,
        *,
        runtime: _RuntimeBundle,
        tier: PlaywrightTierConfig,
        url: str,
        actions: list[dict[str, Any]],
        timeout: int,
    ) -> dict[str, Any]:
        # pylint: disable=too-many-arguments
        """Execute the scraping flow for a single tier/runtime."""
        # pylint: disable=too-many-locals
        start_time = time.perf_counter()
        attempt_errors: list[str] = []

        for attempt in range(1, tier.max_attempts + 1):
            context: BrowserContext | None = None
            page: Page | None = None
            try:
                context = await runtime.browser.new_context(
                    **self._build_context_options(tier)
                )
                page = await context.new_page()

                if tier.enable_stealth and self._should_apply_stealth():
                    await self._apply_stealth(page)

                response: Response | None = await page.goto(
                    url, wait_until="networkidle", timeout=timeout
                )
                status = int(response.status) if response is not None else None

                challenge_detected = await self._detect_challenge(page, tier, status)
                challenge_outcome = "none"
                if challenge_detected:
                    challenge_outcome = "detected"
                    if tier.captcha:
                        solved = await self._attempt_captcha(page, url, tier)
                        if solved:
                            challenge_detected = False
                            challenge_outcome = "solved"

                action_results = await self._execute_actions(page, actions)
                text, html = await self._extract_content(page)
                metadata = await self._build_metadata(
                    page,
                    len(actions),
                    sum(1 for result in action_results if result.get("success")),
                    time.perf_counter() - start_time,
                )
                metadata.update(
                    {
                        "tier": tier.name,
                        "runtime": runtime.name,
                        "attempt": attempt,
                        "http_status": status,
                        "challenge_detected": challenge_detected,
                        "challenge_outcome": challenge_outcome,
                    }
                )

                result = {
                    "success": not challenge_detected,
                    "url": page.url,
                    "content": text,
                    "html": html,
                    "title": metadata.get("title", ""),
                    "metadata": metadata,
                    "action_results": action_results,
                }

                if challenge_detected:
                    result["error"] = (
                        "Bot-detection challenge detected; escalation required"
                    )
                elif metadata.get("challenge_outcome") == "solved":
                    result["metadata"]["challenge_outcome"] = "solved"
                return result

            except (ValidationError, CrawlServiceError) as exc:
                attempt_errors.append(str(exc))
            except PlaywrightTimeoutError as exc:
                attempt_errors.append(f"Timed out: {exc}")
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Unexpected Playwright error for %s", url)
                attempt_errors.append(str(exc))
            finally:
                if context is not None:
                    with suppress(Exception):  # noqa: BLE001
                        await context.close()

        duration = time.perf_counter() - start_time
        return self._build_error_result(
            message="; ".join(attempt_errors) or "Scrape failed",
            url=url,
            tier=tier,
            runtime=runtime,
            duration=duration,
        )

    async def _ensure_runtime(self, use_undetected: bool) -> _RuntimeBundle:
        """Return a runtime bundle, launching one if required."""
        """Return a runtime bundle, launching one if required."""
        key = "undetected" if use_undetected else "baseline"
        if key in self._runtimes:
            return self._runtimes[key]
        runtime = await self._start_runtime(use_undetected)
        self._runtimes[key] = runtime
        return runtime

    async def _start_runtime(self, use_undetected: bool) -> _RuntimeBundle:
        if use_undetected and not REBROWSER_AVAILABLE:
            msg = "rebrowser-playwright missing; install it to enable undetected tiers"
            raise CrawlServiceError(msg)

        factory = undetected_playwright if use_undetected else async_playwright
        playwright_handle = await factory().start()  # type: ignore[call-arg]
        launcher = getattr(playwright_handle, self.config.browser, None)
        if launcher is None:
            await playwright_handle.stop()
            msg = f"Unsupported browser type: {self.config.browser}"
            raise CrawlServiceError(msg)

        browser = await launcher.launch(headless=self.config.headless)
        self._playwright_handles.append(playwright_handle)
        runtime = _RuntimeBundle(
            name="undetected" if use_undetected else "baseline",
            playwright=playwright_handle,
            browser=browser,
            use_undetected=use_undetected,
        )
        logger.info(
            "Started %s Playwright runtime for browser %s",
            runtime.name,
            self.config.browser,
        )
        return runtime

    def _build_context_options(self, tier: PlaywrightTierConfig) -> dict[str, Any]:
        """Translate configuration into Playwright context options."""
        viewport_cfg = self.config.viewport or {}
        viewport: dict[str, Any] = {}
        if "width" in viewport_cfg and "height" in viewport_cfg:
            viewport = {
                "width": int(viewport_cfg["width"]),
                "height": int(viewport_cfg["height"]),
            }
            if "device_scale_factor" in viewport_cfg:
                viewport["device_scale_factor"] = float(
                    viewport_cfg["device_scale_factor"]
                )
            if viewport_cfg.get("is_mobile"):
                viewport["is_mobile"] = bool(viewport_cfg.get("is_mobile"))

        context_options: dict[str, Any] = {
            "ignore_https_errors": True,
            "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
        }

        if viewport:
            context_options["viewport"] = viewport

        if self.config.user_agent:
            context_options["user_agent"] = self.config.user_agent

        if tier.proxy:
            context_options["proxy"] = tier.proxy.to_playwright_dict()

        return context_options

    def _should_apply_stealth(self) -> bool:
        return self.enable_stealth

    async def _apply_stealth(self, page: Page) -> None:
        """Apply stealth transformations if enabled."""
        if not self._should_apply_stealth():
            return
        if not STEALTH_AVAILABLE:
            if not self._stealth_warned:
                logger.warning(
                    "playwright_stealth not installed; proceeding without stealth"
                )
                self._stealth_warned = True
            return

        try:
            await stealth_async(page)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - plugin failures should not abort execution
            logger.warning("Failed to apply stealth plugin", exc_info=True)

    async def _detect_challenge(
        self, page: Page, tier: PlaywrightTierConfig, status: int | None
    ) -> bool:
        """Determine whether a bot-detection challenge is present."""
        if status is not None and status in set(tier.challenge_status_codes):
            return True

        lowered = ""
        with suppress(Exception):
            text = await page.inner_text("body")
            lowered = text.lower()
        return any(keyword.lower() in lowered for keyword in tier.challenge_keywords)

    async def _attempt_captcha(
        self, page: Page, url: str, tier: PlaywrightTierConfig
    ) -> bool:
        """Attempt to solve a captcha when configuration allows it."""
        if not tier.captcha:
            return False
        solver = self._get_captcha_solver(tier)
        if solver is None:
            return False

        try:
            site_key = await self._extract_site_key(page, tier.captcha)
            if not site_key:
                logger.warning("Unable to extract site key for captcha on %s", url)
                return False
            token = await solver.solve(url, site_key)
            script = f"""
            const el = document.querySelector(
                {json.dumps(tier.captcha.response_input_selector)}
            );
            if (el) {{
                el.value = {json.dumps(token)};
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
            """
            await page.evaluate(script)
            await asyncio.sleep(1.0)
            return True
        except Exception:  # noqa: BLE001 - captcha solving is best effort
            logger.exception("Failed to solve captcha for %s", url)
            return False

    async def _extract_site_key(
        self, page: Page, settings: PlaywrightCaptchaSettings
    ) -> str | None:
        """Extract site key from captcha iframe or data attributes."""
        script = """
            selector => {
                const frame = document.querySelector(selector);
                if (!frame) return null;
                const attrKey = frame.getAttribute('data-sitekey');
                if (attrKey) return attrKey;
                const src = frame.getAttribute('src') || '';
                try {
                    const url = new URL(src, window.location.origin);
                    return url.searchParams.get('k') || url.searchParams.get('sitekey');
                } catch (_) {
                    return null;
                }
            }
        """
        with suppress(Exception):
            value = await page.evaluate(script, settings.iframe_selector)
            if value:
                return str(value)
        return None

    def _get_captcha_solver(
        self, tier: PlaywrightTierConfig
    ) -> CapMonsterSolver | None:
        if not tier.captcha:
            return None
        if tier.name not in self._captcha_solvers:
            self._captcha_solvers[tier.name] = CapMonsterSolver(tier.captcha)
        return self._captcha_solvers[tier.name]

    async def _execute_actions(
        self, page: Page, actions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate and execute browser actions sequentially."""
        if not actions:
            return []

        validated_actions = validate_actions(actions)
        results: list[dict[str, Any]] = []

        for index, action in enumerate(validated_actions):
            start = time.perf_counter()
            try:
                output = await self._run_action(page, action)
                results.append(
                    {
                        "action_index": index,
                        "action_type": action.type,
                        "success": True,
                        "execution_time_ms": (time.perf_counter() - start) * 1000,
                        **({"output": output} if output is not None else {}),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - record failure, continue
                if isinstance(exc, asyncio.CancelledError):  # pragma: no cover - safety
                    raise
                results.append(
                    {
                        "action_index": index,
                        "action_type": action.type,
                        "success": False,
                        "error": str(exc),
                        "execution_time_ms": (time.perf_counter() - start) * 1000,
                    }
                )
        return results

    async def _run_action(self, page: Page, action: Any) -> Any:
        """Execute a single validated action on the page."""
        action_type = action.type

        async def wait_handler() -> None:
            await page.wait_for_timeout(action.timeout)

        async def wait_for_selector_handler() -> Any:
            return await page.wait_for_selector(action.selector, timeout=action.timeout)

        async def wait_for_load_state_handler() -> Any:
            return await page.wait_for_load_state(action.state)

        async def scroll_handler() -> Any:
            if action.direction == "top":
                return await page.evaluate("window.scrollTo(0, 0)")
            if action.direction == "bottom":
                return await page.evaluate(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )
            return await page.evaluate("window.scrollTo(0, arguments[0])", action.y)

        async def press_handler() -> Any:
            if action.selector:
                await page.focus(action.selector)
            return await page.keyboard.press(action.key)

        async_handlers = {
            "click": lambda: page.click(action.selector),
            "fill": lambda: page.fill(action.selector, action.text),
            "type": lambda: page.type(action.selector, action.text),
            "evaluate": lambda: page.evaluate(action.script),
            "hover": lambda: page.hover(action.selector),
            "select": lambda: page.select_option(action.selector, action.value),
            "wait": wait_handler,
            "wait_for_selector": wait_for_selector_handler,
            "wait_for_load_state": wait_for_load_state_handler,
            "scroll": scroll_handler,
            "screenshot": lambda: page.screenshot(
                path=action.path or None, full_page=action.full_page
            ),
            "press": press_handler,
            "drag_and_drop": lambda: page.drag_and_drop(action.source, action.target),
        }

        handler = async_handlers.get(action_type)
        if handler is None:
            msg = f"Unsupported action type: {action_type}"
            raise CrawlServiceError(msg)

        return await handler()

    async def _extract_content(self, page: Page) -> tuple[str, str]:
        """Extract text and HTML from the visited page."""
        try:
            text = await page.inner_text("body")
        except Exception:  # noqa: BLE001 - fallback to empty string
            text = ""

        try:
            html = await page.content()
        except Exception:  # noqa: BLE001 - fallback to empty string
            html = ""

        return text, html

    async def _build_metadata(
        self,
        page: Page,
        action_count: int,
        success_count: int,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Compose metadata describing the scraping session."""
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001 - best effort only
            title = ""

        metadata: dict[str, Any] = {
            "title": title,
            "extraction_method": "playwright",
            "actions_executed": action_count,
            "successful_actions": success_count,
            "processing_time_ms": duration_seconds * 1000,
            "browser_type": self.config.browser,
            "viewport": self.config.viewport,
            "user_agent": self.config.user_agent,
            "stealth_enabled": self._should_apply_stealth() and STEALTH_AVAILABLE,
        }

        return metadata

    def _build_error_result(
        self,
        *,
        message: str,
        url: str,
        tier: PlaywrightTierConfig,
        runtime: _RuntimeBundle,
        duration: float,
    ) -> dict[str, Any]:
        # pylint: disable=too-many-arguments
        """Return a standardized error payload."""
        return {
            "success": False,
            "url": url,
            "error": message,
            "content": "",
            "metadata": {
                "extraction_method": "playwright",
                "processing_time_ms": duration * 1000,
                "browser_type": self.config.browser,
                "tier": tier.name,
                "runtime": runtime.name,
                "challenge_detected": False,
                "challenge_outcome": "none",
                "stealth_enabled": self._should_apply_stealth() and STEALTH_AVAILABLE,
            },
            "action_results": [],
        }
