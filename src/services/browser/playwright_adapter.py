import typing

"""Playwright adapter for maximum control browser automation."""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from src.config import PlaywrightConfig

from ..base import BaseService
from ..errors import CrawlServiceError
from .action_schemas import validate_actions
from .anti_detection import EnhancedAntiDetection

logger = logging.getLogger(__name__)

# Try to import Playwright, handle gracefully if not available
try:
    from playwright.async_api import Browser
    from playwright.async_api import BrowserContext
    from playwright.async_api import Page
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None


class PlaywrightAdapter(BaseService):
    """Direct Playwright automation for maximum control.

    Provides full programmatic control over browser automation.
    Best for authentication scenarios and complex scripted interactions.
    """

    def __init__(self, config: PlaywrightConfig, enable_anti_detection: bool = True):
        """Initialize Playwright adapter.

        Args:
            config: Playwright configuration model
            enable_anti_detection: Whether to enable enhanced anti-detection features
        """
        super().__init__(config)
        self.config = config
        self.logger = logger
        self.enable_anti_detection = enable_anti_detection

        if not PLAYWRIGHT_AVAILABLE:
            self.logger.warning("Playwright not available - adapter will be disabled")
            self._available = False
            return

        self._available = True

        # Browser management
        self._playwright: Any | None = None
        self._browser: Any | None = None  # Browser instance when available
        self._initialized = False

        # Enhanced anti-detection system
        self.anti_detection = EnhancedAntiDetection() if enable_anti_detection else None

    async def initialize(self) -> None:
        """Initialize Playwright browser."""
        if not self._available:
            raise CrawlServiceError(
                "Playwright not available - please install playwright package"
            )

        if self._initialized:
            return

        try:
            self._playwright = await async_playwright().start()

            # Get browser launcher
            browser_launcher = getattr(self._playwright, self.config.browser)

            # Determine browser arguments
            if self.anti_detection:
                # Use enhanced anti-detection args
                stealth_config = self.anti_detection.get_stealth_config()
                browser_args = stealth_config.extra_args
                self.logger.info("Using enhanced anti-detection browser configuration")
            else:
                # Use basic args
                browser_args = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ]

            # Launch browser with optimized settings
            self._browser = await browser_launcher.launch(
                headless=self.config.headless,
                args=browser_args,
            )

            self._initialized = True
            anti_detection_status = "enabled" if self.anti_detection else "disabled"
            self.logger.info(
                f"Playwright adapter initialized with {self.config.browser} "
                f"(anti-detection: {anti_detection_status})"
            )

        except Exception as e:
            raise CrawlServiceError(f"Failed to initialize Playwright: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup Playwright resources."""
        try:
            if self._browser:
                await self._browser.close()
                self._browser = None

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

            self._initialized = False
            self.logger.info("Playwright adapter cleaned up")

        except Exception as e:
            self.logger.exception(f"Error cleaning up Playwright: {e}")

    async def scrape(
        self,
        url: str,
        actions: list[dict],
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Scrape with direct Playwright control.

        Args:
            url: URL to scrape
            actions: List of actions to perform
            timeout: Timeout in milliseconds

        Returns:
            Scraping result with standardized format
        """
        if not self._available:
            raise CrawlServiceError("Playwright not available")

        if not self._initialized:
            raise CrawlServiceError("Adapter not initialized")

        start_time = time.time()
        context: Any | None = None
        page: Any | None = None

        # Setup anti-detection configuration
        domain = urlparse(url).netloc
        site_profile, stealth_config = await self._setup_anti_detection(domain)

        try:
            # Create browser context and page
            context, page = await self._create_browser_context_and_page(
                site_profile, stealth_config, timeout
            )

            # Navigate and execute actions
            action_results = await self._navigate_and_execute_actions(
                page, url, actions, timeout, site_profile
            )

            # Extract content and build result
            result = await self._build_success_result(
                page, actions, action_results, start_time, site_profile, stealth_config
            )

            return result

        except Exception as e:
            return await self._build_error_result(e, url, start_time, site_profile)

        finally:
            await self._cleanup_context(context)

    async def _setup_anti_detection(self, domain: str) -> tuple[str, Any]:
        """Setup anti-detection configuration for domain."""
        site_profile = "default"
        stealth_config = None

        if self.anti_detection:
            site_profile = self.anti_detection.get_recommended_strategy(domain)
            stealth_config = self.anti_detection.get_stealth_config(site_profile)

            # Add pre-scrape delay for anti-detection
            delay = await self.anti_detection.get_human_like_delay(site_profile)
            await asyncio.sleep(delay)

        return site_profile, stealth_config

    async def _create_browser_context_and_page(
        self, site_profile: str, stealth_config: Any, timeout: int
    ) -> tuple[Any, Any]:
        """Create browser context and page with appropriate configuration."""
        # Create browser context with enhanced settings
        if self.anti_detection and stealth_config:
            context = await self._browser.new_context(
                viewport={
                    "width": stealth_config.viewport.width,
                    "height": stealth_config.viewport.height,
                },
                user_agent=stealth_config.user_agent,
                ignore_https_errors=True,
                extra_http_headers=stealth_config.headers,
                device_scale_factor=stealth_config.viewport.device_scale_factor,
            )
            self.logger.debug(f"Using anti-detection profile: {site_profile}")
        else:
            # Use original configuration
            context = await self._browser.new_context(
                viewport=self.config.viewport,
                user_agent=self.config.user_agent,
                ignore_https_errors=True,
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

        # Create new page
        page = await context.new_page()

        # Set up event listeners for debugging
        page.on("console", lambda msg: self.logger.debug(f"Console: {msg.text}"))
        page.on("pageerror", lambda error: self.logger.warning(f"Page error: {error}"))

        return context, page

    async def _navigate_and_execute_actions(
        self, page: Any, url: str, actions: list[dict], timeout: int, site_profile: str
    ) -> list[dict[str, Any]]:
        """Navigate to URL and execute all actions."""
        # Inject stealth JavaScript patterns for enhanced anti-detection
        if self.anti_detection:
            stealth_config = self.anti_detection.get_stealth_config(site_profile)
            await self._inject_stealth_scripts(page, stealth_config)

        # Navigate to URL
        await page.goto(url, wait_until="networkidle", timeout=timeout)

        # Add post-navigation delay for anti-detection
        if self.anti_detection:
            post_nav_delay = await self.anti_detection.get_human_like_delay(
                site_profile
            )
            await asyncio.sleep(post_nav_delay * 0.5)  # Shorter delay after navigation

        # Validate actions before executing
        try:
            validated_actions = validate_actions(actions)
        except ValidationError as e:
            raise CrawlServiceError(f"Invalid actions: {e}") from e

        # Execute custom actions
        action_results = []
        for i, action in enumerate(validated_actions):
            try:
                result = await self._execute_action(page, action, i)
                action_results.append(result)
            except Exception as e:
                self.logger.warning(f"Action {i} failed: {e}")
                action_results.append(
                    {
                        "action_index": i,
                        "success": False,
                        "error": str(e),
                    }
                )

        return action_results

    async def _build_success_result(
        self,
        page: Any,
        actions: list[dict],
        action_results: list[dict[str, Any]],
        start_time: float,
        site_profile: str,
        stealth_config: Any,
    ) -> dict[str, Any]:
        """Build successful scraping result."""
        # Extract content
        content_data = await self._extract_content(page)
        metadata = await self._extract_metadata(page)

        processing_time = (time.time() - start_time) * 1000

        # Record successful attempt for anti-detection monitoring
        if self.anti_detection:
            self.anti_detection.record_attempt(True, site_profile)

        result = {
            "success": True,
            "url": page.url,  # May have changed due to redirects
            "content": content_data["text"],
            "html": content_data["html"],
            "title": metadata.get("title", ""),
            "metadata": {
                **metadata,
                "extraction_method": "playwright",
                "actions_executed": len(actions),
                "successful_actions": sum(
                    1 for r in action_results if r.get("success", False)
                ),
                "processing_time_ms": processing_time,
                "browser_type": self.config.browser,
                "viewport": stealth_config.viewport.model_dump()
                if self.anti_detection and stealth_config
                else self.config.viewport,
                "anti_detection_enabled": self.enable_anti_detection,
                "site_profile": site_profile if self.anti_detection else None,
                "user_agent": stealth_config.user_agent
                if self.anti_detection and stealth_config
                else self.config.user_agent,
            },
            "action_results": action_results,
            "performance": await self._get_performance_metrics(page),
        }

        # Add anti-detection success metrics if enabled
        if self.anti_detection:
            result["metadata"]["anti_detection_metrics"] = (
                self.anti_detection.get_success_metrics()
            )

        return result

    async def _build_error_result(
        self, error: Exception, url: str, start_time: float, site_profile: str
    ) -> dict[str, Any]:
        """Build error result for failed scraping."""
        processing_time = (time.time() - start_time) * 1000
        self.logger.error(f"Playwright error for {url}: {error}")

        # Record failed attempt for anti-detection monitoring
        if self.anti_detection:
            self.anti_detection.record_attempt(False, site_profile)

        result = {
            "success": False,
            "url": url,
            "error": str(error),
            "content": "",
            "metadata": {
                "extraction_method": "playwright",
                "processing_time_ms": processing_time,
                "browser_type": self.config.browser,
                "anti_detection_enabled": self.enable_anti_detection,
                "site_profile": site_profile if self.anti_detection else None,
            },
        }

        # Add anti-detection metrics even on failure
        if self.anti_detection:
            result["metadata"]["anti_detection_metrics"] = (
                self.anti_detection.get_success_metrics()
            )

        return result

    async def _cleanup_context(self, context: Any | None) -> None:
        """Clean up browser context."""
        if context:
            try:
                await context.close()
            except Exception as e:
                self.logger.warning(f"Failed to close context: {e}")

    async def _inject_stealth_scripts(self, page: Any, stealth_config: Any) -> None:
        """Inject JavaScript patterns to avoid detection.

        Args:
            page: Playwright page instance
            stealth_config: Browser stealth configuration
        """
        try:
            # Basic stealth script to hide automation indicators
            stealth_script = """
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });

            // Override the chrome property to mimic a regular Chrome browser
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };

            // Override the permissions property
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            // Override the plugins property to mimic a real browser
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });

            // Override the languages property
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });

            // Add realistic screen properties
            Object.defineProperty(screen, 'availWidth', {
                get: () => screen.width,
            });
            Object.defineProperty(screen, 'availHeight', {
                get: () => screen.height - 40, // Account for taskbar
            });
            """

            # Advanced canvas fingerprinting protection
            if (
                hasattr(stealth_config, "canvas_fingerprint_protection")
                and stealth_config.canvas_fingerprint_protection
            ):
                canvas_protection_script = """
                // Canvas fingerprinting protection
                const getContext = HTMLCanvasElement.prototype.getContext;
                HTMLCanvasElement.prototype.getContext = function(contextType, contextAttributes) {
                    const context = getContext.call(this, contextType, contextAttributes);
                    if (contextType === '2d') {
                        const getImageData = context.getImageData;
                        context.getImageData = function(sx, sy, sw, sh) {
                            const imageData = getImageData.call(this, sx, sy, sw, sh);
                            for (let i = 0; i < imageData.data.length; i += 4) {
                                imageData.data[i] += Math.floor(Math.random() * 3) - 1;
                                imageData.data[i + 1] += Math.floor(Math.random() * 3) - 1;
                                imageData.data[i + 2] += Math.floor(Math.random() * 3) - 1;
                            }
                            return imageData;
                        };
                    }
                    return context;
                };
                """
                stealth_script += canvas_protection_script

            # WebGL fingerprinting protection
            if (
                hasattr(stealth_config, "webgl_fingerprint_protection")
                and stealth_config.webgl_fingerprint_protection
            ):
                webgl_protection_script = """
                // WebGL fingerprinting protection
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) {
                        return 'Intel Inc.';
                    }
                    if (parameter === 37446) {
                        return 'Intel(R) Iris(TM) Graphics 6100';
                    }
                    return getParameter.call(this, parameter);
                };
                """
                stealth_script += webgl_protection_script

            # Inject the stealth script
            await page.add_init_script(stealth_script)
            self.logger.debug("Stealth JavaScript patterns injected successfully")

        except Exception as e:
            self.logger.warning(f"Failed to inject stealth scripts: {e}")

    async def _execute_action(
        self, page: Any, action: Any, index: int
    ) -> dict[str, Any]:
        """Execute a single action.

        Args:
            page: Playwright page instance
            action: Validated action model
            index: Action index for tracking

        Returns:
            Action result
        """
        action_type = action.type
        start_time = time.time()

        try:
            result = await self._perform_action(page, action_type, action, index)
            if result:  # Some actions return custom results
                return result

            return self._create_success_result(index, action_type, start_time)

        except Exception as e:
            return self._create_error_result(index, action_type, start_time, str(e))

    async def _perform_action(
        self, page: Any, action_type: str, action: dict, index: int
    ) -> dict[str, Any] | None:
        """Perform the specific action based on type."""
        # Input actions
        if action_type in ("click", "type", "fill", "hover", "select"):
            await self._execute_input_action(page, action_type, action)

        # Wait actions
        elif action_type in ("wait", "wait_for_selector", "wait_for_load_state"):
            await self._execute_wait_action(page, action_type, action)

        # Navigation actions
        elif action_type == "scroll":
            await self._execute_scroll_action(page, action)

        # Actions with custom results
        elif action_type == "screenshot":
            return await self._execute_screenshot_action(page, action, index)
        elif action_type == "evaluate":
            return await self._execute_evaluate_action(page, action, index)

        # Interaction actions
        elif action_type == "press":
            await self._execute_press_action(page, action)
        elif action_type == "drag_and_drop":
            await self._execute_drag_drop_action(page, action)

        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return None  # No custom result

    async def _execute_input_action(self, page: Any, action_type: str, action: Any):
        """Execute input-related actions."""
        selector = action.selector
        await page.wait_for_selector(selector, timeout=5000)

        if action_type == "click":
            await page.click(selector)
        elif action_type == "type":
            await page.type(selector, action.text)
        elif action_type == "fill":
            await page.fill(selector, action.text)
        elif action_type == "hover":
            await page.hover(selector)
        elif action_type == "select":
            await page.select_option(selector, action.value)

    async def _execute_wait_action(self, page: Any, action_type: str, action: Any):
        """Execute wait-related actions."""
        if action_type == "wait":
            await page.wait_for_timeout(action.timeout)
        elif action_type == "wait_for_selector":
            await page.wait_for_selector(action.selector, timeout=action.timeout)
        elif action_type == "wait_for_load_state":
            await page.wait_for_load_state(action.state)

    async def _execute_scroll_action(self, page: Any, action: Any):
        """Execute scroll action."""
        if action.direction == "bottom":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif action.direction == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        else:
            await page.evaluate(f"window.scrollTo(0, {action.y})")

    async def _execute_screenshot_action(
        self, page: Any, action: Any, index: int
    ) -> dict[str, Any]:
        """Execute screenshot action with custom result."""
        path = action.path if action.path else f"screenshot_{index}.png"
        screenshot = await page.screenshot(path=path, full_page=action.full_page)

        return {
            "action_index": index,
            "action_type": "screenshot",
            "success": True,
            "screenshot_path": path,
            "screenshot_size": len(screenshot),
        }

    async def _execute_evaluate_action(
        self, page: Any, action: Any, index: int
    ) -> dict[str, Any]:
        """Execute evaluate action with custom result."""
        result = await page.evaluate(action.script)

        return {
            "action_index": index,
            "action_type": "evaluate",
            "success": True,
            "result": result,
        }

    async def _execute_press_action(self, page: Any, action: Any):
        """Execute press action."""
        if action.selector:
            await page.press(action.selector, action.key)
        else:
            await page.keyboard.press(action.key)

    async def _execute_drag_drop_action(self, page: Any, action: Any):
        """Execute drag and drop action."""
        await page.drag_and_drop(action.source, action.target)

    def _create_success_result(
        self, index: int, action_type: str, start_time: float
    ) -> dict[str, Any]:
        """Create success result dictionary."""
        return {
            "action_index": index,
            "action_type": action_type,
            "success": True,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    def _create_error_result(
        self, index: int, action_type: str, start_time: float, error: str
    ) -> dict[str, Any]:
        """Create error result dictionary."""
        return {
            "action_index": index,
            "action_type": action_type,
            "success": False,
            "error": error,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    async def _extract_content(self, page: Any) -> dict[str, str]:
        """Extract content from page.

        Args:
            page: Playwright page instance

        Returns:
            Content dictionary with text and HTML
        """
        # Try multiple content selectors in order of preference
        content_selectors = [
            "main",
            "article",
            ".content",
            ".documentation",
            "#content",
            ".markdown-body",
            ".doc-content",
            "body",
        ]

        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    html = await element.inner_html()

                    # Check if we got meaningful content
                    if text and len(text.strip()) > 50:
                        return {"text": text, "html": html}

            except Exception as e:
                self.logger.debug(f"Failed to extract with selector {selector}: {e}")
                continue

        # Fallback to full body
        try:
            return {
                "text": await page.inner_text("body"),
                "html": await page.inner_html("body"),
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract body content: {e}")
            return {"text": "", "html": ""}

    async def _extract_metadata(self, page: Any) -> dict[str, Any]:
        """Extract page metadata.

        Args:
            page: Playwright page instance

        Returns:
            Metadata dictionary
        """
        try:
            metadata = await page.evaluate("""
            () => {
                const getMeta = (name) => {
                    const meta = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                    return meta ? meta.content : null;
                };

                const getLinks = () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        text: link.textContent?.trim() || '',
                        href: link.href,
                        title: link.title || null,
                    })).filter(link => link.href && link.text);
                };

                return {
                    title: document.title,
                    description: getMeta('description'),
                    author: getMeta('author'),
                    keywords: getMeta('keywords'),
                    ogTitle: getMeta('og:title'),
                    ogDescription: getMeta('og:description'),
                    ogImage: getMeta('og:image'),
                    ogType: getMeta('og:type'),
                    twitterCard: getMeta('twitter:card'),
                    twitterTitle: getMeta('twitter:title'),
                    canonical: document.querySelector('link[rel="canonical"]')?.href || null,
                    lastModified: document.lastModified,
                    language: document.documentElement.lang || null,
                    links: getLinks(),
                    headings: Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                        level: h.tagName.toLowerCase(),
                        text: h.textContent?.trim() || '',
                        id: h.id || null,
                    })),
                };
            }
            """)

            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")
            return {
                "title": await page.title() if page else "",
                "description": None,
                "links": [],
                "headings": [],
            }

    async def _get_performance_metrics(self, page: Any) -> dict[str, Any]:
        """Get page performance metrics.

        Args:
            page: Playwright page instance

        Returns:
            Performance metrics dictionary
        """
        try:
            metrics = await page.evaluate("""
            () => {
                if (!window.performance) return {};

                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');

                return {
                    loadTime: navigation?.loadEventEnd - navigation?.loadEventStart || 0,
                    domContentLoadedTime: navigation?.domContentLoadedEventEnd - navigation?.domContentLoadedEventStart || 0,
                    responseTime: navigation?.responseEnd - navigation?.responseStart || 0,
                    firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                    firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                    resourceCount: performance.getEntriesByType('resource').length,
                };
            }
            """)

            return metrics

        except Exception as e:
            self.logger.debug(f"Failed to get performance metrics: {e}")
            return {}

    def get_capabilities(self) -> dict[str, Any]:
        """Get adapter capabilities and limitations.

        Returns:
            Capabilities dictionary
        """
        return {
            "name": "playwright",
            "description": "Full programmatic browser control with maximum flexibility",
            "advantages": [
                "Complete control over browser",
                "Excellent for authentication",
                "Complex interaction support",
                "Performance metrics",
                "Screenshot capabilities",
                "Network interception",
            ],
            "limitations": [
                "Slower than simpler tools",
                "More resource intensive",
                "Requires explicit programming",
                "No AI assistance",
            ],
            "best_for": [
                "Authentication flows",
                "Complex interactions",
                "Testing scenarios",
                "Performance analysis",
                "Screenshot automation",
                "Network monitoring",
            ],
            "performance": {
                "avg_speed": "3.5s per page",
                "concurrency": "3-10 pages",
                "success_rate": "99% with proper actions",
            },
            "javascript_support": "complete",
            "dynamic_content": "excellent",
            "authentication": "excellent",
            "cost": 0,
            "available": self._available,
            "browsers": ["chromium", "firefox", "webkit"],
        }

    async def health_check(self) -> dict[str, Any]:
        """Check adapter health and availability.

        Returns:
            Health status dictionary
        """
        if not self._available:
            return {
                "healthy": False,
                "status": "unavailable",
                "message": "Playwright package not installed",
                "available": False,
            }

        try:
            if not self._initialized:
                return {
                    "healthy": False,
                    "status": "not_initialized",
                    "message": "Adapter not initialized",
                    "available": True,
                }

            # Test with a simple action sequence
            test_url = "https://httpbin.org/html"
            test_actions = [
                {"type": "wait_for_load_state", "state": "networkidle"},
                {"type": "evaluate", "script": "document.title"},
            ]

            start_time = time.time()

            result = await asyncio.wait_for(
                self.scrape(test_url, test_actions), timeout=15.0
            )

            response_time = time.time() - start_time

            return {
                "healthy": result.get("success", False),
                "status": "operational" if result.get("success") else "degraded",
                "message": "Health check passed"
                if result.get("success")
                else result.get("error", "Health check failed"),
                "response_time_ms": response_time * 1000,
                "test_url": test_url,
                "available": True,
                "browser_type": self.config.browser,
                "capabilities": self.get_capabilities(),
            }

        except TimeoutError:
            return {
                "healthy": False,
                "status": "timeout",
                "message": "Health check timed out",
                "response_time_ms": 15000,
                "available": True,
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "message": f"Health check failed: {e}",
                "available": True,
            }

    async def test_complex_interaction(
        self, test_url: str = "https://example.com"
    ) -> dict[str, Any]:
        """Test complex interaction capabilities.

        Args:
            test_url: URL to test against

        Returns:
            Test results
        """
        if not self._available or not self._initialized:
            return {
                "success": False,
                "error": "Adapter not available or initialized",
            }

        test_actions = [
            {"type": "wait_for_load_state", "state": "networkidle"},
            {"type": "evaluate", "script": "document.title"},
            {"type": "scroll", "direction": "bottom"},
            {"type": "wait", "timeout": 1000},
            {"type": "screenshot", "path": "test_screenshot.png", "full_page": True},
            {"type": "evaluate", "script": "document.links.length"},
        ]

        try:
            start_time = time.time()
            result = await self.scrape(test_url, test_actions)

            return {
                "success": result.get("success", False),
                "test_url": test_url,
                "actions_count": len(test_actions),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "successful_actions": result.get("metadata", {}).get(
                    "successful_actions", 0
                ),
                "content_length": len(result.get("content", "")),
                "performance_metrics": result.get("performance", {}),
                "error": result.get("error") if not result.get("success") else None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "test_url": test_url,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }
