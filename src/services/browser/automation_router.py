"""Intelligent browser automation router with three-tier hierarchy."""

import logging
import time
from typing import Any
from typing import Literal
from urllib.parse import urlparse

from ...config import UnifiedConfig
from ..base import BaseService
from ..errors import CrawlServiceError

logger = logging.getLogger(__name__)


class AutomationRouter(BaseService):
    """Intelligently route scraping tasks to appropriate automation tool.

    Implements a three-tier automation hierarchy:
    1. Crawl4AI (90% of sites) - 4-6x faster, $0 cost
    2. Stagehand (Complex interactions) - AI-powered automation
    3. Playwright (Maximum control) - Full programmatic control
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize automation router with configuration.

        Args:
            config: Unified configuration containing browser automation settings
        """
        super().__init__(config)
        self.config = config
        self.logger = logger

        # Initialize adapters (lazy loading)
        self._adapters: dict[str, Any] = {}
        self._initialized = False

        # Site-specific routing rules
        self.routing_rules = {
            # Sites that need Stagehand (AI-powered interaction)
            "stagehand": [
                "vercel.com",  # Complex React app
                "clerk.com",  # Heavy client-side rendering
                "supabase.com",  # Dynamic documentation
                "netlify.com",  # Interactive demos
                "railway.app",  # Dynamic content
                "planetscale.com",  # Complex UI components
            ],
            # Sites that need Playwright (specific automation)
            "playwright": [
                "github.com",  # Authentication required
                "stackoverflow.com",  # Complex pagination
                "discord.com",  # Heavy JavaScript
                "slack.com",  # Authentication flows
                "app.posthog.com",  # Dashboard interactions
            ],
            # Default to Crawl4AI for everything else
        }

        # Performance metrics tracking
        self.metrics = {
            "crawl4ai": {"success": 0, "failed": 0, "avg_time": 0.0, "total_time": 0.0},
            "stagehand": {
                "success": 0,
                "failed": 0,
                "avg_time": 0.0,
                "total_time": 0.0,
            },
            "playwright": {
                "success": 0,
                "failed": 0,
                "avg_time": 0.0,
                "total_time": 0.0,
            },
        }

    async def initialize(self) -> None:
        """Initialize available automation adapters."""
        if self._initialized:
            return

        # Import adapters dynamically to avoid circular imports
        try:
            from .crawl4ai_adapter import Crawl4AIAdapter

            adapter_config = {
                "max_concurrent": self.config.performance.max_concurrent_requests,
                "headless": True,
                "browser": "chromium",
                "page_timeout": int(self.config.performance.request_timeout * 1000),
            }

            adapter = Crawl4AIAdapter(adapter_config)
            await adapter.initialize()
            self._adapters["crawl4ai"] = adapter
            self.logger.info("Initialized Crawl4AI adapter")

        except Exception as e:
            self.logger.warning(f"Failed to initialize Crawl4AI adapter: {e}")

        # Initialize Stagehand adapter if available
        try:
            from .stagehand_adapter import StagehandAdapter

            adapter_config = {
                "env": "LOCAL",  # Use local LLM
                "headless": True,
                "model": "ollama/llama2",
                "enable_caching": True,
                "debug": False,
            }

            adapter = StagehandAdapter(adapter_config)
            await adapter.initialize()
            self._adapters["stagehand"] = adapter
            self.logger.info("Initialized Stagehand adapter")

        except Exception as e:
            self.logger.warning(f"Failed to initialize Stagehand adapter: {e}")

        # Initialize Playwright adapter
        try:
            from .playwright_adapter import PlaywrightAdapter

            adapter_config = {
                "browser": "chromium",
                "headless": True,
                "viewport": {"width": 1920, "height": 1080},
            }

            adapter = PlaywrightAdapter(adapter_config)
            await adapter.initialize()
            self._adapters["playwright"] = adapter
            self.logger.info("Initialized Playwright adapter")

        except Exception as e:
            self.logger.warning(f"Failed to initialize Playwright adapter: {e}")

        if not self._adapters:
            raise CrawlServiceError("No automation adapters available")

        self._initialized = True
        self.logger.info(
            f"Automation router initialized with {len(self._adapters)} adapters"
        )

    async def cleanup(self) -> None:
        """Cleanup all adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.cleanup()
                self.logger.info(f"Cleaned up {name} adapter")
            except Exception as e:
                self.logger.error(f"Error cleaning up {name} adapter: {e}")

        self._adapters.clear()
        self._initialized = False

    async def scrape(
        self,
        url: str,
        interaction_required: bool = False,
        custom_actions: list[dict] | None = None,
        force_tool: Literal["crawl4ai", "stagehand", "playwright"] | None = None,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Route scraping to appropriate tool based on URL and requirements.

        Args:
            url: URL to scrape
            interaction_required: Whether interaction with page is needed
            custom_actions: List of custom actions to perform
            force_tool: Specific tool to use (overrides selection logic)
            timeout: Timeout in milliseconds

        Returns:
            Scraping result with success status, content, metadata, and provider

        Raises:
            CrawlServiceError: If router not initialized or all tools fail
        """
        if not self._initialized:
            raise CrawlServiceError("Router not initialized")

        # Determine which tool to use
        if force_tool:
            if force_tool not in self._adapters:
                raise CrawlServiceError(f"Forced tool '{force_tool}' not available")
            tool = force_tool
        else:
            tool = self._select_tool(url, interaction_required, custom_actions)

        self.logger.info(f"Using {tool} for {url}")

        # Execute with selected tool
        start_time = time.time()

        try:
            if tool == "crawl4ai":
                result = await self._try_crawl4ai(url, custom_actions, timeout)
            elif tool == "stagehand":
                result = await self._try_stagehand(url, custom_actions, timeout)
            else:  # playwright
                result = await self._try_playwright(url, custom_actions, timeout)

            # Update metrics
            elapsed = time.time() - start_time
            self._update_metrics(tool, True, elapsed)

            result["provider"] = tool
            result["automation_time_ms"] = elapsed * 1000
            return result

        except Exception as e:
            self.logger.error(f"{tool} failed for {url}: {e}")
            elapsed = time.time() - start_time
            self._update_metrics(tool, False, elapsed)

            # Try fallback
            return await self._fallback_scrape(url, tool, custom_actions, timeout)

    def _select_tool(
        self,
        url: str,
        interaction_required: bool,
        custom_actions: list[dict] | None,
    ) -> str:
        """Select appropriate tool based on URL and requirements.

        Args:
            url: URL to analyze
            interaction_required: Whether interaction is needed
            custom_actions: Custom actions to perform

        Returns:
            Tool name to use
        """
        domain = urlparse(url).netloc.lower()

        # Check explicit routing rules
        for tool, domains in self.routing_rules.items():
            if any(d in domain for d in domains):
                if tool in self._adapters:
                    return tool

        # Check if interaction is required
        if interaction_required or custom_actions:
            # Prefer Stagehand for AI-powered interactions
            if "stagehand" in self._adapters:
                return "stagehand"
            # Fallback to Playwright for manual control
            elif "playwright" in self._adapters:
                return "playwright"

        # Check for complex JavaScript patterns in URL
        if any(
            pattern in url.lower()
            for pattern in ["spa", "react", "vue", "angular", "app"]
        ):
            if "stagehand" in self._adapters:
                return "stagehand"

        # Default to fastest option
        if "crawl4ai" in self._adapters:
            return "crawl4ai"
        elif "playwright" in self._adapters:
            return "playwright"
        else:
            return "stagehand"

    async def _try_crawl4ai(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Try scraping with Crawl4AI.

        Args:
            url: URL to scrape
            custom_actions: Custom actions (converted to JavaScript)
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """
        adapter = self._adapters["crawl4ai"]

        # Convert custom actions to JavaScript if provided
        js_code = None
        if custom_actions:
            js_code = self._convert_actions_to_js(custom_actions)
        else:
            js_code = self._get_basic_js(url)

        return await adapter.scrape(
            url=url,
            wait_for_selector=".content, main, article",
            js_code=js_code,
            timeout=timeout,
        )

    async def _try_stagehand(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Try scraping with Stagehand AI.

        Args:
            url: URL to scrape
            custom_actions: Custom actions (converted to instructions)
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """
        adapter = self._adapters["stagehand"]

        # Convert custom actions to Stagehand instructions
        if custom_actions:
            instructions = self._convert_to_instructions(custom_actions)
        else:
            instructions = [
                "Extract all documentation content",
                "Expand collapsed sections if present",
                "Click on any 'show more' or 'load more' buttons",
                "Wait for dynamic content to load",
            ]

        return await adapter.scrape(
            url=url,
            instructions=instructions,
            timeout=timeout,
        )

    async def _try_playwright(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """Try scraping with Playwright.

        Args:
            url: URL to scrape
            custom_actions: Custom actions to perform
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """
        adapter = self._adapters["playwright"]

        return await adapter.scrape(
            url=url,
            actions=custom_actions or [],
            timeout=timeout,
        )

    async def _fallback_scrape(
        self,
        url: str,
        failed_tool: str,
        custom_actions: list[dict] | None,
        timeout: int,
    ) -> dict[str, Any]:
        """Fallback to next tool in hierarchy.

        Args:
            url: URL to scrape
            failed_tool: Tool that failed
            custom_actions: Custom actions to perform
            timeout: Timeout in milliseconds

        Returns:
            Scraping result or error
        """
        # Define fallback order
        fallback_order = {
            "crawl4ai": ["stagehand", "playwright"],
            "stagehand": ["playwright", "crawl4ai"],
            "playwright": ["stagehand", "crawl4ai"],
        }

        fallback_tools = [
            tool
            for tool in fallback_order.get(failed_tool, [])
            if tool in self._adapters
        ]

        for fallback_tool in fallback_tools:
            try:
                self.logger.info(f"Falling back to {fallback_tool} for {url}")

                start_time = time.time()

                if fallback_tool == "crawl4ai":
                    result = await self._try_crawl4ai(url, custom_actions, timeout)
                elif fallback_tool == "stagehand":
                    result = await self._try_stagehand(url, custom_actions, timeout)
                else:
                    result = await self._try_playwright(url, custom_actions, timeout)

                # Update metrics for successful fallback
                elapsed = time.time() - start_time
                self._update_metrics(fallback_tool, True, elapsed)

                result["provider"] = fallback_tool
                result["fallback_from"] = failed_tool
                result["automation_time_ms"] = elapsed * 1000
                return result

            except Exception as e:
                self.logger.error(f"Fallback {fallback_tool} also failed: {e}")
                elapsed = time.time() - start_time
                self._update_metrics(fallback_tool, False, elapsed)
                continue

        # All tools failed
        return {
            "success": False,
            "error": f"All automation tools failed for {url}",
            "content": "",
            "metadata": {},
            "url": url,
            "provider": "none",
            "failed_tools": [failed_tool] + fallback_tools,
        }

    def _get_basic_js(self, url: str) -> str:
        """Get basic JavaScript for common scenarios.

        Args:
            url: URL to get JavaScript for

        Returns:
            JavaScript code string
        """
        return """
        // Wait for content to load
        await new Promise(r => setTimeout(r, 2000));
        
        // Expand collapsed sections
        document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
            try { el.click(); } catch(e) {}
        });
        
        // Click show more buttons
        document.querySelectorAll('button, a').forEach(el => {
            const text = el.textContent?.toLowerCase() || '';
            if (text.includes('show more') || text.includes('load more') || text.includes('expand')) {
                try { el.click(); } catch(e) {}
            }
        });
        
        // Scroll to load lazy content
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(r => setTimeout(r, 1000));
        """

    def _convert_actions_to_js(self, actions: list[dict]) -> str:
        """Convert custom actions to JavaScript code.

        Args:
            actions: List of action dictionaries

        Returns:
            JavaScript code string
        """
        js_lines = []

        for action in actions:
            action_type = action.get("type", "")

            if action_type == "click":
                selector = action.get("selector", "")
                js_lines.append(f"document.querySelector('{selector}')?.click();")

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")
                js_lines.append(
                    f"document.querySelector('{selector}').value = '{text}';"
                )

            elif action_type == "wait":
                timeout = action.get("timeout", 1000)
                js_lines.append(f"await new Promise(r => setTimeout(r, {timeout}));")

            elif action_type == "scroll":
                js_lines.append("window.scrollTo(0, document.body.scrollHeight);")

            elif action_type == "evaluate":
                script = action.get("script", "")
                js_lines.append(script)

        return "\n".join(js_lines)

    def _convert_to_instructions(self, actions: list[dict]) -> list[str]:
        """Convert custom actions to Stagehand instructions.

        Args:
            actions: List of action dictionaries

        Returns:
            List of instruction strings
        """
        instructions = []

        for action in actions:
            action_type = action.get("type", "")

            if action_type == "click":
                selector = action.get("selector", "")
                instructions.append(f"Click on element with selector '{selector}'")

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")
                instructions.append(
                    f"Type '{text}' in element with selector '{selector}'"
                )

            elif action_type == "wait":
                timeout = action.get("timeout", 1000)
                instructions.append(f"Wait for {timeout} milliseconds")

            elif action_type == "scroll":
                instructions.append("Scroll to the bottom of the page")

            elif action_type == "extract":
                instructions.append("Extract all visible content from the page")

        return instructions

    def _update_metrics(self, tool: str, success: bool, elapsed: float) -> None:
        """Update performance metrics for a tool.

        Args:
            tool: Tool name
            success: Whether the operation succeeded
            elapsed: Time elapsed in seconds
        """
        metrics = self.metrics[tool]

        if success:
            metrics["success"] += 1
        else:
            metrics["failed"] += 1

        metrics["total_time"] += elapsed

        # Update rolling average
        total_attempts = metrics["success"] + metrics["failed"]
        if total_attempts > 0:
            metrics["avg_time"] = metrics["total_time"] / total_attempts

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics for all tools.

        Returns:
            Dictionary with metrics for each tool including success rate
        """
        result = {}

        for tool, metrics in self.metrics.items():
            total_attempts = metrics["success"] + metrics["failed"]
            success_rate = (
                metrics["success"] / total_attempts if total_attempts > 0 else 0.0
            )

            result[tool] = {
                **metrics,
                "success_rate": success_rate,
                "total_attempts": total_attempts,
                "available": tool in self._adapters,
            }

        return result

    def get_recommended_tool(self, url: str) -> str:
        """Get recommended tool for a URL based on performance metrics.

        Args:
            url: URL to analyze

        Returns:
            Recommended tool name
        """
        # Get base recommendation from selection logic
        base_tool = self._select_tool(url, False, None)

        # Check if we have enough data to make performance-based recommendations
        metrics = self.get_metrics()

        if metrics[base_tool]["total_attempts"] < 5:
            # Not enough data, use base recommendation
            return base_tool

        # If success rate is too low, try a different tool
        if metrics[base_tool]["success_rate"] < 0.8:
            # Find best performing available tool
            best_tool = base_tool
            best_rate = metrics[base_tool]["success_rate"]

            for tool, data in metrics.items():
                if (
                    data["available"]
                    and data["total_attempts"] >= 3
                    and data["success_rate"] > best_rate
                ):
                    best_tool = tool
                    best_rate = data["success_rate"]

            return best_tool

        return base_tool
