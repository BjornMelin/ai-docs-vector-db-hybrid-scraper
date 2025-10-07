# pylint: disable=too-many-lines
"""Intelligent browser automation router with three-tier hierarchy."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

from src.config import Config
from src.services.base import BaseService
from src.services.errors import CrawlServiceError

from .browser_use_adapter import BrowserUseAdapter
from .crawl4ai_adapter import Crawl4AIAdapter
from .lightweight_scraper import LightweightScraper
from .playwright_adapter import PlaywrightAdapter


logger = logging.getLogger(__name__)

ROUTING_RULES_ENV = "AI_DOCS_BROWSER_ROUTING_RULES_PATH"
SUPPORTED_ROUTING_TOOLS = {
    "lightweight",
    "crawl4ai",
    "crawl4ai_enhanced",
    "browser_use",
    "playwright",
}


class RoutingRulesModel(BaseModel):
    """Schema for routing rules configuration."""

    routing_rules: dict[str, list[str]] = Field(default_factory=dict)


class ScrapeRequest(BaseModel):
    """Normalized request payload passed to automation adapters."""

    url: str
    timeout_ms: int
    custom_actions: list[dict[str, Any]] | None = None
    interaction_required: bool = False


class ScrapeResult(BaseModel):
    """Standardized automation response structure."""

    success: bool
    url: str = ""
    content: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    provider: str | None = None
    fallback_from: str | None = None
    automation_time_ms: float | None = None
    raw: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize result into legacy dictionary format."""

        return self.model_dump(exclude_none=True)


class TimeoutBudget:
    """Track remaining time for a multi-attempt automation workflow."""

    def __init__(self, timeout_ms: int) -> None:
        normalized_timeout = max(timeout_ms, 0)
        self._deadline = time.monotonic() + normalized_timeout / 1000

    def remaining_ms(self) -> int:
        """Return remaining milliseconds (clamped at zero)."""
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            return 0
        return int(remaining * 1000)


@dataclass
class CircuitBreakerState:
    """Track failure streaks and open intervals for adapters."""

    failures: int = 0
    opened_until: float | None = None


class AutomationRouter(BaseService):  # pylint: disable=too-many-instance-attributes
    """Intelligently route scraping tasks to appropriate automation tool.

    Implements a five-tier automation hierarchy:
    0. Lightweight HTTP (httpx + BS4) - 5-10x faster for static content, $0 cost
    1. Crawl4AI Basic (90% of sites) - 4-6x faster, $0 cost
    2. Crawl4AI Enhanced (Interactive content) - Custom JavaScript, $0 cost
    3. browser-use (Complex interactions) - AI-powered automation with multi-LLM support
    4. Playwright + Firecrawl (Maximum control) - Programmatic control + API fallback
    """

    def __init__(self, config: Config):
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

        # Load site-specific routing rules from configuration
        self.routing_rules = self._load_routing_rules()

        # Performance metrics tracking for all 5 tiers
        self.metrics = {
            "lightweight": {
                "success": 0,
                "failed": 0,
                "avg_time": 0.0,
                "total_time": 0.0,
            },
            "crawl4ai": {"success": 0, "failed": 0, "avg_time": 0.0, "total_time": 0.0},
            "crawl4ai_enhanced": {
                "success": 0,
                "failed": 0,
                "avg_time": 0.0,
                "total_time": 0.0,
            },
            "browser_use": {
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
            "firecrawl": {
                "success": 0,
                "failed": 0,
                "avg_time": 0.0,
                "total_time": 0.0,
            },
        }
        self._breaker_state: dict[str, CircuitBreakerState] = {}
        self._breaker_failure_threshold = 3
        self._breaker_cooldown_seconds = 60.0

    def _load_routing_rules(self) -> dict[str, list[str]]:
        """Load routing rules from configuration file.

        Returns:
            Dictionary mapping tool names to lists of domains

        """
        try:
            env_path = os.getenv(ROUTING_RULES_ENV)
            if env_path:
                config_file = Path(env_path).expanduser()
            else:
                project_root = Path(__file__).parent.parent.parent.parent
                config_file = project_root / "config" / "browser-routing-rules.json"

            with config_file.open(encoding="utf-8") as f:
                payload = json.load(f)

            rules_config = RoutingRulesModel.model_validate(payload)

            filtered_rules: dict[str, list[str]] = {}
            for tool, domains in rules_config.routing_rules.items():
                if tool not in SUPPORTED_ROUTING_TOOLS:
                    self.logger.warning(
                        "Skipping routing rule for unsupported tool '%s'", tool
                    )
                    continue
                clean_domains = [
                    domain.strip().lower()
                    for domain in domains
                    if isinstance(domain, str) and domain.strip()
                ]
                if not clean_domains:
                    continue
                filtered_rules[tool] = clean_domains

            if filtered_rules:
                self.logger.info("Loaded routing rules from %s", config_file)
                return filtered_rules

            self.logger.warning(
                "Routing rules file %s contained no supported entries; using defaults",
                config_file,
            )

        except FileNotFoundError:
            self.logger.warning("Routing rules file not found; using defaults")
        except (OSError, json.JSONDecodeError, ValidationError):
            self.logger.exception(
                "Failed to load routing rules; falling back to defaults"
            )

        # Fallback to default rules if loading fails
        return self._get_default_routing_rules()

    def _get_default_routing_rules(self) -> dict[str, list[str]]:
        """Get default routing rules as fallback.

        Returns:
            Default routing rules dictionary

        """
        return {
            "browser_use": [
                "vercel.com",
                "clerk.com",
                "supabase.com",
                "netlify.com",
                "railway.app",
                "planetscale.com",
                "react.dev",
                "nextjs.org",
                "docs.anthropic.com",
            ],
            "playwright": [
                "github.com",
                "stackoverflow.com",
                "discord.com",
                "slack.com",
                "app.posthog.com",
                "notion.so",
            ],
        }

    async def initialize(self) -> None:
        """Initialize available automation adapters."""
        if self._initialized:
            return

        # Initialize Tier 0: Lightweight HTTP scraper
        try:
            adapter = LightweightScraper(self.config)
            await adapter.initialize()
            self._adapters["lightweight"] = adapter
            self.logger.info("Initialized Lightweight HTTP adapter")

        except (AttributeError, ConnectionError, ImportError, RuntimeError):
            self.logger.warning("Failed to initialize Lightweight adapter")

        # Initialize Tier 1: Crawl4AI Basic
        try:
            adapter = Crawl4AIAdapter(self.config.crawl4ai)
            await adapter.initialize()
            self._adapters["crawl4ai"] = adapter
            self.logger.info("Initialized Crawl4AI adapter")

        except (AttributeError, ImportError, RuntimeError, ValueError):
            self.logger.warning("Failed to initialize Crawl4AI adapter")

        # Initialize Tier 2: BrowserUse (Enhanced)
        try:
            adapter = BrowserUseAdapter(self.config.browser_use)
            await adapter.initialize()
            self._adapters["browser_use"] = adapter
            self.logger.info("Initialized BrowserUse adapter")

        except (AttributeError, ImportError, RuntimeError, ValueError):
            self.logger.warning("Failed to initialize BrowserUse adapter")

        # Initialize Tier 3: Playwright
        try:
            adapter = PlaywrightAdapter(self.config.playwright)
            await adapter.initialize()
            self._adapters["playwright"] = adapter
            self.logger.info("Initialized Playwright adapter")

        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError):
            self.logger.warning("Failed to initialize Playwright adapter")

        # TODO: Initialize Tier 4: Firecrawl adapter when available
        # try:
        #     from .firecrawl_adapter import FirecrawlAdapter
        #     adapter = FirecrawlAdapter(self.config.firecrawl)
        #     await adapter.initialize()
        #     self._adapters["firecrawl"] = adapter
        #     self.logger.info("Initialized Firecrawl adapter")
        # except Exception:
        #     self.logger.warning("Failed to initialize Firecrawl adapter")

        if not self._adapters:
            msg = "No automation adapters available"
            raise CrawlServiceError(msg)

        self._initialized = True
        self.logger.info(
            "5-tier automation router initialized with %d adapters", len(self._adapters)
        )

    async def cleanup(self) -> None:
        """Cleanup all adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.cleanup()
                self.logger.info("Cleaned up %s adapter", name)
            except (OSError, AttributeError, ConnectionError, ImportError):
                self.logger.exception("Error cleaning up %s adapter", name)

        self._adapters.clear()
        self._initialized = False

    async def scrape(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        url: str,
        interaction_required: bool = False,
        custom_actions: list[dict] | None = None,
        force_tool: Literal[
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ]
        | None = None,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
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
            msg = "Router not initialized"
            raise CrawlServiceError(msg)

        # Determine which tool to use
        if force_tool:
            tool = self._validate_forced_tool(force_tool)
        else:
            tool = await self._select_tool(url, interaction_required, custom_actions)

        self.logger.info("Using %s for %s", tool, url)

        budget = TimeoutBudget(timeout)
        start_time = time.time()

        try:
            remaining_ms = budget.remaining_ms()
            if remaining_ms <= 0:
                msg = "Timeout budget exhausted before executing automation"
                raise CrawlServiceError(msg)

            request = ScrapeRequest(
                url=url,
                timeout_ms=remaining_ms,
                custom_actions=custom_actions,
                interaction_required=interaction_required,
            )

            result = await self._execute_tool(tool, request)
            elapsed = time.time() - start_time

            result = result.model_copy(
                update={
                    "provider": tool,
                    "automation_time_ms": elapsed * 1000,
                    "url": result.url or url,
                }
            )

            self._update_metrics(tool, result.success, elapsed)

            if not result.success:
                self.logger.warning(
                    "%s returned unsuccessful result for %s; attempting fallback",
                    tool,
                    url,
                )
                fallback_result = await self._fallback_scrape(
                    url,
                    tool,
                    request,
                    budget,
                    initial_result=result,
                )
                return fallback_result.as_dict()

            return result.as_dict()

        except (
            OSError,
            PermissionError,
            TimeoutError,
            CrawlServiceError,
            KeyError,
        ) as exc:
            self.logger.exception("%s failed for %s", tool, url)
            elapsed = time.time() - start_time
            self._update_metrics(tool, False, elapsed)

            request = ScrapeRequest(
                url=url,
                timeout_ms=budget.remaining_ms(),
                custom_actions=custom_actions,
                interaction_required=interaction_required,
            )

            fallback_result = await self._fallback_scrape(
                url,
                tool,
                request,
                budget,
                failure_reason=str(exc),
            )
            return fallback_result.as_dict()

    async def _select_tool(
        self,
        url: str,
        interaction_required: bool,
        custom_actions: list[dict] | None,
    ) -> str:
        """Select appropriate tool based on URL and requirements using 5-tier hierarchy.

        Args:
            url: URL to analyze
            interaction_required: Whether interaction is needed
            custom_actions: Custom actions to perform

        Returns:
            Tool name to use

        """
        domain = urlparse(url).netloc.lower()

        # Check explicit routing rules first
        route_tool = self._check_routing_rules(domain)
        if route_tool:
            if self._is_tool_available_for_use(route_tool):
                return route_tool
            self.logger.info(
                "Routing rule chose %s but it is unavailable; using heuristics",
                route_tool,
            )

        # Check for interaction requirements (forces higher tiers)
        if interaction_required or custom_actions:
            interaction_tool = self._get_interaction_tool()
            if interaction_tool:
                return interaction_tool

        # Tier 0: Try lightweight HTTP first (fastest, $0 cost)
        if self._is_tool_available_for_use("lightweight"):
            try:
                can_handle = await self._adapters["lightweight"].can_handle(url)
                if self._lightweight_can_handle(can_handle):
                    return "lightweight"
            except TimeoutError:
                self.logger.debug("Lightweight adapter can_handle check failed")

        # Check for JavaScript-heavy patterns that need higher tiers
        js_patterns = ["spa", "react", "vue", "angular", "app", "dashboard", "console"]
        if any(pattern in url.lower() for pattern in js_patterns):
            enhanced_tool = self._get_enhanced_tool()
            if enhanced_tool:
                return enhanced_tool

        # Default fallback hierarchy
        default_tool = self._get_default_tool()
        if default_tool:
            return default_tool

        msg = "No automation adapters available for selection"
        raise CrawlServiceError(msg)

    def _check_routing_rules(self, domain: str) -> str | None:
        """Check if domain matches explicit routing rules."""
        for tool, domains in self.routing_rules.items():
            if not self._is_tool_available_for_use(tool):
                continue

            for configured_domain in domains:
                if self._domain_matches(domain, configured_domain):
                    return tool
        return None

    def _get_default_tool(self) -> str | None:
        """Get default tool based on adapter availability."""
        for tool in ("crawl4ai", "lightweight", "playwright", "browser_use"):
            if self._is_tool_available_for_use(tool):
                return tool
        return None

    def _get_interaction_tool(self) -> str | None:
        """Get best tool for interactive requirements."""
        for tool in ("browser_use", "playwright", "crawl4ai"):
            if self._is_tool_available_for_use(tool):
                return tool
        return None

    def _get_enhanced_tool(self) -> str | None:
        """Get best tool for enhanced/JavaScript-heavy content."""
        if self._is_tool_available_for_use("browser_use"):
            return "browser_use"
        if self._is_tool_available_for_use("crawl4ai_enhanced"):
            return "crawl4ai_enhanced"  # Use enhanced mode
        if self._is_tool_available_for_use("playwright"):
            return "playwright"
        return None

    async def _execute_tool(
        self,
        tool: str,
        request: ScrapeRequest,
    ) -> ScrapeResult:
        """Execute the selected tool respecting the remaining timeout budget."""
        timeout_ms = max(request.timeout_ms, 0)

        if tool == "lightweight":
            raw = await self._try_lightweight(request.url, timeout=timeout_ms)
        elif tool == "crawl4ai":
            raw = await self._try_crawl4ai(
                request.url, request.custom_actions, timeout=timeout_ms
            )
        elif tool == "crawl4ai_enhanced":
            raw = await self._try_crawl4ai_enhanced(
                request.url, request.custom_actions, timeout=timeout_ms
            )
        elif tool == "browser_use":
            raw = await self._try_browser_use(
                request.url, request.custom_actions, timeout=timeout_ms
            )
        elif tool == "firecrawl":
            raw = await self._try_firecrawl(request.url, _timeout=timeout_ms)
        else:
            raw = await self._try_playwright(
                request.url, request.custom_actions, timeout=timeout_ms
            )

        return self._normalize_result(tool, raw, request)

    def _normalize_result(
        self, tool: str, raw: Any, request: ScrapeRequest
    ) -> ScrapeResult:
        """Convert adapter-specific payloads into a standardized result."""

        if isinstance(raw, ScrapeResult):
            return raw

        if isinstance(raw, dict):
            metadata = raw.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}

            return ScrapeResult(
                success=bool(raw.get("success", True)),
                url=str(raw.get("url") or request.url),
                content=raw.get("content"),
                metadata=dict(metadata),
                error=raw.get("error"),
                provider=raw.get("provider") or tool,
                fallback_from=raw.get("fallback_from"),
                automation_time_ms=raw.get("automation_time_ms"),
                raw=raw,
            )

        return ScrapeResult(
            success=False,
            url=request.url,
            error=f"Unsupported result type '{type(raw).__name__}' from {tool}",
            provider=tool,
            metadata={"raw_result": raw},
        )

    def _validate_forced_tool(
        self,
        force_tool: Literal[
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ],
    ) -> str:
        """Ensure the requested forced tool is available and supported."""
        if force_tool == "crawl4ai_enhanced":
            if "crawl4ai" not in self._adapters:
                msg = "Forced tool 'crawl4ai_enhanced' requires Crawl4AI adapter"
                raise CrawlServiceError(msg)
            return force_tool

        if force_tool == "firecrawl":
            msg = "Forced tool 'firecrawl' not available"
            raise CrawlServiceError(msg)

        if force_tool not in self._adapters:
            msg = f"Forced tool '{force_tool}' not available"
            raise CrawlServiceError(msg)

        return force_tool

    @staticmethod
    def _domain_matches(domain: str, rule: str) -> bool:
        """Determine whether a domain matches a routing rule entry."""
        candidate = domain.lower()
        expected = rule.strip().lower()
        if not expected:
            return False
        if candidate == expected:
            return True
        return candidate.endswith(f".{expected}")

    @staticmethod
    def _lightweight_can_handle(result: Any) -> bool:
        """Interpret lightweight adapter capability result safely."""
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("can_handle"))
        if hasattr(result, "can_handle"):
            return bool(result.can_handle)
        return bool(result)

    def _is_tool_available_for_use(self, tool: str) -> bool:
        """Check if a tool is initialized and not blocked by circuit breaker."""

        if tool == "crawl4ai_enhanced":
            return "crawl4ai" in self._adapters and not self._is_breaker_open(tool)

        return tool in self._adapters and not self._is_breaker_open(tool)

    def _is_breaker_open(self, tool: str) -> bool:
        """Return True when the circuit breaker is open for a tool."""

        state = self._breaker_state.get(tool)
        if not state or state.opened_until is None:
            return False
        if state.opened_until <= time.monotonic():
            self._breaker_state[tool] = CircuitBreakerState()
            return False
        return True

    def _record_breaker_outcome(self, tool: str, success: bool) -> None:
        """Update circuit breaker state with the result of an attempt."""

        state = self._breaker_state.setdefault(tool, CircuitBreakerState())
        if success:
            state.failures = 0
            state.opened_until = None
            return

        state.failures += 1
        if state.failures >= self._breaker_failure_threshold:
            cooldown_until = time.monotonic() + self._breaker_cooldown_seconds
            state.opened_until = cooldown_until
            self.logger.warning(
                "Circuit breaker opened for %s for %.0f seconds",
                tool,
                self._breaker_cooldown_seconds,
            )

    async def _try_crawl4ai(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
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

    async def _try_browser_use(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Try scraping with browser-use AI.

        Args:
            url: URL to scrape
            custom_actions: Custom actions (converted to natural language task)
            timeout: Timeout in milliseconds

        Returns:
            Scraping result

        """
        adapter = self._adapters["browser_use"]

        # Convert custom actions to natural language task
        if custom_actions:
            task = self._convert_to_task(custom_actions)
        else:
            task = (
                "Extract all documentation content including code examples. "
                "Expand any collapsed sections or interactive elements. "
                "Click on any 'show more' or 'load more' buttons and wait for "
                "dynamic content to load."
            )

        return await adapter.scrape(
            url=url,
            task=task,
            timeout=timeout,
        )

    async def _try_playwright(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
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

    async def _fallback_scrape(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        url: str,
        failed_tool: str,
        base_request: ScrapeRequest,
        timeout_budget: TimeoutBudget,
        *,
        initial_result: ScrapeResult | None = None,
        failure_reason: str | None = None,
    ) -> ScrapeResult:
        """Fallback to next tool in hierarchy.

        Args:
            url: URL to scrape
            failed_tool: Tool that failed initially
            base_request: Original scrape request context
            timeout_budget: Remaining timeout budget for attempts
            initial_result: Result returned by the failed tool (if any)
            failure_reason: Error message from the last failure

        Returns:
            Scraping result or error

        """
        # Define fallback order
        fallback_order = {
            "lightweight": ["crawl4ai", "browser_use", "playwright"],
            "crawl4ai": ["browser_use", "playwright"],
            "crawl4ai_enhanced": ["browser_use", "playwright", "crawl4ai"],
            "browser_use": ["playwright", "crawl4ai"],
            "playwright": ["browser_use", "crawl4ai"],
        }

        fallback_tools = [
            tool
            for tool in fallback_order.get(failed_tool, [])
            if self._is_tool_available_for_use(tool)
        ]

        attempted_tools = [failed_tool]
        unsuccessful_results: list[ScrapeResult] = []

        for fallback_tool in fallback_tools:
            remaining_ms = timeout_budget.remaining_ms()
            if remaining_ms <= 0:
                break

            start_time = time.time()

            try:
                self.logger.info("Falling back to %s for %s", fallback_tool, url)

                fallback_request = base_request.model_copy(
                    update={"timeout_ms": remaining_ms}
                )
                result = await self._execute_tool(fallback_tool, fallback_request)

                # Update metrics for successful fallback
                elapsed = time.time() - start_time
                result = result.model_copy(
                    update={
                        "provider": fallback_tool,
                        "fallback_from": failed_tool,
                        "automation_time_ms": elapsed * 1000,
                        "url": result.url or base_request.url or url,
                    }
                )
                self._update_metrics(fallback_tool, result.success, elapsed)

                attempted_tools.append(fallback_tool)

                if result.success:
                    return result

                unsuccessful_results.append(result)

            except (OSError, PermissionError, TimeoutError, CrawlServiceError) as exc:
                self.logger.exception("Fallback %s also failed", fallback_tool)
                elapsed = time.time() - start_time
                self._update_metrics(fallback_tool, False, elapsed)
                attempted_tools.append(fallback_tool)
                unsuccessful_results.append(
                    ScrapeResult(
                        success=False,
                        url=base_request.url or url,
                        error=str(exc),
                        provider=fallback_tool,
                    )
                )
                continue

        # All tools failed
        metadata: dict[str, Any] = {
            "failed_tools": attempted_tools,
            "unsuccessful_results": [res.as_dict() for res in unsuccessful_results],
        }
        if failure_reason:
            metadata["failure_reason"] = failure_reason
        if initial_result:
            metadata["initial_result"] = initial_result.as_dict()

        return ScrapeResult(
            success=False,
            url=base_request.url or url,
            error=f"All automation tools failed for {url}",
            provider="none",
            metadata=metadata,
        )

    def _get_basic_js(self, _url: str) -> str:
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
        """  # noqa: E501

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

    def _convert_to_task(self, actions: list[dict]) -> str:
        """Convert custom actions to browser-use natural language task.

        Args:
            actions: List of action dictionaries

        Returns:
            Natural language task description

        """
        task_parts = []

        for action in actions:
            action_type = action.get("type", "")

            if action_type == "click":
                selector = action.get("selector", "")
                task_parts.append(f"click on element with selector '{selector}'")

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")
                task_parts.append(
                    f"type '{text}' in element with selector '{selector}'"
                )

            elif action_type == "wait":
                timeout = action.get("timeout", 1000)
                task_parts.append(f"wait for {timeout} milliseconds")

            elif action_type == "scroll":
                task_parts.append("scroll to the bottom of the page")

            elif action_type == "extract":
                task_parts.append("extract all visible content from the page")

            elif action_type == "expand":
                task_parts.append("expand any collapsed sections or menus")

        if task_parts:
            return (
                f"Navigate to the page, then {', '.join(task_parts)}, and finally "
                "extract all documentation content including code examples."
            )
        return (
            "Navigate to the page and extract all documentation "
            "content including code examples."
        )

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

        self._record_breaker_outcome(tool, success)

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
                "available": self._is_tool_available_for_use(tool),
            }

        return result

    async def get_recommended_tool(self, url: str) -> str:
        """Get recommended tool for a URL based on performance metrics.

        Args:
            url: URL to analyze

        Returns:
            Recommended tool name
        """

        # Get base recommendation from selection logic
        base_tool = await self._select_tool(url, False, None)

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

    async def _try_lightweight(
        self,
        url: str,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Try scraping with Lightweight HTTP tier.

        Args:
            url: URL to scrape
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """

        adapter = self._adapters["lightweight"]

        if timeout <= 0:
            msg = "Timeout budget exhausted before lightweight attempt"
            raise CrawlServiceError(msg)

        # Convert timeout to seconds for LightweightScraper
        timeout_seconds = max(timeout, 0) / 1000

        # LightweightScraper returns different format than other adapters
        scrape_result = await adapter.scrape(url, timeout=timeout_seconds)

        if scrape_result is None:
            # Escalate to higher tier
            msg = "Lightweight scraper returned None - content should escalate"
            raise CrawlServiceError(msg)

        # Convert LightweightScraper format to standard format
        return {
            "success": scrape_result.success,
            "content": scrape_result.text,
            "metadata": {
                "title": scrape_result.title,
                "url": scrape_result.url,
                "tier": scrape_result.tier,
                "headings": scrape_result.headings,
                "links": scrape_result.links,
                **scrape_result.metadata,
            },
            "url": url,
            "extraction_time_ms": scrape_result.extraction_time_ms,
        }

    async def _try_crawl4ai_enhanced(
        self,
        url: str,
        custom_actions: list[dict] | None = None,
        *,
        timeout: int = 30000,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        """Try scraping with Crawl4AI Enhanced mode.

        Args:
            url: URL to scrape
            custom_actions: Custom actions (converted to JavaScript)
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """

        adapter = self._adapters["crawl4ai"]

        # Enhanced mode uses more sophisticated JavaScript
        enhanced_js = self._get_enhanced_js(url, custom_actions)

        return await adapter.scrape(
            url=url,
            wait_for_selector=".content, main, article, [data-content], .documentation",
            js_code=enhanced_js,
            timeout=timeout,
        )

    async def _try_firecrawl(
        self,
        _url: str,
        *,
        _timeout: int = 30000,
    ) -> dict[str, Any]:
        """Try scraping with Firecrawl API (Tier 4).

        Args:
            url: URL to scrape
            timeout: Timeout in milliseconds

        Returns:
            Scraping result
        """

        # TODO: Implement Firecrawl adapter
        # For now, return error indicating not available
        msg = "Firecrawl adapter not yet implemented"
        raise CrawlServiceError(msg)

    def _get_enhanced_js(
        self, _url: str, custom_actions: list[dict] | None = None
    ) -> str:
        """Get enhanced JavaScript for Crawl4AI enhanced mode.

        Args:
            url: URL being scraped
            custom_actions: Custom actions to incorporate

        Returns:
            Enhanced JavaScript code
        """

        enhanced_js = """
        // Enhanced JavaScript for dynamic content
        console.log('Enhanced mode: Starting advanced content extraction');

        // Wait for initial load
        await new Promise(r => setTimeout(r, 3000));

        // Handle lazy loading
        async function handleLazyLoading() {
            // Scroll to trigger lazy loading
            for (let i = 0; i < 3; i++) {
                window.scrollTo(0, document.body.scrollHeight / 3 * (i + 1));
                await new Promise(r => setTimeout(r, 1500));
            }
        }

        // Handle dynamic content expansion
        async function expandDynamicContent() {
            // Click expandable elements
            const expandableSelectors = [
                '[aria-expanded="false"]',
                '.expand-btn', '.show-more', '.load-more',
                'button[data-toggle]', 'button[data-expand]',
                '.collapsible:not(.active)', '.accordion-header'
            ];

            for (const selector of expandableSelectors) {
                const elements = document.querySelectorAll(selector);
                for (const el of elements) {
                    try {
                        el.click();
                        await new Promise(r => setTimeout(r, 800));
                    } catch(e) { console.log('Click failed:', e); }
                }
            }
        }

        // Handle tab content
        async function handleTabContent() {
            const tabSelectors = [
                '.tab:not(.active)', '.tab-item:not(.active)',
                '[role="tab"]:not([aria-selected="true"])',
                '.nav-link:not(.active)'
            ];

            for (const selector of tabSelectors) {
                const tabs = document.querySelectorAll(selector);
                for (const tab of tabs) {
                    try {
                        tab.click();
                        await new Promise(r => setTimeout(r, 1000));
                    } catch(e) { console.log('Tab click failed:', e); }
                }
            }
        }

        // Execute enhancement sequence
        await handleLazyLoading();
        await expandDynamicContent();
        await handleTabContent();

        // Final scroll to ensure all content is loaded
        window.scrollTo(0, 0);
        await new Promise(r => setTimeout(r, 1000));
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(r => setTimeout(r, 2000));

        console.log('Enhanced mode: Content extraction complete');
        """

        # Add custom actions if provided
        if custom_actions:
            custom_js = self._convert_actions_to_js(custom_actions)
            enhanced_js += f"\n\n// Custom actions\n{custom_js}"

        return enhanced_js
