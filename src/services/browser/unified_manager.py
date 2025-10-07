"""Unified browser automation manager providing single interface to all 5 tiers.

Provides a unified interface for browser automation that routes requests to the
appropriate tier based on content complexity, requirements, and performance.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import redis
from pydantic import BaseModel, Field

from src.config import Config
from src.services.base import BaseService
from src.services.errors import CrawlServiceError

from .monitoring import BrowserAutomationMonitor


try:
    from src.services.cache.browser_cache import BrowserCache, BrowserCacheEntry
except ImportError:
    BrowserCache = None
    BrowserCacheEntry = None

if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager
    from src.services.cache.browser_cache import BrowserCache as BrowserCacheType
else:
    BrowserCacheType = Any
    ClientManager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class UnifiedScrapingRequest(BaseModel):
    """Unified request model for all scraping operations."""

    url: str = Field(description="URL to scrape")
    tier: (
        Literal[
            "auto",
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ]
        | None
    ) = Field(
        default="auto",
        description="Specific tier to use (auto for intelligent selection)",
    )
    interaction_required: bool = Field(
        default=False, description="Whether page interaction is required"
    )
    custom_actions: list[dict] | None = Field(
        default=None, description="Custom actions to perform"
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    wait_for_selector: str | None = Field(
        default=None, description="Specific selector to wait for"
    )
    extract_metadata: bool = Field(
        default=True, description="Whether to extract page metadata"
    )


class UnifiedScrapingResponse(BaseModel):
    """Unified response model for all scraping operations."""

    success: bool = Field(description="Whether scraping was successful")
    content: str = Field(description="Extracted content")
    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Page metadata")

    # Execution details
    tier_used: str = Field(description="Which tier was used for scraping")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    fallback_attempted: bool = Field(
        default=False, description="Whether fallback was attempted"
    )

    # Quality metrics
    content_length: int = Field(description="Length of extracted content")
    quality_score: float = Field(default=0.0, description="Content quality score (0-1)")

    # Error information
    error: str | None = Field(default=None, description="Error message if failed")
    failed_tiers: list[str] = Field(
        default_factory=list, description="Tiers that failed before success"
    )


class TierMetrics(BaseModel):
    """Performance metrics for a specific tier."""

    tier_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    success_rate: float = 0.0


class UnifiedBrowserManager(  # pylint: disable=too-many-instance-attributes
    BaseService
):
    """Unified browser automation manager providing single interface to all 5 tiers.

    The UnifiedBrowserManager acts as the single entry point for all browser automation
    tasks, intelligently routing requests to the most appropriate tier:

    - Tier 0: Lightweight HTTP (httpx + BeautifulSoup) - 5-10x faster for static content
    - Tier 1: Crawl4AI Basic - Standard browser automation for dynamic content
    - Tier 2: Crawl4AI Enhanced - Interactive content with custom JavaScript
    - Tier 3: Browser-use AI - Complex interactions with AI-powered automation
    - Tier 4: Playwright + Firecrawl - Maximum control + API fallback
    """

    def __init__(self, config: Config):
        """Initialize unified browser manager."""

        super().__init__(config)
        self._automation_router: Any = None
        self._client_manager: ClientManager | None = None
        self._browser_cache: BrowserCacheType | None = None
        self._tier_metrics: dict[str, TierMetrics] = {}
        self._initialized = False
        self._cache_enabled = bool(getattr(config.cache, "enable_browser_cache", True))

        # Initialize monitoring system
        self._monitor = BrowserAutomationMonitor()
        self._monitoring_enabled = (
            getattr(config.performance, "enable_monitoring", True)
            if hasattr(config, "performance")
            else True
        )

        # Initialize tier metrics
        for tier in [
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ]:
            self._tier_metrics[tier] = TierMetrics(tier_name=tier)

    async def initialize(self) -> None:
        """Initialize the unified browser manager and all dependencies."""

        if self._initialized:
            return

        try:
            await self._initialize_client_manager()
            await self._initialize_browser_cache()
            await self._initialize_monitoring()
            self._finalize_initialization()
        except Exception as exc:
            logger.exception("Failed to initialize UnifiedBrowserManager")
            msg = "Failed to initialize unified browser manager"
            raise CrawlServiceError(msg) from exc

    async def _initialize_client_manager(self) -> None:
        """Initialize client manager and automation router."""
        from src.infrastructure.client_manager import ClientManager as ClientManagerCls

        self._client_manager = ClientManagerCls()
        await self._client_manager.initialize()

        # Get enhanced automation router (lazy-initialized in ClientManager)
        self._automation_router = (
            await self._client_manager.get_browser_automation_router()
        )

    async def _initialize_browser_cache(self) -> None:
        """Initialize browser cache if enabled."""
        if not self._cache_enabled:
            return

        if BrowserCache is None:
            logger.warning("BrowserCache not available, disabling cache")
            self._cache_enabled = False
            return

        client_manager = self._client_manager
        if client_manager is None:
            logger.warning("Client manager not initialized; disabling cache")
            self._cache_enabled = False
            return

        cache_manager = await client_manager.get_cache_manager()

        cache_config = getattr(self.config, "cache", None)
        self._browser_cache = BrowserCache(
            local_cache=getattr(cache_manager, "local_cache", None),
            distributed_cache=getattr(cache_manager, "distributed_cache", None),
            default_ttl=getattr(cache_config, "browser_cache_ttl", 3600),
            dynamic_content_ttl=getattr(cache_config, "browser_dynamic_ttl", 300),
            static_content_ttl=getattr(cache_config, "browser_static_ttl", 86400),
        )
        logger.info("Browser caching enabled for UnifiedBrowserManager")

    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring system if enabled."""
        if not self._monitoring_enabled:
            return

        await self._monitor.start_monitoring()
        logger.info("Browser automation monitoring started")

    def _finalize_initialization(self) -> None:
        """Finalize initialization process."""
        self._initialized = True
        logger.info("UnifiedBrowserManager initialized with 5-tier automation")

    def _is_cache_available(self) -> bool:
        """Check if browser cache is available and enabled.

        Returns:
            True if cache is available and enabled, False otherwise
        """
        return bool(self._cache_enabled and self._browser_cache)

    async def _try_get_cached_result(
        self, request: UnifiedScrapingRequest, start_time: float
    ) -> UnifiedScrapingResponse | None:
        """Try to get cached result for the request."""
        if not self._is_cache_available() or request.interaction_required:
            return None

        assert self._browser_cache is not None  # For type checker

        try:
            cache_key = self._browser_cache.generate_cache_key(
                request.url, None if request.tier == "auto" else request.tier
            )
            cached_entry = await self._browser_cache.get(cache_key)

            if cached_entry:
                return await self._create_cached_response(
                    request, cached_entry, start_time
                )
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Cache error for %s, continuing with fresh scrape: %s",
                request.url,
                exc,
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Unexpected cache error for %s, continuing with fresh scrape",
                request.url,
                exc_info=True,
            )

        return None

    async def _create_cached_response(
        self,
        request: UnifiedScrapingRequest,
        cached_entry,  # BrowserCacheEntry
        start_time: float,
    ) -> UnifiedScrapingResponse:
        """Create response from cached entry."""
        execution_time = (time.time() - start_time) * 1000

        logger.info(
            "Browser cache hit for %s (cached tier: %s, age: %.1fs)",
            request.url,
            cached_entry.tier_used,
            time.time() - cached_entry.timestamp,
        )

        # Update metrics for cache hit
        self._update_tier_metrics(cached_entry.tier_used, True, execution_time)
        await self._record_monitoring_metrics(
            cached_entry.tier_used, True, execution_time, cache_hit=True
        )

        return UnifiedScrapingResponse(
            success=True,
            content=cached_entry.content,
            url=request.url,
            title=cached_entry.metadata.get("title", ""),
            metadata={
                **cached_entry.metadata,
                "cached": True,
                "cache_age_seconds": time.time() - cached_entry.timestamp,
            },
            tier_used=cached_entry.tier_used,
            execution_time_ms=execution_time,
            fallback_attempted=False,
            content_length=len(cached_entry.content),
            quality_score=self._calculate_quality_score(
                {"success": True, "content": cached_entry.content}
            ),
            failed_tiers=[],
        )

    async def _record_monitoring_metrics(  # pylint: disable=too-many-arguments
        self,
        tier: str,
        success: bool,
        execution_time: float,
        *,
        cache_hit: bool = False,
        runtime: str | None = None,
        challenge: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """Record monitoring metrics if monitoring is enabled.

        Args:
            tier: Tier name
            success: Whether operation was successful
            execution_time: Execution time in milliseconds
            cache_hit: Whether this was a cache hit
            runtime: Runtime identifier if applicable
            challenge: Challenge outcome if applicable
            error_type: Error type if failed
        """
        if not (self._monitoring_enabled and self._monitor):
            return

        try:
            await self._monitor.record_request_metrics(
                tier=tier,
                success=success,
                response_time_ms=execution_time,
                cache_hit=cache_hit,
                runtime=runtime,
                challenge=challenge,
                error_type=error_type,
            )
        except (ConnectionError, OSError, PermissionError) as exc:
            logger.debug("Failed to record monitoring metrics: %s", exc)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Unexpected error while recording monitoring metrics", exc_info=True
            )

    async def _try_cache_result(
        self, request: UnifiedScrapingRequest, response: UnifiedScrapingResponse
    ) -> None:
        """Try to cache successful scraping result."""
        if not (
            self._is_cache_available()
            and response.success
            and not request.interaction_required
            and response.content_length > 0
        ):
            return

        try:
            await self._store_cache_entry(request, response)
        except (ConnectionError, OSError, PermissionError):
            logger.warning("Failed to cache result for %s", request.url)

    async def _store_cache_entry(
        self, request: UnifiedScrapingRequest, response: UnifiedScrapingResponse
    ) -> None:
        """Store cache entry for successful response."""
        if BrowserCacheEntry is None or not self._is_cache_available():
            return

        assert self._browser_cache is not None  # For type checker

        cache_entry = BrowserCacheEntry(
            url=request.url,
            content=response.content,
            metadata={
                "title": response.title,
                **response.metadata,
            },
            tier_used=response.tier_used,
        )

        cache_key = self._browser_cache.generate_cache_key(
            request.url, None if request.tier == "auto" else request.tier
        )

        await self._browser_cache.set(cache_key, cache_entry)
        logger.debug(
            "Cached browser result for %s (tier: %s)", request.url, response.tier_used
        )

    async def _create_scraping_response(
        self,
        request: UnifiedScrapingRequest,
        result: dict[str, Any],
        execution_time: float,
    ) -> UnifiedScrapingResponse:
        """Create unified scraping response from automation router result."""
        # Extract tier information
        tier_used = result.get("provider", "unknown")
        fallback_attempted = "fallback_from" in result
        failed_tiers = result.get("failed_tools", [])
        quality_score = self._calculate_quality_score(result)

        metadata = result.get("metadata", {})

        tier_used = metadata.get("tier") or result.get("provider", "unknown")
        runtime_used = metadata.get("runtime")
        challenge_outcome = metadata.get("challenge_outcome")
        success_flag = result.get("success", False)

        # Update metrics
        self._update_tier_metrics(tier_used, success_flag, execution_time)
        await self._record_monitoring_metrics(
            tier_used,
            success_flag,
            execution_time,
            runtime=runtime_used,
            challenge=(
                challenge_outcome
                if challenge_outcome in {"detected", "solved"}
                else None
            ),
        )

        return UnifiedScrapingResponse(
            success=success_flag,
            content=result.get("content", ""),
            url=request.url,
            title=metadata.get("title", ""),
            metadata=metadata,
            tier_used=tier_used,
            execution_time_ms=execution_time,
            fallback_attempted=fallback_attempted,
            content_length=len(result.get("content", "")),
            quality_score=quality_score,
            failed_tiers=failed_tiers,
        )

    async def _perform_url_analysis(self, url: str) -> dict[str, Any]:
        """Perform URL analysis to determine optimal tier."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Get tier recommendation from AutomationRouter
        recommended_tier = await self._automation_router.get_recommended_tool(url)

        # Get metrics for tier performance
        tier_metrics = self._tier_metrics.get(
            recommended_tier, TierMetrics(tier_name=recommended_tier)
        )

        return {
            "url": url,
            "domain": domain,
            "recommended_tier": recommended_tier,
            "expected_performance": {
                "estimated_time_ms": tier_metrics.average_response_time_ms,
                "success_rate": tier_metrics.success_rate,
            },
        }

    async def _cleanup_monitoring(self) -> None:
        """Stop monitoring if active."""
        if not (self._monitoring_enabled and self._monitor):
            return

        try:
            await self._monitor.stop_monitoring()
        except (ConnectionError, OSError, RuntimeError, TimeoutError):
            logger.debug("Failed to stop monitoring during cleanup")

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        await self._cleanup_monitoring()

        if self._client_manager:
            await self._client_manager.cleanup()
            self._client_manager = None

        self._automation_router = None
        self._initialized = False
        logger.info("UnifiedBrowserManager cleaned up")

    async def scrape(
        self,
        request: UnifiedScrapingRequest | None = None,
        url: str | None = None,
        **kwargs,
    ) -> UnifiedScrapingResponse:
        """Unified scraping interface for all tiers.

        Args:
            request: Structured request object (recommended)
            url: URL to scrape (for simple usage)
            **kwargs: Additional parameters (tier, interaction_required, etc.)

        Returns:
            UnifiedScrapingResponse with standardized format

        Raises:
            CrawlServiceError: If manager not initialized or scraping fails
        """

        if not self._initialized:
            msg = "UnifiedBrowserManager not initialized"
            raise CrawlServiceError(msg)

        # Handle both structured and simple request formats
        if request is None:
            if url is None:
                msg = "Either request object or url parameter required"
                raise CrawlServiceError(msg)
            request = UnifiedScrapingRequest(url=url, **kwargs)

        start_time = time.time()

        # Check cache first if enabled
        cache_result = await self._try_get_cached_result(request, start_time)
        if cache_result:
            return cache_result

        response: UnifiedScrapingResponse | None = None
        try:
            result = await self._automation_router.scrape(
                url=request.url,
                interaction_required=request.interaction_required,
                custom_actions=request.custom_actions,
                force_tool=None if request.tier == "auto" else request.tier,
                timeout=request.timeout,
            )
            execution_time = (time.time() - start_time) * 1000
            response = await self._create_scraping_response(
                request, result, execution_time
            )

            # Cache successful results if enabled
            await self._try_cache_result(request, response)

            logger.info(
                "Unified scraping completed: %s via %s (%.1fms, quality: %.2f)",
                request.url,
                response.tier_used,
                execution_time,
                response.quality_score,
            )

        except Exception as exc:
            execution_time = (time.time() - start_time) * 1000
            self._update_tier_metrics("unknown", False, execution_time)
            await self._record_monitoring_metrics(
                "unknown", False, execution_time, error_type=type(exc).__name__
            )
            logger.exception("Unified scraping failed for %s", request.url)

            response = UnifiedScrapingResponse(
                success=False,
                content="",
                url=request.url,
                tier_used="none",
                execution_time_ms=execution_time,
                content_length=0,
                error=str(exc),
            )

        if response is None:
            msg = "Scraping response could not be generated"
            raise CrawlServiceError(msg)
        return response

    async def analyze_url(self, url: str) -> dict[str, Any]:
        """Analyze URL to determine optimal tier and provide insights.

        Args:
            url: URL to analyze

        Returns:
            Analysis results with tier recommendations
        """

        if not self._initialized:
            msg = "UnifiedBrowserManager not initialized"
            raise CrawlServiceError(msg)

        try:
            return await self._perform_url_analysis(url)

        except Exception as exc:
            logger.exception("URL analysis failed for %s", url)
            return {
                "url": url,
                "error": str(exc),
                "recommended_tier": "crawl4ai",  # Safe default
            }

    def get_tier_metrics(self) -> dict[str, TierMetrics]:
        """Get performance metrics for all tiers.

        Returns:
            Dictionary mapping tier names to their metrics

        """
        return self._tier_metrics.copy()

    def _get_router_metrics(self) -> dict[str, Any]:
        """Get automation router metrics safely."""
        if not self._automation_router:
            return {}

        try:
            return self._automation_router.get_metrics()
        except (ConnectionError, OSError, PermissionError):
            logger.debug("Failed to get router metrics")
            return {}

    def _calculate_overall_success_rate(self) -> tuple[int, int, float]:
        """Calculate overall success metrics.

        Returns:
            Tuple of (total_requests, total_successful, success_rate)
        """
        total_requests = sum(m.total_requests for m in self._tier_metrics.values())
        total_successful = sum(
            m.successful_requests for m in self._tier_metrics.values()
        )
        success_rate = total_successful / total_requests if total_requests > 0 else 0.0
        return total_requests, total_successful, success_rate

    def _get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics safely."""
        if not self._is_cache_available():
            return {}

        assert self._browser_cache is not None
        return self._browser_cache.get_stats()

    def _get_monitoring_health(self) -> dict[str, Any]:
        """Get monitoring system health safely."""
        if not (self._monitoring_enabled and self._monitor):
            return {}

        try:
            return self._monitor.get_system_health()
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.debug("Failed to get monitoring health")
            return {"error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status and health information."""
        if not self._initialized:
            return {"status": "not_initialized", "error": "Manager not initialized"}

        total_requests, _total_successful, overall_success_rate = (
            self._calculate_overall_success_rate()
        )

        return {
            "status": "healthy" if overall_success_rate > 0.8 else "degraded",
            "initialized": self._initialized,
            "total_requests": total_requests,
            "overall_success_rate": overall_success_rate,
            "tier_count": len(
                [m for m in self._tier_metrics.values() if m.total_requests > 0]
            ),
            "router_available": self._automation_router is not None,
            "cache_enabled": self._cache_enabled,
            "cache_stats": self._get_cache_stats(),
            "monitoring_enabled": self._monitoring_enabled,
            "monitoring_health": self._get_monitoring_health(),
            "router_metrics": self._get_router_metrics(),
            "tier_metrics": {
                name: metrics.model_dump()
                for name, metrics in self._tier_metrics.items()
            },
        }

    @staticmethod
    def _calculate_quality_score(result: dict[str, Any]) -> float:
        """Calculate content quality score based on content length.

        Args:
            result: Scraping result from AutomationRouter

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not result.get("success"):
            return 0.0

        content_length = len(result.get("content", ""))
        return min(content_length / 5000.0, 1.0)

    def _update_tier_metrics(
        self, tier: str, success: bool, execution_time_ms: float
    ) -> None:
        """Update performance metrics for a tier.

        Args:
            tier: Tier name
            success: Whether the operation was successful
            execution_time_ms: Execution time in milliseconds
        """

        if tier not in self._tier_metrics:
            self._tier_metrics[tier] = TierMetrics(tier_name=tier)

        metrics = self._tier_metrics[tier]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update rolling average response time
        total_time = (
            metrics.average_response_time_ms * (metrics.total_requests - 1)
            + execution_time_ms
        )
        metrics.average_response_time_ms = total_time / metrics.total_requests

        # Update success rate
        metrics.success_rate = metrics.successful_requests / metrics.total_requests
