"""Thin MCP crawling tools that delegate to the automation router."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from fastmcp import Context

from src.services.service_resolver import get_crawl_manager


logger = logging.getLogger(__name__)


def _validate_url(url: str) -> str:
    """Ensure the provided URL contains a scheme and network location."""

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        msg = "Invalid URL format"
        raise ValueError(msg)
    return url


def register_tools(mcp, crawl_manager: Any | None = None) -> None:
    """Register crawling tools with the MCP server."""

    @mcp.tool()
    async def enhanced_5_tier_crawl(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        url: str,
        tier: str | None = None,
        interaction_required: bool = False,
        custom_actions: list[dict[str, Any]] | None = None,
        timeout_ms: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Delegate crawling to the UnifiedBrowserManager router."""

        validated_url = _validate_url(url)
        if ctx:
            await ctx.info(f"Dispatching crawl request for {validated_url}")

        manager = crawl_manager or await get_crawl_manager()

        try:
            result = await manager.scrape_url(
                url=validated_url,
                preferred_provider=tier,
                require_interaction=interaction_required,
                timeout_ms=timeout_ms,
                actions=custom_actions,
            )
        except Exception as exc:
            logger.exception("Enhanced crawl failed for %s", validated_url)
            if ctx:
                await ctx.error(f"Enhanced crawl failed: {exc}")
            raise

        if ctx:
            await ctx.debug(
                f"Unified router selected provider: {result.provider.value}",
            )
        return {
            "success": result.success,
            "provider": result.provider.value,
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "html": result.html,
            "metadata": dict(result.metadata),
            "links": result.links,
            "assets": result.assets,
            "elapsed_ms": result.elapsed_ms,
        }

    @mcp.tool()
    async def get_crawling_capabilities() -> dict[str, Any]:
        """Describe supported tiers and their characteristics."""

        return {
            "available_tiers": [
                "lightweight",
                "crawl4ai",
                "playwright",
                "browser_use",
                "firecrawl",
            ],
            "tier_capabilities": {
                "lightweight": {
                    "speed": "fastest",
                    "js_support": False,
                    "anti_detection": False,
                    "best_for": ["static_content", "simple_pages"],
                },
                "crawl4ai": {
                    "speed": "fast",
                    "js_support": True,
                    "anti_detection": "basic",
                    "best_for": ["dynamic_content", "spa_apps"],
                },
                "playwright": {
                    "speed": "medium",
                    "js_support": True,
                    "anti_detection": "advanced",
                    "best_for": ["complex_interactions", "forms"],
                },
                "browser_use": {
                    "speed": "slow",
                    "js_support": True,
                    "anti_detection": "advanced",
                    "best_for": ["complex_workflows", "human_like_interaction"],
                },
                "firecrawl": {
                    "speed": "medium",
                    "js_support": True,
                    "anti_detection": "professional",
                    "best_for": ["protected_content", "enterprise_sites"],
                },
            },
        }
