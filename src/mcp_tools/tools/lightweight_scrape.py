"""Lightweight HTTP scraping tool for MCP server."""

import logging
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from ...infrastructure.client_manager import ClientManager
from ...services.errors import CrawlServiceError

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register lightweight scraping tools with the MCP server."""

    @mcp.tool()
    async def lightweight_scrape(
        url: str,
        formats: list[Literal["markdown", "html", "text"]] | None = None,
        ctx: Context | None = None,
    ) -> dict:
        """
        Ultra-fast web scraping for simple static pages using httpx + BeautifulSoup.

        This tool provides 5-10x faster scraping for static content compared to
        browser-based scrapers. It's ideal for:
        - Documentation sites
        - Raw text/markdown files (GitHub raw content)
        - Simple HTML pages without JavaScript
        - JSON/XML API endpoints

        **When to use this tool:**
        - You know the page is static (no JavaScript required)
        - You need maximum speed for simple content
        - You're scraping documentation or raw files

        **When NOT to use this tool:**
        - JavaScript-heavy single-page applications (SPAs)
        - Pages requiring user interaction
        - Complex pages with dynamic content
        - When you need browser features (screenshots, etc.)

        Args:
            url: The URL to scrape
            formats: Output formats to return (default: ["markdown"])
                    Options: "markdown", "html", "text"
            ctx: MCP context for logging

        Returns:
            Dict containing:
            - success: Whether scraping succeeded
            - content: Requested formats with extracted content
            - metadata: Page metadata (title, description, etc.)
            - performance: Timing information

        Raises:
            ValueError: If URL is invalid or formats are invalid
            CrawlServiceError: If scraping fails
        """
        if ctx:
            await ctx.info(f"Starting lightweight scrape of {url}")

        # Validate formats
        if formats is None:
            formats = ["markdown"]

        valid_formats = {"markdown", "html", "text"}
        invalid_formats = set(formats) - valid_formats
        if invalid_formats:
            raise ValueError(
                f"Invalid formats: {invalid_formats}. Valid options: {valid_formats}"
            )

        # Get CrawlManager which uses UnifiedBrowserManager
        crawl_manager = await client_manager.get_crawl_manager()

        if ctx:
            await ctx.debug("Using UnifiedBrowserManager with lightweight tier")

        # Get unified browser manager for tier analysis
        unified_manager = crawl_manager._unified_browser_manager
        if unified_manager and hasattr(unified_manager, "analyze_url"):
            try:
                analysis = await unified_manager.analyze_url(url)
                can_handle = analysis.get("recommended_tier") == "lightweight"
                if not can_handle and ctx:
                    await ctx.warning(
                        f"URL {url} may not be optimal for lightweight scraping. "
                        f"Recommended tier: {analysis.get('recommended_tier', 'unknown')}. "
                        "Consider using standard search or crawl tools for complex pages."
                    )
            except Exception:
                can_handle = True  # Default to allowing the attempt
        else:
            can_handle = True  # Default to allowing the attempt

        # Perform the scrape using UnifiedBrowserManager with forced lightweight tier
        try:
            import time

            start_time = time.time()

            # Force lightweight tier by specifying preferred_provider
            result = await crawl_manager.scrape_url(
                url=url, preferred_provider="lightweight"
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if result.get("success"):
                if ctx:
                    await ctx.info(
                        f"Successfully scraped {url} in {elapsed_ms:.0f}ms using {result.get('tier_used', 'unknown')} tier"
                    )

                # Transform result to match expected format
                content_dict = {}
                content = result.get("content", "")

                # Convert to requested formats
                for fmt in formats:
                    if fmt == "markdown":
                        content_dict["markdown"] = (
                            content  # Assume content is already markdown
                        )
                    elif fmt == "html":
                        content_dict["html"] = result.get("metadata", {}).get(
                            "raw_html", content
                        )
                    elif fmt == "text":
                        # Strip markdown formatting for plain text
                        import re

                        text_content = re.sub(r"[*_`#\[\]()]", "", content)
                        content_dict["text"] = text_content

                # Build response matching expected format
                response = {
                    "success": True,
                    "content": content_dict,
                    "metadata": {
                        "title": result.get("title", ""),
                        "url": result.get("url", url),
                        "tier_used": result.get("tier_used", "lightweight"),
                        "quality_score": result.get("quality_score", 0.0),
                        **result.get("metadata", {}),
                    },
                    "performance": {
                        "elapsed_ms": elapsed_ms,
                        "tier": result.get("tier_used", "lightweight"),
                        "suitable_for_tier": can_handle,
                        "fallback_attempted": result.get("fallback_attempted", False),
                    },
                }

                return response
            else:
                error_msg = result.get("error", "Unknown error")
                failed_tiers = result.get("failed_tiers", [])

                if ctx:
                    await ctx.error(f"Failed to scrape {url}: {error_msg}")
                    if "lightweight" in failed_tiers:
                        await ctx.info(
                            "Lightweight tier failed. This content requires browser-based scraping. "
                            "Consider using standard search or crawl tools."
                        )

                raise CrawlServiceError(
                    f"Lightweight scraping failed: {error_msg}. "
                    f"{'Try browser-based tools for this content.' if 'lightweight' in failed_tiers else ''}"
                )

        except Exception as e:
            if ctx:
                await ctx.error(f"Unexpected error during lightweight scrape: {e!s}")
            raise
