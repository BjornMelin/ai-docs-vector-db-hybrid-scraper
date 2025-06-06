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
from ...services.crawling.lightweight_scraper import LightweightScraper
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

        # Get or create lightweight scraper instance
        config = client_manager.config
        if not hasattr(client_manager, "_lightweight_scraper"):
            if ctx:
                await ctx.debug("Creating new lightweight scraper instance")
            client_manager._lightweight_scraper = LightweightScraper(
                config.lightweight_scraper,
                rate_limiter=None,  # Could add rate limiting if needed
            )
            await client_manager._lightweight_scraper.initialize()

        scraper = client_manager._lightweight_scraper

        # Check if URL can be handled by lightweight tier
        can_handle = await scraper.can_handle(url)
        if not can_handle and ctx:
            await ctx.warning(
                f"URL {url} is not suitable for lightweight scraping. "
                "Consider using standard search or crawl tools for complex pages."
            )

        # Perform the scrape
        try:
            import time

            start_time = time.time()

            result = await scraper.scrape_url(url, formats=formats)

            elapsed_ms = (time.time() - start_time) * 1000

            if result.get("success"):
                if ctx:
                    await ctx.info(f"Successfully scraped {url} in {elapsed_ms:.0f}ms")

                # Add performance metrics
                result["performance"] = {
                    "elapsed_ms": elapsed_ms,
                    "tier": "lightweight",
                    "suitable_for_tier": can_handle,
                }

                return result
            else:
                error_msg = result.get("error", "Unknown error")
                should_escalate = result.get("should_escalate", False)

                if ctx:
                    await ctx.error(f"Failed to scrape {url}: {error_msg}")
                    if should_escalate:
                        await ctx.info(
                            "This content requires browser-based scraping. "
                            "Consider using standard search or crawl tools."
                        )

                raise CrawlServiceError(
                    f"Lightweight scraping failed: {error_msg}. "
                    f"{'Try browser-based tools for this content.' if should_escalate else ''}"
                )

        except Exception as e:
            if ctx:
                await ctx.error(f"Unexpected error during lightweight scrape: {e!s}")
            raise
