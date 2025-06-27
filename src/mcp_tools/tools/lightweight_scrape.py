"""Lightweight HTTP scraping tool for MCP server."""

import logging  # noqa: PLC0415
from typing import TYPE_CHECKING, Literal


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


def _validate_formats(formats: list[str] | None) -> list[str]:
    """Validate and normalize format list."""
    if formats is None:
        formats = ["markdown"]

    valid_formats = {"markdown", "html", "text"}
    invalid_formats = set(formats) - valid_formats
    if invalid_formats:
        raise ValueError(
            f"Invalid formats: {invalid_formats}. Valid options: {valid_formats}"
        )
    return formats


async def _analyze_url_suitability(
    unified_manager, url: str, ctx: Context | None
) -> bool:
    """Analyze URL to determine if lightweight tier is suitable."""
    if not (unified_manager and hasattr(unified_manager, "analyze_url")):
        return True  # Default to allowing the attempt

    try:
        analysis = await unified_manager.analyze_url(url)
        can_handle = analysis.get("recommended_tier") == "lightweight"
        if not can_handle and ctx:
            await ctx.warning(
                f"URL {url} may not be optimal for lightweight scraping. "
                f"Recommended tier: {analysis.get('recommended_tier', 'unknown')}. "
                "Consider using standard search or crawl tools for complex pages."
            )
        return can_handle
    except Exception:
        return True  # Default to allowing the attempt


def _convert_content_formats(
    content: str, formats: list[str], result: dict
) -> dict[str, str]:
    """Convert content to requested formats."""
    content_dict = {}

    for fmt in formats:
        if fmt == "markdown":
            content_dict["markdown"] = content  # Assume content is already markdown
        elif fmt == "html":
            content_dict["html"] = result.get("metadata", {}).get("raw_html", content)
        elif fmt == "text":
            # Strip markdown formatting for plain text
            import re  # noqa: PLC0415

            text_content = re.sub(r"[*_`#\[\]()]", "", content)
            content_dict["text"] = text_content

    return content_dict


def _build_success_response(
    result: dict, url: str, formats: list[str], elapsed_ms: float, can_handle: bool
) -> dict:
    """Build successful response dictionary."""
    content = result.get("content", "")
    content_dict = _convert_content_formats(content, formats, result)

    return {
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


async def _handle_scrape_failure(result: dict, url: str, ctx: Context | None) -> None:
    """Handle scrape failure with appropriate logging and error raising."""
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

        # Validate and normalize formats
        formats = _validate_formats(formats)

        # Get CrawlManager which uses UnifiedBrowserManager
        crawl_manager = await client_manager.get_crawl_manager()

        if ctx:
            await ctx.debug("Using UnifiedBrowserManager with lightweight tier")

        # Analyze URL suitability for lightweight tier
        unified_manager = crawl_manager._unified_browser_manager
        can_handle = await _analyze_url_suitability(unified_manager, url, ctx)

        # Perform the scrape using UnifiedBrowserManager with forced lightweight tier
        try:
            import time  # noqa: PLC0415

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
                return _build_success_response(
                    result, url, formats, elapsed_ms, can_handle
                )
            else:
                await _handle_scrape_failure(result, url, ctx)

        except Exception as e:
            if ctx:
                await ctx.error(f"Unexpected error during lightweight scrape: {e!s}")
            raise
