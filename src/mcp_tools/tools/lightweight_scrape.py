"""Lightweight scraping helpers exposed via the MCP tool interface."""

import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fastmcp import Context

from src.services.browser.models import ProviderKind
from src.services.errors import CrawlServiceError
from src.services.service_resolver import get_crawl_manager


logger = logging.getLogger(__name__)


def _validate_formats(formats: Sequence[str] | None) -> list[str]:
    """Validate and normalize requested output formats."""
    if formats is None:
        return ["markdown"]

    normalized_formats = list(formats)
    valid_formats = frozenset({"markdown", "html", "text"})
    if invalid_formats := set(normalized_formats) - valid_formats:
        msg = f"Invalid formats: {invalid_formats}. Valid options: {valid_formats}"
        raise ValueError(msg)
    return normalized_formats


async def _analyze_url_suitability(
    crawl_manager, url: str, ctx: Context | None
) -> bool:
    """Analyze URL to determine if lightweight tier is suitable."""
    if not (crawl_manager and hasattr(crawl_manager, "analyze_url")):
        return True  # Default to allowing the attempt

    try:
        analysis = await crawl_manager.analyze_url(url)
        can_handle = analysis.get("recommended_tier") == "lightweight"
        if not can_handle and ctx:
            await ctx.warning(
                f"URL {url} may not be optimal for lightweight scraping. "
                f"Recommended tier: {analysis.get('recommended_tier', 'unknown')}. "
                "Consider using standard search or crawl tools for complex pages."
            )
    except (AttributeError, ConnectionError, RuntimeError, TimeoutError):
        return True  # Default to allowing the attempt

    return can_handle


def _convert_content_formats(
    content: str, formats: Sequence[str], result: Mapping[str, object]
) -> dict[str, str]:
    """Convert content to requested formats."""
    content_dict = {}

    metadata = result.get("metadata")
    raw_html = content
    if isinstance(metadata, Mapping):
        raw_html = metadata.get("raw_html", content)

    for fmt in formats:
        if fmt == "markdown":
            content_dict["markdown"] = content  # Assume content is already markdown
        elif fmt == "html":
            content_dict["html"] = raw_html
        elif fmt == "text":
            # Strip markdown formatting for plain text
            text_content = re.sub(r"[*_`#\[\]()]", "", content).strip()
            content_dict["text"] = text_content

    return content_dict


def _build_success_response(
    result: Mapping[str, object],
    url: str,
    formats: Sequence[str],
    elapsed_ms: float,
    can_handle: bool,
) -> dict:
    """Build successful response dictionary."""
    metadata = result.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
    content = str(result.get("content", ""))
    content_dict = _convert_content_formats(content, formats, result)
    provider_raw = result.get("provider", ProviderKind.LIGHTWEIGHT.value)
    provider = (
        provider_raw.value
        if isinstance(provider_raw, ProviderKind)
        else str(provider_raw)
    )

    return {
        "success": True,
        "content": content_dict,
        "metadata": {
            "title": result.get("title", ""),
            "url": result.get("url", url),
            "provider": provider,
            "quality_score": result.get("quality_score", 0.0),
            **metadata_dict,
        },
        "performance": {
            "elapsed_ms": elapsed_ms,
            "provider": provider,
            "suitable_for_tier": can_handle,
            "fallback_attempted": result.get("fallback_attempted", False),
        },
    }


async def _handle_scrape_failure(
    result: Mapping[str, object], url: str, ctx: Context | None
) -> None:
    """Handle scrape failure with appropriate logging and error raising."""
    error_value = result.get("error", "Unknown error")
    error_msg = str(error_value)
    failed_tiers_value = result.get("failed_tiers")
    if isinstance(failed_tiers_value, list | tuple | set):
        failed_tiers = list(failed_tiers_value)
    elif failed_tiers_value is not None:
        failed_tiers = [str(failed_tiers_value)]
    else:
        failed_tiers = []

    if ctx:
        await ctx.error(f"Failed to scrape {url}: {error_msg}")
        if "lightweight" in failed_tiers:
            await ctx.info(
                "Lightweight tier failed. This content requires browser-based "
                "scraping. Consider using standard search or crawl tools."
            )

    msg = (
        f"Lightweight scraping failed: {error_msg}. "
        f"{'Try browser-based tools for this content.' if 'lightweight' in failed_tiers else ''}"  # noqa: E501
    )
    raise CrawlServiceError(msg)


def register_tools(
    mcp,
    crawl_manager: Any | None = None,
) -> None:
    """Register lightweight scraping tools with the MCP server."""

    @mcp.tool()
    async def lightweight_scrape(
        url: str,
        formats: Sequence[Literal["markdown", "html", "text"]] | None = None,
        ctx: Context | None = None,
    ) -> dict:
        """Web scraping for simple static pages using httpx + BeautifulSoup.

        This tool provides faster scraping for static content compared to
        browser-based scrapers. Suitable for:
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
        validated_formats = _validate_formats(formats)

        # Get CrawlManager which uses UnifiedBrowserManager
        manager = crawl_manager or await get_crawl_manager()

        if ctx:
            await ctx.debug("Using UnifiedBrowserManager with lightweight tier")

        # Analyze URL suitability for lightweight tier
        can_handle = await _analyze_url_suitability(manager, url, ctx)

        # Perform the scrape using UnifiedBrowserManager with forced lightweight tier
        try:
            start_time = time.time()

            # Force lightweight tier by specifying preferred_provider
            result = await manager.scrape_url(
                url=url, preferred_provider=ProviderKind.LIGHTWEIGHT.value
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if result.get("success"):
                if ctx:
                    provider_raw = result.get("provider", "unknown")
                    provider_display = (
                        provider_raw.value
                        if isinstance(provider_raw, ProviderKind)
                        else str(provider_raw)
                    )
                    await ctx.info(
                        f"Successfully scraped {url} in {elapsed_ms:.0f}ms using "
                        f"{provider_display} provider"
                    )
                return _build_success_response(
                    result, url, validated_formats, elapsed_ms, can_handle
                )
            await _handle_scrape_failure(result, url, ctx)
            raise CrawlServiceError("Lightweight scraping failed without details.")

        except Exception as e:
            if ctx:
                await ctx.error(f"Unexpected error during lightweight scrape: {e!s}")
            raise
