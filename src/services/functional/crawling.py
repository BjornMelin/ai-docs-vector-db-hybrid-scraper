"""Function-based crawling service with FastAPI dependency injection.

Transforms the CrawlManager class into pure functions with dependency injection.
Provides crawling operations with circuit breaker patterns.
"""

import asyncio
import logging
import re
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Depends, HTTPException

from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_crawling_client


logger = logging.getLogger(__name__)

# Constants for crawling limits and thresholds
MAX_PAGES_LIMIT = 1000
MAX_PARALLEL_LIMIT = 20
LOW_RESOURCE_THRESHOLD = 10
MEDIUM_RESOURCE_THRESHOLD = 100
HIGH_PARALLEL_THRESHOLD = 20
LOW_PARALLEL_TASKS = 3
HIGH_PARALLEL_TASKS = 5
DEFAULT_BATCH_SIZE = 10


@circuit_breaker(CircuitBreakerConfig.enterprise_mode())
async def crawl_url(
    url: str,
    preferred_provider: str | None = None,
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Any]:
    """Scrape URL with AutomationRouter selection.

    Pure function replacement for CrawlManager.scrape_url().
    Uses circuit breaker for resilience.

    Args:
        url: URL to scrape
        preferred_provider: Specific tool to force use (overrides selection logic)
        crawling_client: Injected crawl manager

    Returns:
        Scraping result with success status, content, and metadata

    Raises:
        HTTPException: If crawling fails critically

    """
    try:
        if not crawling_client:
            _raise_crawling_client_unavailable()

        if not url:
            _raise_url_required()

        result = await crawling_client.scrape_url(
            url=url,
            preferred_provider=preferred_provider,
        )

        if result.get("success"):
            tier = result.get("tier_used", "unknown")
            time_ms = result.get("automation_time_ms", 0)
            logger.info("Successfully crawled %s using %s in %dms", url, tier, time_ms)
        else:
            logger.warning(
                "Failed to crawl %s: %s",
                url,
                result.get("error", "Unknown error"),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("URL crawling failed for %s", url)
        raise HTTPException(status_code=500, detail=f"Crawling failed: {e!s}") from e
    else:
        return result


@circuit_breaker(CircuitBreakerConfig.enterprise_mode())
async def crawl_site(
    url: str,
    max_pages: int = 50,
    preferred_provider: str | None = None,
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Any]:
    """Crawl entire website from starting URL using AutomationRouter.

    Pure function replacement for CrawlManager.crawl_site().

    Args:
        url: Starting URL for crawl
        max_pages: Maximum pages to crawl (default: 50)
        preferred_provider: Specific tool to use for crawling
        crawling_client: Injected crawl manager

    Returns:
        Crawl results with pages list and metadata

    Raises:
        HTTPException: If site crawling fails

    """
    try:
        if not crawling_client:
            _raise_crawling_client_unavailable()

        if not url:
            _raise_url_required()

        if max_pages <= 0 or max_pages > MAX_PAGES_LIMIT:
            _raise_invalid_max_pages()

        result = await crawling_client.crawl_site(
            url=url,
            max_pages=max_pages,
            preferred_provider=preferred_provider,
        )

        if result.get("success"):
            logger.info(
                "Successfully crawled %d pages from %s using %s provider",
                result.get("total_pages", 0),
                url,
                result.get("provider", "unknown"),
            )
        else:
            logger.warning(
                "Site crawl failed for %s: %s",
                url,
                result.get("error", "Unknown error"),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Site crawling failed for %s", url)
        raise HTTPException(
            status_code=500,
            detail=f"Site crawling failed: {e!s}",
        ) from e
    return result


async def get_crawl_metrics(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, dict]:
    """Get performance metrics for all crawling tiers.

    Pure function replacement for CrawlManager.get_metrics().

    Args:
        crawling_client: Injected crawl manager

    Returns:
        Dictionary with metrics for each tier including success rates and timing

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        if not crawling_client:
            return {}

        metrics = crawling_client.get_metrics()
        logger.debug("Retrieved crawl metrics for %d tiers", len(metrics))

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Crawl metrics retrieval failed")
        return {}
    return metrics


async def get_recommended_tool(
    url: str,
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> str:
    """Get recommended crawling tool for a URL based on performance metrics.

    Pure function replacement for CrawlManager.get_recommended_tool().

    Args:
        url: URL to analyze
        crawling_client: Injected crawl manager

    Returns:
        Recommended tool name based on UnifiedBrowserManager analysis

    Raises:
        HTTPException: If tool recommendation fails

    """
    try:
        if not crawling_client:
            return "crawl4ai"  # Default fallback

        if not url:
            _raise_url_required()

        recommendation = await crawling_client.get_recommended_tool(url)
        logger.debug("Recommended tool for %s: %s", url, recommendation)

    except HTTPException:
        raise
    except (ConnectionError, OSError, PermissionError):
        logger.exception("Tool recommendation failed for %s", url)
        return "crawl4ai"  # Graceful fallback
    else:
        return recommendation


async def get_provider_info(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, dict]:
    """Get information about available automation tools in 5-tier system.

    Pure function replacement for CrawlManager.get_provider_info().

    Args:
        crawling_client: Injected crawl manager

    Returns:
        Tool information including tier assignments and metrics

    Raises:
        HTTPException: If provider info retrieval fails
    """
    try:
        if not crawling_client:
            return {}

        info = crawling_client.get_provider_info()
        logger.debug("Retrieved provider info for %d tools", len(info))

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Provider info retrieval failed")
        return {}
    return info


async def get_tier_metrics(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, dict]:
    """Get performance metrics for each tier from UnifiedBrowserManager.

    Pure function replacement for CrawlManager.get_tier_metrics().

    Args:
        crawling_client: Injected crawl manager

    Returns:
        Tier performance metrics for all 5 tiers

    Raises:
        HTTPException: If tier metrics retrieval fails
    """
    try:
        if not crawling_client:
            return {}

        metrics = crawling_client.get_tier_metrics()
        logger.debug("Retrieved tier metrics for %d tiers", len(metrics))

    except (ConnectionError, OSError, PermissionError):
        logger.exception("Tier metrics retrieval failed")
        return {}
    return metrics


# New function-based capabilities
@circuit_breaker(CircuitBreakerConfig.enterprise_mode())
async def batch_crawl_urls(
    urls: list[str],
    preferred_provider: str | None = None,
    max_parallel: int = 5,
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> list[dict[str, Any]]:
    """Crawl multiple URLs in parallel with concurrency control.

    New function-based capability demonstrating composition patterns.

    Args:
        urls: List of URLs to crawl
        preferred_provider: Specific tool to use for all URLs
        max_parallel: Maximum parallel crawl operations
        crawling_client: Injected crawl manager

    Returns:
        List of crawling results for each URL

    Raises:
        HTTPException: If batch crawling fails

    """
    try:
        if not urls:
            return []

        if max_parallel <= 0 or max_parallel > MAX_PARALLEL_LIMIT:
            _raise_invalid_max_parallel()

        semaphore = asyncio.Semaphore(max_parallel)

        async def crawl_single_url(url: str) -> dict[str, Any]:
            async with semaphore:
                return await crawl_url(
                    url=url,
                    preferred_provider=preferred_provider,
                    crawling_client=crawling_client,
                )

        # Process URLs in parallel with concurrency control
        tasks = [crawl_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("URL %s failed: %s", urls[i], result)
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "url": urls[i],
                        "content": "",
                        "metadata": {},
                    },
                )
            else:
                processed_results.append(result)

        successful_count = sum(1 for r in processed_results if r.get("success", False))
        logger.info(
            "Batch crawled %d URLs with %d successes using max %d parallel operations",
            len(urls),
            successful_count,
            max_parallel,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch URL crawling failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch crawling failed: {e!s}",
        ) from e
    return processed_results


async def validate_url(url: str) -> dict[str, Any]:
    """Validate URL format and accessibility.

    New function-based utility for URL validation.

    Args:
        url: URL to validate

    Returns:
        Validation result with status and details
    """
    try:
        if not url:
            return {"valid": False, "error": "URL is required", "details": {}}

        # Basic URL format validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "valid": False,
                "error": "Invalid URL format",
                "details": {
                    "parsed": {"scheme": parsed.scheme, "netloc": parsed.netloc},
                },
            }

        # Check for supported schemes
        if parsed.scheme not in ["http", "https"]:
            return {
                "valid": False,
                "error": f"Unsupported URL scheme: {parsed.scheme}",
                "details": {"scheme": parsed.scheme},
            }

        # Basic domain validation
        domain_pattern = re.compile(
            r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
        )
        if not domain_pattern.match(parsed.netloc.split(":")[0]):
            return {
                "valid": False,
                "error": "Invalid domain format",
                "details": {"domain": parsed.netloc},
            }

    except Exception as e:
        logger.exception("URL validation failed for %s", url)
        return {"valid": False, "error": f"Validation error: {e!s}", "details": {}}
    return {
        "valid": True,
        "details": {
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
        },
    }


async def estimate_crawl_cost(
    urls: list[str],
    max_pages_per_site: int = 50,
) -> dict[str, Any]:
    """Estimate crawling cost and time for URLs.

    New function-based utility for cost estimation.

    Args:
        urls: List of URLs to estimate
        max_pages_per_site: Maximum pages per site crawl

    Returns:
        Cost estimation with time and resource requirements
    """
    try:
        if not urls:
            return {
                "total_urls": 0,
                "estimated_time_minutes": 0,
                "estimated_pages": 0,
                "resource_requirements": "none",
            }

        # Simple estimation model
        base_time_per_url = 3  # seconds
        time_per_page = 2  # seconds for site crawls

        single_urls = len(urls)
        estimated_pages = single_urls * max_pages_per_site

        estimated_time_seconds = (
            single_urls * base_time_per_url + estimated_pages * time_per_page
        )
        estimated_time_minutes = estimated_time_seconds / 60

        # Resource requirements estimation
        if len(urls) <= LOW_RESOURCE_THRESHOLD:
            resource_requirements = "low"
        elif len(urls) <= MEDIUM_RESOURCE_THRESHOLD:
            resource_requirements = "medium"
        else:
            resource_requirements = "high"

        return {
            "total_urls": len(urls),
            "estimated_time_minutes": round(estimated_time_minutes, 1),
            "estimated_pages": estimated_pages,
            "estimated_time_seconds": estimated_time_seconds,
            "resource_requirements": resource_requirements,
            "recommendations": {
                "batch_size": min(DEFAULT_BATCH_SIZE, len(urls)),
                "parallel_limit": HIGH_PARALLEL_TASKS
                if len(urls) > HIGH_PARALLEL_THRESHOLD
                else LOW_PARALLEL_TASKS,
            },
        }

    except Exception as e:
        logger.exception("Crawl cost estimation failed")
        return {
            "total_urls": len(urls),
            "estimated_time_minutes": 0,
            "estimated_pages": 0,
            "resource_requirements": "unknown",
            "error": str(e),
        }


def _raise_url_required() -> None:
    """Raise HTTPException for missing URL."""
    raise HTTPException(status_code=400, detail="URL is required")


def _raise_invalid_max_pages() -> None:
    """Raise HTTPException for invalid max_pages."""
    raise HTTPException(status_code=400, detail="max_pages must be between 1 and 1000")


def _raise_invalid_max_parallel() -> None:
    """Raise HTTPException for invalid max_parallel."""
    raise HTTPException(status_code=400, detail="max_parallel must be between 1 and 20")


def _raise_crawling_client_unavailable() -> None:
    """Raise HTTPException for unavailable crawling client."""
    raise HTTPException(status_code=500, detail="Crawling client not available")
