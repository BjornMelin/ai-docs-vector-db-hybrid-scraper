
"""Function-based crawling service with FastAPI dependency injection.

Transforms the CrawlManager class into pure functions with dependency injection.
Provides crawling operations with circuit breaker patterns.
"""

import logging
from typing import Annotated
from typing import Any

from fastapi import Depends
from fastapi import HTTPException

from .circuit_breaker import CircuitBreakerConfig
from .circuit_breaker import circuit_breaker
from .dependencies import get_crawling_client

logger = logging.getLogger(__name__)


@circuit_breaker(CircuitBreakerConfig.enterprise_mode())
async def crawl_url(
    url: str,
    preferred_provider: str | None = None,
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Any]:
    """Scrape URL with intelligent 5-tier AutomationRouter selection.

    Pure function replacement for CrawlManager.scrape_url().
    Uses enterprise circuit breaker for higher resilience.

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
            raise HTTPException(status_code=500, detail="Crawling client not available")

        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        result = await crawling_client.scrape_url(
            url=url,
            preferred_provider=preferred_provider,
        )

        if result.get("success"):
            logger.info(
                f"Successfully crawled {url} using {result.get('tier_used', 'unknown')} "
                f"in {result.get('automation_time_ms', 0)}ms"
            )
        else:
            logger.warning(
                f"Failed to crawl {url}: {result.get('error', 'Unknown error')}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"URL crawling failed for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Crawling failed: {e!s}")


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
            raise HTTPException(status_code=500, detail="Crawling client not available")

        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        if max_pages <= 0 or max_pages > 1000:
            raise HTTPException(
                status_code=400, detail="max_pages must be between 1 and 1000"
            )

        result = await crawling_client.crawl_site(
            url=url,
            max_pages=max_pages,
            preferred_provider=preferred_provider,
        )

        if result.get("success"):
            logger.info(
                f"Successfully crawled {result.get('total_pages', 0)} pages from {url} "
                f"using {result.get('provider', 'unknown')} provider"
            )
        else:
            logger.warning(
                f"Site crawl failed for {url}: {result.get('error', 'Unknown error')}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Site crawling failed for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Site crawling failed: {e!s}")


async def get_crawl_metrics(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Dict]:
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

        logger.debug(f"Retrieved crawl metrics for {len(metrics)} tiers")
        return metrics

    except Exception as e:
        logger.exception(f"Crawl metrics retrieval failed: {e}")
        return {}


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
            raise HTTPException(status_code=400, detail="URL is required")

        recommendation = await crawling_client.get_recommended_tool(url)

        logger.debug(f"Recommended tool for {url}: {recommendation}")
        return recommendation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Tool recommendation failed for {url}: {e}")
        return "crawl4ai"  # Graceful fallback


async def get_provider_info(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Dict]:
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

        logger.debug(f"Retrieved provider info for {len(info)} tools")
        return info

    except Exception as e:
        logger.exception(f"Provider info retrieval failed: {e}")
        return {}


async def get_tier_metrics(
    crawling_client: Annotated[object, Depends(get_crawling_client)] = None,
) -> dict[str, Dict]:
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

        logger.debug(f"Retrieved tier metrics for {len(metrics)} tiers")
        return metrics

    except Exception as e:
        logger.exception(f"Tier metrics retrieval failed: {e}")
        return {}


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
        import asyncio

        if not urls:
            return []

        if max_parallel <= 0 or max_parallel > 20:
            raise HTTPException(
                status_code=400, detail="max_parallel must be between 1 and 20"
            )

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
                logger.error(f"URL {urls[i]} failed: {result}")
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "url": urls[i],
                        "content": "",
                        "metadata": {},
                    }
                )
            else:
                processed_results.append(result)

        successful_count = sum(1 for r in processed_results if r.get("success", False))
        logger.info(
            f"Batch crawled {len(urls)} URLs with {successful_count} successes "
            f"using max {max_parallel} parallel operations"
        )

        return processed_results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch URL crawling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch crawling failed: {e!s}")


async def validate_url(url: str) -> dict[str, Any]:
    """Validate URL format and accessibility.

    New function-based utility for URL validation.

    Args:
        url: URL to validate

    Returns:
        Validation result with status and details
    """
    try:
        import re
        from urllib.parse import urlparse

        if not url:
            return {"valid": False, "error": "URL is required", "details": {}}

        # Basic URL format validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "valid": False,
                "error": "Invalid URL format",
                "details": {
                    "parsed": {"scheme": parsed.scheme, "netloc": parsed.netloc}
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

    except Exception as e:
        logger.exception(f"URL validation failed for {url}: {e}")
        return {"valid": False, "error": f"Validation error: {e!s}", "details": {}}


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
        if len(urls) <= 10:
            resource_requirements = "low"
        elif len(urls) <= 100:
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
                "batch_size": min(10, len(urls)),
                "parallel_limit": 5 if len(urls) > 20 else 3,
            },
        }

    except Exception as e:
        logger.exception(f"Crawl cost estimation failed: {e}")
        return {
            "total_urls": len(urls),
            "estimated_time_minutes": 0,
            "estimated_pages": 0,
            "resource_requirements": "unknown",
            "error": str(e),
        }
