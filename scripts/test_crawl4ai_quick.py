#!/usr/bin/env python3
"""Quick test to verify Crawl4AI is working before running full benchmark."""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415
import time  # noqa: PLC0415

from src.services.crawling.crawl4ai_provider import Crawl4AIProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_crawl4ai():
    """Quick test of Crawl4AI functionality."""
    from src.config import Crawl4AIConfig

    config = Crawl4AIConfig(
        max_concurrent_crawls=1,
        headless=True,
        browser_type="chromium",
        page_timeout=30.0,
    )

    provider = Crawl4AIProvider(config=config)

    try:
        # Initialize
        logger.info("Initializing Crawl4AI...")
        await provider.initialize()

        # Test single URL
        test_url = "https://docs.python.org/3/library/asyncio.html"
        logger.info(f"Testing URL: {test_url}")

        start = time.time()
        result = await provider.scrape_url(test_url)
        elapsed = time.time() - start

        if result.get("success"):
            logger.info(f"✅ Success! Crawled in {elapsed:.2f}s")
            logger.info(f"Content length: {len(result.get('content', ''))} chars")
            logger.info(f"Title: {result.get('title', 'N/A')}")
        else:
            logger.error(f"❌ Failed: {result.get('error')}")

        # Cleanup
        await provider.cleanup()

        return result

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        if hasattr(provider, "cleanup"):
            await provider.cleanup()
        raise


if __name__ == "__main__":
    asyncio.run(test_crawl4ai())
