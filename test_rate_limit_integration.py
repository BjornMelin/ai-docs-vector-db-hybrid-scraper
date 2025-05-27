#!/usr/bin/env python3
"""Test rate limiter integration with service managers."""

import asyncio
import os

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["FIRECRAWL_API_KEY"] = "test-key"

from src.config import get_config
from src.mcp.service_manager import UnifiedServiceManager
from src.services.crawling.manager import CrawlManager
from src.services.embeddings.manager import EmbeddingManager


async def test_integration():
    """Test that rate limiters are properly integrated."""

    # Test with UnifiedServiceManager
    print("Testing UnifiedServiceManager...")
    service_manager = UnifiedServiceManager()
    await service_manager.initialize()

    assert service_manager.rate_limiter is not None, (
        "Rate limiter should be initialized"
    )
    assert service_manager.embedding_manager.rate_limiter is not None, (
        "Embedding manager should have rate limiter"
    )
    assert service_manager.crawl_manager.rate_limiter is not None, (
        "Crawl manager should have rate limiter"
    )

    # Check that providers have rate limiters
    if "openai" in service_manager.embedding_manager.providers:
        provider = service_manager.embedding_manager.providers["openai"]
        assert provider.rate_limiter is not None, (
            "OpenAI provider should have rate limiter"
        )
        print("✓ OpenAI provider has rate limiter")

    if "firecrawl" in service_manager.crawl_manager.providers:
        provider = service_manager.crawl_manager.providers["firecrawl"]
        assert provider.rate_limiter is not None, (
            "Firecrawl provider should have rate limiter"
        )
        print("✓ Firecrawl provider has rate limiter")

    await service_manager.cleanup()
    print("✓ UnifiedServiceManager integration test passed")

    # Test direct initialization
    print("\nTesting direct initialization...")
    config = get_config()

    from src.services.rate_limiter import RateLimitManager

    rate_limiter = RateLimitManager(config)

    embedding_manager = EmbeddingManager(config, rate_limiter=rate_limiter)
    await embedding_manager.initialize()

    assert embedding_manager.rate_limiter is not None
    if "openai" in embedding_manager.providers:
        assert embedding_manager.providers["openai"].rate_limiter is not None
        print("✓ Direct EmbeddingManager initialization passed")

    crawl_manager = CrawlManager(config, rate_limiter=rate_limiter)
    await crawl_manager.initialize()

    assert crawl_manager.rate_limiter is not None
    if "firecrawl" in crawl_manager.providers:
        assert crawl_manager.providers["firecrawl"].rate_limiter is not None
        print("✓ Direct CrawlManager initialization passed")

    await embedding_manager.cleanup()
    await crawl_manager.cleanup()
    # rate_limiter doesn't have cleanup method

    print("\n✅ All integration tests passed!")


if __name__ == "__main__":
    asyncio.run(test_integration())
