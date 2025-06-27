#!/usr/bin/env python3
"""Benchmark script to demonstrate Lightweight HTTP Tier performance improvements."""

import asyncio  # noqa: PLC0415

# Configure logging
import logging  # noqa: PLC0415
import time  # noqa: PLC0415
from statistics import mean
from statistics import stdev

from ..config import Config
from src.services.crawling.manager import CrawlManager
from src.utils.imports import log_import_issues

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable import warnings for cleaner output
log_import_issues(show_warnings=False)


class TierBenchmark:
    """Benchmark harness for testing tier performance."""

    def __init__(self):
        """Initialize benchmark with test URLs."""
        self.test_urls = {
            "static_markdown": [
                "https://raw.githubusercontent.com/python/cpython/main/README.rst",
                "https://raw.githubusercontent.com/nodejs/node/main/README.md",
                "https://raw.githubusercontent.com/golang/go/master/README.md",
            ],
            "documentation": [
                "https://docs.python.org/3/tutorial/index.html",
                "https://docs.python.org/3/library/asyncio.html",
                "https://golang.org/doc/tutorial/getting-started",
            ],
            "complex_spa": [
                "https://react.dev/",
                "https://vuejs.org/",
                "https://angular.io/",
            ],
        }
        self.results: dict[str, list[float]] = {}

    async def setup(self):
        """Set up crawl manager with all tiers enabled."""
        self.config = Config()
        self.manager = CrawlManager(self.config)
        await self.manager.initialize()
        logger.info(
            f"Initialized CrawlManager with providers: {list(self.manager.providers.keys())}"
        )

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "manager"):
            await self.manager.cleanup()

    async def benchmark_url(
        self, url: str, preferred_provider: str | None = None
    ) -> float:
        """Benchmark a single URL scraping.

        Args:
            url: URL to scrape
            preferred_provider: Optional provider to force

        Returns:
            Time taken in milliseconds
        """
        start_time = time.time()
        result = await self.manager.scrape_url(
            url, preferred_provider=preferred_provider
        )
        elapsed_ms = (time.time() - start_time) * 1000

        if result.get("success"):
            provider = result.get("provider", "unknown")
            tier = result.get("tier", -1)
            logger.info(
                f"✅ Scraped {url} with {provider} (Tier {tier}) in {elapsed_ms:.0f}ms"
            )
        else:
            logger.warning(f"❌ Failed to scrape {url}: {result.get('error')}")

        return elapsed_ms

    async def run_category_benchmark(self, category: str, urls: list[str]):
        """Run benchmarks for a category of URLs."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking {category.upper()} URLs")
        logger.info(f"{'=' * 60}")

        # Test with automatic tier selection
        auto_times = []
        for url in urls:
            elapsed = await self.benchmark_url(url)
            auto_times.append(elapsed)
            await asyncio.sleep(0.5)  # Avoid rate limiting

        # Test forcing Crawl4AI (Tier 1) for comparison
        if "lightweight" in self.manager.providers:
            logger.info("\nForcing Crawl4AI (Tier 1) for comparison...")
            crawl4ai_times = []
            for url in urls:
                elapsed = await self.benchmark_url(url, preferred_provider="crawl4ai")
                crawl4ai_times.append(elapsed)
                await asyncio.sleep(0.5)

            # Calculate improvement
            avg_auto = mean(auto_times)
            avg_crawl4ai = mean(crawl4ai_times)
            improvement = avg_crawl4ai / avg_auto if avg_auto > 0 else 0

            logger.info(f"\n📊 {category.upper()} Results:")
            logger.info(
                f"  - Auto-tier avg: {avg_auto:.0f}ms (stdev={stdev(auto_times):.0f}ms)"
            )
            logger.info(
                f"  - Crawl4AI avg: {avg_crawl4ai:.0f}ms (stdev={stdev(crawl4ai_times):.0f}ms)"
            )
            logger.info(
                f"  - Performance improvement: {improvement:.1f}x faster with tier selection"
            )

            self.results[category] = {
                "auto_times": auto_times,
                "crawl4ai_times": crawl4ai_times,
                "improvement": improvement,
            }

    async def run_benchmarks(self):
        """Run all benchmarks."""
        await self.setup()

        try:
            # Run benchmarks for each category
            for category, urls in self.test_urls.items():
                await self.run_category_benchmark(
                    category, urls[:2]
                )  # Limit to 2 URLs per category

            # Print summary
            self.print_summary()

        finally:
            await self.cleanup()

    def print_summary(self):
        """Print benchmark summary."""
        logger.info(f"\n{'=' * 60}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'=' * 60}")

        total_auto_time = 0
        total_crawl4ai_time = 0

        for category, data in self.results.items():
            auto_avg = mean(data["auto_times"])
            crawl4ai_avg = mean(data["crawl4ai_times"])
            total_auto_time += sum(data["auto_times"])
            total_crawl4ai_time += sum(data["crawl4ai_times"])

            logger.info(f"\n{category.replace('_', ' ').title()}:")
            logger.info(f"  Auto-tier: {auto_avg:.0f}ms avg")
            logger.info(f"  Crawl4AI:  {crawl4ai_avg:.0f}ms avg")
            logger.info(f"  Speedup:   {data['improvement']:.1f}x")

        # Overall improvement
        overall_improvement = (
            total_crawl4ai_time / total_auto_time if total_auto_time > 0 else 0
        )
        logger.info(
            f"\n🚀 Overall Performance Improvement: {overall_improvement:.1f}x faster"
        )
        logger.info(
            f"   Total time saved: {total_crawl4ai_time - total_auto_time:.0f}ms"
        )

        # Tier metrics
        logger.info("\n📈 Tier Metrics:")
        tier_info = self.manager.get_tier_metrics()
        for tier_name, metrics in tier_info.items():
            if metrics["attempts"] > 0:
                success_rate = (metrics["successes"] / metrics["attempts"]) * 100
                avg_time = (metrics["total_time"] / metrics["attempts"]) * 1000
                logger.info(
                    f"  {tier_name}: {metrics['attempts']} attempts, "
                    f"{success_rate:.0f}% success, {avg_time:.0f}ms avg"
                )


async def main():
    """Run the benchmark."""
    logger.info("🏁 Starting Lightweight HTTP Tier Benchmark...")
    logger.info("This benchmark compares automatic tier selection vs forced Crawl4AI")

    benchmark = TierBenchmark()
    await benchmark.run_benchmarks()

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
