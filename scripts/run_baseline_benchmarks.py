#!/usr/bin/env python3
"""Run baseline performance benchmarks for POA."""

import asyncio
import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings
from src.infrastructure.client_manager import ClientManager
from src.services.cache.intelligent import IntelligentCache
from src.services.embeddings.manager import EmbeddingManager
from src.services.monitoring.performance_monitor import RealTimePerformanceMonitor
from src.services.performance.benchmarks import PerformanceBenchmark
from src.services.query_processing.pipeline import QueryPipeline


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run baseline performance benchmarks."""
    logger.info("Starting baseline performance benchmarks...")

    # Initialize settings and components
    settings = Settings()
    client_manager = ClientManager.from_unified_config()
    monitor = RealTimePerformanceMonitor()

    # Initialize benchmark framework
    benchmark = PerformanceBenchmark(settings, monitor)

    # Create benchmark components
    components = {"settings": settings, "monitor": monitor}

    # Initialize clients
    await client_manager.initialize()

    try:
        # Add embedding manager if available
        if client_manager.openai_client:
            components["embedding_manager"] = EmbeddingManager(
                client=client_manager.openai_client, settings=settings
            )

        # Add Qdrant client if available
        if client_manager.qdrant_client:
            components["qdrant_client"] = client_manager.qdrant_client
            components["collection_name"] = "documents"

        # Add cache if Redis is available
        if client_manager.redis_client:
            components["cache"] = IntelligentCache(
                redis_client=client_manager.redis_client, settings=settings
            )

        # Add query pipeline if components are available
        if all(
            k in components for k in ["embedding_manager", "qdrant_client", "cache"]
        ):
            components["query_pipeline"] = QueryPipeline(
                embedding_manager=components["embedding_manager"],
                qdrant_client=components["qdrant_client"],
                cache=components["cache"],
                settings=settings,
            )

        # Run baseline suite
        suite = await benchmark.run_baseline_suite(components)

        # Save results
        results_path = Path("benchmarks/results/baseline.json")
        benchmark.save_results(suite, results_path)

        # Print summary
        report = suite.to_report()
        logger.info("\n=== Baseline Benchmark Results ===")
        logger.info(f"Total benchmarks: {report['summary']['total_benchmarks']}")
        logger.info(f"Average duration: {report['summary']['avg_duration_ms']:.2f}ms")
        logger.info(f"Average memory: {report['summary']['avg_memory_mb']:.2f}MB")
        logger.info(
            f"Average throughput: {report['summary']['avg_throughput']:.2f} ops/s"
        )

        # Check performance targets
        logger.info("\n=== Performance Target Validation ===")
        for result in suite.results:
            if result.p95_ms < 100:
                logger.info(f"✓ {result.name}: P95={result.p95_ms:.2f}ms (PASS)")
            else:
                logger.warning(
                    f"✗ {result.name}: P95={result.p95_ms:.2f}ms (FAIL - exceeds 100ms target)"
                )

        logger.info(f"\nResults saved to: {results_path}")

    finally:
        # Cleanup
        await client_manager.close()
        monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
