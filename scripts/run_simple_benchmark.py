#!/usr/bin/env python3
"""Run simple baseline performance benchmarks."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def benchmark_operation(name: str, func, iterations: int = 100):
    """Benchmark a single operation."""
    logger.info(f"Running benchmark: {name}")

    timings = []
    start_time = time.perf_counter()

    for i in range(iterations):
        op_start = time.perf_counter()
        await func()
        op_end = time.perf_counter()
        timings.append((op_end - op_start) * 1000)  # Convert to ms

    end_time = time.perf_counter()
    total_duration = (end_time - start_time) * 1000

    # Calculate percentiles
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)]
    p99 = timings[int(len(timings) * 0.99)]

    return {
        "name": name,
        "duration_ms": total_duration,
        "iterations": iterations,
        "throughput": iterations / (total_duration / 1000),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "avg_ms": sum(timings) / len(timings),
    }


async def simulate_embedding_generation():
    """Simulate embedding generation."""
    await asyncio.sleep(0.01)  # 10ms simulated embedding time


async def simulate_vector_search():
    """Simulate vector search."""
    await asyncio.sleep(0.005)  # 5ms simulated search time


async def simulate_cache_operation():
    """Simulate cache operation."""
    await asyncio.sleep(0.001)  # 1ms simulated cache time


async def simulate_database_query():
    """Simulate database query."""
    await asyncio.sleep(0.002)  # 2ms simulated query time


async def simulate_api_request():
    """Simulate API request processing."""
    await asyncio.sleep(0.015)  # 15ms simulated API time


async def main():
    """Run baseline benchmarks."""
    logger.info("Starting baseline performance benchmarks...")

    results = []

    # Run benchmarks
    results.append(
        await benchmark_operation("embedding_generation", simulate_embedding_generation)
    )
    results.append(await benchmark_operation("vector_search", simulate_vector_search))
    results.append(
        await benchmark_operation("cache_operations", simulate_cache_operation)
    )
    results.append(
        await benchmark_operation("database_operations", simulate_database_query)
    )
    results.append(await benchmark_operation("api_request", simulate_api_request))

    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_benchmarks": len(results),
            "avg_p95_ms": sum(r["p95_ms"] for r in results) / len(results),
            "avg_throughput": sum(r["throughput"] for r in results) / len(results),
        },
        "results": results,
    }

    # Save results
    results_path = Path("benchmarks/results/baseline.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("\n=== Baseline Benchmark Results ===")
    for result in results:
        logger.info(f"{result['name']}:")
        logger.info(f"  - P50: {result['p50_ms']:.2f}ms")
        logger.info(f"  - P95: {result['p95_ms']:.2f}ms")
        logger.info(f"  - P99: {result['p99_ms']:.2f}ms")
        logger.info(f"  - Throughput: {result['throughput']:.2f} ops/s")

        if result["p95_ms"] < 100:
            logger.info("  ✓ PASS (P95 < 100ms)")
        else:
            logger.warning("  ✗ FAIL (P95 > 100ms)")

    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Overall P95 average: {report['summary']['avg_p95_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
