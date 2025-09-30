"""Lightweight performance benchmarking helpers for the config stack."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field
from typing import Any

from src.config import CacheConfig, get_config


def run_async(coro: Awaitable[Any]) -> Any:
    """Execute a coroutine in a dedicated event loop for synchronous callers."""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dataclass(slots=True)
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    duration_ms: float
    throughput_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSuite:
    """Collection of benchmark results with derived statistics."""

    name: str
    results: list[BenchmarkResult]

    def summary(self) -> dict[str, Any]:
        durations = [result.duration_ms for result in self.results]
        return {
            "benchmarks": len(self.results),
            "avg_duration_ms": statistics.mean(durations) if durations else 0.0,
            "p95_duration_ms": statistics.quantiles(durations, n=20)[18]
            if len(durations) >= 20
            else max(durations)
            if durations
            else 0.0,
        }


async def _process_parallel(items: Iterable[int], *, concurrency: int) -> list[int]:
    """Process items concurrently by squaring each integer."""

    semaphore = asyncio.Semaphore(concurrency)
    results: list[int] = []

    async def worker(value: int) -> None:
        async with semaphore:
            await asyncio.sleep(0)  # yield control
            results.append(value * value)

    await asyncio.gather(*(worker(item) for item in items))
    return results


class PerformanceBenchmark:
    """Simple benchmarks centred on the configuration runtime behaviour."""

    def __init__(self) -> None:
        self.config = get_config()

    async def benchmark_parallel_processing(self) -> BenchmarkResult:
        """Measure how quickly a small parallel workload completes."""

        payload = list(range(500))

        start = time.perf_counter()
        await _process_parallel(payload, concurrency=16)
        duration_ms = (time.perf_counter() - start) * 1000
        throughput = len(payload) / max(duration_ms / 1000, 1e-6)

        return BenchmarkResult(
            name="parallel_processing",
            duration_ms=duration_ms,
            throughput_per_second=throughput,
            metadata={"items": len(payload), "concurrency": 16},
        )

    async def benchmark_cache_performance(self) -> BenchmarkResult:
        """Exercise cache-related configuration paths."""

        cache_config = CacheConfig()
        cache_store: dict[str, str] = {}
        payload = {f"key-{i}": f"value-{i}" for i in range(1000)}

        start = time.perf_counter()
        for key, value in payload.items():
            if cache_config.enable_local_cache:
                cache_store[key] = value
        for key in payload:
            _ = cache_store.get(key)
        duration_ms = (time.perf_counter() - start) * 1000
        throughput = len(payload) / max(duration_ms / 1000, 1e-6)

        return BenchmarkResult(
            name="cache_operations",
            duration_ms=duration_ms,
            throughput_per_second=throughput,
            metadata={
                "enable_local_cache": cache_config.enable_local_cache,
                "items_cached": len(cache_store),
            },
        )

    async def run(self) -> BenchmarkSuite:
        """Execute the benchmark suite and collect results."""

        tasks = [
            self.benchmark_parallel_processing(),
            self.benchmark_cache_performance(),
        ]
        results = await asyncio.gather(*tasks)
        return BenchmarkSuite(name="config_performance", results=list(results))


def main() -> None:
    """CLI entry point for running the benchmark suite."""

    parser = argparse.ArgumentParser(description="Config performance benchmarks")
    parser.parse_args()
    suite = run_async(PerformanceBenchmark().run())
    for result in suite.results:
        print(
            f"{result.name}: {result.duration_ms:.2f} ms, "
            f"throughput={result.throughput_per_second:.2f}/s"
        )
    print("Summary:", suite.summary())


if __name__ == "__main__":
    main()
