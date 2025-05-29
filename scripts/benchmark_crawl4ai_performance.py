#!/usr/bin/env python3
"""Comprehensive performance benchmark comparing Crawl4AI vs Firecrawl.

This script validates the 4-6x performance improvement claims by:
1. Testing individual URL scraping performance
2. Testing bulk crawling performance
3. Measuring resource usage (memory, CPU)
4. Tracking success rates and error patterns
5. Generating detailed performance comparison report
"""

import asyncio
import contextlib
import logging
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from src.config.loader import ConfigLoader
from src.infrastructure.client_manager import ClientManager
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.firecrawl_provider import FirecrawlProvider
from src.services.utilities.rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


def validate_api_keys() -> None:
    """Validate required API keys for benchmarking."""
    required_keys = {
        "FIRECRAWL_API_KEY": "Required for Firecrawl provider benchmarking",
        "OPENAI_API_KEY": "Required for embedding operations",
    }

    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"  - {key}: {description}")

    if missing_keys:
        console.print(
            "\n[red]Missing required API keys:[/red]\n"
            + "\n".join(missing_keys)
            + "\n\nSet these environment variables before running benchmarks."
        )
        raise ValueError("Missing required API keys")

    console.print("[green]✓ All required API keys validated[/green]")


class BenchmarkMetrics(BaseModel):
    """Metrics collected during benchmark runs."""

    provider: str
    total_urls: int
    successful: int
    failed: int
    total_time: float
    avg_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    min_time: float
    max_time: float
    throughput: float  # URLs per second
    memory_used_mb: float
    cpu_percent: float
    error_types: dict[str, int]
    content_size_avg: int
    cost_estimate: float


class CrawlerBenchmark:
    """Comprehensive benchmark suite for crawler performance comparison."""

    def __init__(self):
        """Initialize benchmark with both providers."""
        self.config = ConfigLoader().load_config()
        self.client_manager = ClientManager(self.config)

        # Initialize providers
        self.crawl4ai_provider = Crawl4AIProvider(
            config={
                "max_concurrent": 50,
                "rate_limit": 60,
                "browser": "chromium",
                "headless": True,
            }
        )

        # Only initialize Firecrawl if API key is available
        self.firecrawl_provider = None
        if os.getenv("FIRECRAWL_API_KEY"):
            self.firecrawl_provider = FirecrawlProvider(
                api_key=os.getenv("FIRECRAWL_API_KEY"),
                rate_limiter=RateLimiter(max_calls=100, time_window=60),
            )

        self.test_urls = self._get_test_urls()

    def _get_test_urls(self) -> list[str]:
        """Get diverse set of test URLs for benchmarking."""
        return [
            # Documentation sites (various frameworks)
            "https://docs.python.org/3/library/asyncio.html",
            "https://fastapi.tiangolo.com/tutorial/",
            "https://react.dev/learn",
            "https://vuejs.org/guide/introduction.html",
            "https://docs.djangoproject.com/en/stable/",
            # API documentation
            "https://platform.openai.com/docs/api-reference",
            "https://stripe.com/docs/api",
            "https://docs.github.com/en/rest",
            # Technical blogs
            "https://engineering.fb.com/",
            "https://netflixtechblog.com/",
            # Markdown-heavy sites
            "https://github.com/microsoft/vscode/blob/main/README.md",
            # JavaScript-heavy SPAs
            "https://angular.io/docs",
            "https://nextjs.org/docs",
            # Complex layouts
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://kubernetes.io/docs/home/",
        ]

    async def measure_single_url(
        self, provider: Any, url: str, provider_name: str
    ) -> tuple[float, dict[str, Any]]:
        """Measure performance for a single URL scrape.

        Returns:
            Tuple of (elapsed_time, result_dict)
        """
        # Record initial resources with proper async timing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Start CPU measurement in background
        cpu_task = asyncio.create_task(self._measure_cpu_async(process))

        start_time = time.time()
        try:
            result = await provider.scrape_url(url)
            elapsed = time.time() - start_time

            # Measure final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024
            avg_cpu = await cpu_task  # Get average CPU usage

            result["memory_delta"] = max(
                0, final_memory - initial_memory
            )  # Ensure non-negative
            result["cpu_usage"] = avg_cpu

            return elapsed, result
        except Exception as e:
            elapsed = time.time() - start_time
            # Cancel CPU monitoring if still running
            if not cpu_task.done():
                cpu_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cpu_task

            error_type = type(e).__name__
            logger.error(f"{provider_name} failed on {url}: {error_type}: {e}")
            return elapsed, {
                "success": False,
                "error": str(e),
                "error_type": error_type,
                "url": url,
                "memory_delta": 0,
                "cpu_usage": 0,
            }

    async def _measure_cpu_async(
        self, process: psutil.Process, duration: float = 1.0
    ) -> float:
        """Measure average CPU usage over a duration.

        Args:
            process: Process to monitor
            duration: Measurement duration in seconds

        Returns:
            Average CPU percentage
        """
        measurements = []
        interval = 0.1  # Sample every 100ms
        samples = int(duration / interval)

        try:
            for _ in range(samples):
                measurements.append(process.cpu_percent(interval=None))
                await asyncio.sleep(interval)

            return sum(measurements) / len(measurements) if measurements else 0.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Handle process termination or access issues
            return 0.0

    async def benchmark_single_urls(self) -> dict[str, BenchmarkMetrics]:
        """Benchmark single URL scraping for both providers."""
        console.print("\n[bold cyan]🔍 Single URL Scraping Benchmark[/bold cyan]")
        results = {}

        for provider_name, provider in [
            ("crawl4ai", self.crawl4ai_provider),
            ("firecrawl", self.firecrawl_provider),
        ]:
            if provider is None:
                console.print(
                    f"[yellow]⚠️  Skipping {provider_name} (not configured)[/yellow]"
                )
                continue

            console.print(f"\n[bold]{provider_name.upper()}[/bold]")

            # Initialize provider
            if hasattr(provider, "initialize"):
                await provider.initialize()

            metrics = {
                "times": [],
                "successful": 0,
                "failed": 0,
                "error_types": {},
                "content_sizes": [],
                "memory_usage": [],
                "cpu_usage": [],
            }

            with Progress() as progress:
                task = progress.add_task(
                    f"Testing {provider_name}...", total=len(self.test_urls)
                )

                for url in self.test_urls:
                    elapsed, result = await self.measure_single_url(
                        provider, url, provider_name
                    )
                    metrics["times"].append(elapsed)

                    if result.get("success", False):
                        metrics["successful"] += 1
                        content_size = len(result.get("content", ""))
                        metrics["content_sizes"].append(content_size)
                    else:
                        metrics["failed"] += 1
                        error_type = result.get("error_type", "Unknown")
                        metrics["error_types"][error_type] = (
                            metrics["error_types"].get(error_type, 0) + 1
                        )

                    metrics["memory_usage"].append(result.get("memory_delta", 0))
                    metrics["cpu_usage"].append(result.get("cpu_usage", 0))

                    progress.advance(task)

            # Cleanup provider
            if hasattr(provider, "cleanup"):
                await provider.cleanup()

            # Calculate final metrics
            times = metrics["times"]
            results[provider_name] = BenchmarkMetrics(
                provider=provider_name,
                total_urls=len(self.test_urls),
                successful=metrics["successful"],
                failed=metrics["failed"],
                total_time=sum(times),
                avg_time=statistics.mean(times) if times else 0,
                p50_time=statistics.median(times) if times else 0,
                p95_time=np.percentile(times, 95) if times else 0,
                p99_time=np.percentile(times, 99) if times else 0,
                min_time=min(times) if times else 0,
                max_time=max(times) if times else 0,
                throughput=len(times) / sum(times) if sum(times) > 0 else 0,
                memory_used_mb=statistics.mean(metrics["memory_usage"]),
                cpu_percent=statistics.mean(metrics["cpu_usage"]),
                error_types=metrics["error_types"],
                content_size_avg=(
                    statistics.mean(metrics["content_sizes"])
                    if metrics["content_sizes"]
                    else 0
                ),
                cost_estimate=self._estimate_cost(provider_name, len(self.test_urls)),
            )

        return results

    async def benchmark_bulk_crawling(self) -> dict[str, BenchmarkMetrics]:
        """Benchmark bulk crawling performance."""
        console.print("\n[bold cyan]🚀 Bulk Crawling Benchmark[/bold cyan]")

        # Use subset of URLs for bulk test
        bulk_urls = self.test_urls[:5]  # First 5 URLs
        results = {}

        for provider_name, provider in [
            ("crawl4ai", self.crawl4ai_provider),
            ("firecrawl", self.firecrawl_provider),
        ]:
            if provider is None:
                console.print(
                    f"[yellow]⚠️  Skipping {provider_name} (not configured)[/yellow]"
                )
                continue

            console.print(f"\n[bold]{provider_name.upper()} - Bulk Crawl[/bold]")

            # Initialize provider
            if hasattr(provider, "initialize"):
                await provider.initialize()

            # Measure bulk crawl
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            try:
                if hasattr(provider, "crawl_bulk"):
                    # Crawl4AI has bulk method
                    bulk_results = await provider.crawl_bulk(bulk_urls)
                else:
                    # Firecrawl - simulate bulk with concurrent requests
                    tasks = [provider.scrape_url(url) for url in bulk_urls]
                    bulk_results = await asyncio.gather(*tasks, return_exceptions=True)

                elapsed = time.time() - start_time
                final_memory = process.memory_info().rss / 1024 / 1024

                # Process results
                successful = sum(
                    1
                    for r in bulk_results
                    if isinstance(r, dict) and r.get("success", False)
                )
                failed = len(bulk_results) - successful

                results[f"{provider_name}_bulk"] = BenchmarkMetrics(
                    provider=f"{provider_name}_bulk",
                    total_urls=len(bulk_urls),
                    successful=successful,
                    failed=failed,
                    total_time=elapsed,
                    avg_time=elapsed / len(bulk_urls),
                    p50_time=elapsed / len(bulk_urls),
                    p95_time=elapsed / len(bulk_urls),
                    p99_time=elapsed / len(bulk_urls),
                    min_time=elapsed / len(bulk_urls),
                    max_time=elapsed / len(bulk_urls),
                    throughput=len(bulk_urls) / elapsed,
                    memory_used_mb=final_memory - initial_memory,
                    cpu_percent=process.cpu_percent(interval=0.1),
                    error_types={},
                    content_size_avg=0,
                    cost_estimate=self._estimate_cost(provider_name, len(bulk_urls)),
                )

            except Exception as e:
                logger.error(f"Bulk crawl failed for {provider_name}: {e}")
                results[f"{provider_name}_bulk"] = BenchmarkMetrics(
                    provider=f"{provider_name}_bulk",
                    total_urls=len(bulk_urls),
                    successful=0,
                    failed=len(bulk_urls),
                    total_time=0,
                    avg_time=0,
                    p50_time=0,
                    p95_time=0,
                    p99_time=0,
                    min_time=0,
                    max_time=0,
                    throughput=0,
                    memory_used_mb=0,
                    cpu_percent=0,
                    error_types={"BulkError": 1},
                    content_size_avg=0,
                    cost_estimate=0,
                )

            # Cleanup
            if hasattr(provider, "cleanup"):
                await provider.cleanup()

        return results

    def _estimate_cost(self, provider: str, num_urls: int) -> float:
        """Estimate cost based on provider pricing."""
        if provider == "crawl4ai":
            return 0.0  # Free
        elif provider == "firecrawl":
            # Firecrawl pricing: ~$15 per 1000 pages
            return (num_urls / 1000) * 15.0
        return 0.0

    def generate_report(
        self,
        single_results: dict[str, BenchmarkMetrics],
        bulk_results: dict[str, BenchmarkMetrics],
    ) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = [
            "# Crawl4AI vs Firecrawl Performance Benchmark Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"Platform: {platform.system()} {platform.release()}",
            f"Python: {platform.python_version()}",
            "\n## Executive Summary\n",
        ]

        # Calculate improvement ratios
        if "crawl4ai" in single_results and "firecrawl" in single_results:
            c4ai = single_results["crawl4ai"]
            fc = single_results["firecrawl"]

            speed_improvement = fc.avg_time / c4ai.avg_time if c4ai.avg_time > 0 else 0
            throughput_improvement = (
                c4ai.throughput / fc.throughput if fc.throughput > 0 else 0
            )
            cost_savings = fc.cost_estimate - c4ai.cost_estimate

            report_lines.extend(
                [
                    f"- **Speed Improvement**: {speed_improvement:.2f}x faster",
                    f"- **Throughput Improvement**: {throughput_improvement:.2f}x higher",
                    f"- **Cost Savings**: ${cost_savings:.2f} per {len(self.test_urls)} URLs",
                    f"- **Success Rate**: Crawl4AI: {c4ai.successful / c4ai.total_urls * 100:.1f}%, "
                    f"Firecrawl: {fc.successful / fc.total_urls * 100:.1f}%",
                    "\n### Key Findings\n",
                ]
            )

            if speed_improvement >= 4:
                report_lines.append(
                    "✅ **VALIDATED**: Crawl4AI achieves 4x+ speed improvement"
                )
            else:
                report_lines.append(
                    f"⚠️  **PARTIAL**: Crawl4AI achieves {speed_improvement:.2f}x speed improvement"
                )

        # Single URL Performance Table
        report_lines.append("\n## Single URL Scraping Performance\n")
        table = Table(title="Single URL Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Crawl4AI", style="green")
        table.add_column("Firecrawl", style="yellow")
        table.add_column("Improvement", style="magenta")

        if "crawl4ai" in single_results and "firecrawl" in single_results:
            c4ai = single_results["crawl4ai"]
            fc = single_results["firecrawl"]

            metrics = [
                ("URLs Tested", c4ai.total_urls, fc.total_urls, "-"),
                ("Successful", c4ai.successful, fc.successful, "-"),
                ("Failed", c4ai.failed, fc.failed, "-"),
                (
                    "Success Rate",
                    f"{c4ai.successful / c4ai.total_urls * 100:.1f}%",
                    f"{fc.successful / fc.total_urls * 100:.1f}%",
                    "-",
                ),
                (
                    "Avg Time (s)",
                    f"{c4ai.avg_time:.3f}",
                    f"{fc.avg_time:.3f}",
                    f"{fc.avg_time / c4ai.avg_time:.2f}x",
                ),
                (
                    "P50 Time (s)",
                    f"{c4ai.p50_time:.3f}",
                    f"{fc.p50_time:.3f}",
                    f"{fc.p50_time / c4ai.p50_time:.2f}x",
                ),
                (
                    "P95 Time (s)",
                    f"{c4ai.p95_time:.3f}",
                    f"{fc.p95_time:.3f}",
                    f"{fc.p95_time / c4ai.p95_time:.2f}x",
                ),
                (
                    "P99 Time (s)",
                    f"{c4ai.p99_time:.3f}",
                    f"{fc.p99_time:.3f}",
                    f"{fc.p99_time / c4ai.p99_time:.2f}x",
                ),
                ("Min Time (s)", f"{c4ai.min_time:.3f}", f"{fc.min_time:.3f}", "-"),
                ("Max Time (s)", f"{c4ai.max_time:.3f}", f"{fc.max_time:.3f}", "-"),
                (
                    "Throughput (URL/s)",
                    f"{c4ai.throughput:.2f}",
                    f"{fc.throughput:.2f}",
                    f"{c4ai.throughput / fc.throughput:.2f}x",
                ),
                (
                    "Avg Memory (MB)",
                    f"{c4ai.memory_used_mb:.1f}",
                    f"{fc.memory_used_mb:.1f}",
                    "-",
                ),
                ("Avg CPU %", f"{c4ai.cpu_percent:.1f}", f"{fc.cpu_percent:.1f}", "-"),
                (
                    "Est. Cost",
                    f"${c4ai.cost_estimate:.2f}",
                    f"${fc.cost_estimate:.2f}",
                    f"${fc.cost_estimate - c4ai.cost_estimate:.2f}",
                ),
            ]

            for row in metrics:
                table.add_row(*[str(x) for x in row])

        # Print table to console
        console.print(table)

        # Convert table to markdown for report
        report_lines.append("| Metric | Crawl4AI | Firecrawl | Improvement |")
        report_lines.append("|--------|----------|-----------|-------------|")
        if "crawl4ai" in single_results and "firecrawl" in single_results:
            for row in metrics:
                report_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

        # Bulk Performance
        if bulk_results:
            report_lines.append("\n## Bulk Crawling Performance\n")
            report_lines.append(
                "| Provider | URLs | Time (s) | Throughput (URL/s) | Success Rate |"
            )
            report_lines.append(
                "|----------|------|----------|-------------------|--------------|"
            )

            for name, metrics in bulk_results.items():
                success_rate = (
                    f"{metrics.successful / metrics.total_urls * 100:.1f}%"
                    if metrics.total_urls > 0
                    else "0%"
                )
                report_lines.append(
                    f"| {name} | {metrics.total_urls} | {metrics.total_time:.2f} | "
                    f"{metrics.throughput:.2f} | {success_rate} |"
                )

        # Error Analysis
        report_lines.append("\n## Error Analysis\n")
        for provider_name, metrics in single_results.items():
            if metrics.error_types:
                report_lines.append(f"\n### {provider_name.capitalize()} Errors:")
                for error_type, count in metrics.error_types.items():
                    report_lines.append(f"- {error_type}: {count}")

        # Recommendations
        report_lines.extend(
            [
                "\n## Recommendations\n",
                "Based on the benchmark results:",
                "",
                "1. **Primary Scraper**: Use Crawl4AI for all bulk documentation scraping",
                "2. **Fallback Strategy**: Keep Firecrawl as fallback for edge cases",
                "3. **Configuration**: Optimize Crawl4AI concurrency based on target sites",
                "4. **Monitoring**: Track success rates and performance in production",
                "",
                "## Next Steps\n",
                "- [ ] Update documentation with Crawl4AI configuration examples",
                "- [ ] Create troubleshooting guide for common Crawl4AI issues",
                "- [ ] Implement performance monitoring for production workloads",
                "- [ ] Consider removing Firecrawl dependency after validation period",
            ]
        )

        return "\n".join(report_lines)

    async def run_full_benchmark(self):
        """Run complete benchmark suite."""
        console.print(
            "[bold green]🚀 Starting Crawl4AI vs Firecrawl Performance Benchmark[/bold green]"
        )

        # Run benchmarks
        single_results = await self.benchmark_single_urls()
        bulk_results = await self.benchmark_bulk_crawling()

        # Generate report
        report = self.generate_report(single_results, bulk_results)

        # Save report
        report_dir = Path("benchmark_results")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"crawl4ai_benchmark_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write(report)

        console.print(
            f"\n[bold green]✅ Benchmark complete! Report saved to: {report_file}[/bold green]"
        )
        console.print("\n[cyan]Report Preview:[/cyan]")
        console.print(report)

        return single_results, bulk_results, report_file


async def main():
    """Run the benchmark."""
    # Validate API keys before starting
    validate_api_keys()

    benchmark = CrawlerBenchmark()
    await benchmark.run_full_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
