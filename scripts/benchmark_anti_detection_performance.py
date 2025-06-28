#!/usr/bin/env python3
"""Performance benchmarks for Enhanced Anti-Detection System.

This script benchmarks different anti-detection levels to measure their
performance impact on scraping operations. Useful for optimizing the
balance between stealth and performance.
"""

import asyncio  # noqa: PLC0415
import json  # noqa: PLC0415
import statistics

# Add src to path for imports
import sys
import time  # noqa: PLC0415
from pathlib import Path
from typing import Any

import click
import psutil
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.models import PlaywrightConfig
from services.browser.enhanced_anti_detection import EnhancedAntiDetection


class PerformanceBenchmark:
    """Performance benchmark for anti-detection configurations."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize benchmark with output directory."""
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function call."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "result": result,
            "execution_time_ms": (end_time - start_time) * 1000,
            "memory_usage_mb": mem_after - mem_before,
            "peak_memory_mb": mem_after,
        }

    async def benchmark_config_generation(
        self, iterations: int = 100
    ) -> dict[str, Any]:
        """Benchmark stealth configuration generation."""
        click.echo("🔧 Benchmarking stealth configuration generation...")

        anti_detection = EnhancedAntiDetection()
        site_profiles = ["default", "github.com", "linkedin.com", "cloudflare.com"]

        results = {}

        for profile in site_profiles:
            profile_results = {
                "execution_times": [],
                "memory_usage": [],
                "config_complexity": {},
            }

            for _ in range(iterations):
                metrics = self.measure_memory_usage(
                    anti_detection.get_stealth_config, profile
                )

                profile_results["execution_times"].append(metrics["execution_time_ms"])
                profile_results["memory_usage"].append(metrics["memory_usage_mb"])

                # Analyze config complexity
                config = metrics["result"]
                profile_results["config_complexity"] = {
                    "extra_args_count": len(config.extra_args),
                    "headers_count": len(config.headers),
                    "viewport_randomization": True,
                    "timing_patterns": True,
                }

            # Calculate statistics
            results[profile] = {
                "avg_execution_time_ms": statistics.mean(
                    profile_results["execution_times"]
                ),
                "p95_execution_time_ms": statistics.quantiles(
                    profile_results["execution_times"], n=20
                )[18],
                "avg_memory_usage_mb": statistics.mean(profile_results["memory_usage"]),
                "config_complexity": profile_results["config_complexity"],
                "iterations": iterations,
            }

        return results

    async def benchmark_playwright_integration(
        self, test_urls: list[str] | None = None
    ) -> dict[str, Any]:
        """Benchmark Playwright integration with different anti-detection levels."""
        click.echo("🎭 Benchmarking Playwright integration...")

        if not test_urls:
            test_urls = [
                "https://httpbin.org/html",  # Simple test endpoint
                "https://httpbin.org/user-agent",  # User agent test
                "https://httpbin.org/headers",  # Headers test
            ]

        anti_detection = EnhancedAntiDetection()
        base_config = PlaywrightConfig()

        results = {}
        site_profiles = ["default", "github.com", "linkedin.com", "cloudflare.com"]

        for profile in site_profiles:
            profile_results = {
                "config_generation_times": [],
                "memory_usage": [],
                "enhanced_configs": [],
            }

            for _url in test_urls:
                metrics = self.measure_memory_usage(
                    lambda p=profile: asyncio.run(
                        anti_detection.apply_stealth_to_playwright_config(
                            base_config, p
                        )
                    )
                )

                profile_results["config_generation_times"].append(
                    metrics["execution_time_ms"]
                )
                profile_results["memory_usage"].append(metrics["memory_usage_mb"])
                profile_results["enhanced_configs"].append(metrics["result"])

            # Analyze enhanced config differences
            config_analysis = self._analyze_config_differences(
                base_config, profile_results["enhanced_configs"][0]
            )

            results[profile] = {
                "avg_config_time_ms": statistics.mean(
                    profile_results["config_generation_times"]
                ),
                "max_config_time_ms": max(profile_results["config_generation_times"]),
                "avg_memory_usage_mb": statistics.mean(profile_results["memory_usage"]),
                "config_changes": config_analysis,
                "test_urls_count": len(test_urls),
            }

        return results

    def _analyze_config_differences(
        self, base_config: PlaywrightConfig, enhanced_config: PlaywrightConfig
    ) -> dict[str, Any]:
        """Analyze differences between base and enhanced configs."""
        return {
            "viewport_changed": base_config.viewport != enhanced_config.viewport,
            "user_agent_changed": base_config.user_agent != enhanced_config.user_agent,
            "timeout_changed": base_config.timeout != enhanced_config.timeout,
            "browser_changed": base_config.browser != enhanced_config.browser,
        }

    async def benchmark_user_agent_rotation(
        self, iterations: int = 1000
    ) -> dict[str, Any]:
        """Benchmark user agent rotation performance and diversity."""
        click.echo("🔄 Benchmarking user agent rotation...")

        anti_detection = EnhancedAntiDetection()

        execution_times = []
        user_agents = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            ua = anti_detection._rotate_user_agents()
            end_time = time.perf_counter()

            execution_times.append((end_time - start_time) * 1000)
            user_agents.append(ua)

        # Analyze diversity
        unique_uas = set(user_agents)
        browser_distribution = self._analyze_browser_distribution(user_agents)

        return {
            "avg_execution_time_ms": statistics.mean(execution_times),
            "total_unique_agents": len(unique_uas),
            "diversity_ratio": len(unique_uas) / len(user_agents),
            "browser_distribution": browser_distribution,
            "iterations": iterations,
        }

    def _analyze_browser_distribution(self, user_agents: list[str]) -> dict[str, float]:
        """Analyze browser type distribution in user agents."""
        browser_counts = {"Chrome": 0, "Firefox": 0, "Safari": 0, "Other": 0}

        for ua in user_agents:
            if "Chrome" in ua and "Safari" in ua:
                browser_counts["Chrome"] += 1
            elif "Firefox" in ua:
                browser_counts["Firefox"] += 1
            elif "Safari" in ua and "Chrome" not in ua:
                browser_counts["Safari"] += 1
            else:
                browser_counts["Other"] += 1

        total = len(user_agents)
        return {browser: count / total for browser, count in browser_counts.items()}

    async def benchmark_viewport_randomization(
        self, iterations: int = 1000
    ) -> dict[str, Any]:
        """Benchmark viewport randomization performance and distribution."""
        click.echo("📱 Benchmarking viewport randomization...")

        anti_detection = EnhancedAntiDetection()

        execution_times = []
        viewports = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            viewport = anti_detection._randomize_viewport()
            end_time = time.perf_counter()

            execution_times.append((end_time - start_time) * 1000)
            viewports.append((viewport.width, viewport.height))

        # Analyze distribution
        unique_viewports = set(viewports)
        width_stats = [v[0] for v in viewports]
        height_stats = [v[1] for v in viewports]

        return {
            "avg_execution_time_ms": statistics.mean(execution_times),
            "total_unique_viewports": len(unique_viewports),
            "diversity_ratio": len(unique_viewports) / len(viewports),
            "width_distribution": {
                "min": min(width_stats),
                "max": max(width_stats),
                "avg": statistics.mean(width_stats),
                "std": statistics.stdev(width_stats) if len(width_stats) > 1 else 0,
            },
            "height_distribution": {
                "min": min(height_stats),
                "max": max(height_stats),
                "avg": statistics.mean(height_stats),
                "std": statistics.stdev(height_stats) if len(height_stats) > 1 else 0,
            },
            "iterations": iterations,
        }

    async def benchmark_delay_patterns(self, iterations: int = 100) -> dict[str, Any]:
        """Benchmark human-like delay generation."""
        click.echo("⏱️ Benchmarking delay patterns...")

        anti_detection = EnhancedAntiDetection()
        site_profiles = ["default", "github.com", "linkedin.com", "cloudflare.com"]

        results = {}

        for profile in site_profiles:
            execution_times = []
            delays = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                delay = asyncio.run(anti_detection.get_human_like_delay(profile))
                end_time = time.perf_counter()

                execution_times.append((end_time - start_time) * 1000)
                delays.append(delay)

            results[profile] = {
                "avg_execution_time_ms": statistics.mean(execution_times),
                "delay_distribution": {
                    "min_delay_s": min(delays),
                    "max_delay_s": max(delays),
                    "avg_delay_s": statistics.mean(delays),
                    "std_delay_s": statistics.stdev(delays) if len(delays) > 1 else 0,
                },
                "iterations": iterations,
            }

        return results

    async def benchmark_success_monitoring(
        self, iterations: int = 100
    ) -> dict[str, Any]:
        """Benchmark success rate monitoring performance."""
        click.echo("📊 Benchmarking success monitoring...")

        anti_detection = EnhancedAntiDetection()

        # Simulate recording attempts
        record_times = []
        for i in range(iterations):
            success = i % 3 != 0  # ~67% success rate

            start_time = time.perf_counter()
            anti_detection.record_attempt(success, "test_strategy")
            end_time = time.perf_counter()

            record_times.append((end_time - start_time) * 1000)

        # Benchmark metrics retrieval
        metrics_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _metrics = anti_detection.get_success_metrics()
            end_time = time.perf_counter()

            metrics_times.append((end_time - start_time) * 1000)

        return {
            "record_attempt": {
                "avg_time_ms": statistics.mean(record_times),
                "max_time_ms": max(record_times),
                "iterations": iterations,
            },
            "get_metrics": {
                "avg_time_ms": statistics.mean(metrics_times),
                "max_time_ms": max(metrics_times),
                "iterations": 10,
            },
            "final_success_rate": anti_detection.get_success_metrics()[
                "overall_success_rate"
            ],
        }

    async def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """Run all benchmarks and compile results."""
        click.echo("🚀 Starting comprehensive anti-detection performance benchmark...")

        start_time = time.time()

        benchmark_results = {
            "benchmark_metadata": {
                "timestamp": time.time(),
                "python_version": sys.version,
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "platform": sys.platform,
                },
            },
            "config_generation": await self.benchmark_config_generation(),
            "playwright_integration": await self.benchmark_playwright_integration(),
            "user_agent_rotation": await self.benchmark_user_agent_rotation(),
            "viewport_randomization": await self.benchmark_viewport_randomization(),
            "delay_patterns": await self.benchmark_delay_patterns(),
            "success_monitoring": await self.benchmark_success_monitoring(),
        }

        total_time = time.time() - start_time
        benchmark_results["benchmark_metadata"]["total_execution_time_s"] = total_time

        # Save results
        results_file = (
            self.output_dir / f"anti_detection_benchmark_{int(time.time())}.json"
        )
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        click.echo(f"✅ Benchmark completed in {total_time:.2f}s")
        click.echo(f"📁 Results saved to: {results_file}")

        return benchmark_results

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate a human-readable report from benchmark results."""
        report_lines = [
            "# Enhanced Anti-Detection Performance Benchmark Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['benchmark_metadata']['timestamp']))}",
            f"**Total Execution Time:** {results['benchmark_metadata']['total_execution_time_s']:.2f}s",
            "",
            "## System Information",
            "",
            f"- **CPU Cores:** {results['benchmark_metadata']['system_info']['cpu_count']}",
            f"- **Total Memory:** {results['benchmark_metadata']['system_info']['memory_total_gb']:.1f} GB",
            f"- **Platform:** {results['benchmark_metadata']['system_info']['platform']}",
            "",
            "## Performance Summary",
            "",
        ]

        # Config generation summary
        config_gen = results["config_generation"]
        report_lines.extend(
            [
                "### Configuration Generation",
                "",
                "| Site Profile | Avg Time (ms) | P95 Time (ms) | Memory (MB) | Args Count |",
                "|--------------|---------------|---------------|-------------|------------|",
            ]
        )

        for profile, data in config_gen.items():
            args_count = data["config_complexity"]["extra_args_count"]
            report_lines.append(
                f"| {profile} | {data['avg_execution_time_ms']:.2f} | "
                f"{data['p95_execution_time_ms']:.2f} | {data['avg_memory_usage_mb']:.2f} | {args_count} |"
            )

        # User agent rotation summary
        ua_rotation = results["user_agent_rotation"]
        report_lines.extend(
            [
                "",
                "### User Agent Rotation",
                "",
                f"- **Average Execution Time:** {ua_rotation['avg_execution_time_ms']:.3f}ms",
                f"- **Unique Agents Generated:** {ua_rotation['total_unique_agents']} / {ua_rotation['iterations']}",
                f"- **Diversity Ratio:** {ua_rotation['diversity_ratio']:.3f}",
                "",
                "**Browser Distribution:**",
            ]
        )

        for browser, ratio in ua_rotation["browser_distribution"].items():
            report_lines.append(f"- {browser}: {ratio:.1%}")

        # Viewport randomization summary
        viewport = results["viewport_randomization"]
        report_lines.extend(
            [
                "",
                "### Viewport Randomization",
                "",
                f"- **Average Execution Time:** {viewport['avg_execution_time_ms']:.3f}ms",
                f"- **Unique Viewports:** {viewport['total_unique_viewports']} / {viewport['iterations']}",
                f"- **Diversity Ratio:** {viewport['diversity_ratio']:.3f}",
                f"- **Width Range:** {viewport['width_distribution']['min']} - {viewport['width_distribution']['max']}px",
                f"- **Height Range:** {viewport['height_distribution']['min']} - {viewport['height_distribution']['max']}px",
            ]
        )

        # Delay patterns summary
        delays = results["delay_patterns"]
        report_lines.extend(
            [
                "",
                "### Delay Patterns",
                "",
                "| Site Profile | Avg Time (ms) | Min Delay (s) | Max Delay (s) | Avg Delay (s) |",
                "|--------------|---------------|---------------|---------------|---------------|",
            ]
        )

        for profile, data in delays.items():
            dist = data["delay_distribution"]
            report_lines.append(
                f"| {profile} | {data['avg_execution_time_ms']:.2f} | "
                f"{dist['min_delay_s']:.2f} | {dist['max_delay_s']:.2f} | {dist['avg_delay_s']:.2f} |"
            )

        # Success monitoring summary
        success_mon = results["success_monitoring"]
        report_lines.extend(
            [
                "",
                "### Success Monitoring",
                "",
                f"- **Record Attempt Avg Time:** {success_mon['record_attempt']['avg_time_ms']:.3f}ms",
                f"- **Get Metrics Avg Time:** {success_mon['get_metrics']['avg_time_ms']:.3f}ms",
                f"- **Final Success Rate:** {success_mon['final_success_rate']:.1%}",
                "",
                "## Performance Recommendations",
                "",
                "Based on the benchmark results:",
                "",
            ]
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        report_lines.extend(recommendations)

        return "\n".join(report_lines)

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate performance recommendations based on results."""
        recommendations = []

        # Analyze config generation performance
        config_gen = results["config_generation"]
        fastest_profile = min(
            config_gen.keys(), key=lambda k: config_gen[k]["avg_execution_time_ms"]
        )
        slowest_profile = max(
            config_gen.keys(), key=lambda k: config_gen[k]["avg_execution_time_ms"]
        )

        recommendations.extend(
            [
                f"1. **Fastest Configuration:** {fastest_profile} profile generates configs most efficiently",
                f"2. **Slowest Configuration:** {slowest_profile} profile has highest overhead",
            ]
        )

        # Analyze user agent diversity
        ua_diversity = results["user_agent_rotation"]["diversity_ratio"]
        if ua_diversity < 0.1:
            recommendations.append(
                "3. **User Agent Pool:** Consider expanding user agent pool for better diversity"
            )
        else:
            recommendations.append(
                "3. **User Agent Pool:** Good diversity achieved in rotation"
            )

        # Analyze viewport diversity
        vp_diversity = results["viewport_randomization"]["diversity_ratio"]
        if vp_diversity < 0.1:
            recommendations.append(
                "4. **Viewport Pool:** Consider expanding viewport profiles for better diversity"
            )
        else:
            recommendations.append(
                "4. **Viewport Pool:** Good diversity achieved in randomization"
            )

        # Analyze delay patterns
        delay_results = results["delay_patterns"]
        extreme_delay = (
            delay_results.get("cloudflare.com", {})
            .get("delay_distribution", {})
            .get("avg_delay_s", 0)
        )
        if extreme_delay > 10:
            recommendations.append(
                "5. **Delay Optimization:** Extreme site delays may impact performance - consider optimization"
            )
        else:
            recommendations.append(
                "5. **Delay Patterns:** Delay patterns are well-balanced for stealth vs performance"
            )

        # Analyze success monitoring overhead
        record_time = results["success_monitoring"]["record_attempt"]["avg_time_ms"]
        if record_time > 1.0:
            recommendations.append(
                "6. **Monitoring Overhead:** Success monitoring has measurable overhead - consider optimization"
            )
        else:
            recommendations.append(
                "6. **Monitoring Performance:** Success monitoring has minimal performance impact"
            )

        return recommendations


@click.command()
@click.option(
    "--output-dir",
    "-o",
    default="benchmark_results",
    help="Output directory for results",
)
@click.option("--report", "-r", is_flag=True, help="Generate human-readable report")
@click.option(
    "--iterations", "-i", default=100, help="Number of iterations for each benchmark"
)
def main(output_dir: str, report: bool, iterations: int):
    """Run Enhanced Anti-Detection performance benchmarks."""
    try:
        benchmark = PerformanceBenchmark(Path(output_dir))

        # Run benchmarks
        results = asyncio.run(benchmark.run_comprehensive_benchmark())

        # Generate report if requested
        if report:
            report_content = benchmark.generate_report(results)
            report_file = (
                Path(output_dir) / f"anti_detection_report_{int(time.time())}.md"
            )

            with open(report_file, "w") as f:
                f.write(report_content)

            click.echo(f"📊 Report generated: {report_file}")

            # Display summary in terminal
            click.echo("\n" + "=" * 60)
            click.echo("PERFORMANCE SUMMARY")
            click.echo("=" * 60)

            # Quick summary table
            summary_data = []
            for profile, data in results["config_generation"].items():
                summary_data.append(
                    [
                        profile,
                        f"{data['avg_execution_time_ms']:.2f}ms",
                        f"{data['avg_memory_usage_mb']:.2f}MB",
                        data["config_complexity"]["extra_args_count"],
                    ]
                )

            click.echo("\nConfiguration Generation Performance:")
            click.echo(
                tabulate(
                    summary_data,
                    headers=["Site Profile", "Avg Time", "Memory", "Args Count"],
                    tablefmt="grid",
                )
            )

    except KeyboardInterrupt:
        click.echo("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception:
        click.echo(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
