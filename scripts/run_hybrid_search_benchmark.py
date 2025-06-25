#!/usr/bin/env python3
"""Main script for running Advanced Hybrid Search benchmarks.

This script provides a command-line interface for executing comprehensive
performance benchmarks of the Advanced Hybrid Search system.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks import HybridSearchBenchmark
from benchmarks import BenchmarkConfig
from benchmarks import BenchmarkResults
from ..config import Config
from services.vector_db.hybrid_search import HybridSearchService


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("benchmark.log")],
    )


async def initialize_search_service(config_path: str) -> HybridSearchService:
    """Initialize the Hybrid Search service.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized search service
    """
    # Load configuration
    config = UnifiedConfig.load_from_file(config_path)

    # Initialize search service
    search_service = HybridSearchService(config)
    await search_service.initialize()

    return search_service


def create_benchmark_config(args) -> BenchmarkConfig:
    """Create benchmark configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Benchmark configuration
    """
    return BenchmarkConfig(
        name=args.benchmark_name,
        description=f"Advanced Hybrid Search benchmark - {args.benchmark_name}",
        target_latency_p95_ms=args.target_latency_p95,
        target_throughput_qps=args.target_throughput,
        target_memory_mb=args.target_memory,
        target_cache_hit_rate=args.target_cache_hit_rate,
        target_accuracy=args.target_accuracy,
        test_queries_per_type=args.test_queries_per_type,
        concurrent_users=args.concurrent_users,
        test_duration_seconds=args.test_duration,
        enable_component_benchmarks=args.enable_component_benchmarks,
        enable_load_testing=args.enable_load_testing,
        enable_profiling=args.enable_profiling,
        enable_optimization_analysis=args.enable_optimization_analysis,
    )


async def run_benchmark_suite(
    search_service: HybridSearchService,
    benchmark_config: BenchmarkConfig,
    output_dir: Path,
) -> BenchmarkResults:
    """Run complete benchmark suite.

    Args:
        search_service: Initialized search service
        benchmark_config: Benchmark configuration
        output_dir: Output directory for results

    Returns:
        Benchmark results
    """
    logger = logging.getLogger(__name__)

    # Create benchmark orchestrator
    config = search_service.config
    benchmark = HybridSearchBenchmark(config, search_service, benchmark_config)

    logger.info(f"Starting benchmark suite: {benchmark_config.name}")
    logger.info(f"Configuration: {benchmark_config.model_dump()}")

    # Run comprehensive benchmark
    results = await benchmark.run_comprehensive_benchmark()

    # Save results
    await benchmark.save_results(results, output_dir)

    return results


def print_benchmark_summary(results: BenchmarkResults) -> None:
    """Print benchmark summary to console.

    Args:
        results: Benchmark results
    """
    print("\n" + "=" * 80)
    print("🔍 ADVANCED HYBRID SEARCH BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Benchmark: {results.benchmark_name}")
    print(f"Duration: {results.duration_seconds:.2f} seconds")
    print(f"Status: {'✅ PASS' if results.meets_targets else '❌ FAIL'}")
    print(f"Timestamp: {results.timestamp}")

    if results.latency_metrics:
        print("\n📊 LATENCY METRICS:")
        for metric, value in results.latency_metrics.items():
            status = "✅" if value < 300 else "⚠️" if value < 500 else "❌"
            print(f"  {status} {metric}: {value:.1f}ms")

    if results.throughput_metrics:
        print("\n⚡ THROUGHPUT METRICS:")
        for metric, value in results.throughput_metrics.items():
            status = "✅" if value > 100 else "⚠️" if value > 50 else "❌"
            print(f"  {status} {metric}: {value:.1f} QPS")

    if results.resource_metrics:
        print("\n💾 RESOURCE METRICS:")
        for metric, value in results.resource_metrics.items():
            if "memory" in metric.lower():
                status = "✅" if value < 1000 else "⚠️" if value < 2000 else "❌"
                unit = "MB"
            else:
                status = "✅" if value < 70 else "⚠️" if value < 90 else "❌"
                unit = "%"
            print(f"  {status} {metric}: {value:.1f}{unit}")

    if results.accuracy_metrics:
        print("\n🎯 ACCURACY METRICS:")
        for metric, value in results.accuracy_metrics.items():
            percentage = value * 100
            status = "✅" if percentage > 80 else "⚠️" if percentage > 60 else "❌"
            print(f"  {status} {metric}: {percentage:.1f}%")

    if results.failed_targets:
        print("\n❌ FAILED TARGETS:")
        for target in results.failed_targets:
            print(f"  • {target}")

    if results.optimization_recommendations:
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for recommendation in results.optimization_recommendations[:5]:  # Show top 5
            print(f"  • {recommendation}")

    print("\n" + "=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Advanced Hybrid Search benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic configuration
    parser.add_argument(
        "--config",
        "-c",
        default="config/development.json",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--benchmark-name",
        "-n",
        default="hybrid_search_benchmark",
        help="Name of the benchmark",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="benchmark_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Performance targets
    parser.add_argument(
        "--target-latency-p95",
        type=float,
        default=300.0,
        help="Target 95th percentile latency in milliseconds",
    )

    parser.add_argument(
        "--target-throughput",
        type=float,
        default=500.0,
        help="Target throughput in queries per second",
    )

    parser.add_argument(
        "--target-memory", type=float, default=2048.0, help="Target memory usage in MB"
    )

    parser.add_argument(
        "--target-cache-hit-rate",
        type=float,
        default=0.8,
        help="Target cache hit rate (0.0-1.0)",
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.85,
        help="Target ML accuracy (0.0-1.0)",
    )

    # Test configuration
    parser.add_argument(
        "--test-queries-per-type",
        type=int,
        default=100,
        help="Number of test queries per type",
    )

    parser.add_argument(
        "--concurrent-users",
        type=int,
        nargs="+",
        default=[10, 50, 200],
        help="Concurrent user levels to test",
    )

    parser.add_argument(
        "--test-duration", type=int, default=300, help="Test duration in seconds"
    )

    # Feature toggles
    parser.add_argument(
        "--disable-component-benchmarks",
        action="store_true",
        help="Disable component-level benchmarks",
    )

    parser.add_argument(
        "--disable-load-testing", action="store_true", help="Disable load testing"
    )

    parser.add_argument(
        "--disable-profiling", action="store_true", help="Disable performance profiling"
    )

    parser.add_argument(
        "--disable-optimization-analysis",
        action="store_true",
        help="Disable optimization analysis",
    )

    # Output options
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON results (no HTML report)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )

    args = parser.parse_args()

    # Convert disable flags to enable flags
    args.enable_component_benchmarks = not args.disable_component_benchmarks
    args.enable_load_testing = not args.disable_load_testing
    args.enable_profiling = not args.disable_profiling
    args.enable_optimization_analysis = not args.disable_optimization_analysis

    return args


async def main():
    """Main benchmark execution function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize search service
        logger.info("Initializing Advanced Hybrid Search service...")
        search_service = await initialize_search_service(args.config)

        # Create benchmark configuration
        benchmark_config = create_benchmark_config(args)

        # Run benchmark suite
        logger.info("Starting benchmark execution...")
        results = await run_benchmark_suite(
            search_service, benchmark_config, output_dir
        )

        # Print summary unless quiet mode
        if not args.quiet:
            print_benchmark_summary(results)

        # Exit with appropriate code
        exit_code = 0 if results.meets_targets else 1
        logger.info(f"Benchmark completed with exit code: {exit_code}")

        return exit_code

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
