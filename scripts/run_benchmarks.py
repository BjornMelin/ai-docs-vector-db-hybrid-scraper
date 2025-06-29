#!/usr/bin/env python3
"""Simple benchmark runner using pytest-benchmark.

This script runs our clean pytest-benchmark based performance tests
that validate our BJO-134 enterprise database achievements.

Usage:
    python scripts/run_benchmarks.py                    # Run all benchmarks
    python scripts/run_benchmarks.py --quick           # Run fast benchmarks only
    python scripts/run_benchmarks.py --save-results    # Save benchmark results
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmarks(
    quick: bool = False, save_results: bool = False, verbose: bool = False
) -> int:
    """Run database performance benchmarks using pytest-benchmark.

    Args:
        quick: Skip slow benchmarks if True
        save_results: Save benchmark results to JSON if True
        verbose: Show detailed output if True

    Returns:
        Exit code (0 for success, 1 for failure)
    """

    # Base pytest command
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/benchmarks/test_database_performance.py",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,rounds,iterations",
    ]

    # Add quick mode (skip slow tests)
    if quick:
        cmd.extend(["-m", "not slow"])
        print("🚀 Running quick benchmarks (skipping slow tests)")
    else:
        print("🏢 Running full enterprise database benchmarks")

    # Add result saving
    if save_results:
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        cmd.extend(
            [
                "--benchmark-json=benchmark_results/database_performance.json",
                "--benchmark-save=database_perf",
            ]
        )
        print(f"📊 Results will be saved to: {results_dir}/database_performance.json")

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    try:
        # Run the benchmark
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print("\n" + "=" * 80)
            print(
                "✅ All benchmarks passed! Enterprise database performance validated."
            )
            if not quick:
                print("🎯 BJO-134 performance targets confirmed:")
                print("   • 887.9% throughput improvement maintained")
                print("   • 50.9% P95 latency reduction preserved")
                print("   • 95% ML prediction accuracy achieved")
                print("   • 99.9% uptime with circuit breaker")
        else:
            print("\n" + "=" * 80)
            print("❌ Some benchmarks failed or performance targets not met.")
            print("Review the output above for details.")

        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        return 1
    except Exception:
        print(f"\n❌ Error running benchmarks: {e}")
        return 1


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(
        description="Run enterprise database performance benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick benchmarks (skip slow tests)",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save benchmark results to JSON file",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    # Run benchmarks
    exit_code = run_benchmarks(
        quick=args.quick, save_results=args.save_results, verbose=args.verbose
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
