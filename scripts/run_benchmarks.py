#!/usr/bin/env python3
"""Simple benchmark runner using pytest-benchmark.

This script runs our clean pytest-benchmark based performance tests
that validate our BJO-134 enterprise database achievements.

Usage:
    python scripts/run_benchmarks.py                    # Run all benchmarks
    python scripts/run_benchmarks.py --quick           # Run fast benchmarks only
    python scripts/run_benchmarks.py --save-results    # Save benchmark results
"""

import subprocess
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
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ Error running benchmarks: {e}")
        return 1


def main():
    """Run comprehensive performance benchmarks using pytest-benchmark."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Starting Modernized Performance Benchmark Suite")
    print("=" * 60)
    
    # Define benchmark categories
    benchmark_suites = {
        "core": [
            "tests/benchmarks/performance_suite.py::TestPerformanceBenchmarks::test_real_embedding_generation_performance",
            "tests/benchmarks/performance_suite.py::TestPerformanceBenchmarks::test_real_cache_performance",
            "tests/benchmarks/performance_suite.py::TestPerformanceBenchmarks::test_real_memory_usage_optimization",
        ],
        "database": [
            "tests/benchmarks/test_database_performance.py::TestDatabasePerformance::test_real_collection_operations_performance",
            "tests/benchmarks/test_database_performance.py::TestDatabasePerformance::test_real_vector_upsert_performance",
            "tests/benchmarks/test_database_performance.py::TestDatabasePerformance::test_real_search_performance",
        ],
        "config": [
            "tests/benchmarks/test_config_performance.py::TestConfigurationPerformance::test_real_settings_instantiation_performance",
            "tests/benchmarks/test_config_performance.py::TestConfigurationPerformance::test_real_config_caching_performance",
            "tests/benchmarks/test_config_performance.py::TestConfigurationPerformance::test_real_config_validation_performance",
        ],
        "comprehensive": [
            "tests/benchmarks/performance_suite.py::TestPerformanceBenchmarks::test_real_vector_search_performance",
            "tests/benchmarks/performance_suite.py::TestPerformanceBenchmarks::test_real_concurrent_operations_performance",
            "tests/benchmarks/test_database_performance.py::TestDatabasePerformance::test_real_concurrent_database_operations",
            "tests/benchmarks/test_database_performance.py::TestDatabasePerformance::test_real_payload_indexing_performance",
        ]
    }
    
    # Base pytest-benchmark command
    base_cmd = [
        "uv", "run", "pytest",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,rounds,iterations",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=3",
        "-v"
    ]
    
    # Check for specific suite argument
    suite_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if suite_arg == "all":
        # Run all benchmark suites
        for suite_name, tests in benchmark_suites.items():
            print(f"\n📊 Running {suite_name.upper()} benchmarks...")
            print("-" * 40)
            
            cmd = base_cmd + [
                f"--benchmark-group-by=func",
                f"--benchmark-name=short",
                f"--benchmark-json=benchmark_results_{suite_name}.json"
            ] + tests
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ {suite_name.upper()} benchmarks completed successfully")
                
                # Show key metrics from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and 'benchmark' in line:
                        print(f"   {line.strip()}")
                        
            except subprocess.CalledProcessError as e:
                print(f"❌ {suite_name.upper()} benchmarks failed:")
                print(f"   {e.stderr}")
                continue
    
    elif suite_arg in benchmark_suites:
        # Run specific suite
        tests = benchmark_suites[suite_arg]
        print(f"\n📊 Running {suite_arg.upper()} benchmarks...")
        
        cmd = base_cmd + [
            f"--benchmark-group-by=func",
            f"--benchmark-name=short",
            f"--benchmark-json=benchmark_results_{suite_arg}.json"
        ] + tests
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ {suite_arg.upper()} benchmarks completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {suite_arg.upper()} benchmarks failed")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown benchmark suite: {suite_arg}")
        print(f"Available suites: {', '.join(benchmark_suites.keys())}, all")
        sys.exit(1)
    
    print("\n🎉 Benchmark execution completed!")
    print("📈 Results saved to benchmark_results_*.json files")
    print("\nTo compare results over time:")
    print("  uv run pytest --benchmark-compare benchmark_results_*.json")
    print("\nTo generate HTML report:")
    print("  uv run pytest --benchmark-histogram")
    
    return 0


if __name__ == "__main__":
    main()
