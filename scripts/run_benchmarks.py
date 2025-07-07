#!/usr/bin/env python3
"""Run performance benchmarks with proper configuration for CI/CD environments."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Set up environment variables for benchmarking."""
    env = os.environ.copy()

    # CI environment setup
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        env.update(
            {
                "CI": "true",
                "TESTING": "true",
                "SKIP_BROWSER_TESTS": "1",
                "SKIP_DOCKER_TESTS": "1",
                "SKIP_INTEGRATION_TESTS": "1",
            }
        )

    return env


def create_directories():
    """Create necessary directories for benchmarks."""
    directories = [
        "tests/fixtures/cache",
        "tests/fixtures/data",
        "tests/fixtures/logs",
        "tests/fixtures/vectors",
        "tests/fixtures/embeddings",
        "tests/benchmarks/cache",
        "tests/benchmarks/data",
        "tests/benchmarks/logs",
        "tests/benchmarks/.benchmarks",
        "logs",
        "cache",
        "data",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_benchmark_suite(suite_name, output_file, env):
    """Run a specific benchmark suite."""
    if suite_name == "config":
        test_path = "tests/benchmarks/test_config_performance.py"
    elif suite_name == "core":
        test_path = "tests/benchmarks"
    elif suite_name == "integration":
        test_path = "tests/performance"
    else:
        raise ValueError(f"Unknown benchmark suite: {suite_name}")

    # Check if path exists
    if not Path(test_path).exists():
        print(f"❌ Benchmark path not found: {test_path}")
        return False

    cmd = [
        "uv",
        "run",
        "pytest",
        test_path,
        "--benchmark-only",
        f"--benchmark-json={output_file}",
        "--benchmark-verbose",
        "--benchmark-min-rounds=3",
        "--benchmark-max-time=10",  # 10 seconds per benchmark
        "-m",
        "not slow",  # Exclude slow tests
        "--maxfail=5",
        "--tb=short",
        "-v",
    ]

    print(f"⚡ Running {suite_name} benchmarks...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for entire suite
        )

        if result.returncode != 0:
            print(f"❌ Benchmark suite {suite_name} failed")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        print(f"✅ Benchmark suite {suite_name} completed successfully")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Benchmark suite {suite_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running benchmark suite {suite_name}: {e}")
        return False


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        "--suite",
        choices=["config", "core", "integration", "all"],
        default="all",
        help="Which benchmark suite to run",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save benchmark results",
    )

    args = parser.parse_args()

    # Setup
    create_directories()
    env = setup_environment()

    # Determine which suites to run
    if args.suite == "all":
        suites = ["config", "core", "integration"]
    else:
        suites = [args.suite]

    # Run benchmarks
    success = True
    results = {}

    for suite in suites:
        output_file = Path(args.output_dir) / f"{suite}_benchmark_results.json"
        suite_success = run_benchmark_suite(suite, str(output_file), env)
        success = success and suite_success

        # Try to load results
        if output_file.exists():
            try:
                with open(output_file) as f:
                    results[suite] = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load results for {suite}: {e}")

    # Summary
    print("\n📊 Benchmark Summary:")
    print(f"Suites run: {', '.join(suites)}")
    print(f"Overall success: {'✅' if success else '❌'}")

    for suite, data in results.items():
        if "benchmarks" in data:
            print(f"\n{suite.upper()} Suite:")
            print(f"  Tests run: {len(data['benchmarks'])}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
