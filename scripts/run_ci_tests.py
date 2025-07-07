#!/usr/bin/env python3
"""Run tests with optimal CI configuration.

This script automatically detects the CI environment and runs tests
with the best configuration for that platform.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.ci.pytest_xdist_config import XDistOptimizer, get_xdist_args
from tests.ci.test_environments import detect_environment, setup_test_environment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tests with optimal CI configuration"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e", "all", "fast", "full"],
        default="all",
        help="Type of tests to run",
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting",
    )
    
    parser.add_argument(
        "--performance-report",
        action="store_true",
        help="Generate performance report",
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity",
    )
    
    parser.add_argument(
        "--markers",
        "-m",
        help="Additional pytest markers to apply",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        help="Override number of workers",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override test timeout in seconds",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    
    return parser.parse_args()


def build_pytest_command(args):
    """Build the pytest command with all optimizations."""
    # Detect environment and set up
    env = detect_environment()
    env_pytest_args = setup_test_environment()
    
    # Get xdist optimization args
    optimizer = XDistOptimizer()
    xdist_config = optimizer.get_optimal_config()
    
    # Override workers if specified
    if args.workers:
        xdist_config.num_workers = args.workers
    
    # Override timeout if specified
    if args.timeout:
        xdist_config.timeout = args.timeout
    
    xdist_args = get_xdist_args(xdist_config)
    
    # Build base command
    cmd = ["pytest"]
    
    # Add configuration file
    if env.name != "LocalEnvironment":
        cmd.extend(["-c", "pytest-ci.ini"])
    
    # Add xdist args
    cmd.extend(xdist_args)
    
    # Add environment-specific args
    cmd.extend(env_pytest_args)
    
    # Test selection based on type
    test_paths = []
    markers = []
    
    if args.test_type == "unit":
        test_paths = ["tests/unit"]
        markers.append("not integration and not e2e")
    elif args.test_type == "integration":
        test_paths = ["tests/integration"]
        markers.append("integration")
    elif args.test_type == "e2e":
        test_paths = ["tests/e2e"]
        markers.append("e2e")
    elif args.test_type == "fast":
        test_paths = ["tests/unit"]
        markers.append("not slow and not integration and not e2e")
    elif args.test_type == "full":
        test_paths = ["tests"]
        # No marker filter for full
    else:  # all
        test_paths = ["tests"]
        markers.append("not e2e")  # Skip e2e by default
    
    # Add custom markers
    if args.markers:
        markers.append(args.markers)
    
    if markers:
        marker_expr = " and ".join(f"({m})" for m in markers)
        cmd.extend(["-m", marker_expr])
    
    # Coverage options
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
        ])
    
    # Performance reporting
    if args.performance_report:
        report_path = f"test-performance-{env.name.lower()}.json"
        cmd.append(f"--performance-report={report_path}")
    
    # Fail fast
    if args.fail_fast:
        cmd.extend(["-x", "--maxfail=1"])
    
    # Verbosity
    if args.verbose:
        cmd.append("-" + "v" * args.verbose)
    else:
        cmd.append("-q")
    
    # Add test paths
    cmd.extend(test_paths)
    
    return cmd


def print_configuration():
    """Print the current configuration for debugging."""
    optimizer = XDistOptimizer()
    env_info = optimizer.env_info
    config = optimizer.get_optimal_config()
    
    print("\n" + "="*60)
    print("CI Test Configuration")
    print("="*60)
    
    print(f"\nEnvironment: {detect_environment().name}")
    print(f"Platform: {env_info['platform']}")
    print(f"CPU Count: {env_info['cpu_count']}")
    print(f"Memory: {env_info['memory_gb']:.1f} GB")
    
    if env_info['ci_provider']:
        print(f"CI Provider: {env_info['ci_provider']}")
        print(f"Runner Specs: {env_info['runner_specs']}")
    
    print(f"\nWorker Configuration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Max Workers: {config.max_workers}")
    print(f"  Distribution: {config.dist_mode}")
    print(f"  Timeout: {config.timeout}s")
    print(f"  Memory/Worker: {config.max_memory_per_worker_mb} MB")
    
    print("\n" + "="*60 + "\n")


def run_tests(cmd, dry_run=False):
    """Run the test command and handle results."""
    if dry_run:
        print("Would run:")
        print(" ".join(cmd))
        return 0
    
    print("Running tests with command:")
    print(" ".join(cmd))
    print()
    
    start_time = time.time()
    
    # Run pytest
    result = subprocess.run(cmd, env=os.environ.copy())
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nTests completed in {duration:.2f} seconds")
    
    # Check for performance report
    perf_reports = list(Path(".").glob("test-performance*.json"))
    if perf_reports:
        print("\nPerformance reports generated:")
        for report in perf_reports:
            print(f"  - {report}")
            
            # Print summary if available
            try:
                with open(report) as f:
                    data = json.load(f)
                    print(f"    Total tests: {data['metadata']['total_tests']}")
                    print(f"    Workers used: {data['metadata']['workers_used']}")
                    print(f"    Total duration: {data['metadata']['total_duration']:.2f}s")
            except Exception:
                pass
    
    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print configuration in verbose mode
    if args.verbose:
        print_configuration()
    
    # Build command
    cmd = build_pytest_command(args)
    
    # Run tests
    return_code = run_tests(cmd, args.dry_run)
    
    # Exit with test result code
    sys.exit(return_code)


if __name__ == "__main__":
    main()