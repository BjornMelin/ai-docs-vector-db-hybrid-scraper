#!/usr/bin/env python3
"""Fast test execution script for optimized CI/CD and development feedback loops."""

import argparse
import os  # noqa: PLC0415
import subprocess
import sys
import time  # noqa: PLC0415
from pathlib import Path


def run_command(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run command with timeout and capture output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run optimized fast test suite")
    parser.add_argument(
        "--profile", 
        choices=["unit", "fast", "medium", "integration", "full"],
        default="fast",
        help="Test execution profile"
    )
    parser.add_argument(
        "--parallel", 
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto)"
    )
    parser.add_argument(
        "--timeout",
        type=int, 
        default=300,
        help="Test execution timeout in seconds"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Enable coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Base command with optimized pytest configuration
    cmd = ["uv", "run", "pytest", "-c", "pytest-optimized.ini"]
    
    # Test selection based on profile
    if args.profile == "unit":
        cmd.extend(["-m", "unit and fast", "--maxfail=1"])
        timeout = 60
    elif args.profile == "fast": 
        cmd.extend(["-m", "(unit or fast) and not slow", "--maxfail=3"])
        timeout = 120
    elif args.profile == "medium":
        cmd.extend(["-m", "(fast or medium) and not slow", "--maxfail=5"])
        timeout = 300
    elif args.profile == "integration":
        cmd.extend(["-m", "integration and not slow"])
        timeout = 600
    else:  # full
        cmd.extend(["--maxfail=10"])
        timeout = 1200
    
    # Parallel execution
    if args.parallel > 0:
        cmd.extend(["-n", str(args.parallel)])
    else:
        cmd.extend(["-n", "auto"])
    
    # Coverage
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing:skip-covered"])
    
    # Verbosity
    if args.verbose:
        cmd.extend(["-v", "--tb=short"])
    else:
        cmd.extend(["-q", "--tb=line"])
    
    # Override timeout
    timeout = min(args.timeout, timeout)
    
    print(f"🚀 Running {args.profile} test profile with {args.parallel or 'auto'} workers")
    print(f"⏱️  Timeout: {timeout}s")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    # Execute tests
    returncode, stdout, stderr = run_command(cmd, timeout)
    
    execution_time = time.time() - start_time
    
    # Output results
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    
    # Performance summary
    print(f"\n📊 Execution Summary:")
    print(f"   Profile: {args.profile}")
    print(f"   Duration: {execution_time:.2f}s")
    print(f"   Status: {'✅ PASSED' if returncode == 0 else '❌ FAILED'}")
    
    # Performance targets
    target_times = {
        "unit": 30,
        "fast": 60, 
        "medium": 180,
        "integration": 300,
        "full": 600
    }
    
    target = target_times.get(args.profile, 300)
    if execution_time > target:
        print(f"⚠️  WARNING: Execution time {execution_time:.2f}s exceeded target {target}s")
    else:
        print(f"✅ Performance target met: {execution_time:.2f}s <= {target}s")
    
    return returncode


if __name__ == "__main__":
    sys.exit(main())