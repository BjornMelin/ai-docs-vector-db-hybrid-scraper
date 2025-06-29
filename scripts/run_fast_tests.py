#!/usr/bin/env python3
"""Fast test execution script for optimized CI/CD and development feedback loops."""

import argparse
import os  # noqa: PLC0415
import subprocess
import sys
import time  # noqa: PLC0415
from pathlib import Path


class FastTestRunner:
    """Optimized test runner for fast feedback loops."""
    
    def __init__(self):
        self.cpu_count = max(1, (os.cpu_count() or 4) - 1)
        self.is_ci = os.getenv('CI', '').lower() in ('true', '1')
        self.project_root = Path(__file__).parent.parent
        
    def run_command(self, cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
        """Run command with timeout and capture output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout} seconds"
    
    def run_fast_tests(self, profile: str = "fast", parallel: int = 0, verbose: bool = False, coverage: bool = False) -> int:
        """Run unit tests with aggressive optimization."""
        cmd = [
            "uv", "run", "pytest",
            "tests/unit/",
            "-x",  # Stop on first failure for fast feedback
            "--ff",  # Failed first
            f"-n{parallel if parallel > 0 else self.cpu_count}",  # Parallel execution
            "--tb=short",  # Shorter tracebacks
            "--disable-warnings",  # Less noise in development
            "--durations=5"  # Show slowest tests
        ]
        
        # Profile-specific optimizations
        if profile == "unit":
            cmd.extend(["-m", "fast", "--maxfail=1"])
            timeout = 60
        elif profile == "fast":
            cmd.extend(["-m", "(unit or fast) and not slow", "--maxfail=3"])
            timeout = 120
        else:
            timeout = 300
            
        if not self.is_ci:
            cmd.extend([
                "--lf",  # Last failed only (for local development)
                "-q" if not verbose else "-v"
            ])
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing:skip-covered"])
            
        return self.run_command(cmd, timeout)
    
    def run_integration_tests(self, parallel: int = 0) -> tuple[int, str, str]:
        """Run integration tests separately."""
        cmd = [
            "uv", "run", "pytest", 
            "tests/integration/",
            "-v",
            f"-n{min(parallel if parallel > 0 else self.cpu_count, 4)}"  # Fewer workers for integration
        ]
        return self.run_command(cmd, 600)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run optimized fast test suite")
    parser.add_argument(
        "--profile",
        choices=["unit", "fast", "medium", "integration", "full"],
        default="fast",
        help="Test execution profile",
    )
    parser.add_argument(
        "--parallel", type=int, default=0, help="Number of parallel workers (0 = auto)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Test execution timeout in seconds"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )

    args = parser.parse_args()
    
    # Initialize test runner
    runner = FastTestRunner()

    print(
        f"🚀 Running {args.profile} test profile with {args.parallel or 'auto'} workers"
    )
    print(f"⏱️  Timeout: {args.timeout}s")
    print()

    start_time = time.time()

    # Execute tests based on profile
    if args.profile in ["unit", "fast"]:
        returncode, stdout, stderr = runner.run_fast_tests(
            profile=args.profile,
            parallel=args.parallel,
            verbose=args.verbose,
            coverage=args.coverage
        )
    elif args.profile == "integration":
        returncode, stdout, stderr = runner.run_integration_tests(
            parallel=args.parallel
        )
    else:  # medium, full
        # Use comprehensive test runner for complex profiles
        cmd = ["uv", "run", "pytest"]
        if args.profile == "medium":
            cmd.extend(["-m", "(fast or medium) and not slow", "--maxfail=5"])
        else:  # full
            cmd.extend(["--maxfail=10"])
            
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
            
        if args.verbose:
            cmd.extend(["-v"])
        else:
            cmd.extend(["-q"])
            
        returncode, stdout, stderr = runner.run_command(cmd, args.timeout)

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
        "full": 600,
    }

    target = target_times.get(args.profile, 300)
    if execution_time > target:
        print(
            f"⚠️  WARNING: Execution time {execution_time:.2f}s exceeded target {target}s"
        )
    else:
        print(f"✅ Performance target met: {execution_time:.2f}s <= {target}s")

    return returncode


if __name__ == "__main__":
    sys.exit(main())
