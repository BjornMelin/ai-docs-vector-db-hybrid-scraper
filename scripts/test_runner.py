#!/usr/bin/env python3
"""Optimized test runner with execution profiles for different scenarios.

This script provides optimized test execution strategies for:
- Local development (quick feedback)
- CI/CD pipelines (parallel, fail-fast)
- Quality assurance (comprehensive coverage)
- Performance benchmarking
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


class TestRunner:
    """Intelligent test runner with multiple execution profiles."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.is_ci = bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))
        
    def get_cpu_count(self) -> int:
        """Get optimal worker count for parallel execution."""
        cpu_count = os.cpu_count() or 4
        if self.is_ci:
            # CI environments: Use fewer workers to avoid resource exhaustion
            return min(cpu_count, 4)
        else:
            # Local: Use more aggressive parallelization but leave one CPU free
            return max(1, cpu_count - 1)
    
    def run_quick_tests(self) -> int:
        """Run fast unit tests for local development."""
        print("🚀 Running quick tests (unit + fast)...")
        
        cmd = [
            "uv", "run", "pytest",
            "-m", "unit and fast and not slow",
            "--tb=short",
            "-q",
            "--maxfail=5",
            f"-n={self.get_cpu_count()}",
            "--dist=worksteal",
        ]
        
        return self._execute_command(cmd)
    
    def run_ci_tests(self) -> int:
        """Run CI-optimized test suite."""
        print("🔄 Running CI test suite...")
        
        cmd = [
            "uv", "run", "pytest",
            "-m", "not local_only and not slow",
            "--tb=short",
            "-q",
            "--maxfail=3",
            f"-n={self.get_cpu_count()}",
            "--dist=worksteal",
            "--cov=src",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=40",
        ]
        
        if self.is_ci:
            cmd.extend([
                "--timeout=180",
                "--durations=20",
            ])
        
        return self._execute_command(cmd)
    
    def run_full_tests(self) -> int:
        """Run comprehensive test suite."""
        print("📋 Running full test suite...")
        
        cmd = [
            "uv", "run", "pytest",
            "--tb=short",
            "-v",
            f"-n={self.get_cpu_count()}",
            "--dist=worksteal",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--durations=20",
        ]
        
        return self._execute_command(cmd)
    
    def run_performance_tests(self) -> int:
        """Run performance and benchmark tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            "uv", "run", "pytest",
            "-m", "performance or benchmark",
            "--tb=short",
            "-v",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-group-by=group",
        ]
        
        return self._execute_command(cmd)
    
    def run_mutation_tests(self) -> int:
        """Run mutation testing for quality validation."""
        print("🧬 Running mutation tests...")
        
        if not (self.project_root / "mutmut_config.ini").exists():
            print("❌ Mutation testing config not found")
            return 1
        
        cmd = [
            "uv", "run", "mutmut", "run",
            "--paths-to-mutate", "src/config/",
            "--tests-dir", "tests/unit/config/",
            "--runner", "uv run pytest -x tests/unit/config/ -q",
            "--processes", str(min(4, self.get_cpu_count())),
        ]
        
        return self._execute_command(cmd)
    
    def run_lint_and_format(self) -> int:
        """Run linting and formatting."""
        print("🔍 Running linting and formatting...")
        
        # Run ruff check and format
        check_result = self._execute_command([
            "uv", "run", "ruff", "check", ".", "--fix"
        ])
        
        format_result = self._execute_command([
            "uv", "run", "ruff", "format", "."
        ])
        
        return max(check_result, format_result)
    
    def check_coverage_quality(self) -> int:
        """Check test coverage quality."""
        print("📊 Checking coverage quality...")
        
        # Run tests with coverage
        cmd = [
            "uv", "run", "pytest", 
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-fail-under=40",
            "-m", "unit",
            "-q",
        ]
        
        return self._execute_command(cmd)
    
    def _execute_command(self, cmd: List[str]) -> int:
        """Execute command and return exit code."""
        print(f"Executing: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=False)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Command completed successfully in {duration:.1f}s")
            else:
                print(f"❌ Command failed with exit code {result.returncode} after {duration:.1f}s")
            
            return result.returncode
            
        except KeyboardInterrupt:
            print("\n🛑 Test execution interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error executing command: {e}")
            return 1


def main():
    """Main entry point for test runner."""
    if len(sys.argv) < 2:
        print("""
Test Runner - Optimized test execution profiles

Usage: python scripts/test_runner.py <profile>

Available profiles:
  quick      - Fast unit tests for local development
  ci         - CI-optimized test suite with coverage
  full       - Comprehensive test suite with detailed output
  performance - Performance and benchmark tests only
  mutation   - Mutation testing for quality validation
  lint       - Linting and code formatting
  coverage   - Coverage quality check

Examples:
  python scripts/test_runner.py quick     # Fast feedback during development
  python scripts/test_runner.py ci        # CI pipeline execution
  python scripts/test_runner.py full      # Complete local testing
        """)
        return 1
    
    profile = sys.argv[1].lower()
    runner = TestRunner()
    
    profile_methods = {
        "quick": runner.run_quick_tests,
        "ci": runner.run_ci_tests,
        "full": runner.run_full_tests,
        "performance": runner.run_performance_tests,
        "mutation": runner.run_mutation_tests,
        "lint": runner.run_lint_and_format,
        "coverage": runner.check_coverage_quality,
    }
    
    if profile not in profile_methods:
        print(f"❌ Unknown profile: {profile}")
        print(f"Available profiles: {', '.join(profile_methods.keys())}")
        return 1
    
    print(f"\n🎯 Executing profile: {profile}")
    return profile_methods[profile]()


if __name__ == "__main__":
    sys.exit(main())