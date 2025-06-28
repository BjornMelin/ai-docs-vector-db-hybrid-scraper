#!/usr/bin/env python3
"""Test performance profiler for identifying slow tests and optimization opportunities."""

import argparse
import json  # noqa: PLC0415
import os  # noqa: PLC0415
import re
import subprocess
import time  # noqa: PLC0415
from collections import defaultdict
from pathlib import Path


class TestPerformanceProfiler:
    """Profiles test execution performance and identifies optimization opportunities."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.slow_tests = []
        self.optimization_suggestions = []

    def profile_tests(self, pattern: str = "", parallel: bool = True) -> Dict:
        """Profile test execution performance."""
        print("🔍 Profiling test performance...")

        # Run tests with detailed timing
        cmd = [
            "uv",
            "run",
            "pytest",
            "--durations=0",  # Show all test durations
            "--tb=no",  # No traceback for speed
            "-v",  # Verbose for test names
            "--disable-warnings",
        ]

        if parallel:
            cmd.extend(["-n", "auto"])

        if pattern:
            cmd.extend(["-k", pattern])

        # Execute and capture output
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )
        total_time = time.time() - start_time

        # Parse timing results
        timing_data = self._parse_timing_output(result.stdout)

        # Analyze results
        self.results = {
            "total_execution_time": total_time,
            "parallel_enabled": parallel,
            "test_count": len(timing_data),
            "timings": timing_data,
            "analysis": self._analyze_performance(timing_data),
        }

        return self.results

    def _parse_timing_output(self, output: str) -> list[Dict]:
        """Parse pytest timing output."""
        timings = []

        # Look for timing lines in format: "0.12s call test_file.py::TestClass::test_method"
        timing_pattern = r"(\d+\.\d+)s\s+(\w+)\s+(.*)"

        for line in output.split("\n"):
            match = re.search(timing_pattern, line)
            if match:
                duration = float(match.group(1))
                phase = match.group(2)
                test_path = match.group(3)

                # Only include call phase (actual test execution)
                if phase == "call":
                    timings.append(
                        {
                            "test": test_path,
                            "duration": duration,
                            "phase": phase,
                        }
                    )

        return sorted(timings, key=lambda x: x["duration"], reverse=True)

    def _analyze_performance(self, timings: list[Dict]) -> Dict:
        """Analyze test performance and identify issues."""
        if not timings:
            return {"error": "No timing data available"}

        durations = [t["duration"] for t in timings]

        # Calculate statistics
        total_time = sum(durations)
        avg_time = total_time / len(durations)

        # Categorize tests by speed
        fast_tests = [t for t in timings if t["duration"] < 0.1]
        medium_tests = [t for t in timings if 0.1 <= t["duration"] < 2.0]
        slow_tests = [t for t in timings if t["duration"] >= 2.0]

        # Identify bottlenecks
        top_10_slowest = timings[:10]

        # Group by test file
        file_timings = defaultdict(list)
        for timing in timings:
            file_path = timing["test"].split("::")[0]
            file_timings[file_path].append(timing["duration"])

        # Calculate file-level statistics
        file_stats = {}
        for file_path, durations in file_timings.items():
            file_stats[file_path] = {
                "test_count": len(durations),
                "total_time": sum(durations),
                "avg_time": sum(durations) / len(durations),
                "max_time": max(durations),
            }

        # Sort files by total time
        slowest_files = sorted(
            file_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        )[:10]

        return {
            "summary": {
                "total_tests": len(timings),
                "total_time": total_time,
                "average_time": avg_time,
                "fast_tests": len(fast_tests),
                "medium_tests": len(medium_tests),
                "slow_tests": len(slow_tests),
            },
            "categories": {
                "fast": {
                    "count": len(fast_tests),
                    "percentage": len(fast_tests) / len(timings) * 100,
                },
                "medium": {
                    "count": len(medium_tests),
                    "percentage": len(medium_tests) / len(timings) * 100,
                },
                "slow": {
                    "count": len(slow_tests),
                    "percentage": len(slow_tests) / len(timings) * 100,
                },
            },
            "bottlenecks": {
                "slowest_tests": top_10_slowest,
                "slowest_files": slowest_files,
            },
            "optimization_opportunities": self._identify_optimizations(
                timings, file_stats
            ),
        }

    def _identify_optimizations(
        self, timings: list[Dict], file_stats: Dict
    ) -> list[Dict]:
        """Identify specific optimization opportunities."""
        optimizations = []

        # Slow tests that could be optimized
        slow_tests = [t for t in timings if t["duration"] > 2.0]
        if slow_tests:
            optimizations.append(
                {
                    "type": "slow_tests",
                    "priority": "high",
                    "description": f"Found {len(slow_tests)} tests taking >2s each",
                    "suggestion": "Consider mocking external dependencies, reducing test data size, or splitting tests",
                    "tests": [t["test"] for t in slow_tests[:5]],
                }
            )

        # Files with many slow tests
        for file_path, stats in file_stats.items():
            if stats["test_count"] > 5 and stats["avg_time"] > 1.0:
                optimizations.append(
                    {
                        "type": "slow_file",
                        "priority": "medium",
                        "description": f"File {file_path} has {stats['test_count']} tests averaging {stats['avg_time']:.2f}s",
                        "suggestion": "Consider shared fixtures, test data optimization, or setup/teardown improvements",
                        "file": file_path,
                    }
                )

        # Tests that could benefit from parallelization
        unit_test_files = [f for f in file_stats.keys() if "unit" in f]
        slow_unit_files = [
            f for f in unit_test_files if file_stats[f]["avg_time"] > 0.5
        ]

        if slow_unit_files:
            optimizations.append(
                {
                    "type": "parallel_optimization",
                    "priority": "medium",
                    "description": f"Found {len(slow_unit_files)} unit test files that could benefit from better parallelization",
                    "suggestion": "Ensure tests are stateless and can run in parallel safely",
                    "files": slow_unit_files,
                }
            )

        return optimizations

    def generate_report(self, output_file: str = None) -> str:
        """Generate performance report."""
        if not self.results:
            return "No profiling results available. Run profile_tests() first."

        report = []
        report.append("🚀 Test Performance Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Summary
        summary = self.results["analysis"]["summary"]
        report.append("📊 Summary Statistics:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Total Time: {summary['total_time']:.2f}s")
        report.append(f"  Average Time: {summary['average_time']:.3f}s")
        report.append(
            f"  Fast Tests (<0.1s): {summary['fast_tests']} ({summary['fast_tests'] / summary['total_tests'] * 100:.1f}%)"
        )
        report.append(
            f"  Medium Tests (0.1-2s): {summary['medium_tests']} ({summary['medium_tests'] / summary['total_tests'] * 100:.1f}%)"
        )
        report.append(
            f"  Slow Tests (>2s): {summary['slow_tests']} ({summary['slow_tests'] / summary['total_tests'] * 100:.1f}%)"
        )
        report.append("")

        # Performance targets
        report.append("🎯 Performance Targets:")
        report.append("  Unit Tests: < 0.1s average")
        report.append("  Integration Tests: < 2s average")
        report.append("  E2E Tests: < 10s average")
        report.append("  Full Suite: < 5 minutes parallel")
        report.append("")

        # Bottlenecks
        bottlenecks = self.results["analysis"]["bottlenecks"]
        report.append("🐌 Top 5 Slowest Tests:")
        for i, test in enumerate(bottlenecks["slowest_tests"][:5], 1):
            report.append(f"  {i}. {test['test']} - {test['duration']:.2f}s")
        report.append("")

        report.append("📁 Top 5 Slowest Files:")
        for i, (file_path, stats) in enumerate(bottlenecks["slowest_files"][:5], 1):
            report.append(
                f"  {i}. {file_path} - {stats['total_time']:.2f}s ({stats['test_count']} tests)"
            )
        report.append("")

        # Optimizations
        optimizations = self.results["analysis"]["optimization_opportunities"]
        if optimizations:
            report.append("💡 Optimization Opportunities:")
            for opt in optimizations:
                report.append(
                    f"  🔧 {opt['type'].title()} ({opt['priority']} priority)"
                )
                report.append(f"     {opt['description']}")
                report.append(f"     💡 {opt['suggestion']}")
                report.append("")

        # Recommendations
        report.append("📋 Recommendations:")

        if summary["slow_tests"] > summary["total_tests"] * 0.1:
            report.append("  ⚠️  More than 10% of tests are slow (>2s)")
            report.append(
                "     Consider mocking external dependencies and optimizing test data"
            )

        if summary["average_time"] > 0.5:
            report.append("  ⚠️  Average test time is high (>0.5s)")
            report.append("     Focus on unit test optimization and fixture efficiency")

        if self.results["parallel_enabled"] and summary["total_time"] > 300:
            report.append("  ⚠️  Total execution time is high even with parallelization")
            report.append("     Consider test categorization and selective execution")

        report.append(
            "  ✅ Consider implementing test markers for fast/slow categorization"
        )
        report.append("  ✅ Use session-scoped fixtures for expensive setup")
        report.append(
            "  ✅ Implement test data factories for consistent, minimal test data"
        )
        report.append("  ✅ Monitor test performance in CI/CD pipelines")

        report_text = "\n".join(report)

        if output_file:
            output_path = Path(output_file)
            output_path.write_text(report_text)
            print(f"📄 Report saved to: {output_path}")

        return report_text

    def save_results(self, output_file: str):
        """Save detailed results to JSON file."""
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Detailed results saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Profile test performance")
    parser.add_argument(
        "--pattern", default="", help="Test pattern to profile (default: all tests)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel execution"
    )
    parser.add_argument(
        "--output", default="test_performance_report.txt", help="Output report file"
    )
    parser.add_argument(
        "--json",
        default="test_performance_results.json",
        help="Output JSON results file",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    profiler = TestPerformanceProfiler(project_root)

    # Profile tests
    results = profiler.profile_tests(
        pattern=args.pattern, parallel=not args.no_parallel
    )

    # Generate and display report
    report = profiler.generate_report(args.output)
    print(report)

    # Save detailed results
    profiler.save_results(args.json)

    print(f"\n🎉 Performance profiling complete!")
    print(f"   Report: {args.output}")
    print(f"   Data: {args.json}")


if __name__ == "__main__":
    main()
