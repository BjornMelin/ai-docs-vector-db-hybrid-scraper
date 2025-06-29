#!/usr/bin/env python3
"""Comprehensive Test Runner for AI Documentation System.

This script provides a unified test execution framework that demonstrates modern
AI/ML testing practices with categorized test execution, performance validation,
and comprehensive reporting.

Features:
- Test categorization by markers (fast, integration, e2e, ai, performance, security)
- Performance benchmarking with P95 latency validation
- Security testing with vulnerability scanning
- Property-based testing with Hypothesis
- Comprehensive coverage reporting
- Parallel test execution with optimization
- CI/CD integration support
- Test result analysis and recommendations

Usage:
    python scripts/run_comprehensive_tests.py --category fast
    python scripts/run_comprehensive_tests.py --category integration --parallel
    python scripts/run_comprehensive_tests.py --category all --coverage --benchmark
    python scripts/run_comprehensive_tests.py --security-only
    python scripts/run_comprehensive_tests.py --performance-only --p95-threshold 100
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


@dataclass
class TestConfig:
    """Configuration for test execution."""
    
    # Test categorization
    categories: List[str] = field(default_factory=lambda: ["fast"])
    markers: List[str] = field(default_factory=list)
    
    # Execution options
    parallel: bool = False
    max_workers: int = 4
    fail_fast: bool = False
    verbose: bool = True
    
    # Coverage options
    coverage: bool = False
    coverage_threshold: int = 80
    coverage_format: str = "html"
    
    # Performance options
    benchmark: bool = False
    performance_threshold_ms: float = 100.0
    memory_threshold_mb: float = 512.0
    
    # Security options
    security: bool = False
    vulnerability_scan: bool = False
    
    # Output options
    output_dir: str = "test_results"
    junit_xml: bool = False
    json_report: bool = True
    
    # AI/ML specific options
    hypothesis_profile: str = "default"
    property_based_tests: bool = True
    ai_model_validation: bool = True


@dataclass
class TestResults:
    """Container for test execution results."""
    
    category: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    coverage_percentage: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exit_code: int = 0


class ComprehensiveTestRunner:
    """Comprehensive test runner with modern AI/ML testing capabilities."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test category configurations
        self.test_categories = {
            "fast": {
                "markers": ["not integration and not e2e and not performance and not security"],
                "timeout": 120,
                "description": "Fast unit tests (<2s each)",
                "max_duration": 120,
            },
            "integration": {
                "markers": ["integration and not e2e"],
                "timeout": 300,
                "description": "Integration tests with service mocking",
                "max_duration": 300,
            },
            "e2e": {
                "markers": ["e2e"],
                "timeout": 600,
                "description": "End-to-end workflow tests",
                "max_duration": 600,
            },
            "ai": {
                "markers": ["ai or embedding or vector_db or rag"],
                "timeout": 300,
                "description": "AI/ML component tests",
                "max_duration": 300,
            },
            "property": {
                "markers": ["property"],
                "timeout": 180,
                "description": "Property-based tests with Hypothesis",
                "max_duration": 180,
            },
            "performance": {
                "markers": ["performance"],
                "timeout": 600,
                "description": "Performance and benchmark tests",
                "max_duration": 600,
            },
            "security": {
                "markers": ["security"],
                "timeout": 300,
                "description": "Security and vulnerability tests",
                "max_duration": 300,
            },
            "all": {
                "markers": [],
                "timeout": 1200,
                "description": "All test categories",
                "max_duration": 1200,
            },
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, TestResults]:
        """Run comprehensive test suite with all specified categories."""
        print("🧪 Starting Comprehensive AI/ML Testing Framework")
        print("=" * 60)
        
        results = {}
        
        for category in self.config.categories:
            if category not in self.test_categories:
                print(f"❌ Unknown test category: {category}")
                continue
            
            print(f"\n📋 Running {category} tests: {self.test_categories[category]['description']}")
            
            try:
                result = await self.run_test_category(category)
                results[category] = result
                
                # Print category summary
                self._print_category_summary(category, result)
                
                # Fail fast if enabled and tests failed
                if self.config.fail_fast and result.failed > 0:
                    print(f"\n🛑 Stopping execution due to failures in {category} tests")
                    break
                    
            except Exception as e:
                print(f"❌ Error running {category} tests: {e}")
                results[category] = TestResults(
                    category=category,
                    errors=1,
                    exit_code=1
                )
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(results)
        
        return results
    
    async def run_test_category(self, category: str) -> TestResults:
        """Run tests for a specific category."""
        start_time = time.time()
        category_config = self.test_categories[category]
        
        # Build pytest command
        cmd = self._build_pytest_command(category, category_config)
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        # Execute tests
        if self.config.parallel and category != "e2e":
            result = await self._run_parallel_tests(cmd, category_config["timeout"])
        else:
            result = await self._run_sequential_tests(cmd, category_config["timeout"])
        
        duration = time.time() - start_time
        
        # Parse results
        test_result = self._parse_test_results(result, category, duration)
        
        # Run additional analysis for specific categories
        if category == "performance":
            await self._analyze_performance_results(test_result)
        elif category == "security":
            await self._analyze_security_results(test_result)
        elif category == "ai":
            await self._analyze_ai_results(test_result)
        
        return test_result
    
    def _build_pytest_command(self, category: str, category_config: Dict[str, Any]) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ["uv", "run", "pytest"]
        
        # Add markers for category
        if category_config["markers"]:
            for marker in category_config["markers"]:
                cmd.extend(["-m", marker])
        
        # Add verbosity
        if self.config.verbose:
            cmd.append("-v")
        
        # Add coverage
        if self.config.coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                f"--cov-report=html:htmlcov_{category}",
                f"--cov-fail-under={self.config.coverage_threshold}"
            ])
        
        # Add junit XML output
        if self.config.junit_xml:
            cmd.extend(["--junit-xml", f"{self.output_dir}/junit_{category}.xml"])
        
        # Add JSON report
        if self.config.json_report:
            cmd.extend(["--json-report", f"--json-report-file={self.output_dir}/report_{category}.json"])
        
        # Add performance options
        if category == "performance" or self.config.benchmark:
            cmd.extend([
                "--benchmark-only",
                "--benchmark-json", f"{self.output_dir}/benchmark_{category}.json",
                "--benchmark-sort=mean"
            ])
        
        # Add parallel execution for fast tests
        if self.config.parallel and category == "fast":
            cmd.extend(["-n", str(self.config.max_workers)])
        
        # Add timeout
        cmd.extend(["--timeout", str(category_config["timeout"])])
        
        # Add fail fast
        if self.config.fail_fast:
            cmd.extend(["-x"])
        
        # Add specific test directories based on category
        if category == "security":
            cmd.append("tests/security/")
        elif category == "performance":
            cmd.append("tests/performance/")
        elif category == "integration":
            cmd.append("tests/integration/")
        elif category == "ai":
            cmd.extend(["tests/unit/ai/", "tests/integration/ai/"])
        elif category == "property":
            cmd.append("tests/unit/ai/test_embedding_properties.py")
        elif category != "all":
            cmd.append("tests/")
        else:
            cmd.append("tests/")
        
        # Add Hypothesis profile for property-based tests
        if category == "property" or self.config.property_based_tests:
            cmd.extend(["--hypothesis-profile", self.config.hypothesis_profile])
        
        return cmd
    
    async def _run_sequential_tests(self, cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Run tests sequentially."""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.project_root
        )
    
    async def _run_parallel_tests(self, cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Run tests in parallel using thread pool."""
        loop = asyncio.get_event_loop()
        
        def run_cmd():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(executor, run_cmd)
            return await future
    
    def _parse_test_results(self, result: subprocess.CompletedProcess, category: str, duration: float) -> TestResults:
        """Parse pytest results and extract metrics."""
        test_result = TestResults(category=category, duration=duration, exit_code=result.returncode)
        
        # Parse pytest output for test counts
        output_lines = result.stdout.split('\n')
        
        for line in output_lines:
            # Look for summary line like "5 passed, 1 failed, 2 skipped in 10.5s"
            if " passed" in line or " failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            status = parts[i + 1]
                            if status.startswith("passed"):
                                test_result.passed = count
                            elif status.startswith("failed"):
                                test_result.failed = count
                            elif status.startswith("skipped"):
                                test_result.skipped = count
                            elif status.startswith("error"):
                                test_result.errors = count
        
        test_result.total_tests = test_result.passed + test_result.failed + test_result.skipped + test_result.errors
        
        # Parse coverage information
        if self.config.coverage:
            for line in output_lines:
                if "TOTAL" in line and "%" in line:
                    # Extract coverage percentage
                    parts = line.split()
                    for part in parts:
                        if part.endswith("%"):
                            try:
                                test_result.coverage_percentage = float(part.rstrip("%"))
                                break
                            except ValueError:
                                pass
        
        # Parse errors and warnings
        if result.returncode != 0:
            error_lines = result.stderr.split('\n')
            test_result.warnings.extend([line for line in error_lines if line.strip()])
        
        return test_result
    
    async def _analyze_performance_results(self, test_result: TestResults):
        """Analyze performance test results."""
        benchmark_file = self.output_dir / f"benchmark_{test_result.category}.json"
        
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                benchmark_data = json.load(f)
            
            # Extract performance metrics
            benchmarks = benchmark_data.get("benchmarks", [])
            performance_metrics = {
                "total_benchmarks": len(benchmarks),
                "average_time": 0.0,
                "p95_time": 0.0,
                "p99_time": 0.0,
                "slowest_test": "",
                "threshold_violations": 0,
            }
            
            if benchmarks:
                times = [b["stats"]["mean"] for b in benchmarks]
                performance_metrics["average_time"] = sum(times) / len(times)
                
                # Calculate percentiles
                sorted_times = sorted(times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                
                performance_metrics["p95_time"] = sorted_times[p95_index] if sorted_times else 0.0
                performance_metrics["p99_time"] = sorted_times[p99_index] if sorted_times else 0.0
                
                # Find slowest test
                slowest = max(benchmarks, key=lambda b: b["stats"]["mean"])
                performance_metrics["slowest_test"] = slowest["name"]
                
                # Check threshold violations
                threshold_ms = self.config.performance_threshold_ms / 1000  # Convert to seconds
                violations = [b for b in benchmarks if b["stats"]["mean"] > threshold_ms]
                performance_metrics["threshold_violations"] = len(violations)
                
                if violations:
                    test_result.warnings.append(
                        f"Performance threshold exceeded in {len(violations)} tests"
                    )
            
            test_result.performance_metrics = performance_metrics
    
    async def _analyze_security_results(self, test_result: TestResults):
        """Analyze security test results."""
        # Look for security findings in test output
        security_findings = []
        
        # Mock security analysis - in real implementation, this would parse
        # actual security scanner output
        if test_result.failed == 0 and test_result.errors == 0:
            security_findings.append({
                "type": "success",
                "message": "All security tests passed",
                "severity": "info"
            })
        else:
            security_findings.append({
                "type": "vulnerability",
                "message": f"{test_result.failed} security tests failed",
                "severity": "high"
            })
        
        test_result.security_findings = security_findings
    
    async def _analyze_ai_results(self, test_result: TestResults):
        """Analyze AI/ML specific test results."""
        # Add AI-specific performance metrics
        ai_metrics = {
            "embedding_validation_passed": test_result.failed == 0,
            "vector_similarity_tests": test_result.passed,
            "property_based_coverage": "high" if test_result.passed > 5 else "low",
        }
        
        test_result.performance_metrics.update(ai_metrics)
    
    def _print_category_summary(self, category: str, result: TestResults):
        """Print summary for a test category."""
        status_icon = "✅" if result.exit_code == 0 else "❌"
        
        print(f"\n{status_icon} {category.upper()} TESTS SUMMARY")
        print(f"   Total: {result.total_tests}")
        print(f"   Passed: {result.passed}")
        print(f"   Failed: {result.failed}")
        print(f"   Skipped: {result.skipped}")
        print(f"   Duration: {result.duration:.2f}s")
        
        if self.config.coverage and result.coverage_percentage > 0:
            coverage_icon = "✅" if result.coverage_percentage >= self.config.coverage_threshold else "⚠️"
            print(f"   Coverage: {coverage_icon} {result.coverage_percentage:.1f}%")
        
        if result.performance_metrics:
            if "p95_time" in result.performance_metrics:
                p95_ms = result.performance_metrics["p95_time"] * 1000
                threshold_icon = "✅" if p95_ms <= self.config.performance_threshold_ms else "⚠️"
                print(f"   P95 Latency: {threshold_icon} {p95_ms:.1f}ms")
        
        if result.warnings:
            print(f"   Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                print(f"     • {warning}")
        
        if result.security_findings:
            high_severity = [f for f in result.security_findings if f.get("severity") == "high"]
            if high_severity:
                print(f"   Security: ⚠️ {len(high_severity)} high-severity findings")
    
    async def _generate_comprehensive_report(self, results: Dict[str, TestResults]):
        """Generate comprehensive test report."""
        report_data = {
            "timestamp": time.time(),
            "config": {
                "categories": self.config.categories,
                "coverage_threshold": self.config.coverage_threshold,
                "performance_threshold_ms": self.config.performance_threshold_ms,
                "parallel": self.config.parallel,
            },
            "summary": {},
            "categories": {},
            "recommendations": [],
        }
        
        # Calculate overall summary
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_duration = sum(r.duration for r in results.values())
        
        report_data["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "exit_code": max((r.exit_code for r in results.values()), default=0),
        }
        
        # Add category details
        for category, result in results.items():
            report_data["categories"][category] = {
                "total_tests": result.total_tests,
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "duration": result.duration,
                "coverage_percentage": result.coverage_percentage,
                "performance_metrics": result.performance_metrics,
                "security_findings": result.security_findings,
                "warnings": result.warnings,
                "exit_code": result.exit_code,
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        report_data["recommendations"] = recommendations
        
        # Save report
        report_file = self.output_dir / "comprehensive_test_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Print final summary
        self._print_final_summary(report_data)
        
        print(f"\n📊 Comprehensive report saved to: {report_file}")
    
    def _generate_recommendations(self, results: Dict[str, TestResults]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Coverage recommendations
        low_coverage_categories = [
            category for category, result in results.items()
            if result.coverage_percentage > 0 and result.coverage_percentage < self.config.coverage_threshold
        ]
        
        if low_coverage_categories:
            recommendations.append(
                f"Improve test coverage in: {', '.join(low_coverage_categories)}"
            )
        
        # Performance recommendations
        for category, result in results.items():
            if result.performance_metrics.get("threshold_violations", 0) > 0:
                recommendations.append(
                    f"Optimize performance in {category} tests - {result.performance_metrics['threshold_violations']} tests exceed threshold"
                )
        
        # Security recommendations
        security_issues = sum(
            len([f for f in result.security_findings if f.get("severity") == "high"])
            for result in results.values()
        )
        
        if security_issues > 0:
            recommendations.append(
                f"Address {security_issues} high-severity security findings"
            )
        
        # Test stability recommendations
        failed_categories = [category for category, result in results.items() if result.failed > 0]
        if failed_categories:
            recommendations.append(
                f"Investigate test failures in: {', '.join(failed_categories)}"
            )
        
        return recommendations
    
    def _print_final_summary(self, report_data: Dict[str, Any]):
        """Print final comprehensive summary."""
        summary = report_data["summary"]
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 60)
        
        # Overall status
        status_icon = "✅" if summary["exit_code"] == 0 else "❌"
        print(f"\n{status_icon} Overall Status: {'PASSED' if summary['exit_code'] == 0 else 'FAILED'}")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['total_passed']}")
        print(f"❌ Failed: {summary['total_failed']}")
        print(f"📈 Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.1f}s")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, data in report_data["categories"].items():
            status = "✅" if data["exit_code"] == 0 else "❌"
            print(f"   {status} {category}: {data['passed']}/{data['total_tests']} passed ({data['duration']:.1f}s)")
        
        # Recommendations
        if report_data["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report_data["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "=" * 60)


def create_test_config_from_args(args: argparse.Namespace) -> TestConfig:
    """Create test configuration from command line arguments."""
    config = TestConfig()
    
    # Categories
    if args.category:
        if args.category == "all":
            config.categories = ["fast", "integration", "ai", "performance", "security"]
        else:
            config.categories = [args.category]
    
    if args.security_only:
        config.categories = ["security"]
        config.security = True
        config.vulnerability_scan = True
    
    if args.performance_only:
        config.categories = ["performance"]
        config.benchmark = True
    
    if args.ai_only:
        config.categories = ["ai", "property"]
        config.property_based_tests = True
        config.ai_model_validation = True
    
    # Execution options
    config.parallel = args.parallel
    config.fail_fast = args.fail_fast
    config.verbose = args.verbose
    config.max_workers = args.max_workers
    
    # Coverage options
    config.coverage = args.coverage
    config.coverage_threshold = args.coverage_threshold
    
    # Performance options
    config.benchmark = args.benchmark or args.performance_only
    config.performance_threshold_ms = args.p95_threshold
    
    # Output options
    config.junit_xml = args.junit_xml
    config.json_report = True  # Always enabled
    config.output_dir = args.output_dir
    
    return config


async def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive AI/ML Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --category fast                    # Run fast unit tests only
  %(prog)s --category integration --parallel  # Run integration tests in parallel
  %(prog)s --category all --coverage          # Run all tests with coverage
  %(prog)s --security-only                    # Run security tests only
  %(prog)s --performance-only --p95-threshold 50  # Performance tests with 50ms threshold
  %(prog)s --ai-only                          # Run AI/ML specific tests only
        """
    )
    
    # Test category selection
    parser.add_argument(
        "--category", 
        choices=["fast", "integration", "e2e", "ai", "property", "performance", "security", "all"],
        default="fast",
        help="Test category to run (default: fast)"
    )
    
    parser.add_argument("--security-only", action="store_true", help="Run security tests only")
    parser.add_argument("--performance-only", action="store_true", help="Run performance tests only")
    parser.add_argument("--ai-only", action="store_true", help="Run AI/ML tests only")
    
    # Execution options
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    # Coverage options
    parser.add_argument("--coverage", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--coverage-threshold", type=int, default=80, help="Coverage threshold percentage")
    
    # Performance options
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--p95-threshold", type=float, default=100.0, help="P95 latency threshold in ms")
    
    # Output options
    parser.add_argument("--junit-xml", action="store_true", help="Generate JUnit XML reports")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = create_test_config_from_args(args)
    
    # Create and run test runner
    runner = ComprehensiveTestRunner(config)
    
    try:
        results = await runner.run_comprehensive_tests()
        
        # Determine exit code
        exit_code = max((result.exit_code for result in results.values()), default=0)
        
        if exit_code == 0:
            print("\n🎉 All tests completed successfully!")
        else:
            print(f"\n💥 Tests completed with failures (exit code: {exit_code})")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())