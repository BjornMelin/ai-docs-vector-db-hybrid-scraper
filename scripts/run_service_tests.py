"""Comprehensive test runner for service layer testing.

Runs all service layer tests including:
- Function-based dependency injection tests
- Circuit breaker pattern tests  
- Database connection pooling tests
- Browser automation monitoring tests
- Service interaction integration tests
- Performance benchmarks
- Mutation testing

Provides detailed reporting and coverage analysis.
"""

import argparse
import asyncio  # noqa: PLC0415
import os  # noqa: PLC0415
import subprocess
import sys
import time  # noqa: PLC0415
from pathlib import Path
from typing import Any

import pytest


class ServiceTestRunner:
    """Comprehensive test runner for service layer."""
    
    def __init__(self, project_root: Path):
        """Initialize test runner.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.test_root = project_root / "tests"
        self.results = {}
        
    def run_dependency_injection_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run function-based dependency injection tests."""
        print("🔧 Running Dependency Injection Tests...")
        
        test_files = [
            "tests/unit/services/functional/test_dependencies_comprehensive.py",
            "tests/unit/services/test_function_based_dependencies.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short",
            "--strict-markers",
            f"--cov=src/services/functional",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['dependency_injection'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': 'Function-based dependency injection patterns'
        }
        
        return self.results['dependency_injection']
    
    def run_circuit_breaker_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run circuit breaker pattern tests."""
        print("⚡ Running Circuit Breaker Tests...")
        
        test_files = [
            "tests/unit/services/functional/test_circuit_breaker.py",
            "tests/unit/services/test_circuit_breakers.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v", 
            "--tb=short",
            "--strict-markers",
            "-m", "circuit_breaker or not circuit_breaker",  # Run all circuit breaker related tests
            f"--cov=src/services/functional/circuit_breaker",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['circuit_breaker'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': 'Circuit breaker resilience patterns'
        }
        
        return self.results['circuit_breaker']
    
    def run_database_pooling_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run database connection pooling and ML-based scaling tests."""
        print("🗄️ Running Database Connection Pooling Tests...")
        
        test_files = [
            "tests/unit/services/functional/test_database_connection_pooling.py",
            "tests/benchmarks/test_database_performance.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short", 
            "--strict-markers",
            "-m", "database_pooling or not database_pooling",
            f"--cov=src/infrastructure/database",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['database_pooling'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': 'Database connection pooling with ML-based scaling'
        }
        
        return self.results['database_pooling']
    
    def run_browser_monitoring_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run browser automation tier health monitoring tests."""
        print("🌐 Running Browser Automation Monitoring Tests...")
        
        test_files = [
            "tests/unit/services/functional/test_browser_automation_monitoring.py",
            "tests/unit/services/browser/test_monitoring.py",
            "tests/unit/services/browser/test_unified_manager.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short",
            "--strict-markers", 
            "-m", "browser_monitoring or not browser_monitoring",
            f"--cov=src/services/browser",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['browser_monitoring'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': '5-tier browser automation health monitoring'
        }
        
        return self.results['browser_monitoring']
    
    def run_service_integration_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run service interaction integration tests."""
        print("🔗 Running Service Integration Tests...")
        
        test_files = [
            "tests/integration/services/test_service_interactions.py",
            "tests/integration/test_mcp_tools_integration.py",
            "tests/integration/test_query_processing_integration.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short",
            "--strict-markers",
            "-m", "service_integration or integration",
            f"--cov=src/services",
            "--cov-append",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['service_integration'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': 'Service interaction and composition patterns'
        }
        
        return self.results['service_integration']
    
    def run_performance_benchmarks(self, verbose: bool = False) -> dict[str, Any]:
        """Run service performance benchmarks."""
        print("🚀 Running Performance Benchmarks...")
        
        test_files = [
            "tests/performance/services/test_service_performance_benchmarks.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short",
            "--strict-markers",
            "-m", "performance_benchmark",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-columns=min,max,mean,stddev,ops,rounds",
            "--benchmark-group-by=group",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['performance_benchmarks'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'test_files': test_files,
            'description': 'Service layer performance benchmarks'
        }
        
        return self.results['performance_benchmarks']
    
    def run_mutation_tests(self, verbose: bool = False) -> dict[str, Any]:
        """Run mutation testing for service logic validation."""
        print("🧬 Running Mutation Tests...")
        
        test_files = [
            "tests/mutation/test_service_mutation_testing.py",
        ]
        
        args = [
            "--verbose" if verbose else "-v",
            "--tb=short",
            "--strict-markers",
            "-m", "mutation_testing",
            f"--cov=src/services/functional",
            "--cov-append",
            "--cov-report=term-missing",
        ] + test_files
        
        result = pytest.main(args)
        
        self.results['mutation_testing'] = {
            'status': 'passed' if result == 0 else 'failed', 
            'exit_code': result,
            'test_files': test_files,
            'description': 'Mutation testing for service logic validation'
        }
        
        return self.results['mutation_testing']
    
    def run_coverage_analysis(self) -> dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("📊 Running Coverage Analysis...")
        
        # Run coverage on all service modules
        args = [
            "--cov=src/services",
            "--cov=src/infrastructure",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/services",
            "--cov-report=xml:coverage.xml", 
            "--cov-fail-under=90",  # Target 90%+ coverage
            "tests/unit/services/",
            "tests/integration/services/",
        ]
        
        result = pytest.main(args)
        
        coverage_result = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'target_coverage': 90,
            'description': 'Service layer coverage analysis'
        }
        
        self.results['coverage_analysis'] = coverage_result
        return coverage_result
    
    def run_all_tests(self, 
                     verbose: bool = False,
                     include_benchmarks: bool = True,
                     include_mutation: bool = False) -> dict[str, Any]:
        """Run all service layer tests.
        
        Args:
            verbose: Enable verbose output
            include_benchmarks: Include performance benchmarks
            include_mutation: Include mutation testing
            
        Returns:
            Dictionary with all test results
        """
        print("🧪 Running Comprehensive Service Layer Tests...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run test suites in order
        test_suites = [
            ('dependency_injection', self.run_dependency_injection_tests),
            ('circuit_breaker', self.run_circuit_breaker_tests),
            ('database_pooling', self.run_database_pooling_tests),
            ('browser_monitoring', self.run_browser_monitoring_tests),
            ('service_integration', self.run_service_integration_tests),
        ]
        
        if include_benchmarks:
            test_suites.append(('performance_benchmarks', self.run_performance_benchmarks))
        
        if include_mutation:
            test_suites.append(('mutation_testing', self.run_mutation_tests))
        
        # Execute test suites
        failed_suites = []
        for suite_name, test_function in test_suites:
            print(f"\n{'-' * 40}")
            try:
                result = test_function(verbose=verbose)
                if result['status'] == 'failed':
                    failed_suites.append(suite_name)
                    print(f"❌ {suite_name} tests FAILED")
                else:
                    print(f"✅ {suite_name} tests PASSED")
            except Exception as e:
                print(f"💥 {suite_name} tests CRASHED: {e}")
                failed_suites.append(suite_name)
                self.results[suite_name] = {
                    'status': 'crashed',
                    'error': str(e),
                    'description': f'{suite_name} test suite'
                }
        
        # Run coverage analysis
        print(f"\n{'-' * 40}")
        try:
            self.run_coverage_analysis()
        except Exception as e:
            print(f"💥 Coverage analysis CRASHED: {e}")
        
        execution_time = time.time() - start_time
        
        # Generate summary
        self.results['summary'] = {
            'total_suites': len(test_suites),
            'passed_suites': len(test_suites) - len(failed_suites),
            'failed_suites': failed_suites,
            'execution_time_seconds': execution_time,
            'overall_status': 'passed' if not failed_suites else 'failed'
        }
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("🎯 SERVICE LAYER TEST SUMMARY")
        print("=" * 60)
        
        if 'summary' not in self.results:
            print("❌ No summary available")
            return
        
        summary = self.results['summary']
        
        # Overall status
        status_emoji = "✅" if summary['overall_status'] == 'passed' else "❌"
        print(f"{status_emoji} Overall Status: {summary['overall_status'].upper()}")
        print(f"⏱️  Execution Time: {summary['execution_time_seconds']:.2f} seconds")
        print(f"📈 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for suite_name, result in self.results.items():
            if suite_name == 'summary':
                continue
                
            status = result.get('status', 'unknown')
            description = result.get('description', 'No description')
            
            if status == 'passed':
                print(f"  ✅ {suite_name}: {description}")
            elif status == 'failed':
                print(f"  ❌ {suite_name}: {description}")
            elif status == 'crashed':
                print(f"  💥 {suite_name}: {description} - {result.get('error', 'Unknown error')}")
            else:
                print(f"  ❓ {suite_name}: {description} - Status: {status}")
        
        # Failed suites details
        if summary['failed_suites']:
            print(f"\n🚨 Failed Test Suites:")
            for suite in summary['failed_suites']:
                result = self.results.get(suite, {})
                print(f"  - {suite}: {result.get('description', 'No description')}")
        
        # Coverage information
        if 'coverage_analysis' in self.results:
            coverage_result = self.results['coverage_analysis']
            coverage_status = "✅" if coverage_result['status'] == 'passed' else "❌"
            print(f"\n{coverage_status} Coverage Target: {coverage_result['target_coverage']}%")
        
        # Next steps
        print(f"\n🔄 Next Steps:")
        if summary['overall_status'] == 'passed':
            print("  - All service layer tests are passing!")
            print("  - Consider running mutation testing for additional validation")
            print("  - Review performance benchmarks for optimization opportunities")
        else:
            print("  - Fix failing test suites before proceeding")
            print("  - Check test logs for detailed error information")
            print("  - Ensure all dependencies are properly mocked")
            print("  - Verify service logic implementation")
    
    def generate_report(self, output_file: Path = None) -> Path:
        """Generate detailed test report.
        
        Args:
            output_file: Path to output file (default: test-report.md)
            
        Returns:
            Path to generated report file
        """
        if output_file is None:
            output_file = self.project_root / "test-report-services.md"
        
        report_content = self._generate_markdown_report()
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Test report generated: {output_file}")
        return output_file
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown test report."""
        if 'summary' not in self.results:
            return "# Service Layer Test Report\n\nNo test results available.\n"
        
        summary = self.results['summary']
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Service Layer Test Report
        
Generated: {timestamp}

## Executive Summary

- **Overall Status**: {summary['overall_status'].upper()}
- **Test Suites**: {summary['passed_suites']}/{summary['total_suites']} passed
- **Execution Time**: {summary['execution_time_seconds']:.2f} seconds

## Test Suite Results

"""
        
        for suite_name, result in self.results.items():
            if suite_name == 'summary':
                continue
            
            status = result.get('status', 'unknown')
            description = result.get('description', 'No description')
            test_files = result.get('test_files', [])
            
            status_emoji = {
                'passed': '✅',
                'failed': '❌', 
                'crashed': '💥'
            }.get(status, '❓')
            
            report += f"### {status_emoji} {suite_name.replace('_', ' ').title()}\n\n"
            report += f"**Status**: {status}\n"
            report += f"**Description**: {description}\n"
            
            if test_files:
                report += f"**Test Files**:\n"
                for test_file in test_files:
                    report += f"- `{test_file}`\n"
            
            if status == 'crashed' and 'error' in result:
                report += f"**Error**: {result['error']}\n"
            
            report += "\n"
        
        if summary['failed_suites']:
            report += "## Failed Test Suites\n\n"
            for suite in summary['failed_suites']:
                result = self.results.get(suite, {})
                report += f"- **{suite}**: {result.get('description', 'No description')}\n"
            report += "\n"
        
        report += "## Recommendations\n\n"
        if summary['overall_status'] == 'passed':
            report += "- All service layer tests are passing\n"
            report += "- Consider implementing mutation testing for additional validation\n"
            report += "- Review performance benchmarks for optimization opportunities\n"
        else:
            report += "- Fix failing test suites before proceeding to production\n"
            report += "- Implement missing test coverage for service logic\n"
            report += "- Verify circuit breaker and dependency injection patterns\n"
        
        return report


def main():
    """Main entry point for service test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive service layer test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_service_tests.py --all                    # Run all tests
  python scripts/run_service_tests.py --dependency-injection  # Run DI tests only
  python scripts/run_service_tests.py --circuit-breaker       # Run circuit breaker tests
  python scripts/run_service_tests.py --database-pooling      # Run DB pooling tests
  python scripts/run_service_tests.py --browser-monitoring    # Run browser monitoring tests
  python scripts/run_service_tests.py --integration           # Run integration tests
  python scripts/run_service_tests.py --performance           # Run performance benchmarks
  python scripts/run_service_tests.py --mutation              # Run mutation tests
  python scripts/run_service_tests.py --coverage              # Run coverage analysis
  python scripts/run_service_tests.py --all --verbose         # Run all with verbose output
  python scripts/run_service_tests.py --all --no-benchmarks   # Run all except benchmarks
        """
    )
    
    # Test suite selection
    parser.add_argument('--all', action='store_true', help='Run all test suites')
    parser.add_argument('--dependency-injection', action='store_true', help='Run dependency injection tests')
    parser.add_argument('--circuit-breaker', action='store_true', help='Run circuit breaker tests')
    parser.add_argument('--database-pooling', action='store_true', help='Run database pooling tests')
    parser.add_argument('--browser-monitoring', action='store_true', help='Run browser monitoring tests')
    parser.add_argument('--integration', action='store_true', help='Run service integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--mutation', action='store_true', help='Run mutation tests')
    parser.add_argument('--coverage', action='store_true', help='Run coverage analysis only')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-benchmarks', action='store_true', help='Exclude performance benchmarks')
    parser.add_argument('--no-mutation', action='store_true', help='Exclude mutation testing')
    parser.add_argument('--report', type=str, help='Generate report to specified file')
    
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Initialize test runner
    runner = ServiceTestRunner(project_root)
    
    # Determine what to run
    if args.all:
        include_benchmarks = not args.no_benchmarks
        include_mutation = not args.no_mutation
        results = runner.run_all_tests(
            verbose=args.verbose,
            include_benchmarks=include_benchmarks,
            include_mutation=include_mutation
        )
    elif args.coverage:
        results = {'coverage_analysis': runner.run_coverage_analysis()}
    else:
        # Run specific test suites
        results = {}
        
        if args.dependency_injection:
            results['dependency_injection'] = runner.run_dependency_injection_tests(args.verbose)
        
        if args.circuit_breaker:
            results['circuit_breaker'] = runner.run_circuit_breaker_tests(args.verbose)
        
        if args.database_pooling:
            results['database_pooling'] = runner.run_database_pooling_tests(args.verbose)
        
        if args.browser_monitoring:
            results['browser_monitoring'] = runner.run_browser_monitoring_tests(args.verbose)
        
        if args.integration:
            results['service_integration'] = runner.run_service_integration_tests(args.verbose)
        
        if args.performance:
            results['performance_benchmarks'] = runner.run_performance_benchmarks(args.verbose)
        
        if args.mutation:
            results['mutation_testing'] = runner.run_mutation_tests(args.verbose)
        
        if not results:
            print("❌ No test suites specified. Use --all or specify individual suites.")
            parser.print_help()
            return 1
        
        # Add summary for partial runs
        failed_suites = [name for name, result in results.items() if result['status'] != 'passed']
        runner.results = results
        runner.results['summary'] = {
            'total_suites': len(results),
            'passed_suites': len(results) - len(failed_suites),
            'failed_suites': failed_suites,
            'execution_time_seconds': 0,  # Not tracked for partial runs
            'overall_status': 'passed' if not failed_suites else 'failed'
        }
        runner.print_summary()
    
    # Generate report if requested
    if args.report:
        runner.generate_report(Path(args.report))
    
    # Exit with appropriate code
    if 'summary' in runner.results:
        return 0 if runner.results['summary']['overall_status'] == 'passed' else 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())