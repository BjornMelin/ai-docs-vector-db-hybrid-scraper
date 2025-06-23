#!/usr/bin/env python3
"""Comprehensive load test runner for AI Documentation Vector DB.

This script provides a command-line interface for running various types of load tests
with different configurations, profiles, and reporting options.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from locust import main as locust_main
from locust.env import Environment
from locust.log import setup_logging

from tests.load.base_load_test import create_load_test_runner
from tests.load.locust_load_runner import (
    VectorDBUser, AdminUser, create_load_test_environment,
    metrics_collector, save_load_test_report
)
from tests.load.load_profiles import LOAD_PROFILES, get_load_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Main load test runner with multiple execution modes."""
    
    def __init__(self):
        self.results_dir = Path("load_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "light": {
                "users": 10,
                "spawn_rate": 2,
                "duration": 300,  # 5 minutes
                "host": "http://localhost:8000",
            },
            "moderate": {
                "users": 50,
                "spawn_rate": 5,
                "duration": 600,  # 10 minutes
                "host": "http://localhost:8000",
            },
            "heavy": {
                "users": 200,
                "spawn_rate": 10,
                "duration": 900,  # 15 minutes
                "host": "http://localhost:8000",
            },
            "stress": {
                "users": 500,
                "spawn_rate": 25,
                "duration": 600,  # 10 minutes
                "host": "http://localhost:8000",
            },
        }
    
    def run_locust_test(self, config: Dict, profile: Optional[str] = None, 
                       headless: bool = True, web_port: int = 8089) -> Dict:
        """Run load test using Locust."""
        logger.info(f"Starting Locust load test with config: {config}")
        
        # Create environment
        env = create_load_test_environment(
            host=config["host"],
            user_classes=[VectorDBUser, AdminUser]
        )
        
        # Apply load profile if specified
        if profile and profile in LOAD_PROFILES:
            load_profile = get_load_profile(profile)
            if load_profile:
                env.shape_class = load_profile
                logger.info(f"Applied load profile: {profile}")
        
        if headless:
            # Run headless test
            runner = env.create_local_runner()
            
            # Start test
            runner.start(config["users"], spawn_rate=config["spawn_rate"])
            
            # Run for specified duration
            start_time = time.time()
            duration = config["duration"]
            
            try:
                while time.time() - start_time < duration:
                    time.sleep(1)
                    
                    # Check if test should stop early
                    if runner.state == "stopped":
                        break
            
            except KeyboardInterrupt:
                logger.info("Test interrupted by user")
            
            finally:
                # Stop test
                runner.stop()
                
                # Generate report
                report = self._generate_test_report(env, config, profile)
                
                # Save report
                report_file = self._save_report(report)
                logger.info(f"Test report saved to: {report_file}")
                
                return report
        
        else:
            # Run with web UI
            logger.info(f"Starting Locust web UI on port {web_port}")
            logger.info(f"Visit http://localhost:{web_port} to control the test")
            
            # Set up Locust arguments for web mode
            locust_args = [
                "--web-port", str(web_port),
                "--host", config["host"],
                "--users", str(config["users"]),
                "--spawn-rate", str(config["spawn_rate"]),
            ]
            
            if config.get("duration"):
                locust_args.extend(["--run-time", f"{config['duration']}s"])
            
            # Run Locust main
            os.environ["LOCUST_USER_CLASSES"] = "VectorDBUser,AdminUser"
            sys.argv = ["locust"] + locust_args
            
            try:
                locust_main.main()
            except SystemExit:
                pass
            
            return {"status": "completed_web_mode"}
    
    def run_pytest_load_tests(self, test_type: str = "all", markers: List[str] = None) -> Dict:
        """Run load tests using pytest."""
        logger.info(f"Running pytest load tests: {test_type}")
        
        import subprocess
        
        # Build pytest command
        cmd = ["uv", "run", "pytest", "tests/load/", "-v"]
        
        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        elif test_type != "all":
            cmd.extend(["-m", test_type])
        
        # Add specific test directories based on type
        if test_type == "load":
            cmd = ["uv", "run", "pytest", "tests/load/load_testing/", "-v"]
        elif test_type == "stress":
            cmd = ["uv", "run", "pytest", "tests/load/stress_testing/", "-v"]
        elif test_type == "spike":
            cmd = ["uv", "run", "pytest", "tests/load/spike_testing/", "-v"]
        elif test_type == "endurance":
            cmd = ["uv", "run", "pytest", "tests/load/endurance_testing/", "-v"]
        elif test_type == "volume":
            cmd = ["uv", "run", "pytest", "tests/load/volume_testing/", "-v"]
        elif test_type == "scalability":
            cmd = ["uv", "run", "pytest", "tests/load/scalability/", "-v"]
        
        # Add output options
        cmd.extend(["--tb=short", "--disable-warnings"])
        
        # Run tests
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }
        
        except Exception as e:
            logger.error(f"Failed to run pytest tests: {e}")
            return {
                "status": "error",
                "error": str(e),
                "command": " ".join(cmd),
            }
    
    def run_custom_scenario(self, scenario_file: str) -> Dict:
        """Run a custom load test scenario from JSON file."""
        logger.info(f"Running custom scenario: {scenario_file}")
        
        try:
            with open(scenario_file, 'r') as f:
                scenario = json.load(f)
            
            # Validate scenario format
            required_fields = ["name", "description", "config"]
            for field in required_fields:
                if field not in scenario:
                    raise ValueError(f"Missing required field: {field}")
            
            config = scenario["config"]
            profile = scenario.get("profile")
            
            # Run the test
            result = self.run_locust_test(config, profile, headless=True)
            
            # Add scenario metadata to result
            result["scenario"] = {
                "name": scenario["name"],
                "description": scenario["description"],
                "file": scenario_file,
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to run custom scenario: {e}")
            return {
                "status": "error",
                "error": str(e),
                "scenario_file": scenario_file,
            }
    
    def benchmark_endpoints(self, endpoints: List[str], config: Dict) -> Dict:
        """Benchmark specific endpoints."""
        logger.info(f"Benchmarking endpoints: {endpoints}")
        
        results = {}
        
        for endpoint in endpoints:
            logger.info(f"Benchmarking endpoint: {endpoint}")
            
            # Create custom user class for this endpoint
            endpoint_config = config.copy()
            endpoint_config["target_endpoint"] = endpoint
            
            # Run focused test on this endpoint
            result = self._run_endpoint_benchmark(endpoint, endpoint_config)
            results[endpoint] = result
        
        # Generate comparative report
        comparative_report = self._generate_endpoint_comparison(results)
        
        # Save report
        report_file = self._save_report(comparative_report, "endpoint_benchmark")
        logger.info(f"Endpoint benchmark report saved to: {report_file}")
        
        return comparative_report
    
    def validate_performance_regression(self, baseline_file: str, current_config: Dict) -> Dict:
        """Validate performance against a baseline."""
        logger.info(f"Running performance regression test against baseline: {baseline_file}")
        
        try:
            # Load baseline results
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            # Run current test
            current_result = self.run_locust_test(current_config, headless=True)
            
            # Compare results
            regression_analysis = self._analyze_performance_regression(baseline, current_result)
            
            # Save regression report
            report_file = self._save_report(regression_analysis, "regression_analysis")
            logger.info(f"Regression analysis saved to: {report_file}")
            
            return regression_analysis
        
        except Exception as e:
            logger.error(f"Failed to run regression test: {e}")
            return {
                "status": "error",
                "error": str(e),
                "baseline_file": baseline_file,
            }
    
    def _generate_test_report(self, env: Environment, config: Dict, profile: Optional[str]) -> Dict:
        """Generate comprehensive test report."""
        stats = env.stats
        
        if not stats or stats.total.num_requests == 0:
            return {
                "status": "no_data",
                "message": "No requests were made during the test",
            }
        
        # Calculate key metrics
        total_requests = stats.total.num_requests
        total_failures = stats.total.num_failures
        success_rate = ((total_requests - total_failures) / total_requests) * 100 if total_requests > 0 else 0
        
        # Get percentile response times
        response_times = []
        for entry in stats.entries.values():
            response_times.extend(entry.response_times)
        
        response_times.sort()
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": self._percentile(response_times, 50),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99),
            }
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "config": config,
            "profile": profile,
            "summary": {
                "total_requests": total_requests,
                "total_failures": total_failures,
                "success_rate_percent": success_rate,
                "avg_response_time_ms": stats.total.avg_response_time,
                "min_response_time_ms": stats.total.min_response_time,
                "max_response_time_ms": stats.total.max_response_time,
                "requests_per_second": stats.total.current_rps,
                "percentiles": percentiles,
            },
            "endpoint_breakdown": {},
            "performance_grade": self._calculate_performance_grade(stats),
            "recommendations": self._generate_recommendations(stats),
        }
        
        # Add endpoint breakdown
        for name, entry in stats.entries.items():
            if entry.num_requests > 0:
                report["endpoint_breakdown"][name] = {
                    "requests": entry.num_requests,
                    "failures": entry.num_failures,
                    "avg_response_time": entry.avg_response_time,
                    "min_response_time": entry.min_response_time,
                    "max_response_time": entry.max_response_time,
                    "requests_per_second": entry.current_rps,
                    "success_rate_percent": ((entry.num_requests - entry.num_failures) / entry.num_requests) * 100,
                }
        
        return report
    
    def _run_endpoint_benchmark(self, endpoint: str, config: Dict) -> Dict:
        """Run benchmark for a specific endpoint."""
        # This would create a custom user class focused on the specific endpoint
        # For now, return a simplified result
        return {
            "endpoint": endpoint,
            "avg_response_time_ms": 150,  # Placeholder
            "requests_per_second": 25,    # Placeholder
            "success_rate_percent": 99.5, # Placeholder
        }
    
    def _generate_endpoint_comparison(self, results: Dict) -> Dict:
        """Generate comparative analysis of endpoint performance."""
        comparison = {
            "timestamp": time.time(),
            "endpoints": results,
            "analysis": {
                "fastest_endpoint": min(results.keys(), key=lambda k: results[k]["avg_response_time_ms"]),
                "slowest_endpoint": max(results.keys(), key=lambda k: results[k]["avg_response_time_ms"]),
                "highest_throughput": max(results.keys(), key=lambda k: results[k]["requests_per_second"]),
                "most_reliable": max(results.keys(), key=lambda k: results[k]["success_rate_percent"]),
            },
        }
        
        return comparison
    
    def _analyze_performance_regression(self, baseline: Dict, current: Dict) -> Dict:
        """Analyze performance regression between baseline and current results."""
        analysis = {
            "timestamp": time.time(),
            "baseline": baseline,
            "current": current,
            "regression_detected": False,
            "improvements": [],
            "regressions": [],
            "overall_assessment": "pass",
        }
        
        # Compare key metrics
        baseline_summary = baseline.get("summary", {})
        current_summary = current.get("summary", {})
        
        # Response time comparison
        baseline_rt = baseline_summary.get("avg_response_time_ms", 0)
        current_rt = current_summary.get("avg_response_time_ms", 0)
        
        if current_rt > baseline_rt * 1.2:  # 20% degradation threshold
            analysis["regressions"].append({
                "metric": "avg_response_time_ms",
                "baseline": baseline_rt,
                "current": current_rt,
                "degradation_percent": ((current_rt - baseline_rt) / baseline_rt) * 100,
            })
            analysis["regression_detected"] = True
        
        # Throughput comparison
        baseline_rps = baseline_summary.get("requests_per_second", 0)
        current_rps = current_summary.get("requests_per_second", 0)
        
        if current_rps < baseline_rps * 0.8:  # 20% reduction threshold
            analysis["regressions"].append({
                "metric": "requests_per_second",
                "baseline": baseline_rps,
                "current": current_rps,
                "degradation_percent": ((baseline_rps - current_rps) / baseline_rps) * 100,
            })
            analysis["regression_detected"] = True
        
        # Success rate comparison
        baseline_success = baseline_summary.get("success_rate_percent", 0)
        current_success = current_summary.get("success_rate_percent", 0)
        
        if current_success < baseline_success - 5:  # 5% absolute reduction
            analysis["regressions"].append({
                "metric": "success_rate_percent",
                "baseline": baseline_success,
                "current": current_success,
                "degradation_percent": baseline_success - current_success,
            })
            analysis["regression_detected"] = True
        
        # Set overall assessment
        if analysis["regression_detected"]:
            analysis["overall_assessment"] = "fail"
        
        return analysis
    
    def _calculate_performance_grade(self, stats) -> str:
        """Calculate performance grade based on test results."""
        if stats.total.num_requests == 0:
            return "N/A"
        
        score = 100
        
        # Deduct for high response times
        avg_response_time = stats.total.avg_response_time
        if avg_response_time > 100:
            score -= min(40, (avg_response_time - 100) / 25)
        
        # Deduct for errors
        error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        score -= error_rate * 10
        
        # Deduct for low throughput
        rps = stats.total.current_rps
        if rps < 10:
            score -= (10 - rps) * 2
        
        # Convert to letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, stats) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        if stats.total.num_requests == 0:
            return ["No requests were made - check test configuration"]
        
        # Response time recommendations
        avg_response_time = stats.total.avg_response_time
        if avg_response_time > 1000:
            recommendations.append("High response times detected - consider optimizing database queries and adding caching")
        elif avg_response_time > 500:
            recommendations.append("Moderate response times - review application logic for performance bottlenecks")
        
        # Error rate recommendations
        error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        if error_rate > 5:
            recommendations.append("High error rate - implement better error handling and retry mechanisms")
        elif error_rate > 1:
            recommendations.append("Some errors detected - review error logs and improve system reliability")
        
        # Throughput recommendations
        rps = stats.total.current_rps
        if rps < 10:
            recommendations.append("Low throughput - consider horizontal scaling or performance optimization")
        
        if not recommendations:
            recommendations.append("Performance looks good - continue monitoring in production")
        
        return recommendations
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        index = int(len(data) * percentile / 100)
        return data[min(index, len(data) - 1)]
    
    def _save_report(self, report: Dict, report_type: str = "load_test") -> str:
        """Save test report to file."""
        timestamp = int(time.time())
        filename = f"{report_type}_report_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(filepath)


def main():
    """Main entry point for load test runner."""
    parser = argparse.ArgumentParser(description="AI Documentation Vector DB Load Test Runner")
    
    # Test execution mode
    parser.add_argument(
        "--mode", 
        choices=["locust", "pytest", "scenario", "benchmark", "regression"],
        default="locust",
        help="Test execution mode"
    )
    
    # Test configuration
    parser.add_argument(
        "--config", 
        choices=["light", "moderate", "heavy", "stress"],
        default="light",
        help="Load test configuration"
    )
    
    # Load profile
    parser.add_argument(
        "--profile",
        choices=list(LOAD_PROFILES.keys()),
        help="Load test profile (steady, ramp_up, spike, etc.)"
    )
    
    # Test type for pytest mode
    parser.add_argument(
        "--test-type",
        choices=["all", "load", "stress", "spike", "endurance", "volume", "scalability"],
        default="all",
        help="Type of load tests to run (pytest mode)"
    )
    
    # Markers for pytest mode
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Pytest markers to run (e.g., load stress spike)"
    )
    
    # Custom scenario file
    parser.add_argument(
        "--scenario",
        help="Path to custom scenario JSON file"
    )
    
    # Endpoints for benchmark mode
    parser.add_argument(
        "--endpoints",
        nargs="+",
        help="Endpoints to benchmark"
    )
    
    # Baseline file for regression mode
    parser.add_argument(
        "--baseline",
        help="Path to baseline results file for regression testing"
    )
    
    # Locust options
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Locust in headless mode"
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run Locust with web UI"
    )
    
    parser.add_argument(
        "--web-port",
        type=int,
        default=8089,
        help="Port for Locust web UI"
    )
    
    # Host configuration
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="Target host URL"
    )
    
    # Test parameters
    parser.add_argument(
        "--users",
        type=int,
        help="Number of concurrent users"
    )
    
    parser.add_argument(
        "--spawn-rate",
        type=int,
        help="User spawn rate per second"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="Test duration in seconds"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="load_test_results",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level, None)
    
    # Create load test runner
    runner = LoadTestRunner()
    runner.results_dir = Path(args.output_dir)
    runner.results_dir.mkdir(exist_ok=True)
    
    # Get base configuration
    if args.config in runner.default_configs:
        config = runner.default_configs[args.config].copy()
    else:
        config = runner.default_configs["light"].copy()
    
    # Override with command line arguments
    if args.host:
        config["host"] = args.host
    if args.users:
        config["users"] = args.users
    if args.spawn_rate:
        config["spawn_rate"] = args.spawn_rate
    if args.duration:
        config["duration"] = args.duration
    
    # Execute based on mode
    try:
        if args.mode == "locust":
            headless = args.headless and not args.web
            result = runner.run_locust_test(
                config=config,
                profile=args.profile,
                headless=headless,
                web_port=args.web_port
            )
        
        elif args.mode == "pytest":
            result = runner.run_pytest_load_tests(
                test_type=args.test_type,
                markers=args.markers
            )
        
        elif args.mode == "scenario":
            if not args.scenario:
                logger.error("Scenario file required for scenario mode")
                sys.exit(1)
            result = runner.run_custom_scenario(args.scenario)
        
        elif args.mode == "benchmark":
            if not args.endpoints:
                logger.error("Endpoints required for benchmark mode")
                sys.exit(1)
            result = runner.benchmark_endpoints(args.endpoints, config)
        
        elif args.mode == "regression":
            if not args.baseline:
                logger.error("Baseline file required for regression mode")
                sys.exit(1)
            result = runner.validate_performance_regression(args.baseline, config)
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"Load Test Summary - Mode: {args.mode}")
        print("=" * 60)
        
        if isinstance(result, dict):
            if "status" in result:
                print(f"Status: {result['status']}")
            
            if "summary" in result:
                summary = result["summary"]
                print(f"Total Requests: {summary.get('total_requests', 'N/A')}")
                print(f"Success Rate: {summary.get('success_rate_percent', 'N/A'):.1f}%")
                print(f"Avg Response Time: {summary.get('avg_response_time_ms', 'N/A'):.1f}ms")
                print(f"Requests/Second: {summary.get('requests_per_second', 'N/A'):.1f}")
            
            if "performance_grade" in result:
                print(f"Performance Grade: {result['performance_grade']}")
            
            if "recommendations" in result and result["recommendations"]:
                print("\nRecommendations:")
                for i, rec in enumerate(result["recommendations"], 1):
                    print(f"  {i}. {rec}")
        
        print("=" * 60)
        
        # Exit with appropriate code
        if isinstance(result, dict):
            if result.get("status") == "failed" or result.get("regression_detected"):
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()