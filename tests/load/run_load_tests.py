#!/usr/bin/env python3
"""Comprehensive load test runner for AI Documentation Vector DB.

This script provides a command-line interface for running various types of load tests
with different configurations, profiles, and reporting options.
"""

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import time

# Add the project root to Python path
import traceback
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from locust import main as locust_main  # noqa: E402
from locust.env import Environment  # noqa: E402
from locust.log import setup_logging  # noqa: E402

from tests.load.load_profiles import LOAD_PROFILES, get_load_profile  # noqa: E402
from tests.load.locust_load_runner import (  # noqa: E402
    AdminUser,
    VectorDBUser,
    create_load_test_environment,
)


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

    def run_locust_test(
        self,
        config: dict,
        profile: str | None = None,
        headless: bool = True,
        web_port: int = 8089,
    ) -> dict:
        """Run load test using Locust."""
        logger.info(
            "Starting Locust load test with config: %s", config
        )  # TODO: Convert f-string to logging format

        # Create environment
        env = create_load_test_environment(
            host=config["host"], user_classes=[VectorDBUser, AdminUser]
        )

        # Apply load profile if specified
        if profile and profile in LOAD_PROFILES:
            load_profile = get_load_profile(profile)
            if load_profile:
                env.shape_class = load_profile
                logger.info(
                    "Applied load profile: %s", profile
                )  # TODO: Convert f-string to logging format

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
                logger.info(
                    "Test report saved to: %s", report_file
                )  # TODO: Convert f-string to logging format

            return report
        # Run with web UI
        logger.info(
            "Starting Locust web UI on port %s", web_port
        )  # TODO: Convert f-string to logging format
        logger.info(
            "Visit http://localhost:%s to control the test", web_port
        )  # TODO: Convert f-string to logging format

        # Set up Locust arguments for web mode
        locust_args = [
            "--web-port",
            str(web_port),
            "--host",
            config["host"],
            "--users",
            str(config["users"]),
            "--spawn-rate",
            str(config["spawn_rate"]),
        ]

        if config.get("duration"):
            locust_args.extend(["--run-time", f"{config['duration']}s"])

        # Run Locust main
        os.environ["LOCUST_USER_CLASSES"] = "VectorDBUser,AdminUser"
        sys.argv = ["locust", *locust_args]

        with contextlib.suppress(SystemExit):
            locust_main.main()

        return {"status": "completed_web_mode"}

    def run_pytest_load_tests(
        self, test_type: str = "all", markers: list[str] | None = None
    ) -> dict:
        """Run load tests using pytest with comprehensive security validation."""
        logger.info("Running pytest load tests: %s", test_type)

        # Security: Validate and sanitize all inputs
        sanitized_test_type = self._validate_test_type(test_type)
        sanitized_markers = self._validate_markers(markers) if markers else None

        # Build secure pytest command with validated inputs
        cmd = self._build_secure_pytest_command(sanitized_test_type, sanitized_markers)

        # Enhanced security validation
        self._validate_command_security(cmd)

        # Run tests with security constraints
        try:
            result = subprocess.run(  # noqa: S603  # Secure: validated executable, no shell, no user input
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                check=False,
                timeout=3600,  # Security: Prevent hanging processes
                env=self._get_secure_environment(),  # Security: Clean environment
            )

            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            logger.exception("Pytest execution timed out after 1 hour")
            return {
                "status": "timeout",
                "error": "Test execution exceeded 1 hour timeout",
                "command": " ".join(cmd),
            }
        except Exception as e:
            logger.exception("Failed to run pytest tests")
            return {
                "status": "error",
                "error": str(e),
                "command": " ".join(cmd),
            }

    def run_custom_scenario(self, scenario_file: str) -> dict:
        """Run a custom load test scenario from JSON file."""
        logger.info(
            "Running custom scenario: %s", scenario_file
        )  # TODO: Convert f-string to logging format

        try:
            with Path(scenario_file).open() as f:
                scenario = json.load(f)

            # Validate scenario format
            required_fields = ["name", "description", "config"]
            for field in required_fields:
                if field not in scenario:
                    self._raise_missing_field_error(field)

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

        except Exception:
            logger.exception("Error in test execution")
        else:
            return result

        try:
            pass
        except Exception:
            logger.exception("Failed to run custom scenario")
            return {
                "status": "error",
                "error": "Unknown error",
                "scenario_file": scenario_file,
            }

    def _raise_missing_field_error(self, field: str) -> None:
        """Raise ValueError for missing required field."""
        msg = f"Missing required field: {field}"
        raise ValueError(msg)

    def benchmark_endpoints(self, endpoints: list[str], config: dict) -> dict:
        """Benchmark specific endpoints."""
        logger.info(
            "Benchmarking endpoints: %s", endpoints
        )  # TODO: Convert f-string to logging format

        results = {}

        for endpoint in endpoints:
            logger.info(
                "Benchmarking endpoint: %s", endpoint
            )  # TODO: Convert f-string to logging format

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
        logger.info(
            "Endpoint benchmark report saved to: %s", report_file
        )  # TODO: Convert f-string to logging format

        return comparative_report

    def validate_performance_regression(
        self, baseline_file: str, current_config: dict
    ) -> dict:
        """Validate performance against a baseline."""
        logger.info(
            "Running performance regression test against baseline: %s", baseline_file
        )

        try:
            # Load baseline results
            with Path(baseline_file).open() as f:
                baseline = json.load(f)

            # Run current test
            current_result = self.run_locust_test(current_config, headless=True)

            # Compare results
            regression_analysis = self._analyze_performance_regression(
                baseline, current_result
            )

            # Save regression report
            report_file = self._save_report(regression_analysis, "regression_analysis")
            logger.info(
                "Regression analysis saved to: %s", report_file
            )  # TODO: Convert f-string to logging format

        except Exception:
            logger.exception("Error in test execution")
        else:
            return regression_analysis

        try:
            pass
        except Exception:
            logger.exception("Failed to run regression test")
            return {
                "status": "error",
                "error": "Unknown error",
                "baseline_file": baseline_file,
            }

    def _generate_test_report(
        self, env: Environment, config: dict, profile: str | None
    ) -> dict:
        """Generate comprehensive test report."""
        stats = env.stats

        if not stats or stats.total.num_requests == 0:
            return {
                "status": "no_data",
                "message": "No requests were made during the test",
            }

        # Calculate key metrics
        _total_requests = stats.total.num_requests
        _total_failures = stats.total.num_failures
        success_rate = (
            ((_total_requests - _total_failures) / _total_requests) * 100
            if _total_requests > 0
            else 0
        )

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
                "_total_requests": _total_requests,
                "_total_failures": _total_failures,
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
                    "success_rate_percent": (
                        (entry.num_requests - entry.num_failures) / entry.num_requests
                    )
                    * 100,
                }

        return report

    def _run_endpoint_benchmark(self, endpoint: str, _config: dict) -> dict:
        """Run benchmark for a specific endpoint."""
        # This would create a custom user class focused on the specific endpoint
        # For now, return a simplified result
        return {
            "endpoint": endpoint,
            "avg_response_time_ms": 150,  # Placeholder
            "requests_per_second": 25,  # Placeholder
            "success_rate_percent": 99.5,  # Placeholder
        }

    def _generate_endpoint_comparison(self, results: dict) -> dict:
        """Generate comparative analysis of endpoint performance."""
        return {
            "timestamp": time.time(),
            "endpoints": results,
            "analysis": {
                "fastest_endpoint": min(
                    results.keys(), key=lambda k: results[k]["avg_response_time_ms"]
                ),
                "slowest_endpoint": max(
                    results.keys(), key=lambda k: results[k]["avg_response_time_ms"]
                ),
                "highest_throughput": max(
                    results.keys(), key=lambda k: results[k]["requests_per_second"]
                ),
                "most_reliable": max(
                    results.keys(), key=lambda k: results[k]["success_rate_percent"]
                ),
            },
        }

    def _analyze_performance_regression(self, baseline: dict, current: dict) -> dict:
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
            analysis["regressions"].append(
                {
                    "metric": "avg_response_time_ms",
                    "baseline": baseline_rt,
                    "current": current_rt,
                    "degradation_percent": ((current_rt - baseline_rt) / baseline_rt)
                    * 100,
                }
            )
            analysis["regression_detected"] = True

        # Throughput comparison
        baseline_rps = baseline_summary.get("requests_per_second", 0)
        current_rps = current_summary.get("requests_per_second", 0)

        if current_rps < baseline_rps * 0.8:  # 20% reduction threshold
            analysis["regressions"].append(
                {
                    "metric": "requests_per_second",
                    "baseline": baseline_rps,
                    "current": current_rps,
                    "degradation_percent": ((baseline_rps - current_rps) / baseline_rps)
                    * 100,
                }
            )
            analysis["regression_detected"] = True

        # Success rate comparison
        baseline_success = baseline_summary.get("success_rate_percent", 0)
        current_success = current_summary.get("success_rate_percent", 0)

        if current_success < baseline_success - 5:  # 5% absolute reduction
            analysis["regressions"].append(
                {
                    "metric": "success_rate_percent",
                    "baseline": baseline_success,
                    "current": current_success,
                    "degradation_percent": baseline_success - current_success,
                }
            )
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
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    def _generate_recommendations(self, stats) -> list[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []

        if stats.total.num_requests == 0:
            return ["No requests were made - check test configuration"]

        # Response time recommendations
        avg_response_time = stats.total.avg_response_time
        if avg_response_time > 1000:
            recommendations.append(
                "High response times detected - consider optimizing database "
                "queries and adding caching"
            )
        elif avg_response_time > 500:
            recommendations.append(
                "Moderate response times - review application logic for "
                "performance bottlenecks"
            )

        # Error rate recommendations
        error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        if error_rate > 5:
            recommendations.append(
                "High error rate - implement better error handling and retry mechanisms"
            )
        elif error_rate > 1:
            recommendations.append(
                "Some errors detected - review error logs and "
                "improve system reliability"
            )

        # Throughput recommendations
        rps = stats.total.current_rps
        if rps < 10:
            recommendations.append(
                "Low throughput - consider horizontal scaling or "
                "performance optimization"
            )

        if not recommendations:
            recommendations.append(
                "Performance looks good - continue monitoring in production"
            )

        return recommendations

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        index = int(len(data) * percentile / 100)
        return data[min(index, len(data) - 1)]

    def _save_report(self, report: dict, report_type: str = "load_test") -> str:
        """Save test report to file."""
        timestamp = int(time.time())
        filename = f"{report_type}_report_{timestamp}.json"
        filepath = self.results_dir / filename

        with Path(filepath).open("w") as f:
            json.dump(report, f, indent=2)

        return str(filepath)

    def _validate_test_type(self, test_type: str) -> str:
        """Validate and sanitize test type parameter against injection attacks."""
        if not isinstance(test_type, str):
            msg = "Test type must be a string"
            raise TypeError(msg)

        # Security: Allowlist of valid test types
        valid_test_types = {
            "all",
            "load",
            "stress",
            "spike",
            "endurance",
            "volume",
            "scalability",
        }

        # Security: Remove any non-alphanumeric characters except underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", test_type.strip())

        if not sanitized:
            msg = "Invalid test type: empty after sanitization"
            raise ValueError(msg)

        if sanitized not in valid_test_types:
            msg = f"Invalid test type: {sanitized}. Must be one of: {valid_test_types}"
            raise ValueError(msg)

        return sanitized

    def _validate_markers(self, markers: list[str]) -> list[str]:
        """Validate and sanitize pytest markers against injection attacks."""
        if not isinstance(markers, list):
            msg = "Markers must be a list"
            raise TypeError(msg)

        sanitized_markers = []
        # Security: Allowlist of valid marker patterns
        valid_marker_pattern = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$")

        for marker in markers:
            if not isinstance(marker, str):
                msg = "Each marker must be a string"
                raise TypeError(msg)

            # Security: Remove any dangerous characters
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", marker.strip())

            if not sanitized:
                continue  # Skip empty markers

            if not valid_marker_pattern.match(sanitized):
                msg = f"Invalid marker format: {marker}"
                raise ValueError(msg)

            # Security: Limit marker length to prevent buffer overflow
            if len(sanitized) > 50:
                msg = f"Marker too long: {marker}"
                raise ValueError(msg)

            sanitized_markers.append(sanitized)

        # Security: Limit total number of markers
        if len(sanitized_markers) > 10:
            msg = "Too many markers specified (maximum 10)"
            raise ValueError(msg)

        return sanitized_markers

    def _build_secure_pytest_command(
        self, test_type: str, markers: list[str] | None
    ) -> list[str]:
        """Build pytest command with security constraints."""
        # Security: Start with hardcoded base command
        cmd = ["uv", "run", "pytest"]

        # Security: Map test types to secure paths within tests/load/
        test_type_paths = {
            "all": "tests/load/",
            "load": "tests/load/load_testing/",
            "stress": "tests/load/stress_testing/",
            "spike": "tests/load/spike_testing/",
            "endurance": "tests/load/endurance_testing/",
            "volume": "tests/load/volume_testing/",
            "scalability": "tests/load/scalability/",
        }

        # Security: Use predefined path mapping instead of dynamic construction
        test_path = test_type_paths.get(test_type, "tests/load/")
        cmd.append(test_path)

        # Security: Add hardcoded safe options
        cmd.extend(["-v", "--tb=short", "--disable-warnings"])

        # Security: Add validated markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        elif test_type != "all":
            cmd.extend(["-m", test_type])

        return cmd

    def _validate_command_security(self, cmd: list[str]) -> None:
        """Perform comprehensive security validation on the command."""
        if not isinstance(cmd, list):
            msg = "Command must be a list"
            raise TypeError(msg)

        if not cmd:
            msg = "Command cannot be empty"
            raise ValueError(msg)

        # Security: Validate executable
        allowed_executables = {"uv", "python", "python3"}
        if cmd[0] not in allowed_executables:
            msg = f"Executable '{cmd[0]}' not allowed"
            raise ValueError(msg)

        # Security: Validate all command components
        dangerous_patterns = [
            r"[;&|`$(){}[\]<>]",  # Shell metacharacters
            r"\.\.",  # Directory traversal
            r"/dev/",  # Device files
            r"/proc/",  # Process files
            r"/sys/",  # System files
            r"~",  # Home directory expansion
            r"\$[A-Za-z_]",  # Environment variable expansion
        ]

        for component in cmd:
            if not isinstance(component, str):
                msg = "All command components must be strings"
                raise TypeError(msg)

            # Security: Check for dangerous patterns
            for pattern in dangerous_patterns:
                if re.search(pattern, component):
                    msg = f"Dangerous pattern detected in command: {component}"
                    raise ValueError(msg)

            # Security: Validate path components
            if component.startswith("/") and not component.startswith("tests/"):
                # Allow absolute paths only if they're in the tests directory
                tests_path = str(project_root / "tests")
                if not component.startswith(tests_path):
                    msg = f"Absolute path not allowed: {component}"
                    raise ValueError(msg)

        # Security: Validate command length to prevent argument overflow
        if len(cmd) > 50:
            msg = "Command too long (maximum 50 arguments)"
            raise ValueError(msg)

        # Security: Validate total command string length
        cmd_str = " ".join(cmd)
        if len(cmd_str) > 2000:
            msg = "Command string too long (maximum 2000 characters)"
            raise ValueError(msg)

    def _get_secure_environment(self) -> dict[str, str]:
        """Get a secure environment for subprocess execution."""
        # Security: Start with minimal environment
        secure_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "PYTHONPATH": str(project_root),
        }

        # Security: Add only safe environment variables
        safe_vars = {
            "TERM",
            "LANG",
            "LC_ALL",
            "TZ",
            "UV_CACHE_DIR",
            "PYTEST_CURRENT_TEST",
            "CI",
            "GITHUB_ACTIONS",
        }

        for var in safe_vars:
            if var in os.environ:
                secure_env[var] = os.environ[var]

        # Security: Remove any variables that could be used for injection
        dangerous_vars = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONINSPECT",
            "PYTHONSTARTUP",
            "PYTHONEXECUTABLE",
        }

        for var in dangerous_vars:
            secure_env.pop(var, None)

        return secure_env


def main():
    """Main entry point for load test runner."""
    parser = argparse.ArgumentParser(
        description="AI Documentation Vector DB Load Test Runner"
    )

    # Test execution mode
    parser.add_argument(
        "--mode",
        choices=["locust", "pytest", "scenario", "benchmark", "regression"],
        default="locust",
        help="Test execution mode",
    )

    # Test configuration
    parser.add_argument(
        "--config",
        choices=["light", "moderate", "heavy", "stress"],
        default="light",
        help="Load test configuration",
    )

    # Load profile
    parser.add_argument(
        "--profile",
        choices=list(LOAD_PROFILES.keys()),
        help="Load test profile (steady, ramp_up, spike, etc.)",
    )

    # Test type for pytest mode
    parser.add_argument(
        "--test-type",
        choices=[
            "all",
            "load",
            "stress",
            "spike",
            "endurance",
            "volume",
            "scalability",
        ],
        default="all",
        help="Type of load tests to run (pytest mode)",
    )

    # Markers for pytest mode
    parser.add_argument(
        "--markers", nargs="+", help="Pytest markers to run (e.g., load stress spike)"
    )

    # Custom scenario file
    parser.add_argument("--scenario", help="Path to custom scenario JSON file")

    # Endpoints for benchmark mode
    parser.add_argument("--endpoints", nargs="+", help="Endpoints to benchmark")

    # Baseline file for regression mode
    parser.add_argument(
        "--baseline", help="Path to baseline results file for regression testing"
    )

    # Locust options
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Locust in headless mode",
    )

    parser.add_argument("--web", action="store_true", help="Run Locust with web UI")

    parser.add_argument(
        "--web-port", type=int, default=8089, help="Port for Locust web UI"
    )

    # Host configuration
    parser.add_argument(
        "--host", default="http://localhost:8000", help="Target host URL"
    )

    # Test parameters
    parser.add_argument("--users", type=int, help="Number of concurrent users")

    parser.add_argument("--spawn-rate", type=int, help="User spawn rate per second")

    parser.add_argument("--duration", type=int, help="Test duration in seconds")

    # Output options
    parser.add_argument(
        "--output-dir",
        default="load_test_results",
        help="Output directory for test results",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

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
                web_port=args.web_port,
            )

        elif args.mode == "pytest":
            result = runner.run_pytest_load_tests(
                test_type=args.test_type, markers=args.markers
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
                print(f"Total Requests: {summary.get('_total_requests', 'N/A')}")
                print(
                    f"Success Rate: {summary.get('success_rate_percent', 'N/A'):.1f}%"
                )
                print(
                    "Avg Response Time: "
                    f"{summary.get('avg_response_time_ms', 'N/A'):.1f}ms"
                )
                print(
                    f"Requests/Second: {summary.get('requests_per_second', 'N/A'):.1f}"
                )

            if "performance_grade" in result:
                print(f"Performance Grade: {result['performance_grade']}")

            if result.get("recommendations"):
                print("\nRecommendations:")
                for i, rec in enumerate(result["recommendations"], 1):
                    print(f"  {i}. {rec}")

        print("=" * 60)

        # Exit with appropriate code
        if isinstance(result, dict) and (
            result.get("status") == "failed" or result.get("regression_detected")
        ):
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)

    except Exception:
        logger.exception("Test execution failed")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
