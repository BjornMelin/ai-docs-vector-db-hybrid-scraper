"""Core implementation details for the load testing runner."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from locust import main as locust_main
from locust.env import Environment

from tests.load.load_profiles import LOAD_PROFILES, get_load_profile
from tests.load.locust_load_runner import (
    AdminUser,
    VectorDBUser,
    create_load_test_environment,
)
from tests.load.performance_utils import grade_from_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_EXECUTABLES = frozenset({"uv", "python", "python3"})
DANGEROUS_COMMAND_PATTERNS = (
    r"[;&|`$(){}[\]<>]",  # Shell metacharacters
    r"\.\.",  # Directory traversal
    r"/dev/",  # Device files
    r"/proc/",  # Process files
    r"/sys/",  # System files
    r"~",  # Home directory expansion
    r"\$[A-Za-z_]",  # Environment variable expansion
)
MAX_CMD_ARGUMENTS = 50
MAX_CMD_LENGTH = 2000
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Main load test runner with multiple execution modes."""

    def __init__(self):
        """Initialize load test runner."""
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
        logger.info("Starting Locust load test with config: %s", config)

        # Create environment
        env = create_load_test_environment(
            host=config["host"], user_classes=[VectorDBUser, AdminUser]
        )

        # Apply load profile if specified
        if profile and profile in LOAD_PROFILES:
            load_profile = get_load_profile(profile)
            if load_profile:
                env.shape_class = load_profile
                logger.info("Applied load profile: %s", profile)

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
                logger.info("Test report saved to: %s", report_file)

            return report
        # Run with web UI
        logger.info("Starting Locust web UI on port %s", web_port)
        logger.info("Visit http://localhost:%s to control the test", web_port)

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
        """Run load tests using pytest with security validation."""
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
                cwd=PROJECT_ROOT,
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
        logger.info("Running custom scenario: %s", scenario_file)

        try:
            with Path(scenario_file).open(encoding="utf-8") as f:
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
            return result

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
        logger.info("Benchmarking endpoints: %s", endpoints)

        results = {}

        for endpoint in endpoints:
            logger.info("Benchmarking endpoint: %s", endpoint)

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
        logger.info("Endpoint benchmark report saved to: %s", report_file)

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
            with Path(baseline_file).open(encoding="utf-8") as f:
                baseline = json.load(f)

            # Run current test
            current_result = self.run_locust_test(current_config, headless=True)

            # Compare results
            regression_analysis = self._analyze_performance_regression(
                baseline, current_result
            )

            # Save regression report
            report_file = self._save_report(regression_analysis, "regression_analysis")
            logger.info("Regression analysis saved to: %s", report_file)
            return regression_analysis

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
        """Generate test report."""
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

        return grade_from_score(score)

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

        with Path(filepath).open("w", encoding="utf-8") as f:
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
        """Perform security validation on the command."""
        self._ensure_command_structure(cmd)
        self._ensure_allowed_executable(cmd[0])
        self._validate_command_components(cmd)
        self._validate_command_lengths(cmd)

    def _ensure_command_structure(self, cmd: list[str]) -> None:
        """Ensure the command is a non-empty list."""
        if not isinstance(cmd, list):
            msg = "Command must be a list"
            raise TypeError(msg)

        if not cmd:
            msg = "Command cannot be empty"
            raise ValueError(msg)

    def _ensure_allowed_executable(self, executable: str) -> None:
        """Validate that the command executable is explicitly allowed."""
        if executable not in ALLOWED_EXECUTABLES:
            msg = f"Executable '{executable}' not allowed"
            raise ValueError(msg)

    def _validate_command_components(self, cmd: list[str]) -> None:
        """Validate each component of the command string."""
        tests_path = str(PROJECT_ROOT / "tests")

        for component in cmd:
            self._validate_command_component(component, tests_path)

    def _validate_command_component(self, component: object, tests_path: str) -> None:
        """Validate individual command component for type and content."""
        if not isinstance(component, str):
            msg = "All command components must be strings"
            raise TypeError(msg)

        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, component):
                msg = f"Dangerous pattern detected in command: {component}"
                raise ValueError(msg)

        if component.startswith("/") and not component.startswith(tests_path):
            msg = f"Absolute path not allowed: {component}"
            raise ValueError(msg)

    def _validate_command_lengths(self, cmd: list[str]) -> None:
        """Validate command length to prevent overflow attacks."""
        if len(cmd) > MAX_CMD_ARGUMENTS:
            msg = f"Command too long (maximum {MAX_CMD_ARGUMENTS} arguments)"
            raise ValueError(msg)

        cmd_str = " ".join(cmd)
        if len(cmd_str) > MAX_CMD_LENGTH:
            msg = f"Command string too long (maximum {MAX_CMD_LENGTH} characters)"
            raise ValueError(msg)

    def _get_secure_environment(self) -> dict[str, str]:
        """Get a secure environment for subprocess execution."""
        # Security: Start with minimal environment
        secure_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "PYTHONPATH": str(PROJECT_ROOT),
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
