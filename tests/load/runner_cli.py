"""CLI utilities for the load test runner."""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any

from locust.log import setup_logging

from tests.load.load_profiles import LOAD_PROFILES
from tests.load.runner_core import LoadTestRunner


MODE_CHOICES = (
    "locust",
    "pytest",
    "scenario",
    "benchmark",
    "regression",
)

LOAD_TEST_TYPE_CHOICES = (
    "all",
    "load",
    "stress",
    "spike",
    "endurance",
    "volume",
    "scalability",
)

PROFILE_CHOICES = tuple(sorted(LOAD_PROFILES))

ARGUMENT_DEFINITIONS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("--mode",),
        {
            "choices": MODE_CHOICES,
            "default": "locust",
            "help": "Test execution mode",
        },
    ),
    (
        ("--config",),
        {
            "choices": ("light", "moderate", "heavy", "stress"),
            "default": "light",
            "help": "Load test configuration",
        },
    ),
    (
        ("--profile",),
        {
            "choices": PROFILE_CHOICES,
            "help": "Load test profile (steady, ramp_up, spike, etc.)",
        },
    ),
    (
        ("--test-type",),
        {
            "choices": LOAD_TEST_TYPE_CHOICES,
            "default": "all",
            "help": "Type of load tests to run (pytest mode)",
        },
    ),
    (
        ("--markers",),
        {
            "nargs": "+",
            "help": "Pytest markers to run (e.g., load stress spike)",
        },
    ),
    (("--scenario",), {"help": "Path to custom scenario JSON file"}),
    (
        ("--endpoints",),
        {"nargs": "+", "help": "Endpoints to benchmark"},
    ),
    (
        ("--baseline",),
        {"help": "Path to baseline results file for regression testing"},
    ),
    (
        ("--headless",),
        {
            "action": "store_true",
            "default": True,
            "help": "Run Locust in headless mode",
        },
    ),
    (("--web",), {"action": "store_true", "help": "Run Locust with web UI"}),
    (
        ("--web-port",),
        {"type": int, "default": 8089, "help": "Port for Locust web UI"},
    ),
    (
        ("--host",),
        {"default": "http://localhost:8000", "help": "Target host URL"},
    ),
    (
        ("--users",),
        {"type": int, "help": "Number of concurrent users"},
    ),
    (
        ("--spawn-rate",),
        {"type": int, "help": "User spawn rate per second"},
    ),
    (
        ("--duration",),
        {"type": int, "help": "Test duration in seconds"},
    ),
    (
        ("--output-dir",),
        {
            "default": "load_test_results",
            "help": "Output directory for test results",
        },
    ),
    (
        ("--verbose", "-v"),
        {"action": "store_true", "help": "Verbose output"},
    ),
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the load test runner."""

    parser = argparse.ArgumentParser(
        description="AI Documentation Vector DB Load Test Runner"
    )

    for flags, kwargs in ARGUMENT_DEFINITIONS:
        parser.add_argument(*flags, **kwargs)

    return parser


def configure_logging(verbose: bool) -> None:
    """Configure logging verbosity for both standard logging and Locust."""

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    setup_logging(log_level, None)


def configure_runner_output(runner: LoadTestRunner, output_dir: str) -> None:
    """Ensure the runner's output directory exists and is configured."""

    runner.results_dir = Path(output_dir)
    runner.results_dir.mkdir(exist_ok=True)


def build_base_config(runner: LoadTestRunner, args: argparse.Namespace) -> dict:
    """Create the base configuration for load tests using CLI arguments."""

    base_config = runner.default_configs.get(
        args.config, runner.default_configs["light"]
    ).copy()

    overrides = {
        "host": args.host,
        "users": args.users,
        "spawn_rate": args.spawn_rate,
        "duration": args.duration,
    }

    for key, value in overrides.items():
        if value is not None:
            base_config[key] = value

    return base_config


def _require_argument(value: object, message: str) -> None:
    """Ensure a required CLI argument is provided, exiting with an error if missing."""

    if value:
        return

    logger.error(message)
    raise SystemExit(1)


def execute_mode(
    args: argparse.Namespace, runner: LoadTestRunner, config: dict
) -> dict:
    """Execute the selected mode and return the resulting payload."""

    mode = args.mode

    if mode == "locust":
        headless = args.headless and not args.web
        return runner.run_locust_test(
            config=config,
            profile=args.profile,
            headless=headless,
            web_port=args.web_port,
        )

    if mode == "pytest":
        return runner.run_pytest_load_tests(
            test_type=args.test_type,
            markers=args.markers,
        )

    if mode == "scenario":
        _require_argument(args.scenario, "Scenario file required for scenario mode")
        return runner.run_custom_scenario(args.scenario)

    if mode == "benchmark":
        _require_argument(args.endpoints, "Endpoints required for benchmark mode")
        return runner.benchmark_endpoints(args.endpoints, config)

    if mode == "regression":
        _require_argument(args.baseline, "Baseline file required for regression mode")
        return runner.validate_performance_regression(args.baseline, config)

    return {"status": "error", "error": f"Unknown mode: {mode}"}


def print_summary(mode: str, result: dict[str, object]) -> None:
    """Render a concise summary of the load test execution."""

    print("\n" + "=" * 60)
    print(f"Load Test Summary - Mode: {mode}")
    print("=" * 60)

    status = result.get("status")
    if status is not None:
        print(f"Status: {status}")

    summary = result.get("summary")
    if isinstance(summary, dict):
        total_requests = summary.get("_total_requests", "N/A")
        success_rate = summary.get("success_rate_percent")
        avg_response = summary.get("avg_response_time_ms")
        requests_per_second = summary.get("requests_per_second")

        print(f"Total Requests: {total_requests}")
        if success_rate is not None:
            print(f"Success Rate: {float(success_rate):.1f}%")
        if avg_response is not None:
            print(f"Avg Response Time: {float(avg_response):.1f}ms")
        if requests_per_second is not None:
            print(f"Requests/Second: {float(requests_per_second):.1f}")

    performance_grade = result.get("performance_grade")
    if performance_grade is not None:
        print(f"Performance Grade: {performance_grade}")

    recommendations = result.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        print("\nRecommendations:")
        for idx, recommendation in enumerate(recommendations, 1):
            print(f"  {idx}. {recommendation}")

    print("=" * 60)


def should_exit_with_error(result: dict[str, object]) -> bool:
    """Determine whether the process should exit with a non-zero status."""

    status = result.get("status")
    if status == "failed":
        return True

    regression_detected = result.get("regression_detected")
    return bool(regression_detected)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for load test runner."""

    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)

    runner = LoadTestRunner()
    configure_runner_output(runner, args.output_dir)

    config = build_base_config(runner, args)

    try:
        result = execute_mode(args, runner, config)
    except KeyboardInterrupt as exc:
        logger.info("Test interrupted by user")
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001 - capture for structured logging
        logger.exception("Test execution failed")
        if args.verbose:
            traceback.print_exc()
        raise SystemExit(1) from exc

    if not isinstance(result, dict):
        logger.error("Unexpected result type: %s", type(result))
        raise SystemExit(1)

    print_summary(args.mode, result)

    if should_exit_with_error(result):
        raise SystemExit(1)
