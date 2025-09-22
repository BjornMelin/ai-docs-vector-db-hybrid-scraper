"""Performance reporting for pytest-xdist parallel execution.

This module provides performance monitoring and reporting capabilities
for parallel test execution with detailed metrics and insights.
"""

import importlib
import json
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _pytest.config import Config
from _pytest.reports import TestReport


def _load_psutil():  # pragma: no cover - optional dependency helper
    try:
        return importlib.import_module("psutil")  # type: ignore[import-not-found]
    except ImportError:
        return None


psutil = _load_psutil()


@dataclass
class TestMetrics:
    """Metrics for individual test execution."""

    nodeid: str
    worker_id: str
    duration: float
    outcome: str
    start_time: float
    end_time: float
    memory_before: int | None = None
    memory_after: int | None = None
    cpu_percent: float | None = None

    @property
    def memory_delta(self) -> int | None:
        """Calculate memory usage change."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None


@dataclass
class WorkerMetrics:
    """Aggregated metrics for a test worker."""

    worker_id: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration: float = 0.0
    idle_time: float = 0.0
    test_metrics: list[TestMetrics] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Calculate worker efficiency (active time / total time)."""
        total_time = self.total_duration + self.idle_time
        if total_time > 0:
            return (self.total_duration / total_time) * 100
        return 0.0

    @property
    def average_test_duration(self) -> float:
        """Calculate average test duration."""
        if self.total_tests > 0:
            return self.total_duration / self.total_tests
        return 0.0


class PerformanceReporter:
    """pytest plugin for performance reporting in parallel execution."""

    def __init__(self, config: Config):
        self.config = config
        self.test_metrics: list[TestMetrics] = []
        self.worker_metrics: dict[str, WorkerMetrics] = defaultdict(
            lambda: WorkerMetrics(worker_id="unknown")
        )
        self.session_start_time: float | None = None
        self.session_end_time: float | None = None
        self.report_path = Path(
            config.getoption("--performance-report", default="test-performance.json")
        )
        self._last_metrics_by_nodeid: dict[str, TestMetrics] = {}

        # Try to import psutil for memory monitoring
        self.psutil = psutil
        if self.psutil is not None:
            self.process = self.psutil.Process()
            # Prime CPU sampling to avoid the first call returning a meaningless 0.0
            with suppress(self.psutil.Error):
                self.process.cpu_percent(None)
        else:
            self.process = None

    # pytest hooks

    def pytest_sessionstart(self, session):
        """Called at the start of the test session."""
        self.session_start_time = time.time()

    def pytest_sessionfinish(self, session):
        """Called at the end of the test session."""
        self.session_end_time = time.time()
        self.generate_report()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Monitor individual test execution."""
        worker_id = getattr(item.config, "workerinput", {}).get("workerid", "master")

        # Capture metrics before test
        memory_before = None
        if self.process and self.psutil:
            with suppress(self.psutil.Error):
                memory_before = self.process.memory_info().rss

        start_time = time.time()

        # Run the test
        outcome = yield

        end_time = time.time()
        duration = end_time - start_time

        # Capture metrics after test
        memory_after = None
        cpu_percent = None
        if self.process and self.psutil:
            with suppress(self.psutil.Error):
                memory_after = self.process.memory_info().rss
                cpu_percent = self.process.cpu_percent(interval=None)

        # Get test outcome
        report = outcome.get_result()
        test_outcome = report.outcome if hasattr(report, "outcome") else "unknown"

        # Record metrics
        metrics = TestMetrics(
            nodeid=item.nodeid,
            worker_id=worker_id,
            duration=duration,
            outcome=test_outcome,
            start_time=start_time,
            end_time=end_time,
            memory_before=memory_before,
            memory_after=memory_after,
            cpu_percent=cpu_percent,
        )

        self.test_metrics.append(metrics)
        self.update_worker_metrics(worker_id, metrics)
        self._last_metrics_by_nodeid[item.nodeid] = metrics

    def pytest_report_teststatus(self, report: TestReport, config: Config):
        """Process test reports for outcome tracking."""
        if report.when == "call":
            worker_id = getattr(config, "workerinput", {}).get("workerid", "master")
            worker_metrics = self.worker_metrics[worker_id]

            last_metrics = self._last_metrics_by_nodeid.get(report.nodeid)
            if last_metrics:
                last_metrics.outcome = report.outcome

            if report.passed:
                worker_metrics.passed_tests += 1
            elif report.failed:
                worker_metrics.failed_tests += 1
            elif report.skipped:
                worker_metrics.skipped_tests += 1

    def update_worker_metrics(self, worker_id: str, test_metrics: TestMetrics):
        """Update worker-level metrics."""
        worker = self.worker_metrics[worker_id]
        worker.worker_id = worker_id
        worker.total_tests += 1
        worker.total_duration += test_metrics.duration
        worker.test_metrics.append(test_metrics)

    def generate_report(self):
        """Generate the performance report."""
        if not self.session_start_time or not self.session_end_time:
            return

        total_duration = self.session_end_time - self.session_start_time

        # Analyze test distribution
        test_distribution = self.analyze_test_distribution()

        # Find slow tests
        slow_tests = self.find_slow_tests(threshold_percentile=90)

        # Calculate worker efficiency
        worker_efficiency = self.calculate_worker_efficiency()

        # Memory usage analysis
        memory_analysis = self.analyze_memory_usage()

        report = {
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "pytest_version": pytest.__version__,
                "total_duration": total_duration,
                "total_tests": len(self.test_metrics),
                "workers_used": len(self.worker_metrics),
            },
            "summary": {
                "passed": sum(w.passed_tests for w in self.worker_metrics.values()),
                "failed": sum(w.failed_tests for w in self.worker_metrics.values()),
                "skipped": sum(w.skipped_tests for w in self.worker_metrics.values()),
                "average_test_duration": sum(t.duration for t in self.test_metrics)
                / len(self.test_metrics)
                if self.test_metrics
                else 0,
            },
            "worker_metrics": {
                worker_id: {
                    "total_tests": metrics.total_tests,
                    "passed": metrics.passed_tests,
                    "failed": metrics.failed_tests,
                    "skipped": metrics.skipped_tests,
                    "total_duration": metrics.total_duration,
                    "average_duration": metrics.average_test_duration,
                    "efficiency": metrics.efficiency,
                }
                for worker_id, metrics in self.worker_metrics.items()
            },
            "test_distribution": test_distribution,
            "slow_tests": slow_tests,
            "worker_efficiency": worker_efficiency,
            "memory_analysis": memory_analysis,
            "recommendations": self.generate_recommendations(),
        }

        # Write JSON report
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Print summary to console
        self.print_summary(report)

    def analyze_test_distribution(self) -> dict[str, Any]:
        """Analyze how tests were distributed across workers."""
        distribution = {}

        for worker_id, metrics in self.worker_metrics.items():
            distribution[worker_id] = {
                "test_count": metrics.total_tests,
                "percentage": (metrics.total_tests / len(self.test_metrics) * 100)
                if self.test_metrics
                else 0,
                "duration_percentage": (
                    metrics.total_duration
                    / sum(t.duration for t in self.test_metrics)
                    * 100
                )
                if self.test_metrics
                else 0,
            }

        # Calculate load balance score (0-100, higher is better)
        if len(self.worker_metrics) > 1:
            test_counts = [m.total_tests for m in self.worker_metrics.values()]
            avg_tests = sum(test_counts) / len(test_counts)
            variance = sum((count - avg_tests) ** 2 for count in test_counts) / len(
                test_counts
            )
            std_dev = variance**0.5
            load_balance_score = (
                max(0, 100 - (std_dev / avg_tests * 100)) if avg_tests > 0 else 0
            )
        else:
            load_balance_score = 100

        distribution["load_balance_score"] = load_balance_score

        return distribution

    def find_slow_tests(self, threshold_percentile: int = 90) -> list[dict[str, Any]]:
        """Find the slowest tests."""
        if not self.test_metrics:
            return []

        sorted_tests = sorted(self.test_metrics, key=lambda t: t.duration, reverse=True)

        # Calculate percentile threshold
        durations = [t.duration for t in sorted_tests]
        threshold_idx = int(len(durations) * threshold_percentile / 100)
        threshold_duration = (
            durations[threshold_idx] if threshold_idx < len(durations) else 0
        )

        return [
            {
                "nodeid": test.nodeid,
                "duration": test.duration,
                "worker": test.worker_id,
                "outcome": test.outcome,
            }
            for test in sorted_tests[:10]
            if test.duration >= threshold_duration
        ]

    def calculate_worker_efficiency(self) -> dict[str, float]:
        """Calculate efficiency metrics for each worker."""
        efficiency = {}

        for worker_id, metrics in self.worker_metrics.items():
            efficiency[worker_id] = metrics.efficiency

        return efficiency

    def analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.psutil:
            return {"available": False}

        memory_deltas = []
        peak_memory = 0

        for test in self.test_metrics:
            if test.memory_delta is not None:
                memory_deltas.append(test.memory_delta)
            if test.memory_after is not None:
                peak_memory = max(peak_memory, test.memory_after)

        if not memory_deltas:
            return {"available": False}

        return {
            "available": True,
            "average_delta_mb": sum(memory_deltas) / len(memory_deltas) / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "total_delta_mb": sum(memory_deltas) / (1024 * 1024),
        }

    def generate_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        # Check load balance
        distribution = self.analyze_test_distribution()
        if distribution.get("load_balance_score", 100) < 80:
            recommendations.append(
                "Consider using --dist=loadfile or --dist=loadgroup for better test distribution"
            )

        # Check for slow tests
        slow_tests = self.find_slow_tests()
        if slow_tests and slow_tests[0]["duration"] > 10:
            recommendations.append(
                f"Optimize slow tests: {slow_tests[0]['nodeid']} takes {slow_tests[0]['duration']:.2f}s"
            )

        # Check worker efficiency
        efficiency_scores = list(self.calculate_worker_efficiency().values())
        if efficiency_scores and min(efficiency_scores) < 70:
            recommendations.append(
                "Some workers have low efficiency - consider reducing worker count"
            )

        # Memory recommendations
        memory_analysis = self.analyze_memory_usage()
        if (
            memory_analysis.get("available")
            and memory_analysis.get("peak_memory_mb", 0) > 2048
        ):
            recommendations.append(
                f"High memory usage detected ({memory_analysis['peak_memory_mb']:.0f} MB peak)"
            )

        return recommendations

    def print_summary(self, report: dict[str, Any]):
        """Print a summary of the performance report to console."""
        print("\n" + "=" * 60)
        print("pytest-xdist Performance Report")
        print("=" * 60)

        print(f"\nTotal Duration: {report['metadata']['total_duration']:.2f}s")
        print(f"Total Tests: {report['metadata']['total_tests']}")
        print(f"Workers Used: {report['metadata']['workers_used']}")

        print("\nTest Results:")
        print(f"  Passed: {report['summary']['passed']}")
        print(f"  Failed: {report['summary']['failed']}")
        print(f"  Skipped: {report['summary']['skipped']}")

        print("\nWorker Performance:")
        for worker_id, metrics in report["worker_metrics"].items():
            print(f"  {worker_id}:")
            print(f"    Tests: {metrics['total_tests']}")
            print(f"    Duration: {metrics['total_duration']:.2f}s")
            print(f"    Efficiency: {metrics['efficiency']:.1f}%")

        if report["recommendations"]:
            print("\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")

        print(f"\nDetailed report saved to: {self.report_path}")
        print("=" * 60)


def pytest_configure(config):
    """Register the performance reporter plugin."""
    if config.getoption("--performance-report", default=None) is not None:
        config.pluginmanager.register(
            PerformanceReporter(config), "performance_reporter"
        )


def pytest_addoption(parser):
    """Add performance reporting options."""
    parser.addoption(
        "--performance-report",
        default=None,
        help="Generate performance report at the specified path",
    )
