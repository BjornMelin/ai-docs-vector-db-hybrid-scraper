"""Breaking point analysis tests for stress testing.

This module identifies the exact breaking points of the system under various
stress conditions, including maximum throughput, connection limits, memory
thresholds, and response time degradation points.
"""

import random
import statistics

import pytest

from tests.load.conftest import LoadTestConfig, LoadTestType


class TestError(Exception):
    """Custom exception for this module."""


@pytest.mark.stress
@pytest.mark.asyncio
class TestBreakingPointAnalysis:
    """Identify system breaking points under various stress conditions."""

    async def test_throughput_breaking_point(
        self, load_test_runner, mock_load_test_service
    ):
        """Find the maximum sustainable throughput before system breaks."""

        class ThroughputBreakingPointAnalyzer:
            def __init__(self, service):
                self.service = service
                self.breaking_point_data = []

            async def find_throughput_limit(
                self, max_rps: int = 200, step_size: int = 20
            ):
                """Binary search approach to find throughput breaking point."""
                test_results = []

                # Test increasing RPS levels
                for target_rps in range(step_size, max_rps + 1, step_size):
                    users = min(target_rps * 2, 500)  # 2 users per RPS, capped at 500

                    config = LoadTestConfig(
                        test_type=LoadTestType.STRESS,
                        concurrent_users=users,
                        requests_per_second=target_rps,
                        duration_seconds=45.0,
                        success_criteria={
                            "max_error_rate_percent": 20.0,
                            "max_avg_response_time_ms": 5000.0,
                        },
                    )

                    print(f"Testing throughput: {target_rps} RPS with {users} users")

                    result = await load_test_runner.run_load_test(
                        config=config, target_function=self.service.process_request
                    )

                    # Calculate key metrics
                    success_rate = (
                        result.metrics.successful_requests
                        / max(result.metrics.total_requests, 1)
                    ) * 100
                    actual_rps = result.metrics.throughput_rps
                    avg_response_time = (
                        sum(result.metrics.response_times)
                        / len(result.metrics.response_times)
                        if result.metrics.response_times
                        else 0
                    )

                    test_result = {
                        "target_rps": target_rps,
                        "actual_rps": actual_rps,
                        "users": users,
                        "success_rate": success_rate,
                        "avg_response_time_ms": avg_response_time * 1000,
                        "_total_requests": result.metrics.total_requests,
                        "successful_requests": result.metrics.successful_requests,
                        "failed_requests": result.metrics.failed_requests,
                        "system_stable": success_rate >= 80.0
                        and avg_response_time < 2.0,
                    }

                    test_results.append(test_result)

                    # Early break if system is clearly broken
                    if success_rate < 50.0:
                        print(
                            f"System breaking detected at {target_rps} RPS (success rate: {success_rate:.1f}%)"
                        )
                        break

                # Analyze results to find breaking point
                stable_results = [r for r in test_results if r["system_stable"]]
                [r for r in test_results if not r["system_stable"]]

                if stable_results:
                    max_stable_rps = max(r["actual_rps"] for r in stable_results)
                    breaking_point = next(
                        (r for r in test_results if not r["system_stable"]), None
                    )
                else:
                    max_stable_rps = 0
                    breaking_point = test_results[0] if test_results else None

                return {
                    "max_stable_rps": max_stable_rps,
                    "breaking_point": breaking_point,
                    "all_results": test_results,
                }

        analyzer = ThroughputBreakingPointAnalyzer(mock_load_test_service)

        # Find throughput breaking point
        analysis = await analyzer.find_throughput_limit(max_rps=120, step_size=15)

        assert len(analysis["all_results"]) > 0, "No throughput tests were run"

        print("\nThroughput Breaking Point Analysis:")
        print(f"Maximum stable RPS: {analysis['max_stable_rps']:.2f}")

        if analysis["breaking_point"]:
            bp = analysis["breaking_point"]
            print(
                f"Breaking point: {bp['target_rps']} RPS target ({bp['actual_rps']:.2f} actual)"
            )
            print(f"  Success rate: {bp['success_rate']:.1f}%")
            print(f"  Avg response time: {bp['avg_response_time_ms']:.1f}ms")

        # Validate that we found meaningful results
        assert analysis["max_stable_rps"] > 0, "No stable throughput level found"

    async def test_concurrent_users_breaking_point(
        self, load_test_runner, mock_load_test_service
    ):
        """Find the maximum number of concurrent users before system breaks."""

        class ConcurrentUsersBreakingPointAnalyzer:
            def __init__(self, service):
                self.service = service

            async def find_concurrency_limit(
                self, max_users: int = 1000, step_size: int = 50
            ):
                """Find the point where too many concurrent users break the system."""
                test_results = []

                for target_users in range(step_size, max_users + 1, step_size):
                    rps = target_users * 0.5  # 0.5 RPS per user

                    config = LoadTestConfig(
                        test_type=LoadTestType.STRESS,
                        concurrent_users=target_users,
                        requests_per_second=rps,
                        duration_seconds=60.0,
                        success_criteria={
                            "max_error_rate_percent": 25.0,
                            "max_avg_response_time_ms": 8000.0,
                        },
                    )

                    print(f"Testing concurrency: {target_users} users at {rps:.1f} RPS")

                    result = await load_test_runner.run_load_test(
                        config=config, target_function=self.service.process_request
                    )

                    success_rate = (
                        result.metrics.successful_requests
                        / max(result.metrics.total_requests, 1)
                    ) * 100
                    avg_response_time = (
                        sum(result.metrics.response_times)
                        / len(result.metrics.response_times)
                        if result.metrics.response_times
                        else 0
                    )

                    # Calculate response time percentiles
                    if result.metrics.response_times:
                        sorted_times = sorted(result.metrics.response_times)
                        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
                    else:
                        p95_response_time = 0
                        p99_response_time = 0

                    test_result = {
                        "concurrent_users": target_users,
                        "target_rps": rps,
                        "actual_rps": result.metrics.throughput_rps,
                        "success_rate": success_rate,
                        "avg_response_time_ms": avg_response_time * 1000,
                        "p95_response_time_ms": p95_response_time * 1000,
                        "p99_response_time_ms": p99_response_time * 1000,
                        "_total_requests": result.metrics.total_requests,
                        "peak_concurrent": result.metrics.peak_concurrent_users,
                        "system_stable": success_rate >= 75.0
                        and avg_response_time < 3.0,
                    }

                    test_results.append(test_result)

                    # Stop if system is completely broken
                    if success_rate < 30.0:
                        print(
                            f"System failure at {target_users} users (success rate: {success_rate:.1f}%)"
                        )
                        break

                    # Stop if response times are extremely high
                    if avg_response_time > 10.0:  # 10 second average
                        print(
                            f"Unacceptable response times at {target_users} users ({avg_response_time:.2f}s avg)"
                        )
                        break

                # Find breaking point
                stable_results = [r for r in test_results if r["system_stable"]]

                if stable_results:
                    max_stable_users = max(
                        r["concurrent_users"] for r in stable_results
                    )
                    breaking_point = next(
                        (r for r in test_results if not r["system_stable"]), None
                    )
                else:
                    max_stable_users = 0
                    breaking_point = test_results[0] if test_results else None

                return {
                    "max_stable_users": max_stable_users,
                    "breaking_point": breaking_point,
                    "all_results": test_results,
                }

        analyzer = ConcurrentUsersBreakingPointAnalyzer(mock_load_test_service)

        # Find concurrency breaking point
        analysis = await analyzer.find_concurrency_limit(max_users=400, step_size=50)

        assert len(analysis["all_results"]) > 0, "No concurrency tests were run"

        print("\nConcurrent Users Breaking Point Analysis:")
        print(f"Maximum stable users: {analysis['max_stable_users']}")

        if analysis["breaking_point"]:
            bp = analysis["breaking_point"]
            print(f"Breaking point: {bp['concurrent_users']} users")
            print(f"  Success rate: {bp['success_rate']:.1f}%")
            print(f"  Avg response time: {bp['avg_response_time_ms']:.1f}ms")
            print(f"  P95 response time: {bp['p95_response_time_ms']:.1f}ms")

        # Print summary of all tests
        print("\nConcurrency test progression:")
        for result in analysis["all_results"]:
            stability = "✓ STABLE" if result["system_stable"] else "✗ UNSTABLE"
            print(
                f"  {result['concurrent_users']:3d} users: {result['success_rate']:5.1f}% success, {result['avg_response_time_ms']:6.1f}ms avg - {stability}"
            )

    async def test_response_time_degradation_point(
        self, load_test_runner, mock_load_test_service
    ):
        """Find the point where response times degrade unacceptably."""

        class ResponseTimeDegradationAnalyzer:
            def __init__(self, service):
                self.service = service

            async def analyze_response_time_degradation(self):
                """Analyze how response times degrade under increasing load."""

                # Progressive load tests to see response time degradation
                load_levels = [
                    {"users": 10, "rps": 5, "name": "baseline"},
                    {"users": 25, "rps": 15, "name": "light_load"},
                    {"users": 50, "rps": 30, "name": "moderate_load"},
                    {"users": 100, "rps": 60, "name": "heavy_load"},
                    {"users": 200, "rps": 120, "name": "stress_load"},
                    {"users": 400, "rps": 200, "name": "extreme_load"},
                ]

                degradation_results = []
                baseline_response_time = None

                for load_level in load_levels:
                    config = LoadTestConfig(
                        test_type=LoadTestType.STRESS,
                        concurrent_users=load_level["users"],
                        requests_per_second=load_level["rps"],
                        duration_seconds=45.0,
                        success_criteria={
                            "max_error_rate_percent": 30.0,
                        },
                    )

                    print(
                        f"Testing {load_level['name']}: {load_level['users']} users, {load_level['rps']} RPS"
                    )

                    result = await load_test_runner.run_load_test(
                        config=config, target_function=self.service.process_request
                    )

                    if result.metrics.response_times:
                        response_times = result.metrics.response_times
                        avg_response_time = statistics.mean(response_times)
                        median_response_time = statistics.median(response_times)

                        sorted_times = sorted(response_times)
                        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
                        max_response_time = max(response_times)
                        min_response_time = min(response_times)

                        # Calculate coefficient of variation (measure of variability)
                        stdev_response_time = (
                            statistics.stdev(response_times)
                            if len(response_times) > 1
                            else 0
                        )
                        cv = (
                            (stdev_response_time / avg_response_time)
                            if avg_response_time > 0
                            else 0
                        )
                    else:
                        avg_response_time = 0
                        median_response_time = 0
                        p95_response_time = 0
                        p99_response_time = 0
                        max_response_time = 0
                        min_response_time = 0
                        cv = 0

                    success_rate = (
                        result.metrics.successful_requests
                        / max(result.metrics.total_requests, 1)
                    ) * 100

                    # Set baseline for comparison
                    if load_level["name"] == "baseline":
                        baseline_response_time = avg_response_time

                    # Calculate degradation factor
                    degradation_factor = (
                        (avg_response_time / baseline_response_time)
                        if baseline_response_time > 0
                        else 1
                    )

                    degradation_result = {
                        "load_level": load_level["name"],
                        "users": load_level["users"],
                        "rps": load_level["rps"],
                        "avg_response_time_ms": avg_response_time * 1000,
                        "median_response_time_ms": median_response_time * 1000,
                        "p95_response_time_ms": p95_response_time * 1000,
                        "p99_response_time_ms": p99_response_time * 1000,
                        "max_response_time_ms": max_response_time * 1000,
                        "min_response_time_ms": min_response_time * 1000,
                        "response_time_cv": cv,
                        "degradation_factor": degradation_factor,
                        "success_rate": success_rate,
                        "_total_requests": result.metrics.total_requests,
                        "acceptable_performance": avg_response_time < 2.0
                        and success_rate >= 90.0,
                    }

                    degradation_results.append(degradation_result)

                    # Stop if response times become completely unacceptable
                    if avg_response_time > 30.0:  # 30 second average
                        print(
                            f"Stopping test due to extreme response times: {avg_response_time:.2f}s"
                        )
                        break

                # Analyze degradation pattern
                acceptable_results = [
                    r for r in degradation_results if r["acceptable_performance"]
                ]
                unacceptable_results = [
                    r for r in degradation_results if not r["acceptable_performance"]
                ]

                if acceptable_results:
                    last_acceptable = max(acceptable_results, key=lambda x: x["users"])
                else:
                    last_acceptable = None

                if unacceptable_results:
                    first_unacceptable = min(
                        unacceptable_results, key=lambda x: x["users"]
                    )
                else:
                    first_unacceptable = None

                return {
                    "baseline_response_time_ms": baseline_response_time * 1000
                    if baseline_response_time
                    else 0,
                    "last_acceptable_load": last_acceptable,
                    "first_unacceptable_load": first_unacceptable,
                    "all_results": degradation_results,
                }

        analyzer = ResponseTimeDegradationAnalyzer(mock_load_test_service)

        # Analyze response time degradation
        analysis = await analyzer.analyze_response_time_degradation()

        assert len(analysis["all_results"]) > 0, "No response time tests were run"

        print("\nResponse Time Degradation Analysis:")
        print(f"Baseline response time: {analysis['baseline_response_time_ms']:.1f}ms")

        if analysis["last_acceptable_load"]:
            last_ok = analysis["last_acceptable_load"]
            print(
                f"Last acceptable load: {last_ok['users']} users ({last_ok['avg_response_time_ms']:.1f}ms avg)"
            )

        if analysis["first_unacceptable_load"]:
            first_bad = analysis["first_unacceptable_load"]
            print(
                f"First unacceptable load: {first_bad['users']} users ({first_bad['avg_response_time_ms']:.1f}ms avg)"
            )
            print(f"Degradation factor: {first_bad['degradation_factor']:.1f}x")

        print("\nResponse time progression:")
        for result in analysis["all_results"]:
            performance = "✓ GOOD" if result["acceptable_performance"] else "✗ POOR"
            print(
                f"  {result['load_level']:15s}: avg={result['avg_response_time_ms']:6.1f}ms, "
                f"p95={result['p95_response_time_ms']:6.1f}ms, degradation={result['degradation_factor']:4.1f}x - {performance}"
            )

    async def test_error_cascade_breaking_point(
        self, load_test_runner, mock_load_test_service
    ):
        """Find the point where errors cascade and cause complete system failure."""

        class ErrorCascadeAnalyzer:
            def __init__(self, service):
                self.service = service
                self.error_cascade_level = 0.0

            async def error_prone_task(self, **_kwargs):
                """Task that becomes more error-prone under load."""
                # Error probability increases with load and cascade level
                load_factor = min(1.0, self.service.request_count / 500.0)
                base_error_rate = 0.02  # 2% base error rate
                cascade_error_rate = self.error_cascade_level * 0.1

                _total_error_rate = min(
                    0.8, base_error_rate + load_factor * 0.15 + cascade_error_rate
                )

                if random.random() < _total_error_rate:
                    # Increase cascade level on error
                    msg = f"Cascading error (cascade level: {self.error_cascade_level:.1f}, load: {load_factor:.2f})"
                    raise TestError(msg)
                # Decrease cascade level on success
                self.error_cascade_level = max(0.0, self.error_cascade_level - 0.05)

                result = await self.service.process_request(**_kwargs)
                result["cascade_level"] = self.error_cascade_level
                result["load_factor"] = load_factor
                return result

            async def find_error_cascade_point(self):
                """Find the load level that triggers error cascading."""
                test_levels = [
                    {"users": 20, "rps": 10, "name": "stable"},
                    {"users": 50, "rps": 30, "name": "moderate"},
                    {"users": 100, "rps": 70, "name": "heavy"},
                    {"users": 200, "rps": 150, "name": "overload"},
                    {"users": 400, "rps": 300, "name": "extreme"},
                ]

                cascade_results = []

                for level in test_levels:
                    # Reset cascade level for each test
                    initial_cascade = self.error_cascade_level

                    config = LoadTestConfig(
                        test_type=LoadTestType.STRESS,
                        concurrent_users=level["users"],
                        requests_per_second=level["rps"],
                        duration_seconds=60.0,
                        success_criteria={
                            "max_error_rate_percent": 90.0,  # Very lenient - we expect errors
                        },
                    )

                    print(
                        f"Testing error cascade at {level['name']}: {level['users']} users, {level['rps']} RPS"
                    )

                    result = await load_test_runner.run_load_test(
                        config=config, target_function=self.error_prone_task
                    )

                    final_cascade = self.error_cascade_level
                    success_rate = (
                        result.metrics.successful_requests
                        / max(result.metrics.total_requests, 1)
                    ) * 100
                    error_rate = (
                        result.metrics.failed_requests
                        / max(result.metrics.total_requests, 1)
                    ) * 100

                    cascade_result = {
                        "level": level["name"],
                        "users": level["users"],
                        "rps": level["rps"],
                        "_total_requests": result.metrics.total_requests,
                        "success_rate": success_rate,
                        "error_rate": error_rate,
                        "initial_cascade_level": initial_cascade,
                        "final_cascade_level": final_cascade,
                        "cascade_progression": final_cascade - initial_cascade,
                        "cascade_triggered": final_cascade > 2.0,
                        "system_failed": success_rate < 20.0,
                    }

                    cascade_results.append(cascade_result)

                    # Stop if system has completely failed
                    if success_rate < 10.0:
                        print(
                            f"System complete failure at {level['name']} ({success_rate:.1f}% success rate)"
                        )
                        break

                # Find cascade breaking point
                cascade_triggered = [
                    r for r in cascade_results if r["cascade_triggered"]
                ]
                system_failed = [r for r in cascade_results if r["system_failed"]]

                return {
                    "cascade_results": cascade_results,
                    "cascade_breaking_point": cascade_triggered[0]
                    if cascade_triggered
                    else None,
                    "system_failure_point": system_failed[0] if system_failed else None,
                }

        analyzer = ErrorCascadeAnalyzer(mock_load_test_service)

        # Find error cascade breaking point
        analysis = await analyzer.find_error_cascade_point()

        assert len(analysis["cascade_results"]) > 0, "No error cascade tests were run"

        print("\nError Cascade Breaking Point Analysis:")

        if analysis["cascade_breaking_point"]:
            cbp = analysis["cascade_breaking_point"]
            print(f"Cascade triggered at: {cbp['level']} ({cbp['users']} users)")
            print(f"  Error rate: {cbp['error_rate']:.1f}%")
            print(
                f"  Cascade level: {cbp['initial_cascade_level']:.1f} → {cbp['final_cascade_level']:.1f}"
            )

        if analysis["system_failure_point"]:
            sfp = analysis["system_failure_point"]
            print(f"System failure at: {sfp['level']} ({sfp['users']} users)")
            print(f"  Success rate: {sfp['success_rate']:.1f}%")

        print("\nError cascade progression:")
        for result in analysis["cascade_results"]:
            cascade_status = "CASCADING" if result["cascade_triggered"] else "STABLE"
            failure_status = "FAILED" if result["system_failed"] else "OPERATIONAL"
            print(
                f"  {result['level']:10s}: {result['success_rate']:5.1f}% success, "
                f"cascade {result['initial_cascade_level']:4.1f}→{result['final_cascade_level']:4.1f}, "
                f"{cascade_status}, {failure_status}"
            )

        # Validate that we detected meaningful patterns
        assert (
            len([r for r in analysis["cascade_results"] if r["_total_requests"] > 0])
            > 0
        ), "No meaningful test results"
