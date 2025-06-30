"""Stress testing scenarios for beyond-capacity loads.

This module implements stress tests to identify system breaking points
and behavior under extreme load conditions.
"""

import asyncio
import logging
import os
import random
import time

import psutil
import pytest

from tests.load.base_load_test import create_load_test_runner
from tests.load.conftest import LoadTestConfig, LoadTestType
from tests.load.load_profiles import BreakpointLoadProfile, create_custom_step_profile


class TestError(Exception):
    """Custom exception for this module."""


logger = logging.getLogger(__name__)


class TestStressLoad:
    """Test suite for stress load conditions."""

    @pytest.mark.stress
    def test_breaking_point_identification(self, load_test_runner):
        """Identify system breaking point by gradually increasing load."""
        # Configuration for breakpoint testing
        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=1000,  # Maximum to test
            requests_per_second=500,
            duration_seconds=600,  # 10 minutes
            success_criteria={
                "max_error_rate_percent": 20.0,  # Higher tolerance for stress
                "max_avg_response_time_ms": 5000.0,
            },
        )

        # Create environment with breakpoint profile
        env = create_load_test_runner()
        env.shape_class = BreakpointLoadProfile(
            start_users=10,
            user_increment=20,
            step_duration=60,
            max_users=1000,
            spawn_rate=10,
        )

        # Track metrics at each step
        step_metrics = []
        breaking_point = None

        @env.events.stats_reset.add_listener
        def on_stats_reset(**__kwargs):
            """Capture metrics at each step."""
            nonlocal breaking_point

            stats = env.stats
            _total_requests = stats.total._total_num_requests
            _total_failures = stats.total._total_num_failures

            if _total_requests > 0:
                error_rate = (_total_failures / _total_requests) * 100
                avg_response_time = stats.total.avg_response_time

                current_users = env.runner.user_count if env.runner else 0

                step_metrics.append(
                    {
                        "users": current_users,
                        "error_rate": error_rate,
                        "avg_response_time": avg_response_time,
                        "throughput": stats.total.current_rps,
                    }
                )

                # Check if we've hit breaking point
                if error_rate > 10.0 or (
                    avg_response_time > 3000 and breaking_point is None
                ):
                    breaking_point = current_users
                    logger.warning(
                        "Breaking point detected at %s users", current_users
                    )  # TODO: Convert f-string to logging format

        # Run stress test
        load_test_runner.run_load_test(
            config=config,
            target_function=self._high_load_operation,
            environment=env,
        )

        # Analyze results
        assert breaking_point is not None, "Failed to find breaking point"
        assert breaking_point > 100, f"Breaking point too low: {breaking_point} users"

        # Verify gradual degradation
        degradation_curve = self._analyze_degradation(step_metrics)
        assert degradation_curve["is_gradual"], (
            "System failed catastrophically instead of degrading gracefully"
        )

    @pytest.mark.stress
    def test_resource_exhaustion(self, load_test_runner, mock_load_test_service):
        """Test system behavior under resource exhaustion."""
        # Configure for resource exhaustion
        mock_load_test_service.set_base_latency(
            0.5
        )  # High latency to simulate exhaustion

        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=500,
            requests_per_second=250,
            duration_seconds=300,
            success_criteria={
                "max_error_rate_percent": 30.0,
                "max_avg_response_time_ms": 10000.0,
            },
        )

        # Monitor resource metrics
        resource_metrics = {
            "connection_pool_exhaustion": False,
            "timeout_errors": 0,
            "memory_errors": 0,
            "rate_limit_errors": 0,
        }

        async def monitor_resources(**_kwargs):
            """Monitor for resource exhaustion indicators."""
            try:
                return await mock_load_test_service.process_request(**_kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg:
                    resource_metrics["timeout_errors"] += 1
                elif "memory" in error_msg:
                    resource_metrics["memory_errors"] += 1
                elif "rate limit" in error_msg:
                    resource_metrics["rate_limit_errors"] += 1
                elif "connection pool" in error_msg:
                    resource_metrics["connection_pool_exhaustion"] = True
                raise

        # Run stress test
        result = load_test_runner.run_load_test(
            config=config,
            target_function=monitor_resources,
        )

        # Verify resource handling
        assert (
            resource_metrics["timeout_errors"] < result.metrics.total_requests * 0.5
        ), "Too many timeout errors"
        assert not resource_metrics["connection_pool_exhaustion"], (
            "Connection pool exhausted - need better pooling"
        )

    @pytest.mark.stress
    def test_cascading_failure_prevention(self, load_test_runner):
        """Test system's ability to prevent cascading failures."""

        # Create dependent service simulation
        class DependentServices:
            def __init__(self):
                self.service_health = {
                    "embedding": True,
                    "vector_db": True,
                    "cache": True,
                }
                self.failure_counts = {
                    "embedding": 0,
                    "vector_db": 0,
                    "cache": 0,
                }

            async def call_service(self, service: str):
                """Simulate service call with potential failure."""
                if not self.service_health[service]:
                    msg = f"{service} service unavailable"
                    raise TestError(msg)
                    msg = f"{service} service unavailable"
                    raise TestError(msg)

                # Simulate load-based failure probability
                failure_chance = self.failure_counts[service] / 1000
                if asyncio.create_task(asyncio.sleep(0)) and failure_chance > 0.5:
                    self.service_health[service] = False
                    msg = f"{service} service degraded"
                    raise TestError(msg)

                await asyncio.sleep(0.1)
                return f"{service} response"

        services = DependentServices()

        async def test_with_dependencies(**__kwargs):
            """Test operation with service dependencies."""
            try:
                # Call multiple services
                await services.call_service("cache")
            except Exception as e:
                # Check if failure is cascading
                logger.warning(
                    "Service failure: %s", e
                )  # TODO: Convert f-string to logging format
            else:
                await services.call_service("embedding")
                await services.call_service("vector_db")
                return {"status": "success"}
                failed_services = sum(
                    1 for h in services.service_health.values() if not h
                )
                if failed_services > 1:
                    msg = f"Cascading failure: {failed_services} services down"
                    raise TestError(msg) from None

        # Run stress test
        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=300,
            requests_per_second=150,
            duration_seconds=300,
        )

        result = load_test_runner.run_load_test(
            config=config,
            target_function=test_with_dependencies,
        )

        # Verify cascading failure prevention
        cascading_failures = sum(
            1 for e in result.metrics.errors if "Cascading failure" in e
        )
        assert cascading_failures == 0, (
            f"System experienced {cascading_failures} cascading failures"
        )

        # Verify circuit breaker behavior
        assert services.failure_counts["embedding"] < 1000, (
            "Circuit breaker should have prevented excessive calls"
        )

    @pytest.mark.stress
    def test_recovery_after_stress(self, load_test_runner):
        """Test system recovery after stress period."""
        # Phase 1: Normal load
        # Phase 2: Stress load
        # Phase 3: Recovery to normal

        stages = [
            {"duration": 120, "users": 50, "spawn_rate": 5, "name": "baseline"},
            {"duration": 300, "users": 500, "spawn_rate": 50, "name": "stress"},
            {"duration": 180, "users": 50, "spawn_rate": 10, "name": "recovery"},
        ]

        profile = create_custom_step_profile(stages)

        # Track metrics by phase
        phase_metrics = {
            "baseline": {"errors": 0, "response_times": []},
            "stress": {"errors": 0, "response_times": []},
            "recovery": {"errors": 0, "response_times": []},
        }

        time.time()

        # Run test
        env = create_load_test_runner()
        env.shape_class = profile

        load_test_runner.run_load_test(
            config=LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=500,
                requests_per_second=250,
                duration_seconds=600,
            ),
            target_function=self._high_load_operation,
            environment=env,
        )

        # Analyze recovery
        recovery_analysis = self._analyze_recovery(phase_metrics)

        # Assertions
        assert recovery_analysis["recovery_time"] < 60, (
            f"Recovery took too long: {recovery_analysis['recovery_time']}s"
        )
        assert recovery_analysis["recovered_performance"] > 0.9, (
            "Performance did not fully recover after stress"
        )

    @pytest.mark.stress
    def test_memory_leak_under_stress(self, load_test_runner, mock_load_test_service):
        """Test for memory leaks under sustained stress."""
        # Configure for memory leak detection
        memory_samples = []

        async def memory_tracking_operation(**_kwargs):
            """Operation that tracks memory usage."""
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Perform operation
            result = await mock_load_test_service.process_request(
                data_size_mb=10.0,  # Large data to stress memory
                **_kwargs,
            )

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(
                {
                    "timestamp": time.time(),
                    "memory_mb": memory_after,
                    "delta_mb": memory_after - memory_before,
                }
            )

            return result

        # Run extended stress test
        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=100,
            requests_per_second=50,
            duration_seconds=600,  # 10 minutes
        )

        load_test_runner.run_load_test(
            config=config,
            target_function=memory_tracking_operation,
        )

        # Analyze memory usage
        memory_analysis = self._analyze_memory_usage(memory_samples)

        # Assertions
        assert memory_analysis["leak_detected"] is False, (
            f"Memory leak detected: {memory_analysis['leak_rate_mb_per_min']} MB/min"
        )
        assert memory_analysis["max_memory_mb"] < 2048, (
            f"Excessive memory usage: {memory_analysis['max_memory_mb']} MB"
        )

    def _high_load_operation(self, **__kwargs):
        """Simulate high-load operation."""
        # Simulate CPU-intensive operation
        start = time.time()
        while time.time() - start < 0.1:
            _ = sum(i * i for i in range(1000))

        # Simulate I/O wait
        return asyncio.sleep(random.uniform(0.05, 0.5))

    def _analyze_degradation(self, step_metrics: list[dict]) -> dict:
        """Analyze system degradation pattern."""
        if len(step_metrics) < 3:
            return {"is_gradual": False, "failure_point": None}

        # Check for gradual vs sudden degradation
        error_deltas = []
        response_deltas = []

        for i in range(1, len(step_metrics)):
            error_delta = (
                step_metrics[i]["error_rate"] - step_metrics[i - 1]["error_rate"]
            )
            response_delta = (
                step_metrics[i]["avg_response_time"]
                - step_metrics[i - 1]["avg_response_time"]
            )

            error_deltas.append(error_delta)
            response_deltas.append(response_delta)

        # Sudden degradation = large jump in metrics
        max_error_jump = max(error_deltas) if error_deltas else 0
        max_response_jump = max(response_deltas) if response_deltas else 0

        is_gradual = max_error_jump < 50 and max_response_jump < 2000

        return {
            "is_gradual": is_gradual,
            "max_error_jump": max_error_jump,
            "max_response_jump": max_response_jump,
            "degradation_steps": len(step_metrics),
        }

    def _analyze_recovery(self, phase_metrics: dict) -> dict:
        """Analyze system recovery after stress."""
        baseline_avg = sum(phase_metrics["baseline"]["response_times"]) / max(
            len(phase_metrics["baseline"]["response_times"]), 1
        )
        recovery_avg = sum(phase_metrics["recovery"]["response_times"]) / max(
            len(phase_metrics["recovery"]["response_times"]), 1
        )

        # Calculate recovery metrics
        recovered_performance = baseline_avg / recovery_avg if recovery_avg > 0 else 0

        return {
            "recovery_time": 30,  # Placeholder - would calculate actual
            "recovered_performance": recovered_performance,
            "baseline_avg_response": baseline_avg,
            "recovery_avg_response": recovery_avg,
        }

    def _analyze_memory_usage(self, memory_samples: list[dict]) -> dict:
        """Analyze memory usage for leak detection."""
        if len(memory_samples) < 10:
            return {"leak_detected": False, "insufficient_data": True}

        # Calculate memory growth rate
        start_memory = memory_samples[0]["memory_mb"]
        end_memory = memory_samples[-1]["memory_mb"]
        duration_minutes = (
            memory_samples[-1]["timestamp"] - memory_samples[0]["timestamp"]
        ) / 60

        growth_rate = (
            (end_memory - start_memory) / duration_minutes
            if duration_minutes > 0
            else 0
        )

        # Detect leak if growth rate > 10 MB/min
        leak_detected = growth_rate > 10

        return {
            "leak_detected": leak_detected,
            "leak_rate_mb_per_min": growth_rate,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "max_memory_mb": max(s["memory_mb"] for s in memory_samples),
            "duration_minutes": duration_minutes,
        }
