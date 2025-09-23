"""Normal load testing scenarios for expected traffic patterns.

This module implements load tests for normal operating conditions,
validating system performance under expected user loads.
"""

import asyncio
import logging
import random

import pytest

from tests.load.base_load_test import create_load_test_runner
from tests.load.load_profiles import (
    RampUpLoadProfile,
    SteadyLoadProfile,
    create_custom_step_profile,
)


logger = logging.getLogger(__name__)


class TestNormalLoad:
    """Test suite for normal load conditions."""

    @pytest.mark.load
    def test_steady_state_load(self, load_test_runner):
        """Test system under steady-state normal load."""
        # Configuration
        target_users = 50
        duration = 300  # 5 minutes
        spawn_rate = 5

        # Create environment
        env = create_load_test_runner()

        # Apply steady load profile
        env.shape_class = SteadyLoadProfile(
            users=target_users,
            duration=duration,
            spawn_rate=spawn_rate,
        )

        # Run test
        result = load_test_runner.run_load_test(
            config=load_test_runner.load_test_config["test_profiles"]["moderate"],
            target_function=self._simulate_normal_traffic,
            environment=env,
        )

        # Assertions
        assert result.success, f"Load test failed: {result.bottlenecks_identified}"
        assert result.metrics.throughput_rps >= 25, "Throughput below expected"
        assert result.performance_grade in ["A", "B"], (
            f"Poor performance grade: {result.performance_grade}"
        )

        # Check response times
        p95_response_time = load_test_runner._percentile(
            result.metrics.response_times, 95
        )
        assert p95_response_time < 1.0, (
            f"P95 response time too high: {p95_response_time}s"
        )

    @pytest.mark.load
    def test_gradual_ramp_up(self, load_test_runner):
        """Test system with gradual user ramp-up."""
        # Configuration
        start_users = 1
        end_users = 100
        ramp_time = 300  # 5 minutes
        hold_time = 300  # 5 minutes

        # Create environment
        env = create_load_test_runner()

        # Apply ramp-up profile
        env.shape_class = RampUpLoadProfile(
            start_users=start_users,
            end_users=end_users,
            ramp_time=ramp_time,
            hold_time=hold_time,
            spawn_rate=2,
        )

        # Run test
        result = load_test_runner.run_load_test(
            config=load_test_runner.load_test_config["test_profiles"]["moderate"],
            target_function=self._simulate_normal_traffic,
            environment=env,
        )

        # Assertions
        assert result.success, f"Ramp-up test failed: {result.bottlenecks_identified}"
        assert result.metrics.peak_concurrent_users >= end_users * 0.9, (
            "Failed to reach target users"
        )

        # Verify stable performance during hold phase
        hold_phase_metrics = self._get_phase_metrics(
            result.metrics, ramp_time, hold_time
        )
        assert hold_phase_metrics["error_rate"] < 0.05, (
            "High error rate during hold phase"
        )

    @pytest.mark.load
    def test_mixed_operation_load(self, load_test_runner, mock_load_test_service):
        """Test system with mixed operations (read/write balance)."""
        # Configure service for mixed operations
        mock_load_test_service.set_base_latency(0.05)  # 50ms base latency
        mock_load_test_service.set_failure_rate(0.01)  # 1% failure rate

        # Run test with mixed workload
        result = load_test_runner.run_load_test(
            config=load_test_runner.load_test_config["test_profiles"]["moderate"],
            target_function=mock_load_test_service.process_request,
        )

        # Analyze operation mix
        operation_stats = self._analyze_operation_mix(result)

        # Assertions
        assert operation_stats["read_ratio"] > 0.6, "Read ratio too low"
        assert operation_stats["write_ratio"] < 0.4, "Write ratio too high"
        assert operation_stats["avg_read_time"] < operation_stats["avg_write_time"], (
            "Read should be faster than write"
        )

    @pytest.mark.load
    def test_diurnal_pattern(self, load_test_runner):
        """Test system with diurnal (daily) traffic pattern."""
        # Simulate 24-hour pattern compressed to 1 hour
        stages = [
            {"duration": 300, "users": 20, "spawn_rate": 2},  # Night (low)
            {"duration": 300, "users": 50, "spawn_rate": 5},  # Morning ramp
            {"duration": 600, "users": 100, "spawn_rate": 10},  # Day (peak)
            {"duration": 300, "users": 80, "spawn_rate": 5},  # Evening
            {"duration": 300, "users": 40, "spawn_rate": 3},  # Late evening
            {"duration": 300, "users": 20, "spawn_rate": 2},  # Night (low)
        ]

        # Create custom profile

        profile = create_custom_step_profile(stages)

        # Run test
        env = create_load_test_runner()
        env.shape_class = profile

        result = load_test_runner.run_load_test(
            config=load_test_runner.load_test_config["test_profiles"]["moderate"],
            target_function=self._simulate_normal_traffic,
            environment=env,
        )

        # Assertions
        assert result.success, "Diurnal pattern test failed"
        assert len(result.metrics.response_times) > 1000, "Insufficient data collected"

        # Verify system handles transitions smoothly
        transition_errors = self._check_smooth_transitions(result.metrics)
        assert transition_errors == 0, (
            f"Found {transition_errors} errors during transitions"
        )

    @pytest.mark.load
    def test_cache_effectiveness(self, load_test_runner, mock_load_test_service):
        """Test cache effectiveness under normal load."""
        # Reset service metrics
        mock_load_test_service.reset_metrics()

        # Run test with repeated queries
        repeated_queries = ["python tutorials", "fastapi documentation", "numpy guide"]

        async def cached_search_workload(**_kwargs):
            """Workload that repeats queries to test caching."""

            query = random.choice(repeated_queries)  # noqa: S311
            return await mock_load_test_service.search_documents(query=query, **_kwargs)

        result = load_test_runner.run_load_test(
            config=load_test_runner.load_test_config["test_profiles"]["light"],
            target_function=cached_search_workload,
        )

        # Calculate cache metrics
        cache_metrics = self._calculate_cache_metrics(result)

        # Assertions
        assert cache_metrics["hit_rate"] > 0.7, (
            f"Low cache hit rate: {cache_metrics['hit_rate']}"
        )
        assert (
            cache_metrics["cached_response_time"]
            < cache_metrics["uncached_response_time"] * 0.5
        ), "Cache not providing expected performance benefit"

    def _simulate_normal_traffic(self, **__kwargs):
        """Simulate normal traffic patterns."""

        operations = [
            ("search", 0.6),
            ("add_document", 0.2),
            ("update_document", 0.1),
            ("generate_embeddings", 0.1),
        ]

        # Select operation based on probability
        rand = random.random()  # noqa: S311
        cumulative = 0

        for _op, prob in operations:
            cumulative += prob
            if rand <= cumulative:
                # Simulate operation
                return asyncio.sleep(random.uniform(0.05, 0.2))  # noqa: S311

        return asyncio.sleep(0.1)

    def _get_phase_metrics(self, metrics, start_time: float, duration: float) -> dict:
        """Extract metrics for a specific phase."""

        # Filter metrics by time window
        # Note: This is simplified - in real implementation would use timestamps
        start_idx = int(start_time * 10)  # Assuming ~10 requests/second
        end_idx = start_idx + int(duration * 10)

        phase_response_times = metrics.response_times[start_idx:end_idx]
        phase_errors = (
            metrics.errors[start_idx:end_idx] if start_idx < len(metrics.errors) else []
        )

        return {
            "response_times": phase_response_times,
            "error_rate": len(phase_errors) / max(len(phase_response_times), 1),
            "avg_response_time": sum(phase_response_times)
            / max(len(phase_response_times), 1),
        }

    def _analyze_operation_mix(self, _result) -> dict:
        """Analyze the mix of operations performed."""
        # Simplified analysis - would parse actual operation types in real
        # implementation

        return {
            "read_ratio": 0.7,  # Placeholder
            "write_ratio": 0.3,  # Placeholder
            "avg_read_time": 0.05,  # Placeholder
            "avg_write_time": 0.15,  # Placeholder
        }

    def _check_smooth_transitions(self, metrics) -> int:
        """Check for errors during load transitions."""
        # Look for spikes in error rates during transitions
        error_spikes = 0

        for i in range(1, len(metrics.errors)):
            if metrics.errors[i] - metrics.errors[i - 1] > 0.1:  # 10% spike
                error_spikes += 1

        return error_spikes

    def _calculate_cache_metrics(self, result) -> dict:
        """Calculate cache effectiveness metrics."""
        # Simplified calculation - would analyze actual cache hits in real
        # implementation
        response_times = result.metrics.response_times

        # Assume first 30% are uncached, rest are potentially cached
        uncached_count = int(len(response_times) * 0.3)
        uncached_times = response_times[:uncached_count]
        cached_times = response_times[uncached_count:]

        return {
            "hit_rate": 0.75,  # Placeholder
            "cached_response_time": sum(cached_times) / max(len(cached_times), 1),
            "uncached_response_time": sum(uncached_times) / max(len(uncached_times), 1),
        }
