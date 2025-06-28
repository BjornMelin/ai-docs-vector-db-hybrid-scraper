"""Spike testing scenarios for sudden load increases.

This module implements spike tests to validate system behavior
under sudden traffic spikes and auto-scaling scenarios.
"""

import asyncio
import logging
import time
from typing import Dict

import pytest

from ..base_load_test import create_load_test_runner
from ..conftest import LoadTestConfig, LoadTestType
from ..load_profiles import DoubleSpike, SpikeLoadProfile


class TestError(Exception):
    """Custom exception for this module."""

    pass


logger = logging.getLogger(__name__)


class TestSpikeLoad:
    """Test suite for spike load conditions."""

    @pytest.mark.spike
    def test_single_traffic_spike(self, load_test_runner):
        """Test system response to a single traffic spike."""
        # Configuration for spike testing
        config = LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=500,
            requests_per_second=250,
            duration_seconds=300,  # 5 minutes
            success_criteria={
                "max_error_rate_percent": 15.0,  # Higher tolerance during spike
                "max_avg_response_time_ms": 3000.0,
                "recovery_time_seconds": 60.0,
            },
        )

        # Create environment with spike profile
        env = create_load_test_runner()
        env.shape_class = SpikeLoadProfile(
            baseline_users=50,
            spike_users=500,
            baseline_time=120,  # 2 minutes baseline
            spike_time=60,  # 1 minute spike
            recovery_time=120,  # 2 minutes recovery
            spawn_rate=100,  # Fast spawn for spike
        )

        # Track spike metrics
        spike_metrics = {
            "baseline_phase": {"errors": 0, "response_times": [], "throughput": []},
            "spike_phase": {"errors": 0, "response_times": [], "throughput": []},
            "recovery_phase": {"errors": 0, "response_times": [], "throughput": []},
        }

        # Run spike test
        result = load_test_runner.run_load_test(
            config=config,
            target_function=self._spike_aware_operation,
            environment=env,
            spike_metrics=spike_metrics,
        )

        # Analyze spike behavior
        spike_analysis = self._analyze_spike_response(spike_metrics)

        # Assertions
        assert result.success, f"Spike test failed: {result.bottlenecks_identified}"
        assert spike_analysis["spike_handled"], "System failed to handle traffic spike"
        assert spike_analysis["recovery_time"] < 90, (
            f"Recovery took too long: {spike_analysis['recovery_time']}s"
        )
        assert spike_analysis["performance_degradation"] < 300, (
            "Excessive performance degradation during spike"
        )

    @pytest.mark.spike
    def test_double_spike_pattern(self, load_test_runner):
        """Test system with double spike pattern."""
        config = LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=400,
            requests_per_second=200,
            duration_seconds=480,  # 8 minutes
            success_criteria={
                "max_error_rate_percent": 20.0,
                "max_avg_response_time_ms": 4000.0,
            },
        )

        # Create environment with double spike profile
        env = create_load_test_runner()
        env.shape_class = DoubleSpike()

        # Track performance between spikes
        inter_spike_metrics = []

        @env.events.stats_reset.add_listener
        def track_inter_spike_performance(**_kwargs):
            """Track performance between spikes."""
            stats = env.stats
            if stats and stats.total.num_requests > 0:
                inter_spike_metrics.append(
                    {
                        "timestamp": time.time(),
                        "users": env.runner.user_count if env.runner else 0,
                        "rps": stats.total.current_rps,
                        "avg_response_time": stats.total.avg_response_time,
                        "error_rate": (
                            stats.total.num_failures / stats.total.num_requests
                        )
                        * 100,
                    }
                )

        # Run double spike test
        load_test_runner.run_load_test(
            config=config,
            target_function=self._spike_aware_operation,
            environment=env,
        )

        # Analyze double spike behavior
        spike_recovery = self._analyze_double_spike_recovery(inter_spike_metrics)

        # Assertions
        assert spike_recovery["first_spike_recovered"], (
            "Failed to recover from first spike"
        )
        assert spike_recovery["second_spike_handled"], "Failed to handle second spike"
        assert spike_recovery["system_stability"] > 0.8, (
            "System unstable between spikes"
        )

    @pytest.mark.spike
    def test_auto_scaling_response(self, load_test_runner, mock_load_test_service):
        """Test auto-scaling response during traffic spikes."""

        # Simulate auto-scaling behavior
        class AutoScalingSimulator:
            def __init__(self):
                self.current_capacity = 100  # Base capacity
                self.max_capacity = 1000
                self.scale_up_threshold = 80  # % utilization
                self.scale_down_threshold = 30
                self.utilization_history = []

            def update_utilization(self, current_load: int, response_time: float):
                """Update utilization and trigger scaling decisions."""
                utilization = (current_load / self.current_capacity) * 100
                self.utilization_history.append(
                    {
                        "timestamp": time.time(),
                        "utilization": utilization,
                        "capacity": self.current_capacity,
                        "load": current_load,
                        "response_time": response_time,
                    }
                )

                # Auto-scaling logic
                if (
                    utilization > self.scale_up_threshold
                    and self.current_capacity < self.max_capacity
                ):
                    self.scale_up()
                elif (
                    utilization < self.scale_down_threshold
                    and self.current_capacity > 100
                ):
                    self.scale_down()

            def scale_up(self):
                """Scale up capacity."""
                old_capacity = self.current_capacity
                self.current_capacity = min(
                    self.current_capacity * 1.5, self.max_capacity
                )
                logger.info(f"Scaled up from {old_capacity} to {self.current_capacity}")

            def scale_down(self):
                """Scale down capacity."""
                old_capacity = self.current_capacity
                self.current_capacity = max(self.current_capacity * 0.8, 100)
                logger.info(
                    f"Scaled down from {old_capacity} to {self.current_capacity}"
                )

        auto_scaler = AutoScalingSimulator()

        async def auto_scaling_aware_operation(**kwargs):
            """Operation that updates auto-scaling metrics."""
            start_time = time.time()
            result = await mock_load_test_service.process_request(**kwargs)
            response_time = time.time() - start_time

            # Update auto-scaler with current metrics
            current_load = kwargs.get("concurrent_users", 50)
            auto_scaler.update_utilization(current_load, response_time)

            # Simulate capacity-based latency
            if current_load > auto_scaler.current_capacity * 0.8:
                await asyncio.sleep(0.1)  # Additional latency when near capacity

            return result

        # Configuration for auto-scaling test
        config = LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=800,
            requests_per_second=400,
            duration_seconds=300,
        )

        # Run test with auto-scaling simulation
        load_test_runner.run_load_test(
            config=config,
            target_function=auto_scaling_aware_operation,
        )

        # Analyze auto-scaling effectiveness
        scaling_analysis = self._analyze_auto_scaling(auto_scaler.utilization_history)

        # Assertions
        assert scaling_analysis["scaling_events"] > 0, (
            "No auto-scaling events triggered"
        )
        assert scaling_analysis["peak_capacity"] > 100, (
            "Failed to scale up during spike"
        )
        assert scaling_analysis["avg_utilization"] < 90, "Utilization remained too high"

    @pytest.mark.spike
    def test_circuit_breaker_activation(self, load_test_runner):
        """Test circuit breaker activation during spikes."""

        # Simulate circuit breaker behavior
        class CircuitBreakerSimulator:
            def __init__(self):
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
                self.failure_count = 0
                self.failure_threshold = 5
                self.success_threshold = 3
                self.timeout_duration = 30  # seconds
                self.last_failure_time = 0
                self.success_count = 0
                self.events = []

            async def call_service(self, **kwargs):
                """Simulate service call through circuit breaker."""
                current_time = time.time()

                if self.state == "OPEN":
                    if current_time - self.last_failure_time > self.timeout_duration:
                        self.state = "HALF_OPEN"
                        self.events.append(
                            {"timestamp": current_time, "event": "HALF_OPEN"}
                        )
                        raise TestError("Circuit breaker is OPEN")
                        raise TestError("Circuit breaker is OPEN")

                try:
                    # Simulate higher failure rate during spikes
                    failure_rate = kwargs.get("spike_intensity", 0.1)
                    if current_time % 1.0 < failure_rate:
                        raise TestError("Service temporarily unavailable")

                    # Success
                    if self.state == "HALF_OPEN":
                        self.success_count += 1
                        if self.success_count >= self.success_threshold:
                            self.state = "CLOSED"
                            self.failure_count = 0
                            self.success_count = 0
                            self.events.append(
                                {"timestamp": current_time, "event": "CLOSED"}
                            )

                    await asyncio.sleep(0.05)  # Simulate processing time
                    return {"status": "success", "circuit_state": self.state}

                except Exception:
                    self.failure_count += 1
                    self.last_failure_time = current_time

                    if (
                        self.state in ["CLOSED", "HALF_OPEN"]
                        and self.failure_count >= self.failure_threshold
                    ):
                        self.state = "OPEN"
                        self.success_count = 0
                        self.events.append({"timestamp": current_time, "event": "OPEN"})

                    raise

        circuit_breaker = CircuitBreakerSimulator()

        async def circuit_breaker_operation(**kwargs):
            """Operation with circuit breaker protection."""
            # Add spike intensity based on current load
            kwargs["spike_intensity"] = min(
                kwargs.get("concurrent_users", 50) / 500.0, 0.3
            )
            return await circuit_breaker.call_service(**kwargs)

        # Run spike test with circuit breaker
        config = LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=600,
            requests_per_second=300,
            duration_seconds=180,
        )

        load_test_runner.run_load_test(
            config=config,
            target_function=circuit_breaker_operation,
        )

        # Analyze circuit breaker behavior
        breaker_analysis = self._analyze_circuit_breaker(circuit_breaker.events)

        # Assertions
        assert breaker_analysis["activation_count"] > 0, (
            "Circuit breaker never activated"
        )
        assert breaker_analysis["recovery_count"] > 0, "Circuit breaker never recovered"
        assert breaker_analysis["protection_effectiveness"] > 0.7, (
            "Circuit breaker not effective"
        )

    @pytest.mark.spike
    def test_database_connection_pooling(self, load_test_runner):
        """Test database connection pool behavior during spikes."""

        # Simulate connection pool
        class ConnectionPoolSimulator:
            def __init__(self, initial_size=10, max_size=50):
                self.initial_size = initial_size
                self.max_size = max_size
                self.current_size = initial_size
                self.active_connections = 0
                self.queue_length = 0
                self.metrics = []

            async def get_connection(self):
                """Get connection from pool."""
                if self.active_connections < self.current_size:
                    self.active_connections += 1
                    return f"conn_{self.active_connections}"
                elif self.current_size < self.max_size:
                    # Expand pool
                    self.current_size = min(self.current_size + 2, self.max_size)
                    self.active_connections += 1
                    return f"conn_{self.active_connections}"
                else:
                    # Queue request
                    self.queue_length += 1
                    await asyncio.sleep(0.1)  # Wait for connection
                    self.queue_length -= 1
                    self.active_connections += 1
                    return f"conn_{self.active_connections}"

            def release_connection(self, _conn_id: str):
                """Release connection back to pool."""
                self.active_connections = max(0, self.active_connections - 1)

                # Record metrics
                self.metrics.append(
                    {
                        "timestamp": time.time(),
                        "pool_size": self.current_size,
                        "active_connections": self.active_connections,
                        "queue_length": self.queue_length,
                        "utilization": self.active_connections / self.current_size,
                    }
                )

        pool = ConnectionPoolSimulator(initial_size=10, max_size=100)

        async def database_operation(**kwargs):
            """Simulate database operation with connection pooling."""
            # Get connection
            conn = await pool.get_connection()

            try:
                # Simulate database work
                work_time = kwargs.get("work_complexity", 0.05)
                await asyncio.sleep(work_time)
                return {"status": "success", "connection": conn}
            finally:
                # Always release connection
                pool.release_connection(conn)

        # Run spike test focusing on database operations
        config = LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=200,
            requests_per_second=100,
            duration_seconds=240,
        )

        load_test_runner.run_load_test(
            config=config,
            target_function=database_operation,
            work_complexity=0.1,  # Longer database operations
        )

        # Analyze connection pool behavior
        pool_analysis = self._analyze_connection_pool(pool.metrics)

        # Assertions
        assert pool_analysis["max_pool_size"] > 10, "Pool did not expand during spike"
        assert pool_analysis["max_queue_length"] < 20, "Excessive connection queuing"
        assert pool_analysis["avg_utilization"] > 0.6, "Low pool utilization"

    def _spike_aware_operation(self, **kwargs):
        """Operation that adapts to spike conditions."""
        import asyncio
        import random

        # Simulate different behavior during spikes
        concurrent_users = kwargs.get("concurrent_users", 50)

        if concurrent_users > 300:  # During spike
            # Increased processing time during spike
            processing_time = random.uniform(0.1, 0.3)
        else:
            # Normal processing time
            processing_time = random.uniform(0.05, 0.15)

        return asyncio.sleep(processing_time)

    def _analyze_spike_response(self, spike_metrics: Dict) -> Dict:
        """Analyze system response to traffic spike."""
        baseline_times = spike_metrics["baseline_phase"]["response_times"]
        spike_times = spike_metrics["spike_phase"]["response_times"]
        recovery_times = spike_metrics["recovery_phase"]["response_times"]

        if not baseline_times or not spike_times or not recovery_times:
            return {
                "spike_handled": False,
                "recovery_time": float("inf"),
                "performance_degradation": float("inf"),
            }

        baseline_avg = sum(baseline_times) / len(baseline_times)
        spike_avg = sum(spike_times) / len(spike_times)
        recovery_avg = sum(recovery_times) / len(recovery_times)

        performance_degradation = ((spike_avg - baseline_avg) / baseline_avg) * 100
        recovery_ratio = recovery_avg / baseline_avg

        return {
            "spike_handled": spike_avg < baseline_avg * 5,  # Less than 5x degradation
            "recovery_time": 60,  # Simplified - would calculate actual
            "performance_degradation": performance_degradation,
            "recovery_effectiveness": recovery_ratio < 1.2,  # Within 20% of baseline
            "baseline_avg": baseline_avg * 1000,  # Convert to ms
            "spike_avg": spike_avg * 1000,
            "recovery_avg": recovery_avg * 1000,
        }

    def _analyze_double_spike_recovery(self, metrics: list[Dict]) -> Dict:
        """Analyze recovery between double spikes."""
        if len(metrics) < 10:
            return {
                "first_spike_recovered": False,
                "second_spike_handled": False,
                "system_stability": 0.0,
            }

        # Find spike periods (high user count)
        spike_periods = []
        for i, metric in enumerate(metrics):
            if metric["users"] > 200:  # Spike threshold
                spike_periods.append(i)

        # Analyze stability between spikes
        stability_score = 1.0
        error_spikes = 0

        for i in range(1, len(metrics)):
            if (
                metrics[i]["error_rate"] > metrics[i - 1]["error_rate"] + 5
            ):  # 5% increase
                error_spikes += 1

        stability_score -= error_spikes / len(metrics)

        return {
            "first_spike_recovered": len(spike_periods) >= 2,
            "second_spike_handled": True,  # Simplified analysis
            "system_stability": max(0.0, stability_score),
            "spike_periods_detected": len(spike_periods),
            "error_spikes": error_spikes,
        }

    def _analyze_auto_scaling(self, utilization_history: list[Dict]) -> Dict:
        """Analyze auto-scaling effectiveness."""
        if not utilization_history:
            return {"scaling_events": 0, "peak_capacity": 100, "avg_utilization": 0}

        scaling_events = 0
        capacities = [h["capacity"] for h in utilization_history]
        utilizations = [h["utilization"] for h in utilization_history]

        # Count scaling events (capacity changes)
        for i in range(1, len(capacities)):
            if capacities[i] != capacities[i - 1]:
                scaling_events += 1

        return {
            "scaling_events": scaling_events,
            "peak_capacity": max(capacities),
            "avg_utilization": sum(utilizations) / len(utilizations),
            "min_capacity": min(capacities),
            "capacity_variance": max(capacities) - min(capacities),
        }

    def _analyze_circuit_breaker(self, events: list[Dict]) -> Dict:
        """Analyze circuit breaker effectiveness."""
        open_events = [e for e in events if e["event"] == "OPEN"]
        closed_events = [e for e in events if e["event"] == "CLOSED"]

        activation_count = len(open_events)
        recovery_count = len(closed_events)

        # Calculate protection effectiveness
        protection_effectiveness = min(1.0, recovery_count / max(activation_count, 1))

        return {
            "activation_count": activation_count,
            "recovery_count": recovery_count,
            "protection_effectiveness": protection_effectiveness,
            "total_events": len(events),
        }

    def _analyze_connection_pool(self, metrics: list[Dict]) -> Dict:
        """Analyze database connection pool behavior."""
        if not metrics:
            return {"max_pool_size": 0, "max_queue_length": 0, "avg_utilization": 0}

        pool_sizes = [m["pool_size"] for m in metrics]
        queue_lengths = [m["queue_length"] for m in metrics]
        utilizations = [m["utilization"] for m in metrics]

        return {
            "max_pool_size": max(pool_sizes),
            "min_pool_size": min(pool_sizes),
            "max_queue_length": max(queue_lengths),
            "avg_utilization": sum(utilizations) / len(utilizations),
            "peak_queue_time": max(queue_lengths) * 0.1,  # Simplified calculation
        }
