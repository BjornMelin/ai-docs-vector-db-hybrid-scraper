"""Stress testing scenarios for system breaking point analysis.

This module implements various stress testing scenarios to identify system
breaking points, resource exhaustion conditions, and failure modes under
extreme load conditions.
"""

import asyncio
import random
import time

import pytest

from tests.load.conftest import LoadTestConfig, LoadTestType


class TestError(Exception):
    """Custom exception for this module."""


@pytest.mark.stress
@pytest.mark.asyncio
class TestStressScenarios:
    """Test various stress scenarios to find system limits."""

    @pytest.mark.asyncio
    async def test_cpu_intensive_stress(self, load_test_runner, mock_load_test_service):
        """Test CPU-intensive stress scenarios."""

        class CPUStressSimulator:
            def __init__(self, service):
                self.service = service
                self.cpu_load_factor = 1.0

            async def cpu_intensive_task(self, **_kwargs):
                """Simulate CPU-intensive processing."""
                # Simulate CPU-bound work by increasing latency based on load
                cpu_latency = self.service.base_latency * self.cpu_load_factor
                await asyncio.sleep(cpu_latency)

                # Simulate some CPU work (simplified)
                result = await self.service.process_request(**_kwargs)
                result["cpu_processing_time"] = cpu_latency
                return result

            def increase_cpu_load(self, factor: float):
                """Increase CPU load simulation."""
                self.cpu_load_factor = factor

        cpu_stress = CPUStressSimulator(mock_load_test_service)

        # Progressive CPU stress test
        stress_levels = [
            {"factor": 1.0, "users": 50, "name": "baseline"},
            {"factor": 2.0, "users": 100, "name": "2x_cpu_load"},
            {"factor": 5.0, "users": 150, "name": "5x_cpu_load"},
            {"factor": 10.0, "users": 200, "name": "10x_cpu_load"},
        ]

        stress_results = []

        for stress_level in stress_levels:
            cpu_stress.increase_cpu_load(stress_level["factor"])

            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=stress_level["users"],
                requests_per_second=stress_level["users"] * 0.5,
                duration_seconds=60.0,
                success_criteria={
                    "max_error_rate_percent": 50.0,  # Very lenient for stress testing
                    "max_avg_response_time_ms": 10000.0,  # 10 second timeout
                },
            )

            print(
                f"Testing {stress_level['name']}: {stress_level['factor']}x CPU load, {stress_level['users']} users"
            )

            result = await load_test_runner.run_load_test(
                config=config, target_function=cpu_stress.cpu_intensive_task
            )

            stress_result = {
                "level": stress_level["name"],
                "cpu_factor": stress_level["factor"],
                "users": stress_level["users"],
                "_total_requests": result.metrics.total_requests,
                "successful_requests": result.metrics.successful_requests,
                "failed_requests": result.metrics.failed_requests,
                "success_rate": (
                    result.metrics.successful_requests
                    / max(result.metrics.total_requests, 1)
                )
                * 100,
                "avg_response_time": sum(result.metrics.response_times)
                / len(result.metrics.response_times)
                if result.metrics.response_times
                else 0,
                "throughput": result.metrics.throughput_rps,
            }

            stress_results.append(stress_result)

            # Check if system is still responsive
            assert result.metrics.total_requests > 0, (
                f"System completely unresponsive at {stress_level['name']}"
            )

        # Analyze stress progression
        assert len(stress_results) == 4

        # Find breaking point (where success rate drops below 50%)
        breaking_point = None
        for result in stress_results:
            if result["success_rate"] < 50.0:
                breaking_point = result
                break

        print("Stress test analysis:")
        for result in stress_results:
            print(
                f"  {result['level']}: {result['success_rate']:.1f}% success, {result['avg_response_time']:.2f}ms avg response"
            )

        if breaking_point:
            print(
                f"Breaking point identified at: {breaking_point['level']} ({breaking_point['cpu_factor']}x CPU load)"
            )

    @pytest.mark.asyncio
    async def test_memory_pressure_stress(
        self, load_test_runner, mock_load_test_service
    ):
        """Test stress scenarios with memory pressure simulation."""

        class MemoryStressSimulator:
            def __init__(self, service):
                self.service = service
                self.memory_pressure = {}  # Simulated memory usage

            async def memory_intensive_task(self, data_size_mb: float = 1.0, **_kwargs):
                """Simulate memory-intensive processing."""
                # Simulate memory allocation
                memory_key = f"mem_{time.time()}_{id(asyncio.current_task())}"
                self.memory_pressure[memory_key] = "x" * int(
                    data_size_mb * 1024
                )  # Rough simulation

                try:
                    # Process with memory pressure
                    result = await self.service.process_request(
                        data_size_mb=data_size_mb, **_kwargs
                    )
                    result["memory_allocated_mb"] = data_size_mb
                    result["_total_memory_objects"] = len(self.memory_pressure)

                    # Simulate memory cleanup (sometimes fails under stress)
                    if len(self.memory_pressure) > 100:  # Memory pressure threshold
                        # Clean up some memory
                        old_keys = list(self.memory_pressure.keys())[:10]
                        for key in old_keys:
                            del self.memory_pressure[key]

                except Exception:
                    # Clean up on error
                    if memory_key in self.memory_pressure:
                        del self.memory_pressure[memory_key]
                    raise
                else:
                    return result

        memory_stress = MemoryStressSimulator(mock_load_test_service)

        # Progressive memory stress test
        memory_levels = [
            {"data_mb": 1.0, "users": 30, "name": "low_memory"},
            {"data_mb": 5.0, "users": 50, "name": "medium_memory"},
            {"data_mb": 15.0, "users": 75, "name": "high_memory"},
            {"data_mb": 50.0, "users": 100, "name": "extreme_memory"},
        ]

        memory_results = []

        for memory_level in memory_levels:
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=memory_level["users"],
                requests_per_second=memory_level["users"]
                * 0.3,  # Lower RPS for memory tests
                duration_seconds=45.0,
                data_size_mb=memory_level["data_mb"],
                success_criteria={
                    "max_error_rate_percent": 40.0,
                    "max_avg_response_time_ms": 8000.0,
                },
            )

            print(
                f"Testing {memory_level['name']}: {memory_level['data_mb']}MB per request, {memory_level['users']} users"
            )

            result = await load_test_runner.run_load_test(
                config=config, target_function=memory_stress.memory_intensive_task
            )

            memory_result = {
                "level": memory_level["name"],
                "data_size_mb": memory_level["data_mb"],
                "users": memory_level["users"],
                "_total_requests": result.metrics.total_requests,
                "success_rate": (
                    result.metrics.successful_requests
                    / max(result.metrics.total_requests, 1)
                )
                * 100,
                "avg_response_time": sum(result.metrics.response_times)
                / len(result.metrics.response_times)
                if result.metrics.response_times
                else 0,
                "simulated_memory_objects": len(memory_stress.memory_pressure),
            }

            memory_results.append(memory_result)

            assert result.metrics.total_requests > 0, (
                f"No requests processed at {memory_level['name']}"
            )

        assert len(memory_results) == 4

        print("Memory stress analysis:")
        for result in memory_results:
            print(
                f"  {result['level']}: {result['success_rate']:.1f}% success, {result['simulated_memory_objects']} memory objects"
            )

    @pytest.mark.asyncio
    async def test_connection_exhaustion_stress(
        self, load_test_runner, mock_load_test_service
    ):
        """Test stress scenarios that exhaust connection pools."""

        class ConnectionStressSimulator:
            def __init__(self, service):
                self.service = service
                self.active_connections = 0
                self.max_connections = 50  # Simulated connection pool limit
                self.connection_failures = 0

            async def connection_intensive_task(self, **_kwargs):
                """Simulate connection-intensive operations."""
                # Try to acquire connection
                if self.active_connections >= self.max_connections:
                    msg = f"Connection pool exhausted ({self.active_connections}/{self.max_connections})"
                    raise TestError(msg)

                # Acquire connection
                self.active_connections += 1

                try:
                    # Simulate connection hold time
                    connection_hold_time = (
                        0.05 + (self.active_connections / self.max_connections) * 0.1
                    )
                    await asyncio.sleep(connection_hold_time)

                    result = await self.service.process_request(**_kwargs)
                    result["connections_used"] = self.active_connections
                    result["connection_hold_time"] = connection_hold_time

                    return result

                finally:
                    # Release connection
                    self.active_connections = max(0, self.active_connections - 1)

        connection_stress = ConnectionStressSimulator(mock_load_test_service)

        # Test connection exhaustion scenarios
        connection_tests = [
            {"users": 25, "rps": 20, "name": "within_limits"},
            {"users": 60, "rps": 50, "name": "at_limits"},
            {"users": 100, "rps": 80, "name": "over_limits"},
            {"users": 150, "rps": 120, "name": "extreme_over_limits"},
        ]

        connection_results = []

        for test in connection_tests:
            # Reset connection failures for each test
            connection_stress.connection_failures = 0

            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=test["users"],
                requests_per_second=test["rps"],
                duration_seconds=30.0,
                success_criteria={
                    "max_error_rate_percent": 70.0,  # Very lenient for connection stress
                },
            )

            print(f"Testing {test['name']}: {test['users']} users, {test['rps']} RPS")

            result = await load_test_runner.run_load_test(
                config=config,
                target_function=connection_stress.connection_intensive_task,
            )

            connection_result = {
                "test": test["name"],
                "users": test["users"],
                "rps": test["rps"],
                "_total_requests": result.metrics.total_requests,
                "successful_requests": result.metrics.successful_requests,
                "connection_failures": connection_stress.connection_failures,
                "success_rate": (
                    result.metrics.successful_requests
                    / max(result.metrics.total_requests, 1)
                )
                * 100,
                "_total_failures": result.metrics.failed_requests,
            }

            connection_results.append(connection_result)

            assert result.metrics.total_requests > 0, (
                f"No requests attempted at {test['name']}"
            )

        assert len(connection_results) == 4

        print("Connection stress analysis:")
        for result in connection_results:
            print(
                f"  {result['test']}: {result['success_rate']:.1f}% success, {result['connection_failures']} connection failures"
            )

    @pytest.mark.asyncio
    async def test_cascading_failure_stress(
        self, load_test_runner, mock_load_test_service
    ):
        """Test cascading failure scenarios under stress."""

        class CascadingFailureSimulator:
            def __init__(self, service):
                self.service = service
                self.failure_cascade_level = 0
                self.consecutive_failures = 0
                self.circuit_breaker_open = False

            async def cascading_failure_task(self, **_kwargs):
                """Simulate services that can cascade into failure."""
                # Circuit breaker logic
                if self.circuit_breaker_open:
                    if self.consecutive_failures > 50:  # Stay open for a while
                        self.consecutive_failures -= 1
                        msg = "Circuit breaker is OPEN - rejecting requests"
                        raise TestError(msg)
                    # Try to close circuit breaker
                    self.circuit_breaker_open = False
                    self.consecutive_failures = 0

                # Simulate failure probability that increases with load

                current_load = self.service.request_count / 100.0  # Normalize load
                failure_probability = min(
                    0.3, current_load * 0.05 + self.failure_cascade_level * 0.1
                )

                if random.random() < failure_probability:
                    self.consecutive_failures += 1
                    self.failure_cascade_level = min(
                        5, self.failure_cascade_level + 0.1
                    )

                    # Open circuit breaker if too many consecutive failures
                    if self.consecutive_failures > 10:
                        self.circuit_breaker_open = True

                    msg = f"Cascading failure (level {self.failure_cascade_level:.1f}, consecutive: {self.consecutive_failures})"
                    raise TestError(msg)
                # Success - reduce cascade level
                self.failure_cascade_level = max(0, self.failure_cascade_level - 0.05)
                self.consecutive_failures = max(0, self.consecutive_failures - 1)

                result = await self.service.process_request(**_kwargs)
                result["cascade_level"] = self.failure_cascade_level
                result["circuit_breaker_open"] = self.circuit_breaker_open

                return result

        cascade_simulator = CascadingFailureSimulator(mock_load_test_service)

        # Test cascading failure under increasing stress
        cascade_tests = [
            {"users": 20, "duration": 30, "name": "initial_load"},
            {"users": 50, "duration": 45, "name": "increased_load"},
            {"users": 100, "duration": 60, "name": "high_load"},
            {"users": 200, "duration": 30, "name": "extreme_load"},
        ]

        cascade_results = []

        for test in cascade_tests:
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=test["users"],
                requests_per_second=test["users"] * 0.8,
                duration_seconds=test["duration"],
                success_criteria={
                    "max_error_rate_percent": 80.0,  # Expect high failure rates
                },
            )

            print(
                f"Testing {test['name']}: {test['users']} users for {test['duration']}s"
            )

            initial_cascade_level = cascade_simulator.failure_cascade_level

            result = await load_test_runner.run_load_test(
                config=config, target_function=cascade_simulator.cascading_failure_task
            )

            final_cascade_level = cascade_simulator.failure_cascade_level

            cascade_result = {
                "test": test["name"],
                "users": test["users"],
                "duration": test["duration"],
                "_total_requests": result.metrics.total_requests,
                "successful_requests": result.metrics.successful_requests,
                "failed_requests": result.metrics.failed_requests,
                "success_rate": (
                    result.metrics.successful_requests
                    / max(result.metrics.total_requests, 1)
                )
                * 100,
                "initial_cascade_level": initial_cascade_level,
                "final_cascade_level": final_cascade_level,
                "cascade_progression": final_cascade_level - initial_cascade_level,
                "circuit_breaker_triggered": cascade_simulator.circuit_breaker_open,
            }

            cascade_results.append(cascade_result)

            assert result.metrics.total_requests > 0, (
                f"No requests processed during {test['name']}"
            )

        assert len(cascade_results) == 4

        print("Cascading failure analysis:")
        for result in cascade_results:
            print(
                f"  {result['test']}: {result['success_rate']:.1f}% success, cascade level: {result['initial_cascade_level']:.1f} → {result['final_cascade_level']:.1f}"
            )

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(
        self, load_test_runner, mock_load_test_service
    ):
        """Test system recovery after resource exhaustion."""

        class ResourceExhaustionSimulator:
            def __init__(self, service):
                self.service = service
                self.resource_pool = 100  # Available resources
                self.max_resources = 100
                self.exhaustion_triggered = False

            async def resource_consuming_task(self, **_kwargs):
                """Simulate resource-consuming operations."""
                # Try to acquire resources
                resources_needed = 1

                if self.resource_pool < resources_needed:
                    self.exhaustion_triggered = True
                    msg = f"Resource exhaustion: {self.resource_pool}/{self.max_resources} available"
                    raise TestError(msg)

                # Consume resources
                self.resource_pool -= resources_needed

                try:
                    # Simulate work
                    work_time = (
                        0.02 + (1 - self.resource_pool / self.max_resources) * 0.08
                    )  # Slower when resources low
                    await asyncio.sleep(work_time)

                    result = await self.service.process_request(**_kwargs)
                    result["resources_available"] = self.resource_pool
                    result["work_time"] = work_time

                    return result

                finally:
                    # Release resources (with some recovery time)
                    await asyncio.sleep(0.01)  # Recovery time
                    self.resource_pool = min(self.max_resources, self.resource_pool + 1)

        resource_simulator = ResourceExhaustionSimulator(mock_load_test_service)

        # Test resource exhaustion and recovery pattern
        exhaustion_phases = [
            {"users": 30, "duration": 45, "phase": "warm_up"},
            {
                "users": 120,
                "duration": 30,
                "phase": "exhaustion",
            },  # Should exhaust resources
            {"users": 40, "duration": 60, "phase": "recovery"},  # Allow recovery
            {"users": 80, "duration": 30, "phase": "stress_test"},  # Test resilience
        ]

        exhaustion_results = []

        for phase in exhaustion_phases:
            resource_simulator.exhaustion_triggered = False
            initial_resources = resource_simulator.resource_pool

            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=phase["users"],
                requests_per_second=phase["users"] * 0.6,
                duration_seconds=phase["duration"],
                success_criteria={
                    "max_error_rate_percent": 60.0,  # Expect failures during exhaustion
                },
            )

            print(
                f"Testing {phase['phase']}: {phase['users']} users for {phase['duration']}s"
            )

            result = await load_test_runner.run_load_test(
                config=config,
                target_function=resource_simulator.resource_consuming_task,
            )

            final_resources = resource_simulator.resource_pool

            phase_result = {
                "phase": phase["phase"],
                "users": phase["users"],
                "_total_requests": result.metrics.total_requests,
                "successful_requests": result.metrics.successful_requests,
                "success_rate": (
                    result.metrics.successful_requests
                    / max(result.metrics.total_requests, 1)
                )
                * 100,
                "initial_resources": initial_resources,
                "final_resources": final_resources,
                "resource_change": final_resources - initial_resources,
                "exhaustion_triggered": resource_simulator.exhaustion_triggered,
                "avg_response_time": sum(result.metrics.response_times)
                / len(result.metrics.response_times)
                if result.metrics.response_times
                else 0,
            }

            exhaustion_results.append(phase_result)

            assert result.metrics.total_requests > 0, (
                f"No requests processed during {phase['phase']}"
            )

        assert len(exhaustion_results) == 4

        # Analyze recovery pattern
        exhaustion_phase = next(
            r for r in exhaustion_results if r["phase"] == "exhaustion"
        )
        recovery_phase = next(r for r in exhaustion_results if r["phase"] == "recovery")

        print("Resource exhaustion analysis:")
        for result in exhaustion_results:
            print(
                f"  {result['phase']}: {result['success_rate']:.1f}% success, resources: {result['initial_resources']} → {result['final_resources']}"
            )

        # Recovery should show improvement
        assert (
            recovery_phase["success_rate"] > exhaustion_phase["success_rate"] * 0.5
        ), "System did not recover properly"
