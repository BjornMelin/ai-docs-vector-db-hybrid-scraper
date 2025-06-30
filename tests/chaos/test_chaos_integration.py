"""Chaos engineering integration tests.

This module implements comprehensive integration tests that combine all chaos
engineering components to validate system resilience under real-world failure
scenarios.
"""

import asyncio
import logging
import time

import pytest

from tests.chaos.conftest import ChaosExperiment, FailureType
from tests.chaos.test_chaos_runner import ChaosTestRunner, ChaosTestSuite


logger = logging.getLogger(__name__)


class TestError(Exception):
    """Custom exception for this module."""


@pytest.mark.chaos
@pytest.mark.integration
class TestChaosIntegration:
    """Integration tests for chaos engineering components."""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for comprehensive testing."""

        class IntegratedSystemSimulator:
            def __init__(self):
                # System components
                self.services = {
                    "api_gateway": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.05,
                    },
                    "search_service": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.1,
                    },
                    "vector_db": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.03,
                    },
                    "embedding_service": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.08,
                    },
                    "cache": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.01,
                    },
                    "auth_service": {
                        "status": "healthy",
                        "connections": 0,
                        "response_time": 0.02,
                    },
                }

                # Resource usage
                self.resources = {
                    "memory": {"used": 40, "_total": 100},  # 40% used
                    "cpu": {"used": 30, "_total": 100},  # 30% used
                    "disk": {"used": 50, "_total": 100},  # 50% used
                    "connections": {"used": 20, "_total": 1000},  # 20 connections used
                }

                # Data stores
                self.data_stores = {
                    "documents": 10000,
                    "embeddings": 10000,
                    "cache_entries": 5000,
                    "user_sessions": 500,
                }

                # Failure states
                self.active_failures = {}
                self.circuit_breakers = {}
                self.degraded_mode = False

                # Metrics
                self.metrics = {
                    "throughput": 1000,  # requests/second
                    "latency_p95": 0.2,  # 200ms
                    "error_rate": 0.01,  # 1%
                    "availability": 0.999,  # 99.9%
                }

            async def inject_failure(
                self, failure_type: FailureType, target_service: str
            ):
                """Inject failure into the system."""
                self.active_failures[target_service] = failure_type

                # Update service status
                if target_service in self.services:
                    if failure_type == FailureType.SERVICE_UNAVAILABLE:
                        self.services[target_service]["status"] = "failed"
                        self.services[target_service]["response_time"] = 0.0
                    elif failure_type == FailureType.NETWORK_TIMEOUT:
                        self.services[target_service]["status"] = "degraded"
                        self.services[target_service]["response_time"] *= (
                            10  # 10x slower
                        )
                    elif failure_type == FailureType.MEMORY_EXHAUSTION:
                        self.resources["memory"]["used"] = min(
                            95, self.resources["memory"]["_total"]
                        )
                        self.services[target_service]["status"] = "degraded"

                # Trigger cascading effects
                await self._handle_failure_cascade(failure_type, target_service)

            async def _handle_failure_cascade(
                self, _failure_type: FailureType, failed_service: str
            ):
                """Handle cascading effects of failures."""
                # Service dependencies
                dependencies = {
                    "api_gateway": ["auth_service", "search_service"],
                    "search_service": ["vector_db", "embedding_service"],
                    "embedding_service": [],
                    "vector_db": [],
                    "cache": [],
                    "auth_service": [],
                }

                # Find services that depend on the failed service
                for service, deps in dependencies.items():
                    if (
                        failed_service in deps
                        and service not in self.active_failures
                        and self.services[service]["status"] == "healthy"
                    ):
                        # Dependent service may degrade
                        self.services[service]["status"] = "degraded"
                        self.services[service]["response_time"] *= 2

                # Update system metrics
                self._update_system_metrics()

            def _update_system_metrics(self):
                """Update system-wide metrics based on service states."""
                # Count healthy vs degraded/failed services
                healthy_services = len(
                    [s for s in self.services.values() if s["status"] == "healthy"]
                )
                _total_services = len(self.services)

                # Update availability
                self.metrics["availability"] = healthy_services / _total_services

                # Update error rate
                failed_services = len(
                    [s for s in self.services.values() if s["status"] == "failed"]
                )
                self.metrics["error_rate"] = min(0.5, 0.01 + (failed_services * 0.1))

                # Update throughput
                degraded_services = len(
                    [s for s in self.services.values() if s["status"] == "degraded"]
                )
                throughput_factor = max(
                    0.1, 1.0 - (failed_services * 0.3) - (degraded_services * 0.1)
                )
                self.metrics["throughput"] = int(1000 * throughput_factor)

                # Update latency
                avg_response_time = (
                    sum([s["response_time"] for s in self.services.values()])
                    / _total_services
                )
                self.metrics["latency_p95"] = (
                    avg_response_time * 5
                )  # P95 is ~5x average

            async def stop_failure_injection(self, target_service: str):
                """Stop failure injection and begin recovery."""
                if target_service in self.active_failures:
                    del self.active_failures[target_service]

                # Begin service recovery
                if target_service in self.services:
                    await self._recover_service(target_service)

                # Update metrics
                self._update_system_metrics()

            async def _recover_service(self, service_name: str):
                """Recover a service from failure."""
                # Simulate recovery time
                await asyncio.sleep(0.05)  # 50ms recovery

                # Restore service to healthy state
                self.services[service_name]["status"] = "healthy"

                # Reset response time to baseline (with slight degradation during recovery)
                baseline_times = {
                    "api_gateway": 0.05,
                    "search_service": 0.1,
                    "vector_db": 0.03,
                    "embedding_service": 0.08,
                    "cache": 0.01,
                    "auth_service": 0.02,
                }

                recovery_factor = 1.2  # 20% slower during recovery
                self.services[service_name]["response_time"] = (
                    baseline_times.get(service_name, 0.05) * recovery_factor
                )

            async def health_check(self):
                """Perform system health check."""
                failed_services = [
                    name
                    for name, service in self.services.items()
                    if service["status"] == "failed"
                ]

                if failed_services:
                    msg = f"Health check failed - services down: {failed_services}"
                    raise TestError(msg)

                return {
                    "status": "healthy",
                    "services": self.services,
                    "metrics": self.metrics,
                    "resources": self.resources,
                }

            async def get_metrics(self):
                """Get current system metrics."""
                return {
                    "timestamp": time.time(),
                    "services": self.services,
                    "resources": self.resources,
                    "data_stores": self.data_stores,
                    "system_metrics": self.metrics,
                    "active_failures": self.active_failures,
                }

            async def check_data_integrity(self) -> bool:
                """Check data integrity."""
                # Simulate data integrity check
                expected_docs = 10000
                actual_docs = self.data_stores.get("documents", 0)

                # Allow small variance (5%)
                return abs(actual_docs - expected_docs) / expected_docs <= 0.05

            async def check_degraded_service(self) -> bool:
                """Check if system provides degraded service."""
                # System provides degraded service if at least 50% of services are operational
                operational_services = len(
                    [
                        s
                        for s in self.services.values()
                        if s["status"] in ["healthy", "degraded"]
                    ]
                )

                return operational_services >= len(self.services) * 0.5

        return IntegratedSystemSimulator()

    @pytest.mark.usefixtures("fault_injector", "resilience_validator")
    async def test_comprehensive_chaos_scenario(
        self,
        integrated_system,
        chaos_experiment_runner,
    ):
        """Test comprehensive chaos scenario with multiple failure types."""
        # Define comprehensive chaos experiments
        experiments = [
            chaos_experiment_runner.define_experiment(
                name="cascade_failure_test",
                description="Test cascade failure resilience",
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                target_service="vector_db",
                duration_seconds=0.2,
                failure_rate=1.0,
                blast_radius="service",
                recovery_time_seconds=0.1,
                success_criteria=["system_recovers", "graceful_degradation"],
                rollback_strategy="immediate",
            ),
            chaos_experiment_runner.define_experiment(
                name="network_partition_test",
                description="Test network partition handling",
                failure_type=FailureType.NETWORK_TIMEOUT,
                target_service="search_service",
                duration_seconds=0.15,
                failure_rate=1.0,
                blast_radius="single",
                recovery_time_seconds=0.1,
                success_criteria=["system_recovers", "no_data_loss"],
                rollback_strategy="immediate",
            ),
            chaos_experiment_runner.define_experiment(
                name="memory_pressure_test",
                description="Test memory pressure handling",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                target_service="embedding_service",
                duration_seconds=0.1,
                failure_rate=0.8,
                blast_radius="single",
                recovery_time_seconds=0.05,
                success_criteria=["system_recovers", "graceful_degradation"],
                rollback_strategy="immediate",
            ),
        ]

        # Execute experiments sequentially
        results = []
        for experiment in experiments:
            result = await chaos_experiment_runner.run_experiment(
                experiment,
                integrated_system,
                monitoring_func=integrated_system.get_metrics,
            )
            results.append(result)

        # Validate overall system resilience
        assert len(results) == 3

        # Check that all experiments completed
        for result in results:
            assert result.failure_injected, (
                f"Failure not injected in {result.experiment_name}"
            )
            assert result.system_recovered, (
                f"System did not recover in {result.experiment_name}"
            )
            assert result.duration > 0, f"Invalid duration in {result.experiment_name}"

        # Verify system is healthy after all experiments
        final_health = await integrated_system.health_check()
        assert final_health["status"] == "healthy"

    @pytest.mark.usefixtures("fault_injector", "resilience_validator")
    async def test_real_world_failure_simulation(self, integrated_system):
        """Test realistic failure simulation combining multiple chaos types."""
        # Capture baseline metrics
        baseline_metrics = await integrated_system.get_metrics()
        baseline_throughput = baseline_metrics["system_metrics"]["throughput"]

        # Phase 1: Gradual memory pressure
        await integrated_system.inject_failure(
            FailureType.MEMORY_EXHAUSTION, "embedding_service"
        )

        # Monitor system adaptation
        phase1_metrics = await integrated_system.get_metrics()
        assert (
            phase1_metrics["resources"]["memory"]["used"]
            > baseline_metrics["resources"]["memory"]["used"]
        )

        # Phase 2: Add network latency
        await integrated_system.inject_failure(
            FailureType.NETWORK_TIMEOUT, "search_service"
        )

        # Phase 3: Service failure
        await integrated_system.inject_failure(FailureType.SERVICE_UNAVAILABLE, "cache")

        # Verify system degrades but continues operating
        degraded_metrics = await integrated_system.get_metrics()

        # Should have reduced performance but still functional
        assert degraded_metrics["system_metrics"]["throughput"] < baseline_throughput
        assert (
            degraded_metrics["system_metrics"]["error_rate"]
            > baseline_metrics["system_metrics"]["error_rate"]
        )
        assert degraded_metrics["system_metrics"]["availability"] < 1.0

        # Verify graceful degradation
        can_provide_degraded_service = await integrated_system.check_degraded_service()
        assert can_provide_degraded_service, "System should provide degraded service"

        # Phase 4: Recovery
        await integrated_system.stop_failure_injection("cache")
        await integrated_system.stop_failure_injection("search_service")
        await integrated_system.stop_failure_injection("embedding_service")

        # Wait for recovery
        await asyncio.sleep(0.2)

        # Verify recovery
        recovery_metrics = await integrated_system.get_metrics()

        # System should recover most of its performance
        assert (
            recovery_metrics["system_metrics"]["throughput"]
            > degraded_metrics["system_metrics"]["throughput"]
        )
        assert (
            recovery_metrics["system_metrics"]["error_rate"]
            < degraded_metrics["system_metrics"]["error_rate"]
        )

        # Final health check
        final_health = await integrated_system.health_check()
        assert final_health["status"] == "healthy"

    async def test_circuit_breaker_integration(
        self, integrated_system, resilience_validator
    ):
        """Test circuit breaker integration with chaos testing."""
        # Test circuit breaker with simulated service
        failure_count = 0

        async def failing_service():
            nonlocal failure_count
            failure_count += 1

            # Simulate vector_db failure
            if "vector_db" in integrated_system.active_failures:
                msg = "Vector DB unavailable"
                raise TestError(msg)

            return {"status": "success", "data": "vector_search_results"}

        # Inject failure to trigger circuit breaker
        await integrated_system.inject_failure(
            FailureType.SERVICE_UNAVAILABLE, "vector_db"
        )

        # Validate circuit breaker behavior
        circuit_result = await resilience_validator.validate_circuit_breaker(
            service_func=failing_service, failure_threshold=3, recovery_timeout=0.1
        )

        assert circuit_result["circuit_breaker_triggered"], (
            "Circuit breaker should be triggered"
        )
        assert circuit_result["failure_count"] >= 3, "Should accumulate failures"

        # Recover service
        await integrated_system.stop_failure_injection("vector_db")

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Circuit breaker should allow requests again
        assert circuit_result["recovery_successful"] or failure_count > 3

    async def test_retry_mechanism_validation(
        self, integrated_system, resilience_validator
    ):
        """Test retry mechanism validation during chaos events."""
        call_count = 0

        async def unreliable_service():
            nonlocal call_count
            call_count += 1

            # Simulate search service issues
            if (
                "search_service" in integrated_system.active_failures
                and call_count <= 3
            ):  # Fail first 3 attempts
                msg = "Search service temporarily unavailable"
                raise TestError(msg)

            return {"results": ["doc1", "doc2", "doc3"]}

        # Inject failure
        await integrated_system.inject_failure(
            FailureType.NETWORK_TIMEOUT, "search_service"
        )

        # Validate retry behavior
        retry_result = await resilience_validator.validate_retry_behavior(
            service_func=unreliable_service,
            max_retries=5,
            backoff_factor=0.01,  # Fast backoff for testing
        )

        assert retry_result["retry_attempts"] > 1, "Should attempt retries"
        assert retry_result["success_on_retry"], "Should eventually succeed"
        assert retry_result["backoff_respected"], "Should respect backoff timing"

        # Stop failure injection
        await integrated_system.stop_failure_injection("search_service")

    async def test_system_recovery_validation(
        self, integrated_system, resilience_validator
    ):
        """Test comprehensive system recovery validation."""
        # Inject multiple failures
        await integrated_system.inject_failure(
            FailureType.SERVICE_UNAVAILABLE, "auth_service"
        )
        await integrated_system.inject_failure(
            FailureType.MEMORY_EXHAUSTION, "embedding_service"
        )

        # Verify system is degraded
        def _assert_health_failure():
            msg = "Health check should fail with multiple failures"
            raise AssertionError(msg)

        try:
            await integrated_system.health_check()
            _assert_health_failure()
        except (TestError, ValueError, ConnectionError, TimeoutError):
            # Expected failure
            logger.debug("Expected failure during chaos testing")

        # Measure recovery time
        recovery_result = await resilience_validator.measure_system_recovery(
            health_check_func=integrated_system.health_check,
            recovery_timeout=5.0,
            check_interval=0.1,
        )

        # Start recovery process
        await integrated_system.stop_failure_injection("auth_service")
        await integrated_system.stop_failure_injection("embedding_service")

        # Wait a bit for recovery
        await asyncio.sleep(0.2)

        # Validate final recovery
        assert recovery_result["health_checks_performed"] > 0

        # Final health check should pass
        final_health = await integrated_system.health_check()
        assert final_health["status"] == "healthy"

    async def test_data_consistency_during_chaos(self, integrated_system):
        """Test data consistency during chaos events."""
        # Check initial data integrity
        initial_integrity = await integrated_system.check_data_integrity()
        assert initial_integrity, "Initial data should be intact"

        # Inject failures that could affect data
        await integrated_system.inject_failure(
            FailureType.SERVICE_UNAVAILABLE, "vector_db"
        )

        # Simulate data operations during failure
        # (In real implementation, this would test actual data operations)

        # Check data integrity during failure
        during_failure_integrity = await integrated_system.check_data_integrity()
        # Data should still be consistent (no corruption)
        assert during_failure_integrity, (
            "Data integrity should be maintained during failure"
        )

        # Recover service
        await integrated_system.stop_failure_injection("vector_db")
        await asyncio.sleep(0.1)

        # Check data integrity after recovery
        post_recovery_integrity = await integrated_system.check_data_integrity()
        assert post_recovery_integrity, (
            "Data integrity should be maintained after recovery"
        )

    async def test_performance_impact_measurement(self, integrated_system):
        """Test measurement of performance impact during chaos events."""
        # Capture baseline performance
        baseline_metrics = await integrated_system.get_metrics()
        baseline_throughput = baseline_metrics["system_metrics"]["throughput"]
        baseline_latency = baseline_metrics["system_metrics"]["latency_p95"]
        baseline_error_rate = baseline_metrics["system_metrics"]["error_rate"]

        # Inject performance-affecting failure
        await integrated_system.inject_failure(
            FailureType.NETWORK_TIMEOUT, "api_gateway"
        )

        # Measure degraded performance
        degraded_metrics = await integrated_system.get_metrics()
        degraded_throughput = degraded_metrics["system_metrics"]["throughput"]
        degraded_latency = degraded_metrics["system_metrics"]["latency_p95"]
        degraded_error_rate = degraded_metrics["system_metrics"]["error_rate"]

        # Verify performance degradation
        assert degraded_throughput < baseline_throughput, "Throughput should decrease"
        assert degraded_latency > baseline_latency, "Latency should increase"
        assert degraded_error_rate > baseline_error_rate, "Error rate should increase"

        # Calculate performance impact
        throughput_impact = (
            baseline_throughput - degraded_throughput
        ) / baseline_throughput
        latency_impact = (degraded_latency - baseline_latency) / baseline_latency

        # Performance impact should be significant but not _total
        assert 0.1 < throughput_impact < 0.9, (
            f"Throughput impact should be 10-90%, was {throughput_impact:.2%}"
        )
        assert latency_impact > 0.1, (
            f"Latency impact should be >10%, was {latency_impact:.2%}"
        )

        # Recover and measure recovery
        await integrated_system.stop_failure_injection("api_gateway")
        await asyncio.sleep(0.15)  # Allow time for recovery

        recovered_metrics = await integrated_system.get_metrics()
        recovered_throughput = recovered_metrics["system_metrics"]["throughput"]

        # Performance should improve after recovery
        assert recovered_throughput > degraded_throughput, (
            "Performance should improve after recovery"
        )

    async def test_chaos_test_automation(self, integrated_system):
        """Test automated chaos testing pipeline."""
        # Create chaos test runner
        runner = ChaosTestRunner()

        # Define automated test suite
        automated_suite = ChaosTestSuite(
            suite_name="automated_resilience_suite",
            experiments=[
                ChaosExperiment(
                    name="auto_network_chaos",
                    description="Automated network chaos test",
                    failure_type=FailureType.NETWORK_TIMEOUT,
                    target_service="search_service",
                    duration_seconds=0.1,
                    failure_rate=1.0,
                    blast_radius="single",
                    recovery_time_seconds=0.05,
                    success_criteria=["system_recovers", "graceful_degradation"],
                    rollback_strategy="immediate",
                ),
                ChaosExperiment(
                    name="auto_service_chaos",
                    description="Automated service chaos test",
                    failure_type=FailureType.SERVICE_UNAVAILABLE,
                    target_service="cache",
                    duration_seconds=0.1,
                    failure_rate=1.0,
                    blast_radius="single",
                    recovery_time_seconds=0.05,
                    success_criteria=["system_recovers", "no_data_loss"],
                    rollback_strategy="immediate",
                ),
            ],
            parallel_execution=False,
            prerequisites=["baseline_captured", "monitoring_enabled"],
        )

        runner.register_test_suite(automated_suite)

        # Execute automated test suite
        suite_result = await runner.execute_test_suite(
            "automated_resilience_suite", integrated_system
        )

        # Verify automated execution
        assert suite_result["status"] == "completed"
        assert suite_result["_total_experiments"] == 2
        assert (
            suite_result["successful_experiments"] >= 1
        )  # At least some should succeed

        # Generate automated report
        report = runner.generate_report("automated_resilience_suite")

        # Verify report generation
        assert "summary" in report
        assert "failure_type_analysis" in report
        assert "recommendations" in report
        assert len(report["recommendations"]) > 0

        # Verify system is healthy after automated testing
        final_health = await integrated_system.health_check()
        assert final_health["status"] == "healthy"

    async def test_chaos_monitoring_integration(self, integrated_system):
        """Test integration with monitoring systems during chaos events."""
        monitoring_events = []

        # Mock monitoring system
        async def monitoring_callback():
            """Collect monitoring data during chaos events."""
            metrics = await integrated_system.get_metrics()

            event = {
                "timestamp": time.time(),
                "system_health": "healthy"
                if len(integrated_system.active_failures) == 0
                else "degraded",
                "active_failures": list(integrated_system.active_failures.keys()),
                "throughput": metrics["system_metrics"]["throughput"],
                "error_rate": metrics["system_metrics"]["error_rate"],
                "availability": metrics["system_metrics"]["availability"],
            }

            monitoring_events.append(event)
            return event

        # Execute chaos experiment with monitoring
        await integrated_system.inject_failure(
            FailureType.SERVICE_UNAVAILABLE, "vector_db"
        )

        # Collect monitoring data during failure
        await monitoring_callback()
        await asyncio.sleep(0.05)
        await monitoring_callback()

        # Recover and collect data
        await integrated_system.stop_failure_injection("vector_db")
        await asyncio.sleep(0.1)
        await monitoring_callback()

        # Verify monitoring data was collected
        assert len(monitoring_events) >= 3

        # Verify monitoring detected failure and recovery
        failure_events = [
            e for e in monitoring_events if e["system_health"] == "degraded"
        ]
        recovery_events = [
            e for e in monitoring_events if e["system_health"] == "healthy"
        ]

        assert len(failure_events) > 0, "Monitoring should detect failure"
        assert len(recovery_events) > 0, "Monitoring should detect recovery"

        # Verify metrics changed appropriately
        baseline_event = monitoring_events[0]
        failure_event = next(
            (e for e in monitoring_events if len(e["active_failures"]) > 0), None
        )
        recovery_event = monitoring_events[-1]

        if failure_event:
            assert failure_event["throughput"] < baseline_event["throughput"], (
                "Throughput should decrease during failure"
            )
            assert failure_event["error_rate"] > baseline_event["error_rate"], (
                "Error rate should increase during failure"
            )

        # Recovery should show improvement
        assert recovery_event["system_health"] == "healthy"
        assert len(recovery_event["active_failures"]) == 0
