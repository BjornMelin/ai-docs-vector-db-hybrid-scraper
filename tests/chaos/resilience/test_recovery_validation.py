"""Recovery validation tests for chaos engineering.

This module implements comprehensive recovery validation to test system
ability to recover from failures, maintain data consistency, and resume
normal operations after chaos events.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


class TestError(Exception):
    """Custom exception for this module."""

    pass


class RecoveryStage(Enum):
    """Recovery process stages."""

    DETECTION = "detection"
    ISOLATION = "isolation"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    RESTORATION = "restoration"


@dataclass
class RecoveryMetrics:
    """Metrics for recovery validation."""

    detection_time: float = 0.0
    isolation_time: float = 0.0
    recovery_time: float = 0.0
    validation_time: float = 0.0
    total_recovery_time: float = 0.0
    data_consistency_score: float = 1.0
    service_availability_score: float = 1.0
    performance_degradation: float = 0.0


@dataclass
class SystemState:
    """Represents system state for recovery validation."""

    services: dict[str, Any] = field(default_factory=dict)
    data_stores: dict[str, Any] = field(default_factory=dict)
    connections: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@pytest.mark.chaos
@pytest.mark.resilience
class TestRecoveryValidation:
    """Test recovery validation scenarios."""

    @pytest.fixture
    def system_monitor(self):
        """Mock system monitor for testing."""

        class SystemMonitor:
            def __init__(self):
                self.baseline_state = None
                self.current_state = None
                self.failure_detected = False
                self.recovery_started = False

            async def capture_baseline(self) -> SystemState:
                """Capture baseline system state."""
                self.baseline_state = SystemState(
                    services={
                        "api_gateway": {"status": "healthy", "response_time": 0.05},
                        "search_service": {"status": "healthy", "response_time": 0.1},
                        "vector_db": {"status": "healthy", "response_time": 0.03},
                        "cache": {"status": "healthy", "response_time": 0.01},
                    },
                    data_stores={
                        "primary_db": {"connections": 20, "query_time": 0.02},
                        "vector_store": {"documents": 10000, "embeddings": 10000},
                        "cache_store": {"hit_rate": 0.85, "memory_usage": 0.6},
                    },
                    connections={
                        "external_apis": {"openai": "connected", "search": "connected"},
                        "internal_services": {
                            "auth": "connected",
                            "logging": "connected",
                        },
                    },
                    performance_metrics={
                        "throughput": 1000,  # requests per second
                        "latency_p95": 0.2,
                        "error_rate": 0.01,
                        "cpu_usage": 0.4,
                        "memory_usage": 0.6,
                    },
                )
                return self.baseline_state

            async def monitor_current_state(self) -> SystemState:
                """Monitor current system state."""
                # This would be updated by the monitoring system
                if not self.current_state:
                    self.current_state = self.baseline_state
                return self.current_state

            async def detect_failure(self) -> dict[str, Any]:
                """Detect system failures."""
                current = await self.monitor_current_state()

                failures = []

                # Check service health
                for service, state in current.services.items():
                    if state["status"] != "healthy":
                        failures.append(f"service_{service}_unhealthy")
                    if state["response_time"] > 1.0:  # Threshold
                        failures.append(f"service_{service}_slow")

                # Check data integrity
                if current.data_stores.get("vector_store", {}).get("documents", 0) == 0:
                    failures.append("data_loss_detected")

                # Check performance degradation
                baseline_throughput = self.baseline_state.performance_metrics[
                    "throughput"
                ]
                current_throughput = current.performance_metrics.get("throughput", 0)

                if current_throughput < baseline_throughput * 0.5:  # 50% degradation
                    failures.append("performance_degradation")

                self.failure_detected = len(failures) > 0

                return {
                    "failures_detected": len(failures) > 0,
                    "failure_types": failures,
                    "severity": "high"
                    if len(failures) > 2
                    else "medium"
                    if failures
                    else "low",
                }

            async def validate_recovery(self) -> dict[str, Any]:
                """Validate system recovery."""
                current = await self.monitor_current_state()

                if not self.baseline_state:
                    raise TestError("No baseline state available for comparison")

                validation_results = {
                    "services_recovered": True,
                    "data_consistency": True,
                    "performance_acceptable": True,
                    "recovery_complete": True,
                }

                # Validate services
                for service, baseline in self.baseline_state.services.items():
                    current_service = current.services.get(service, {})
                    if current_service.get("status") != "healthy":
                        validation_results["services_recovered"] = False

                    # Allow some performance degradation during recovery
                    if (
                        current_service.get("response_time", 0)
                        > baseline["response_time"] * 3
                    ):  # 3x slower is unacceptable
                        validation_results["performance_acceptable"] = False

                # Validate data consistency
                baseline_docs = self.baseline_state.data_stores.get(
                    "vector_store", {}
                ).get("documents", 0)
                current_docs = current.data_stores.get("vector_store", {}).get(
                    "documents", 0
                )

                if current_docs < baseline_docs * 0.95:  # Allow 5% data loss
                    validation_results["data_consistency"] = False

                # Overall recovery status
                validation_results["recovery_complete"] = all(
                    [
                        validation_results["services_recovered"],
                        validation_results["data_consistency"],
                        validation_results["performance_acceptable"],
                    ]
                )

                return validation_results

        return SystemMonitor()

    @pytest.fixture
    def recovery_orchestrator(self):
        """Mock recovery orchestrator for testing."""

        class RecoveryOrchestrator:
            def __init__(self):
                self.recovery_procedures = {}
                self.recovery_history = []

            def register_recovery_procedure(
                self, failure_type: str, procedure: Callable
            ):
                """Register recovery procedure for failure type."""
                self.recovery_procedures[failure_type] = procedure

            async def execute_recovery(
                self, failure_type: str, context: dict[str, Any] | None = None
            ) -> RecoveryMetrics:
                """Execute recovery procedure."""
                if failure_type not in self.recovery_procedures:
                    raise TestError(f"No recovery procedure for {failure_type}")

                metrics = RecoveryMetrics()
                start_time = time.time()

                try:
                    # Execute recovery procedure
                    procedure = self.recovery_procedures[failure_type]
                    recovery_result = await procedure(context or {})

                    metrics.total_recovery_time = time.time() - start_time

                    # Extract metrics from recovery result
                    if isinstance(recovery_result, dict):
                        metrics.detection_time = recovery_result.get(
                            "detection_time", 0
                        )
                        metrics.isolation_time = recovery_result.get(
                            "isolation_time", 0
                        )
                        metrics.recovery_time = recovery_result.get("recovery_time", 0)
                        metrics.validation_time = recovery_result.get(
                            "validation_time", 0
                        )
                        metrics.data_consistency_score = recovery_result.get(
                            "data_consistency", 1.0
                        )
                        metrics.service_availability_score = recovery_result.get(
                            "availability", 1.0
                        )
                        metrics.performance_degradation = recovery_result.get(
                            "degradation", 0.0
                        )

                    # Record recovery attempt
                    self.recovery_history.append(
                        {
                            "failure_type": failure_type,
                            "timestamp": start_time,
                            "metrics": metrics,
                            "success": True,
                        }
                    )

                    return metrics

                except Exception as e:
                    # Record failed recovery
                    self.recovery_history.append(
                        {
                            "failure_type": failure_type,
                            "timestamp": start_time,
                            "error": str(e),
                            "success": False,
                        }
                    )
                    raise

        return RecoveryOrchestrator()

    async def test_service_failure_recovery(
        self, system_monitor, recovery_orchestrator, resilience_validator
    ):
        """Test recovery from service failures."""
        # Capture baseline
        baseline = await system_monitor.capture_baseline()

        # Simulate service failure
        system_monitor.current_state = SystemState(
            services={
                "api_gateway": {"status": "healthy", "response_time": 0.05},
                "search_service": {"status": "failed", "response_time": 0.0},  # Failed
                "vector_db": {"status": "healthy", "response_time": 0.03},
                "cache": {"status": "healthy", "response_time": 0.01},
            },
            data_stores=baseline.data_stores.copy(),
            connections=baseline.connections.copy(),
            performance_metrics={
                **baseline.performance_metrics,
                "throughput": 300,  # Reduced due to service failure
                "error_rate": 0.15,  # Higher error rate
            },
        )

        # Detect failure
        failure_detection = await system_monitor.detect_failure()
        assert failure_detection["failures_detected"]
        assert "service_search_service_unhealthy" in failure_detection["failure_types"]

        # Define recovery procedure
        async def search_service_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Recovery procedure for search service."""
            time.time()

            # Step 1: Detection (already done)
            detection_time = 0.05

            # Step 2: Isolation - stop sending traffic to failed service
            isolation_start = time.time()
            await asyncio.sleep(0.02)  # Simulate isolation
            isolation_time = time.time() - isolation_start

            # Step 3: Recovery - restart service
            recovery_start = time.time()
            await asyncio.sleep(0.1)  # Simulate service restart

            # Update system state to reflect recovery
            system_monitor.current_state.services["search_service"] = {
                "status": "healthy",
                "response_time": 0.15,  # Slightly slower after restart
            }
            system_monitor.current_state.performance_metrics.update(
                {
                    "throughput": 900,  # Partially recovered
                    "error_rate": 0.02,  # Reduced errors
                }
            )

            recovery_time = time.time() - recovery_start

            # Step 4: Validation
            validation_start = time.time()
            await system_monitor.validate_recovery()
            validation_time = time.time() - validation_start

            return {
                "detection_time": detection_time,
                "isolation_time": isolation_time,
                "recovery_time": recovery_time,
                "validation_time": validation_time,
                "data_consistency": 1.0,
                "availability": 0.9,  # 90% availability during recovery
                "degradation": 0.1,  # 10% performance degradation
            }

        # Register and execute recovery
        recovery_orchestrator.register_recovery_procedure(
            "service_failure", search_service_recovery
        )
        metrics = await recovery_orchestrator.execute_recovery("service_failure")

        # Validate recovery metrics
        assert metrics.total_recovery_time > 0
        assert metrics.detection_time > 0
        assert metrics.isolation_time > 0
        assert metrics.recovery_time > 0
        assert metrics.validation_time > 0
        assert metrics.data_consistency_score >= 0.95  # High data consistency expected
        assert metrics.service_availability_score >= 0.8  # Reasonable availability

        # Validate final system state
        final_validation = await system_monitor.validate_recovery()
        assert final_validation["services_recovered"]
        assert final_validation["recovery_complete"]

    async def test_data_corruption_recovery(
        self, system_monitor, recovery_orchestrator
    ):
        """Test recovery from data corruption."""
        # Capture baseline
        baseline = await system_monitor.capture_baseline()

        # Simulate data corruption
        system_monitor.current_state = SystemState(
            services=baseline.services.copy(),
            data_stores={
                "primary_db": baseline.data_stores["primary_db"].copy(),
                "vector_store": {
                    "documents": 5000,  # Lost half the documents
                    "embeddings": 5000,  # Lost half the embeddings
                },
                "cache_store": baseline.data_stores["cache_store"].copy(),
            },
            connections=baseline.connections.copy(),
            performance_metrics=baseline.performance_metrics.copy(),
        )

        # Mock backup system
        backup_data = {
            "vector_store": {
                "documents": 10000,
                "embeddings": 10000,
                "last_backup": time.time() - 300,  # 5 minutes ago
            }
        }

        # Define data recovery procedure
        async def data_corruption_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Recovery procedure for data corruption."""

            # Step 1: Detect corruption
            detection_start = time.time()
            await system_monitor.detect_failure()
            detection_time = time.time() - detection_start

            # Step 2: Stop writes to prevent further corruption
            isolation_start = time.time()
            await asyncio.sleep(0.01)  # Simulate stopping writes
            isolation_time = time.time() - isolation_start

            # Step 3: Restore from backup
            recovery_start = time.time()
            await asyncio.sleep(0.2)  # Simulate data restoration

            # Update system state with restored data
            system_monitor.current_state.data_stores["vector_store"] = backup_data[
                "vector_store"
            ].copy()
            recovery_time = time.time() - recovery_start

            # Step 4: Validate data integrity
            validation_start = time.time()
            await asyncio.sleep(0.05)  # Simulate integrity check
            validation_time = time.time() - validation_start

            # Calculate data loss
            backup_age = time.time() - backup_data["vector_store"]["last_backup"]
            data_loss_score = max(0, 1.0 - (backup_age / 3600))  # 1 hour = full loss

            return {
                "detection_time": detection_time,
                "isolation_time": isolation_time,
                "recovery_time": recovery_time,
                "validation_time": validation_time,
                "data_consistency": data_loss_score,
                "availability": 0.7,  # Reduced during restoration
                "degradation": 0.3,  # Significant degradation during recovery
            }

        # Execute data recovery
        recovery_orchestrator.register_recovery_procedure(
            "data_corruption", data_corruption_recovery
        )
        metrics = await recovery_orchestrator.execute_recovery("data_corruption")

        # Validate recovery
        assert metrics.data_consistency_score > 0.8  # Should recover most data
        assert metrics.total_recovery_time < 5.0  # Should complete reasonably quickly

        # Verify data restoration
        final_state = await system_monitor.monitor_current_state()
        assert final_state.data_stores["vector_store"]["documents"] == 10000
        assert final_state.data_stores["vector_store"]["embeddings"] == 10000

    async def test_cascade_failure_recovery(
        self, system_monitor, recovery_orchestrator
    ):
        """Test recovery from cascade failures."""
        baseline = await system_monitor.capture_baseline()

        # Simulate cascade failure (multiple services affected)
        system_monitor.current_state = SystemState(
            services={
                "api_gateway": {"status": "degraded", "response_time": 2.0},  # Slow
                "search_service": {"status": "failed", "response_time": 0.0},  # Failed
                "vector_db": {"status": "degraded", "response_time": 1.5},  # Slow
                "cache": {"status": "failed", "response_time": 0.0},  # Failed
            },
            data_stores=baseline.data_stores.copy(),
            connections={
                "external_apis": {"openai": "connected", "search": "timeout"},
                "internal_services": {"auth": "degraded", "logging": "failed"},
            },
            performance_metrics={
                "throughput": 100,  # Severely reduced
                "latency_p95": 5.0,  # Very high latency
                "error_rate": 0.6,  # High error rate
                "cpu_usage": 0.9,  # High CPU usage
                "memory_usage": 0.95,  # High memory usage
            },
        )

        # Define cascade recovery procedure
        async def cascade_failure_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Recovery procedure for cascade failures."""

            # Step 1: Rapid detection
            detection_start = time.time()
            await system_monitor.detect_failure()
            detection_time = time.time() - detection_start

            # Step 2: Prioritized isolation and recovery
            isolation_start = time.time()

            # Priority 1: Critical services (cache, then vector_db)
            await asyncio.sleep(0.05)  # Restart cache
            system_monitor.current_state.services["cache"] = {
                "status": "healthy",
                "response_time": 0.02,
            }

            await asyncio.sleep(0.1)  # Restart vector_db
            system_monitor.current_state.services["vector_db"] = {
                "status": "healthy",
                "response_time": 0.05,
            }

            isolation_time = time.time() - isolation_start

            # Step 3: Dependent services recovery
            recovery_start = time.time()

            await asyncio.sleep(0.08)  # Restart search_service
            system_monitor.current_state.services["search_service"] = {
                "status": "healthy",
                "response_time": 0.12,
            }

            await asyncio.sleep(0.03)  # API gateway should recover automatically
            system_monitor.current_state.services["api_gateway"] = {
                "status": "healthy",
                "response_time": 0.08,
            }

            # Update system performance
            system_monitor.current_state.performance_metrics.update(
                {
                    "throughput": 800,  # Mostly recovered
                    "latency_p95": 0.3,  # Acceptable latency
                    "error_rate": 0.05,  # Low error rate
                    "cpu_usage": 0.5,  # Normal CPU
                    "memory_usage": 0.7,  # Normal memory
                }
            )

            recovery_time = time.time() - recovery_start

            # Step 4: Comprehensive validation
            validation_start = time.time()
            await system_monitor.validate_recovery()
            validation_time = time.time() - validation_start

            return {
                "detection_time": detection_time,
                "isolation_time": isolation_time,
                "recovery_time": recovery_time,
                "validation_time": validation_time,
                "data_consistency": 1.0,  # No data loss
                "availability": 0.75,  # Reduced during recovery
                "degradation": 0.2,  # Some performance impact
            }

        # Execute cascade recovery
        recovery_orchestrator.register_recovery_procedure(
            "cascade_failure", cascade_failure_recovery
        )
        metrics = await recovery_orchestrator.execute_recovery("cascade_failure")

        # Validate cascade recovery
        assert (
            metrics.total_recovery_time < 10.0
        )  # Should recover within reasonable time
        assert metrics.service_availability_score >= 0.7  # Reasonable availability

        # Verify all services recovered
        final_validation = await system_monitor.validate_recovery()
        assert final_validation["services_recovered"]
        assert final_validation["performance_acceptable"]

    async def test_partial_recovery_scenarios(
        self, system_monitor, recovery_orchestrator
    ):
        """Test scenarios where recovery is only partial."""
        baseline = await system_monitor.capture_baseline()

        # Simulate scenario where some components cannot fully recover
        system_monitor.current_state = SystemState(
            services={
                "api_gateway": {"status": "healthy", "response_time": 0.05},
                "search_service": {
                    "status": "degraded",
                    "response_time": 0.5,
                },  # Partial recovery
                "vector_db": {
                    "status": "failed",
                    "response_time": 0.0,
                },  # Cannot recover
                "cache": {"status": "healthy", "response_time": 0.01},
            },
            data_stores={
                "primary_db": baseline.data_stores["primary_db"].copy(),
                "vector_store": {
                    "documents": 7000,  # Partial data loss
                    "embeddings": 7000,
                },
                "cache_store": baseline.data_stores["cache_store"].copy(),
            },
            connections=baseline.connections.copy(),
            performance_metrics={
                "throughput": 500,  # Reduced capacity
                "latency_p95": 0.8,  # Higher latency
                "error_rate": 0.1,  # Elevated error rate
                "cpu_usage": 0.6,
                "memory_usage": 0.7,
            },
        )

        # Define partial recovery procedure
        async def partial_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Recovery procedure that achieves partial recovery."""

            # Attempt recovery but some components remain degraded
            await asyncio.sleep(0.1)

            # Search service improves but doesn't fully recover
            system_monitor.current_state.services["search_service"] = {
                "status": "degraded",
                "response_time": 0.3,
            }

            # Vector DB cannot be recovered - implement fallback
            # Use cached/degraded search capabilities
            system_monitor.current_state.performance_metrics.update(
                {
                    "throughput": 600,  # Improved but not full
                    "latency_p95": 0.6,  # Better but still high
                    "error_rate": 0.08,  # Reduced errors
                }
            )

            return {
                "detection_time": 0.02,
                "isolation_time": 0.03,
                "recovery_time": 0.1,
                "validation_time": 0.02,
                "data_consistency": 0.7,  # 30% data loss
                "availability": 0.6,  # Partial availability
                "degradation": 0.4,  # Significant degradation remains
            }

        # Execute partial recovery
        recovery_orchestrator.register_recovery_procedure(
            "partial_failure", partial_recovery
        )
        metrics = await recovery_orchestrator.execute_recovery("partial_failure")

        # Validate partial recovery acceptance
        assert metrics.data_consistency_score >= 0.6  # Acceptable partial recovery
        assert metrics.service_availability_score >= 0.5  # Partial service

        # Verify graceful degradation is in effect
        final_state = await system_monitor.monitor_current_state()
        assert final_state.services["search_service"]["status"] == "degraded"
        assert final_state.performance_metrics["throughput"] > 0  # Still functional

    async def test_recovery_time_objectives(
        self, _system_monitor, recovery_orchestrator
    ):
        """Test recovery time objectives (RTO) compliance."""

        # Define RTO requirements for different failure types

        # Mock fast recovery procedure
        async def fast_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Fast recovery procedure for RTO testing."""
            start_time = time.time()

            # Simulate rapid recovery
            await asyncio.sleep(0.05)  # 50ms recovery

            return {
                "detection_time": 0.01,
                "isolation_time": 0.01,
                "recovery_time": 0.02,
                "validation_time": 0.01,
                "total_time": time.time() - start_time,
                "data_consistency": 1.0,
                "availability": 0.95,
                "degradation": 0.05,
            }

        # Mock slow recovery procedure
        async def slow_recovery(_context: dict[str, Any]) -> dict[str, Any]:
            """Slow recovery procedure for RTO testing."""
            start_time = time.time()

            # Simulate slow recovery that might exceed RTO
            await asyncio.sleep(0.2)  # 200ms recovery

            return {
                "detection_time": 0.05,
                "isolation_time": 0.05,
                "recovery_time": 0.1,
                "validation_time": 0.05,
                "total_time": time.time() - start_time,
                "data_consistency": 1.0,
                "availability": 0.9,
                "degradation": 0.1,
            }

        # Test fast recovery meets RTO
        recovery_orchestrator.register_recovery_procedure("fast_failure", fast_recovery)
        fast_metrics = await recovery_orchestrator.execute_recovery("fast_failure")

        # For testing, scale down RTO requirements
        test_rto = 0.1  # 100ms for testing
        assert fast_metrics.total_recovery_time <= test_rto, (
            f"Fast recovery should meet RTO: {fast_metrics.total_recovery_time} > {test_rto}"
        )

        # Test slow recovery might exceed RTO
        recovery_orchestrator.register_recovery_procedure("slow_failure", slow_recovery)
        slow_metrics = await recovery_orchestrator.execute_recovery("slow_failure")

        # Slow recovery might exceed tight RTO

        # Even if RTO is exceeded, recovery should still succeed
        assert slow_metrics.data_consistency_score >= 0.9
        assert slow_metrics.service_availability_score >= 0.8

    async def test_recovery_point_objectives(
        self, _system_monitor, recovery_orchestrator
    ):
        """Test recovery point objectives (RPO) compliance."""

        # Mock data with timestamps for RPO testing
        mock_data_timeline = [
            {"timestamp": time.time() - 600, "data": "old_data_1"},  # 10 minutes ago
            {"timestamp": time.time() - 300, "data": "recent_data_1"},  # 5 minutes ago
            {"timestamp": time.time() - 60, "data": "current_data_1"},  # 1 minute ago
            {"timestamp": time.time() - 10, "data": "latest_data_1"},  # 10 seconds ago
        ]

        # Define RPO requirements

        async def rpo_compliant_recovery(context: dict[str, Any]) -> dict[str, Any]:
            """Recovery that meets RPO requirements."""
            rpo_target = context.get("rpo_target", 300.0)  # 5 minutes default

            # Find latest backup within RPO window
            cutoff_time = time.time() - rpo_target
            recoverable_data = [
                item for item in mock_data_timeline if item["timestamp"] >= cutoff_time
            ]

            # Calculate data loss
            total_data_points = len(mock_data_timeline)
            recovered_data_points = len(recoverable_data)
            data_recovery_ratio = (
                recovered_data_points / total_data_points
                if total_data_points > 0
                else 0
            )

            return {
                "detection_time": 0.01,
                "isolation_time": 0.02,
                "recovery_time": 0.05,
                "validation_time": 0.02,
                "data_consistency": data_recovery_ratio,
                "data_points_lost": total_data_points - recovered_data_points,
                "rpo_compliance": True,
                "availability": 0.9,
                "degradation": 0.1,
            }

        # Test RPO compliance with different targets
        recovery_orchestrator.register_recovery_procedure(
            "rpo_test", rpo_compliant_recovery
        )

        # Test with strict RPO (1 minute)
        strict_metrics = await recovery_orchestrator.execute_recovery(
            "rpo_test", {"rpo_target": 60.0}
        )

        # Should recover recent data within 1 minute RPO
        assert strict_metrics.data_consistency_score >= 0.5  # At least 50% of data

        # Test with lenient RPO (10 minutes)
        lenient_metrics = await recovery_orchestrator.execute_recovery(
            "rpo_test", {"rpo_target": 600.0}
        )

        # Should recover all data within 10 minute RPO
        assert lenient_metrics.data_consistency_score >= 0.9  # 90%+ of data

    async def test_automated_recovery_validation(
        self, system_monitor, recovery_orchestrator
    ):
        """Test automated recovery validation and health checks."""

        class AutomatedValidator:
            def __init__(self):
                self.validation_checks = []

            def register_check(self, check_name: str, check_func: Callable):
                """Register automated validation check."""
                self.validation_checks.append((check_name, check_func))

            async def run_validation_suite(self) -> dict[str, Any]:
                """Run all registered validation checks."""
                results = {}

                for check_name, check_func in self.validation_checks:
                    try:
                        result = await check_func()
                        results[check_name] = {"status": "pass", "result": result}
                    except Exception as e:
                        results[check_name] = {"status": "fail", "error": str(e)}

                # Calculate overall health score
                passed_checks = len(
                    [r for r in results.values() if r["status"] == "pass"]
                )
                health_score = passed_checks / len(results) if results else 0

                return {
                    "overall_health": health_score,
                    "checks": results,
                    "validation_complete": True,
                }

        validator = AutomatedValidator()

        # Register validation checks
        async def service_health_check():
            """Check all services are healthy."""
            state = await system_monitor.monitor_current_state()
            unhealthy_services = [
                name
                for name, service in state.services.items()
                if service.get("status") != "healthy"
            ]
            if unhealthy_services:
                raise TestError(f"Unhealthy services: {unhealthy_services}")
            return {"healthy_services": len(state.services)}

        async def performance_check():
            """Check system performance is acceptable."""
            state = await system_monitor.monitor_current_state()
            if state.performance_metrics.get("error_rate", 0) > 0.1:
                raise TestError("Error rate too high")
            if state.performance_metrics.get("latency_p95", 0) > 1.0:
                raise TestError("Latency too high")
            return {"performance": "acceptable"}

        async def data_integrity_check():
            """Check data integrity."""
            state = await system_monitor.monitor_current_state()
            docs = state.data_stores.get("vector_store", {}).get("documents", 0)
            if docs < 5000:  # Minimum acceptable data
                raise TestError("Insufficient data available")
            return {"documents": docs}

        validator.register_check("service_health", service_health_check)
        validator.register_check("performance", performance_check)
        validator.register_check("data_integrity", data_integrity_check)

        # Set up healthy system state
        await system_monitor.capture_baseline()

        # Run validation suite
        validation_results = await validator.run_validation_suite()

        # Verify automated validation works
        assert validation_results["validation_complete"]
        assert validation_results["overall_health"] >= 0.8  # 80% of checks should pass
        assert "service_health" in validation_results["checks"]
        assert "performance" in validation_results["checks"]
        assert "data_integrity" in validation_results["checks"]

        # Test validation with degraded system
        system_monitor.current_state.services["search_service"]["status"] = "degraded"
        system_monitor.current_state.performance_metrics["error_rate"] = 0.15

        degraded_results = await validator.run_validation_suite()

        # Should detect degraded state
        assert degraded_results["overall_health"] < 0.8  # Lower health score
        assert degraded_results["checks"]["performance"]["status"] == "fail"
