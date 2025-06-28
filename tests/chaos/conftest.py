"""Chaos engineering fixtures and configuration.

This module provides pytest fixtures for comprehensive chaos engineering testing including
fault injection, failure scenarios, network chaos, dependency failures, resource exhaustion,
and resilience validation.
"""

import asyncio
import random
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class CustomError(Exception):
    """Custom exception for this module."""

    pass


class FailureType(Enum):
    """Types of failures that can be injected."""

    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_REFUSED = "connection_refused"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    DATABASE_FAILURE = "database_failure"
    AUTHENTICATION_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    PARTIAL_FAILURE = "partial_failure"


@dataclass
class ChaosExperiment:
    """Definition of a chaos engineering experiment."""

    name: str
    description: str
    failure_type: FailureType
    target_service: str
    duration_seconds: float
    failure_rate: float  # 0.0 to 1.0
    blast_radius: str  # "single", "service", "system"
    recovery_time_seconds: float
    success_criteria: list[str]
    rollback_strategy: str


@dataclass
class ExperimentResult:
    """Result of a chaos engineering experiment."""

    experiment_name: str
    started_at: float
    ended_at: float
    duration: float
    failure_injected: bool
    system_recovered: bool
    recovery_time: float
    success_criteria_met: list[bool]
    metrics: dict[str, Any]
    errors: list[str]


@pytest.fixture(scope="session")
def chaos_test_config():
    """Provide chaos engineering test configuration."""
    return {
        "experiments": {
            "max_duration_seconds": 300,
            "min_recovery_time_seconds": 5,
            "max_failure_rate": 0.5,  # Never fail more than 50% of requests
            "safety_mode": True,  # Enable safety checks
        },
        "targets": {
            "services": ["vector_db", "cache", "embedding_service", "web_scraper"],
            "dependencies": ["qdrant", "redis", "openai_api", "external_apis"],
            "infrastructure": ["network", "disk", "memory", "cpu"],
        },
        "monitoring": {
            "metrics_interval_seconds": 1.0,
            "alert_thresholds": {
                "error_rate": 0.1,  # 10% error rate
                "response_time_p95": 5.0,  # 5 seconds
                "system_recovery_time": 60.0,  # 1 minute
            },
        },
        "safety": {
            "circuit_breaker_threshold": 0.2,
            "auto_stop_on_critical_failure": True,
            "max_concurrent_experiments": 1,
        },
    }


@pytest.fixture
def fault_injector():
    """Fault injection utilities for chaos testing."""

    class FaultInjector:
        def __init__(self):
            self.active_faults = {}
            self.fault_history = []

        async def inject_network_timeout(
            self, target: str, timeout_seconds: float = 30.0, failure_rate: float = 1.0
        ):
            """Inject network timeout failures."""
            fault_id = f"network_timeout_{target}_{time.time()}"

            async def timeout_fault():
                if random.random() < failure_rate:
                    await asyncio.sleep(timeout_seconds)
                    raise TimeoutError(f"Simulated timeout for {target}")

            self.active_faults[fault_id] = {
                "type": FailureType.NETWORK_TIMEOUT,
                "target": target,
                "fault_func": timeout_fault,
                "started_at": time.time(),
            }

            return fault_id

        async def inject_connection_failure(
            self, target: str, failure_rate: float = 1.0
        ):
            """Inject connection failures."""
            fault_id = f"connection_failure_{target}_{time.time()}"

            async def connection_fault():
                if random.random() < failure_rate:
                    raise ConnectionError(f"Simulated connection failure for {target}")

            self.active_faults[fault_id] = {
                "type": FailureType.CONNECTION_REFUSED,
                "target": target,
                "fault_func": connection_fault,
                "started_at": time.time(),
            }

            return fault_id

        async def inject_service_unavailable(
            self, target: str, failure_rate: float = 1.0
        ):
            """Inject service unavailable responses."""
            fault_id = f"service_unavailable_{target}_{time.time()}"

            async def service_fault():
                raise CustomError(f"Service {target} is temporarily unavailable")
                raise CustomError(f"Service {target} is temporarily unavailable")

            self.active_faults[fault_id] = {
                "type": FailureType.SERVICE_UNAVAILABLE,
                "target": target,
                "fault_func": service_fault,
                "started_at": time.time(),
            }

            return fault_id

        async def inject_partial_failure(self, target: str, success_rate: float = 0.7):
            """Inject partial failures (some requests succeed, some fail)."""
            fault_id = f"partial_failure_{target}_{time.time()}"

            async def partial_fault():
                if random.random() > success_rate:
                    failure_types = [
                        "Internal server error",
                        "Temporary service degradation",
                        "Rate limit exceeded",
                        "Partial data corruption",
                    ]
                    raise CustomError(f"{random.choice(failure_types)} for {target}")

            self.active_faults[fault_id] = {
                "type": FailureType.PARTIAL_FAILURE,
                "target": target,
                "fault_func": partial_fault,
                "started_at": time.time(),
            }

            return fault_id

        async def inject_latency_spike(
            self, target: str, latency_seconds: float = 5.0, spike_rate: float = 0.3
        ):
            """Inject latency spikes."""
            fault_id = f"latency_spike_{target}_{time.time()}"

            async def latency_fault():
                if random.random() < spike_rate:
                    await asyncio.sleep(latency_seconds)

            self.active_faults[fault_id] = {
                "type": "latency_spike",
                "target": target,
                "fault_func": latency_fault,
                "started_at": time.time(),
            }

            return fault_id

        async def inject_memory_pressure(
            self, target: str, pressure_level: str = "moderate"
        ):
            """Inject memory pressure (simulated)."""
            fault_id = f"memory_pressure_{target}_{time.time()}"

            # This is a simulation - in real chaos testing, you'd use tools like stress-ng
            pressure_levels = {
                "low": 0.1,
                "moderate": 0.3,
                "high": 0.6,
                "critical": 0.9,
            }

            failure_rate = pressure_levels.get(pressure_level, 0.3)

            async def memory_fault():
                if random.random() < failure_rate:
                    raise MemoryError(
                        f"Simulated memory pressure ({pressure_level}) for {target}"
                    )

            self.active_faults[fault_id] = {
                "type": FailureType.MEMORY_EXHAUSTION,
                "target": target,
                "fault_func": memory_fault,
                "started_at": time.time(),
            }

            return fault_id

        def remove_fault(self, fault_id: str):
            """Remove an active fault."""
            if fault_id in self.active_faults:
                fault = self.active_faults.pop(fault_id)
                fault["ended_at"] = time.time()
                fault["duration"] = fault["ended_at"] - fault["started_at"]
                self.fault_history.append(fault)

        def clear_all_faults(self):
            """Clear all active faults."""
            for fault_id in list(self.active_faults.keys()):
                self.remove_fault(fault_id)

        @asynccontextmanager
        async def temporary_fault(
            self, fault_type: str, target: str, duration_seconds: float, **kwargs
        ):
            """Context manager for temporary fault injection."""
            # Inject fault based on type
            if fault_type == "network_timeout":
                fault_id = await self.inject_network_timeout(target, **kwargs)
            elif fault_type == "connection_failure":
                fault_id = await self.inject_connection_failure(target, **kwargs)
            elif fault_type == "service_unavailable":
                fault_id = await self.inject_service_unavailable(target, **kwargs)
            elif fault_type == "partial_failure":
                fault_id = await self.inject_partial_failure(target, **kwargs)
            elif fault_type == "latency_spike":
                fault_id = await self.inject_latency_spike(target, **kwargs)
            elif fault_type == "memory_pressure":
                fault_id = await self.inject_memory_pressure(target, **kwargs)
            else:
                raise ValueError(f"Unknown fault type: {fault_type}")

            try:
                # Let the fault run for the specified duration
                await asyncio.sleep(duration_seconds)
                yield fault_id
            finally:
                # Clean up the fault
                self.remove_fault(fault_id)

        def get_active_faults(self) -> dict[str, Any]:
            """Get information about active faults."""
            return self.active_faults

        def get_fault_history(self) -> list[dict[str, Any]]:
            """Get history of injected faults."""
            return self.fault_history

    return FaultInjector()


@pytest.fixture
def resilience_validator():
    """Resilience validation utilities."""

    class ResilienceValidator:
        def __init__(self):
            self.test_results = []
            self.recovery_times = []

        async def validate_circuit_breaker(
            self,
            service_func: Callable,
            failure_threshold: int = 5,
            recovery_timeout: float = 30.0,
        ) -> dict[str, Any]:
            """Validate circuit breaker behavior."""
            results = {
                "circuit_breaker_triggered": False,
                "recovery_successful": False,
                "failure_count": 0,
                "recovery_time": None,
            }

            start_time = time.time()

            # Simulate failures to trigger circuit breaker
            for _i in range(failure_threshold + 2):
                try:
                    await service_func()
                except Exception:
                    results["failure_count"] += 1

                    # Check if circuit breaker should be triggered
                    if results["failure_count"] >= failure_threshold:
                        results["circuit_breaker_triggered"] = True
                        break

            # Wait for recovery period
            if results["circuit_breaker_triggered"]:
                await asyncio.sleep(recovery_timeout)

                # Test recovery
                try:
                    await service_func()
                    results["recovery_successful"] = True
                    results["recovery_time"] = time.time() - start_time
                except Exception:
                    results["recovery_successful"] = False

            return results

        async def validate_retry_behavior(
            self,
            service_func: Callable,
            max_retries: int = 3,
            backoff_factor: float = 1.0,
        ) -> dict[str, Any]:
            """Validate retry behavior."""
            results = {
                "retry_attempts": 0,
                "total_time": 0.0,
                "success_on_retry": False,
                "backoff_respected": True,
            }

            start_time = time.time()
            last_attempt_time = start_time

            for attempt in range(max_retries + 1):
                results["retry_attempts"] = attempt + 1
                attempt_start = time.time()

                # Check backoff timing
                if attempt > 0:
                    expected_delay = backoff_factor * (2 ** (attempt - 1))
                    actual_delay = attempt_start - last_attempt_time
                    if actual_delay < expected_delay * 0.8:  # Allow 20% tolerance
                        results["backoff_respected"] = False

                last_attempt_time = attempt_start

                try:
                    await service_func()
                    results["success_on_retry"] = True
                    break
                except Exception:
                    if attempt < max_retries:
                        await asyncio.sleep(backoff_factor * (2**attempt))

            results["total_time"] = time.time() - start_time
            return results

        async def validate_graceful_degradation(
            self, service_func: Callable, fallback_func: Callable
        ) -> dict[str, Any]:
            """Validate graceful degradation behavior."""
            results = {
                "primary_service_failed": False,
                "fallback_triggered": False,
                "fallback_successful": False,
                "response_time": 0.0,
            }

            start_time = time.time()

            try:
                await service_func()
            except Exception:
                results["primary_service_failed"] = True
                results["fallback_triggered"] = True

                try:
                    await fallback_func()
                    results["fallback_successful"] = True
                except Exception:
                    results["fallback_successful"] = False

            results["response_time"] = time.time() - start_time
            return results

        async def measure_system_recovery(
            self,
            health_check_func: Callable,
            recovery_timeout: float = 60.0,
            check_interval: float = 1.0,
        ) -> dict[str, Any]:
            """Measure system recovery time after failure."""
            start_time = time.time()
            recovery_time = None
            health_checks = 0

            while time.time() - start_time < recovery_timeout:
                health_checks += 1

                try:
                    await health_check_func()
                    recovery_time = time.time() - start_time
                    break
                except Exception:
                    await asyncio.sleep(check_interval)

            return {
                "recovered": recovery_time is not None,
                "recovery_time": recovery_time,
                "health_checks_performed": health_checks,
                "timeout_exceeded": recovery_time is None,
            }

        def calculate_resilience_score(
            self, test_results: list[dict[str, Any]]
        ) -> dict[str, Any]:
            """Calculate overall resilience score."""
            if not test_results:
                return {"score": 0.0, "breakdown": {}}

            scores = {
                "circuit_breaker": 0.0,
                "retry_behavior": 0.0,
                "graceful_degradation": 0.0,
                "recovery_time": 0.0,
            }

            # Circuit breaker scoring
            cb_results = [r for r in test_results if "circuit_breaker_triggered" in r]
            if cb_results:
                cb_triggered = sum(
                    1 for r in cb_results if r["circuit_breaker_triggered"]
                )
                cb_recovered = sum(
                    1 for r in cb_results if r.get("recovery_successful", False)
                )
                scores["circuit_breaker"] = (
                    (cb_triggered + cb_recovered) / (len(cb_results) * 2) * 100
                )

            # Retry behavior scoring
            retry_results = [r for r in test_results if "retry_attempts" in r]
            if retry_results:
                successful_retries = sum(
                    1 for r in retry_results if r.get("success_on_retry", False)
                )
                proper_backoff = sum(
                    1 for r in retry_results if r.get("backoff_respected", False)
                )
                scores["retry_behavior"] = (
                    (successful_retries + proper_backoff)
                    / (len(retry_results) * 2)
                    * 100
                )

            # Graceful degradation scoring
            degradation_results = [r for r in test_results if "fallback_triggered" in r]
            if degradation_results:
                successful_fallbacks = sum(
                    1
                    for r in degradation_results
                    if r.get("fallback_successful", False)
                )
                scores["graceful_degradation"] = (
                    successful_fallbacks / len(degradation_results) * 100
                )

            # Recovery time scoring
            recovery_results = [r for r in test_results if "recovery_time" in r]
            if recovery_results:
                avg_recovery_time = sum(
                    r["recovery_time"] or 60.0 for r in recovery_results
                ) / len(recovery_results)
                # Score based on recovery time (lower is better)
                scores["recovery_time"] = max(0, 100 - (avg_recovery_time / 60.0) * 100)

            overall_score = sum(scores.values()) / len(
                [s for s in scores.values() if s > 0]
            )

            return {
                "overall_score": round(overall_score, 2),
                "breakdown": {k: round(v, 2) for k, v in scores.items()},
                "grade": self._get_resilience_grade(overall_score),
            }

        def _get_resilience_grade(self, score: float) -> str:
            """Convert resilience score to letter grade."""
            if score >= 90:
                return "A"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"

    return ResilienceValidator()


@pytest.fixture
def chaos_experiment_runner():
    """Chaos experiment execution utilities."""

    class ChaosExperimentRunner:
        def __init__(self):
            self.experiments = []
            self.running_experiments = {}

        def define_experiment(
            self,
            name: str,
            description: str,
            failure_type: FailureType,
            target_service: str,
            **kwargs,
        ) -> ChaosExperiment:
            """Define a chaos experiment."""
            experiment = ChaosExperiment(
                name=name,
                description=description,
                failure_type=failure_type,
                target_service=target_service,
                duration_seconds=kwargs.get("duration_seconds", 30.0),
                failure_rate=kwargs.get("failure_rate", 1.0),
                blast_radius=kwargs.get("blast_radius", "single"),
                recovery_time_seconds=kwargs.get("recovery_time_seconds", 10.0),
                success_criteria=kwargs.get("success_criteria", []),
                rollback_strategy=kwargs.get("rollback_strategy", "immediate"),
            )

            self.experiments.append(experiment)
            return experiment

        async def run_experiment(
            self,
            experiment: ChaosExperiment,
            target_system: Any,
            monitoring_func: Callable | None = None,
        ) -> ExperimentResult:
            """Run a chaos experiment."""
            start_time = time.time()
            result = ExperimentResult(
                experiment_name=experiment.name,
                started_at=start_time,
                ended_at=0.0,
                duration=0.0,
                failure_injected=False,
                system_recovered=False,
                recovery_time=0.0,
                success_criteria_met=[],
                metrics={},
                errors=[],
            )

            try:
                # Mark experiment as running
                self.running_experiments[experiment.name] = experiment

                # Inject failure
                await self._inject_failure(experiment, target_system)
                result.failure_injected = True

                # Monitor system during failure
                if monitoring_func:
                    metrics = await monitoring_func()
                    result.metrics.update(metrics)

                # Wait for experiment duration
                await asyncio.sleep(experiment.duration_seconds)

                # Stop failure injection
                await self._stop_failure_injection(experiment, target_system)

                # Wait for recovery
                recovery_start = time.time()
                await asyncio.sleep(experiment.recovery_time_seconds)

                # Check system recovery
                result.system_recovered = await self._check_system_recovery(
                    target_system
                )
                result.recovery_time = time.time() - recovery_start

                # Evaluate success criteria
                result.success_criteria_met = await self._evaluate_success_criteria(
                    experiment, target_system
                )

            except Exception as e:
                result.errors.append(str(e))

            finally:
                # Clean up
                if experiment.name in self.running_experiments:
                    del self.running_experiments[experiment.name]

                result.ended_at = time.time()
                result.duration = result.ended_at - result.started_at

            return result

        async def _inject_failure(
            self, experiment: ChaosExperiment, target_system: Any
        ):
            """Inject failure based on experiment configuration."""
            # This is a simplified implementation
            # In practice, you'd use specialized chaos engineering tools

            if experiment.failure_type == FailureType.NETWORK_TIMEOUT:
                # Simulate network timeout
                if hasattr(target_system, "simulate_network_timeout"):
                    await target_system.simulate_network_timeout(
                        experiment.duration_seconds
                    )

            elif experiment.failure_type == FailureType.SERVICE_UNAVAILABLE:
                # Simulate service unavailable
                if hasattr(target_system, "simulate_service_unavailable"):
                    await target_system.simulate_service_unavailable()

            elif experiment.failure_type == FailureType.MEMORY_EXHAUSTION:
                # Simulate memory pressure
                if hasattr(target_system, "simulate_memory_pressure"):
                    await target_system.simulate_memory_pressure()

        async def _stop_failure_injection(
            self, _experiment: ChaosExperiment, target_system: Any
        ):
            """Stop failure injection."""
            if hasattr(target_system, "stop_failure_simulation"):
                await target_system.stop_failure_simulation()

        async def _check_system_recovery(self, target_system: Any) -> bool:
            """Check if system has recovered from failure."""
            if hasattr(target_system, "health_check"):
                try:
                    await target_system.health_check()
                except Exception:
                    return False
                else:
                    return True

            # Default: assume recovered
            return True

        async def _evaluate_success_criteria(
            self, experiment: ChaosExperiment, target_system: Any
        ) -> list[bool]:
            """Evaluate success criteria for the experiment."""
            results = []

            for criterion in experiment.success_criteria:
                if criterion == "system_recovers":
                    results.append(await self._check_system_recovery(target_system))
                elif criterion == "no_data_loss":
                    # Check for data consistency
                    if hasattr(target_system, "check_data_integrity"):
                        results.append(await target_system.check_data_integrity())
                    else:
                        results.append(True)  # Assume no data loss
                elif criterion == "graceful_degradation":
                    # Check if system provided degraded service
                    if hasattr(target_system, "check_degraded_service"):
                        results.append(await target_system.check_degraded_service())
                    else:
                        results.append(True)
                else:
                    results.append(True)  # Unknown criterion

            return results

        def get_experiment_summary(
            self, results: list[ExperimentResult]
        ) -> dict[str, Any]:
            """Generate summary of experiment results."""
            if not results:
                return {"total_experiments": 0}

            return {
                "total_experiments": len(results),
                "successful_injections": sum(1 for r in results if r.failure_injected),
                "successful_recoveries": sum(1 for r in results if r.system_recovered),
                "average_recovery_time": sum(r.recovery_time for r in results)
                / len(results),
                "success_rate": sum(1 for r in results if all(r.success_criteria_met))
                / len(results),
                "total_errors": sum(len(r.errors) for r in results),
            }

    return ChaosExperimentRunner()


@pytest.fixture
def mock_resilient_service():
    """Mock service with resilience patterns for testing."""

    class MockResilientService:
        def __init__(self):
            self.failure_mode = None
            self.circuit_breaker_open = False
            self.failure_count = 0
            self.last_failure_time = 0

        def simulate_network_timeout(self, _duration: float):
            """Simulate network timeout."""
            self.failure_mode = "network_timeout"
            self.last_failure_time = time.time()

        def simulate_service_unavailable(self):
            """Simulate service unavailable."""
            self.failure_mode = "service_unavailable"
            self.last_failure_time = time.time()

        def simulate_memory_pressure(self):
            """Simulate memory pressure."""
            self.failure_mode = "memory_pressure"
            self.last_failure_time = time.time()

        def stop_failure_simulation(self):
            """Stop failure simulation."""
            self.failure_mode = None

        async def health_check(self):
            """Health check endpoint."""
            if self.failure_mode == "service_unavailable":
                raise CustomError("Service unavailable")
            return {"status": "healthy"}

        async def process_request(self):
            """Process a request with potential failures."""
            if self.failure_mode == "network_timeout":
                raise TimeoutError("Network timeout")
            elif self.failure_mode == "service_unavailable":
                raise CustomError("Service unavailable")
            elif self.failure_mode == "memory_pressure":
                raise MemoryError("Out of memory")

            return {"status": "success", "data": "processed"}

        async def check_data_integrity(self) -> bool:
            """Check data integrity."""
            # Simulate data integrity check
            return self.failure_mode != "memory_pressure"

        async def check_degraded_service(self) -> bool:
            """Check if degraded service is available."""
            # Simulate degraded service check
            return self.failure_mode in [None, "network_timeout"]

    return MockResilientService()


@pytest.fixture
def chaos_test_data():
    """Provide test data for chaos engineering."""
    return {
        "failure_scenarios": [
            {
                "name": "database_connection_failure",
                "type": FailureType.CONNECTION_REFUSED,
                "target": "vector_db",
                "duration": 30,
                "expected_behavior": "graceful_degradation",
            },
            {
                "name": "api_rate_limiting",
                "type": FailureType.RATE_LIMIT_EXCEEDED,
                "target": "openai_api",
                "duration": 60,
                "expected_behavior": "retry_with_backoff",
            },
            {
                "name": "cache_service_timeout",
                "type": FailureType.NETWORK_TIMEOUT,
                "target": "redis",
                "duration": 15,
                "expected_behavior": "bypass_cache",
            },
        ],
        "success_criteria": [
            "system_recovers",
            "no_data_loss",
            "graceful_degradation",
            "error_rate_below_threshold",
            "recovery_time_within_sla",
        ],
        "blast_radius_configs": {
            "single": {"affect_single_instance": True},
            "service": {"affect_entire_service": True},
            "system": {"affect_multiple_services": True},
        },
    }


# Pytest markers for chaos test categorization
def pytest_configure(config):
    """Configure chaos engineering testing markers."""
    config.addinivalue_line("markers", "chaos: mark test as chaos engineering test")
    config.addinivalue_line(
        "markers", "fault_injection: mark test as fault injection test"
    )
    config.addinivalue_line("markers", "resilience: mark test as resilience test")
    config.addinivalue_line(
        "markers", "failure_scenarios: mark test as failure scenario test"
    )
    config.addinivalue_line("markers", "network_chaos: mark test as network chaos test")
    config.addinivalue_line(
        "markers", "resource_exhaustion: mark test as resource exhaustion test"
    )
    config.addinivalue_line(
        "markers", "dependency_failure: mark test as dependency failure test"
    )
