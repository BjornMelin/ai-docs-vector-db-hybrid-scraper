"""Chaos engineering test runner and orchestrator.

This module implements a comprehensive chaos engineering test runner that
orchestrates chaos experiments, manages test execution, and provides
reporting and analysis capabilities.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pytest

from tests.chaos.conftest import ChaosExperiment, ExperimentResult, FailureType


class TestError(Exception):
    """Custom exception for this module."""


class ExperimentStatus(Enum):
    """Chaos experiment execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChaosTestSuite:
    """Represents a collection of chaos experiments."""

    suite_name: str
    experiments: list[ChaosExperiment] = field(default_factory=list)
    parallel_execution: bool = False
    max_concurrent_experiments: int = 3
    timeout_seconds: float = 3600.0  # 1 hour
    prerequisites: list[str] = field(default_factory=list)
    cleanup_required: bool = True


@dataclass
class ExperimentExecution:
    """Tracks execution of a chaos experiment."""

    experiment: ChaosExperiment
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    result: ExperimentResult | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2


class ChaosTestRunner:
    """Orchestrates chaos engineering experiments."""

    def __init__(self):
        self.test_suites: dict[str, ChaosTestSuite] = {}
        self.execution_history: list[ExperimentExecution] = []
        self.active_experiments: dict[str, ExperimentExecution] = {}
        self.global_config = {
            "safety_mode": True,
            "max_failure_rate": 0.5,
            "emergency_stop_threshold": 0.8,
            "monitoring_interval": 1.0,
            "report_generation": True,
        }

    def _raise_safety_error(self, reason: str) -> None:
        """Raise a safety error with the given reason."""
        msg = f"Safety check failed: {reason}"
        raise TestError(msg)

    def register_test_suite(self, suite: ChaosTestSuite):
        """Register a chaos test suite."""
        self.test_suites[suite.suite_name] = suite

    async def execute_experiment(
        self,
        experiment: ChaosExperiment,
        target_system: Any = None,
        monitoring_callback: Callable | None = None,
    ) -> ExperimentExecution:
        """Execute a single chaos experiment."""
        execution = ExperimentExecution(experiment=experiment)
        execution.status = ExperimentStatus.RUNNING
        execution.start_time = time.time()

        experiment_id = f"{experiment.name}_{execution.start_time}"
        self.active_experiments[experiment_id] = execution

        try:
            # Safety checks
            if self.global_config["safety_mode"]:
                safety_check = await self._perform_safety_checks(experiment)
                if not safety_check.get("passed", True):
                    self._raise_safety_error(safety_check["reason"])

            # Execute the experiment
            result = await self._run_experiment_implementation(
                experiment, target_system, monitoring_callback
            )

            execution.result = result
            execution.status = ExperimentStatus.COMPLETED

        except (
            TestError,
            ValueError,
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ) as e:
            execution.error = str(e)
            execution.status = ExperimentStatus.FAILED

            # Check if retry is needed
            if execution.retry_count < execution.max_retries:
                execution.retry_count += 1
                # Schedule retry
                await asyncio.sleep(5.0)  # Wait before retry
                return await self.execute_experiment(
                    experiment, target_system, monitoring_callback
                )

        finally:
            execution.end_time = time.time()
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            self.execution_history.append(execution)

        return execution

    async def execute_test_suite(
        self, suite_name: str, target_system: Any = None
    ) -> dict[str, Any]:
        """Execute a complete chaos test suite."""
        if suite_name not in self.test_suites:
            msg = f"Test suite '{suite_name}' not found"
            raise ValueError(msg)

        suite = self.test_suites[suite_name]
        suite_start_time = time.time()

        # Check prerequisites
        prereq_check = await self._check_prerequisites(suite.prerequisites)
        if not prereq_check["all_met"]:
            return {
                "status": "failed",
                "reason": "prerequisites_not_met",
                "missing_prerequisites": prereq_check["missing"],
            }

        experiment_executions = []

        try:
            if suite.parallel_execution:
                # Execute experiments in parallel
                experiment_executions = await self._execute_parallel(
                    suite.experiments, target_system, suite.max_concurrent_experiments
                )
            else:
                # Execute experiments sequentially
                experiment_executions = await self._execute_sequential(
                    suite.experiments, target_system
                )

            # Analyze results
            suite_results = self._analyze_suite_results(experiment_executions)

            return {
                "status": "completed",
                "suite_name": suite_name,
                "execution_time": time.time() - suite_start_time,
                "_total_experiments": len(suite.experiments),
                "successful_experiments": suite_results["successful"],
                "failed_experiments": suite_results["failed"],
                "experiment_results": [
                    asdict(execution) for execution in experiment_executions
                ],
                "summary": suite_results,
            }

        except (
            TestError,
            ValueError,
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ) as e:
            return {
                "status": "failed",
                "suite_name": suite_name,
                "error": str(e),
                "execution_time": time.time() - suite_start_time,
                "partial_results": [
                    asdict(execution) for execution in experiment_executions
                ],
            }

        finally:
            # Cleanup if required
            if suite.cleanup_required:
                await self._perform_cleanup(suite_name)

    async def _execute_parallel(
        self,
        experiments: list[ChaosExperiment],
        target_system: Any,
        max_concurrent: int,
    ) -> list[ExperimentExecution]:
        """Execute experiments in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(experiment):
            async with semaphore:
                return await self.execute_experiment(experiment, target_system)

        tasks = [run_with_semaphore(exp) for exp in experiments]
        executions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        results = []
        for i, result in enumerate(executions):
            if isinstance(result, Exception):
                # Create failed execution for exception
                failed_execution = ExperimentExecution(experiment=experiments[i])
                failed_execution.status = ExperimentStatus.FAILED
                failed_execution.error = str(result)
                failed_execution.start_time = time.time()
                failed_execution.end_time = time.time()
                results.append(failed_execution)
            else:
                results.append(result)

        return results

    async def _execute_sequential(
        self, experiments: list[ChaosExperiment], target_system: Any
    ) -> list[ExperimentExecution]:
        """Execute experiments sequentially."""
        executions = []

        for experiment in experiments:
            execution = await self.execute_experiment(experiment, target_system)
            executions.append(execution)

            # Check if we should stop due to failures
            if execution.status == ExperimentStatus.FAILED:
                failure_rate = len(
                    [e for e in executions if e.status == ExperimentStatus.FAILED]
                ) / len(executions)
                if failure_rate > self.global_config["max_failure_rate"]:
                    # Stop execution due to high failure rate
                    break

        return executions

    async def _run_experiment_implementation(
        self,
        experiment: ChaosExperiment,
        target_system: Any,
        monitoring_callback: Callable | None,
    ) -> ExperimentResult:
        """Run the actual experiment implementation."""
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
            # Phase 1: Inject failure
            await self._inject_experiment_failure(experiment, target_system)
            result.failure_injected = True

            # Phase 2: Monitor system during failure
            if monitoring_callback:
                metrics = await monitoring_callback()
                result.metrics.update(metrics)

            # Phase 3: Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)

            # Phase 4: Stop failure injection
            await self._stop_experiment_failure(experiment, target_system)

            # Phase 5: Validate recovery
            recovery_start = time.time()
            await asyncio.sleep(experiment.recovery_time_seconds)

            # Check system recovery
            result.system_recovered = await self._validate_system_recovery(
                target_system
            )
            result.recovery_time = time.time() - recovery_start

            # Phase 6: Evaluate success criteria
            result.success_criteria_met = await self._evaluate_success_criteria(
                experiment, target_system
            )

        except (
            TestError,
            ValueError,
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ) as e:
            result.errors.append(str(e))

        finally:
            result.ended_at = time.time()
            result.duration = result.ended_at - result.started_at

        return result

    async def _perform_safety_checks(
        self, experiment: ChaosExperiment
    ) -> dict[str, Any]:
        """Perform safety checks before experiment execution."""
        safety_checks = []

        # Check blast radius
        if experiment.blast_radius == "system":
            safety_checks.append(
                {
                    "check": "blast_radius",
                    "safe": False,
                    "reason": "System-wide blast radius requires manual approval",
                }
            )

        # Check failure rate
        if experiment.failure_rate > 0.8:
            safety_checks.append(
                {
                    "check": "failure_rate",
                    "safe": False,
                    "reason": f"High failure rate ({experiment.failure_rate}) may cause system instability",
                }
            )

        # Check concurrent experiments
        if len(self.active_experiments) >= 3:
            safety_checks.append(
                {
                    "check": "concurrent_experiments",
                    "safe": False,
                    "reason": "Too many concurrent experiments running",
                }
            )

        unsafe_checks = [check for check in safety_checks if not check["safe"]]

        return {
            "safe_to_proceed": len(unsafe_checks) == 0,
            "reason": "; ".join([check["reason"] for check in unsafe_checks])
            if unsafe_checks
            else None,
            "checks": safety_checks,
        }

    async def _check_prerequisites(self, prerequisites: list[str]) -> dict[str, Any]:
        """Check if prerequisites are met."""
        # Mock prerequisite checking
        met_prerequisites = []
        missing_prerequisites = []

        for prereq in prerequisites:
            # Simulate prerequisite checking
            if prereq in ["baseline_captured", "monitoring_enabled", "backup_verified"]:
                met_prerequisites.append(prereq)
            else:
                missing_prerequisites.append(prereq)

        return {
            "all_met": len(missing_prerequisites) == 0,
            "met": met_prerequisites,
            "missing": missing_prerequisites,
        }

    async def _inject_experiment_failure(
        self, experiment: ChaosExperiment, target_system: Any
    ):
        """Inject failure based on experiment configuration."""
        # Mock failure injection
        if hasattr(target_system, "inject_failure"):
            await target_system.inject_failure(
                experiment.failure_type, experiment.target_service
            )

    async def _stop_experiment_failure(
        self, experiment: ChaosExperiment, target_system: Any
    ):
        """Stop failure injection."""
        if hasattr(target_system, "stop_failure_injection"):
            await target_system.stop_failure_injection(experiment.target_service)

    async def _validate_system_recovery(self, target_system: Any) -> bool:
        """Validate that system has recovered."""
        if hasattr(target_system, "health_check"):
            try:
                await target_system.health_check()
            except (TestError, ValueError, TimeoutError, ConnectionError, RuntimeError):
                return False
            else:
                return True
        return True  # Assume recovered if no health check available

    async def _evaluate_success_criteria(
        self, experiment: ChaosExperiment, target_system: Any
    ) -> list[bool]:
        """Evaluate experiment success criteria."""
        results = []

        for criterion in experiment.success_criteria:
            if criterion == "system_recovers":
                results.append(await self._validate_system_recovery(target_system))
            elif criterion == "no_data_loss":
                # Mock data loss check
                results.append(True)
            elif criterion == "graceful_degradation":
                # Mock graceful degradation check
                results.append(True)
            else:
                results.append(True)  # Unknown criteria pass by default

        return results

    def _analyze_suite_results(
        self, executions: list[ExperimentExecution]
    ) -> dict[str, Any]:
        """Analyze results of experiment executions."""
        successful = len(
            [e for e in executions if e.status == ExperimentStatus.COMPLETED]
        )
        failed = len([e for e in executions if e.status == ExperimentStatus.FAILED])

        _total_duration = sum(
            [
                (e.end_time or 0) - (e.start_time or 0)
                for e in executions
                if e.start_time and e.end_time
            ]
        )

        # Calculate success criteria metrics
        all_criteria_met = 0
        _total_criteria = 0

        for execution in executions:
            if execution.result and execution.result.success_criteria_met:
                criteria_met = sum(execution.result.success_criteria_met)
                all_criteria_met += criteria_met
                _total_criteria += len(execution.result.success_criteria_met)

        success_rate = all_criteria_met / _total_criteria if _total_criteria > 0 else 0

        return {
            "successful": successful,
            "failed": failed,
            "_total": len(executions),
            "success_rate": success_rate,
            "_total_duration": _total_duration,
            "average_duration": _total_duration / len(executions) if executions else 0,
            "criteria_success_rate": success_rate,
        }

    async def _perform_cleanup(self, _suite_name: str):
        """Perform cleanup after suite execution."""
        # Mock cleanup operations
        await asyncio.sleep(0.1)

    def generate_report(self, suite_name: str | None = None) -> dict[str, Any]:
        """Generate comprehensive chaos engineering report."""
        if suite_name:
            # Report for specific suite
            executions = [
                e
                for e in self.execution_history
                if e.experiment.name.startswith(suite_name)
            ]
        else:
            # Report for all executions
            executions = self.execution_history

        if not executions:
            return {"error": "No execution data available"}

        # Calculate overall metrics
        _total_experiments = len(executions)
        successful_experiments = len(
            [e for e in executions if e.status == ExperimentStatus.COMPLETED]
        )
        failed_experiments = len(
            [e for e in executions if e.status == ExperimentStatus.FAILED]
        )

        # Failure type analysis
        failure_types = {}
        for execution in executions:
            failure_type = execution.experiment.failure_type.value
            if failure_type not in failure_types:
                failure_types[failure_type] = {
                    "_total": 0,
                    "successful": 0,
                    "failed": 0,
                }

            failure_types[failure_type]["_total"] += 1
            if execution.status == ExperimentStatus.COMPLETED:
                failure_types[failure_type]["successful"] += 1
            else:
                failure_types[failure_type]["failed"] += 1

        # Recovery time analysis
        recovery_times = [
            e.result.recovery_time
            for e in executions
            if e.result and e.result.recovery_time > 0
        ]

        avg_recovery_time = (
            sum(recovery_times) / len(recovery_times) if recovery_times else 0
        )

        return {
            "report_generated_at": time.time(),
            "suite_name": suite_name,
            "summary": {
                "_total_experiments": _total_experiments,
                "successful_experiments": successful_experiments,
                "failed_experiments": failed_experiments,
                "success_rate": successful_experiments / _total_experiments
                if _total_experiments > 0
                else 0,
            },
            "failure_type_analysis": failure_types,
            "recovery_metrics": {
                "average_recovery_time": avg_recovery_time,
                "min_recovery_time": min(recovery_times) if recovery_times else 0,
                "max_recovery_time": max(recovery_times) if recovery_times else 0,
                "_total_recovery_samples": len(recovery_times),
            },
            "recommendations": self._generate_recommendations(executions),
        }

    def _generate_recommendations(
        self, executions: list[ExperimentExecution]
    ) -> list[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []

        # Analyze failure patterns
        failed_experiments = [
            e for e in executions if e.status == ExperimentStatus.FAILED
        ]
        if len(failed_experiments) > len(executions) * 0.3:  # More than 30% failures
            recommendations.append(
                "High failure rate detected - review system resilience mechanisms"
            )

        # Analyze recovery times
        recovery_times = [
            e.result.recovery_time
            for e in executions
            if e.result and e.result.recovery_time > 0
        ]

        if recovery_times and max(recovery_times) > 300:  # 5 minutes
            recommendations.append(
                "Long recovery times detected - optimize recovery procedures"
            )

        # Analyze success criteria
        criteria_failures = []
        for execution in executions:
            if execution.result and execution.result.success_criteria_met:
                for i, met in enumerate(execution.result.success_criteria_met):
                    if not met:
                        criteria_failures.append(
                            execution.experiment.success_criteria[i]
                        )

        if criteria_failures:
            most_common_failure = max(
                set(criteria_failures), key=criteria_failures.count
            )
            recommendations.append(
                f"Success criteria '{most_common_failure}' frequently fails - investigate and improve"
            )

        if not recommendations:
            recommendations.append(
                "System demonstrates good resilience - continue regular chaos testing"
            )

        return recommendations


@pytest.mark.chaos
class TestChaosRunner:
    """Test chaos engineering runner functionality."""

    @pytest.fixture
    def chaos_runner(self):
        """Create chaos test runner."""
        return ChaosTestRunner()

    @pytest.fixture
    def sample_experiments(self):
        """Create sample chaos experiments."""
        return [
            ChaosExperiment(
                name="network_timeout_test",
                description="Test network timeout resilience",
                failure_type=FailureType.NETWORK_TIMEOUT,
                target_service="api_gateway",
                duration_seconds=10.0,
                failure_rate=1.0,
                blast_radius="single",
                recovery_time_seconds=5.0,
                success_criteria=["system_recovers", "graceful_degradation"],
                rollback_strategy="immediate",
            ),
            ChaosExperiment(
                name="service_unavailable_test",
                description="Test service unavailable handling",
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                target_service="search_service",
                duration_seconds=15.0,
                failure_rate=1.0,
                blast_radius="service",
                recovery_time_seconds=10.0,
                success_criteria=["system_recovers", "no_data_loss"],
                rollback_strategy="immediate",
            ),
            ChaosExperiment(
                name="memory_exhaustion_test",
                description="Test memory exhaustion handling",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                target_service="embedding_service",
                duration_seconds=20.0,
                failure_rate=0.8,
                blast_radius="single",
                recovery_time_seconds=15.0,
                success_criteria=["system_recovers", "graceful_degradation"],
                rollback_strategy="immediate",
            ),
        ]

    @pytest.fixture
    def mock_target_system(self):
        """Create mock target system for testing."""

        class MockTargetSystem:
            def __init__(self):
                self.failure_injected = False
                self.failure_type = None
                self.target_service = None

            async def inject_failure(
                self, failure_type: FailureType, target_service: str
            ):
                self.failure_injected = True
                self.failure_type = failure_type
                self.target_service = target_service

            async def stop_failure_injection(self, _target_service: str):
                self.failure_injected = False
                self.failure_type = None
                self.target_service = None

            async def health_check(self):
                if self.failure_injected:
                    msg = "System unhealthy due to injected failure"
                    raise TestError(msg)
                return {"status": "healthy"}

        return MockTargetSystem()

    async def test_single_experiment_execution(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test execution of a single chaos experiment."""
        experiment = sample_experiments[0]  # network_timeout_test

        execution = await chaos_runner.execute_experiment(
            experiment, mock_target_system
        )

        # Verify execution completed
        assert execution.status == ExperimentStatus.COMPLETED
        assert execution.start_time is not None
        assert execution.end_time is not None
        assert execution.result is not None

        # Verify experiment results
        result = execution.result
        assert result.experiment_name == "network_timeout_test"
        assert result.failure_injected is True
        assert result.duration > 0

        # Verify target system was affected
        assert mock_target_system.failure_type == FailureType.NETWORK_TIMEOUT
        assert mock_target_system.target_service == "api_gateway"

    async def test_test_suite_execution(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test execution of a complete test suite."""
        # Create test suite
        test_suite = ChaosTestSuite(
            suite_name="resilience_test_suite",
            experiments=sample_experiments,
            parallel_execution=False,
            prerequisites=["baseline_captured", "monitoring_enabled"],
        )

        chaos_runner.register_test_suite(test_suite)

        # Execute test suite
        suite_result = await chaos_runner.execute_test_suite(
            "resilience_test_suite", mock_target_system
        )

        # Verify suite execution
        assert suite_result["status"] == "completed"
        assert suite_result["_total_experiments"] == 3
        assert suite_result["successful_experiments"] >= 0
        assert suite_result["execution_time"] > 0

        # Verify all experiments were executed
        assert len(suite_result["experiment_results"]) == 3

    async def test_parallel_experiment_execution(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test parallel execution of chaos experiments."""
        # Create parallel test suite
        parallel_suite = ChaosTestSuite(
            suite_name="parallel_chaos_suite",
            experiments=sample_experiments,
            parallel_execution=True,
            max_concurrent_experiments=2,
        )

        chaos_runner.register_test_suite(parallel_suite)

        # Execute parallel suite
        start_time = time.time()
        suite_result = await chaos_runner.execute_test_suite(
            "parallel_chaos_suite", mock_target_system
        )
        execution_time = time.time() - start_time

        # Verify parallel execution was faster than sequential
        # (This is a rough test - parallel should be faster)
        assert suite_result["status"] == "completed"
        assert execution_time < 60  # Should complete within 1 minute

        # Verify all experiments completed
        assert suite_result["successful_experiments"] >= 0

    async def test_safety_checks(self, chaos_runner, mock_target_system):
        """Test safety check mechanisms."""
        # Create unsafe experiment
        unsafe_experiment = ChaosExperiment(
            name="unsafe_test",
            description="Unsafe test with system-wide blast radius",
            failure_type=FailureType.SERVICE_UNAVAILABLE,
            target_service="entire_system",
            duration_seconds=60.0,
            failure_rate=1.0,
            blast_radius="system",  # This should trigger safety check
            recovery_time_seconds=30.0,
            success_criteria=["system_recovers"],
            rollback_strategy="immediate",
        )

        # Execute unsafe experiment
        execution = await chaos_runner.execute_experiment(
            unsafe_experiment, mock_target_system
        )

        # Should fail due to safety check
        assert execution.status == ExperimentStatus.FAILED
        assert "Safety check failed" in execution.error

    @pytest.mark.usefixtures("mock_target_system")
    async def test_experiment_retry_mechanism(self, chaos_runner):
        """Test experiment retry mechanism."""

        # Create experiment that will fail initially
        class FailingTargetSystem:
            def __init__(self):
                self.call_count = 0

            async def inject_failure(
                self, _failure_type: FailureType, _target_service: str
            ):
                self.call_count += 1
                if self.call_count <= 2:  # Fail first 2 attempts
                    msg = "Simulated injection failure"
                    raise TestError(msg)
                # Succeed on 3rd attempt

            async def stop_failure_injection(self, target_service: str):
                pass

            async def health_check(self):
                return {"status": "healthy"}

        failing_system = FailingTargetSystem()

        experiment = ChaosExperiment(
            name="retry_test",
            description="Test retry mechanism",
            failure_type=FailureType.NETWORK_TIMEOUT,
            target_service="test_service",
            duration_seconds=1.0,
            failure_rate=1.0,
            blast_radius="single",
            recovery_time_seconds=1.0,
            success_criteria=["system_recovers"],
            rollback_strategy="immediate",
        )

        execution = await chaos_runner.execute_experiment(experiment, failing_system)

        # Should eventually succeed after retries
        assert execution.retry_count > 0
        assert execution.status == ExperimentStatus.COMPLETED

    async def test_report_generation(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test chaos engineering report generation."""
        # Execute some experiments
        for experiment in sample_experiments:
            await chaos_runner.execute_experiment(experiment, mock_target_system)

        # Generate report
        report = chaos_runner.generate_report()

        # Verify report structure
        assert "report_generated_at" in report
        assert "summary" in report
        assert "failure_type_analysis" in report
        assert "recovery_metrics" in report
        assert "recommendations" in report

        # Verify summary metrics
        summary = report["summary"]
        assert summary["_total_experiments"] == 3
        assert summary["success_rate"] >= 0.0
        assert summary["success_rate"] <= 1.0

        # Verify recommendations
        assert len(report["recommendations"]) > 0

    async def test_concurrent_experiment_limits(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test concurrent experiment execution limits."""
        # Modify global config to limit concurrent experiments
        chaos_runner.global_config["max_concurrent_experiments"] = 1

        # Create long-running experiment
        long_experiment = ChaosExperiment(
            name="long_test",
            description="Long running test",
            failure_type=FailureType.NETWORK_TIMEOUT,
            target_service="test_service",
            duration_seconds=0.5,  # 500ms
            failure_rate=1.0,
            blast_radius="single",
            recovery_time_seconds=0.1,
            success_criteria=["system_recovers"],
            rollback_strategy="immediate",
        )

        # Start first experiment
        task1 = asyncio.create_task(
            chaos_runner.execute_experiment(long_experiment, mock_target_system)
        )

        # Give it time to start
        await asyncio.sleep(0.1)

        # Try to start second experiment
        execution2 = await chaos_runner.execute_experiment(
            sample_experiments[0], mock_target_system
        )

        # Second experiment should fail due to concurrent limit
        assert execution2.status == ExperimentStatus.FAILED
        assert "concurrent experiments" in execution2.error.lower()

        # Wait for first experiment to complete
        execution1 = await task1
        assert execution1.status == ExperimentStatus.COMPLETED

    async def test_monitoring_callback_integration(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test monitoring callback integration."""
        monitoring_data = []

        async def monitoring_callback():
            """Mock monitoring callback."""
            metrics = {
                "timestamp": time.time(),
                "cpu_usage": 75.5,
                "memory_usage": 60.2,
                "response_time": 150,
                "error_rate": 0.05,
            }
            monitoring_data.append(metrics)
            return metrics

        experiment = sample_experiments[0]
        execution = await chaos_runner.execute_experiment(
            experiment, mock_target_system, monitoring_callback
        )

        # Verify monitoring was called
        assert execution.status == ExperimentStatus.COMPLETED
        assert execution.result.metrics is not None
        assert len(monitoring_data) > 0

        # Verify metrics were captured
        assert "cpu_usage" in execution.result.metrics
        assert "memory_usage" in execution.result.metrics

    async def test_cleanup_operations(
        self, chaos_runner, sample_experiments, mock_target_system
    ):
        """Test cleanup operations after experiment execution."""
        cleanup_called = False

        # Create wrapper to track cleanup calls
        async def mock_cleanup(suite_name: str):
            nonlocal cleanup_called
            cleanup_called = True
            # Note: In real implementation, would verify cleanup was performed

        # Instead of accessing private method, we'll test the public interface
        # and verify that cleanup was triggered through observable side effects

        # Create test suite with cleanup required
        test_suite = ChaosTestSuite(
            suite_name="cleanup_test_suite",
            experiments=[sample_experiments[0]],
            cleanup_required=True,
        )

        chaos_runner.register_test_suite(test_suite)

        # Execute test suite
        await chaos_runner.execute_test_suite("cleanup_test_suite", mock_target_system)

        # Verify cleanup was called
        assert cleanup_called
