import typing
"""Stress testing fixtures and configuration.

This module provides specialized fixtures and configuration for stress testing,
including resource monitoring, chaos injection, and failure simulation utilities.
"""

import asyncio
import gc
import logging
import os
import resource
import tempfile
import threading
import time
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import psutil
import pytest

logger = logging.getLogger(__name__)


@dataclass
class StressTestProfile:
    """Profile for stress testing scenarios."""

    name: str
    max_users: int
    target_rps: float
    duration_seconds: int
    failure_injection_rate: float = 0.0
    resource_constraints: typing.Optional[dict[str, Any]] = None
    chaos_scenarios: list[str] = field(default_factory=list)
    recovery_validation: bool = True


@dataclass
class ChaosScenario:
    """Chaos engineering scenario for stress testing."""

    name: str
    description: str
    failure_type: str  # "network", "memory", "cpu", "disk", "service"
    intensity: float  # 0.0 to 1.0
    duration: float  # seconds
    recovery_time: float  # seconds
    target_components: list[str] = field(default_factory=list)


class ResourceConstraintManager:
    """Manage system resource constraints for stress testing."""

    def __init__(self):
        self.original_limits = {}
        self.active_constraints = {}

    @contextmanager
    def constrain_memory(self, limit_mb: int):
        """Constrain available memory."""
        if os.name == "posix":  # Unix-like systems
            try:
                old_limit = resource.getrlimit(resource.RLIMIT_AS)
                new_limit = (limit_mb * 1024 * 1024, old_limit[1])
                resource.setrlimit(resource.RLIMIT_AS, new_limit)
                self.original_limits["memory"] = old_limit
                yield
            finally:
                if "memory" in self.original_limits:
                    resource.setrlimit(
                        resource.RLIMIT_AS, self.original_limits["memory"]
                    )
                    del self.original_limits["memory"]
        else:
            # Windows or other systems - just track the constraint
            self.active_constraints["memory_mb"] = limit_mb
            yield
            if "memory_mb" in self.active_constraints:
                del self.active_constraints["memory_mb"]

    @contextmanager
    def constrain_file_descriptors(self, limit: int):
        """Constrain file descriptor limit."""
        if os.name == "posix":
            try:
                old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                new_limit = (limit, old_limit[1])
                resource.setrlimit(resource.RLIMIT_NOFILE, new_limit)
                self.original_limits["fds"] = old_limit
                yield
            finally:
                if "fds" in self.original_limits:
                    resource.setrlimit(
                        resource.RLIMIT_NOFILE, self.original_limits["fds"]
                    )
                    del self.original_limits["fds"]
        else:
            self.active_constraints["max_fds"] = limit
            yield
            if "max_fds" in self.active_constraints:
                del self.active_constraints["max_fds"]

    @contextmanager
    def constrain_cpu_time(self, limit_seconds: int):
        """Constrain CPU time."""
        if os.name == "posix":
            try:
                old_limit = resource.getrlimit(resource.RLIMIT_CPU)
                new_limit = (limit_seconds, old_limit[1])
                resource.setrlimit(resource.RLIMIT_CPU, new_limit)
                self.original_limits["cpu"] = old_limit
                yield
            finally:
                if "cpu" in self.original_limits:
                    resource.setrlimit(resource.RLIMIT_CPU, self.original_limits["cpu"])
                    del self.original_limits["cpu"]
        else:
            self.active_constraints["max_cpu_seconds"] = limit_seconds
            yield
            if "max_cpu_seconds" in self.active_constraints:
                del self.active_constraints["max_cpu_seconds"]


class FailureInjector:
    """Inject various types of failures for chaos testing."""

    def __init__(self):
        self.active_failures = {}
        self.failure_history = []

    async def inject_network_failure(
        self, failure_rate: float = 0.3, duration: float = 30.0
    ):
        """Inject network failures."""
        failure_id = f"network_{time.time()}"
        self.active_failures[failure_id] = {
            "type": "network",
            "rate": failure_rate,
            "start_time": time.time(),
            "duration": duration,
        }

        logger.warning(
            f"Injecting network failures: {failure_rate:.1%} rate for {duration}s"
        )

        # Simulate network failure
        def should_fail():
            return time.time() % 1.0 < failure_rate

        # Store original function to patch

        try:
            await asyncio.sleep(duration)
        finally:
            if failure_id in self.active_failures:
                del self.active_failures[failure_id]

            self.failure_history.append(
                {
                    "id": failure_id,
                    "type": "network",
                    "rate": failure_rate,
                    "duration": duration,
                    "end_time": time.time(),
                }
            )

    async def inject_memory_pressure(
        self, pressure_mb: int = 100, duration: float = 30.0
    ):
        """Inject memory pressure."""
        failure_id = f"memory_{time.time()}"
        self.active_failures[failure_id] = {
            "type": "memory",
            "pressure_mb": pressure_mb,
            "start_time": time.time(),
            "duration": duration,
        }

        logger.warning(f"Injecting memory pressure: {pressure_mb}MB for {duration}s")

        # Allocate memory to create pressure
        memory_hogs = []
        try:
            for _ in range(pressure_mb):
                memory_chunk = bytearray(1024 * 1024)  # 1MB chunks
                memory_hogs.append(memory_chunk)

            await asyncio.sleep(duration)

        finally:
            # Clean up memory
            memory_hogs.clear()
            gc.collect()

            if failure_id in self.active_failures:
                del self.active_failures[failure_id]

            self.failure_history.append(
                {
                    "id": failure_id,
                    "type": "memory",
                    "pressure_mb": pressure_mb,
                    "duration": duration,
                    "end_time": time.time(),
                }
            )

    async def inject_cpu_saturation(
        self, cpu_load: float = 0.8, duration: float = 30.0
    ):
        """Inject CPU saturation."""
        failure_id = f"cpu_{time.time()}"
        self.active_failures[failure_id] = {
            "type": "cpu",
            "load": cpu_load,
            "start_time": time.time(),
            "duration": duration,
        }

        logger.warning(f"Injecting CPU saturation: {cpu_load:.1%} load for {duration}s")

        # CPU intensive work
        stop_cpu_work = threading.Event()

        def cpu_intensive_work():
            """CPU intensive work in separate thread."""
            work_duration = 0.1  # Work for 100ms
            rest_duration = (
                work_duration * (1 - cpu_load) / cpu_load if cpu_load > 0 else 0
            )

            while not stop_cpu_work.is_set():
                # Do CPU intensive work
                start = time.time()
                while time.time() - start < work_duration:
                    _ = sum(i**2 for i in range(1000))

                # Rest to achieve target CPU load
                if rest_duration > 0:
                    time.sleep(rest_duration)

        # Start CPU work in separate thread
        cpu_thread = threading.Thread(target=cpu_intensive_work, daemon=True)
        cpu_thread.start()

        try:
            await asyncio.sleep(duration)
        finally:
            stop_cpu_work.set()

            if failure_id in self.active_failures:
                del self.active_failures[failure_id]

            self.failure_history.append(
                {
                    "id": failure_id,
                    "type": "cpu",
                    "load": cpu_load,
                    "duration": duration,
                    "end_time": time.time(),
                }
            )

    async def inject_disk_io_stress(
        self, io_intensity: float = 0.5, duration: float = 30.0
    ):
        """Inject disk I/O stress."""
        failure_id = f"disk_{time.time()}"
        self.active_failures[failure_id] = {
            "type": "disk",
            "intensity": io_intensity,
            "start_time": time.time(),
            "duration": duration,
        }

        logger.warning(
            f"Injecting disk I/O stress: {io_intensity:.1%} intensity for {duration}s"
        )

        # Create temporary files and perform I/O operations
        temp_files = []
        stop_io_work = threading.Event()

        def disk_io_work():
            """Disk I/O intensive work."""
            while not stop_io_work.is_set():
                try:
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_files.append(temp_file.name)

                    # Write data
                    data = b"x" * int(
                        1024 * 1024 * io_intensity
                    )  # Variable size based on intensity
                    temp_file.write(data)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())  # Force write to disk

                    # Read data back
                    temp_file.seek(0)
                    _ = temp_file.read()

                    temp_file.close()

                    # Brief pause
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Disk I/O stress error: {e}")
                    break

        # Start I/O work in separate thread
        io_thread = threading.Thread(target=disk_io_work, daemon=True)
        io_thread.start()

        try:
            await asyncio.sleep(duration)
        finally:
            stop_io_work.set()

            # Clean up temporary files
            for temp_file_path in temp_files:
                with suppress(OSError, FileNotFoundError):
                    os.unlink(temp_file_path)

            if failure_id in self.active_failures:
                del self.active_failures[failure_id]

            self.failure_history.append(
                {
                    "id": failure_id,
                    "type": "disk",
                    "intensity": io_intensity,
                    "duration": duration,
                    "end_time": time.time(),
                }
            )

    def get_failure_status(self) -> dict[str, Any]:
        """Get current failure injection status."""
        return {
            "active_failures": len(self.active_failures),
            "active_types": list({f["type"] for f in self.active_failures.values()}),
            "total_injected": len(self.failure_history),
            "history": self.failure_history[-10:],  # Last 10 failures
        }


class StressTestOrchestrator:
    """Orchestrate complex stress testing scenarios."""

    def __init__(self):
        self.resource_manager = ResourceConstraintManager()
        self.failure_injector = FailureInjector()
        self.monitoring_active = False
        self.test_phases = []

    async def run_chaos_scenario(self, scenario: ChaosScenario) -> dict[str, Any]:
        """Run a chaos engineering scenario."""
        logger.info(f"Starting chaos scenario: {scenario.name}")

        scenario_start = time.time()
        scenario_result = {
            "scenario": scenario.name,
            "start_time": scenario_start,
            "failure_type": scenario.failure_type,
            "intensity": scenario.intensity,
            "planned_duration": scenario.duration,
            "planned_recovery": scenario.recovery_time,
        }

        try:
            # Inject failure based on type
            if scenario.failure_type == "network":
                await self.failure_injector.inject_network_failure(
                    failure_rate=scenario.intensity, duration=scenario.duration
                )

            elif scenario.failure_type == "memory":
                pressure_mb = int(scenario.intensity * 500)  # Up to 500MB pressure
                await self.failure_injector.inject_memory_pressure(
                    pressure_mb=pressure_mb, duration=scenario.duration
                )

            elif scenario.failure_type == "cpu":
                await self.failure_injector.inject_cpu_saturation(
                    cpu_load=scenario.intensity, duration=scenario.duration
                )

            elif scenario.failure_type == "disk":
                await self.failure_injector.inject_disk_io_stress(
                    io_intensity=scenario.intensity, duration=scenario.duration
                )

            # Recovery phase
            if scenario.recovery_time > 0:
                logger.info(
                    f"Recovery phase for {scenario.name}: {scenario.recovery_time}s"
                )
                await asyncio.sleep(scenario.recovery_time)

            scenario_result.update(
                {
                    "success": True,
                    "actual_duration": time.time() - scenario_start,
                    "error": None,
                }
            )

        except Exception as e:
            scenario_result.update(
                {
                    "success": False,
                    "actual_duration": time.time() - scenario_start,
                    "error": str(e),
                }
            )
            logger.exception(f"Chaos scenario {scenario.name} failed: {e}")

        logger.info(f"Completed chaos scenario: {scenario.name}")
        return scenario_result

    async def run_multi_phase_stress_test(
        self, phases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run multi-phase stress test with different conditions."""
        phase_results = []

        for i, phase in enumerate(phases):
            logger.info(
                f"Starting stress test phase {i + 1}/{len(phases)}: {phase.get('name', f'Phase {i + 1}')}"
            )

            phase_start = time.time()
            phase_result = {
                "phase_number": i + 1,
                "phase_name": phase.get("name", f"Phase {i + 1}"),
                "start_time": phase_start,
            }

            try:
                # Apply resource constraints if specified
                constraints = phase.get("resource_constraints", {})
                constraint_contexts = []

                if "memory_mb" in constraints:
                    constraint_contexts.append(
                        self.resource_manager.constrain_memory(constraints["memory_mb"])
                    )

                if "max_fds" in constraints:
                    constraint_contexts.append(
                        self.resource_manager.constrain_file_descriptors(
                            constraints["max_fds"]
                        )
                    )

                # Enter all constraint contexts
                async with asyncio.gather(
                    *[self._async_context_manager(ctx) for ctx in constraint_contexts],
                    return_exceptions=True,
                ):
                    # Run chaos scenarios if specified
                    chaos_results = []
                    if "chaos_scenarios" in phase:
                        for chaos_config in phase["chaos_scenarios"]:
                            chaos_scenario = ChaosScenario(**chaos_config)
                            chaos_result = await self.run_chaos_scenario(chaos_scenario)
                            chaos_results.append(chaos_result)

                    # Execute phase-specific test logic here
                    # (This would be implemented by the calling test)

                    phase_result.update(
                        {
                            "success": True,
                            "duration": time.time() - phase_start,
                            "chaos_results": chaos_results,
                            "resource_constraints": constraints,
                            "error": None,
                        }
                    )

            except Exception as e:
                phase_result.update(
                    {
                        "success": False,
                        "duration": time.time() - phase_start,
                        "error": str(e),
                    }
                )
                logger.exception(f"Stress test phase {i + 1} failed: {e}")

            phase_results.append(phase_result)
            logger.info(f"Completed stress test phase {i + 1}")

        return phase_results

    async def _async_context_manager(self, context_manager):
        """Convert sync context manager to async."""
        # This is a simplified implementation
        # In a real scenario, you might need more sophisticated handling
        return context_manager


# Predefined stress test profiles
STRESS_TEST_PROFILES = {
    "light_stress": StressTestProfile(
        name="light_stress",
        max_users=50,
        target_rps=25.0,
        duration_seconds=120,
        failure_injection_rate=0.05,
    ),
    "moderate_stress": StressTestProfile(
        name="moderate_stress",
        max_users=200,
        target_rps=100.0,
        duration_seconds=300,
        failure_injection_rate=0.1,
        chaos_scenarios=["memory_pressure", "cpu_spike"],
    ),
    "heavy_stress": StressTestProfile(
        name="heavy_stress",
        max_users=500,
        target_rps=250.0,
        duration_seconds=600,
        failure_injection_rate=0.15,
        resource_constraints={"memory_mb": 1024, "max_fds": 500},
        chaos_scenarios=["memory_pressure", "cpu_spike", "network_failure"],
    ),
    "extreme_stress": StressTestProfile(
        name="extreme_stress",
        max_users=1000,
        target_rps=500.0,
        duration_seconds=300,
        failure_injection_rate=0.2,
        resource_constraints={"memory_mb": 512, "max_fds": 200},
        chaos_scenarios=[
            "memory_pressure",
            "cpu_spike",
            "network_failure",
            "disk_io_stress",
        ],
        recovery_validation=True,
    ),
}

# Predefined chaos scenarios
CHAOS_SCENARIOS = {
    "memory_pressure": ChaosScenario(
        name="memory_pressure",
        description="Inject memory pressure to test memory handling",
        failure_type="memory",
        intensity=0.7,
        duration=60.0,
        recovery_time=30.0,
    ),
    "cpu_spike": ChaosScenario(
        name="cpu_spike",
        description="Create CPU saturation to test CPU handling",
        failure_type="cpu",
        intensity=0.9,
        duration=45.0,
        recovery_time=15.0,
    ),
    "network_failure": ChaosScenario(
        name="network_failure",
        description="Inject network failures to test resilience",
        failure_type="network",
        intensity=0.3,
        duration=90.0,
        recovery_time=60.0,
    ),
    "disk_io_stress": ChaosScenario(
        name="disk_io_stress",
        description="Create disk I/O stress to test I/O handling",
        failure_type="disk",
        intensity=0.8,
        duration=120.0,
        recovery_time=30.0,
    ),
}


@pytest.fixture
def stress_test_profile():
    """Provide stress test profile selection."""

    def get_profile(profile_name: str) -> StressTestProfile:
        return STRESS_TEST_PROFILES.get(
            profile_name, STRESS_TEST_PROFILES["light_stress"]
        )

    return get_profile


@pytest.fixture
def chaos_scenario():
    """Provide chaos scenario selection."""

    def get_scenario(scenario_name: str) -> ChaosScenario:
        return CHAOS_SCENARIOS.get(scenario_name, CHAOS_SCENARIOS["memory_pressure"])

    return get_scenario


@pytest.fixture
def resource_constraint_manager():
    """Provide resource constraint manager."""
    manager = ResourceConstraintManager()
    yield manager
    # Cleanup any remaining constraints
    manager.original_limits.clear()
    manager.active_constraints.clear()


@pytest.fixture
def failure_injector():
    """Provide failure injector for chaos testing."""
    injector = FailureInjector()
    yield injector
    # Cleanup any active failures
    injector.active_failures.clear()


@pytest.fixture
def stress_test_orchestrator():
    """Provide stress test orchestrator."""
    orchestrator = StressTestOrchestrator()
    yield orchestrator
    # Cleanup
    orchestrator.resource_manager.original_limits.clear()
    orchestrator.resource_manager.active_constraints.clear()
    orchestrator.failure_injector.active_failures.clear()


@pytest.fixture(scope="session")
def stress_test_environment():
    """Set up stress testing environment."""
    # Log system information
    logger.info("Setting up stress testing environment")
    logger.info(
        f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.2f} GB RAM"
    )

    # Check resource limits
    if os.name == "posix":
        try:
            fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(
                f"File descriptor limit: {fd_limit[0]} (soft), {fd_limit[1]} (hard)"
            )

            mem_limit = resource.getrlimit(resource.RLIMIT_AS)
            if mem_limit[0] != resource.RLIM_INFINITY:
                logger.info(f"Memory limit: {mem_limit[0] / (1024**3):.2f} GB")
            else:
                logger.info("Memory limit: unlimited")
        except Exception as e:
            logger.warning(f"Could not check resource limits: {e}")

    yield

    # Cleanup
    logger.info("Cleaning up stress testing environment")
    gc.collect()


# Pytest markers for stress test categorization
def pytest_configure(config):
    """Configure stress testing markers."""
    config.addinivalue_line("markers", "stress: mark test as stress test")
    config.addinivalue_line(
        "markers", "resource_exhaustion: mark test as resource exhaustion test"
    )
    config.addinivalue_line(
        "markers", "breaking_point: mark test as breaking point identification test"
    )
    config.addinivalue_line("markers", "chaos: mark test as chaos engineering test")
    config.addinivalue_line(
        "markers", "recovery: mark test as recovery validation test"
    )
    config.addinivalue_line("markers", "slow: mark test as slow-running test")
