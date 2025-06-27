class TestError(Exception):
    """Custom exception for this module."""

    pass


"""Resource exhaustion tests for chaos engineering.

This module implements resource exhaustion scenarios to test system behavior
under extreme resource constraints including memory, CPU, disk, and connection
pool exhaustion.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


class ResourceType(Enum):
    """Types of system resources."""

    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK_CONNECTIONS = "network_connections"
    FILE_DESCRIPTORS = "file_descriptors"
    DATABASE_CONNECTIONS = "database_connections"
    THREAD_POOL = "thread_pool"


@dataclass
class ResourceMonitor:
    """Monitor system resource usage."""

    resource_type: ResourceType
    current_usage: float = 0.0
    max_capacity: float = 100.0
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    history: list[float] = field(default_factory=list)

    def update_usage(self, usage: float):
        """Update resource usage."""
        self.current_usage = usage
        self.history.append(usage)
        # Keep only recent history
        if len(self.history) > 100:
            self.history.pop(0)

    def get_usage_percentage(self) -> float:
        """Get usage as percentage."""
        return (self.current_usage / self.max_capacity) * 100

    def is_critical(self) -> bool:
        """Check if resource usage is critical."""
        return self.get_usage_percentage() >= self.critical_threshold

    def is_warning(self) -> bool:
        """Check if resource usage is at warning level."""
        return self.get_usage_percentage() >= self.warning_threshold


class ResourceExhaustionSimulator:
    """Simulate resource exhaustion scenarios."""

    def __init__(self):
        self.monitors = {
            ResourceType.MEMORY: ResourceMonitor(
                ResourceType.MEMORY, max_capacity=1000
            ),  # MB
            ResourceType.CPU: ResourceMonitor(
                ResourceType.CPU, max_capacity=100
            ),  # Percentage
            ResourceType.DISK: ResourceMonitor(
                ResourceType.DISK, max_capacity=1000
            ),  # GB
            ResourceType.NETWORK_CONNECTIONS: ResourceMonitor(
                ResourceType.NETWORK_CONNECTIONS, max_capacity=1000
            ),
            ResourceType.FILE_DESCRIPTORS: ResourceMonitor(
                ResourceType.FILE_DESCRIPTORS, max_capacity=1024
            ),
            ResourceType.DATABASE_CONNECTIONS: ResourceMonitor(
                ResourceType.DATABASE_CONNECTIONS, max_capacity=100
            ),
            ResourceType.THREAD_POOL: ResourceMonitor(
                ResourceType.THREAD_POOL, max_capacity=50
            ),
        }
        self.active_exhaustions = {}

    def get_monitor(self, resource_type: ResourceType) -> ResourceMonitor:
        """Get resource monitor."""
        return self.monitors[resource_type]

    async def simulate_memory_exhaustion(
        self, target_usage: float, duration: float = 5.0
    ):
        """Simulate memory exhaustion."""
        monitor = self.monitors[ResourceType.MEMORY]
        original_usage = monitor.current_usage

        # Gradually increase memory usage
        steps = 20
        step_duration = duration / steps
        usage_increment = (target_usage - original_usage) / steps

        try:
            for step in range(steps):
                new_usage = original_usage + (usage_increment * (step + 1))
                monitor.update_usage(min(new_usage, monitor.max_capacity))
                await asyncio.sleep(step_duration)

                if monitor.is_critical():
                    # Simulate system slowdown due to memory pressure
                    await asyncio.sleep(0.01)
        finally:
            # Restore original usage
            monitor.update_usage(original_usage)

    async def simulate_cpu_exhaustion(self, target_usage: float, duration: float = 5.0):
        """Simulate CPU exhaustion."""
        monitor = self.monitors[ResourceType.CPU]
        original_usage = monitor.current_usage

        try:
            monitor.update_usage(target_usage)

            # Simulate CPU-intensive work
            work_duration = duration
            if target_usage > 90:
                work_duration *= 2  # Slower response under high CPU load

            await asyncio.sleep(work_duration)
        finally:
            monitor.update_usage(original_usage)

    async def simulate_connection_pool_exhaustion(
        self, pool_type: ResourceType, target_usage: int
    ):
        """Simulate connection pool exhaustion."""
        monitor = self.monitors[pool_type]
        monitor.update_usage(target_usage)

        if monitor.is_critical():
            return {
                "status": "exhausted",
                "available_connections": max(0, monitor.max_capacity - target_usage),
                "queue_length": max(0, target_usage - monitor.max_capacity),
            }

        return {
            "status": "available",
            "available_connections": monitor.max_capacity - target_usage,
            "queue_length": 0,
        }

    async def simulate_disk_exhaustion(self, target_usage: float):
        """Simulate disk space exhaustion."""
        monitor = self.monitors[ResourceType.DISK]
        monitor.update_usage(target_usage)

        if monitor.is_critical():
            # Simulate disk operations becoming slow/failing
            await asyncio.sleep(0.1)  # Simulate slow disk I/O
            raise TestError("Disk full - cannot write data")

        return {
            "available_space": monitor.max_capacity - target_usage,
            "usage_percentage": monitor.get_usage_percentage(),
            "is_critical": monitor.is_critical(),
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health based on resource usage."""
        critical_resources = []
        warning_resources = []

        for resource_type, monitor in self.monitors.items():
            if monitor.is_critical():
                critical_resources.append(
                    {
                        "resource": resource_type.value,
                        "usage": monitor.get_usage_percentage(),
                        "status": "critical",
                    }
                )
            elif monitor.is_warning():
                warning_resources.append(
                    {
                        "resource": resource_type.value,
                        "usage": monitor.get_usage_percentage(),
                        "status": "warning",
                    }
                )

        if critical_resources:
            overall_health = "critical"
        elif warning_resources:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "critical_resources": critical_resources,
            "warning_resources": warning_resources,
            "total_resources": len(self.monitors),
        }


@pytest.mark.chaos
@pytest.mark.resource_exhaustion
class TestResourceExhaustion:
    """Test resource exhaustion scenarios."""

    @pytest.fixture
    def resource_simulator(self):
        """Create resource exhaustion simulator."""
        return ResourceExhaustionSimulator()

    async def test_memory_exhaustion_scenario(
        self, resource_simulator, _fault_injector
    ):
        """Test system behavior under memory exhaustion."""
        # Monitor initial memory state
        memory_monitor = resource_simulator.get_monitor(ResourceType.MEMORY)

        # Simulate gradual memory exhaustion
        await resource_simulator.simulate_memory_exhaustion(
            target_usage=950,  # 95% of capacity
            duration=0.1,  # Fast for testing
        )

        # Check that memory usage increased
        assert memory_monitor.get_usage_percentage() >= 90
        assert memory_monitor.is_critical()

        # Simulate application behavior under memory pressure
        async def memory_intensive_operation():
            """Simulate operation that requires memory."""
            monitor = resource_simulator.get_monitor(ResourceType.MEMORY)
            if monitor.is_critical():
                # Simulate memory allocation failure
                raise MemoryError("Insufficient memory available")
            return {"status": "success", "memory_used": 50}

        # Test that operations fail under memory pressure
        with pytest.raises(MemoryError):
            await memory_intensive_operation()

        # Test graceful degradation
        async def memory_aware_operation():
            """Operation that adapts to memory pressure."""
            monitor = resource_simulator.get_monitor(ResourceType.MEMORY)
            if monitor.is_critical():
                # Use less memory-intensive algorithm
                return {"status": "degraded", "memory_used": 10, "mode": "low_memory"}
            return {"status": "normal", "memory_used": 50, "mode": "normal"}

        result = await memory_aware_operation()
        assert result["status"] == "degraded"
        assert result["mode"] == "low_memory"

    async def test_cpu_exhaustion_scenario(self, resource_simulator):
        """Test system behavior under CPU exhaustion."""
        cpu_monitor = resource_simulator.get_monitor(ResourceType.CPU)

        # Simulate CPU-intensive workload
        async def cpu_intensive_task():
            """Simulate CPU-intensive task."""
            await resource_simulator.simulate_cpu_exhaustion(
                target_usage=95,  # 95% CPU usage
                duration=0.1,
            )
            return {
                "task": "completed",
                "cpu_usage": cpu_monitor.get_usage_percentage(),
            }

        # Measure task execution time under high CPU load
        start_time = time.time()
        result = await cpu_intensive_task()
        execution_time = time.time() - start_time

        assert result["cpu_usage"] >= 90
        assert execution_time > 0.1  # Should take longer due to CPU pressure

        # Test task prioritization under CPU pressure
        async def priority_aware_scheduler():
            """Scheduler that adapts to CPU pressure."""
            monitor = resource_simulator.get_monitor(ResourceType.CPU)

            if monitor.is_critical():
                # Drop low-priority tasks
                return {
                    "high_priority_tasks": 5,
                    "medium_priority_tasks": 2,
                    "low_priority_tasks": 0,  # Dropped
                    "cpu_throttling": True,
                }

            return {
                "high_priority_tasks": 5,
                "medium_priority_tasks": 5,
                "low_priority_tasks": 5,
                "cpu_throttling": False,
            }

        schedule_result = await priority_aware_scheduler()
        assert schedule_result["cpu_throttling"] is True
        assert schedule_result["low_priority_tasks"] == 0

    async def test_connection_pool_exhaustion(self, resource_simulator):
        """Test connection pool exhaustion scenarios."""
        # Test database connection pool exhaustion
        db_pool_status = await resource_simulator.simulate_connection_pool_exhaustion(
            ResourceType.DATABASE_CONNECTIONS,
            target_usage=100,  # Exhaust all connections
        )

        assert db_pool_status["status"] == "exhausted"
        assert db_pool_status["available_connections"] == 0

        # Simulate connection request when pool is exhausted
        async def request_database_connection():
            """Request database connection."""
            monitor = resource_simulator.get_monitor(ResourceType.DATABASE_CONNECTIONS)
            available = monitor.max_capacity - monitor.current_usage

            if available <= 0:
                # No connections available - implement queuing or rejection
                raise TestError("Connection pool exhausted - request queued")

            return {"connection_id": "conn_123", "status": "acquired"}

        # Should fail when pool is exhausted
        with pytest.raises(Exception, match="Connection pool exhausted"):
            await request_database_connection()

        # Test connection pool with graceful degradation
        async def connection_with_fallback():
            """Connection request with fallback strategy."""
            try:
                return await request_database_connection()
            except Exception:
                # Fallback to read-only replica or cached data
                return {
                    "connection_id": "fallback_readonly",
                    "status": "fallback",
                    "capabilities": ["read_only"],
                }

        fallback_result = await connection_with_fallback()
        assert fallback_result["status"] == "fallback"
        assert "read_only" in fallback_result["capabilities"]

    async def test_disk_space_exhaustion(self, resource_simulator):
        """Test disk space exhaustion scenarios."""
        # Simulate disk space filling up
        disk_status = await resource_simulator.simulate_disk_exhaustion(950)  # 95% full

        assert disk_status["is_critical"] is True
        assert disk_status["available_space"] <= 50  # Less than 50GB available

        # Test write operations under disk pressure
        async def write_operation(data_size: float):
            """Simulate write operation."""
            monitor = resource_simulator.get_monitor(ResourceType.DISK)
            available_space = monitor.max_capacity - monitor.current_usage

            if data_size > available_space:
                raise TestError(
                    f"Insufficient disk space: need {data_size}GB, have {available_space}GB"
                )

            # Simulate successful write
            monitor.update_usage(monitor.current_usage + data_size)
            return {
                "bytes_written": data_size * 1024 * 1024 * 1024,
                "status": "success",
            }

        # Small write should succeed
        small_write = await write_operation(1)  # 1GB
        assert small_write["status"] == "success"

        # Large write should fail
        with pytest.raises(Exception, match="Insufficient disk space"):
            await write_operation(100)  # 100GB

        # Test disk cleanup strategy
        async def cleanup_disk_space():
            """Cleanup disk space when running low."""
            monitor = resource_simulator.get_monitor(ResourceType.DISK)

            if monitor.is_critical():
                # Simulate cleanup operations
                cleanup_amount = 200  # Clean up 200GB
                new_usage = max(0, monitor.current_usage - cleanup_amount)
                monitor.update_usage(new_usage)

                return {
                    "cleanup_performed": True,
                    "space_freed": cleanup_amount,
                    "new_usage": monitor.get_usage_percentage(),
                }

            return {"cleanup_performed": False}

        cleanup_result = await cleanup_disk_space()
        assert cleanup_result["cleanup_performed"] is True
        assert cleanup_result["space_freed"] > 0

    async def test_file_descriptor_exhaustion(self, resource_simulator):
        """Test file descriptor exhaustion."""
        fd_monitor = resource_simulator.get_monitor(ResourceType.FILE_DESCRIPTORS)

        # Simulate file descriptor usage
        open_files = []

        async def open_file(filename: str):
            """Simulate opening a file."""
            if fd_monitor.current_usage >= fd_monitor.max_capacity:
                raise OSError("Too many open files")

            file_handle = {"name": filename, "fd": len(open_files)}
            open_files.append(file_handle)
            fd_monitor.update_usage(len(open_files))

            return file_handle

        async def close_file(file_handle: dict[str, Any]):
            """Simulate closing a file."""
            if file_handle in open_files:
                open_files.remove(file_handle)
                fd_monitor.update_usage(len(open_files))

        # Open files until exhaustion
        for i in range(int(fd_monitor.max_capacity)):
            await open_file(f"file_{i}")

        # Next file open should fail
        with pytest.raises(OSError, match="Too many open files"):
            await open_file("one_too_many")

        # Test file descriptor management
        async def managed_file_operation():
            """File operation with proper resource management."""
            try:
                file_handle = await open_file("managed_file")
                # Simulate file operation
                await asyncio.sleep(0.001)
                return {"status": "success", "file": file_handle["name"]}
            except OSError:
                # Handle file descriptor exhaustion
                return {"status": "failed", "reason": "no_file_descriptors"}
            finally:
                # Always close file
                if "file_handle" in locals():
                    await close_file(file_handle)

        # Should fail due to exhaustion
        result = await managed_file_operation()
        assert result["status"] == "failed"
        assert result["reason"] == "no_file_descriptors"

    async def test_thread_pool_exhaustion(self, resource_simulator):
        """Test thread pool exhaustion scenarios."""
        thread_monitor = resource_simulator.get_monitor(ResourceType.THREAD_POOL)
        active_tasks = []

        async def submit_task(task_id: str, duration: float = 0.01):
            """Submit task to thread pool."""
            if len(active_tasks) >= thread_monitor.max_capacity:
                # Thread pool exhausted - task queued or rejected
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "queue_position": len(active_tasks)
                    - thread_monitor.max_capacity
                    + 1,
                }

            # Task can be executed immediately
            active_tasks.append(task_id)
            thread_monitor.update_usage(len(active_tasks))

            try:
                await asyncio.sleep(duration)
                return {"task_id": task_id, "status": "completed"}
            finally:
                active_tasks.remove(task_id)
                thread_monitor.update_usage(len(active_tasks))

        # Submit tasks to exhaust thread pool
        tasks = []
        for i in range(int(thread_monitor.max_capacity) + 5):  # Exceed capacity
            task = submit_task(f"task_{i}", duration=0.1)
            tasks.append(task)

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks)

        # Some tasks should be queued
        queued_tasks = [r for r in results if r["status"] == "queued"]
        completed_tasks = [r for r in results if r["status"] == "completed"]

        assert len(queued_tasks) > 0, "Some tasks should be queued"
        assert len(completed_tasks) <= thread_monitor.max_capacity, (
            "Not more tasks than pool capacity should complete immediately"
        )

    async def test_cascading_resource_exhaustion(self, resource_simulator):
        """Test cascading resource exhaustion scenarios."""
        # Simulate scenario where one resource exhaustion leads to others

        # Start with memory exhaustion
        await resource_simulator.simulate_memory_exhaustion(950, duration=0.05)

        # Memory pressure leads to increased disk I/O (swap)
        await resource_simulator.simulate_disk_exhaustion(900)

        # Disk pressure leads to CPU pressure (I/O wait)
        await resource_simulator.simulate_cpu_exhaustion(90, duration=0.05)

        # Check system health
        health_status = resource_simulator.get_system_health()

        assert health_status["overall_health"] == "critical"
        assert (
            len(health_status["critical_resources"]) >= 2
        )  # Multiple resources in critical state

        # Test system shutdown/restart decision
        async def emergency_resource_management():
            """Emergency resource management when multiple resources are critical."""
            health = resource_simulator.get_system_health()

            if health["overall_health"] == "critical":
                critical_count = len(health["critical_resources"])

                if critical_count >= 3:
                    # System in critical state - emergency measures
                    return {
                        "action": "emergency_shutdown",
                        "reason": "multiple_critical_resources",
                        "affected_resources": [
                            r["resource"] for r in health["critical_resources"]
                        ],
                    }
                elif critical_count >= 2:
                    # Partial shutdown of non-critical services
                    return {
                        "action": "partial_shutdown",
                        "reason": "cascading_resource_exhaustion",
                        "services_stopped": [
                            "analytics",
                            "background_tasks",
                            "non_essential_apis",
                        ],
                    }
                else:
                    # Individual resource management
                    return {
                        "action": "resource_throttling",
                        "reason": "single_resource_critical",
                    }

            return {"action": "monitor", "reason": "system_healthy"}

        emergency_result = await emergency_resource_management()
        assert emergency_result["action"] in ["emergency_shutdown", "partial_shutdown"]
        assert "resource" in emergency_result["reason"]

    async def test_resource_monitoring_and_alerting(self, resource_simulator):
        """Test resource monitoring and alerting systems."""

        class ResourceAlertManager:
            def __init__(self, simulator: ResourceExhaustionSimulator):
                self.simulator = simulator
                self.alerts = []
                self.alert_thresholds = {
                    ResourceType.MEMORY: {"warning": 80, "critical": 95},
                    ResourceType.CPU: {"warning": 85, "critical": 95},
                    ResourceType.DISK: {"warning": 85, "critical": 95},
                }

            async def check_and_alert(self):
                """Check resource levels and generate alerts."""
                new_alerts = []

                for resource_type, monitor in self.simulator.monitors.items():
                    if resource_type in self.alert_thresholds:
                        thresholds = self.alert_thresholds[resource_type]
                        usage = monitor.get_usage_percentage()

                        if usage >= thresholds["critical"]:
                            alert = {
                                "level": "critical",
                                "resource": resource_type.value,
                                "usage": usage,
                                "timestamp": time.time(),
                                "message": f"{resource_type.value} usage critical: {usage:.1f}%",
                            }
                            new_alerts.append(alert)
                        elif usage >= thresholds["warning"]:
                            alert = {
                                "level": "warning",
                                "resource": resource_type.value,
                                "usage": usage,
                                "timestamp": time.time(),
                                "message": f"{resource_type.value} usage high: {usage:.1f}%",
                            }
                            new_alerts.append(alert)

                self.alerts.extend(new_alerts)
                return new_alerts

            def get_alert_summary(self) -> dict[str, Any]:
                """Get summary of all alerts."""
                critical_alerts = [a for a in self.alerts if a["level"] == "critical"]
                warning_alerts = [a for a in self.alerts if a["level"] == "warning"]

                return {
                    "total_alerts": len(self.alerts),
                    "critical_alerts": len(critical_alerts),
                    "warning_alerts": len(warning_alerts),
                    "latest_critical": critical_alerts[-1] if critical_alerts else None,
                    "latest_warning": warning_alerts[-1] if warning_alerts else None,
                }

        alert_manager = ResourceAlertManager(resource_simulator)

        # Trigger resource exhaustion to generate alerts
        await resource_simulator.simulate_memory_exhaustion(950, duration=0.05)
        await resource_simulator.simulate_cpu_exhaustion(90, duration=0.05)

        # Check for alerts
        alerts = await alert_manager.check_and_alert()

        assert len(alerts) > 0, "Should generate alerts for high resource usage"

        # Verify alert content
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        [a for a in alerts if a["level"] == "warning"]

        assert len(critical_alerts) > 0, "Should have critical alerts"

        # Get alert summary
        summary = alert_manager.get_alert_summary()
        assert summary["total_alerts"] > 0
        assert summary["critical_alerts"] > 0
        assert summary["latest_critical"] is not None

    async def test_resource_quota_and_limits(self, resource_simulator):
        """Test resource quota and limit enforcement."""

        class ResourceQuotaManager:
            def __init__(self, simulator: ResourceExhaustionSimulator):
                self.simulator = simulator
                self.quotas = {
                    "service_a": {"memory": 200, "cpu": 25, "connections": 20},
                    "service_b": {"memory": 300, "cpu": 30, "connections": 30},
                    "service_c": {"memory": 150, "cpu": 15, "connections": 15},
                }
                self.current_usage = {
                    "service_a": {"memory": 0, "cpu": 0, "connections": 0},
                    "service_b": {"memory": 0, "cpu": 0, "connections": 0},
                    "service_c": {"memory": 0, "cpu": 0, "connections": 0},
                }

            async def allocate_resource(
                self, service: str, resource_type: str, amount: float
            ) -> dict[str, Any]:
                """Allocate resource with quota checking."""
                if service not in self.quotas:
                    return {"status": "error", "reason": "unknown_service"}

                current = self.current_usage[service][resource_type]
                quota = self.quotas[service][resource_type]

                if current + amount > quota:
                    return {
                        "status": "denied",
                        "reason": "quota_exceeded",
                        "requested": amount,
                        "available": max(0, quota - current),
                        "quota": quota,
                    }

                # Allocate resource
                self.current_usage[service][resource_type] += amount

                return {
                    "status": "allocated",
                    "service": service,
                    "resource_type": resource_type,
                    "amount": amount,
                    "new_usage": self.current_usage[service][resource_type],
                    "quota": quota,
                }

            def get_quota_status(self, service: str) -> dict[str, Any]:
                """Get quota status for service."""
                if service not in self.quotas:
                    return {"error": "unknown_service"}

                quota = self.quotas[service]
                usage = self.current_usage[service]

                status = {}
                for resource_type in quota:
                    utilization = (usage[resource_type] / quota[resource_type]) * 100
                    status[resource_type] = {
                        "used": usage[resource_type],
                        "quota": quota[resource_type],
                        "available": quota[resource_type] - usage[resource_type],
                        "utilization_percent": utilization,
                    }

                return status

        quota_manager = ResourceQuotaManager(resource_simulator)

        # Test normal allocation
        allocation_result = await quota_manager.allocate_resource(
            "service_a", "memory", 100
        )
        assert allocation_result["status"] == "allocated"

        # Test quota enforcement
        over_quota_result = await quota_manager.allocate_resource(
            "service_a", "memory", 150
        )  # Exceeds quota
        assert over_quota_result["status"] == "denied"
        assert over_quota_result["reason"] == "quota_exceeded"

        # Check quota status
        quota_status = quota_manager.get_quota_status("service_a")
        assert quota_status["memory"]["used"] == 100
        assert quota_status["memory"]["available"] == 100
        assert quota_status["memory"]["utilization_percent"] == 50.0
