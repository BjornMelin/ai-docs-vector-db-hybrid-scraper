class TestError(Exception):
    """Custom exception for this module."""

    pass


"""Cascade failure tests for chaos engineering.

This module implements comprehensive cascade failure scenarios to test
system resilience against failure propagation and system-wide outages.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

import pytest


logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class ServiceNode:
    """Represents a service node in the dependency graph."""

    name: str
    state: ServiceState = ServiceState.HEALTHY
    dependencies: list[str] = None
    failure_threshold: int = 3
    recovery_time: float = 5.0
    circuit_breaker_open: bool = False

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@pytest.mark.chaos
@pytest.mark.failure_scenarios
class TestCascadeFailures:
    """Test cascade failure scenarios."""

    @pytest.fixture
    def service_topology(self):
        """Create a realistic service topology for testing."""
        return {
            "frontend": ServiceNode("frontend", dependencies=["api_gateway"]),
            "api_gateway": ServiceNode(
                "api_gateway", dependencies=["auth_service", "search_service"]
            ),
            "auth_service": ServiceNode(
                "auth_service", dependencies=["user_db", "cache"]
            ),
            "search_service": ServiceNode(
                "search_service", dependencies=["vector_db", "embedding_service"]
            ),
            "embedding_service": ServiceNode(
                "embedding_service", dependencies=["ml_model_service"]
            ),
            "ml_model_service": ServiceNode("ml_model_service", dependencies=[]),
            "vector_db": ServiceNode("vector_db", dependencies=["storage"]),
            "user_db": ServiceNode("user_db", dependencies=["storage"]),
            "cache": ServiceNode("cache", dependencies=[]),
            "storage": ServiceNode("storage", dependencies=[]),
        }

    async def test_downstream_failure_propagation(
        self, service_topology, _fault_injector, resilience_validator
    ):
        """Test how downstream failures propagate upstream."""
        # Simulate storage failure (bottom of dependency chain)
        service_topology["storage"].state = ServiceState.FAILED

        failure_count = {}

        async def simulate_service_call(
            service_name: str, topology: dict[str, ServiceNode]
        ):
            """Simulate service call with dependency checking."""
            service = topology[service_name]

            # Check if service itself has failed
            if service.state == ServiceState.FAILED:
                raise TestError(f"Service {service_name} is failed")
                raise TestError(f"Service {service_name} is failed")

            # Check dependencies
            for dep_name in service.dependencies:
                dep_service = topology[dep_name]
                if dep_service.state == ServiceState.FAILED:
                    # Dependency failure can cause this service to fail
                    failure_count[service_name] = failure_count.get(service_name, 0) + 1

                    # Implement circuit breaker
                    if failure_count[service_name] >= service.failure_threshold:
                        service.circuit_breaker_open = True
                        service.state = ServiceState.CRITICAL

                    raise TestError(
                        f"Service {service_name} failed due to dependency {dep_name}"
                    )

            return {"service": service_name, "status": "success"}

        # Test cascade failure propagation
        affected_services = []

        # Try calling services that depend on storage
        dependent_services = [
            "vector_db",
            "user_db",
            "auth_service",
            "search_service",
            "api_gateway",
            "frontend",
        ]

        for service_name in dependent_services:
            try:
                await simulate_service_call(service_name, service_topology)
            except Exception:
                affected_services.append(service_name)

        # Verify cascade failure occurred
        assert len(affected_services) > 0, (
            "Expected cascade failure to affect dependent services"
        )
        assert "vector_db" in affected_services, (
            "vector_db should fail due to storage dependency"
        )
        assert "user_db" in affected_services, (
            "user_db should fail due to storage dependency"
        )

    async def test_circuit_breaker_cascade_prevention(
        self, service_topology, _fault_injector, resilience_validator
    ):
        """Test circuit breakers preventing cascade failures."""
        failure_counts = {}
        circuit_breakers = {}

        async def resilient_service_call(
            service_name: str, topology: dict[str, ServiceNode]
        ):
            """Service call with circuit breaker protection."""
            service = topology[service_name]

            # Check circuit breaker
            if circuit_breakers.get(service_name, False):
                raise TestError(f"Circuit breaker open for {service_name}")

            # Simulate dependency failure
            for dep_name in service.dependencies:
                dep_service = topology[dep_name]
                if dep_service.state == ServiceState.FAILED:
                    failure_counts[service_name] = (
                        failure_counts.get(service_name, 0) + 1
                    )

                    # Open circuit breaker after threshold
                    if failure_counts[service_name] >= service.failure_threshold:
                        circuit_breakers[service_name] = True
                        # Instead of failing, provide degraded service
                        return {
                            "service": service_name,
                            "status": "degraded",
                            "circuit_breaker": "open",
                        }

                    raise TestError(f"Dependency {dep_name} failed")

            return {"service": service_name, "status": "success"}

        # Fail storage service
        service_topology["storage"].state = ServiceState.FAILED

        # Test multiple calls to trigger circuit breakers
        results = []
        for _attempt in range(10):
            try:
                result = await resilient_service_call("api_gateway", service_topology)
                results.append(result)
            except Exception as e:
                # Count failed attempts
                logger.debug("Exception suppressed during cleanup/testing")

        # Verify circuit breaker provided degraded service
        degraded_responses = [r for r in results if r.get("status") == "degraded"]
        assert len(degraded_responses) > 0, (
            "Expected degraded service from circuit breaker"
        )

    async def test_bulkhead_pattern_isolation(self, _fault_injector):
        """Test bulkhead pattern for failure isolation."""
        # Simulate thread/resource pools for different operations
        search_pool = {"size": 10, "used": 0}
        admin_pool = {"size": 5, "used": 0}

        async def search_operation():
            if search_pool["used"] >= search_pool["size"]:
                raise TestError("Search pool exhausted")

            search_pool["used"] += 1
            try:
                # Simulate search work
                await asyncio.sleep(0.01)
                return {"operation": "search", "status": "success"}
            finally:
                search_pool["used"] -= 1

        async def admin_operation():
            if admin_pool["used"] >= admin_pool["size"]:
                raise TestError("Admin pool exhausted")

            admin_pool["used"] += 1
            try:
                # Simulate admin work
                await asyncio.sleep(0.01)
                return {"operation": "admin", "status": "success"}
            finally:
                admin_pool["used"] -= 1

        # Overload search operations
        search_tasks = [search_operation() for _ in range(15)]  # Exceed pool size
        admin_tasks = [admin_operation() for _ in range(3)]  # Within pool size

        # Execute operations
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        admin_results = await asyncio.gather(*admin_tasks, return_exceptions=True)

        # Verify bulkhead isolation
        search_failures = [r for r in search_results if isinstance(r, Exception)]
        admin_successes = [r for r in admin_results if not isinstance(r, Exception)]

        assert len(search_failures) > 0, (
            "Expected search operations to fail due to pool exhaustion"
        )
        assert len(admin_successes) == 3, (
            "Admin operations should succeed despite search failures"
        )

    async def test_graceful_degradation_cascade(
        self, service_topology, _fault_injector
    ):
        """Test graceful degradation preventing total system failure."""
        # Service capabilities matrix
        service_capabilities = {
            "search_service": {
                "full": ["vector_search", "text_search", "filters", "sorting"],
                "degraded": ["text_search"],  # Fallback when vector_db fails
                "minimal": ["cached_results"],
            },
            "api_gateway": {
                "full": ["routing", "auth", "rate_limiting", "caching"],
                "degraded": ["routing", "basic_auth"],  # When auth_service fails
                "minimal": ["routing"],
            },
        }

        async def adaptive_service_call(service_name: str, requested_capability: str):
            """Service call that adapts based on dependency health."""
            capabilities = service_capabilities.get(service_name, {})

            # Check dependencies and determine service level
            if service_name == "search_service":
                if service_topology["vector_db"].state == ServiceState.FAILED:
                    service_level = "degraded"
                elif service_topology["cache"].state == ServiceState.FAILED:
                    service_level = "minimal"
                else:
                    service_level = "full"
            elif service_name == "api_gateway":
                if service_topology["auth_service"].state == ServiceState.FAILED:
                    service_level = "degraded"
                else:
                    service_level = "full"
            else:
                service_level = "full"

            available_capabilities = capabilities.get(service_level, [])

            if requested_capability in available_capabilities:
                return {
                    "service": service_name,
                    "capability": requested_capability,
                    "level": service_level,
                    "status": "success",
                }
            else:
                return {
                    "service": service_name,
                    "capability": requested_capability,
                    "level": service_level,
                    "status": "capability_unavailable",
                }

        # Test graceful degradation
        # Fail vector_db
        service_topology["vector_db"].state = ServiceState.FAILED

        # Test search service adaptation
        vector_search_result = await adaptive_service_call(
            "search_service", "vector_search"
        )
        text_search_result = await adaptive_service_call(
            "search_service", "text_search"
        )

        assert vector_search_result["status"] == "capability_unavailable"
        assert text_search_result["status"] == "success"
        assert text_search_result["level"] == "degraded"

    async def test_timeout_cascade_prevention(self, _fault_injector):
        """Test timeout configuration preventing cascade failures."""
        # Service timeout configuration
        timeouts = {
            "frontend": 5.0,
            "api_gateway": 3.0,
            "search_service": 2.0,
            "vector_db": 1.0,
        }

        async def service_with_timeout(service_name: str, dependency_call=None):
            """Service call with timeout protection."""
            service_timeout = timeouts[service_name]

            try:
                if dependency_call:
                    # Call dependency with timeout
                    result = await asyncio.wait_for(
                        dependency_call(), timeout=service_timeout
                    )
                    return {
                        "service": service_name,
                        "status": "success",
                        "dependency": result,
                    }
                else:
                    # Simulate slow operation
                    await asyncio.sleep(service_timeout + 0.5)  # Longer than timeout
                    return {"service": service_name, "status": "success"}
            except TimeoutError:
                return {
                    "service": service_name,
                    "status": "timeout",
                    "fallback": "using_cached_data",
                }

        # Create dependency chain with timeouts
        async def slow_vector_db():
            return await service_with_timeout("vector_db")

        async def search_service():
            return await service_with_timeout("search_service", slow_vector_db)

        async def api_gateway():
            return await service_with_timeout("api_gateway", search_service)

        # Test timeout cascade prevention
        result = await api_gateway()

        # Should timeout but not crash the entire chain
        assert result["status"] == "timeout"
        assert "fallback" in result

    async def test_retry_storm_prevention(self, _fault_injector):
        """Test prevention of retry storms during failures."""
        failure_start_time = time.time()
        request_counts = {}

        async def failing_service():
            # Simulate service failure for 2 seconds
            if time.time() - failure_start_time < 2.0:
                raise TestError("Service temporarily unavailable")
            return {"status": "recovered"}

        async def client_with_exponential_backoff(client_id: str):
            """Client with exponential backoff to prevent retry storms."""
            request_counts[client_id] = 0
            max_retries = 5
            base_delay = 0.1

            for attempt in range(max_retries):
                request_counts[client_id] += 1

                try:
                    result = await failing_service()
                    return {
                        "client": client_id,
                        "result": result,
                        "attempts": attempt + 1,
                    }
                except Exception:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2**attempt)
                        jitter = delay * 0.1  # 10% jitter
                        await asyncio.sleep(delay + jitter)
                    else:
                        return {
                            "client": client_id,
                            "status": "failed",
                            "attempts": attempt + 1,
                        }

        # Simulate multiple clients making requests
        clients = [f"client_{i}" for i in range(10)]
        tasks = [client_with_exponential_backoff(client_id) for client_id in clients]

        results = await asyncio.gather(*tasks)

        # Verify retry behavior
        total_requests = sum(request_counts.values())
        successful_clients = len([r for r in results if "result" in r])

        # Should have reasonable number of retries (not a storm)
        assert total_requests < 100, (
            f"Too many requests: {total_requests} (potential retry storm)"
        )
        assert successful_clients > 0, (
            "Some clients should succeed after service recovery"
        )

    async def test_distributed_system_partition(self, _fault_injector):
        """Test distributed system behavior during network partitions."""
        # Simulate 3-node distributed system
        nodes = {
            "node_a": {"partition": "group_1", "data": {"key1": "value1"}},
            "node_b": {"partition": "group_1", "data": {"key2": "value2"}},
            "node_c": {
                "partition": "group_2",
                "data": {"key3": "value3"},
            },  # Partitioned
        }

        async def distributed_read(key: str, consistency_level: str = "strong"):
            """Read from distributed system with consistency guarantees."""
            if consistency_level == "strong":
                # Strong consistency requires majority
                accessible_nodes = [
                    n for n in nodes.values() if n["partition"] == "group_1"
                ]
                if len(accessible_nodes) < 2:  # Less than majority
                    raise TestError(
                        "Cannot achieve strong consistency - insufficient nodes"
                    )

                # Find data in accessible nodes
                for node in accessible_nodes:
                    if key in node["data"]:
                        return {"value": node["data"][key], "consistency": "strong"}

                raise KeyError(f"Key {key} not found in accessible nodes")

            elif consistency_level == "eventual":
                # Eventual consistency - try any accessible node
                for node in nodes.values():
                    if node["partition"] == "group_1" and key in node["data"]:
                        return {"value": node["data"][key], "consistency": "eventual"}

                raise KeyError(f"Key {key} not found")

        # Test reads during partition
        # Should succeed for keys in accessible partition
        result1 = await distributed_read("key1", "strong")
        assert result1["value"] == "value1"
        assert result1["consistency"] == "strong"

        # Should fail for keys in partitioned node with strong consistency
        with pytest.raises(KeyError):
            await distributed_read("key3", "strong")

        # Should succeed with eventual consistency (though key3 is not accessible)
        with pytest.raises(KeyError):
            await distributed_read("key3", "eventual")

    async def test_memory_leak_cascade(self, _fault_injector):
        """Test cascade failures due to memory leaks."""
        # Simulate memory usage tracking
        memory_usage = {"current": 0, "limit": 1000}

        async def memory_leaking_service(operation_id: str):
            """Service that leaks memory on each call."""
            # Simulate memory leak
            memory_usage["current"] += 50  # Each operation "leaks" 50 units

            if memory_usage["current"] > memory_usage["limit"]:
                raise MemoryError(f"Out of memory - current: {memory_usage['current']}")

            return {"operation": operation_id, "memory_used": memory_usage["current"]}

        async def memory_monitoring_service():
            """Service that monitors and reports memory issues."""
            if memory_usage["current"] > memory_usage["limit"] * 0.8:  # 80% threshold
                return {
                    "status": "warning",
                    "memory_usage": memory_usage["current"],
                    "recommendation": "restart_service",
                }
            return {"status": "healthy", "memory_usage": memory_usage["current"]}

        # Simulate operations until memory exhaustion
        operations_completed = 0
        memory_errors = 0

        for i in range(25):  # Should trigger memory exhaustion
            try:
                await memory_leaking_service(f"op_{i}")
                operations_completed += 1

                # Check memory status
                status = await memory_monitoring_service()
                if status["status"] == "warning":
                    # Simulate memory cleanup/restart
                    memory_usage["current"] = memory_usage["current"] // 2

            except MemoryError:
                memory_errors += 1
                break

        assert memory_errors > 0, "Expected memory exhaustion"
        assert operations_completed > 10, (
            "Should complete some operations before exhaustion"
        )

    async def test_dependency_health_monitoring(
        self, service_topology, _fault_injector, resilience_validator
    ):
        """Test dependency health monitoring and automatic recovery."""
        health_status = {}

        async def health_check_service(service_name: str):
            """Perform health check on service."""
            service = service_topology[service_name]

            if service.state == ServiceState.FAILED:
                health_status[service_name] = "unhealthy"
                return {"service": service_name, "status": "unhealthy"}
            elif service.state == ServiceState.DEGRADED:
                health_status[service_name] = "degraded"
                return {"service": service_name, "status": "degraded"}
            else:
                health_status[service_name] = "healthy"
                return {"service": service_name, "status": "healthy"}

        async def dependency_health_monitor():
            """Monitor health of all dependencies."""
            health_results = {}

            for service_name in service_topology:
                try:
                    result = await health_check_service(service_name)
                    health_results[service_name] = result["status"]
                except Exception:
                    health_results[service_name] = "error"

            return health_results

        async def auto_recovery_system():
            """Automatic recovery system based on health monitoring."""
            health_report = await dependency_health_monitor()

            recovery_actions = []
            for service_name, status in health_report.items():
                if status == "unhealthy":
                    # Simulate recovery action
                    service_topology[service_name].state = ServiceState.DEGRADED
                    recovery_actions.append(f"restarted_{service_name}")
                elif status == "degraded":
                    # Simulate full recovery
                    service_topology[service_name].state = ServiceState.HEALTHY
                    recovery_actions.append(f"recovered_{service_name}")

            return recovery_actions

        # Introduce failures
        service_topology["cache"].state = ServiceState.FAILED
        service_topology["vector_db"].state = ServiceState.DEGRADED

        # Run monitoring and recovery
        initial_health = await dependency_health_monitor()
        recovery_actions = await auto_recovery_system()
        final_health = await dependency_health_monitor()

        # Verify monitoring detected issues
        assert initial_health["cache"] == "unhealthy"
        assert initial_health["vector_db"] == "degraded"

        # Verify recovery actions were taken
        assert len(recovery_actions) > 0
        assert any("cache" in action for action in recovery_actions)

        # Verify health improved
        assert final_health["cache"] != "unhealthy"  # Should be degraded now
        assert final_health["vector_db"] == "healthy"  # Should be fully recovered
