"""Service fault injection tests for chaos engineering.

This module implements service-level fault injection to test resilience
against service failures, resource exhaustion, and degraded performance.
"""

import asyncio
import time
from typing import Any

import pytest

from tests.chaos.conftest import FailureType


@pytest.mark.chaos
@pytest.mark.fault_injection
class TestServiceFaultInjection:
    """Test service-level fault injection scenarios."""

    async def test_service_unavailable_injection(
        self, fault_injector, mock_resilient_service
    ):
        """Test service unavailable fault injection."""
        # Inject service unavailable fault
        fault_id = await fault_injector.inject_service_unavailable(
            target="web_scraper_service", failure_rate=1.0
        )

        # Verify fault is active
        active_faults = fault_injector.get_active_faults()
        assert fault_id in active_faults
        assert active_faults[fault_id]["type"] == FailureType.SERVICE_UNAVAILABLE

        # Test service failure
        with pytest.raises(Exception, match="Service .* is temporarily unavailable"):
            await active_faults[fault_id]["fault_func"]()

        # Clean up
        fault_injector.remove_fault(fault_id)

    async def test_memory_exhaustion_injection(self, fault_injector):
        """Test memory exhaustion fault injection."""
        # Test different pressure levels
        pressure_levels = ["low", "moderate", "high", "critical"]

        for pressure_level in pressure_levels:
            fault_id = await fault_injector.inject_memory_pressure(
                target="embedding_service", pressure_level=pressure_level
            )

            # Verify fault configuration
            active_fault = fault_injector.get_active_faults()[fault_id]
            assert active_fault["type"] == FailureType.MEMORY_EXHAUSTION

            # Test memory pressure effect (may or may not fail depending on randomness)
            try:
                await active_fault["fault_func"]()
            except MemoryError as e:
                assert "Simulated memory pressure" in str(e)
                assert pressure_level in str(e)

            # Clean up
            fault_injector.remove_fault(fault_id)

    async def test_database_connection_failure(
        self, _fault_injector, resilience_validator
    ):
        """Test database connection failure scenarios."""
        # Mock database service
        db_connection_count = 0
        max_connections = 3

        async def database_operation():
            nonlocal db_connection_count
            db_connection_count += 1

            if db_connection_count > max_connections:
                raise Exception("Database connection pool exhausted")

            # Simulate database operation
            await asyncio.sleep(0.01)
            return {"result": "database_data"}

        # Test connection pool exhaustion
        tasks = []
        for _i in range(5):  # Exceed max connections
            tasks.append(database_operation())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some operations should fail due to connection exhaustion
        failures = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        assert len(failures) > 0, "Expected some database connection failures"
        assert len(successes) <= max_connections, "Too many successful connections"

    async def test_authentication_service_failure(self, _fault_injector):
        """Test authentication service failure scenarios."""
        auth_failure_modes = [
            "token_expired",
            "invalid_credentials",
            "auth_service_down",
            "rate_limit_exceeded",
        ]

        for failure_mode in auth_failure_modes:

            async def auth_operation():
                if failure_mode == "token_expired":
                    raise Exception("Authentication token has expired")
                elif failure_mode == "invalid_credentials":
                    raise Exception("Invalid authentication credentials")
                elif failure_mode == "auth_service_down":
                    raise ConnectionError("Authentication service unavailable")
                elif failure_mode == "rate_limit_exceeded":
                    raise Exception("Authentication rate limit exceeded")

            # Test each failure mode
            with pytest.raises(Exception):
                await auth_operation()

    async def test_cache_service_failures(self, _fault_injector, _resilience_validator):
        """Test cache service failure scenarios."""
        cache_hit_count = 0
        cache_failure_rate = 0.3

        async def cache_get(key: str):
            nonlocal cache_hit_count
            import random

            if random.random() < cache_failure_rate:
                raise Exception("Cache service temporarily unavailable")

            cache_hit_count += 1
            return f"cached_value_for_{key}"

        async def fallback_db_get(key: str):
            # Simulate database fallback
            await asyncio.sleep(0.02)  # DB is slower than cache
            return f"db_value_for_{key}"

        # Test cache failure with fallback
        async def get_data_with_fallback(key: str):
            try:
                return await cache_get(key)
            except Exception:
                return await fallback_db_get(key)

        # Run multiple operations
        results = []
        for i in range(20):
            result = await get_data_with_fallback(f"key_{i}")
            results.append(result)

        # Verify all operations succeeded (either cache or fallback)
        assert len(results) == 20
        assert all(
            result.startswith(("cached_value", "db_value")) for result in results
        )

    async def test_api_rate_limiting_failure(self, _fault_injector):
        """Test API rate limiting scenarios."""
        request_count = 0
        rate_limit = 5  # requests per second
        window_start = time.time()

        async def rate_limited_api_call():
            nonlocal request_count, window_start
            current_time = time.time()

            # Reset counter every second
            if current_time - window_start >= 1.0:
                request_count = 0
                window_start = current_time

            request_count += 1

            if request_count > rate_limit:
                raise Exception("API rate limit exceeded - too many requests")

            return {"status": "success", "data": "api_response"}

        # Test rate limiting
        successes = 0
        failures = 0

        # Make rapid requests
        for _i in range(10):
            try:
                await rate_limited_api_call()
                successes += 1
            except Exception as e:
                if "rate limit exceeded" in str(e):
                    failures += 1
                else:
                    raise

        assert successes <= rate_limit, (
            f"Too many successes: {successes} > {rate_limit}"
        )
        assert failures > 0, "Expected some rate limit failures"

    async def test_microservice_cascade_failure(
        self, _fault_injector, resilience_validator
    ):
        """Test cascade failure prevention in microservices."""
        # Service dependency chain: A -> B -> C
        service_b_failures = 0
        service_c_failures = 0

        async def service_c():
            nonlocal service_c_failures
            service_c_failures += 1
            if service_c_failures > 2:
                raise Exception("Service C is overloaded")
            return {"service": "C", "data": "service_c_data"}

        async def service_b():
            nonlocal service_b_failures
            try:
                result_c = await service_c()
                return {"service": "B", "dependency": result_c}
            except Exception:
                service_b_failures += 1
                # Circuit breaker: fail fast after dependency failure
                if service_b_failures > 1:
                    raise Exception("Service B circuit breaker open")
                raise

        async def service_a():
            try:
                result_b = await service_b()
                return {"service": "A", "dependency": result_b}
            except Exception:
                # Fallback mechanism
                return {"service": "A", "dependency": None, "fallback": True}

        # Test cascade failure prevention
        results = []
        for _i in range(5):
            result = await service_a()
            results.append(result)

        # Verify that Service A can still respond even when B and C fail
        assert len(results) == 5
        fallback_results = [r for r in results if r.get("fallback")]
        assert len(fallback_results) > 0, "Expected some fallback responses"

    async def test_data_corruption_simulation(self, _fault_injector):
        """Test data corruption scenarios."""
        corruption_rate = 0.2

        async def data_processing_service(data: dict[str, Any]):
            import random

            # Simulate data corruption
            if random.random() < corruption_rate:
                # Corrupt the data
                corrupted_data = data.copy()
                corrupted_data["corrupted"] = True
                corrupted_data["checksum_invalid"] = True
                return corrupted_data

            # Normal processing
            processed_data = data.copy()
            processed_data["processed"] = True
            processed_data["checksum_valid"] = True
            return processed_data

        async def data_validator(data: dict[str, Any]):
            # Validate data integrity
            if data.get("checksum_invalid"):
                raise ValueError("Data integrity check failed")
            return data

        # Test data processing with validation
        input_data = {"id": "test_001", "content": "test_content"}

        corruptions_detected = 0
        successful_processing = 0

        for _i in range(20):
            try:
                processed_data = await data_processing_service(input_data)
                await data_validator(processed_data)
                successful_processing += 1
            except ValueError:
                corruptions_detected += 1

        assert corruptions_detected > 0, "Expected some data corruption detection"
        assert successful_processing > 0, "Expected some successful processing"

    async def test_partial_service_degradation(
        self, _fault_injector, resilience_validator
    ):
        """Test partial service degradation scenarios."""
        service_health = "healthy"  # healthy, degraded, critical

        async def adaptive_service():
            if service_health == "healthy":
                # Full functionality
                await asyncio.sleep(0.01)  # Normal response time
                return {
                    "status": "success",
                    "features": ["feature_a", "feature_b", "feature_c"],
                    "response_time": "normal",
                }
            elif service_health == "degraded":
                # Reduced functionality
                await asyncio.sleep(0.05)  # Slower response
                return {
                    "status": "success",
                    "features": ["feature_a"],  # Only core features
                    "response_time": "slow",
                }
            else:  # critical
                raise Exception("Service in critical state")

        # Test healthy state
        result = await adaptive_service()
        assert result["status"] == "success"
        assert len(result["features"]) == 3

        # Test degraded state
        service_health = "degraded"
        result = await adaptive_service()
        assert result["status"] == "success"
        assert len(result["features"]) == 1
        assert result["response_time"] == "slow"

        # Test critical state
        service_health = "critical"
        with pytest.raises(Exception, match="Service in critical state"):
            await adaptive_service()

    async def test_service_startup_failure(self, _fault_injector):
        """Test service startup failure scenarios."""
        startup_attempts = 0
        max_startup_attempts = 3

        async def service_startup():
            nonlocal startup_attempts
            startup_attempts += 1

            if startup_attempts < max_startup_attempts:
                # Simulate startup failure
                raise Exception(f"Service startup failed (attempt {startup_attempts})")

            # Successful startup
            return {"status": "started", "attempts": startup_attempts}

        async def service_manager():
            for attempt in range(5):  # Try up to 5 times
                try:
                    result = await service_startup()
                    return result
                except Exception:
                    if attempt < 4:  # Not the last attempt
                        await asyncio.sleep(0.01)  # Brief delay before retry
                        continue
                    raise

        # Test service startup with retries
        result = await service_manager()
        assert result["status"] == "started"
        assert result["attempts"] == max_startup_attempts

    async def test_service_dependency_timeout(
        self, _fault_injector, resilience_validator
    ):
        """Test service dependency timeout scenarios."""

        async def slow_dependency_service():
            # Simulate slow dependency
            await asyncio.sleep(0.2)
            return {"status": "success", "data": "dependency_data"}

        async def main_service_with_timeout():
            try:
                # Set timeout for dependency call
                result = await asyncio.wait_for(slow_dependency_service(), timeout=0.1)
                return {"status": "success", "dependency": result}
            except TimeoutError:
                # Handle timeout gracefully
                return {
                    "status": "success",
                    "dependency": None,
                    "timeout": True,
                    "fallback_used": True,
                }

        # Test timeout handling
        result = await main_service_with_timeout()
        assert result["status"] == "success"
        assert result.get("timeout") is True
        assert result.get("fallback_used") is True


@pytest.mark.chaos
@pytest.mark.fault_injection
class TestAdvancedServiceFaults:
    """Test advanced service fault scenarios."""

    async def test_split_brain_scenario(self, _fault_injector):
        """Test split-brain scenario prevention."""
        # Simulate distributed system with leader election
        leader = "node_a"
        network_partition = False

        async def elect_leader(node: str):
            nonlocal leader

            if network_partition and node in ["node_b", "node_c"]:
                # Simulate partition: nodes B and C can't communicate with A
                if leader == "node_a":
                    # Potential split-brain: elect new leader
                    leader = "node_b"
                    return {"leader": "node_b", "warning": "potential_split_brain"}

            return {"leader": leader, "status": "normal"}

        # Normal operation
        result = await elect_leader("node_a")
        assert result["leader"] == "node_a"
        assert result["status"] == "normal"

        # Simulate network partition
        network_partition = True
        result = await elect_leader("node_b")
        assert "split_brain" in result.get("warning", "")

    async def test_resource_starvation(self, _fault_injector):
        """Test resource starvation scenarios."""
        # Simulate resource pools
        cpu_pool = {"available": 100, "used": 0}
        memory_pool = {"available": 1000, "used": 0}

        async def resource_intensive_operation(cpu_required: int, memory_required: int):
            # Check resource availability
            if (
                cpu_pool["used"] + cpu_required > cpu_pool["available"]
                or memory_pool["used"] + memory_required > memory_pool["available"]
            ):
                raise Exception("Insufficient resources - operation rejected")

            # Allocate resources
            cpu_pool["used"] += cpu_required
            memory_pool["used"] += memory_required

            try:
                # Simulate work
                await asyncio.sleep(0.01)
                return {
                    "status": "completed",
                    "resources_used": {"cpu": cpu_required, "memory": memory_required},
                }
            finally:
                # Release resources
                cpu_pool["used"] -= cpu_required
                memory_pool["used"] -= memory_required

        # Test resource starvation
        # Schedule operations that exceed available resources
        operations = [
            (30, 300),  # CPU: 30, Memory: 300
            (40, 400),  # CPU: 40, Memory: 400
            (50, 500),  # CPU: 50, Memory: 500 - This should cause starvation
        ]

        results = []
        for cpu, memory in operations:
            try:
                result = await resource_intensive_operation(cpu, memory)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        # Some operations should fail due to resource exhaustion
        errors = [r for r in results if "error" in r]
        assert len(errors) > 0, "Expected resource exhaustion errors"

    async def test_deadlock_detection(self, _fault_injector):
        """Test deadlock detection and prevention."""
        # Simulate resource locks
        lock_a = asyncio.Lock()
        lock_b = asyncio.Lock()

        async def task_1():
            async with lock_a:
                await asyncio.sleep(0.01)
                # Try to acquire lock_b while holding lock_a
                try:
                    async with asyncio.wait_for(lock_b.acquire(), timeout=0.05):
                        await asyncio.sleep(0.01)
                        lock_b.release()
                        return {"task": "task_1", "status": "completed"}
                except TimeoutError:
                    return {"task": "task_1", "status": "deadlock_detected"}

        async def task_2():
            async with lock_b:
                await asyncio.sleep(0.01)
                # Try to acquire lock_a while holding lock_b
                try:
                    async with asyncio.wait_for(lock_a.acquire(), timeout=0.05):
                        await asyncio.sleep(0.01)
                        lock_a.release()
                        return {"task": "task_2", "status": "completed"}
                except TimeoutError:
                    return {"task": "task_2", "status": "deadlock_detected"}

        # Run tasks concurrently to create potential deadlock
        results = await asyncio.gather(task_1(), task_2())

        # At least one task should detect deadlock
        deadlock_detected = any(r.get("status") == "deadlock_detected" for r in results)
        assert deadlock_detected, "Deadlock detection should have been triggered"

    async def test_byzantine_fault_tolerance(self, _fault_injector):
        """Test Byzantine fault tolerance scenarios."""
        # Simulate distributed consensus with Byzantine nodes
        nodes = {
            "node_1": {"value": 42, "byzantine": False},
            "node_2": {"value": 42, "byzantine": False},
            "node_3": {"value": 42, "byzantine": False},
            "node_4": {"value": 999, "byzantine": True},  # Byzantine node
            "node_5": {"value": 42, "byzantine": False},
        }

        async def consensus_algorithm():
            values = [node["value"] for node in nodes.values()]

            # Simple majority consensus
            from collections import Counter

            value_counts = Counter(values)
            consensus_value, count = value_counts.most_common(1)[0]

            # Check if we have majority (> 50%)
            if count > len(nodes) / 2:
                return {
                    "consensus": True,
                    "value": consensus_value,
                    "agreement_count": count,
                }
            else:
                return {"consensus": False, "reason": "no_majority"}

        # Test consensus with Byzantine node
        result = await consensus_algorithm()
        assert result["consensus"] is True
        assert result["value"] == 42  # Correct value despite Byzantine node
        assert result["agreement_count"] >= 4  # 4 out of 5 nodes agree

    async def test_jitter_injection(self, _fault_injector):
        """Test jitter injection for timing-based chaos."""
        import random

        async def timing_sensitive_operation():
            # Add random jitter
            jitter = random.uniform(0.001, 0.02)  # 1-20ms jitter
            await asyncio.sleep(jitter)

            return {"status": "completed", "jitter": jitter}

        # Measure timing variance
        times = []
        for _ in range(10):
            start = time.time()
            await timing_sensitive_operation()
            duration = time.time() - start
            times.append(duration)

        # Verify jitter is present
        min_time = min(times)
        max_time = max(times)
        variance = max_time - min_time

        assert variance > 0.001, "Expected timing variance due to jitter"
        assert max_time < 0.1, "Jitter should not cause excessive delays"
