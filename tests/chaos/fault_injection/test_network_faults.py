"""Network fault injection tests for chaos engineering.

This module implements network-level fault injection to test system resilience
against network failures, timeouts, and connectivity issues.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from tests.chaos.conftest import FailureType


class TestError(Exception):
    """Custom exception for this module."""

    pass


@pytest.mark.chaos
@pytest.mark.fault_injection
@pytest.mark.network_chaos
class TestNetworkFaultInjection:
    """Test network fault injection scenarios."""

    async def test_network_timeout_injection(
        self, fault_injector, _resilience_validator, mock_resilient_service
    ):
        """Test network timeout fault injection."""

        # Create a service that will timeout
        async def timeout_service():
            await asyncio.sleep(0.1)  # Simulate normal operation
            raise TimeoutError("Simulated timeout")

        # Inject network timeout fault
        fault_id = await fault_injector.inject_network_timeout(
            target="test_service", timeout_seconds=0.05, failure_rate=1.0
        )

        # Verify fault is active
        active_faults = fault_injector.get_active_faults()
        assert fault_id in active_faults
        assert active_faults[fault_id]["type"] == FailureType.NETWORK_TIMEOUT

        # Test that the fault is triggered
        with pytest.raises(asyncio.TimeoutError):
            await active_faults[fault_id]["fault_func"]()

        # Clean up fault
        fault_injector.remove_fault(fault_id)

        # Verify fault is removed
        assert fault_id not in fault_injector.get_active_faults()
        assert len(fault_injector.get_fault_history()) == 1

    async def test_connection_failure_injection(
        self, fault_injector, mock_resilient_service
    ):
        """Test connection failure injection."""
        # Inject connection failure
        fault_id = await fault_injector.inject_connection_failure(
            target="database", failure_rate=1.0
        )

        # Verify fault injection
        active_faults = fault_injector.get_active_faults()
        assert fault_id in active_faults
        assert active_faults[fault_id]["type"] == FailureType.CONNECTION_REFUSED

        # Test that connection fails
        with pytest.raises(ConnectionError, match="Simulated connection failure"):
            await active_faults[fault_id]["fault_func"]()

        # Clean up
        fault_injector.remove_fault(fault_id)

    async def test_partial_network_failure(self, fault_injector, _resilience_validator):
        """Test partial network failure with intermittent connectivity."""
        # Inject partial failure (70% success rate)
        fault_id = await fault_injector.inject_partial_failure(
            target="api_gateway", success_rate=0.7
        )

        # Test multiple requests to verify partial failure behavior
        successes = 0
        failures = 0
        total_requests = 100

        for _ in range(total_requests):
            try:
                await fault_injector.get_active_faults()[fault_id]["fault_func"]()
                successes += 1
            except Exception:
                failures += 1

        # Verify partial failure behavior (should have some failures)
        assert failures > 0, "No failures detected in partial failure test"
        assert successes > 0, "No successes detected in partial failure test"

        # Success rate should be roughly around 70% (with some tolerance)
        actual_success_rate = successes / total_requests
        assert 0.6 <= actual_success_rate <= 0.8, (
            f"Success rate {actual_success_rate} outside expected range"
        )

        # Clean up
        fault_injector.remove_fault(fault_id)

    async def test_latency_spike_injection(self, fault_injector):
        """Test latency spike injection."""
        # Inject latency spikes
        fault_id = await fault_injector.inject_latency_spike(
            target="search_service",
            latency_seconds=0.1,
            spike_rate=1.0,  # Always spike for testing
        )

        # Measure latency
        start_time = time.time()
        await fault_injector.get_active_faults()[fault_id]["fault_func"]()
        latency = time.time() - start_time

        # Verify latency spike occurred
        assert latency >= 0.1, f"Expected latency >= 0.1s, got {latency}s"

        # Clean up
        fault_injector.remove_fault(fault_id)

    async def test_temporary_fault_context_manager(self, fault_injector):
        """Test temporary fault injection using context manager."""
        # Test different fault types with context manager
        fault_types = [
            ("network_timeout", {"timeout_seconds": 0.1}),
            ("connection_failure", {}),
            ("service_unavailable", {}),
            ("latency_spike", {"latency_seconds": 0.05}),
        ]

        for fault_type, kwargs in fault_types:
            async with fault_injector.temporary_fault(
                fault_type=fault_type,
                target="test_service",
                duration_seconds=0.1,
                **kwargs,
            ) as fault_id:
                # Verify fault is active during context
                assert fault_id in fault_injector.get_active_faults()

                # Test fault behavior
                active_fault = fault_injector.get_active_faults()[fault_id]
                if fault_type in [
                    "network_timeout",
                    "connection_failure",
                    "service_unavailable",
                ]:
                    with pytest.raises(Exception):
                        await active_fault["fault_func"]()
                else:  # latency_spike
                    start_time = time.time()
                    await active_fault["fault_func"]()
                    latency = time.time() - start_time
                    assert latency >= 0.05

            # Verify fault is removed after context
            assert fault_id not in fault_injector.get_active_faults()

    async def test_concurrent_fault_injection(self, fault_injector):
        """Test multiple concurrent fault injections."""
        # Inject multiple faults simultaneously
        fault_ids = []

        # Network timeout on service A
        fault_ids.append(
            await fault_injector.inject_network_timeout(
                target="service_a", timeout_seconds=0.1
            )
        )

        # Connection failure on service B
        fault_ids.append(
            await fault_injector.inject_connection_failure(target="service_b")
        )

        # Latency spike on service C
        fault_ids.append(
            await fault_injector.inject_latency_spike(
                target="service_c", latency_seconds=0.05
            )
        )

        # Verify all faults are active
        active_faults = fault_injector.get_active_faults()
        assert len(active_faults) == 3

        for fault_id in fault_ids:
            assert fault_id in active_faults

        # Test each fault independently
        for fault_id in fault_ids:
            fault = active_faults[fault_id]
            if fault["type"] in [
                FailureType.NETWORK_TIMEOUT,
                FailureType.CONNECTION_REFUSED,
            ]:
                with pytest.raises(Exception):
                    await fault["fault_func"]()
            else:  # latency spike
                start_time = time.time()
                await fault["fault_func"]()
                latency = time.time() - start_time
                assert latency >= 0.05

        # Clean up all faults
        fault_injector.clear_all_faults()

        # Verify all faults are cleared
        assert len(fault_injector.get_active_faults()) == 0
        assert len(fault_injector.get_fault_history()) == 3

    async def test_network_fault_with_circuit_breaker(
        self, _fault_injector, resilience_validator
    ):
        """Test network fault injection with circuit breaker validation."""
        failure_count = 0
        circuit_breaker_triggered = False

        async def failing_service():
            nonlocal failure_count, circuit_breaker_triggered
            failure_count += 1

            # Simulate circuit breaker after 3 failures
            if failure_count >= 3:
                raise TestError("Circuit breaker open")
                raise TestError("Circuit breaker open")

            raise ConnectionError("Network failure")

        # Validate circuit breaker behavior
        result = await resilience_validator.validate_circuit_breaker(
            service_func=failing_service, failure_threshold=3, recovery_timeout=0.1
        )

        # Verify circuit breaker was triggered
        assert result["circuit_breaker_triggered"], (
            "Circuit breaker should have been triggered"
        )
        assert result["failure_count"] >= 3, (
            f"Expected >= 3 failures, got {result['failure_count']}"
        )
        assert circuit_breaker_triggered, "Circuit breaker flag should be set"

    @pytest.mark.parametrize("failure_rate", [0.1, 0.5, 0.9])
    async def test_variable_failure_rates(self, fault_injector, failure_rate: float):
        """Test fault injection with different failure rates."""
        fault_id = await fault_injector.inject_partial_failure(
            target="variable_service", success_rate=1.0 - failure_rate
        )

        # Test multiple requests
        failures = 0
        total_requests = 50

        for _ in range(total_requests):
            try:
                await fault_injector.get_active_faults()[fault_id]["fault_func"]()
            except Exception:
                failures += 1

        actual_failure_rate = failures / total_requests

        # Allow 20% tolerance for randomness
        expected_min = max(0, failure_rate - 0.2)
        expected_max = min(1, failure_rate + 0.2)

        assert expected_min <= actual_failure_rate <= expected_max, (
            f"Failure rate {actual_failure_rate} outside expected range "
            f"[{expected_min}, {expected_max}] for configured rate {failure_rate}"
        )

        # Clean up
        fault_injector.remove_fault(fault_id)


@pytest.mark.chaos
@pytest.mark.fault_injection
@pytest.mark.network_chaos
class TestNetworkChaosWithRealServices:
    """Test network chaos with mock real service interactions."""

    @pytest.fixture
    def mock_vector_db_service(self):
        """Mock vector database service."""
        service = AsyncMock()
        service.health_check = AsyncMock(return_value={"status": "healthy"})
        service.search = AsyncMock(return_value={"results": []})
        service.insert = AsyncMock(return_value={"success": True})
        return service

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = AsyncMock()
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        service.health_check = AsyncMock(return_value={"status": "healthy"})
        return service

    async def test_vector_db_connection_failure_scenario(
        self, _fault_injector, resilience_validator, mock_vector_db_service
    ):
        """Test vector database connection failure and recovery."""
        # Simulate connection failure
        mock_vector_db_service.search.side_effect = ConnectionError(
            "DB connection failed"
        )
        mock_vector_db_service.health_check.side_effect = ConnectionError(
            "Health check failed"
        )

        # Test graceful degradation
        async def primary_search():
            return await mock_vector_db_service.search({"query": "test"})

        async def fallback_search():
            # Simulate fallback to cached results
            return {"results": [], "source": "cache"}

        # Validate graceful degradation
        result = await resilience_validator.validate_graceful_degradation(
            service_func=primary_search, fallback_func=fallback_search
        )

        assert result["primary_service_failed"], "Primary service should fail"
        assert result["fallback_triggered"], "Fallback should be triggered"
        assert result["fallback_successful"], "Fallback should succeed"

    async def test_embedding_service_timeout_with_retry(
        self, _fault_injector, resilience_validator, mock_embedding_service
    ):
        """Test embedding service timeout with retry mechanism."""
        call_count = 0

        async def unreliable_embedding_service():
            nonlocal call_count
            call_count += 1

            # Fail first 2 attempts, succeed on 3rd
            if call_count <= 2:
                raise TimeoutError("Embedding service timeout")

            return [0.1, 0.2, 0.3]

        # Validate retry behavior
        result = await resilience_validator.validate_retry_behavior(
            service_func=unreliable_embedding_service,
            max_retries=3,
            backoff_factor=0.1,  # Fast backoff for testing
        )

        assert result["retry_attempts"] == 3, (
            f"Expected 3 attempts, got {result['retry_attempts']}"
        )
        assert result["success_on_retry"], "Should succeed after retries"
        assert result["backoff_respected"], "Backoff timing should be respected"

    async def test_cascade_failure_prevention(
        self,
        _fault_injector,
        resilience_validator,
        mock_vector_db_service,
        mock_embedding_service,
    ):
        """Test cascade failure prevention mechanisms."""
        # Simulate primary service failure
        mock_vector_db_service.search.side_effect = Exception("Vector DB overloaded")

        # Ensure dependent service doesn't fail due to cascade
        mock_embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]

        # Test that embedding service remains healthy even when vector DB fails
        embedding_result = await mock_embedding_service.embed_text("test query")
        assert embedding_result == [0.1, 0.2, 0.3], (
            "Embedding service should remain functional"
        )

        # Test circuit breaker prevents cascade
        failure_count = 0

        async def protected_vector_search():
            nonlocal failure_count
            failure_count += 1

            # After 3 failures, circuit breaker should open
            if failure_count > 3:
                raise TestError("Circuit breaker open - preventing cascade")

            raise TestError("Vector DB failure")

        # Validate circuit breaker prevents cascade
        circuit_result = await resilience_validator.validate_circuit_breaker(
            service_func=protected_vector_search,
            failure_threshold=3,
            recovery_timeout=0.1,
        )

        assert circuit_result["circuit_breaker_triggered"], (
            "Circuit breaker should prevent cascade"
        )

    async def test_network_partition_simulation(
        self, _fault_injector, resilience_validator
    ):
        """Test network partition simulation and split-brain prevention."""
        # Simulate network partition between services
        partition_active = True

        async def service_a_operation():
            if partition_active:
                raise ConnectionError("Network partition - cannot reach service B")
            return {"status": "success", "service": "A"}

        async def service_b_operation():
            if partition_active:
                raise ConnectionError("Network partition - cannot reach service A")
            return {"status": "success", "service": "B"}

        # Test that both services detect partition
        with pytest.raises(ConnectionError, match="Network partition"):
            await service_a_operation()

        with pytest.raises(ConnectionError, match="Network partition"):
            await service_b_operation()

        # Simulate partition healing
        partition_active = False

        # Test recovery after partition heals
        result_a = await service_a_operation()
        result_b = await service_b_operation()

        assert result_a["status"] == "success"
        assert result_b["status"] == "success"

    async def test_bandwidth_limitation_simulation(self, _fault_injector):
        """Test bandwidth limitation simulation."""

        # Simulate bandwidth limitation by adding delays
        async def bandwidth_limited_operation():
            # Simulate slow data transfer
            data_size_mb = 10
            bandwidth_mbps = 1  # 1 Mbps
            transfer_time = data_size_mb / bandwidth_mbps

            await asyncio.sleep(transfer_time * 0.01)  # Scale down for testing
            return {"transferred": f"{data_size_mb}MB"}

        # Measure operation time
        start_time = time.time()
        result = await bandwidth_limited_operation()
        elapsed_time = time.time() - start_time

        # Verify bandwidth limitation effect
        assert elapsed_time >= 0.01, (
            "Operation should be slowed by bandwidth limitation"
        )
        assert result["transferred"] == "10MB"

    async def test_dns_failure_simulation(self, _fault_injector):
        """Test DNS failure simulation."""
        dns_failure_active = True

        async def dns_dependent_operation(hostname: str):
            if dns_failure_active and hostname not in ["localhost", "127.0.0.1"]:
                raise TestError(f"DNS resolution failed for {hostname}")

            return {"resolved": True, "hostname": hostname}

        # Test DNS failure
        with pytest.raises(Exception, match="DNS resolution failed"):
            await dns_dependent_operation("external-service.com")

        # Test localhost still works (fallback behavior)
        result = await dns_dependent_operation("localhost")
        assert result["resolved"] is True
        assert result["hostname"] == "localhost"
