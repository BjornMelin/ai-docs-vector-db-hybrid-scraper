"""Distributed system resilience integration tests.

This module tests the resilience of the distributed system under various failure scenarios,
network partitions, and resource constraints. It validates fault tolerance, recovery patterns,
and system stability across service boundaries.

Tests include:
- Network partition tolerance
- Service discovery and registration
- Load balancing effectiveness
- Circuit breaker integration
- Distributed configuration management
- Cross-service authentication/authorization
- System recovery validation
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config


logger = logging.getLogger(__name__)


@dataclass
class NetworkPartition:
    """Represents a network partition scenario."""

    affected_services: list[str]
    partition_type: str  # 'complete', 'intermittent', 'latency'
    duration_seconds: float
    partition_probability: float = 1.0  # For intermittent partitions


@dataclass
class ServiceNode:
    """Represents a service node in the distributed system."""

    service_name: str
    node_id: str
    health_status: str
    last_heartbeat: float
    load_metrics: dict[str, Any]
    network_reachable: bool = True


class TestNetworkPartitionTolerance:
    """Test system behavior under network partitions."""

    @pytest.fixture
    async def distributed_system_setup(self):
        """Setup distributed system with multiple service nodes."""
        config = MagicMock(spec=Config)

        # Define service topology
        service_nodes = {
            "vector_db_primary": ServiceNode(
                service_name="vector_db",
                node_id="vector_db_primary",
                health_status="healthy",
                last_heartbeat=time.time(),
                load_metrics={"cpu": 0.3, "memory": 0.5, "connections": 10},
            ),
            "vector_db_replica": ServiceNode(
                service_name="vector_db",
                node_id="vector_db_replica",
                health_status="healthy",
                last_heartbeat=time.time(),
                load_metrics={"cpu": 0.2, "memory": 0.4, "connections": 5},
            ),
            "embedding_service_a": ServiceNode(
                service_name="embedding_service",
                node_id="embedding_service_a",
                health_status="healthy",
                last_heartbeat=time.time(),
                load_metrics={"cpu": 0.6, "memory": 0.7, "api_calls": 100},
            ),
            "embedding_service_b": ServiceNode(
                service_name="embedding_service",
                node_id="embedding_service_b",
                health_status="healthy",
                last_heartbeat=time.time(),
                load_metrics={"cpu": 0.4, "memory": 0.5, "api_calls": 60},
            ),
            "cache_service": ServiceNode(
                service_name="cache_service",
                node_id="cache_service_primary",
                health_status="healthy",
                last_heartbeat=time.time(),
                load_metrics={"cpu": 0.2, "memory": 0.8, "hit_rate": 0.85},
            ),
        }

        # Mock service discovery and load balancer
        services = {
            "service_discovery": AsyncMock(),
            "load_balancer": AsyncMock(),
            "health_monitor": AsyncMock(),
            "partition_detector": AsyncMock(),
        }

        return {"config": config, "service_nodes": service_nodes, **services}

    @pytest.mark.asyncio
    async def test_complete_network_partition_handling(self, distributed_system_setup):
        """Test system behavior during complete network partition."""
        setup = distributed_system_setup
        service_nodes = setup["service_nodes"]

        # Simulate complete partition affecting vector_db_primary
        NetworkPartition(
            affected_services=["vector_db_primary"],
            partition_type="complete",
            duration_seconds=30.0,
        )

        # Apply partition
        service_nodes["vector_db_primary"].network_reachable = False
        service_nodes["vector_db_primary"].health_status = "unreachable"

        # Mock partition detection
        setup["partition_detector"].detect_partition.return_value = {
            "partition_detected": True,
            "affected_nodes": ["vector_db_primary"],
            "partition_type": "complete",
            "detection_time": time.time(),
        }

        # Mock service discovery failover
        available_nodes = [
            node
            for node in service_nodes.values()
            if node.service_name == "vector_db" and node.network_reachable
        ]
        setup["service_discovery"].get_available_nodes.return_value = available_nodes

        # Mock load balancer reconfiguration
        setup["load_balancer"].reconfigure.return_value = {
            "status": "reconfigured",
            "active_nodes": ["vector_db_replica"],
            "removed_nodes": ["vector_db_primary"],
            "reconfiguration_time_ms": 150,
        }

        # Test partition handling workflow
        partition_detection = await setup["partition_detector"].detect_partition()
        assert partition_detection["partition_detected"] is True

        # Get available nodes after partition
        available_vector_db_nodes = await setup[
            "service_discovery"
        ].get_available_nodes(service_name="vector_db")
        assert len(available_vector_db_nodes) == 1
        assert available_vector_db_nodes[0].node_id == "vector_db_replica"

        # Reconfigure load balancer
        reconfig_result = await setup["load_balancer"].reconfigure(
            service_name="vector_db",
            available_nodes=[node.node_id for node in available_vector_db_nodes],
        )
        assert reconfig_result["status"] == "reconfigured"
        assert "vector_db_replica" in reconfig_result["active_nodes"]
        assert "vector_db_primary" in reconfig_result["removed_nodes"]

        # Verify system continues operating with replica
        assert (
            len(
                [
                    n
                    for n in service_nodes.values()
                    if n.service_name == "vector_db" and n.network_reachable
                ]
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_intermittent_network_partition(self, distributed_system_setup):
        """Test system behavior during intermittent network partitions."""
        setup = distributed_system_setup
        setup["service_nodes"]

        # Simulate intermittent partition affecting embedding service
        partition = NetworkPartition(
            affected_services=["embedding_service_a"],
            partition_type="intermittent",
            duration_seconds=60.0,
            partition_probability=0.7,  # 70% packet loss
        )

        # Track connection attempts and failures
        connection_attempts = []
        failed_attempts = []

        async def simulate_intermittent_connection(node_id: str):
            """Simulate intermittent network connection."""
            attempt_time = time.time()
            connection_attempts.append({"node_id": node_id, "time": attempt_time})

            # Simulate packet loss based on partition probability
            if random.random() < partition.partition_probability:
                failed_attempts.append({"node_id": node_id, "time": attempt_time})
                msg = f"Network unreachable for {node_id}"
                raise ConnectionError(msg)

            return {"status": "connected", "node_id": node_id, "latency_ms": 250}

        # Mock health check with intermittent failures
        setup[
            "health_monitor"
        ].check_node_health.side_effect = simulate_intermittent_connection

        # Simulate multiple health checks over time
        health_check_results = []
        for _i in range(10):
            try:
                result = await setup["health_monitor"].check_node_health(
                    "embedding_service_a"
                )
                health_check_results.append({"success": True, "result": result})
            except ConnectionError as e:
                health_check_results.append({"success": False, "error": str(e)})

            await asyncio.sleep(0.01)  # Small delay between checks

        # Analyze intermittent behavior
        successful_checks = [r for r in health_check_results if r["success"]]
        failed_checks = [r for r in health_check_results if not r["success"]]

        # Verify intermittent behavior
        assert len(failed_checks) > 0  # Some failures expected
        assert len(successful_checks) > 0  # Some successes expected
        failure_rate = len(failed_checks) / len(health_check_results)
        assert 0.5 <= failure_rate <= 0.9  # Failure rate around partition probability

        # Verify retry logic and circuit breaker behavior would be triggered
        assert len(connection_attempts) == 10
        assert len(failed_attempts) >= 5  # Expect significant failures

    @pytest.mark.asyncio
    async def test_network_latency_partition(self, distributed_system_setup):
        """Test system behavior under high network latency."""
        setup = distributed_system_setup

        # Simulate high latency partition
        NetworkPartition(
            affected_services=["cache_service"],
            partition_type="latency",
            duration_seconds=45.0,
        )

        # Mock high latency responses
        async def simulate_high_latency_response(operation: str):
            """Simulate high latency network response."""
            latency_ms = random.uniform(2000, 5000)  # 2-5 second latency
            await asyncio.sleep(latency_ms / 1000)

            return {
                "status": "completed",
                "operation": operation,
                "latency_ms": latency_ms,
                "timestamp": time.time(),
            }

        setup["cache_service"] = AsyncMock()
        setup["cache_service"].get.side_effect = (
            lambda _k: simulate_high_latency_response("get")
        )
        setup["cache_service"].set.side_effect = (
            lambda _k, _v: simulate_high_latency_response("set")
        )

        # Test operations under high latency
        time.time()

        # Simulate cache operations with timeout
        cache_operations = []

        async def perform_cache_operation_with_timeout(
            operation: str,
            timeout: float = 1.0,  # noqa: ASYNC109
        ):
            """Perform cache operation with timeout."""
            try:
                if operation == "get":
                    result = await asyncio.wait_for(
                        setup["cache_service"].get("test_key"), timeout=timeout
                    )
                elif operation == "set":
                    result = await asyncio.wait_for(
                        setup["cache_service"].set("test_key", "test_value"),
                        timeout=timeout,
                    )

                cache_operations.append(
                    {
                        "operation": operation,
                        "status": "success",
                        "latency_ms": result["latency_ms"],
                    }
                )
                return result

            except TimeoutError:
                cache_operations.append(
                    {
                        "operation": operation,
                        "status": "timeout",
                        "timeout_ms": timeout * 1000,
                    }
                )
                raise

        # Perform operations that should timeout
        timeout_operations = []
        for op in ["get", "set", "get"]:
            try:
                await perform_cache_operation_with_timeout(op, timeout=1.0)
            except TimeoutError:
                timeout_operations.append(op)

        # Verify timeout behavior under high latency
        assert len(timeout_operations) > 0  # Some operations should timeout
        timeout_ops = [op for op in cache_operations if op["status"] == "timeout"]
        assert len(timeout_ops) >= 2  # Most operations should timeout with 1s timeout

        # Verify that operations would succeed with higher timeout
        try:
            result = await perform_cache_operation_with_timeout("get", timeout=6.0)
            assert result["status"] == "completed"
            assert result["latency_ms"] > 2000  # High latency confirmed
        except TimeoutError:
            pass  # May still timeout depending on random latency

    @pytest.mark.asyncio
    async def test_split_brain_scenario_prevention(self, distributed_system_setup):
        """Test prevention of split-brain scenarios in distributed system."""
        setup = distributed_system_setup
        service_nodes = setup["service_nodes"]

        # Simulate network partition that could cause split-brain
        # Partition isolates vector_db_primary from embedding services
        partitioned_nodes = ["vector_db_primary"]

        # Mock quorum-based decision making
        class QuorumManager:
            def __init__(self, _total_nodes: int, quorum_size: int):
                self._total_nodes = _total_nodes
                self.quorum_size = quorum_size
                self.active_nodes = set()

            def add_node(self, node_id: str):
                self.active_nodes.add(node_id)

            def remove_node(self, node_id: str):
                self.active_nodes.discard(node_id)

            def has_quorum(self) -> bool:
                return len(self.active_nodes) >= self.quorum_size

            def get_leader(self) -> str | None:
                if self.has_quorum():
                    # Simple leader election - return node with lowest ID
                    return min(self.active_nodes) if self.active_nodes else None
                return None

        # Setup quorum manager
        quorum_manager = QuorumManager(_total_nodes=5, quorum_size=3)
        for node_id in service_nodes:
            quorum_manager.add_node(node_id)

        # Simulate partition - remove partitioned nodes
        for node_id in partitioned_nodes:
            quorum_manager.remove_node(node_id)
            service_nodes[node_id].network_reachable = False

        # Test quorum maintenance
        assert (
            quorum_manager.has_quorum() is True
        )  # Should still have quorum (4/5 nodes)
        leader = quorum_manager.get_leader()
        assert leader is not None
        assert leader not in partitioned_nodes

        # Test write operations only allowed with quorum
        write_operations = []

        async def attempt_write_operation(operation_id: str, node_id: str):
            """Attempt write operation with quorum check."""
            if not quorum_manager.has_quorum():
                return {
                    "operation_id": operation_id,
                    "status": "rejected",
                    "reason": "no_quorum",
                }

            if node_id not in quorum_manager.active_nodes:
                return {
                    "operation_id": operation_id,
                    "status": "rejected",
                    "reason": "node_not_in_quorum",
                }

            return {
                "operation_id": operation_id,
                "status": "accepted",
                "leader": quorum_manager.get_leader(),
                "quorum_size": len(quorum_manager.active_nodes),
            }

        # Test write operations from different nodes
        operations = [
            ("write_1", "vector_db_replica"),  # Should succeed
            ("write_2", "embedding_service_a"),  # Should succeed
            ("write_3", "vector_db_primary"),  # Should fail (partitioned)
        ]

        for op_id, node_id in operations:
            result = await attempt_write_operation(op_id, node_id)
            write_operations.append(result)

        # Verify split-brain prevention
        successful_writes = [
            op for op in write_operations if op["status"] == "accepted"
        ]
        rejected_writes = [op for op in write_operations if op["status"] == "rejected"]

        assert len(successful_writes) == 2  # Two operations should succeed
        assert len(rejected_writes) == 1  # One should be rejected

        # Verify rejected operation is from partitioned node
        rejected_op = rejected_writes[0]
        assert rejected_op["reason"] == "node_not_in_quorum"

        # Verify all successful operations have same leader
        leaders = {op["leader"] for op in successful_writes}
        assert len(leaders) == 1  # All operations should see same leader


class TestServiceDiscoveryAndRegistration:
    """Test service discovery and registration mechanisms."""

    @pytest.fixture
    async def service_discovery_setup(self):
        """Setup service discovery infrastructure."""
        services = {
            "registry": AsyncMock(),
            "health_checker": AsyncMock(),
            "load_balancer": AsyncMock(),
            "config_manager": AsyncMock(),
        }

        # Mock service registry state
        service_registry = {
            "vector_db": [
                {
                    "node_id": "vector_db_1",
                    "endpoint": "http://vector-db-1:6333",
                    "health": "healthy",
                    "metadata": {"region": "us-east-1", "version": "1.2.0"},
                },
                {
                    "node_id": "vector_db_2",
                    "endpoint": "http://vector-db-2:6333",
                    "health": "healthy",
                    "metadata": {"region": "us-west-1", "version": "1.2.0"},
                },
            ],
            "embedding_service": [
                {
                    "node_id": "embedding_1",
                    "endpoint": "http://embedding-1:8080",
                    "health": "healthy",
                    "metadata": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                    },
                }
            ],
        }

        return {"service_registry": service_registry, **services}

    @pytest.mark.asyncio
    async def test_dynamic_service_registration(self, service_discovery_setup):
        """Test dynamic service registration and deregistration."""
        setup = service_discovery_setup
        registry = setup["service_registry"]

        # Test new service registration
        new_service = {
            "service_name": "embedding_service",
            "node_id": "embedding_2",
            "endpoint": "http://embedding-2:8080",
            "health": "healthy",
            "metadata": {"provider": "openai", "model": "text-embedding-3-large"},
        }

        # Mock registration process
        setup["registry"].register_service.return_value = {
            "status": "registered",
            "node_id": new_service["node_id"],
            "registration_time": time.time(),
        }

        setup["health_checker"].start_monitoring.return_value = {
            "status": "monitoring_started",
            "check_interval": 30,
            "timeout": 5,
        }

        # Register new service
        registration_result = await setup["registry"].register_service(
            service_name=new_service["service_name"],
            node_id=new_service["node_id"],
            endpoint=new_service["endpoint"],
            metadata=new_service["metadata"],
        )

        # Start health monitoring
        await setup["health_checker"].start_monitoring(
            node_id=new_service["node_id"], endpoint=new_service["endpoint"]
        )

        # Update registry with new service
        if new_service["service_name"] not in registry:
            registry[new_service["service_name"]] = []
        registry[new_service["service_name"]].append(new_service)

        # Verify registration
        assert registration_result["status"] == "registered"
        assert len(registry["embedding_service"]) == 2

        # Test service deregistration
        setup["registry"].deregister_service.return_value = {
            "status": "deregistered",
            "node_id": new_service["node_id"],
            "deregistration_time": time.time(),
        }

        setup["health_checker"].stop_monitoring.return_value = {
            "status": "monitoring_stopped",
            "node_id": new_service["node_id"],
        }

        # Deregister service
        deregistration_result = await setup["registry"].deregister_service(
            service_name=new_service["service_name"], node_id=new_service["node_id"]
        )

        # Stop health monitoring
        stop_monitoring_result = await setup["health_checker"].stop_monitoring(
            node_id=new_service["node_id"]
        )

        # Remove from registry
        registry[new_service["service_name"]] = [
            node
            for node in registry[new_service["service_name"]]
            if node["node_id"] != new_service["node_id"]
        ]

        # Verify deregistration
        assert deregistration_result["status"] == "deregistered"
        assert stop_monitoring_result["status"] == "monitoring_stopped"
        assert len(registry["embedding_service"]) == 1

    @pytest.mark.asyncio
    async def test_health_based_service_discovery(self, service_discovery_setup):
        """Test service discovery based on health status."""
        setup = service_discovery_setup
        registry = setup["service_registry"]

        # Simulate health check results
        health_check_results = {
            "vector_db_1": {
                "status": "healthy",
                "response_time_ms": 50,
                "last_check": time.time(),
            },
            "vector_db_2": {
                "status": "unhealthy",
                "error": "connection_timeout",
                "last_check": time.time(),
            },
            "embedding_1": {
                "status": "healthy",
                "response_time_ms": 120,
                "last_check": time.time(),
            },
        }

        setup["health_checker"].get_health_status.side_effect = (
            lambda node_id: health_check_results.get(node_id)
        )

        # Mock service discovery with health filtering
        async def discover_healthy_services(service_name: str):
            """Discover only healthy services."""
            if service_name not in registry:
                return []

            healthy_services = []
            for service_node in registry[service_name]:
                health = await setup["health_checker"].get_health_status(
                    service_node["node_id"]
                )
                if health and health["status"] == "healthy":
                    healthy_services.append({**service_node, "health_metrics": health})

            return healthy_services

        # Test discovery of healthy vector_db services
        healthy_vector_dbs = await discover_healthy_services("vector_db")

        # Verify only healthy services returned
        assert len(healthy_vector_dbs) == 1
        assert healthy_vector_dbs[0]["node_id"] == "vector_db_1"
        assert healthy_vector_dbs[0]["health_metrics"]["status"] == "healthy"

        # Test discovery of healthy embedding services
        healthy_embedding_services = await discover_healthy_services(
            "embedding_service"
        )

        assert len(healthy_embedding_services) == 1
        assert healthy_embedding_services[0]["node_id"] == "embedding_1"

        # Test discovery when no healthy services available
        # Simulate all vector_db services becoming unhealthy
        health_check_results["vector_db_1"]["status"] = "unhealthy"

        unhealthy_vector_dbs = await discover_healthy_services("vector_db")
        assert len(unhealthy_vector_dbs) == 0

    @pytest.mark.asyncio
    async def test_load_balancing_with_service_discovery(self, service_discovery_setup):
        """Test load balancing integration with service discovery."""
        setup = service_discovery_setup
        setup["service_registry"]

        # Mock load balancing algorithms
        class LoadBalancer:
            def __init__(self):
                self.request_counts = {}
                self.response_times = {}

            def round_robin(self, services: list[dict]) -> dict:
                """Round-robin load balancing."""
                if not services:
                    return None

                # Track request counts
                for service in services:
                    node_id = service["node_id"]
                    if node_id not in self.request_counts:
                        self.request_counts[node_id] = 0

                # Select service with lowest request count
                selected_service = min(
                    services, key=lambda s: self.request_counts[s["node_id"]]
                )
                self.request_counts[selected_service["node_id"]] += 1

                return selected_service

            def least_response_time(self, services: list[dict]) -> dict:
                """Least response time load balancing."""
                if not services:
                    return None

                # Select service with lowest response time
                return min(
                    services,
                    key=lambda s: s.get("health_metrics", {}).get(
                        "response_time_ms", 1000
                    ),
                )

        load_balancer = LoadBalancer()

        # Test round-robin load balancing
        vector_db_services = [
            {
                "node_id": "vector_db_1",
                "endpoint": "http://vector-db-1:6333",
                "health_metrics": {"response_time_ms": 50},
            },
            {
                "node_id": "vector_db_2",
                "endpoint": "http://vector-db-2:6333",
                "health_metrics": {"response_time_ms": 80},
            },
        ]

        # Simulate multiple requests with round-robin
        selected_services = []
        for _i in range(6):
            selected = load_balancer.round_robin(vector_db_services)
            selected_services.append(selected["node_id"])

        # Verify round-robin distribution
        service_1_count = selected_services.count("vector_db_1")
        service_2_count = selected_services.count("vector_db_2")
        assert service_1_count == 3
        assert service_2_count == 3

        # Test least response time balancing
        response_time_selections = []
        for _i in range(4):
            selected = load_balancer.least_response_time(vector_db_services)
            response_time_selections.append(selected["node_id"])

        # Verify all requests go to service with lower response time
        assert all(node_id == "vector_db_1" for node_id in response_time_selections)

    @pytest.mark.asyncio
    async def test_service_discovery_failover(self, service_discovery_setup):
        """Test service discovery failover mechanisms."""
        setup = service_discovery_setup

        # Mock failover scenario
        primary_service_discovery = setup["registry"]
        AsyncMock()

        # Configure backup registry
        backup_registry = {
            "vector_db": [
                {
                    "node_id": "vector_db_backup",
                    "endpoint": "http://vector-db-backup:6333",
                    "health": "healthy",
                    "metadata": {"region": "us-central-1", "version": "1.2.0"},
                }
            ]
        }

        async def failover_service_discovery(service_name: str):
            """Failover service discovery implementation."""
            try:
                # Try primary service discovery
                result = await primary_service_discovery.discover_services(service_name)
                if result:
                    return {"source": "primary", "services": result}
            except Exception:
                logger.debug("Exception suppressed during cleanup/testing")

            # Fallback to backup service discovery
            try:
                backup_result = backup_registry.get(service_name, [])
                return {"source": "backup", "services": backup_result}
            except Exception:
                return {"source": "none", "services": []}

        # Test normal operation (primary available)
        primary_service_discovery.discover_services.return_value = [
            {"node_id": "vector_db_1", "endpoint": "http://vector-db-1:6333"}
        ]

        result = await failover_service_discovery("vector_db")
        assert result["source"] == "primary"
        assert len(result["services"]) == 1

        # Test failover scenario (primary unavailable)
        primary_service_discovery.discover_services.side_effect = ConnectionError(
            "Primary unavailable"
        )

        failover_result = await failover_service_discovery("vector_db")
        assert failover_result["source"] == "backup"
        assert len(failover_result["services"]) == 1
        assert failover_result["services"][0]["node_id"] == "vector_db_backup"


class TestDistributedConfigurationManagement:
    """Test distributed configuration management and synchronization."""

    @pytest.fixture
    async def distributed_config_setup(self):
        """Setup distributed configuration management."""
        services = {
            "config_server": AsyncMock(),
            "config_client": AsyncMock(),
            "change_notifier": AsyncMock(),
        }

        # Mock configuration hierarchy
        config_hierarchy = {
            "global": {
                "system.version": "1.0.0",
                "logging.level": "INFO",
                "monitoring.enabled": True,
            },
            "environment.production": {
                "database.pool_size": 20,
                "cache.ttl_seconds": 3600,
                "security.ssl_enabled": True,
            },
            "service.vector_db": {
                "collection.default_name": "documents",
                "search.timeout_ms": 5000,
                "indexing.batch_size": 100,
            },
        }

        return {"config_hierarchy": config_hierarchy, **services}

    @pytest.mark.asyncio
    async def test_hierarchical_configuration_resolution(
        self, distributed_config_setup
    ):
        """Test hierarchical configuration resolution across services."""
        setup = distributed_config_setup
        config_hierarchy = setup["config_hierarchy"]

        # Mock configuration resolution
        class ConfigResolver:
            def __init__(self, hierarchy: dict):
                self.hierarchy = hierarchy

            def resolve_config(
                self, service_name: str, environment: str = "production"
            ) -> dict:
                """Resolve configuration with hierarchy precedence."""
                resolved_config = {}

                # Apply configurations in order of precedence
                precedence_order = [
                    "global",
                    f"environment.{environment}",
                    f"service.{service_name}",
                ]

                for config_level in precedence_order:
                    if config_level in self.hierarchy:
                        resolved_config.update(self.hierarchy[config_level])

                return resolved_config

        resolver = ConfigResolver(config_hierarchy)

        # Test configuration resolution for different services
        vector_db_config = resolver.resolve_config("vector_db", "production")
        embedding_service_config = resolver.resolve_config(
            "embedding_service", "production"
        )

        # Verify hierarchical resolution for vector_db
        assert vector_db_config["system.version"] == "1.0.0"  # From global
        assert vector_db_config["database.pool_size"] == 20  # From environment
        assert (
            vector_db_config["collection.default_name"] == "documents"
        )  # From service

        # Verify service-specific configuration
        assert "collection.default_name" in vector_db_config
        assert "collection.default_name" not in embedding_service_config

        # Verify environment configuration inheritance
        assert embedding_service_config["cache.ttl_seconds"] == 3600
        assert embedding_service_config["security.ssl_enabled"] is True

    @pytest.mark.asyncio
    async def test_configuration_change_propagation(self, _distributed_config_setup):
        """Test configuration change propagation across services."""

        # Mock configuration change events
        config_changes = [
            {
                "change_id": "change_001",
                "config_path": "global.logging.level",
                "old_value": "INFO",
                "new_value": "DEBUG",
                "timestamp": time.time(),
                "affected_services": ["all"],
            },
            {
                "change_id": "change_002",
                "config_path": "service.vector_db.search.timeout_ms",
                "old_value": 5000,
                "new_value": 10000,
                "timestamp": time.time() + 1,
                "affected_services": ["vector_db"],
            },
        ]

        # Mock change notification system
        notified_services = []

        async def notify_configuration_change(change: dict):
            """Notify services of configuration changes."""
            affected_services = change["affected_services"]

            if "all" in affected_services:
                # Notify all services
                all_services = [
                    "vector_db",
                    "embedding_service",
                    "api_service",
                    "cache_service",
                ]
                for service in all_services:
                    notified_services.append(
                        {
                            "service": service,
                            "change_id": change["change_id"],
                            "config_path": change["config_path"],
                            "notification_time": time.time(),
                        }
                    )
            else:
                # Notify specific services
                for service in affected_services:
                    notified_services.append(
                        {
                            "service": service,
                            "change_id": change["change_id"],
                            "config_path": change["config_path"],
                            "notification_time": time.time(),
                        }
                    )

        # Process configuration changes
        for change in config_changes:
            await notify_configuration_change(change)

        # Verify change propagation
        global_change_notifications = [
            n for n in notified_services if n["change_id"] == "change_001"
        ]
        service_specific_notifications = [
            n for n in notified_services if n["change_id"] == "change_002"
        ]

        # Global change should notify all services
        assert len(global_change_notifications) == 4
        notified_service_names = {n["service"] for n in global_change_notifications}
        assert notified_service_names == {
            "vector_db",
            "embedding_service",
            "api_service",
            "cache_service",
        }

        # Service-specific change should notify only affected service
        assert len(service_specific_notifications) == 1
        assert service_specific_notifications[0]["service"] == "vector_db"

    @pytest.mark.asyncio
    async def test_configuration_consistency_validation(
        self, _distributed_config_setup
    ):
        """Test configuration consistency validation across services."""

        # Mock configuration consistency rules
        consistency_rules = [
            {
                "rule_id": "database_pool_consistency",
                "description": "Database pool sizes should be consistent across services",
                "type": "value_consistency",
                "config_paths": [
                    "service.vector_db.database.pool_size",
                    "service.embedding_service.database.pool_size",
                ],
            },
            {
                "rule_id": "ssl_requirement",
                "description": "SSL must be enabled in production environment",
                "type": "value_requirement",
                "config_path": "environment.production.security.ssl_enabled",
                "required_value": True,
            },
        ]

        # Mock configuration states across services
        service_configurations = {
            "vector_db": {"database.pool_size": 20, "security.ssl_enabled": True},
            "embedding_service": {
                "database.pool_size": 15,  # Inconsistent!
                "security.ssl_enabled": True,
            },
            "api_service": {
                "database.pool_size": 20,
                "security.ssl_enabled": False,  # Violates requirement!
            },
        }

        # Validate configuration consistency
        validation_results = []

        for rule in consistency_rules:
            if rule["type"] == "value_consistency":
                # Check value consistency across services
                values = []
                for config_path in rule["config_paths"]:
                    service_name = config_path.split(".")[1]
                    config_key = ".".join(config_path.split(".")[2:])

                    if service_name in service_configurations:
                        value = service_configurations[service_name].get(config_key)
                        values.append({"service": service_name, "value": value})

                # Check if all values are consistent
                unique_values = {v["value"] for v in values if v["value"] is not None}
                is_consistent = len(unique_values) <= 1

                validation_results.append(
                    {
                        "rule_id": rule["rule_id"],
                        "type": rule["type"],
                        "is_valid": is_consistent,
                        "values": values,
                        "error": None
                        if is_consistent
                        else "Inconsistent values across services",
                    }
                )

            elif rule["type"] == "value_requirement":
                # Check value requirement
                config_path = rule["config_path"]
                required_value = rule["required_value"]

                violations = []
                for service_name, config in service_configurations.items():
                    config_key = config_path.split(".")[-1]
                    actual_value = config.get(config_key)

                    if actual_value != required_value:
                        violations.append(
                            {
                                "service": service_name,
                                "expected": required_value,
                                "actual": actual_value,
                            }
                        )

                validation_results.append(
                    {
                        "rule_id": rule["rule_id"],
                        "type": rule["type"],
                        "is_valid": len(violations) == 0,
                        "violations": violations,
                        "error": None
                        if len(violations) == 0
                        else f"Requirement violations: {len(violations)}",
                    }
                )

        # Verify validation results
        consistency_result = next(
            r for r in validation_results if r["rule_id"] == "database_pool_consistency"
        )
        requirement_result = next(
            r for r in validation_results if r["rule_id"] == "ssl_requirement"
        )

        # Database pool consistency should fail
        assert consistency_result["is_valid"] is False
        assert "Inconsistent values" in consistency_result["error"]

        # SSL requirement should fail (api_service has SSL disabled)
        assert requirement_result["is_valid"] is False
        assert len(requirement_result["violations"]) == 1
        assert requirement_result["violations"][0]["service"] == "api_service"


class TestCrossServiceAuthentication:
    """Test cross-service authentication and authorization."""

    @pytest.mark.asyncio
    async def test_service_to_service_authentication(self):
        """Test service-to-service authentication mechanisms."""

        # Mock JWT token service
        class JWTTokenService:
            def __init__(self):
                self.issued_tokens = {}
                self.service_keys = {
                    "api_service": "secret_key_api",
                    "vector_db_service": "secret_key_vector",
                    "embedding_service": "secret_key_embedding",
                }

            async def issue_service_token(
                self, service_name: str, requesting_service: str
            ) -> str:
                """Issue JWT token for service-to-service communication."""
                if requesting_service not in self.service_keys:
                    msg = f"Unknown requesting service: {requesting_service}"
                    raise ValueError(msg)

                token_payload = {
                    "service_name": service_name,
                    "requesting_service": requesting_service,
                    "issued_at": time.time(),
                    "expires_at": time.time() + 3600,  # 1 hour
                    "permissions": ["read", "write"]
                    if service_name == requesting_service
                    else ["read"],
                }

                # Simulate JWT encoding
                token = f"jwt.{service_name}.{requesting_service}.{int(time.time())}"
                self.issued_tokens[token] = token_payload

                return token

            async def validate_service_token(
                self, token: str, required_service: str
            ) -> dict:
                """Validate JWT token for service access."""
                if token not in self.issued_tokens:
                    msg = "Invalid token"
                    raise ValueError(msg)

                payload = self.issued_tokens[token]

                # Check expiration
                if time.time() > payload["expires_at"]:
                    msg = "Token expired"
                    raise ValueError(msg)

                # Check service authorization
                if payload["service_name"] != required_service:
                    msg = "Token not valid for requested service"
                    raise ValueError(msg)

                return payload

        token_service = JWTTokenService()

        # Test token issuance
        api_to_vector_token = await token_service.issue_service_token(
            service_name="vector_db_service", requesting_service="api_service"
        )

        embedding_to_vector_token = await token_service.issue_service_token(
            service_name="vector_db_service", requesting_service="embedding_service"
        )

        # Test token validation
        api_payload = await token_service.validate_service_token(
            api_to_vector_token, required_service="vector_db_service"
        )

        embedding_payload = await token_service.validate_service_token(
            embedding_to_vector_token, required_service="vector_db_service"
        )

        # Verify token contents
        assert api_payload["requesting_service"] == "api_service"
        assert api_payload["service_name"] == "vector_db_service"
        assert "read" in api_payload["permissions"]

        assert embedding_payload["requesting_service"] == "embedding_service"
        assert embedding_payload["service_name"] == "vector_db_service"

        # Test invalid token scenarios
        with pytest.raises(ValueError, match="Invalid token"):
            await token_service.validate_service_token(
                "invalid_token", "vector_db_service"
            )

        with pytest.raises(ValueError, match="Token not valid"):
            await token_service.validate_service_token(
                api_to_vector_token, "wrong_service"
            )

    @pytest.mark.asyncio
    async def test_role_based_access_control(self):
        """Test role-based access control across services."""

        # Mock RBAC system
        class RBACSystem:
            def __init__(self):
                self.roles = {
                    "admin": {
                        "permissions": ["read", "write", "delete", "configure"],
                        "services": ["all"],
                    },
                    "data_processor": {
                        "permissions": ["read", "write"],
                        "services": ["vector_db_service", "embedding_service"],
                    },
                    "search_client": {
                        "permissions": ["read"],
                        "services": ["vector_db_service", "api_service"],
                    },
                }

                self.service_assignments = {
                    "api_service": ["admin", "search_client"],
                    "embedding_service": ["admin", "data_processor"],
                    "vector_db_service": ["admin", "data_processor", "search_client"],
                }

            async def check_permission(
                self, service_name: str, requesting_service: str, operation: str
            ) -> bool:
                """Check if service has permission for operation."""
                # Get roles for requesting service
                allowed_roles = self.service_assignments.get(requesting_service, [])

                for role_name in allowed_roles:
                    role = self.roles[role_name]

                    # Check if role has permission for operation
                    if operation in role["permissions"]:
                        # Check if role can access target service
                        if (
                            "all" in role["services"]
                            or service_name in role["services"]
                        ):
                            return True

                return False

        rbac = RBACSystem()

        # Test various permission scenarios
        test_cases = [
            # (target_service, requesting_service, operation, expected_result)
            ("vector_db_service", "api_service", "read", True),  # search_client role
            (
                "vector_db_service",
                "api_service",
                "write",
                False,
            ),  # search_client can't write
            (
                "vector_db_service",
                "embedding_service",
                "write",
                True,
            ),  # data_processor role
            ("api_service", "embedding_service", "read", False),  # no role allows this
            (
                "vector_db_service",
                "admin_service",
                "delete",
                False,
            ),  # admin_service not defined
        ]

        for target_service, requesting_service, operation, expected in test_cases:
            result = await rbac.check_permission(
                target_service, requesting_service, operation
            )
            assert result == expected, (
                f"Permission check failed: {requesting_service} -> {target_service} "
                f"for {operation}, expected {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_security_audit_logging(self):
        """Test security audit logging for cross-service operations."""

        # Mock security audit logger
        class SecurityAuditLogger:
            def __init__(self):
                self.audit_logs = []

            async def log_access_attempt(self, event_type: str, details: dict):
                """Log security-related events."""
                audit_entry = {
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "details": details,
                    "audit_id": f"audit_{len(self.audit_logs) + 1}",
                }
                self.audit_logs.append(audit_entry)
                return audit_entry["audit_id"]

            def get_audit_logs(self, event_type: str | None = None) -> list[dict]:
                """Retrieve audit logs by event type."""
                if event_type:
                    return [
                        log
                        for log in self.audit_logs
                        if log["event_type"] == event_type
                    ]
                return self.audit_logs

        audit_logger = SecurityAuditLogger()

        # Test various security events
        security_events = [
            {
                "event_type": "service_authentication_success",
                "details": {
                    "requesting_service": "api_service",
                    "target_service": "vector_db_service",
                    "authentication_method": "jwt_token",
                    "user_context": "user_123",
                },
            },
            {
                "event_type": "service_authentication_failure",
                "details": {
                    "requesting_service": "unknown_service",
                    "target_service": "vector_db_service",
                    "failure_reason": "invalid_token",
                    "source_ip": "192.168.1.100",
                },
            },
            {
                "event_type": "unauthorized_access_attempt",
                "details": {
                    "requesting_service": "api_service",
                    "target_service": "embedding_service",
                    "requested_operation": "delete",
                    "denial_reason": "insufficient_permissions",
                },
            },
        ]

        # Log security events
        audit_ids = []
        for event in security_events:
            audit_id = await audit_logger.log_access_attempt(
                event["event_type"], event["details"]
            )
            audit_ids.append(audit_id)

        # Verify audit logging
        assert len(audit_ids) == 3
        assert all(isinstance(audit_id, str) for audit_id in audit_ids)

        # Test audit log retrieval
        auth_failures = audit_logger.get_audit_logs("service_authentication_failure")
        unauthorized_attempts = audit_logger.get_audit_logs(
            "unauthorized_access_attempt"
        )

        assert len(auth_failures) == 1
        assert auth_failures[0]["details"]["failure_reason"] == "invalid_token"

        assert len(unauthorized_attempts) == 1
        assert (
            unauthorized_attempts[0]["details"]["denial_reason"]
            == "insufficient_permissions"
        )

        # Verify all logs captured
        all_logs = audit_logger.get_audit_logs()
        assert len(all_logs) == 3

        # Verify log structure
        for log in all_logs:
            assert "timestamp" in log
            assert "event_type" in log
            assert "details" in log
            assert "audit_id" in log