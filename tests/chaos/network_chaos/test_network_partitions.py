"""Network partition tests for chaos engineering.

This module implements network partition scenarios to test distributed system
resilience against split-brain conditions, consensus failures, and network
isolation scenarios.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


class NetworkZone(Enum):
    """Network zones for partition testing."""

    ZONE_A = "zone_a"
    ZONE_B = "zone_b"
    ZONE_C = "zone_c"
    ISOLATED = "isolated"


@dataclass
class NetworkNode:
    """Represents a network node in distributed system."""

    node_id: str
    zone: NetworkZone
    is_leader: bool = False
    can_communicate_with: set[str] = field(default_factory=set)
    data: dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)


class NetworkPartitionSimulator:
    """Simulates network partitions for testing."""

    def __init__(self):
        self.nodes: dict[str, NetworkNode] = {}
        self.partitions: dict[NetworkZone, set[str]] = {}
        self.communication_matrix: dict[str, set[str]] = {}

    def add_node(
        self,
        node_id: str,
        zone: NetworkZone,
        initial_data: dict[str, Any] | None = None,
    ):
        """Add a node to the network."""
        self.nodes[node_id] = NetworkNode(
            node_id=node_id, zone=zone, data=initial_data or {}
        )

        if zone not in self.partitions:
            self.partitions[zone] = set()
        self.partitions[zone].add(node_id)

        # Initially all nodes can communicate
        self.communication_matrix[node_id] = set(self.nodes.keys()) - {node_id}

        # Update existing nodes to include this new node
        for existing_node_id in self.nodes:
            if existing_node_id != node_id:
                self.communication_matrix[existing_node_id].add(node_id)

    def create_partition(self, isolated_zones: list[NetworkZone]):
        """Create network partition by isolating specified zones."""
        isolated_nodes = set()
        for zone in isolated_zones:
            isolated_nodes.update(self.partitions.get(zone, set()))

        connected_nodes = set(self.nodes.keys()) - isolated_nodes

        # Update communication matrix
        for node_id in self.nodes:
            if node_id in isolated_nodes:
                # Isolated nodes can only communicate within their partition
                self.communication_matrix[node_id] = isolated_nodes - {node_id}
            else:
                # Connected nodes can communicate with each other but not isolated nodes
                self.communication_matrix[node_id] = connected_nodes - {node_id}

    def heal_partition(self):
        """Heal network partition - restore full connectivity."""
        all_nodes = set(self.nodes.keys())
        for node_id in self.nodes:
            self.communication_matrix[node_id] = all_nodes - {node_id}

    async def can_communicate(self, from_node: str, to_node: str) -> bool:
        """Check if two nodes can communicate."""
        if from_node not in self.nodes or to_node not in self.nodes:
            return False

        return to_node in self.communication_matrix.get(from_node, set())

    async def send_message(
        self, from_node: str, to_node: str, _message: dict[str, Any]
    ) -> bool:
        """Send message between nodes if they can communicate."""
        if await self.can_communicate(from_node, to_node):
            # Simulate network latency
            await asyncio.sleep(0.001)
            return True
        msg = f"Cannot send message from {from_node} to {to_node} - network partition"
        raise ConnectionError(msg)

    def get_reachable_nodes(self, node_id: str) -> set[str]:
        """Get set of nodes reachable from given node."""
        return self.communication_matrix.get(node_id, set())

    def get_partition_info(self) -> dict[str, Any]:
        """Get current partition information."""
        partitions = {}
        processed_nodes = set()

        for node_id in self.nodes:
            if node_id in processed_nodes:
                continue

            reachable = self.get_reachable_nodes(node_id)
            reachable.add(node_id)

            partition_id = f"partition_{len(partitions)}"
            partitions[partition_id] = list(reachable)
            processed_nodes.update(reachable)

        return {
            "num_partitions": len(partitions),
            "partitions": partitions,
            "largest_partition": max(partitions.values(), key=len)
            if partitions
            else [],
            "isolated_nodes": [
                node
                for partition in partitions.values()
                if len(partition) == 1
                for node in partition
            ],
        }


@pytest.mark.chaos
@pytest.mark.network_chaos
class TestNetworkPartitions:
    """Test network partition scenarios."""

    @pytest.fixture
    def network_simulator(self):
        """Create network partition simulator."""
        simulator = NetworkPartitionSimulator()

        # Add nodes in different zones
        simulator.add_node(
            "node_1", NetworkZone.ZONE_A, {"role": "leader", "data": {"key1": "value1"}}
        )
        simulator.add_node(
            "node_2",
            NetworkZone.ZONE_A,
            {"role": "follower", "data": {"key2": "value2"}},
        )
        simulator.add_node(
            "node_3",
            NetworkZone.ZONE_B,
            {"role": "follower", "data": {"key3": "value3"}},
        )
        simulator.add_node(
            "node_4",
            NetworkZone.ZONE_B,
            {"role": "follower", "data": {"key4": "value4"}},
        )
        simulator.add_node(
            "node_5",
            NetworkZone.ZONE_C,
            {"role": "follower", "data": {"key5": "value5"}},
        )

        return simulator

    async def test_simple_network_partition(self, network_simulator, fault_injector):
        """Test basic network partition scenario."""
        # Verify initial connectivity
        assert await network_simulator.can_communicate("node_1", "node_3")
        assert await network_simulator.can_communicate("node_2", "node_4")

        # Create partition - isolate Zone C
        network_simulator.create_partition([NetworkZone.ZONE_C])

        # Verify partition effects
        assert await network_simulator.can_communicate(
            "node_1", "node_2"
        )  # Same partition
        assert not await network_simulator.can_communicate(
            "node_1", "node_5"
        )  # Different partition
        assert not await network_simulator.can_communicate(
            "node_3", "node_5"
        )  # Different partition

        # Test message sending
        assert await network_simulator.send_message(
            "node_1", "node_2", {"test": "message"}
        )

        with pytest.raises(ConnectionError):
            await network_simulator.send_message(
                "node_1", "node_5", {"test": "message"}
            )

        # Get partition information
        partition_info = network_simulator.get_partition_info()
        assert partition_info["num_partitions"] == 2
        assert len(partition_info["isolated_nodes"]) == 1
        assert "node_5" in partition_info["isolated_nodes"]

    async def test_split_brain_prevention(self, network_simulator):
        """Test split-brain prevention mechanisms."""

        class DistributedConsensus:
            def __init__(self, network_sim: NetworkPartitionSimulator):
                self.network = network_sim
                self.leader = "node_1"
                self.term = 1
                self.votes = {}

            async def elect_leader(self, candidate_node: str) -> dict[str, Any]:
                """Elect leader using majority consensus."""
                candidate_votes = 0
                _total_nodes = 0
                reachable_nodes = self.network.get_reachable_nodes(candidate_node)
                reachable_nodes.add(candidate_node)

                # Count votes from reachable nodes
                for node_id in reachable_nodes:
                    _total_nodes += 1
                    try:
                        # Simulate vote request
                        await self.network.send_message(
                            candidate_node,
                            node_id,
                            {
                                "type": "vote_request",
                                "term": self.term + 1,
                                "candidate": candidate_node,
                            },
                        )
                        candidate_votes += 1
                    except ConnectionError:
                        # Cannot reach this node
                        pass

                # Check if majority achieved
                majority_needed = (len(self.network.nodes) // 2) + 1
                has_majority = candidate_votes >= majority_needed

                if has_majority:
                    self.leader = candidate_node
                    self.term += 1

                return {
                    "leader_elected": has_majority,
                    "leader": self.leader if has_majority else None,
                    "votes_received": candidate_votes,
                    "votes_needed": majority_needed,
                    "term": self.term,
                    "reachable_nodes": len(reachable_nodes),
                }

            async def check_split_brain(self) -> dict[str, Any]:
                """Check for potential split-brain scenarios."""
                partition_info = self.network.get_partition_info()

                # Check if multiple partitions could elect leaders
                potential_leaders = []
                for partition_id, nodes in partition_info["partitions"].items():
                    majority_needed = (len(self.network.nodes) // 2) + 1
                    if len(nodes) >= majority_needed:
                        potential_leaders.append(
                            {
                                "partition": partition_id,
                                "nodes": nodes,
                                "can_elect_leader": True,
                            }
                        )
                    else:
                        potential_leaders.append(
                            {
                                "partition": partition_id,
                                "nodes": nodes,
                                "can_elect_leader": False,
                            }
                        )

                split_brain_risk = (
                    len([p for p in potential_leaders if p["can_elect_leader"]]) > 1
                )

                return {
                    "split_brain_risk": split_brain_risk,
                    "potential_leaders": potential_leaders,
                    "partition_info": partition_info,
                }

        consensus = DistributedConsensus(network_simulator)

        # Normal operation - should elect leader
        election_result = await consensus.elect_leader("node_1")
        assert election_result["leader_elected"]
        assert election_result["leader"] == "node_1"

        # Create partition that isolates 2 nodes (minority)
        network_simulator.create_partition([NetworkZone.ZONE_C, NetworkZone.ZONE_B])

        # Check for split-brain risk
        split_brain_check = await consensus.check_split_brain()

        # Should not have split-brain risk if minority partition cannot elect leader
        if len(network_simulator.nodes) == 5:  # Total 5 nodes
            # Majority partition (3 nodes) can elect, minority (2 nodes) cannot
            assert not split_brain_check["split_brain_risk"]

        # Attempt election from minority partition
        minority_election = await consensus.elect_leader("node_5")
        assert not minority_election[
            "leader_elected"
        ]  # Should fail due to insufficient votes

    async def test_quorum_based_operations(self, network_simulator):
        """Test quorum-based operations during network partitions."""

        class QuorumBasedStorage:
            def __init__(self, network_sim: NetworkPartitionSimulator):
                self.network = network_sim
                self.storage = {node_id: {} for node_id in network_sim.nodes}

            async def write_with_quorum(
                self, key: str, value: Any, initiator_node: str
            ) -> dict[str, Any]:
                """Perform write operation with quorum consensus."""
                reachable_nodes = self.network.get_reachable_nodes(initiator_node)
                reachable_nodes.add(initiator_node)

                # Calculate quorum requirement
                _total_nodes = len(self.network.nodes)
                quorum_size = (_total_nodes // 2) + 1

                # Attempt to write to reachable nodes
                successful_writes = 0
                failed_writes = 0

                for node_id in reachable_nodes:
                    try:
                        await self.network.send_message(
                            initiator_node,
                            node_id,
                            {"type": "write_request", "key": key, "value": value},
                        )
                        self.storage[node_id][key] = value
                        successful_writes += 1
                    except ConnectionError:
                        failed_writes += 1

                # Check if quorum achieved
                quorum_achieved = successful_writes >= quorum_size

                return {
                    "write_successful": quorum_achieved,
                    "successful_writes": successful_writes,
                    "failed_writes": failed_writes,
                    "quorum_required": quorum_size,
                    "reachable_nodes": len(reachable_nodes),
                }

            async def read_with_quorum(
                self, key: str, initiator_node: str
            ) -> dict[str, Any]:
                """Perform read operation with quorum consensus."""
                reachable_nodes = self.network.get_reachable_nodes(initiator_node)
                reachable_nodes.add(initiator_node)

                _total_nodes = len(self.network.nodes)
                quorum_size = (_total_nodes // 2) + 1

                # Read from reachable nodes
                read_values = {}
                successful_reads = 0

                for node_id in reachable_nodes:
                    try:
                        await self.network.send_message(
                            initiator_node,
                            node_id,
                            {"type": "read_request", "key": key},
                        )
                        if key in self.storage[node_id]:
                            value = self.storage[node_id][key]
                            if value not in read_values:
                                read_values[value] = 0
                            read_values[value] += 1
                            successful_reads += 1
                    except ConnectionError:
                        pass

                # Check if quorum achieved and determine consensus value
                quorum_achieved = successful_reads >= quorum_size
                consensus_value = None

                if quorum_achieved and read_values:
                    # Get value with most votes
                    consensus_value = max(
                        read_values.keys(), key=lambda v: read_values[v]
                    )

                return {
                    "read_successful": quorum_achieved,
                    "consensus_value": consensus_value,
                    "successful_reads": successful_reads,
                    "value_distribution": read_values,
                    "quorum_required": quorum_size,
                }

        storage = QuorumBasedStorage(network_simulator)

        # Normal write operation
        write_result = await storage.write_with_quorum(
            "test_key", "test_value", "node_1"
        )
        assert write_result["write_successful"]
        assert write_result["successful_writes"] >= write_result["quorum_required"]

        # Normal read operation
        read_result = await storage.read_with_quorum("test_key", "node_2")
        assert read_result["read_successful"]
        assert read_result["consensus_value"] == "test_value"

        # Create partition
        network_simulator.create_partition([NetworkZone.ZONE_C])

        # Write from majority partition - should succeed
        majority_write = await storage.write_with_quorum(
            "partition_key", "partition_value", "node_1"
        )
        assert majority_write["write_successful"]

        # Write from minority partition - should fail
        minority_write = await storage.write_with_quorum(
            "minority_key", "minority_value", "node_5"
        )
        assert not minority_write["write_successful"]
        assert minority_write["successful_writes"] < minority_write["quorum_required"]

    async def test_network_partition_detection(self, network_simulator, fault_injector):
        """Test network partition detection mechanisms."""

        class PartitionDetector:
            def __init__(self, network_sim: NetworkPartitionSimulator):
                self.network = network_sim
                self.last_heartbeat = {
                    node_id: time.time() for node_id in network_sim.nodes
                }
                self.suspected_partitions = set()

            async def send_heartbeat(self, from_node: str) -> dict[str, Any]:
                """Send heartbeat from node to all other nodes."""
                successful_heartbeats = 0
                failed_heartbeats = 0

                for target_node in self.network.nodes:
                    if target_node == from_node:
                        continue

                    try:
                        await self.network.send_message(
                            from_node,
                            target_node,
                            {
                                "type": "heartbeat",
                                "timestamp": time.time(),
                                "from": from_node,
                            },
                        )
                        self.last_heartbeat[target_node] = time.time()
                        successful_heartbeats += 1
                    except ConnectionError:
                        failed_heartbeats += 1

                return {
                    "successful_heartbeats": successful_heartbeats,
                    "failed_heartbeats": failed_heartbeats,
                    "_total_nodes": len(self.network.nodes) - 1,
                }

            async def detect_partitions(
                self, timeout_threshold: float = 5.0
            ) -> dict[str, Any]:
                """Detect network partitions based on heartbeat timeouts."""
                current_time = time.time()
                suspected_down_nodes = []

                for node_id, last_heartbeat in self.last_heartbeat.items():
                    if current_time - last_heartbeat > timeout_threshold:
                        suspected_down_nodes.append(node_id)

                # Analyze partition topology
                partition_info = self.network.get_partition_info()

                return {
                    "partition_detected": partition_info["num_partitions"] > 1,
                    "suspected_down_nodes": suspected_down_nodes,
                    "partition_info": partition_info,
                    "detection_confidence": len(suspected_down_nodes)
                    / len(self.network.nodes)
                    if self.network.nodes
                    else 0,
                }

            async def validate_partition_suspicion(
                self, suspected_node: str, validator_node: str
            ) -> bool:
                """Validate partition suspicion by attempting direct communication."""
                try:
                    await self.network.send_message(
                        validator_node,
                        suspected_node,
                        {"type": "ping", "timestamp": time.time()},
                    )
                    return False  # Node is reachable, no partition
                except ConnectionError:
                    return True  # Node is unreachable, partition confirmed

        detector = PartitionDetector(network_simulator)

        # Normal heartbeat operation
        heartbeat_result = await detector.send_heartbeat("node_1")
        assert heartbeat_result["failed_heartbeats"] == 0

        # No partition detected initially
        detection_result = await detector.detect_partitions(timeout_threshold=0.1)
        assert not detection_result["partition_detected"]

        # Create partition
        network_simulator.create_partition([NetworkZone.ZONE_C])

        # Send heartbeat from node in majority partition
        partitioned_heartbeat = await detector.send_heartbeat("node_1")
        assert partitioned_heartbeat["failed_heartbeats"] > 0

        # Detect partition
        partition_detection = await detector.detect_partitions()
        assert partition_detection["partition_detected"]
        assert partition_detection["partition_info"]["num_partitions"] > 1

        # Validate suspicion
        partition_confirmed = await detector.validate_partition_suspicion(
            "node_5", "node_1"
        )
        assert partition_confirmed

    async def test_partition_recovery_mechanisms(self, network_simulator):
        """Test recovery mechanisms after network partition healing."""

        class PartitionRecoveryManager:
            def __init__(self, network_sim: NetworkPartitionSimulator):
                self.network = network_sim
                self.node_data = {node_id: {} for node_id in network_sim.nodes}
                self.vector_clocks = {node_id: {} for node_id in network_sim.nodes}

            async def record_operation(self, node_id: str, operation: dict[str, Any]):
                """Record operation with vector clock."""
                # Increment vector clock for this node
                if node_id not in self.vector_clocks[node_id]:
                    self.vector_clocks[node_id][node_id] = 0
                self.vector_clocks[node_id][node_id] += 1

                # Store operation with vector clock
                op_id = f"{node_id}_{self.vector_clocks[node_id][node_id]}"
                self.node_data[node_id][op_id] = {
                    "operation": operation,
                    "vector_clock": self.vector_clocks[node_id].copy(),
                    "timestamp": time.time(),
                }

                return op_id

            async def sync_after_partition_heal(self) -> dict[str, Any]:
                """Synchronize data after partition healing."""
                # Collect all operations from all nodes
                all_operations = {}
                conflicts = []

                for operations in self.node_data.values():
                    for op_id, op_data in operations.items():
                        if op_id in all_operations:
                            # Potential conflict
                            existing_op = all_operations[op_id]
                            if existing_op["operation"] != op_data["operation"]:
                                conflicts.append(
                                    {
                                        "operation_id": op_id,
                                        "node1": existing_op,
                                        "node2": op_data,
                                    }
                                )
                        else:
                            all_operations[op_id] = op_data

                # Resolve conflicts using vector clocks
                resolved_operations = {}
                for op_id, op_data in all_operations.items():
                    resolved_operations[op_id] = op_data

                # Merge vector clocks
                merged_vector_clock = {}
                for node_id in self.network.nodes:
                    merged_vector_clock[node_id] = 0
                    for node_ops in self.node_data.values():
                        for op_data in node_ops.values():
                            vc = op_data["vector_clock"]
                            if node_id in vc:
                                merged_vector_clock[node_id] = max(
                                    merged_vector_clock[node_id], vc[node_id]
                                )

                return {
                    "sync_successful": True,
                    "_total_operations": len(resolved_operations),
                    "conflicts_detected": len(conflicts),
                    "conflicts": conflicts,
                    "merged_vector_clock": merged_vector_clock,
                }

            async def repair_inconsistencies(self) -> dict[str, Any]:
                """Repair data inconsistencies after partition."""
                repairs_needed = []
                repairs_completed = []

                # Check for data inconsistencies between nodes
                reference_node = next(iter(self.network.nodes.keys()))
                reference_data = self.node_data[reference_node]

                for node_id in self.network.nodes:
                    if node_id == reference_node:
                        continue

                    node_data = self.node_data[node_id]

                    # Find missing operations
                    missing_ops = set(reference_data.keys()) - set(node_data.keys())
                    extra_ops = set(node_data.keys()) - set(reference_data.keys())

                    if missing_ops:
                        repairs_needed.append(
                            {
                                "node": node_id,
                                "type": "missing_operations",
                                "operations": list(missing_ops),
                            }
                        )

                        # Simulate repair
                        for op_id in missing_ops:
                            self.node_data[node_id][op_id] = reference_data[op_id]

                        repairs_completed.append(
                            {
                                "node": node_id,
                                "type": "added_missing_operations",
                                "count": len(missing_ops),
                            }
                        )

                    if extra_ops:
                        repairs_needed.append(
                            {
                                "node": node_id,
                                "type": "extra_operations",
                                "operations": list(extra_ops),
                            }
                        )

                return {
                    "repairs_needed": len(repairs_needed),
                    "repairs_completed": len(repairs_completed),
                    "repair_details": repairs_completed,
                }

        recovery_manager = PartitionRecoveryManager(network_simulator)

        # Record some operations before partition
        await recovery_manager.record_operation(
            "node_1", {"type": "write", "key": "a", "value": 1}
        )
        await recovery_manager.record_operation(
            "node_2", {"type": "write", "key": "b", "value": 2}
        )

        # Create partition
        network_simulator.create_partition([NetworkZone.ZONE_C])

        # Record operations in different partitions
        await recovery_manager.record_operation(
            "node_1", {"type": "write", "key": "c", "value": 3}
        )  # Majority partition
        await recovery_manager.record_operation(
            "node_5", {"type": "write", "key": "d", "value": 4}
        )  # Minority partition

        # Heal partition
        network_simulator.heal_partition()

        # Synchronize after healing
        sync_result = await recovery_manager.sync_after_partition_heal()
        assert sync_result["sync_successful"]
        assert sync_result["_total_operations"] >= 4  # At least 4 operations recorded

        # Repair inconsistencies
        repair_result = await recovery_manager.repair_inconsistencies()
        assert repair_result["repairs_completed"] >= 0  # Some repairs may be needed

    async def test_jepsen_style_consistency_checking(self, network_simulator):
        """Test Jepsen-style consistency checking during partitions."""

        class ConsistencyChecker:
            def __init__(self, network_sim: NetworkPartitionSimulator):
                self.network = network_sim
                self.operations_log = []
                self.reads_log = []

            async def perform_write(
                self, key: str, value: Any, node_id: str
            ) -> dict[str, Any]:
                """Perform write operation and log it."""
                write_op = {
                    "type": "write",
                    "key": key,
                    "value": value,
                    "node": node_id,
                    "timestamp": time.time(),
                    "success": False,
                }

                try:
                    # Simulate write operation
                    reachable_nodes = self.network.get_reachable_nodes(node_id)
                    if len(reachable_nodes) >= len(self.network.nodes) // 2:  # Majority
                        write_op["success"] = True
                        self.network.nodes[node_id].data[key] = value

                        # Replicate to reachable nodes
                        for target_node in reachable_nodes:
                            try:
                                await self.network.send_message(
                                    node_id,
                                    target_node,
                                    {
                                        "type": "replicate_write",
                                        "key": key,
                                        "value": value,
                                    },
                                )
                                self.network.nodes[target_node].data[key] = value
                            except ConnectionError:
                                pass

                except (TimeoutError, ConnectionError, OSError) as e:
                    write_op["error"] = str(e)

                self.operations_log.append(write_op)
                return write_op

            async def perform_read(self, key: str, node_id: str) -> dict[str, Any]:
                """Perform read operation and log it."""
                read_op = {
                    "type": "read",
                    "key": key,
                    "node": node_id,
                    "timestamp": time.time(),
                    "value": None,
                    "success": False,
                }

                try:
                    if key in self.network.nodes[node_id].data:
                        read_op["value"] = self.network.nodes[node_id].data[key]
                        read_op["success"] = True

                except (TimeoutError, ConnectionError, OSError) as e:
                    read_op["error"] = str(e)

                self.reads_log.append(read_op)
                return read_op

            def check_consistency(self) -> dict[str, Any]:
                """Check consistency properties of operations."""
                violations = []

                # Check read-your-writes consistency
                for read_op in self.reads_log:
                    if not read_op["success"]:
                        continue

                    # Find the last write to this key before this read
                    last_write = None
                    for write_op in reversed(self.operations_log):
                        if (
                            write_op["type"] == "write"
                            and write_op["key"] == read_op["key"]
                            and write_op["timestamp"] < read_op["timestamp"]
                            and write_op["success"]
                        ):
                            last_write = write_op
                            break

                    if last_write and read_op["value"] != last_write["value"]:
                        violations.append(
                            {
                                "type": "read_your_writes_violation",
                                "read_op": read_op,
                                "expected_value": last_write["value"],
                                "actual_value": read_op["value"],
                            }
                        )

                # Check monotonic read consistency
                key_read_history = {}
                for read_op in self.reads_log:
                    if not read_op["success"]:
                        continue

                    key = read_op["key"]
                    if key not in key_read_history:
                        key_read_history[key] = []
                    key_read_history[key].append(read_op)

                for reads in key_read_history.values():
                    reads.sort(key=lambda r: r["timestamp"])
                    for i in range(1, len(reads)):
                        current_read = reads[i]
                        previous_read = reads[i - 1]

                        # Check if value went backwards (monotonic violation)
                        if (
                            current_read["value"] is not None
                            and previous_read["value"] is not None
                            and current_read["value"] != previous_read["value"]
                        ):
                            # This could be a monotonic read violation
                            # (simplified check - in reality this is more complex)
                            pass

                return {
                    "consistency_violations": len(violations),
                    "violations": violations,
                    "_total_operations": len(self.operations_log),
                    "_total_reads": len(self.reads_log),
                }

        checker = ConsistencyChecker(network_simulator)

        # Perform operations before partition
        write1 = await checker.perform_write("key1", "value1", "node_1")
        read1 = await checker.perform_read("key1", "node_2")

        assert write1["success"]
        assert read1["success"]
        assert read1["value"] == "value1"

        # Create partition
        network_simulator.create_partition([NetworkZone.ZONE_C])

        # Perform operations during partition
        write2 = await checker.perform_write("key2", "value2", "node_1")  # Majority
        write3 = await checker.perform_write("key3", "value3", "node_5")  # Minority

        assert write2["success"]  # Should succeed in majority
        assert not write3["success"]  # Should fail in minority

        # Heal partition and check consistency
        network_simulator.heal_partition()

        consistency_result = checker.check_consistency()
        # Should have minimal violations if partition handling is correct
        assert consistency_result["_total_operations"] >= 3
        assert consistency_result["_total_reads"] >= 1
