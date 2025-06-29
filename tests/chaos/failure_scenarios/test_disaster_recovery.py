"""Disaster recovery tests for chaos engineering.

This module implements disaster recovery scenarios to test system resilience
against major outages, data loss, and catastrophic failures.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class TestError(Exception):
    """Custom exception for this module."""


class DisasterType(Enum):
    """Types of disaster scenarios."""

    DATACENTER_OUTAGE = "datacenter_outage"
    NETWORK_PARTITION = "network_partition"
    HARDWARE_FAILURE = "hardware_failure"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    NATURAL_DISASTER = "natural_disaster"
    HUMAN_ERROR = "human_error"


@dataclass
class BackupData:
    """Represents backup data for disaster recovery."""

    timestamp: float
    data: dict[str, Any]
    checksum: str
    location: str
    type: str = "full"  # full, incremental, differential


@dataclass
class RecoveryPlan:
    """Disaster recovery plan definition."""

    disaster_type: DisasterType
    recovery_steps: list[str]
    rto: float  # Recovery Time Objective (seconds)
    rpo: float  # Recovery Point Objective (seconds)
    priority: int = 1  # 1=highest, 5=lowest


@pytest.mark.chaos
@pytest.mark.failure_scenarios
class TestDisasterRecovery:
    """Test disaster recovery scenarios."""

    @pytest.fixture
    def backup_system(self):
        """Mock backup system for testing."""
        backups = []

        class BackupSystem:
            def __init__(self):
                self.backups = backups
                self.replication_lag = 0.0

            async def create_backup(
                self, data: dict[str, Any], backup_type: str = "full"
            ):
                backup = BackupData(
                    timestamp=time.time(),
                    data=data.copy(),
                    checksum=str(hash(str(data))),
                    location=f"backup_location_{len(self.backups)}",
                    type=backup_type,
                )
                self.backups.append(backup)
                return backup

            async def restore_from_backup(
                self, backup_id: str | None = None
            ) -> dict[str, Any]:
                if not self.backups:
                    msg = "No backups available"
                    raise TestError(msg)

                if backup_id:
                    backup = next(
                        (b for b in self.backups if b.location == backup_id), None
                    )
                    if not backup:
                        msg = f"Backup {backup_id} not found"
                        raise TestError(msg)
                else:
                    # Get latest backup
                    backup = max(self.backups, key=lambda b: b.timestamp)

                # Simulate restore time
                await asyncio.sleep(0.01)
                return backup.data

            async def verify_backup_integrity(self, backup: BackupData) -> bool:
                # Simulate integrity check
                await asyncio.sleep(0.005)
                current_checksum = str(hash(str(backup.data)))
                return current_checksum == backup.checksum

            def get_latest_backup_age(self) -> float:
                if not self.backups:
                    return float("inf")
                latest = max(self.backups, key=lambda b: b.timestamp)
                return time.time() - latest.timestamp

        return BackupSystem()

    @pytest.fixture
    def recovery_orchestrator(self):
        """Mock recovery orchestrator for testing."""

        class RecoveryOrchestrator:
            def __init__(self):
                self.recovery_plans = {}
                self.recovery_in_progress = False
                self.recovery_start_time = None

            def register_recovery_plan(
                self, disaster_type: DisasterType, plan: RecoveryPlan
            ):
                self.recovery_plans[disaster_type] = plan

            async def execute_recovery(
                self, disaster_type: DisasterType
            ) -> dict[str, Any]:
                if self.recovery_in_progress:
                    msg = "Recovery already in progress"
                    raise TestError(msg)

                plan = self.recovery_plans.get(disaster_type)
                if not plan:
                    msg = f"No recovery plan for {disaster_type}"
                    raise TestError(msg)

                self.recovery_in_progress = True
                self.recovery_start_time = time.time()

                try:
                    # Execute recovery steps
                    completed_steps = []
                    for step in plan.recovery_steps:
                        await self._execute_recovery_step(step)
                        completed_steps.append(step)

                    recovery_time = time.time() - self.recovery_start_time

                    return {
                        "status": "success",
                        "recovery_time": recovery_time,
                        "rto_met": recovery_time <= plan.rto,
                        "completed_steps": completed_steps,
                    }

                finally:
                    self.recovery_in_progress = False

            async def _execute_recovery_step(self, step: str):
                # Simulate recovery step execution
                if "restore_data" in step:
                    await asyncio.sleep(0.05)  # Data restore takes time
                elif "restart_services" in step:
                    await asyncio.sleep(0.02)  # Service restart
                elif "verify_integrity" in step:
                    await asyncio.sleep(0.01)  # Verification
                else:
                    await asyncio.sleep(0.005)  # Generic step

        return RecoveryOrchestrator()

    async def test_datacenter_outage_recovery(
        self, backup_system, recovery_orchestrator, _resilience_validator
    ):
        """Test recovery from complete datacenter outage."""
        # Setup data and backups
        critical_data = {
            "user_accounts": {"user1": "data1", "user2": "data2"},
            "vector_embeddings": {"doc1": [0.1, 0.2, 0.3], "doc2": [0.4, 0.5, 0.6]},
            "search_index": {"term1": ["doc1"], "term2": ["doc2"]},
        }

        # Create backup before disaster
        backup = await backup_system.create_backup(critical_data)

        # Register recovery plan
        recovery_plan = RecoveryPlan(
            disaster_type=DisasterType.DATACENTER_OUTAGE,
            recovery_steps=[
                "activate_secondary_datacenter",
                "restore_data_from_backup",
                "restart_services",
                "verify_data_integrity",
                "redirect_traffic",
            ],
            rto=300.0,  # 5 minutes
            rpo=60.0,  # 1 minute data loss acceptable
        )
        recovery_orchestrator.register_recovery_plan(
            DisasterType.DATACENTER_OUTAGE, recovery_plan
        )

        # Simulate datacenter outage
        datacenter_healthy = False

        async def check_datacenter_health():
            if not datacenter_healthy:
                msg = "Primary datacenter is offline"
                raise TestError(msg)
            return {"status": "healthy"}

        # Verify outage detected
        with pytest.raises(Exception, match="Primary datacenter is offline"):
            await check_datacenter_health()

        # Execute disaster recovery
        recovery_result = await recovery_orchestrator.execute_recovery(
            DisasterType.DATACENTER_OUTAGE
        )

        # Restore data from backup
        restored_data = await backup_system.restore_from_backup(backup.location)

        # Verify recovery
        assert recovery_result["status"] == "success"
        assert recovery_result["rto_met"], (
            f"RTO not met: {recovery_result['recovery_time']}s > {recovery_plan.rto}s"
        )
        assert restored_data == critical_data, "Data not fully restored"

        # Verify backup age meets RPO
        backup_age = backup_system.get_latest_backup_age()
        assert backup_age <= recovery_plan.rpo, (
            f"RPO not met: backup age {backup_age}s > {recovery_plan.rpo}s"
        )

    async def test_data_corruption_recovery(self, backup_system, recovery_orchestrator):
        """Test recovery from data corruption."""
        # Setup original data
        original_data = {
            "documents": {
                "doc1": {
                    "content": "important document",
                    "metadata": {"author": "user1"},
                },
                "doc2": {
                    "content": "another document",
                    "metadata": {"author": "user2"},
                },
            },
            "embeddings": {"doc1": [0.1, 0.2], "doc2": [0.3, 0.4]},
        }

        # Create multiple backups over time
        await backup_system.create_backup(original_data)
        await asyncio.sleep(0.01)  # Time passes

        # Modify data
        modified_data = original_data.copy()
        modified_data["documents"]["doc3"] = {
            "content": "new document",
            "metadata": {"author": "user3"},
        }
        await backup_system.create_backup(modified_data, "incremental")

        await asyncio.sleep(0.01)  # Time passes

        # Simulate data corruption
        corrupted_data = modified_data.copy()
        corrupted_data["documents"]["doc1"]["content"] = "CORRUPTED_DATA_###"
        corrupted_data["embeddings"]["doc1"] = [999.9, 999.9]  # Corrupted embeddings

        async def detect_corruption(data: dict[str, Any]) -> bool:
            """Detect data corruption."""
            # Check for corruption markers
            for doc_id, doc in data.get("documents", {}).items():
                if "CORRUPTED" in doc.get("content", ""):
                    return True

                # Check for anomalous embeddings
                embeddings = data.get("embeddings", {}).get(doc_id, [])
                if any(abs(val) > 100 for val in embeddings):
                    return True

            return False

        # Detect corruption
        corruption_detected = await detect_corruption(corrupted_data)
        assert corruption_detected, "Corruption should be detected"

        # Register recovery plan for data corruption
        corruption_recovery_plan = RecoveryPlan(
            disaster_type=DisasterType.DATA_CORRUPTION,
            recovery_steps=[
                "stop_write_operations",
                "identify_corruption_scope",
                "restore_clean_backup",
                "verify_data_integrity",
                "resume_operations",
            ],
            rto=180.0,  # 3 minutes
            rpo=300.0,  # 5 minutes data loss acceptable
        )
        recovery_orchestrator.register_recovery_plan(
            DisasterType.DATA_CORRUPTION, corruption_recovery_plan
        )

        # Execute corruption recovery
        recovery_result = await recovery_orchestrator.execute_recovery(
            DisasterType.DATA_CORRUPTION
        )

        # Restore from latest good backup
        restored_data = await backup_system.restore_from_backup()

        # Verify recovery
        assert recovery_result["status"] == "success"
        assert not await detect_corruption(restored_data), (
            "Restored data should not be corrupted"
        )
        assert "doc3" in restored_data["documents"], (
            "Should restore latest good version"
        )

    async def test_network_partition_recovery(self, _recovery_orchestrator):
        """Test recovery from network partition (split-brain scenario)."""
        # Simulate distributed system nodes
        nodes = {
            "node_a": {
                "region": "us-east",
                "data": {"key1": "value1"},
                "accessible": True,
            },
            "node_b": {
                "region": "us-west",
                "data": {"key2": "value2"},
                "accessible": True,
            },
            "node_c": {
                "region": "eu-west",
                "data": {"key3": "value3"},
                "accessible": False,
            },  # Partitioned
        }

        quorum_size = 2  # Need majority for operations

        async def check_quorum() -> bool:
            """Check if we have quorum of nodes."""
            accessible_nodes = sum(1 for node in nodes.values() if node["accessible"])
            return accessible_nodes >= quorum_size

        async def distributed_write(key: str, value: str) -> bool:
            """Perform distributed write with quorum."""
            if not await check_quorum():
                msg = "Cannot perform write - insufficient quorum"
                raise TestError(msg)

            # Write to accessible nodes
            success_count = 0
            for node in nodes.values():
                if node["accessible"]:
                    node["data"][key] = value
                    success_count += 1

            return success_count >= quorum_size

        async def resolve_split_brain():
            """Resolve split-brain scenario."""
            # Identify partitioned nodes
            partitioned_nodes = [
                nid for nid, node in nodes.items() if not node["accessible"]
            ]

            # Restore connectivity
            for node_id in partitioned_nodes:
                nodes[node_id]["accessible"] = True

            # Merge data from partitioned nodes
            all_data = {}
            for node in nodes.values():
                all_data.update(node["data"])

            # Replicate merged data to all nodes
            for node in nodes.values():
                node["data"] = all_data.copy()

            return {"resolved": True, "merged_keys": list(all_data.keys())}

        # Test operations with partition
        assert await check_quorum(), "Should have quorum initially"

        # Perform write before partition resolution
        await distributed_write("key4", "value4")

        # Resolve partition
        resolution_result = await resolve_split_brain()

        # Verify resolution
        assert resolution_result["resolved"]
        assert "key3" in resolution_result["merged_keys"], (
            "Data from partitioned node should be merged"
        )
        assert "key4" in resolution_result["merged_keys"], "New data should be included"

        # Verify all nodes have consistent data
        all_node_data = [node["data"] for node in nodes.values()]
        assert all(data == all_node_data[0] for data in all_node_data), (
            "All nodes should have consistent data"
        )

    async def test_security_breach_recovery(self, backup_system, recovery_orchestrator):
        """Test recovery from security breach."""
        # Setup system state before breach
        secure_data = {
            "user_credentials": {
                "user1": "hashed_password1",
                "user2": "hashed_password2",
            },
            "access_tokens": {
                "token1": {"user": "user1", "expires": time.time() + 3600}
            },
            "audit_logs": [
                {"timestamp": time.time() - 100, "action": "login", "user": "user1"},
                {"timestamp": time.time() - 50, "action": "search", "user": "user2"},
            ],
        }

        # Create backup
        await backup_system.create_backup(secure_data)

        # Simulate security breach detection

        async def detect_security_breach() -> dict[str, Any]:
            """Detect security breach indicators."""
            suspicious_activities = [
                {
                    "type": "unauthorized_access",
                    "severity": "high",
                    "source": "unknown_ip",
                },
                {
                    "type": "privilege_escalation",
                    "severity": "critical",
                    "user": "compromised_user",
                },
                {"type": "data_exfiltration", "severity": "high", "volume": "large"},
            ]

            return {
                "breach_detected": True,
                "indicators": suspicious_activities,
                "confidence": 0.95,
            }

        async def containment_actions():
            """Execute containment actions."""
            actions = [
                "revoke_all_access_tokens",
                "disable_compromised_accounts",
                "isolate_affected_systems",
                "collect_forensic_evidence",
            ]

            completed_actions = []
            for action in actions:
                # Simulate containment action
                await asyncio.sleep(0.01)
                completed_actions.append(action)

            return completed_actions

        # Detect breach
        breach_info = await detect_security_breach()
        assert breach_info["breach_detected"], "Security breach should be detected"

        # Execute containment
        containment_result = await containment_actions()
        assert "revoke_all_access_tokens" in containment_result, (
            "Tokens should be revoked"
        )

        # Register security breach recovery plan
        security_recovery_plan = RecoveryPlan(
            disaster_type=DisasterType.SECURITY_BREACH,
            recovery_steps=[
                "contain_breach",
                "assess_damage",
                "restore_clean_backup",
                "reset_all_credentials",
                "strengthen_security",
                "monitor_for_reoccurrence",
            ],
            rto=600.0,  # 10 minutes
            rpo=0.0,  # No data loss acceptable for security
        )
        recovery_orchestrator.register_recovery_plan(
            DisasterType.SECURITY_BREACH, security_recovery_plan
        )

        # Execute security recovery
        recovery_result = await recovery_orchestrator.execute_recovery(
            DisasterType.SECURITY_BREACH
        )

        # Verify security recovery
        assert recovery_result["status"] == "success"
        assert "reset_all_credentials" in recovery_result["completed_steps"]

    async def test_human_error_recovery(self, backup_system, recovery_orchestrator):
        """Test recovery from human error (accidental deletion, configuration changes)."""
        # Setup initial system state
        production_config = {
            "database": {"host": "prod-db", "port": 5432, "pool_size": 20},
            "cache": {"host": "prod-cache", "port": 6379, "ttl": 3600},
            "search": {"index": "production", "shards": 5, "replicas": 2},
        }

        production_data = {
            "collections": ["docs", "embeddings", "metadata"],
            "indexes": {"docs": 10000, "embeddings": 10000, "metadata": 5000},
        }

        # Create backups
        config_backup = await backup_system.create_backup(production_config)
        data_backup = await backup_system.create_backup(production_data)

        # Simulate human error scenarios
        async def accidental_deletion():
            """Simulate accidental data deletion."""
            # "Accidentally" delete collections
            damaged_data = {"collections": [], "indexes": {}}
            return damaged_data

        async def wrong_configuration():
            """Simulate incorrect configuration deployment."""
            # "Accidentally" deploy test config to production
            wrong_config = {
                "database": {"host": "test-db", "port": 5432, "pool_size": 1},
                "cache": {"host": "test-cache", "port": 6379, "ttl": 60},
                "search": {"index": "test", "shards": 1, "replicas": 0},
            }
            return wrong_config

        async def detect_human_error(
            current_config: dict, current_data: dict
        ) -> dict[str, Any]:
            """Detect human error by comparing with expected state."""
            errors = []

            # Check configuration
            if current_config.get("database", {}).get("host") != "prod-db":
                errors.append("incorrect_database_config")

            if current_config.get("search", {}).get("index") != "production":
                errors.append("incorrect_search_config")

            # Check data
            if not current_data.get("collections"):
                errors.append("missing_collections")

            actual__total_docs = sum(current_data.get("indexes", {}).values())
            if actual__total_docs == 0:
                errors.append("data_loss")

            return {
                "errors_detected": len(errors) > 0,
                "error_types": errors,
                "confidence": 1.0 if errors else 0.0,
            }

        # Simulate human errors
        damaged_data = await accidental_deletion()
        wrong_config = await wrong_configuration()

        # Detect errors
        data_error = await detect_human_error(production_config, damaged_data)
        config_error = await detect_human_error(wrong_config, production_data)

        assert data_error["errors_detected"], "Data deletion should be detected"
        assert config_error["errors_detected"], "Config error should be detected"
        assert "data_loss" in data_error["error_types"]
        assert "incorrect_database_config" in config_error["error_types"]

        # Register human error recovery plan
        human_error_recovery_plan = RecoveryPlan(
            disaster_type=DisasterType.HUMAN_ERROR,
            recovery_steps=[
                "detect_change_scope",
                "rollback_configuration",
                "restore_data_from_backup",
                "verify_system_state",
                "implement_safeguards",
            ],
            rto=120.0,  # 2 minutes
            rpo=900.0,  # 15 minutes data loss acceptable
        )
        recovery_orchestrator.register_recovery_plan(
            DisasterType.HUMAN_ERROR, human_error_recovery_plan
        )

        # Execute recovery
        recovery_result = await recovery_orchestrator.execute_recovery(
            DisasterType.HUMAN_ERROR
        )

        # Restore configuration and data
        restored_config = await backup_system.restore_from_backup(
            config_backup.location
        )
        restored_data = await backup_system.restore_from_backup(data_backup.location)

        # Verify recovery
        assert recovery_result["status"] == "success"
        assert restored_config == production_config, "Configuration should be restored"
        assert restored_data == production_data, "Data should be restored"

        # Verify no errors after recovery
        post_recovery_check = await detect_human_error(restored_config, restored_data)
        assert not post_recovery_check["errors_detected"], (
            "No errors should remain after recovery"
        )

    async def test_multi_region_failover(self, _recovery_orchestrator):
        """Test multi-region failover and recovery."""
        # Setup multi-region deployment
        regions = {
            "us-east-1": {
                "status": "healthy",
                "services": ["api", "db", "cache"],
                "traffic_weight": 60,
                "latency": 50,
            },
            "us-west-2": {
                "status": "healthy",
                "services": ["api", "db", "cache"],
                "traffic_weight": 30,
                "latency": 80,
            },
            "eu-west-1": {
                "status": "healthy",
                "services": ["api", "db", "cache"],
                "traffic_weight": 10,
                "latency": 120,
            },
        }

        async def region_health_check(region: str) -> dict[str, Any]:
            """Check health of a specific region."""
            region_info = regions[region]

            if region_info["status"] == "failed":
                msg = f"Region {region} is offline"
                raise TestError(msg)

            return {
                "region": region,
                "status": region_info["status"],
                "services": region_info["services"],
                "latency": region_info["latency"],
            }

        async def failover_to_region(failed_region: str, target_region: str):
            """Failover traffic from failed region to target region."""
            if regions[failed_region]["status"] != "failed":
                msg = f"Region {failed_region} is not marked as failed"
                raise TestError(msg)

            # Redistribute traffic
            failed_weight = regions[failed_region]["traffic_weight"]
            regions[target_region]["traffic_weight"] += failed_weight
            regions[failed_region]["traffic_weight"] = 0

            return {
                "failover_completed": True,
                "failed_region": failed_region,
                "target_region": target_region,
                "new_traffic_weight": regions[target_region]["traffic_weight"],
            }

        # Simulate region failure
        regions["us-east-1"]["status"] = "failed"

        # Detect region failure
        with pytest.raises(Exception, match="Region us-east-1 is offline"):
            await region_health_check("us-east-1")

        # Execute failover
        failover_result = await failover_to_region("us-east-1", "us-west-2")

        # Verify failover
        assert failover_result["failover_completed"]
        assert regions["us-west-2"]["traffic_weight"] == 90  # 30 + 60
        assert regions["us-east-1"]["traffic_weight"] == 0

        # Test remaining regions are healthy
        west_health = await region_health_check("us-west-2")
        eu_health = await region_health_check("eu-west-1")

        assert west_health["status"] == "healthy"
        assert eu_health["status"] == "healthy"

    async def test_backup_verification_and_restoration_testing(self, backup_system):
        """Test backup verification and restoration testing procedures."""
        # Create test data with various types
        test_data = {
            "structured_data": {
                "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "settings": {"theme": "dark", "notifications": True},
            },
            "binary_data": b"sample binary content",
            "large_text": "x" * 10000,  # Large text data
            "nested_structure": {
                "level1": {"level2": {"level3": ["item1", "item2", "item3"]}}
            },
        }

        # Create backup
        backup = await backup_system.create_backup(test_data)

        # Verify backup integrity
        integrity_ok = await backup_system.verify_backup_integrity(backup)
        assert integrity_ok, "Backup integrity check should pass"

        # Test restoration
        restored_data = await backup_system.restore_from_backup(backup.location)
        assert restored_data == test_data, "Restored data should match original"

        # Test restoration with corruption simulation
        # Corrupt the backup
        backup.data["structured_data"]["users"][0]["name"] = "CORRUPTED"
        backup.checksum = "invalid_checksum"

        # Verify corruption is detected
        integrity_ok = await backup_system.verify_backup_integrity(backup)
        assert not integrity_ok, "Backup corruption should be detected"

        # Test partial restoration capabilities
        async def restore_partial_data(
            backup_data: dict[str, Any], keys: list[str]
        ) -> dict[str, Any]:
            """Restore only specific keys from backup."""
            partial_data = {}
            for key in keys:
                if key in backup_data:
                    partial_data[key] = backup_data[key]
            return partial_data

        # Restore only structured data
        partial_restored = await restore_partial_data(test_data, ["structured_data"])
        assert "structured_data" in partial_restored
        assert "binary_data" not in partial_restored

    async def test_rto_rpo_compliance_monitoring(
        self, backup_system, _recovery_orchestrator
    ):
        """Test RTO/RPO compliance monitoring and alerting."""
        # Setup monitoring metrics
        recovery_metrics = {
            "last_backup_time": time.time() - 30,  # 30 seconds ago
            "backup_frequency": 60,  # Every 60 seconds
            "last_recovery_test": time.time() - 86400,  # 24 hours ago
            "recovery_test_frequency": 3600,  # Every hour
        }

        async def check_rpo_compliance(target_rpo: float) -> dict[str, Any]:
            """Check if current backup age meets RPO requirements."""
            backup_age = time.time() - recovery_metrics["last_backup_time"]

            return {
                "rpo_compliant": backup_age <= target_rpo,
                "current_backup_age": backup_age,
                "target_rpo": target_rpo,
                "next_backup_due": recovery_metrics["last_backup_time"]
                + recovery_metrics["backup_frequency"],
            }

        async def check_rto_compliance(target_rto: float) -> dict[str, Any]:
            """Check if recovery procedures can meet RTO requirements."""
            last_test_age = time.time() - recovery_metrics["last_recovery_test"]
            test_overdue = last_test_age > recovery_metrics["recovery_test_frequency"]

            return {
                "rto_testable": not test_overdue,
                "last_recovery_test_age": last_test_age,
                "target_rto": target_rto,
                "next_test_due": recovery_metrics["last_recovery_test"]
                + recovery_metrics["recovery_test_frequency"],
            }

        async def automated_recovery_test() -> dict[str, Any]:
            """Perform automated recovery test."""
            test_start = time.time()

            # Create test backup
            test_data = {"test": "recovery_test_data", "timestamp": test_start}
            test_backup = await backup_system.create_backup(test_data)

            # Test restoration
            try:
                restored_data = await backup_system.restore_from_backup(
                    test_backup.location
                )
                test_duration = time.time() - test_start

                # Update metrics
                recovery_metrics["last_recovery_test"] = test_start

                return {
                    "test_successful": restored_data == test_data,
                    "test_duration": test_duration,
                    "timestamp": test_start,
                }
            except Exception as e:
                return {
                    "test_successful": False,
                    "error": str(e),
                    "timestamp": test_start,
                }

        # Test RPO compliance
        rpo_check = await check_rpo_compliance(60.0)  # 1 minute RPO
        assert rpo_check["rpo_compliant"], "RPO should be compliant"

        # Test RTO testing compliance
        rto_check = await check_rto_compliance(300.0)  # 5 minute RTO
        assert not rto_check["rto_testable"], "Recovery test should be overdue"

        # Run automated recovery test
        test_result = await automated_recovery_test()
        assert test_result["test_successful"], "Recovery test should succeed"
        assert test_result["test_duration"] < 5.0, (
            "Recovery test should complete quickly"
        )

        # Verify RTO testing compliance after test
        rto_check_after = await check_rto_compliance(300.0)
        assert rto_check_after["rto_testable"], (
            "RTO testing should be compliant after test"
        )
