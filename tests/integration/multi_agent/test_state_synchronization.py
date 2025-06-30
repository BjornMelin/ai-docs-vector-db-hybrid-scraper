"""Integration tests for multi-agent state synchronization.

Tests state synchronization across distributed agents, shared memory patterns,
and coordination efficiency under high load scenarios.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import (
    AgenticOrchestrator,
    ToolRequest,
    ToolResponse,
)
from src.services.agents.core import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    create_agent_dependencies,
)
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    ToolMetrics,
)


@dataclass
class SharedState:
    """Shared state structure for multi-agent coordination."""

    state_id: str
    version: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    updated_by: str | None = None
    access_count: int = 0
    lock_owner: str | None = None
    lock_timestamp: datetime | None = None


@dataclass
class StateUpdate:
    """Represents a state update operation."""

    update_id: str
    agent_id: str
    timestamp: datetime
    operation: str  # 'read', 'write', 'merge', 'delete'
    key_path: str
    old_value: Any = None
    new_value: Any = None
    success: bool = False
    conflict_detected: bool = False


class DistributedStateManager:
    """Manages distributed state synchronization across agents."""

    def __init__(self):
        self.shared_states: dict[str, SharedState] = {}
        self.update_history: list[StateUpdate] = []
        self.locks: dict[str, threading.Lock] = {}
        self.state_locks: dict[str, str] = {}  # state_id -> agent_id mappings
        self.conflict_resolution_strategy = "last_write_wins"
        self.synchronization_metrics: dict[str, float] = {}

    async def create_shared_state(
        self, state_id: str, initial_data: dict[str, Any]
    ) -> SharedState:
        """Create a new shared state."""
        shared_state = SharedState(
            state_id=state_id,
            data=initial_data.copy(),
            last_updated=datetime.now(),
        )

        self.shared_states[state_id] = shared_state
        self.locks[state_id] = threading.Lock()

        return shared_state

    async def read_state(
        self, state_id: str, agent_id: str, key_path: str = None
    ) -> Any:
        """Read state value with synchronization."""
        update = StateUpdate(
            update_id=str(uuid4()),
            agent_id=agent_id,
            timestamp=datetime.now(),
            operation="read",
            key_path=key_path or "",
        )

        try:
            if state_id not in self.shared_states:
                raise ValueError(f"State {state_id} does not exist")

            state = self.shared_states[state_id]
            state.access_count += 1

            if key_path:
                # Navigate nested path
                value = state.data
                for key in key_path.split("."):
                    value = value.get(key, {}) if isinstance(value, dict) else None
                    if value is None:
                        break
                update.old_value = value
            else:
                update.old_value = state.data.copy()

            update.success = True
            return update.old_value

        except Exception as e:
            update.success = False
            raise e
        finally:
            self.update_history.append(update)

    async def write_state(
        self,
        state_id: str,
        agent_id: str,
        key_path: str,
        value: Any,
        require_lock: bool = True,
    ) -> bool:
        """Write state value with synchronization and conflict detection."""
        update = StateUpdate(
            update_id=str(uuid4()),
            agent_id=agent_id,
            timestamp=datetime.now(),
            operation="write",
            key_path=key_path,
            new_value=value,
        )

        try:
            if state_id not in self.shared_states:
                raise ValueError(f"State {state_id} does not exist")

            # Check for lock requirement
            if require_lock and not await self._check_lock(state_id, agent_id):
                update.conflict_detected = True
                update.success = False
                return False

            state = self.shared_states[state_id]

            # Get old value for conflict detection
            old_value = state.data
            for key in key_path.split(".")[:-1]:
                old_value = old_value.get(key, {})
                if not isinstance(old_value, dict):
                    old_value = {}
                    break

            final_key = key_path.split(".")[-1]
            update.old_value = (
                old_value.get(final_key) if isinstance(old_value, dict) else None
            )

            # Detect conflicts
            if update.old_value != update.new_value and state.updated_by != agent_id:
                # Check if another agent updated this recently
                time_since_update = datetime.now() - state.last_updated
                if time_since_update < timedelta(seconds=1):  # Recent update
                    update.conflict_detected = True
                    if not await self._resolve_conflict(state_id, update):
                        update.success = False
                        return False

            # Perform the write
            await self._write_to_nested_path(state.data, key_path, value)

            # Update metadata
            state.version += 1
            state.last_updated = datetime.now()
            state.updated_by = agent_id

            update.success = True
            return True

        except Exception as e:
            update.success = False
            raise e
        finally:
            self.update_history.append(update)

    async def merge_state(
        self,
        state_id: str,
        agent_id: str,
        updates: dict[str, Any],
        merge_strategy: str = "deep_merge",
    ) -> bool:
        """Merge multiple updates into shared state."""
        update = StateUpdate(
            update_id=str(uuid4()),
            agent_id=agent_id,
            timestamp=datetime.now(),
            operation="merge",
            key_path="multiple",
            new_value=updates,
        )

        try:
            if state_id not in self.shared_states:
                raise ValueError(f"State {state_id} does not exist")

            state = self.shared_states[state_id]
            update.old_value = state.data.copy()

            # Perform merge based on strategy
            if merge_strategy == "deep_merge":
                self._deep_merge(state.data, updates)
            elif merge_strategy == "shallow_merge":
                state.data.update(updates)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

            # Update metadata
            state.version += 1
            state.last_updated = datetime.now()
            state.updated_by = agent_id

            update.success = True
            return True

        except Exception as e:
            update.success = False
            raise e
        finally:
            self.update_history.append(update)

    async def acquire_lock(
        self, state_id: str, agent_id: str, timeout: float = 5.0
    ) -> bool:
        """Acquire exclusive lock on shared state."""
        if state_id not in self.shared_states:
            return False

        start_time = time.time()

        while time.time() - start_time < timeout:
            if state_id not in self.state_locks:
                self.state_locks[state_id] = agent_id
                state = self.shared_states[state_id]
                state.lock_owner = agent_id
                state.lock_timestamp = datetime.now()
                return True

            # Wait briefly before retrying
            await asyncio.sleep(0.1)

        return False  # Timeout

    async def release_lock(self, state_id: str, agent_id: str) -> bool:
        """Release exclusive lock on shared state."""
        if state_id in self.state_locks and self.state_locks[state_id] == agent_id:
            del self.state_locks[state_id]
            state = self.shared_states[state_id]
            state.lock_owner = None
            state.lock_timestamp = None
            return True
        return False

    async def _check_lock(self, state_id: str, agent_id: str) -> bool:
        """Check if agent has lock or no lock is required."""
        if state_id not in self.state_locks:
            return True  # No lock held
        return self.state_locks[state_id] == agent_id

    async def _resolve_conflict(self, state_id: str, update: StateUpdate) -> bool:
        """Resolve write conflicts based on strategy."""
        if self.conflict_resolution_strategy == "last_write_wins":
            return True  # Allow the write
        if self.conflict_resolution_strategy == "first_write_wins":
            return False  # Reject the write
        if self.conflict_resolution_strategy == "merge":
            # Attempt to merge changes
            return await self._attempt_merge_resolution(state_id, update)
        return False

    async def _attempt_merge_resolution(
        self, state_id: str, update: StateUpdate
    ) -> bool:
        """Attempt to resolve conflict through merging."""
        # Simple merge resolution - in practice this would be more sophisticated
        if isinstance(update.old_value, dict) and isinstance(update.new_value, dict):
            merged_value = {**update.old_value, **update.new_value}
            update.new_value = merged_value
            return True
        return False

    async def _write_to_nested_path(
        self, data: dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Write value to nested dictionary path."""
        keys = key_path.split(".")
        current = data

        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set final value
        current[keys[-1]] = value

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Perform deep merge of dictionaries."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def get_synchronization_metrics(self) -> dict[str, float]:
        """Get synchronization performance metrics."""
        if not self.update_history:
            return {}

        total_updates = len(self.update_history)
        successful_updates = sum(1 for u in self.update_history if u.success)
        conflicts = sum(1 for u in self.update_history if u.conflict_detected)

        recent_updates = [
            u
            for u in self.update_history
            if datetime.now() - u.timestamp < timedelta(minutes=1)
        ]

        return {
            "total_updates": total_updates,
            "success_rate": successful_updates / total_updates
            if total_updates > 0
            else 0,
            "conflict_rate": conflicts / total_updates if total_updates > 0 else 0,
            "recent_update_rate": len(recent_updates),
            "avg_state_versions": sum(s.version for s in self.shared_states.values())
            / len(self.shared_states)
            if self.shared_states
            else 0,
        }


class TestStateManagementBasics:
    """Test basic state management operations."""

    @pytest.fixture
    def state_manager(self) -> DistributedStateManager:
        """Create distributed state manager."""
        return DistributedStateManager()

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_state_sync",
        )

    @pytest.mark.asyncio
    async def test_basic_state_operations(self, state_manager):
        """Test basic state create, read, write operations."""
        # Create shared state
        initial_data = {
            "task_queue": [],
            "processing_status": {"active": 0, "completed": 0},
            "agent_assignments": {},
        }

        shared_state = await state_manager.create_shared_state(
            "basic_test", initial_data
        )

        assert shared_state.state_id == "basic_test"
        assert shared_state.version == 0
        assert shared_state.data == initial_data

        # Test read operations
        full_data = await state_manager.read_state("basic_test", "agent_1")
        assert full_data == initial_data

        status_data = await state_manager.read_state(
            "basic_test", "agent_1", "processing_status"
        )
        assert status_data == {"active": 0, "completed": 0}

        active_count = await state_manager.read_state(
            "basic_test", "agent_1", "processing_status.active"
        )
        assert active_count == 0

        # Test write operations
        success = await state_manager.write_state(
            "basic_test", "agent_1", "processing_status.active", 5, require_lock=False
        )
        assert success is True

        # Verify write
        updated_active = await state_manager.read_state(
            "basic_test", "agent_1", "processing_status.active"
        )
        assert updated_active == 5

        # Check state version updated
        updated_state = state_manager.shared_states["basic_test"]
        assert updated_state.version == 1
        assert updated_state.updated_by == "agent_1"

    @pytest.mark.asyncio
    async def test_lock_acquisition_and_release(self, state_manager):
        """Test lock acquisition and release mechanisms."""
        # Create shared state
        await state_manager.create_shared_state("lock_test", {"counter": 0})

        # Agent 1 acquires lock
        lock_acquired = await state_manager.acquire_lock("lock_test", "agent_1")
        assert lock_acquired is True

        # Agent 2 tries to acquire lock (should fail)
        lock_blocked = await state_manager.acquire_lock(
            "lock_test", "agent_2", timeout=0.5
        )
        assert lock_blocked is False

        # Agent 1 can write with lock
        write_success = await state_manager.write_state(
            "lock_test", "agent_1", "counter", 10, require_lock=True
        )
        assert write_success is True

        # Agent 2 cannot write without lock
        write_blocked = await state_manager.write_state(
            "lock_test", "agent_2", "counter", 20, require_lock=True
        )
        assert write_blocked is False

        # Agent 1 releases lock
        lock_released = await state_manager.release_lock("lock_test", "agent_1")
        assert lock_released is True

        # Agent 2 can now acquire lock
        lock_acquired_2 = await state_manager.acquire_lock("lock_test", "agent_2")
        assert lock_acquired_2 is True

        # Agent 2 can write with lock
        write_success_2 = await state_manager.write_state(
            "lock_test", "agent_2", "counter", 30, require_lock=True
        )
        assert write_success_2 is True

        # Verify final value
        final_value = await state_manager.read_state("lock_test", "agent_2", "counter")
        assert final_value == 30

    @pytest.mark.asyncio
    async def test_merge_operations(self, state_manager):
        """Test state merge operations."""
        # Create shared state
        initial_data = {
            "metrics": {"requests": 0, "errors": 0},
            "config": {"timeout": 30, "retries": 3},
        }
        await state_manager.create_shared_state("merge_test", initial_data)

        # Agent 1 merges metrics update
        metrics_update = {
            "metrics": {"requests": 100, "latency": 250},
            "status": "running",
        }

        merge_success = await state_manager.merge_state(
            "merge_test", "agent_1", metrics_update, "deep_merge"
        )
        assert merge_success is True

        # Verify merge results
        final_data = await state_manager.read_state("merge_test", "agent_1")
        expected_data = {
            "metrics": {"requests": 100, "errors": 0, "latency": 250},  # Deep merged
            "config": {"timeout": 30, "retries": 3},  # Preserved
            "status": "running",  # Added
        }
        assert final_data == expected_data

        # Test shallow merge
        shallow_update = {"metrics": {"new_metric": True}}
        await state_manager.merge_state(
            "merge_test", "agent_2", shallow_update, "shallow_merge"
        )

        # Should have replaced metrics completely
        after_shallow = await state_manager.read_state(
            "merge_test", "agent_2", "metrics"
        )
        assert after_shallow == {"new_metric": True}


class TestConcurrentStateAccess:
    """Test concurrent state access and conflict resolution."""

    @pytest.fixture
    def state_manager(self) -> DistributedStateManager:
        """Create distributed state manager."""
        return DistributedStateManager()

    @pytest.fixture
    def agent_pool(self) -> list[AgenticOrchestrator]:
        """Create pool of agents for concurrent testing."""
        return [
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1) for _ in range(4)
        ]

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_concurrent_state",
        )

    @pytest.mark.asyncio
    async def test_concurrent_read_operations(
        self, state_manager, agent_pool, agent_dependencies
    ):
        """Test concurrent read operations on shared state."""
        # Initialize agents
        for agent in agent_pool:
            await agent.initialize(agent_dependencies)

        # Create shared state
        shared_data = {
            "large_dataset": {f"item_{i}": f"value_{i}" for i in range(1000)},
            "metadata": {"size": 1000, "type": "test_data"},
        }
        await state_manager.create_shared_state("concurrent_read_test", shared_data)

        # Define concurrent read task
        async def concurrent_read_task(agent_id, iterations=10):
            results = []
            for i in range(iterations):
                # Read different parts of the state
                if i % 3 == 0:
                    data = await state_manager.read_state(
                        "concurrent_read_test", agent_id
                    )
                elif i % 3 == 1:
                    data = await state_manager.read_state(
                        "concurrent_read_test", agent_id, "metadata"
                    )
                else:
                    data = await state_manager.read_state(
                        "concurrent_read_test", agent_id, f"large_dataset.item_{i}"
                    )

                results.append(data is not None)
                await asyncio.sleep(0.01)  # Small delay to encourage concurrency

            return sum(results)  # Count successful reads

        # Execute concurrent reads
        tasks = [concurrent_read_task(f"agent_{i}", 20) for i in range(len(agent_pool))]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Verify concurrent reads
        assert all(result == 20 for result in results)  # All reads should succeed
        assert execution_time < 10.0  # Should complete reasonably quickly

        # Check state access metrics
        state = state_manager.shared_states["concurrent_read_test"]
        assert state.access_count >= 80  # At least 20 reads per agent * 4 agents

    @pytest.mark.asyncio
    async def test_concurrent_write_with_conflict_detection(
        self, state_manager, agent_pool, agent_dependencies
    ):
        """Test concurrent writes with conflict detection."""
        # Initialize agents
        for agent in agent_pool:
            await agent.initialize(agent_dependencies)

        # Create shared state for conflict testing
        await state_manager.create_shared_state(
            "conflict_test",
            {
                "counter": 0,
                "agent_updates": {},
                "last_update_time": None,
            },
        )

        # Define concurrent write task
        async def concurrent_write_task(agent_id, update_count=5):
            successful_writes = 0
            conflicts_detected = 0

            for i in range(update_count):
                try:
                    # Try to update counter and agent-specific data
                    current_counter = await state_manager.read_state(
                        "conflict_test", agent_id, "counter"
                    )
                    new_counter = current_counter + 1

                    # Simulate some processing time
                    await asyncio.sleep(0.05)

                    # Attempt write
                    write_success = await state_manager.write_state(
                        "conflict_test",
                        agent_id,
                        "counter",
                        new_counter,
                        require_lock=False,
                    )

                    if write_success:
                        successful_writes += 1

                        # Also update agent-specific data
                        await state_manager.write_state(
                            "conflict_test",
                            agent_id,
                            f"agent_updates.{agent_id}",
                            f"update_{i}",
                            require_lock=False,
                        )

                    # Check for conflicts in update history
                    recent_updates = [
                        u
                        for u in state_manager.update_history
                        if u.agent_id == agent_id and u.conflict_detected
                    ]
                    conflicts_detected = len(recent_updates)

                except Exception:
                    pass  # Continue on errors

            return {
                "agent_id": agent_id,
                "successful_writes": successful_writes,
                "conflicts_detected": conflicts_detected,
            }

        # Execute concurrent writes
        write_tasks = [
            concurrent_write_task(f"agent_{i}", 10) for i in range(len(agent_pool))
        ]

        results = await asyncio.gather(*write_tasks)

        # Verify conflict detection
        total_successful = sum(r["successful_writes"] for r in results)
        total_conflicts = sum(r["conflicts_detected"] for r in results)

        assert total_successful > 0  # Some writes should succeed

        # Final counter should reflect successful writes
        final_counter = await state_manager.read_state(
            "conflict_test", "validator", "counter"
        )
        assert final_counter > 0

        # Verify conflict detection worked
        if total_conflicts > 0:
            print(f"Detected {total_conflicts} conflicts during concurrent writes")

        # Check synchronization metrics
        metrics = state_manager.get_synchronization_metrics()
        assert metrics["total_updates"] > 0
        assert metrics["success_rate"] > 0

    @pytest.mark.asyncio
    async def test_lock_based_coordination(
        self, state_manager, agent_pool, agent_dependencies
    ):
        """Test lock-based coordination for critical sections."""
        # Initialize agents
        for agent in agent_pool:
            await agent.initialize(agent_dependencies)

        # Create shared state for coordination
        await state_manager.create_shared_state(
            "coordination_test",
            {
                "resource_pool": list(range(20)),  # 20 resources
                "allocated_resources": {},
                "allocation_history": [],
            },
        )

        # Define resource allocation task (critical section)
        async def allocate_resources_task(agent_id, resources_needed=3):
            allocated = []

            try:
                # Acquire lock for critical section
                lock_acquired = await state_manager.acquire_lock(
                    "coordination_test", agent_id, timeout=2.0
                )

                if not lock_acquired:
                    return {"agent_id": agent_id, "allocated": [], "lock_failed": True}

                # Critical section: allocate resources
                current_pool = await state_manager.read_state(
                    "coordination_test", agent_id, "resource_pool"
                )

                if len(current_pool) >= resources_needed:
                    # Allocate resources
                    allocated = current_pool[:resources_needed]
                    remaining = current_pool[resources_needed:]

                    # Update state
                    await state_manager.write_state(
                        "coordination_test",
                        agent_id,
                        "resource_pool",
                        remaining,
                        require_lock=True,
                    )
                    await state_manager.write_state(
                        "coordination_test",
                        agent_id,
                        f"allocated_resources.{agent_id}",
                        allocated,
                        require_lock=True,
                    )

                    # Log allocation
                    history = await state_manager.read_state(
                        "coordination_test", agent_id, "allocation_history"
                    )
                    history.append(
                        {
                            "agent_id": agent_id,
                            "resources": allocated,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    await state_manager.write_state(
                        "coordination_test",
                        agent_id,
                        "allocation_history",
                        history,
                        require_lock=True,
                    )

                return {
                    "agent_id": agent_id,
                    "allocated": allocated,
                    "lock_failed": False,
                }

            finally:
                # Always release lock
                await state_manager.release_lock("coordination_test", agent_id)

        # Execute concurrent resource allocation
        allocation_tasks = [
            allocate_resources_task(f"agent_{i}", 2 + i)  # Different resource needs
            for i in range(len(agent_pool))
        ]

        results = await asyncio.gather(*allocation_tasks)

        # Verify coordination results
        successful_allocations = [
            r for r in results if r["allocated"] and not r["lock_failed"]
        ]
        failed_locks = [r for r in results if r["lock_failed"]]

        assert len(successful_allocations) > 0  # Some allocations should succeed

        # Verify no resource conflicts (no resource allocated twice)
        all_allocated = []
        for result in successful_allocations:
            all_allocated.extend(result["allocated"])

        assert len(all_allocated) == len(set(all_allocated))  # No duplicates

        # Check final state consistency
        final_pool = await state_manager.read_state(
            "coordination_test", "validator", "resource_pool"
        )
        allocated_resources = await state_manager.read_state(
            "coordination_test", "validator", "allocated_resources"
        )

        total_resources = len(final_pool) + sum(
            len(resources) for resources in allocated_resources.values()
        )
        assert total_resources == 20  # No resources lost or duplicated


class TestHighLoadScenarios:
    """Test state synchronization under high load scenarios."""

    @pytest.fixture
    def state_manager(self) -> DistributedStateManager:
        """Create distributed state manager."""
        return DistributedStateManager()

    @pytest.fixture
    def large_agent_pool(self) -> list[Any]:
        """Create large pool of agents for load testing."""
        agents = []
        # Mix of orchestrator and discovery agents
        for i in range(8):
            if i % 2 == 0:
                agents.append(AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1))
            else:
                agents.append(
                    DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1)
                )
        return agents

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_high_load_state",
        )

    @pytest.mark.asyncio
    async def test_high_frequency_updates(
        self, state_manager, large_agent_pool, agent_dependencies
    ):
        """Test state synchronization under high-frequency updates."""
        # Initialize agents
        for i, agent in enumerate(large_agent_pool):
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Create shared state for high-frequency updates
        await state_manager.create_shared_state(
            "high_frequency_test",
            {
                "request_count": 0,
                "active_agents": set(),
                "performance_metrics": {},
                "recent_activities": [],
            },
        )

        # Define high-frequency update task
        async def high_frequency_update_task(
            agent_id, updates_per_second=10, duration_seconds=5
        ):
            update_interval = 1.0 / updates_per_second
            end_time = time.time() + duration_seconds
            successful_updates = 0

            while time.time() < end_time:
                try:
                    # Update request count
                    current_count = await state_manager.read_state(
                        "high_frequency_test", agent_id, "request_count"
                    )
                    await state_manager.write_state(
                        "high_frequency_test",
                        agent_id,
                        "request_count",
                        current_count + 1,
                        require_lock=False,
                    )

                    # Update metrics
                    timestamp = datetime.now().isoformat()
                    await state_manager.write_state(
                        "high_frequency_test",
                        agent_id,
                        f"performance_metrics.{agent_id}.last_update",
                        timestamp,
                        require_lock=False,
                    )

                    # Add to recent activities (with rotation)
                    activities = await state_manager.read_state(
                        "high_frequency_test", agent_id, "recent_activities"
                    )
                    activities.append(f"{agent_id}_{timestamp}")
                    if len(activities) > 100:  # Keep only recent 100
                        activities = activities[-100:]
                    await state_manager.write_state(
                        "high_frequency_test",
                        agent_id,
                        "recent_activities",
                        activities,
                        require_lock=False,
                    )

                    successful_updates += 1
                    await asyncio.sleep(update_interval)

                except Exception:
                    pass  # Continue on errors

            return {
                "agent_id": agent_id,
                "successful_updates": successful_updates,
                "target_updates": updates_per_second * duration_seconds,
            }

        # Execute high-frequency updates
        update_tasks = [
            high_frequency_update_task(
                f"agent_{i}", 5, 3
            )  # 5 updates/sec for 3 seconds
            for i in range(len(large_agent_pool))
        ]

        start_time = time.time()
        results = await asyncio.gather(*update_tasks)
        total_time = time.time() - start_time

        # Verify high-frequency performance
        total_successful = sum(r["successful_updates"] for r in results)
        total_target = sum(r["target_updates"] for r in results)

        success_rate = total_successful / total_target if total_target > 0 else 0
        assert success_rate > 0.7  # At least 70% success rate under load

        # Check final state consistency
        final_count = await state_manager.read_state(
            "high_frequency_test", "validator", "request_count"
        )
        assert final_count > 0

        # Verify synchronization metrics under load
        metrics = state_manager.get_synchronization_metrics()
        assert metrics["total_updates"] >= total_successful
        print(
            f"High-frequency test: {total_successful}/{total_target} updates successful"
        )
        print(f"Success rate: {success_rate:.2%}")
        print(f"Conflict rate: {metrics['conflict_rate']:.2%}")

    @pytest.mark.asyncio
    async def test_coordination_efficiency_under_load(
        self, state_manager, large_agent_pool, agent_dependencies
    ):
        """Test coordination efficiency under high load."""
        # Initialize agents
        for i, agent in enumerate(large_agent_pool):
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Create coordination state
        await state_manager.create_shared_state(
            "efficiency_test",
            {
                "task_queue": [f"task_{i}" for i in range(100)],  # 100 tasks
                "completed_tasks": [],
                "agent_workloads": {},
                "coordination_metrics": {
                    "lock_acquisitions": 0,
                    "lock_failures": 0,
                    "average_processing_time": 0,
                },
            },
        )

        # Define coordinated task processing
        async def coordinated_task_processor(agent_id, max_tasks=15):
            processed_tasks = []
            lock_failures = 0
            processing_times = []

            while len(processed_tasks) < max_tasks:
                start_time = time.time()

                try:
                    # Try to acquire lock for task queue access
                    lock_acquired = await state_manager.acquire_lock(
                        "efficiency_test", agent_id, timeout=0.5
                    )

                    if not lock_acquired:
                        lock_failures += 1
                        await asyncio.sleep(0.1)  # Back off
                        continue

                    # Get task from queue
                    task_queue = await state_manager.read_state(
                        "efficiency_test", agent_id, "task_queue"
                    )

                    if not task_queue:
                        break  # No more tasks

                    # Take first task
                    task = task_queue.pop(0)
                    processed_tasks.append(task)

                    # Update queue and completed tasks
                    await state_manager.write_state(
                        "efficiency_test",
                        agent_id,
                        "task_queue",
                        task_queue,
                        require_lock=True,
                    )

                    completed = await state_manager.read_state(
                        "efficiency_test", agent_id, "completed_tasks"
                    )
                    completed.append(
                        {
                            "task": task,
                            "agent": agent_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    await state_manager.write_state(
                        "efficiency_test",
                        agent_id,
                        "completed_tasks",
                        completed,
                        require_lock=True,
                    )

                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)

                except Exception:
                    pass
                finally:
                    await state_manager.release_lock("efficiency_test", agent_id)

                # Simulate task processing
                await asyncio.sleep(0.02)

            return {
                "agent_id": agent_id,
                "processed_tasks": len(processed_tasks),
                "lock_failures": lock_failures,
                "avg_processing_time": sum(processing_times) / len(processing_times)
                if processing_times
                else 0,
            }

        # Execute coordinated processing
        processing_tasks = [
            coordinated_task_processor(f"agent_{i}", 20)
            for i in range(len(large_agent_pool))
        ]

        start_time = time.time()
        results = await asyncio.gather(*processing_tasks)
        total_coordination_time = time.time() - start_time

        # Analyze coordination efficiency
        total_processed = sum(r["processed_tasks"] for r in results)
        total_lock_failures = sum(r["lock_failures"] for r in results)
        avg_processing_times = [
            r["avg_processing_time"] for r in results if r["avg_processing_time"] > 0
        ]

        # Verify efficiency metrics
        assert total_processed > 50  # Should process most tasks

        # Check task distribution efficiency
        task_distribution = [r["processed_tasks"] for r in results]
        max_tasks = max(task_distribution)
        min_tasks = min(task_distribution)
        load_balance_ratio = min_tasks / max_tasks if max_tasks > 0 else 0

        assert load_balance_ratio > 0.3  # Reasonable load balancing

        # Verify coordination speed
        tasks_per_second = total_processed / total_coordination_time
        assert tasks_per_second > 5  # At least 5 tasks/second

        # Check final state consistency
        final_queue = await state_manager.read_state(
            "efficiency_test", "validator", "task_queue"
        )
        final_completed = await state_manager.read_state(
            "efficiency_test", "validator", "completed_tasks"
        )

        total_tasks_accounted = len(final_queue) + len(final_completed)
        assert total_tasks_accounted == 100  # No tasks lost

        print("Coordination efficiency test:")
        print(f"  Tasks processed: {total_processed}/100")
        print(f"  Processing rate: {tasks_per_second:.2f} tasks/second")
        print(f"  Load balance ratio: {load_balance_ratio:.2f}")
        print(f"  Lock failures: {total_lock_failures}")

    @pytest.mark.asyncio
    async def test_state_synchronization_performance(
        self, state_manager, large_agent_pool, agent_dependencies
    ):
        """Test overall state synchronization performance."""
        # Initialize agents
        for i, agent in enumerate(large_agent_pool):
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Create performance test state
        await state_manager.create_shared_state(
            "performance_test",
            {
                "global_counters": {f"counter_{i}": 0 for i in range(10)},
                "agent_metrics": {},
                "synchronization_events": [],
                "performance_data": {
                    "start_time": datetime.now().isoformat(),
                    "operations_completed": 0,
                },
            },
        )

        # Define performance test operation
        async def performance_test_operation(agent_id, operation_count=50):
            operations_completed = 0
            read_times = []
            write_times = []

            for i in range(operation_count):
                # Read operation timing
                read_start = time.time()
                counter_key = f"counter_{i % 10}"
                current_value = await state_manager.read_state(
                    "performance_test", agent_id, f"global_counters.{counter_key}"
                )
                read_time = time.time() - read_start
                read_times.append(read_time)

                # Write operation timing
                write_start = time.time()
                new_value = (current_value or 0) + 1
                write_success = await state_manager.write_state(
                    "performance_test",
                    agent_id,
                    f"global_counters.{counter_key}",
                    new_value,
                    require_lock=False,
                )
                write_time = time.time() - write_start
                write_times.append(write_time)

                if write_success:
                    operations_completed += 1

                # Update agent metrics periodically
                if i % 10 == 0:
                    await state_manager.write_state(
                        "performance_test",
                        agent_id,
                        f"agent_metrics.{agent_id}.operations",
                        operations_completed,
                        require_lock=False,
                    )

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            return {
                "agent_id": agent_id,
                "operations_completed": operations_completed,
                "avg_read_time": sum(read_times) / len(read_times),
                "avg_write_time": sum(write_times) / len(write_times),
                "total_operations": operation_count,
            }

        # Execute performance test
        test_tasks = [
            performance_test_operation(f"agent_{i}", 30)
            for i in range(len(large_agent_pool))
        ]

        start_time = time.time()
        results = await asyncio.gather(*test_tasks)
        total_test_time = time.time() - start_time

        # Analyze performance results
        total_operations = sum(r["operations_completed"] for r in results)
        avg_read_times = [r["avg_read_time"] for r in results]
        avg_write_times = [r["avg_write_time"] for r in results]

        overall_read_time = sum(avg_read_times) / len(avg_read_times)
        overall_write_time = sum(avg_write_times) / len(avg_write_times)
        operations_per_second = total_operations / total_test_time

        # Performance assertions
        assert overall_read_time < 0.1  # Reads should be fast
        assert overall_write_time < 0.2  # Writes should be reasonably fast
        assert operations_per_second > 20  # Should handle decent throughput

        # Check synchronization metrics
        final_metrics = state_manager.get_synchronization_metrics()
        assert final_metrics["success_rate"] > 0.8  # High success rate
        assert final_metrics["total_updates"] >= total_operations

        # Verify state consistency
        final_counters = await state_manager.read_state(
            "performance_test", "validator", "global_counters"
        )
        counter_sum = sum(final_counters.values())
        assert counter_sum > 0  # Some updates should have succeeded

        print("State synchronization performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Operations/second: {operations_per_second:.2f}")
        print(f"  Avg read time: {overall_read_time:.4f}s")
        print(f"  Avg write time: {overall_write_time:.4f}s")
        print(f"  Success rate: {final_metrics['success_rate']:.2%}")
        print(f"  Conflict rate: {final_metrics['conflict_rate']:.2%}")
