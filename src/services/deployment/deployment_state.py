"""Centralized deployment state management for canary deployments.

This module provides shared state management between ARQ tasks and routing
components using DragonflyDB for distributed coordination with optimal performance.
"""

import json
import logging
import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeploymentState:
    """Deployment state shared between components."""

    deployment_id: str
    alias: str
    old_collection: str
    new_collection: str
    current_stage: int
    current_percentage: float
    status: str  # pending, running, paused, completed, failed, rolled_back
    start_time: float
    last_updated: float
    metrics: dict[str, Any]
    error: str | None = None
    job_id: str | None = None


class DeploymentStateManager:
    """Manages shared deployment state in DragonflyDB with optimizations.

    Leverages DragonflyDB's performance advantages:
    - Multi-core utilization for better concurrency
    - Faster atomic operations and locking
    - More efficient memory usage
    - Native compression support
    """

    def __init__(self, redis_client):
        """Initialize state manager.

        Args:
            redis_client: DragonflyDB client (Redis-compatible)
        """
        self.redis = redis_client
        self._state_prefix = "deployment:state:"
        self._lock_prefix = "deployment:lock:"
        self._ttl = 86400 * 7  # 7 days retention

    async def get_state(self, deployment_id: str) -> DeploymentState | None:
        """Get deployment state from DragonflyDB.

        Args:
            deployment_id: Deployment ID

        Returns:
            DeploymentState if found, None otherwise
        """
        try:
            key = f"{self._state_prefix}{deployment_id}"
            data = await self.redis.get(key)

            if not data:
                return None

            state_dict = json.loads(data)
            # Remove metadata fields before creating dataclass
            state_dict.pop("version", None)
            return DeploymentState(**state_dict)

        except Exception as e:
            logger.error(f"Failed to get deployment state: {e}")
            return None

    async def update_state(
        self,
        deployment_id: str,
        updates: dict[str, Any],
        expected_version: int | None = None,
    ) -> bool:
        """Update deployment state atomically using DragonflyDB's optimized locking.

        Args:
            deployment_id: Deployment ID
            updates: Fields to update
            expected_version: Optional version for optimistic concurrency

        Returns:
            True if updated successfully
        """
        key = f"{self._state_prefix}{deployment_id}"
        lock_key = f"{self._lock_prefix}{deployment_id}"

        try:
            # DragonflyDB handles locks more efficiently than Redis
            # Using shorter lock timeout due to DragonflyDB's faster operations
            lock_acquired = await self.redis.set(lock_key, "1", nx=True, ex=3)

            if not lock_acquired:
                logger.warning(f"Could not acquire lock for {deployment_id}")
                return False

            try:
                # Use pipeline for atomic read-modify-write
                async with self.redis.pipeline() as pipe:
                    # Watch key for changes
                    await pipe.watch(key)

                    # Get current state
                    current_data = await self.redis.get(key)
                    if current_data:
                        current_state = json.loads(current_data)

                        # Check version if specified
                        if expected_version is not None:
                            current_version = current_state.get("version", 0)
                            if current_version != expected_version:
                                logger.warning(
                                    f"Version mismatch for {deployment_id}: "
                                    f"expected {expected_version}, got {current_version}"
                                )
                                await pipe.unwatch()
                                return False
                    else:
                        current_state = {}

                    # Apply updates
                    current_state.update(updates)
                    current_state["last_updated"] = time.time()
                    current_state["version"] = current_state.get("version", 0) + 1

                    # Execute transaction
                    pipe.multi()
                    pipe.setex(key, self._ttl, json.dumps(current_state))
                    await pipe.execute()

                # Publish state change event
                await self._publish_state_change(deployment_id, current_state, updates)

                return True

            finally:
                # Release lock
                await self.redis.delete(lock_key)

        except Exception as e:
            logger.error(f"Failed to update deployment state: {e}")
            return False

    async def create_state(self, state: DeploymentState) -> bool:
        """Create new deployment state.

        Args:
            state: Deployment state to create

        Returns:
            True if created successfully
        """
        key = f"{self._state_prefix}{state.deployment_id}"

        try:
            # Convert to dict and add metadata
            state_dict = asdict(state)
            state_dict["version"] = 1
            state_dict["last_updated"] = time.time()

            # Create only if not exists
            success = await self.redis.set(
                key, json.dumps(state_dict), nx=True, ex=self._ttl
            )

            if success:
                await self._publish_state_change(
                    state.deployment_id, state_dict, state_dict
                )

            return bool(success)

        except Exception as e:
            logger.error(f"Failed to create deployment state: {e}")
            return False

    async def delete_state(self, deployment_id: str) -> bool:
        """Delete deployment state.

        Args:
            deployment_id: Deployment ID

        Returns:
            True if deleted successfully
        """
        key = f"{self._state_prefix}{deployment_id}"

        try:
            result = await self.redis.delete(key)

            if result > 0:
                await self._publish_state_change(
                    deployment_id, None, {"status": "deleted"}
                )

            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete deployment state: {e}")
            return False

    async def list_deployments(
        self, status_filter: str | None = None
    ) -> list[DeploymentState]:
        """List all deployments with optional status filter.

        Optimized for DragonflyDB's efficient SCAN operation.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of deployment states
        """
        try:
            # DragonflyDB has more efficient SCAN implementation
            pattern = f"{self._state_prefix}*"
            keys = []
            cursor = 0

            # Increase count for better performance with DragonflyDB
            while True:
                cursor, batch = await self.redis.scan(cursor, match=pattern, count=1000)
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                return []

            # Use pipeline for batch fetching (DragonflyDB optimizes this well)
            states = []
            async with self.redis.pipeline() as pipe:
                for key in keys:
                    pipe.get(key)

                results = await pipe.execute()

                for data in results:
                    if data:
                        state_dict = json.loads(data)
                        # Remove metadata fields
                        state_dict.pop("version", None)
                        state = DeploymentState(**state_dict)

                        # Apply filter if specified
                        if status_filter and state.status != status_filter:
                            continue

                        states.append(state)

            # Sort by start time (newest first)
            states.sort(key=lambda s: s.start_time, reverse=True)

            return states

        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            return []

    async def get_active_deployments(self) -> list[DeploymentState]:
        """Get all active (running or paused) deployments.

        Returns:
            List of active deployment states
        """
        all_deployments = await self.list_deployments()
        return [d for d in all_deployments if d.status in ["running", "paused"]]

    async def _publish_state_change(
        self,
        deployment_id: str,
        full_state: dict[str, Any] | None,
        changes: dict[str, Any],
    ) -> None:
        """Publish state change event to Redis Stream.

        DragonflyDB has better Redis Streams performance.

        Args:
            deployment_id: Deployment ID
            full_state: Full state after change
            changes: What changed
        """
        try:
            event_data = {
                "type": "state_changed",
                "deployment_id": deployment_id,
                "timestamp": str(time.time()),
                "changes": json.dumps(changes),
            }

            if full_state:
                event_data["status"] = full_state.get("status", "unknown")
                event_data["current_percentage"] = str(
                    full_state.get("current_percentage", 0)
                )

            # DragonflyDB handles XADD more efficiently
            await self.redis.xadd("deployment:state:events", event_data, maxlen=10000)

        except Exception as e:
            logger.warning(f"Failed to publish state change event: {e}")

    async def acquire_deployment_lock(
        self, deployment_id: str, holder_id: str, ttl: int = 60
    ) -> bool:
        """Acquire exclusive lock on deployment for coordination.

        Uses DragonflyDB's optimized SET NX operations.

        Args:
            deployment_id: Deployment ID
            holder_id: ID of lock holder (e.g., worker ID)
            ttl: Lock TTL in seconds

        Returns:
            True if lock acquired
        """
        lock_key = f"{self._lock_prefix}exclusive:{deployment_id}"

        try:
            # DragonflyDB has faster atomic operations
            result = await self.redis.set(lock_key, holder_id, nx=True, ex=ttl)
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to acquire deployment lock: {e}")
            return False

    async def release_deployment_lock(self, deployment_id: str, holder_id: str) -> bool:
        """Release deployment lock if held by holder.

        Uses Lua script for atomic check-and-delete.

        Args:
            deployment_id: Deployment ID
            holder_id: ID of lock holder

        Returns:
            True if released
        """
        lock_key = f"{self._lock_prefix}exclusive:{deployment_id}"

        # Lua script for atomic check and delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            # DragonflyDB executes Lua scripts efficiently
            result = await self.redis.eval(lua_script, 1, lock_key, holder_id)
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to release deployment lock: {e}")
            return False

    async def extend_deployment_lock(
        self, deployment_id: str, holder_id: str, ttl: int = 60
    ) -> bool:
        """Extend deployment lock if held by holder.

        Uses Lua script for atomic check-and-extend.

        Args:
            deployment_id: Deployment ID
            holder_id: ID of lock holder
            ttl: New TTL in seconds

        Returns:
            True if extended
        """
        lock_key = f"{self._lock_prefix}exclusive:{deployment_id}"

        # Lua script for atomic check and expire
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        try:
            # DragonflyDB handles Lua efficiently
            result = await self.redis.eval(lua_script, 1, lock_key, holder_id, ttl)
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to extend deployment lock: {e}")
            return False

    async def get_deployment_metrics(
        self, deployment_id: str, time_window: int = 300
    ) -> dict[str, Any]:
        """Get real-time metrics for a deployment from event stream.

        Leverages DragonflyDB's fast XRANGE operations.

        Args:
            deployment_id: Deployment ID
            time_window: Time window in seconds (default 5 minutes)

        Returns:
            Aggregated metrics
        """
        try:
            # Calculate time range
            end_time = time.time()
            start_time = end_time - time_window

            # Convert to Redis Stream IDs
            start_id = f"{int(start_time * 1000)}-0"
            end_id = f"{int(end_time * 1000)}-0"

            # Read events from stream
            events = await self.redis.xrange(
                "search:events", start_id, end_id, count=1000
            )

            # Aggregate metrics
            metrics = {
                "total_searches": 0,
                "canary_searches": 0,
                "errors": 0,
                "avg_latency_ms": 0,
            }

            latencies = []

            for _, data in events:
                if data.get(b"deployment_id", b"").decode() == deployment_id:
                    metrics["total_searches"] += 1

                    if data.get(b"is_canary", b"").decode() == "true":
                        metrics["canary_searches"] += 1

                    if data.get(b"is_error", b"").decode() == "true":
                        metrics["errors"] += 1

                    if b"latency_ms" in data:
                        latencies.append(float(data[b"latency_ms"].decode()))

            if latencies:
                metrics["avg_latency_ms"] = sum(latencies) / len(latencies)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get deployment metrics: {e}")
            return {}
