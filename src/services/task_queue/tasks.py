"""ARQ task definitions for background processing."""

import asyncio
import contextlib
import logging
import time
from typing import Any

from arq import func

from ...config import get_config
from ...infrastructure.client_manager import ClientManager
from ..core.qdrant_alias_manager import QdrantAliasManager

logger = logging.getLogger(__name__)


async def delete_collection(
    ctx: dict[str, Any],
    collection_name: str,
    grace_period_minutes: int = 60,
) -> dict[str, Any]:
    """Delete a Qdrant collection after grace period.

    This task replaces the asyncio.create_task pattern in QdrantAliasManager.

    Args:
        ctx: ARQ context (provided by worker)
        collection_name: Name of collection to delete
        grace_period_minutes: Minutes to wait before deletion

    Returns:
        Task result with status
    """
    start_time = time.time()
    logger.info(
        f"Starting delayed deletion of {collection_name} "
        f"(grace period: {grace_period_minutes} minutes)"
    )

    try:
        # Wait for grace period
        await asyncio.sleep(grace_period_minutes * 60)

        # Initialize required services
        config = get_config()
        client_manager = ClientManager(config)
        await client_manager.initialize()

        try:
            qdrant_client = await client_manager.get_qdrant_client()
            alias_manager = QdrantAliasManager(config, qdrant_client)

            # Double-check no aliases point to this collection
            aliases = await alias_manager.list_aliases()
            if collection_name in aliases.values():
                logger.warning(
                    f"Collection {collection_name} still has aliases, skipping deletion"
                )
                return {
                    "status": "skipped",
                    "reason": "collection_has_aliases",
                    "collection": collection_name,
                    "duration": time.time() - start_time,
                }

            # Delete the collection
            await qdrant_client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection {collection_name}")

            return {
                "status": "success",
                "collection": collection_name,
                "duration": time.time() - start_time,
            }

        finally:
            await client_manager.cleanup()

    except asyncio.CancelledError:
        logger.info(f"Deletion of {collection_name} was cancelled")
        return {
            "status": "cancelled",
            "collection": collection_name,
            "duration": time.time() - start_time,
        }
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}")
        return {
            "status": "failed",
            "collection": collection_name,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def persist_cache(
    ctx: dict[str, Any],
    key: str,
    value: Any,
    persist_func_module: str,
    persist_func_name: str,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Persist cache data to storage after delay.

    This task replaces the asyncio.create_task pattern in CachePatterns.

    Args:
        ctx: ARQ context (provided by worker)
        key: Cache key
        value: Value to persist
        persist_func_module: Module containing persist function
        persist_func_name: Name of persist function
        delay: Delay before persisting in seconds

    Returns:
        Task result with status
    """
    start_time = time.time()
    logger.info(f"Starting delayed persistence for {key} (delay: {delay}s)")

    try:
        # Wait for delay
        await asyncio.sleep(delay)

        # Import and call the persist function dynamically
        # This allows different persist functions to be used
        import importlib

        module = importlib.import_module(persist_func_module)
        persist_func = getattr(module, persist_func_name)

        # Call the persist function
        if asyncio.iscoroutinefunction(persist_func):
            await persist_func(key, value)
        else:
            persist_func(key, value)

        logger.info(f"Successfully persisted data for {key}")

        return {
            "status": "success",
            "key": key,
            "duration": time.time() - start_time,
        }

    except Exception as e:
        logger.error(f"Failed to persist data for {key}: {e}")
        return {
            "status": "failed",
            "key": key,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def run_canary_deployment(
    ctx: dict[str, Any],
    deployment_id: str,
    deployment_config: dict[str, Any],
    auto_rollback: bool = True,
) -> dict[str, Any]:
    """Run canary deployment stages.

    This task replaces the asyncio.create_task pattern in CanaryDeployment.
    Publishes events to Redis Streams for coordination with other services.

    Args:
        ctx: ARQ context (provided by worker)
        deployment_id: ID of the deployment
        deployment_config: Serialized deployment configuration
        auto_rollback: Whether to auto-rollback on failure

    Returns:
        Task result with deployment status
    """
    start_time = time.time()
    logger.info(f"Starting canary deployment {deployment_id}")

    # Get Redis connection from ARQ context
    redis_conn = ctx.get("redis")

    try:
        # Publish deployment started event
        if redis_conn:
            await redis_conn.xadd(
                "deployment:events",
                {
                    "type": "deployment_started",
                    "deployment_id": deployment_id,
                    "alias": deployment_config.get("alias"),
                    "old_collection": deployment_config.get("old_collection"),
                    "new_collection": deployment_config.get("new_collection"),
                    "timestamp": str(time.time()),
                },
            )

        # Initialize required services
        config = get_config()
        client_manager = ClientManager(config)
        await client_manager.initialize()

        try:
            # Import CanaryDeployment to avoid circular imports
            from ..deployment.canary import CanaryDeployment
            from ..deployment.canary import CanaryDeploymentConfig
            from ..deployment.canary import CanaryMetrics
            from ..deployment.canary import CanaryStage
            from ..vector_db.service import QdrantService

            # Initialize services
            qdrant_client = await client_manager.get_qdrant_client()
            # Note: Pass None for task_queue_manager since we're already in a task context
            # and canary deployments typically don't need to schedule collection deletions
            alias_manager = QdrantAliasManager(config, qdrant_client, None)
            qdrant_service = QdrantService(config, qdrant_client)

            canary = CanaryDeployment(
                config,
                alias_manager,
                None,  # task_queue_manager - not needed in task context
                qdrant_service,
                client_manager,
            )
            await canary.initialize()

            # Reconstruct deployment config from dict
            stages = [
                CanaryStage(**stage_data) for stage_data in deployment_config["stages"]
            ]

            metrics = CanaryMetrics(
                latency=deployment_config["metrics"]["latency"],
                error_rate=deployment_config["metrics"]["error_rate"],
                success_count=deployment_config["metrics"]["success_count"],
                error_count=deployment_config["metrics"]["error_count"],
                stage_start_time=deployment_config["metrics"]["stage_start_time"],
            )

            deployment = CanaryDeploymentConfig(
                alias=deployment_config["alias"],
                old_collection=deployment_config["old_collection"],
                new_collection=deployment_config["new_collection"],
                stages=stages,
                current_stage=deployment_config["current_stage"],
                metrics=metrics,
                start_time=deployment_config["start_time"],
                status=deployment_config["status"],
            )

            # Set the deployment in the canary instance
            canary.deployments[deployment_id] = deployment

            # Run the canary deployment
            await canary._run_canary(deployment_id, auto_rollback)

            # Get final status
            final_status = await canary.get_deployment_status(deployment_id)

            # Publish deployment completed event
            if redis_conn and final_status.get("status") == "completed":
                await redis_conn.xadd(
                    "deployment:events",
                    {
                        "type": "deployment_completed",
                        "deployment_id": deployment_id,
                        "alias": deployment_config.get("alias"),
                        "new_collection": deployment_config.get("new_collection"),
                        "duration": str(time.time() - start_time),
                        "timestamp": str(time.time()),
                    },
                )

            return {
                "status": "success",
                "deployment_id": deployment_id,
                "deployment_status": final_status,
                "duration": time.time() - start_time,
            }

        finally:
            await client_manager.cleanup()

    except Exception as e:
        logger.error(f"Canary deployment {deployment_id} failed: {e}")

        # Publish deployment failed event
        if redis_conn:
            await redis_conn.xadd(
                "deployment:events",
                {
                    "type": "deployment_failed",
                    "deployment_id": deployment_id,
                    "alias": deployment_config.get("alias"),
                    "error": str(e),
                    "timestamp": str(time.time()),
                },
            )

        return {
            "status": "failed",
            "deployment_id": deployment_id,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def monitor_deployment_events(
    ctx: dict[str, Any],
    consumer_group: str = "deployment_monitor",
    stream_key: str = "deployment:events",
) -> dict[str, Any]:
    """Monitor deployment events and trigger actions.

    This task consumes events from Redis Streams and can trigger
    additional tasks based on deployment lifecycle events.

    Args:
        ctx: ARQ context (provided by worker)
        consumer_group: Redis consumer group name
        stream_key: Redis stream key to monitor

    Returns:
        Task result with monitoring status
    """
    start_time = time.time()
    logger.info(f"Starting deployment event monitor for {stream_key}")

    redis_conn = ctx.get("redis")
    if not redis_conn:
        return {
            "status": "failed",
            "error": "No Redis connection available",
            "duration": time.time() - start_time,
        }

    try:
        # Create consumer group if it doesn't exist
        with contextlib.suppress(Exception):
            # Try to create group - will fail silently if already exists
            await redis_conn.xgroup_create(stream_key, consumer_group, id="0")

        # Process events
        events_processed = 0
        while True:
            # Read events from stream
            events = await redis_conn.xreadgroup(
                consumer_group,
                "worker-1",  # Consumer name
                {stream_key: ">"},  # Read new messages
                count=10,
                block=5000,  # Block for 5 seconds
            )

            if not events:
                # No new events, check if we should continue
                if events_processed > 0:
                    # We've processed some events, exit
                    break
                continue

            for _stream_name, messages in events:
                for msg_id, data in messages:
                    events_processed += 1
                    event_type = data.get(b"type", b"").decode("utf-8")
                    deployment_id = data.get(b"deployment_id", b"").decode("utf-8")

                    logger.info(f"Processing event: {event_type} for {deployment_id}")

                    # Handle different event types
                    if event_type == "deployment_completed":
                        # Could trigger cleanup tasks, notifications, etc.
                        logger.info(
                            f"Deployment {deployment_id} completed successfully"
                        )
                    elif event_type == "deployment_failed":
                        # Could trigger rollback, alerts, etc.
                        error = data.get(b"error", b"").decode("utf-8")
                        logger.error(f"Deployment {deployment_id} failed: {error}")

                    # Acknowledge message
                    await redis_conn.xack(stream_key, consumer_group, msg_id)

        return {
            "status": "success",
            "events_processed": events_processed,
            "duration": time.time() - start_time,
        }

    except Exception as e:
        logger.error(f"Error monitoring deployment events: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time,
        }


# Create decorated versions for ARQ worker
delete_collection_task = func(
    delete_collection,
    name="delete_collection",
    max_tries=3,
    timeout=300,  # 5 minutes
)

persist_cache_task = func(
    persist_cache,
    name="persist_cache",
    max_tries=5,
    timeout=60,  # 1 minute
)

run_canary_deployment_task = func(
    run_canary_deployment,
    name="run_canary_deployment",
    max_tries=1,  # Don't retry canary deployments
    timeout=7200,  # 2 hours
)

monitor_deployment_events_task = func(
    monitor_deployment_events,
    name="monitor_deployment_events",
    max_tries=3,
    timeout=300,  # 5 minutes
)

# Task registry for worker
TASK_REGISTRY = {
    "delete_collection": delete_collection_task,
    "persist_cache": persist_cache_task,
    "run_canary_deployment": run_canary_deployment_task,
    "monitor_deployment_events": monitor_deployment_events_task,
}
