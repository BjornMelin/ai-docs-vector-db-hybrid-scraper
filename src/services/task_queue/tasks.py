"""ARQ task definitions for background processing."""

import asyncio
import logging
import time
from typing import Any

from arq import func

from src.config import get_config

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
        logger.exception(f"Failed to delete collection {collection_name}: {e}")
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
        logger.exception(f"Failed to persist data for {key}: {e}")
        return {
            "status": "failed",
            "key": key,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def config_drift_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    """Take configuration snapshots for drift detection.
    
    Args:
        ctx: ARQ context (provided by worker)
        
    Returns:
        Task result with snapshot status
    """
    start_time = time.time()
    logger.info("Starting configuration drift snapshot task")
    
    try:
        # Import here to avoid circular imports
        from ..config_drift_service import get_drift_service
        
        service = get_drift_service()
        result = await service.take_configuration_snapshot()
        
        logger.info(
            f"Configuration snapshot completed - "
            f"snapshots taken: {result['snapshots_taken']}, "
            f"errors: {len(result['errors'])}"
        )
        
        result.update({
            "status": "success",
            "task_duration": time.time() - start_time,
        })
        
        return result
        
    except Exception as e:
        logger.exception(f"Configuration snapshot task failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "task_duration": time.time() - start_time,
        }


async def config_drift_comparison(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compare configurations to detect drift.
    
    Args:
        ctx: ARQ context (provided by worker)
        
    Returns:
        Task result with comparison status
    """
    start_time = time.time()
    logger.info("Starting configuration drift comparison task")
    
    try:
        # Import here to avoid circular imports
        from ..config_drift_service import get_drift_service
        
        service = get_drift_service()
        result = await service.compare_configurations()
        
        logger.info(
            f"Configuration comparison completed - "
            f"sources compared: {result['sources_compared']}, "
            f"drift events: {len(result['drift_events'])}, "
            f"alerts sent: {result['alerts_sent']}"
        )
        
        result.update({
            "status": "success",
            "task_duration": time.time() - start_time,
        })
        
        return result
        
    except Exception as e:
        logger.exception(f"Configuration comparison task failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "task_duration": time.time() - start_time,
        }


async def config_drift_remediation(
    ctx: dict[str, Any],
    event_id: str,
    source: str,
    drift_type: str,
    suggestion: str,
) -> dict[str, Any]:
    """Execute configuration drift auto-remediation.
    
    Args:
        ctx: ARQ context (provided by worker)
        event_id: Drift event ID
        source: Configuration source
        drift_type: Type of drift detected
        suggestion: Remediation suggestion
        
    Returns:
        Task result with remediation status
    """
    start_time = time.time()
    logger.info(f"Starting configuration drift remediation for event {event_id}")
    
    try:
        # Log the remediation attempt (actual implementation would apply changes)
        logger.info(
            f"Remediating drift event {event_id} in {source}: "
            f"Type: {drift_type}, Suggestion: {suggestion}"
        )
        
        # For now, this is a placeholder for actual remediation logic
        # In a real implementation, this would:
        # 1. Validate the remediation is still safe
        # 2. Apply the suggested changes
        # 3. Verify the changes were successful
        # 4. Take a new snapshot to confirm drift is resolved
        
        await asyncio.sleep(0.1)  # Simulate remediation work
        
        logger.info(f"Configuration drift remediation completed for event {event_id}")
        
        return {
            "status": "success",
            "event_id": event_id,
            "source": source,
            "drift_type": drift_type,
            "remediation_applied": False,  # Would be True when actually implemented
            "task_duration": time.time() - start_time,
        }
        
    except Exception as e:
        logger.exception(f"Configuration drift remediation failed for event {event_id}: {e}")
        return {
            "status": "failed",
            "event_id": event_id,
            "error": str(e),
            "task_duration": time.time() - start_time,
        }


# Removed enterprise deployment infrastructure:
# - run_canary_deployment()
# - monitor_deployment_events()
# These were over-engineered for V1 with 0 users

# ARQ task definitions (simplified for V1)
delete_collection_task = func(
    delete_collection,
    name="delete_collection",
    max_tries=2,
)

persist_cache_task = func(
    persist_cache,
    name="persist_cache",
    max_tries=3,
)

config_drift_snapshot_task = func(
    config_drift_snapshot,
    name="config_drift_snapshot",
    max_tries=2,
)

config_drift_comparison_task = func(
    config_drift_comparison,
    name="config_drift_comparison",
    max_tries=2,
)

config_drift_remediation_task = func(
    config_drift_remediation,
    name="config_drift_remediation",
    max_tries=1,
)

# Task registry for ARQ worker
TASK_MAP = {
    "delete_collection": delete_collection_task,
    "persist_cache": persist_cache_task,
    "config_drift_snapshot": config_drift_snapshot_task,
    "config_drift_comparison": config_drift_comparison_task,
    "config_drift_remediation": config_drift_remediation_task,
}

# Legacy alias for backward compatibility
TASK_REGISTRY = TASK_MAP
