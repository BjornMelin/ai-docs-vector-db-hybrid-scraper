"""ARQ task definitions for background processing."""

import asyncio
import importlib
import logging
import time
from datetime import timedelta
from typing import Any

from arq import func

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.core.qdrant_alias_manager import QdrantAliasManager


logger = logging.getLogger(__name__)


# Security whitelist for dynamic imports to prevent arbitrary code execution
ALLOWED_PERSIST_MODULES = {
    "src.services.cache.manager",
    "src.services.cache.dragonfly_cache",
    "src.services.cache.embedding_cache",
    "src.services.cache.search_cache",
    "src.services.core.persistence",
}

ALLOWED_PERSIST_FUNCTIONS = {
    "persist_to_disk",
    "persist_embeddings",
    "persist_search_results",
    "persist_cache_data",
    "save_to_storage",
    "write_cache_backup",
}


def _validate_dynamic_import(module_name: str, function_name: str) -> bool:
    """Validate that dynamic import parameters are from allowed whitelist.

    Args:
        module_name: Module to import
        function_name: Function to call

    Returns:
        True if both module and function are whitelisted

    Raises:
        ValueError: If module or function is not whitelisted
    """
    if module_name not in ALLOWED_PERSIST_MODULES:
        msg = f"Module '{module_name}' not in security whitelist"
        raise ValueError(msg)

    if function_name not in ALLOWED_PERSIST_FUNCTIONS:
        msg = f"Function '{function_name}' not in security whitelist"
        raise ValueError(msg)

    return True


async def delete_collection(
    _ctx: dict[str, Any],
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
        await asyncio.sleep(grace_period_minutes * 60)

        config = get_config()
        client_manager = ClientManager()
        await client_manager.initialize()

        try:
            qdrant_client = await client_manager.get_qdrant_client()
            alias_manager = QdrantAliasManager(config, qdrant_client, None)

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

            await qdrant_client.delete_collection(collection_name)
            logger.info(
                f"Successfully deleted collection {collection_name}"
            )  # TODO: Convert f-string to logging format

            return {
                "status": "success",
                "collection": collection_name,
                "duration": time.time() - start_time,
            }
        finally:
            await client_manager.cleanup()

    except asyncio.CancelledError:
        logger.info(
            f"Deletion of {collection_name} was cancelled"
        )  # TODO: Convert f-string to logging format
        return {
            "status": "cancelled",
            "collection": collection_name,
            "duration": time.time() - start_time,
        }
    except Exception as e:
        logger.exception("Failed to delete collection {collection_name}")
        return {
            "status": "failed",
            "collection": collection_name,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def persist_cache(
    _ctx: dict[str, Any],
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
    logger.info(
        f"Starting delayed persistence for {key} (delay: {delay}s)"
    )  # TODO: Convert f-string to logging format

    try:
        await asyncio.sleep(delay)

        # Validate dynamic import for security
        try:
            _validate_dynamic_import(persist_func_module, persist_func_name)
        except ValueError as ve:
            logger.exception("Security validation failed for dynamic import: ")
            return {
                "status": "failed",
                "key": key,
                "error": f"Security validation failed: {ve}",
                "duration": time.time() - start_time,
            }

        # Import and call persist function
        module = importlib.import_module(persist_func_module)
        persist_func = getattr(module, persist_func_name)

        if asyncio.iscoroutinefunction(persist_func):
            await persist_func(key, value)
        else:
            persist_func(key, value)

        logger.info(
            f"Successfully persisted data for {key}"
        )  # TODO: Convert f-string to logging format

        return {
            "status": "success",
            "key": key,
            "duration": time.time() - start_time,
        }

    except Exception as e:
        logger.exception("Failed to persist data for {key}")
        return {
            "status": "failed",
            "key": key,
            "error": str(e),
            "duration": time.time() - start_time,
        }


async def config_drift_snapshot(_ctx: dict[str, Any]) -> dict[str, Any]:
    """Take configuration snapshots for drift detection.

    Args:
        ctx: ARQ context (provided by worker)

    Returns:
        Task result with snapshot status

    """
    start_time = time.time()
    logger.info("Starting configuration drift snapshot task")

    try:
        # Use dynamic import to avoid circular dependency

        config_drift_module = importlib.import_module(
            "src.services.config_drift_service"
        )
        get_drift_service = config_drift_module.get_drift_service

        service = get_drift_service()
        result = await service.take_configuration_snapshot()

        logger.info(
            f"Configuration snapshot completed - "
            f"snapshots taken: {result['snapshots_taken']}, "
            "errors"
        )

        result.update(
            {
                "status": "success",
                "task_duration": time.time() - start_time,
            }
        )

    except Exception as e:
        logger.exception("Configuration snapshot task failed")
        return {
            "status": "failed",
            "error": str(e),
            "task_duration": time.time() - start_time,
        }

    else:
        return result


async def config_drift_comparison(_ctx: dict[str, Any]) -> dict[str, Any]:
    """Compare configurations to detect drift.

    Args:
        ctx: ARQ context (provided by worker)

    Returns:
        Task result with comparison status

    """
    start_time = time.time()
    logger.info("Starting configuration drift comparison task")

    try:
        # Use dynamic import to avoid circular dependency

        config_drift_module = importlib.import_module(
            "src.services.config_drift_service"
        )
        get_drift_service = config_drift_module.get_drift_service

        service = get_drift_service()
        result = await service.compare_configurations()

        logger.info(
            f"Configuration comparison completed - "
            f"sources compared: {result['sources_compared']}, "
            f"drift events: {len(result['drift_events'])}, "
            "alerts sent"
        )

        result.update(
            {
                "status": "success",
                "task_duration": time.time() - start_time,
            }
        )

    except Exception as e:
        logger.exception("Configuration comparison task failed")
        return {
            "status": "failed",
            "error": str(e),
            "task_duration": time.time() - start_time,
        }

    else:
        return result


async def config_drift_remediation(
    _ctx: dict[str, Any],
    event_id: str,
    source: str,
    drift_type: str,
    _suggestion: str,
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
    logger.info(
        f"Starting configuration drift remediation for event {event_id}"
    )  # TODO: Convert f-string to logging format

    try:
        # Log the remediation attempt (actual implementation would apply changes)
        logger.info(
            f"Remediating drift event {event_id} in {source}: "
            "Type: {drift_type}, Suggestion"
        )

        # For now, this is a placeholder for actual remediation logic
        # In a real implementation, this would:
        # 1. Validate the remediation is still safe
        # 2. Apply the suggested changes
        # 3. Verify the changes were successful
        # 4. Take a new snapshot to confirm drift is resolved

        await asyncio.sleep(0.1)  # Simulate remediation work

        logger.info(
            f"Configuration drift remediation completed for event {event_id}"
        )  # TODO: Convert f-string to logging format

        return {
            "status": "success",
            "event_id": event_id,
            "source": source,
            "drift_type": drift_type,
            "remediation_applied": False,  # Would be True when actually implemented
            "task_duration": time.time() - start_time,
        }

    except Exception as e:
        logger.exception("Configuration drift remediation failed for event {event_id}")
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


async def create_task(
    task_name: str,  # noqa: ARG001
    task_data: dict[str, Any],  # noqa: ARG001
    delay: timedelta | None = None,  # noqa: ARG001
) -> str | None:
    """Create and enqueue a task for background processing.

    This is a helper function to provide a consistent interface for
    enqueueing tasks from other services.

    Args:
        task_name: Name of the task to execute
        task_data: Data to pass to the task
        delay: Optional delay before execution

    Returns:
        Job ID if successful, None otherwise
    """
    try:
        # Get task queue manager
        client_manager = ClientManager()
        await client_manager.initialize()

        try:
            # Note: get_task_queue_manager method not yet implemented in ClientManager
            # TODO: Implement task queue manager in ClientManager
            logger.warning("Task queue manager not available - task queuing disabled")
            return None

        finally:
            await client_manager.cleanup()

    except Exception:
        logger.exception("Failed to create task {task_name}")
        return None
