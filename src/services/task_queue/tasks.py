
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

# Task registry for ARQ worker
TASK_MAP = {
    "delete_collection": delete_collection_task,
    "persist_cache": persist_cache_task,
}

# Legacy alias for backward compatibility
TASK_REGISTRY = TASK_MAP
