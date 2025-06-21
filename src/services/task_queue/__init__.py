"""Task queue module for persistent background tasks."""

from .manager import TaskQueueManager
from .tasks import delete_collection_task
from .tasks import persist_cache_task

# Removed: from .tasks import run_canary_deployment_task (deployment infrastructure removed)

__all__ = [
    "TaskQueueManager",
    "delete_collection_task",
    "persist_cache_task",
    # Removed: "run_canary_deployment_task", (deployment infrastructure removed)
]
