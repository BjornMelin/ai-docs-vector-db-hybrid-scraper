"""Task queue configuration module.

This module provides backward compatibility for task queue configuration
that is now unified in the main config system.
"""

# Import task queue config from the unified settings system
from .settings import (
    TaskQueueConfig,
    get_config,
)


# Export everything for backward compatibility
__all__ = [
    "TaskQueueConfig",
    "get_config",
]
