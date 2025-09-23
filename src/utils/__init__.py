"""Utilities package for the AI documentation vector database system."""

# Re-export functions from parent utils.py and local imports module
# GPU utilities (optional)
from contextlib import suppress
from pathlib import Path

from .imports import resolve_imports, setup_import_paths


with suppress(ImportError):
    from .gpu import (
        get_gpu_device,
        get_gpu_memory_info,
        get_gpu_stats,
        is_gpu_available,
        optimize_gpu_memory,
        safe_gpu_operation,
    )


# Get parent directory and import directly from utils.py
parent_dir = Path(__file__).parent.parent
utils_path = parent_dir / "utils.py"

# Import functions from the utils.py file by executing it
if utils_path.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location("parent_utils", utils_path)
    parent_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_utils)

    async_to_sync_click = parent_utils.async_to_sync_click
    async_command = parent_utils.async_command
else:
    # Fallback implementations
    def async_to_sync_click(*_args, **_kwargs):
        """Fallback async_to_sync_click function."""
        msg = "async_to_sync_click not available"
        raise ImportError(msg)

    def async_command(*_args, **_kwargs):
        """Fallback async_command function."""
        msg = "async_command not available"
        raise ImportError(msg)


__all__ = [
    "async_command",
    "async_to_sync_click",
    "get_gpu_device",
    "get_gpu_memory_info",
    "get_gpu_stats",
    "is_gpu_available",
    "optimize_gpu_memory",
    "resolve_imports",
    "safe_gpu_operation",
    "setup_import_paths",
]
