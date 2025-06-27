"""Utilities package for the AI documentation vector database system."""

# Re-export functions from parent utils.py and local imports module
from pathlib import Path

from .imports import resolve_imports, setup_import_paths


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
    def async_to_sync_click(*args, **kwargs):
        """Fallback async_to_sync_click function."""
        raise ImportError("async_to_sync_click not available")

    def async_command(*args, **kwargs):
        """Fallback async_command function."""
        raise ImportError("async_command not available")


__all__ = [
    "async_command",
    "async_to_sync_click",
    "resolve_imports",
    "setup_import_paths",
]
