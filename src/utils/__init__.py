"""Utilities package for the AI documentation vector database system."""

# Import async_to_sync_click from the main utils module
import sys
from pathlib import Path

from .imports import resolve_imports
from .imports import setup_import_paths

# Add the src directory to import the main utils.py file
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from utils import async_to_sync_click
except ImportError:
    # Fallback implementation
    def async_to_sync_click(*args, **kwargs):
        """Fallback async_to_sync_click function."""
        raise ImportError("async_to_sync_click not available")


__all__ = ["async_to_sync_click", "resolve_imports", "setup_import_paths"]
