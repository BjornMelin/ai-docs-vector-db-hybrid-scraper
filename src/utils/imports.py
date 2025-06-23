
"""Import resolution utilities for handling module vs script execution contexts."""

import sys
from pathlib import Path


def setup_import_paths():
    """Add necessary paths to sys.path for proper import resolution.

    This ensures imports work correctly whether the code is run as:
    - A module (python -m src.module)
    - A script (python src/module.py)
    - From tests
    - From the MCP server
    """
    # Get the src directory
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent
    project_root = src_dir.parent

    # Add paths if not already present
    paths_to_add = [str(src_dir), str(project_root)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_imports():
    """Decorator to handle import resolution for a module.

    Usage:
        @resolve_imports()
        def main():
            # Your imports will work correctly here
            from src.infrastructure.client_manager import ClientManager
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            setup_import_paths()
            return func(*args, **kwargs)

        return wrapper

    return decorator
