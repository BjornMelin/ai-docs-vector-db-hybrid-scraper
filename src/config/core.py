"""Core configuration functions for backward compatibility.

This module re-exports the main configuration functions from the unified
settings system to maintain backward compatibility with existing code.
"""

# Re-export main configuration functions
from .settings import Config, get_config, reset_config


__all__ = [
    "Config",
    "get_config",
    "reset_config",
]
