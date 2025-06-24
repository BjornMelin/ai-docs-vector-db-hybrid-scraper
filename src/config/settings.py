"""Settings module for backward compatibility and simple access to configuration.

This module provides a simplified interface to the main configuration system.
"""

from .core import Config, get_config


# Global configuration instance
_config: Config | None = None


def get_settings() -> Config:
    """Get the global configuration instance.

    Returns:
        Config: The global configuration instance.
    """
    global _config
    if _config is None:
        _config = get_config()
    return _config


# Convenience aliases for common configuration access patterns
def get_qdrant_config():
    """Get Qdrant configuration."""
    return get_settings().qdrant


def get_embedding_config():
    """Get embedding configuration."""
    return get_settings().embedding


def get_cache_config():
    """Get cache configuration."""
    return get_settings().cache


def get_performance_config():
    """Get performance configuration."""
    return get_settings().performance


# Legacy alias for backward compatibility
settings = get_settings()

__all__ = [
    "get_cache_config",
    "get_embedding_config",
    "get_performance_config",
    "get_qdrant_config",
    "get_settings",
    "settings",
]
