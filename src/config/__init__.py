"""Simplified configuration system for AI Documentation Vector DB.

This module provides a clean, simple configuration system using pydantic-settings v2
with built-in .env support, validation, and secure handling of sensitive data.
"""

# Main configuration interface
# Configuration models (for backward compatibility)
from .core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    Config,
    Crawl4AIConfig,
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    MonitoringConfig,
    OpenAIConfig,
    PerformanceConfig,
    PlaywrightConfig,
    QdrantConfig,
    SecurityConfig,
    get_config_with_auto_detection,
    reset_config,
    set_config,
)

# Enums for validation (minimal subset actually used)
from .enums import (
    CacheType,
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    LogLevel,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)

# Helper utilities
from .helpers import (
    ensure_path_exists,
    get_env_bool,
    get_env_list,
    mask_secret,
    merge_config_dicts,
)
from .settings import Settings, get_settings, reload_settings


# For backward compatibility, map old imports to new settings
def get_config():
    """Get configuration (backward compatibility)."""
    return get_settings()


async def get_config_with_auto_detection_compat():
    """Get configuration with auto-detection (backward compatibility)."""
    return await get_config_with_auto_detection()


def reset_config_compat():
    """Reset configuration (backward compatibility)."""
    return reload_settings()


# Legacy type aliases
Config = Settings
UnifiedConfig = Settings

__all__ = [
    # Main interface
    "Settings",
    "get_settings",
    "reload_settings",
    # Helpers
    "ensure_path_exists",
    "get_env_bool",
    "get_env_list",
    "mask_secret",
    "merge_config_dicts",
    # Enums
    "CacheType",
    "ChunkingStrategy",
    "CrawlProvider",
    "EmbeddingProvider",
    "Environment",
    "LogLevel",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
    # Configuration models
    "BrowserUseConfig",
    "CacheConfig",
    "ChunkingConfig",
    "Config",
    "Crawl4AIConfig",
    "DocumentationSite",
    "EmbeddingConfig",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "HyDEConfig",
    "MonitoringConfig",
    "OpenAIConfig",
    "PerformanceConfig",
    "PlaywrightConfig",
    "QdrantConfig",
    "SecurityConfig",
    # Auto-detection functions
    "get_config_with_auto_detection",
    # Backward compatibility
    "UnifiedConfig",
    "get_config",
    "reset_config",
    "set_config",
]
