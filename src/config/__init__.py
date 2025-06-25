"""Simplified configuration system for AI Documentation Vector DB.

This module provides a clean, simple configuration system using pydantic-settings v2
with built-in .env support, validation, and secure handling of sensitive data.
"""

# Main configuration interface
from .settings import Settings, get_settings, reload_settings

# Configuration models (for backward compatibility)
from .core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
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
    set_config,
)

# Helper utilities
from .helpers import (
    ensure_path_exists,
    get_env_bool,
    get_env_list,
    mask_secret,
    merge_config_dicts,
)

# Enums for validation (minimal subset actually used)
from .enums import (
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    LogLevel,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)

# For backward compatibility, map old imports to new settings
def get_config():
    """Get configuration (backward compatibility)."""
    return get_settings()

def reset_config():
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
    
    # Backward compatibility
    "Config",
    "UnifiedConfig",
    "get_config",
    "reset_config",
    "set_config",
]
