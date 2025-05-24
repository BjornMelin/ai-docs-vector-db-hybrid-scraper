"""Unified configuration system for AI Documentation Vector DB.

This module provides a comprehensive configuration system that consolidates all
settings across the application into a single, well-structured configuration model.
"""

# Core configuration models
from .models import (
    CacheConfig,
    ChunkingConfig,
    ChunkingStrategy,
    CrawlProvider,
    Crawl4AIConfig,
    DocumentationSite,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    FirecrawlConfig,
    LogLevel,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    SecurityConfig,
    UnifiedConfig,
    get_config,
    reset_config,
    set_config,
)

# Configuration loading and management
from .loader import ConfigLoader

# Configuration validation
from .validator import ConfigValidator

# Schema generation
from .schema import ConfigSchemaGenerator

# Configuration migration
from .migrator import ConfigMigrator

__all__ = [
    # Core models
    "UnifiedConfig",
    "Environment",
    "LogLevel",
    "EmbeddingProvider",
    "CrawlProvider",
    "ChunkingStrategy",
    # Component configs
    "CacheConfig",
    "QdrantConfig",
    "OpenAIConfig",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "Crawl4AIConfig",
    "ChunkingConfig",
    "DocumentationSite",
    "PerformanceConfig",
    "SecurityConfig",
    # Functions
    "get_config",
    "set_config",
    "reset_config",
    # Utilities
    "ConfigLoader",
    "ConfigValidator",
    "ConfigSchemaGenerator",
    "ConfigMigrator",
]