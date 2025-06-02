"""Unified configuration system for AI Documentation Vector DB.

This module provides a comprehensive configuration system that consolidates all
settings across the application into a single, well-structured configuration model.
"""

# Core configuration models
# Configuration loading and management
# Configuration enums
from .enums import ChunkingStrategy
from .enums import CrawlProvider
from .enums import EmbeddingProvider
from .enums import Environment
from .enums import LogLevel
from .loader import ConfigLoader

# Configuration migration
from .migrator import ConfigMigrator
from .models import CacheConfig
from .models import ChunkingConfig
from .models import CollectionHNSWConfigs
from .models import Crawl4AIConfig
from .models import DocumentationSite
from .models import EmbeddingConfig
from .models import FastEmbedConfig
from .models import FirecrawlConfig
from .models import HNSWConfig
from .models import HyDEConfig
from .models import ModelBenchmark
from .models import OpenAIConfig
from .models import PerformanceConfig
from .models import QdrantConfig
from .models import SecurityConfig
from .models import SmartSelectionConfig
from .models import UnifiedConfig
from .models import get_config
from .models import reset_config
from .models import set_config

# Schema generation
from .schema import ConfigSchemaGenerator

# Configuration validation
from .validator import ConfigValidator

__all__ = [
    # Component configs
    "CacheConfig",
    "ChunkingConfig",
    "ChunkingStrategy",
    "CollectionHNSWConfigs",
    # Utilities
    "ConfigLoader",
    "ConfigMigrator",
    "ConfigSchemaGenerator",
    "ConfigValidator",
    "Crawl4AIConfig",
    "CrawlProvider",
    "DocumentationSite",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "Environment",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "HNSWConfig",
    "HyDEConfig",
    "LogLevel",
    "ModelBenchmark",
    "OpenAIConfig",
    "PerformanceConfig",
    "QdrantConfig",
    "SecurityConfig",
    "SmartSelectionConfig",
    # Core models
    "UnifiedConfig",
    # Functions
    "get_config",
    "reset_config",
    "set_config",
]
