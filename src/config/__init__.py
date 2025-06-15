"""Simplified configuration system for AI Documentation Vector DB.

Consolidated from 14 files (6,797 lines) to 3 files (~400 lines) following KISS principles.
Provides all essential configuration without over-engineering.
"""

# Core configuration
from .core import BrowserUseConfig
from .core import CacheConfig
from .core import ChunkingConfig
from .core import Config
from .core import Crawl4AIConfig
from .core import DocumentationSite
from .core import EmbeddingConfig
from .core import FastEmbedConfig
from .core import FirecrawlConfig
from .core import HyDEConfig
from .core import MonitoringConfig
from .core import OpenAIConfig
from .core import PerformanceConfig
from .core import PlaywrightConfig
from .core import QdrantConfig
from .core import SecurityConfig
from .core import SQLAlchemyConfig
from .core import get_config
from .core import reset_config
from .core import set_config

# Legacy aliases for backward compatibility
UnifiedConfig = Config

# Enums
from .enums import ABTestVariant
from .enums import CacheType
from .enums import ChunkingStrategy
from .enums import CrawlProvider
from .enums import DocumentStatus
from .enums import EmbeddingModel
from .enums import EmbeddingProvider
from .enums import Environment
from .enums import FusionAlgorithm
from .enums import LogLevel
from .enums import ModelType
from .enums import OptimizationStrategy
from .enums import QueryComplexity
from .enums import QueryType
from .enums import SearchAccuracy
from .enums import SearchStrategy
from .enums import VectorType

__all__ = [
    # Main config
    "Config",
    "UnifiedConfig",  # Legacy alias
    "get_config",
    "set_config",
    "reset_config",
    # Config components
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
    "SQLAlchemyConfig",
    # Enums
    "ABTestVariant",
    "CacheType",
    "ChunkingStrategy",
    "CrawlProvider",
    "DocumentStatus",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "FusionAlgorithm",
    "LogLevel",
    "ModelType",
    "OptimizationStrategy",
    "QueryComplexity",
    "QueryType",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
]
