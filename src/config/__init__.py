import typing

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
from .core import DeploymentConfig
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
from .core import TaskQueueConfig
from .core import get_config
from .core import reset_config
from .core import set_config

# Deployment tier configuration
from .deployment_tiers import DeploymentTier
from .deployment_tiers import TierCapability
from .deployment_tiers import TierConfiguration
from .deployment_tiers import TierManager
from .deployment_tiers import default_tier_manager
from .deployment_tiers import get_current_tier_config
from .deployment_tiers import is_feature_enabled

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

# Legacy aliases for backward compatibility
UnifiedConfig = Config

__all__ = [
    "ABTestVariant",
    "BrowserUseConfig",
    "CacheConfig",
    "CacheType",
    "ChunkingConfig",
    "ChunkingStrategy",
    "Config",
    "Crawl4AIConfig",
    "CrawlProvider",
    "DeploymentConfig",
    "DeploymentTier",
    "DocumentStatus",
    "DocumentationSite",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "FusionAlgorithm",
    "HyDEConfig",
    "LogLevel",
    "ModelType",
    "MonitoringConfig",
    "OpenAIConfig",
    "OptimizationStrategy",
    "PerformanceConfig",
    "PlaywrightConfig",
    "QdrantConfig",
    "QueryComplexity",
    "QueryType",
    "SQLAlchemyConfig",
    "SearchAccuracy",
    "SearchStrategy",
    "SecurityConfig",
    "TaskQueueConfig",
    "TierCapability",
    "TierConfiguration",
    "TierManager",
    "UnifiedConfig",
    "VectorType",
    "default_tier_manager",
    "get_config",
    "get_current_tier_config",
    "is_feature_enabled",
    "reset_config",
    "set_config",
]
