"""Simplified configuration system for AI Documentation Vector DB.

Consolidated from 14 files (6,797 lines) to 3 files (~400 lines) following KISS principles.
Provides all essential configuration without over-engineering.
"""

# Core configuration
from .core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    Config,
    Crawl4AIConfig,
    DeploymentConfig,
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
    SQLAlchemyConfig,
    TaskQueueConfig,
    get_config,
    reset_config,
    set_config,
)

# Deployment tier configuration
from .deployment_tiers import (
    DeploymentTier,
    TierCapability,
    TierConfiguration,
    TierManager,
    default_tier_manager,
    get_current_tier_config,
    is_feature_enabled,
)

# Enums
from .enums import (
    ABTestVariant,
    CacheType,
    ChunkingStrategy,
    CrawlProvider,
    DocumentStatus,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    FusionAlgorithm,
    LogLevel,
    ModelType,
    OptimizationStrategy,
    QueryComplexity,
    QueryType,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)


# Legacy aliases for backward compatibility
UnifiedConfig = Config

__all__: list[str] = [
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
