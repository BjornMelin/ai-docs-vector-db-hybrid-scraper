"""Simplified configuration system for AI Documentation Vector DB.

Consolidated from 14 files (6,797 lines) to 3 files (~400 lines) following KISS principles.
Provides all essential configuration without over-engineering.
"""

# Core configuration
from .core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    CircuitBreakerConfig,
    Config,
    Crawl4AIConfig,
    DeploymentConfig,
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    PlaywrightConfig,
    QdrantConfig,
    RAGConfig,
    SecurityConfig,
    SQLAlchemyConfig,
    TaskQueueConfig,
    get_config,
    get_config_with_auto_detection,
    reset_config,
    set_config,
)

# Instrumented configuration with OpenTelemetry tracing
# Disabled due to circular imports and missing module
# from .instrumented_core import (
#     InstrumentedConfig,
#     create_instrumented_config,
#     create_instrumented_config_with_auto_detection,
#     load_instrumented_config_from_file,
#     load_instrumented_config_from_file_async,
# )

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
    "CircuitBreakerConfig",
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
    # "InstrumentedConfig",  # Disabled due to circular imports
    "LogLevel",
    "ModelType",
    "MonitoringConfig",
    "ObservabilityConfig",
    "OpenAIConfig",
    "OptimizationStrategy",
    "PerformanceConfig",
    "PlaywrightConfig",
    "QdrantConfig",
    "QueryComplexity",
    "QueryType",
    "RAGConfig",
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
    # "create_instrumented_config",  # Disabled due to circular imports
    # "create_instrumented_config_with_auto_detection",  # Disabled due to circular imports
    "default_tier_manager",
    "get_config",
    "get_config_with_auto_detection",
    "get_current_tier_config",
    "is_feature_enabled",
    # "load_instrumented_config_from_file",  # Disabled due to circular imports
    # "load_instrumented_config_from_file_async",  # Disabled due to circular imports
    "reset_config",
    "set_config",
]
