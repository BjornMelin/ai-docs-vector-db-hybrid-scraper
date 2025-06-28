"""Modern configuration system for AI Documentation Vector DB.

MIGRATION IN PROGRESS: Replacing 18-file configuration system (8,599 lines) 
with modern Pydantic Settings 2.0 (2-3 files, ~500 lines) - 94% reduction.

NEW: Use src.config.modern for the new configuration system
LEGACY: Old configuration system maintained for backward compatibility
"""

import os
import warnings
from typing import Union

# NEW MODERN CONFIGURATION SYSTEM
from .modern import (
    Config as ModernConfig,
    ApplicationMode,
    Environment,
    LogLevel,
    EmbeddingProvider,
    CrawlProvider,
    ChunkingStrategy,
    SearchStrategy,
    CacheConfig as ModernCacheConfig,
    PerformanceConfig as ModernPerformanceConfig,
    OpenAIConfig as ModernOpenAIConfig,
    QdrantConfig as ModernQdrantConfig,
    FirecrawlConfig as ModernFirecrawlConfig,
    SecurityConfig as ModernSecurityConfig,
    ChunkingConfig as ModernChunkingConfig,
    HyDEConfig as ModernHyDEConfig,
    ReRankingConfig,
    get_config as get_modern_config,
    set_config as set_modern_config,
    reset_config as reset_modern_config,
    create_simple_config,
    create_enterprise_config,
    get_config_with_auto_detection as get_modern_config_with_auto_detection,
)

# MIGRATION UTILITIES
from .migration import (
    ConfigMigrator,
    migrate_legacy_config,
    create_migration_compatibility_wrapper,
)

# LEGACY CONFIGURATION SYSTEM (for backward compatibility)
from .core import (
    BrowserUseConfig,
    CacheConfig as LegacyCacheConfig,
    ChunkingConfig as LegacyChunkingConfig,
    CircuitBreakerConfig,
    Config as LegacyConfig,
    Crawl4AIConfig,
    DeploymentConfig,
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig as LegacyFirecrawlConfig,
    HyDEConfig as LegacyHyDEConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig as LegacyOpenAIConfig,
    PerformanceConfig as LegacyPerformanceConfig,
    PlaywrightConfig,
    QdrantConfig as LegacyQdrantConfig,
    RAGConfig,
    SecurityConfig as LegacySecurityConfig,
    SQLAlchemyConfig,
    TaskQueueConfig,
    get_config as get_legacy_config,
    get_config_with_auto_detection as get_legacy_config_with_auto_detection,
    reset_config as reset_legacy_config,
    set_config as set_legacy_config,
)

from .config_manager import (
    ConfigManager,
    create_and_load_config_async,
    get_config_manager,
    set_config_manager,
)

# MIGRATION ENVIRONMENT VARIABLE
_USE_MODERN_CONFIG = os.environ.get("AI_DOCS__USE_MODERN_CONFIG", "true").lower() == "true"

# UNIFIED CONFIGURATION INTERFACE
# These functions provide a unified interface that switches between legacy and modern config
# based on the AI_DOCS__USE_MODERN_CONFIG environment variable

def get_config() -> Union[ModernConfig, LegacyConfig]:
    """Get configuration instance.
    
    Returns modern config by default, or legacy config if AI_DOCS__USE_MODERN_CONFIG=false.
    
    Returns:
        Configuration instance (modern or legacy based on environment variable).
    """
    if _USE_MODERN_CONFIG:
        return get_modern_config()
    else:
        warnings.warn(
            "Using legacy configuration system. Consider migrating to modern config. "
            "Set AI_DOCS__USE_MODERN_CONFIG=true to use the new system.",
            DeprecationWarning,
            stacklevel=2
        )
        return get_legacy_config()


def set_config(config: Union[ModernConfig, LegacyConfig]) -> None:
    """Set configuration instance.
    
    Args:
        config: Configuration instance to set.
    """
    if isinstance(config, ModernConfig):
        set_modern_config(config)
    else:
        set_legacy_config(config)


def reset_config() -> None:
    """Reset configuration instance."""
    if _USE_MODERN_CONFIG:
        reset_modern_config()
    else:
        reset_legacy_config()


def get_config_with_auto_detection() -> Union[ModernConfig, LegacyConfig]:
    """Get configuration with auto-detection.
    
    Returns:
        Configuration instance with auto-detection applied.
    """
    if _USE_MODERN_CONFIG:
        return get_modern_config_with_auto_detection()
    else:
        return get_legacy_config_with_auto_detection()


# MIGRATION HELPER FUNCTIONS

def migrate_to_modern_config() -> ModernConfig:
    """Migrate current configuration to modern system.
    
    Returns:
        Modern configuration instance migrated from current legacy config.
    """
    if _USE_MODERN_CONFIG:
        return get_modern_config()
    
    legacy_config = get_legacy_config()
    return migrate_legacy_config(legacy_config)


def is_using_modern_config() -> bool:
    """Check if using modern configuration system.
    
    Returns:
        True if using modern config, False if using legacy config.
    """
    return _USE_MODERN_CONFIG


def get_migration_status() -> dict:
    """Get configuration migration status.
    
    Returns:
        Dictionary with migration status information.
    """
    config_type = "modern" if _USE_MODERN_CONFIG else "legacy"
    
    try:
        if _USE_MODERN_CONFIG:
            config = get_modern_config()
            mode = config.mode.value
            provider_count = 2  # embedding + crawl providers
        else:
            config = get_legacy_config()
            mode = getattr(config, 'mode', 'unknown')
            provider_count = len([
                getattr(config, 'embedding_provider', None),
                getattr(config, 'crawl_provider', None)
            ])
        
        return {
            "config_type": config_type,
            "mode": mode,
            "providers_configured": provider_count,
            "migration_available": not _USE_MODERN_CONFIG,
            "environment_variable": "AI_DOCS__USE_MODERN_CONFIG",
            "current_value": str(_USE_MODERN_CONFIG).lower(),
        }
    except Exception as e:
        return {
            "config_type": config_type,
            "error": str(e),
            "migration_available": not _USE_MODERN_CONFIG,
        }


# BACKWARD COMPATIBILITY ALIASES
# These maintain compatibility with existing code during migration

# Main Config class (defaults to modern, falls back to legacy)
if _USE_MODERN_CONFIG:
    Config = ModernConfig
    CacheConfig = ModernCacheConfig
    PerformanceConfig = ModernPerformanceConfig
    OpenAIConfig = ModernOpenAIConfig
    QdrantConfig = ModernQdrantConfig
    FirecrawlConfig = ModernFirecrawlConfig
    SecurityConfig = ModernSecurityConfig
    ChunkingConfig = ModernChunkingConfig
    HyDEConfig = ModernHyDEConfig
else:
    Config = LegacyConfig
    CacheConfig = LegacyCacheConfig
    PerformanceConfig = LegacyPerformanceConfig
    OpenAIConfig = LegacyOpenAIConfig
    QdrantConfig = LegacyQdrantConfig
    FirecrawlConfig = LegacyFirecrawlConfig
    SecurityConfig = LegacySecurityConfig
    ChunkingConfig = LegacyChunkingConfig
    HyDEConfig = LegacyHyDEConfig

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

# Enhanced error handling
from .error_handling import (
    ConfigError,
    ConfigFileWatchError,
    ConfigLoadError,
    ConfigReloadError,
    ConfigValidationError,
    ErrorContext,
    GracefulDegradationHandler,
    RetryableConfigOperation,
    SafeConfigLoader,
    async_error_context,
    get_degradation_handler,
    handle_validation_error,
    retry_config_operation,
)


# Legacy aliases for backward compatibility
UnifiedConfig = Config

__all__: list[str] = [
    # Modern Configuration System
    "ModernConfig",
    "ApplicationMode",
    "Environment",
    "LogLevel",
    "EmbeddingProvider", 
    "CrawlProvider",
    "ChunkingStrategy",
    "SearchStrategy",
    "ModernCacheConfig",
    "ModernPerformanceConfig",
    "ModernOpenAIConfig",
    "ModernQdrantConfig",
    "ModernFirecrawlConfig",
    "ModernSecurityConfig",
    "ModernChunkingConfig",
    "ModernHyDEConfig",
    "ReRankingConfig",
    "create_simple_config",
    "create_enterprise_config",
    
    # Migration Utilities
    "ConfigMigrator",
    "migrate_legacy_config",
    "create_migration_compatibility_wrapper",
    "migrate_to_modern_config",
    "is_using_modern_config",
    "get_migration_status",
    
    # Unified Interface (switches between modern/legacy)
    "Config",
    "CacheConfig",
    "PerformanceConfig",
    "OpenAIConfig",
    "QdrantConfig",
    "FirecrawlConfig",
    "SecurityConfig",
    "ChunkingConfig",
    "HyDEConfig",
    "get_config",
    "set_config",
    "reset_config",
    "get_config_with_auto_detection",
    
    # Legacy Configuration System (for backward compatibility)
    "LegacyConfig",
    "LegacyCacheConfig",
    "LegacyPerformanceConfig",
    "LegacyOpenAIConfig",
    "LegacyQdrantConfig",
    "LegacyFirecrawlConfig",
    "LegacySecurityConfig",
    "LegacyChunkingConfig",
    "LegacyHyDEConfig",
    "BrowserUseConfig",
    "CircuitBreakerConfig",
    "Crawl4AIConfig",
    "DeploymentConfig",
    "DocumentationSite",
    "EmbeddingConfig",
    "FastEmbedConfig",
    "MonitoringConfig",
    "ObservabilityConfig",
    "PlaywrightConfig",
    "RAGConfig",
    "SQLAlchemyConfig",
    "TaskQueueConfig",
    
    # Configuration Management
    "ConfigManager",
    "create_and_load_config_async",
    "get_config_manager",
    "set_config_manager",
    
    # Deployment Tiers
    "DeploymentTier",
    "TierCapability", 
    "TierConfiguration",
    "TierManager",
    "default_tier_manager",
    "get_current_tier_config",
    "is_feature_enabled",
    
    # Enums (from legacy system)
    "ABTestVariant",
    "CacheType",
    "DocumentStatus",
    "EmbeddingModel",
    "FusionAlgorithm",
    "ModelType",
    "OptimizationStrategy",
    "QueryComplexity",
    "QueryType",
    "SearchAccuracy",
    "VectorType",
    
    # Error Handling
    "ConfigError",
    "ConfigFileWatchError",
    "ConfigLoadError",
    "ConfigReloadError",
    "ConfigValidationError",
    "ErrorContext",
    "GracefulDegradationHandler",
    "RetryableConfigOperation",
    "SafeConfigLoader",
    "async_error_context",
    "get_degradation_handler",
    "handle_validation_error",
    "retry_config_operation",
    
    # Legacy aliases
    "UnifiedConfig",
]
