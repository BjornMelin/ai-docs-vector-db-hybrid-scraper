"""Progressive Configuration System for AI Documentation Vector DB.

This module provides a sophisticated configuration system with progressive complexity:

Features:
- Persona-based configuration builders (Development, Production, Research, Enterprise)
- Progressive disclosure of advanced features with guided discovery
- Intelligent auto-detection and smart defaults
- Advanced Pydantic v2 validation with helpful error messages
- Enterprise-grade configuration management patterns

Quick Start:
    # Simple configuration
    config = await quick_config("development")

    # Guided setup with discovery
    guide = await guided_config_setup()
    config = await guide.build_configuration("essential")

    # Advanced builder pattern
    builder = ConfigBuilderFactory.create_builder("enterprise")
    config = builder.build()

Portfolio showcase elements:
- Advanced Pydantic v2 validation patterns
- Sophisticated auto-detection and service discovery
- Enterprise-grade configuration management
- Clean API design hiding complex validation logic
"""

# Progressive configuration builders - NEW SHOWCASE FEATURES
from .builders import (
    BaseConfigBuilder,
    ConfigBuilderFactory,
    ConfigValidationError,
    DevelopmentConfigBuilder,
    EnterpriseConfigBuilder,
    ProductionConfigBuilder,
    ProgressiveConfigurationGuide,
    ResearchConfigBuilder,
    guided_config_setup,
    quick_config,
    validate_configuration_with_help,
)

# Core configuration models (backward compatibility)
from .core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    CircuitBreakerConfig,
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
    TaskQueueConfig,
    get_config_with_auto_detection,
    reset_config,
    set_config,
)

# Configuration discovery and intelligence - NEW SHOWCASE FEATURES
from .discovery import (
    ConfigurationOptimizer,
    ConfigurationRecommendation,
    ConfigurationValidationReport,
    IntelligentValidator,
    SystemEnvironment,
    discover_optimal_configuration,
    get_system_recommendations,
    validate_configuration_intelligently,
)

# Enums for validation (minimal subset actually used)
from .enums import (
    CacheType,
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    FusionAlgorithm,
    LogLevel,
    QueryComplexity,
    QueryType,
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


# Legacy type aliases (backward compatibility)
UnifiedConfig = Settings

__all__ = [
    # Progressive Configuration Builders - NEW SHOWCASE FEATURES
    "BaseConfigBuilder",
    "ConfigBuilderFactory",
    "ConfigValidationError",
    "DevelopmentConfigBuilder",
    "EnterpriseConfigBuilder",
    "ProductionConfigBuilder",
    "ProgressiveConfigurationGuide",
    "ResearchConfigBuilder",
    "guided_config_setup",
    "quick_config",
    "validate_configuration_with_help",
    # Configuration Discovery and Intelligence - NEW SHOWCASE FEATURES
    "ConfigurationOptimizer",
    "ConfigurationRecommendation",
    "ConfigurationValidationReport",
    "IntelligentValidator",
    "SystemEnvironment",
    "discover_optimal_configuration",
    "get_system_recommendations",
    "validate_configuration_intelligently",
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
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "FusionAlgorithm",
    "LogLevel",
    "QueryComplexity",
    "QueryType",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
    # Configuration models
    "BrowserUseConfig",
    "CacheConfig",
    "ChunkingConfig",
    "CircuitBreakerConfig",
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
    "TaskQueueConfig",
    # Auto-detection functions
    "get_config_with_auto_detection",
    # Backward compatibility
    "UnifiedConfig",
    "get_config",
    "reset_config",
    "set_config",
]
