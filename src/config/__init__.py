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
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    OpenAIConfig,
    PerformanceConfig,
    PlaywrightConfig,
    QdrantConfig,
    SecurityConfig,
    SQLAlchemyConfig,
    get_config,
    reset_config,
    set_config,
)

# Legacy aliases for backward compatibility
UnifiedConfig = Config

# Enums
from .enums import (
    CacheType,
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    LogLevel,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)

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
    "OpenAIConfig",
    "PerformanceConfig",
    "PlaywrightConfig",
    "QdrantConfig",
    "SecurityConfig",
    "SQLAlchemyConfig",
    
    # Enums
    "CacheType",
    "ChunkingStrategy",
    "CrawlProvider",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "LogLevel",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
]