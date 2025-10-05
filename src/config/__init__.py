"""Public configuration API for the AI documentation platform."""

# pylint: disable=global-statement

from __future__ import annotations

from typing import Any

from .loader import (
    Config,
    create_config_from_env,
    create_enterprise_config,
    create_simple_config,
    get_cache_config,
    get_config,
    get_embedding_config,
    get_openai_config,
    get_performance_config,
    get_qdrant_config,
    get_security_config,
    load_config,
    reset_config,
    set_config,
)
from .models import (
    ABTestVariant,
    ApplicationMode,
    BrowserUseConfig,
    CacheConfig,
    CacheType,
    ChunkingConfig,
    ChunkingStrategy,
    CircuitBreakerConfig,
    Crawl4AIConfig,
    CrawlProvider,
    DatabaseConfig,
    DeploymentConfig,
    DeploymentTier,
    DocumentationSite,
    DocumentStatus,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    FirecrawlConfig,
    FusionAlgorithm,
    HyDEConfig,
    LogLevel,
    MCPClientConfig,
    MCPServerConfig,
    MCPTransport,
    ModelType,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    OptimizationStrategy,
    PerformanceConfig,
    PlaywrightCaptchaSettings,
    PlaywrightConfig,
    PlaywrightProxySettings,
    PlaywrightTierConfig,
    QdrantConfig,
    QueryComplexity,
    QueryProcessingConfig,
    QueryType,
    RAGConfig,
    ReRankingConfig,
    ScoreNormalizationStrategy,
    SearchAccuracy,
    SearchMode,
    SearchPipeline,
    SearchStrategy,
    VectorType,
)
from .reloader import (
    ConfigBackup,
    ConfigError,
    ConfigLoadError,
    ConfigReloader,
    ConfigReloadError,
    ReloadOperation,
    ReloadStatus,
    ReloadTrigger,
)
from .security.config import (
    ConfigAccessLevel,
    ConfigDataClassification,
    ConfigOperationType,
    ConfigurationAuditEvent,
    EncryptedConfigItem,
    SecureConfigManager,
    SecurityConfig,
)


_reloader_instance: ConfigReloader | None = None


def get_config_reloader(**kwargs: Any) -> ConfigReloader:
    """Return the globally shared :class:`ConfigReloader` instance."""

    global _reloader_instance
    if _reloader_instance is None:
        _reloader_instance = ConfigReloader(**kwargs)
    return _reloader_instance


def set_config_reloader(reloader: ConfigReloader) -> None:
    """Override the globally shared :class:`ConfigReloader`."""

    global _reloader_instance
    _reloader_instance = reloader


__all__ = [
    "Config",
    "create_config_from_env",
    "create_enterprise_config",
    "create_simple_config",
    "get_cache_config",
    "get_config",
    "get_embedding_config",
    "get_openai_config",
    "get_performance_config",
    "get_qdrant_config",
    "get_security_config",
    "load_config",
    "reset_config",
    "set_config",
    "ABTestVariant",
    "ApplicationMode",
    "BrowserUseConfig",
    "CacheConfig",
    "CacheType",
    "ChunkingConfig",
    "ChunkingStrategy",
    "CircuitBreakerConfig",
    "Crawl4AIConfig",
    "CrawlProvider",
    "DatabaseConfig",
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
    "MCPClientConfig",
    "MCPServerConfig",
    "MCPTransport",
    "ModelType",
    "MonitoringConfig",
    "ObservabilityConfig",
    "OpenAIConfig",
    "OptimizationStrategy",
    "PerformanceConfig",
    "PlaywrightCaptchaSettings",
    "PlaywrightConfig",
    "PlaywrightProxySettings",
    "PlaywrightTierConfig",
    "QdrantConfig",
    "QueryProcessingConfig",
    "QueryComplexity",
    "QueryType",
    "RAGConfig",
    "ReRankingConfig",
    "SearchAccuracy",
    "SearchMode",
    "SearchPipeline",
    "SearchStrategy",
    "ScoreNormalizationStrategy",
    "VectorType",
    "ConfigAccessLevel",
    "ConfigDataClassification",
    "ConfigOperationType",
    "ConfigurationAuditEvent",
    "EncryptedConfigItem",
    "SecureConfigManager",
    "SecurityConfig",
    "get_config_reloader",
    "set_config_reloader",
    "ConfigBackup",
    "ConfigError",
    "ConfigLoadError",
    "ConfigReloadError",
    "ConfigReloader",
    "ReloadOperation",
    "ReloadStatus",
    "ReloadTrigger",
]
