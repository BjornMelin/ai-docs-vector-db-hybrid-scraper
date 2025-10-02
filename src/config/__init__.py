"""Public configuration API for the AI documentation platform."""

# pylint: disable=global-statement

from __future__ import annotations

from typing import Any

from .drift import (
    ConfigDriftDetector,
    DriftSeverity,
    get_drift_detector,
    get_drift_summary,
    initialize_drift_detector,
    run_drift_detection,
)
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
from .manager import (
    ConfigManager,
    GracefulDegradationHandler,
    get_degradation_handler as _get_degradation_handler,
)
from .models import (
    ABTestVariant,
    ApplicationMode,
    AutoDetectedServices,
    AutoDetectionConfig,
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
    DetectedEnvironment,
    DetectedService,
    DocumentationSite,
    DocumentStatus,
    DriftDetectionConfig,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    EnvironmentDetector,
    FastEmbedConfig,
    FirecrawlConfig,
    FusionAlgorithm,
    HyDEConfig,
    LogLevel,
    ModelType,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    OptimizationStrategy,
    PerformanceConfig,
    PlaywrightConfig,
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
    TaskQueueConfig,
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


async def get_config_with_auto_detection(*, force_reload: bool = False) -> Config:
    """Return configuration after running lightweight auto-detection."""

    config = get_config(force_reload=force_reload)
    if not config.auto_detection.enabled:
        config.set_auto_detected_services(None)
        return config

    detector = EnvironmentDetector(config=config.auto_detection)
    environment = await detector.detect()
    autodetected = AutoDetectedServices(environment=environment)
    config.set_auto_detected_services(autodetected)
    return config


_reloader_instance: ConfigReloader | None = None
get_degradation_handler = _get_degradation_handler


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
    "AutoDetectionConfig",
    "AutoDetectedServices",
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
    "DetectedEnvironment",
    "DetectedService",
    "DocumentStatus",
    "DocumentationSite",
    "DriftDetectionConfig",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "EnvironmentDetector",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "FusionAlgorithm",
    "HyDEConfig",
    "LogLevel",
    "ModelType",
    "MonitoringConfig",
    "ObservabilityConfig",
    "OpenAIConfig",
    "OptimizationStrategy",
    "PerformanceConfig",
    "PlaywrightConfig",
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
    "TaskQueueConfig",
    "VectorType",
    "ConfigManager",
    "GracefulDegradationHandler",
    "ConfigAccessLevel",
    "ConfigDataClassification",
    "ConfigOperationType",
    "ConfigurationAuditEvent",
    "EncryptedConfigItem",
    "SecureConfigManager",
    "SecurityConfig",
    "ConfigDriftDetector",
    "DriftSeverity",
    "get_drift_detector",
    "get_drift_summary",
    "initialize_drift_detector",
    "run_drift_detection",
    "get_config_reloader",
    "set_config_reloader",
    "get_config_with_auto_detection",
    "get_degradation_handler",
    "ConfigBackup",
    "ConfigError",
    "ConfigLoadError",
    "ConfigReloadError",
    "ConfigReloader",
    "ReloadOperation",
    "ReloadStatus",
    "ReloadTrigger",
]
