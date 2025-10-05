"""Configuration loading utilities built on Pydantic settings."""

# pylint: disable=no-member, global-statement

from __future__ import annotations

import threading
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import Field, model_validator
from pydantic.fields import ModelPrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import (
    ApplicationMode,
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    ChunkingStrategy,
    Crawl4AIConfig,
    CrawlProvider,
    DatabaseConfig,
    DeploymentConfig,
    DocumentationSite,
    EmbeddingConfig,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    LogLevel,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    PlaywrightConfig,
    QdrantConfig,
    QueryProcessingConfig,
    RAGConfig,
    ReRankingConfig,
    SearchStrategy,
)
from .security.config import SecurityConfig


class Config(BaseSettings):
    """Unified application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        env_ignore_empty=True,
        arbitrary_types_allowed=True,
    )
    model_private_attrs: ClassVar[dict[str, ModelPrivateAttr]] = {}

    # Core application metadata
    app_name: str = Field(
        default="AI Documentation Vector DB", description="Application name"
    )
    version: str = Field(default="1.0.0", description="Application version")
    mode: ApplicationMode = Field(
        default=ApplicationMode.SIMPLE, description="Application mode"
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug features")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")

    # Provider selection
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.FASTEMBED, description="Embedding provider"
    )
    crawl_provider: CrawlProvider = Field(
        default=CrawlProvider.CRAWL4AI, description="Crawling provider"
    )

    # Simple mode conveniences
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Default Qdrant URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379", description="Default Redis URL"
    )
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    firecrawl_api_key: str | None = Field(default=None, description="Firecrawl API key")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    # Nested configuration sections
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache configuration"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig, description="Qdrant configuration"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="OpenAI configuration"
    )
    fastembed: FastEmbedConfig = Field(
        default_factory=FastEmbedConfig, description="FastEmbed configuration"
    )
    firecrawl: FirecrawlConfig = Field(
        default_factory=FirecrawlConfig, description="Firecrawl configuration"
    )
    crawl4ai: Crawl4AIConfig = Field(
        default_factory=Crawl4AIConfig, description="Crawl4AI configuration"
    )
    playwright: PlaywrightConfig = Field(
        default_factory=PlaywrightConfig, description="Playwright configuration"
    )
    browser_use: BrowserUseConfig = Field(
        default_factory=BrowserUseConfig, description="browser-use configuration"
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Document chunking settings"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding configuration"
    )
    hyde: HyDEConfig = Field(
        default_factory=HyDEConfig, description="HyDE configuration"
    )
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    reranking: ReRankingConfig = Field(
        default_factory=ReRankingConfig, description="Re-ranking configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )
    query_processing: QueryProcessingConfig = Field(
        default_factory=QueryProcessingConfig,
        description="Query processing configuration",
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="Observability configuration"
    )
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig, description="Deployment configuration"
    )
    documentation_sites: list[DocumentationSite] = Field(
        default_factory=list, description="Documentation sites to crawl"
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> Self:
        if self.environment == Environment.TESTING:
            return self
        if (
            self.embedding_provider == EmbeddingProvider.OPENAI
            and not self.openai_api_key
            and not self.openai.api_key
        ):
            msg = "OpenAI API key required when using OpenAI embedding provider"
            raise ValueError(msg)
        if (
            self.crawl_provider == CrawlProvider.FIRECRAWL
            and not self.firecrawl_api_key
            and not self.firecrawl.api_key
        ):
            msg = "Firecrawl API key required when using Firecrawl provider"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def sync_api_keys(self) -> Self:
        if self.openai_api_key:
            self.openai.api_key = self.openai_api_key
        if self.firecrawl_api_key:
            self.firecrawl.api_key = self.firecrawl_api_key
        if self.qdrant_api_key:
            self.qdrant.api_key = self.qdrant_api_key
        return self

    @model_validator(mode="after")
    def sync_service_urls(self) -> Self:
        self.qdrant.url = self.qdrant_url
        self.cache.redis_url = self.redis_url
        return self

    @model_validator(mode="after")
    def configure_by_mode(self) -> Self:
        if self.mode == ApplicationMode.SIMPLE:
            self.performance.max_concurrent_crawls = min(
                self.performance.max_concurrent_crawls, 10
            )
            self.cache.local_max_memory_mb = min(self.cache.local_max_memory_mb, 200)
            self.reranking.enabled = False
            self.observability.enabled = False
        elif self.mode == ApplicationMode.ENTERPRISE:
            self.performance.max_concurrent_crawls = min(
                self.performance.max_concurrent_crawls, 50
            )
        return self

    @model_validator(mode="after")
    def create_directories(self) -> Self:
        for directory in (self.data_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    # Convenience helpers -------------------------------------------------

    def is_enterprise_mode(self) -> bool:
        return self.mode == ApplicationMode.ENTERPRISE

    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    def get_effective_chunking_strategy(self) -> ChunkingStrategy:
        if self.mode == ApplicationMode.SIMPLE:
            return ChunkingStrategy.BASIC
        return self.chunking.strategy

    def get_effective_search_strategy(self) -> SearchStrategy:
        if self.mode == ApplicationMode.SIMPLE:
            return SearchStrategy.DENSE
        return SearchStrategy.HYBRID


_config_lock = threading.Lock()
_config_instance: Config | None = None


def load_config(**overrides: Any) -> Config:
    """Load configuration from the environment with optional overrides."""

    return Config(**overrides)


def _refresh_observability(_config: Config) -> None:
    try:
        observability_module = import_module("src.services.observability.config")
        sync_observability_config = getattr(
            observability_module, "get_observability_config", None
        )
        if sync_observability_config is not None:
            sync_observability_config(force_refresh=True)
    except ImportError:  # pragma: no cover - optional dependency
        pass


def get_config(force_reload: bool = False) -> Config:
    """Return the cached configuration instance."""

    global _config_instance
    if force_reload or _config_instance is None:
        with _config_lock:
            if force_reload or _config_instance is None:
                _config_instance = load_config()
                _refresh_observability(_config_instance)
    return _config_instance


def set_config(new_config: Config) -> None:
    """Replace the cached configuration instance."""

    global _config_instance
    with _config_lock:
        _config_instance = new_config
        _refresh_observability(new_config)


def reset_config() -> None:
    """Clear the cached configuration instance."""

    global _config_instance
    with _config_lock:
        _config_instance = None


def get_cache_config() -> CacheConfig:
    return get_config().cache


def get_qdrant_config() -> QdrantConfig:
    return get_config().qdrant


def get_embedding_config() -> EmbeddingConfig:
    return get_config().embedding


def get_performance_config() -> PerformanceConfig:
    return get_config().performance


def get_openai_config() -> OpenAIConfig:
    return get_config().openai


def get_security_config() -> SecurityConfig:
    return get_config().security


def create_simple_config() -> Config:
    return load_config(mode=ApplicationMode.SIMPLE)


def create_enterprise_config() -> Config:
    return load_config(mode=ApplicationMode.ENTERPRISE)


def create_config_from_env() -> Config:
    return load_config()


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
]
