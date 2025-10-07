"""Settings provider for the AI documentation platform."""

# pylint: disable=no-member

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.architecture.modes import ApplicationMode

from .models import (
    AgenticConfig,
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
    MCPClientConfig,
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


SettingsCallback = Callable[["Config", "Config | None"], None]


class Config(BaseSettings):
    """Normalized application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=False,
        env_ignore_empty=True,
        arbitrary_types_allowed=True,
    )

    # Core application metadata
    app_name: str = Field(
        default="AI Documentation Vector DB", description="Application name"
    )
    version: str = Field(default="1.0.0", description="Application version")
    mode: ApplicationMode = Field(
        default=ApplicationMode.SIMPLE, description="Deployment mode profile"
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug features")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    # Provider selection
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.FASTEMBED, description="Embedding provider"
    )
    crawl_provider: CrawlProvider = Field(
        default=CrawlProvider.CRAWL4AI, description="Crawling provider"
    )

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
    mcp_client: MCPClientConfig = Field(
        default_factory=cast(Callable[[], MCPClientConfig], MCPClientConfig),
        description="MCP client configuration",
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
    agentic: AgenticConfig = Field(
        default_factory=AgenticConfig, description="Agentic workflow configuration"
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
    def validate_provider_keys(self) -> Config:
        if self.environment == Environment.TESTING:
            return self
        if (
            self.embedding_provider is EmbeddingProvider.OPENAI
            and not self.openai.api_key
        ):
            msg = "OpenAI API key required when using OpenAI embedding provider"
            raise ValueError(msg)
        if (
            self.crawl_provider is CrawlProvider.FIRECRAWL
            and not self.firecrawl.api_key
        ):
            msg = "Firecrawl API key required when using Firecrawl provider"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def create_runtime_directories(self) -> Config:
        for directory in (self.data_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def apply_mode_adjustments(self) -> Config:
        if self.mode is ApplicationMode.SIMPLE:
            self.performance.max_concurrent_crawls = min(
                self.performance.max_concurrent_crawls, 10
            )
            self.cache.local_max_memory_mb = min(self.cache.local_max_memory_mb, 200)
            self.reranking.enabled = False
            self.observability.enabled = False
        elif self.mode is ApplicationMode.ENTERPRISE:
            self.performance.max_concurrent_crawls = min(
                self.performance.max_concurrent_crawls, 50
            )
        return self

    def is_enterprise_mode(self) -> bool:
        """Return True when the enterprise application mode is active."""

        return self.mode is ApplicationMode.ENTERPRISE

    def is_development(self) -> bool:
        """Return True when running in development environment."""

        return self.environment is Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Return True when running in production environment."""

        return self.environment is Environment.PRODUCTION

    def get_effective_chunking_strategy(self) -> ChunkingStrategy:
        """Return chunking strategy, forcing BASIC for simple mode."""

        if self.mode is ApplicationMode.SIMPLE:
            return ChunkingStrategy.BASIC
        return self.chunking.strategy

    def get_effective_search_strategy(self) -> SearchStrategy:
        """Return search strategy, forcing DENSE for simple mode."""

        if self.mode is ApplicationMode.SIMPLE:
            return SearchStrategy.DENSE
        return SearchStrategy.HYBRID


def ensure_runtime_directories(settings: Config) -> None:
    """Create runtime directories required by the application.

    Args:
        settings: Active settings instance whose directory attributes should exist.
    """

    for directory in (settings.data_dir, settings.cache_dir, settings.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


class SettingsProvider:
    """Thread-safe provider for application settings."""

    def __init__(self, *, factory: Callable[..., Config] | None = None) -> None:
        self._factory: Callable[..., Config] = factory or Config
        self._lock = threading.RLock()
        self._settings: Config | None = None
        self._callbacks: list[SettingsCallback] = []

    def get(
        self,
        *,
        force_reload: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Return the cached settings, reloading when requested.

        Args:
            force_reload: When ``True`` the provider rebuilds the settings model even
                if a cached instance exists.
            overrides: Optional field overrides applied when refreshing settings.

        Returns:
            The cached or newly constructed settings instance.
        """

        with self._lock:
            if force_reload or overrides or self._settings is None:
                overrides = overrides or {}
                settings = self._factory(**overrides)
                ensure_runtime_directories(settings)
                previous = self._settings
                self._settings = settings
                self._notify_listeners(settings, previous)
            return self._settings

    def set(self, settings: Config) -> None:
        """Replace the cached settings instance.

        Args:
            settings: Settings instance that becomes the new cached value.
        """

        with self._lock:
            ensure_runtime_directories(settings)
            previous = self._settings
            self._settings = settings
            self._notify_listeners(settings, previous)

    def reset(self) -> None:
        """Clear the cached settings instance."""

        with self._lock:
            self._settings = None

    def register_callback(self, callback: SettingsCallback) -> Callable[[], None]:
        """Register a callback invoked after settings are applied.

        Args:
            callback: Callable receiving the new settings and the previous instance.

        Returns:
            Callable that removes the callback when invoked.
        """

        with self._lock:
            self._callbacks.append(callback)

        def unregister() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unregister

    def _notify_listeners(
        self,
        new_settings: Config,
        previous_settings: Config | None,
    ) -> None:
        callbacks: Iterable[SettingsCallback] = list(self._callbacks)
        for callback in callbacks:
            callback(new_settings, previous_settings)


_SETTINGS_PROVIDER = SettingsProvider()


def load_config(**overrides: Any) -> Config:
    """Instantiate settings from the environment without caching.

    Args:
        **overrides: Field overrides passed directly to :class:`Config`.

    Returns:
        Newly constructed settings instance.
    """

    settings = Config(**overrides)
    ensure_runtime_directories(settings)
    return settings


def get_config(*, force_reload: bool = False, **overrides: Any) -> Config:
    """Return cached application settings.

    Args:
        force_reload: Reload the settings even if a cached copy exists.
        **overrides: Field overrides that force the provider to rebuild settings.

    Returns:
        Cached or newly built settings instance.
    """

    if overrides:
        return _SETTINGS_PROVIDER.get(force_reload=True, overrides=overrides)
    return _SETTINGS_PROVIDER.get(force_reload=force_reload)


def set_config(settings: Config) -> None:
    """Replace the cached settings instance.

    Args:
        settings: Settings instance that should become the cached value.
    """

    _SETTINGS_PROVIDER.set(settings)


def reset_config() -> None:
    """Clear the cached settings instance."""

    _SETTINGS_PROVIDER.reset()


def on_settings_applied(callback: SettingsCallback) -> Callable[[], None]:
    """Register a callback invoked after settings changes are applied.

    Args:
        callback: Callable invoked after settings refresh with the new and previous
            instances.

    Returns:
        Callable that unregisters the callback when invoked.
    """

    return _SETTINGS_PROVIDER.register_callback(callback)


__all__ = [
    "Config",
    "SettingsProvider",
    "ensure_runtime_directories",
    "get_config",
    "load_config",
    "on_settings_applied",
    "reset_config",
    "set_config",
]
