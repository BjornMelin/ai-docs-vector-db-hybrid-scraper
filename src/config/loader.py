"""Application settings definitions and providers."""

# pylint: disable=global-statement

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import (  # pyright: ignore[reportMissingImports]
    Field,
    ValidationError,
    model_validator,
)
from pydantic_settings import (  # pyright: ignore[reportMissingImports]
    BaseSettings,
    SettingsConfigDict,
)

from src.config.browser import BrowserAutomationConfig

from .models import (
    AgenticConfig,
    CacheConfig,
    ChunkingConfig,
    ChunkingStrategy,
    CircuitBreakerConfig,
    CrawlProvider,
    DatabaseConfig,
    DeploymentConfig,
    DocumentationSite,
    EmbeddingConfig,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    HyDEConfig,
    LogLevel,
    MCPClientConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    QueryProcessingConfig,
    RAGConfig,
    ReRankingConfig,
    SearchStrategy,
)
from .security.config import SecurityConfig


class Settings(BaseSettings):
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
    mode: str = Field(default="production", description="Deployment mode label")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug features")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    enable_advanced_monitoring: bool = Field(
        default=True, description="Enable advanced monitoring features"
    )
    enable_deployment_features: bool = Field(
        default=True, description="Enable deployment and operations APIs"
    )
    enable_ab_testing: bool = Field(
        default=False, description="Enable A/B testing and experimentation"
    )

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
    browser: BrowserAutomationConfig = Field(
        default_factory=BrowserAutomationConfig,
        description="Browser automation configuration",
    )
    mcp_client: MCPClientConfig = Field(
        default_factory=MCPClientConfig,  # type: ignore[call-arg]
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
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
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
    def validate_provider_keys(self) -> Settings:
        if self.environment == Environment.TESTING:
            return self
        openai_api_key = getattr(self.openai, "api_key", None)
        firecrawl_settings = getattr(self.browser, "firecrawl", None)
        firecrawl_api_key = getattr(firecrawl_settings, "api_key", None)

        if self.embedding_provider is EmbeddingProvider.OPENAI and not openai_api_key:
            msg = "OpenAI API key required when using OpenAI embedding provider"
            raise ValueError(msg)
        if self.crawl_provider is CrawlProvider.FIRECRAWL and not firecrawl_api_key:
            msg = "Firecrawl API key required when using Firecrawl provider"
            raise ValueError(msg)
        return self

    def is_development(self) -> bool:
        """Return True when running in development environment."""

        return self.environment is Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Return True when running in production environment."""

        return self.environment is Environment.PRODUCTION

    def get_effective_chunking_strategy(self) -> ChunkingStrategy:
        """Return the configured chunking strategy."""

        return getattr(self.chunking, "strategy", ChunkingStrategy.BASIC)

    def get_effective_search_strategy(self) -> SearchStrategy:
        """Return the configured search strategy."""

        if hasattr(self.embedding, "retrieval_mode"):
            return self.embedding.retrieval_mode
        return SearchStrategy.DENSE

    def get_feature_flags(self) -> dict[str, bool]:
        """Return the active feature flags for the unified application."""

        return {
            "advanced_monitoring": self.enable_advanced_monitoring,
            "deployment_features": self.enable_deployment_features,
            "a_b_testing": self.enable_ab_testing,
            "comprehensive_observability": bool(
                getattr(self.observability, "enabled", False)
            ),
        }


def ensure_runtime_directories(settings: Settings) -> None:
    """Create runtime directories required by the application."""

    for directory in (settings.data_dir, settings.cache_dir, settings.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides: Any) -> Settings:
    """Instantiate settings from the environment without caching."""

    settings = Settings(**overrides)
    ensure_runtime_directories(settings)
    return settings


_ACTIVE_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings instance."""

    global _ACTIVE_SETTINGS
    if _ACTIVE_SETTINGS is None:
        _ACTIVE_SETTINGS = load_settings()
    return _ACTIVE_SETTINGS


def refresh_settings(
    *,
    settings: Settings | None = None,
    **overrides: Any,
) -> Settings:
    """Replace the cached settings instance.

    Args:
        settings: Optional pre-built settings instance to promote to the cache.
        **overrides: Keyword arguments forwarded to ``load_settings``.

    Returns:
        The newly cached settings instance.
    """

    global _ACTIVE_SETTINGS
    if settings is not None and overrides:
        msg = "Provide either a concrete settings instance or overrides, not both."
        raise ValueError(msg)
    if settings is not None:
        _ACTIVE_SETTINGS = settings
        return _ACTIVE_SETTINGS
    if overrides:
        _ACTIVE_SETTINGS = load_settings(**overrides)
        return _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = load_settings()
    return _ACTIVE_SETTINGS


def validate_settings_payload(
    payload: dict[str, Any], *, base: dict[str, Any] | None = None
) -> tuple[bool, list[str], Settings | None]:
    """Validate configuration data using the Settings model."""

    merged: dict[str, Any] = {
        "environment": Environment.DEVELOPMENT,
        "debug": False,
        "log_level": LogLevel.INFO,
    }
    if base:
        merged.update(base)
    merged.update(payload)

    try:
        settings = load_settings(**merged)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(part) for part in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
        return False, errors, None
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        return False, [str(exc)], None

    return True, [], settings


def load_settings_from_file(path: Path) -> Settings:
    """Load settings overrides from a JSON or YAML file."""

    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    try:
        if suffix in {".json"}:
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore import  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "Loading YAML configurations requires PyYAML. "
                    "Install with `pip install pyyaml`."
                ) from exc
            payload = yaml.safe_load(text)
        else:
            msg = f"Unsupported configuration file format: {path.suffix}"
            raise ValueError(msg)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration file: {exc}") from exc

    if not isinstance(payload, dict):
        msg = "Configuration file must define a JSON/YAML object"
        raise ValueError(msg)

    return load_settings(**payload)


__all__ = [
    "Settings",
    "ensure_runtime_directories",
    "get_settings",
    "load_settings",
    "load_settings_from_file",
    "refresh_settings",
    "validate_settings_payload",
]
