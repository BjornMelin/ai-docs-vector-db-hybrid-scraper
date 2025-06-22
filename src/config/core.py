"""Core configuration using Pydantic Settings.

Consolidated configuration system following KISS principles and Pydantic best practices.
All configuration models in one place for V1 release.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from .enums import ChunkingStrategy
from .enums import CrawlProvider
from .enums import EmbeddingModel
from .enums import EmbeddingProvider
from .enums import Environment
from .enums import LogLevel
from .enums import SearchStrategy


class CacheConfig(BaseModel):
    """Cache configuration with local and distributed options."""

    enable_caching: bool = Field(default=True)
    enable_local_cache: bool = Field(default=True)
    enable_dragonfly_cache: bool = Field(default=False)
    dragonfly_url: str = Field(default="redis://localhost:6379")
    local_max_size: int = Field(default=1000, gt=0)
    local_max_memory_mb: int = Field(default=100, gt=0)
    ttl_seconds: int = Field(default=3600, gt=0)
    cache_ttl_seconds: dict[str, int] = Field(
        default_factory=lambda: {
            "search_results": 3600,
            "embeddings": 86400,
            "collections": 7200,
        }
    )


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333")
    api_key: str | None = Field(default=None)
    timeout: float = Field(default=30.0, gt=0)
    collection_name: str = Field(default="documents")
    batch_size: int = Field(default=100, gt=0, le=1000)
    prefer_grpc: bool = Field(default=False)
    grpc_port: int = Field(default=6334, gt=0)


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str | None = Field(default=None)
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536, gt=0, le=3072)
    batch_size: int = Field(default=100, gt=0, le=2048)
    max_requests_per_minute: int = Field(default=3000, gt=0)
    cost_per_million_tokens: float = Field(default=0.02, gt=0)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class FastEmbedConfig(BaseModel):
    """FastEmbed local embeddings configuration."""

    model: str = Field(default="BAAI/bge-small-en-v1.5")
    cache_dir: str | None = Field(default=None)
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=32, gt=0)


class FirecrawlConfig(BaseModel):
    """Firecrawl API configuration."""

    api_key: str | None = Field(default=None)
    api_url: str = Field(default="https://api.firecrawl.dev")
    timeout: float = Field(default=30.0, gt=0)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        if v and not v.startswith("fc-"):
            raise ValueError("Firecrawl API key must start with 'fc-'")
        return v


class Crawl4AIConfig(BaseModel):
    """Crawl4AI configuration."""

    browser_type: str = Field(default="chromium")
    headless: bool = Field(default=True)
    max_concurrent_crawls: int = Field(default=10, gt=0, le=50)
    page_timeout: float = Field(default=30.0, gt=0)
    remove_scripts: bool = Field(default=True)
    remove_styles: bool = Field(default=True)


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    chunk_size: int = Field(default=1600, gt=0)
    chunk_overlap: int = Field(default=320, ge=0)
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.ENHANCED)
    min_chunk_size: int = Field(default=100, gt=0)
    max_chunk_size: int = Field(default=3000, gt=0)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")
        return self


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: EmbeddingProvider = Field(default=EmbeddingProvider.FASTEMBED)
    dense_model: EmbeddingModel = Field(default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
    search_strategy: SearchStrategy = Field(default=SearchStrategy.DENSE)
    enable_quantization: bool = Field(default=True)


class SecurityConfig(BaseModel):
    """Basic security settings."""

    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    require_api_keys: bool = Field(default=True)
    api_key_header: str = Field(default="X-API-Key")
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, gt=0)


class SQLAlchemyConfig(BaseModel):
    """Database configuration."""

    database_url: str = Field(default="sqlite+aiosqlite:///data/app.db")
    echo_queries: bool = Field(default=False)
    pool_size: int = Field(default=20, gt=0, le=100)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_timeout: float = Field(default=30.0, gt=0)


class PlaywrightConfig(BaseModel):
    """Playwright browser configuration."""

    browser: str = Field(default="chromium")
    headless: bool = Field(default=True)
    timeout: int = Field(default=30000, gt=0)


class BrowserUseConfig(BaseModel):
    """BrowserUse automation configuration."""

    llm_provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    headless: bool = Field(default=True)
    timeout: int = Field(default=30000, gt=0)


class HyDEConfig(BaseModel):
    """HyDE configuration."""

    enable_hyde: bool = Field(default=True)
    num_generations: int = Field(default=5, ge=1, le=10)
    generation_temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for external services."""

    # Core circuit breaker settings
    failure_threshold: int = Field(
        default=5, gt=0, le=20, description="Failures before opening circuit"
    )
    recovery_timeout: float = Field(
        default=60.0, gt=0, description="Seconds before attempting recovery"
    )
    half_open_max_calls: int = Field(
        default=3, gt=0, le=10, description="Max calls in half-open state"
    )

    # Advanced features
    enable_adaptive_timeout: bool = Field(
        default=True, description="Enable adaptive timeout adjustment"
    )
    enable_bulkhead_isolation: bool = Field(
        default=True, description="Enable bulkhead pattern isolation"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Enable circuit breaker metrics"
    )

    # Service-specific overrides
    service_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "openai": {"failure_threshold": 3, "recovery_timeout": 30.0},
            "firecrawl": {"failure_threshold": 5, "recovery_timeout": 60.0},
            "qdrant": {"failure_threshold": 3, "recovery_timeout": 15.0},
            "redis": {"failure_threshold": 2, "recovery_timeout": 10.0},
        }
    )


class PerformanceConfig(BaseModel):
    """Performance settings."""

    max_concurrent_requests: int = Field(default=10, gt=0, le=100)
    request_timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0)
    max_memory_usage_mb: float = Field(default=1000.0, gt=0)


class DocumentationSite(BaseModel):
    """Documentation site to crawl."""

    name: str = Field(...)
    url: HttpUrl = Field(...)
    max_pages: int = Field(default=50, gt=0)
    max_depth: int = Field(default=2, gt=0)
    priority: str = Field(default="medium")


class MonitoringConfig(BaseModel):
    """Basic monitoring configuration."""

    enable_metrics: bool = Field(default=False)
    enable_health_checks: bool = Field(default=True)
    metrics_port: int = Field(default=8001, gt=0, le=65535)


class ObservabilityConfig(BaseModel):
    """OpenTelemetry observability configuration."""

    # Core observability toggle
    enabled: bool = Field(default=False, description="Enable OpenTelemetry observability")
    
    # Service identification
    service_name: str = Field(default="ai-docs-vector-db", description="Service name for traces")
    service_version: str = Field(default="1.0.0", description="Service version")
    service_namespace: str = Field(default="ai-docs", description="Service namespace")
    
    # OTLP Exporter configuration
    otlp_endpoint: str = Field(
        default="http://localhost:4317", 
        description="OTLP gRPC endpoint for trace export"
    )
    otlp_headers: dict[str, str] = Field(
        default_factory=dict, 
        description="Headers for OTLP export"
    )
    otlp_insecure: bool = Field(default=True, description="Use insecure OTLP connection")
    
    # Sampling configuration
    trace_sample_rate: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Trace sampling rate (0.0-1.0)"
    )
    
    # AI/ML specific configuration
    track_ai_operations: bool = Field(
        default=True, 
        description="Track AI operations (embeddings, LLM calls)"
    )
    track_costs: bool = Field(default=True, description="Track AI service costs")
    
    # Instrumentation configuration
    instrument_fastapi: bool = Field(default=True, description="Auto-instrument FastAPI")
    instrument_httpx: bool = Field(default=True, description="Auto-instrument HTTP clients")
    instrument_redis: bool = Field(default=True, description="Auto-instrument Redis")
    instrument_sqlalchemy: bool = Field(default=True, description="Auto-instrument SQLAlchemy")
    
    # Console debugging (development)
    console_exporter: bool = Field(
        default=False, 
        description="Export traces to console (development)"
    )


class RAGConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration."""

    # Core settings
    enable_rag: bool = Field(default=False, description="Enable RAG answer generation")
    model: str = Field(
        default="gpt-3.5-turbo", description="LLM model for answer generation"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(
        default=1000, gt=0, le=4000, description="Maximum response tokens"
    )
    timeout_seconds: float = Field(default=30.0, gt=0, description="Generation timeout")

    # Context configuration
    max_context_length: int = Field(
        default=4000, gt=0, description="Max context length in tokens"
    )
    max_results_for_context: int = Field(
        default=5, gt=0, le=20, description="Max search results for context"
    )
    min_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Min confidence for answers"
    )

    # Features
    include_sources: bool = Field(default=True, description="Include source citations")
    include_confidence_score: bool = Field(
        default=True, description="Include confidence scoring"
    )
    enable_answer_metrics: bool = Field(
        default=True, description="Track answer quality metrics"
    )
    enable_caching: bool = Field(default=True, description="Enable answer caching")

    # Performance
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL")
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing"
    )


class DeploymentConfig(BaseModel):
    """Deployment tier and feature flag configuration."""

    # Tier settings
    tier: str = Field(
        default="enterprise",
        description="Deployment tier: personal, professional, enterprise",
    )

    # Feature flag integration
    enable_feature_flags: bool = Field(
        default=True, description="Enable feature flag management"
    )
    flagsmith_api_key: str | None = Field(default=None, description="Flagsmith API key")
    flagsmith_environment_key: str | None = Field(
        default=None, description="Flagsmith environment key"
    )
    flagsmith_api_url: str = Field(
        default="https://edge.api.flagsmith.com/api/v1/",
        description="Flagsmith API URL",
    )

    # Deployment services
    enable_deployment_services: bool = Field(
        default=True, description="Enable enterprise deployment services"
    )
    enable_ab_testing: bool = Field(
        default=True, description="Enable A/B testing capabilities"
    )
    enable_blue_green: bool = Field(
        default=True, description="Enable blue-green deployments"
    )
    enable_canary: bool = Field(default=True, description="Enable canary deployments")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    deployment_tier: str = Field(
        default="enterprise", description="Legacy deployment tier field"
    )

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        valid_tiers = {"personal", "professional", "enterprise"}
        if v.lower() not in valid_tiers:
            raise ValueError(f"Tier must be one of {valid_tiers}")
        return v.lower()

    @field_validator("flagsmith_api_key")
    @classmethod
    def validate_flagsmith_key(cls, v: str | None) -> str | None:
        if v and not (v.startswith("fs_") or v.startswith("env_")):
            raise ValueError("Flagsmith API key must start with 'fs_' or 'env_'")
        return v


class Config(BaseSettings):
    """Main application configuration.

    Consolidated configuration using Pydantic Settings best practices.
    Follows KISS principles with only essential settings for V1.
    """

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # App info
    app_name: str = Field(default="AI Documentation Vector DB")
    version: str = Field(default="0.1.0")

    # Provider preferences
    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.FASTEMBED)
    crawl_provider: CrawlProvider = Field(default=CrawlProvider.CRAWL4AI)

    # Component configs
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: SQLAlchemyConfig = Field(default_factory=SQLAlchemyConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    fastembed: FastEmbedConfig = Field(default_factory=FastEmbedConfig)
    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)
    crawl4ai: Crawl4AIConfig = Field(default_factory=Crawl4AIConfig)
    playwright: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    browser_use: BrowserUseConfig = Field(default_factory=BrowserUseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    # Documentation sites
    documentation_sites: list[DocumentationSite] = Field(default_factory=list)

    # File paths
    data_dir: Path = Field(default=Path("data"))
    cache_dir: Path = Field(default=Path("cache"))
    logs_dir: Path = Field(default=Path("logs"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Config":
        """Validate required API keys for selected providers."""
        if (
            self.embedding_provider == EmbeddingProvider.OPENAI
            and not self.openai.api_key
        ):
            raise ValueError(
                "OpenAI API key required when using OpenAI embedding provider"
            )
        if (
            self.crawl_provider == CrawlProvider.FIRECRAWL
            and not self.firecrawl.api_key
        ):
            raise ValueError("Firecrawl API key required when using Firecrawl provider")
        return self

    @model_validator(mode="after")
    def create_directories(self) -> "Config":
        """Create required directories."""
        for dir_path in [self.data_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def load_from_file(cls, config_path: Path | str) -> "Config":
        """Load configuration from a specific file."""
        config_path = Path(config_path)
        if config_path.suffix == ".json":
            import json

            with open(config_path) as f:
                data = json.load(f)
            return cls(**data)
        elif config_path.suffix in [".yaml", ".yml"]:
            import yaml

            with open(config_path) as f:
                data = yaml.safe_load(f)
            return cls(**data)
        elif config_path.suffix == ".toml":
            import tomllib

            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return cls(**data)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


# Singleton pattern
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config  # noqa: PLW0603
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config  # noqa: PLW0603
    _config = None
