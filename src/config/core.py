"""Core configuration using Pydantic Settings.

Consolidated configuration system following KISS principles and Pydantic best practices.
All configuration models in one place for V1 release.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class CacheConfig(BaseModel):
    """Simple cache configuration."""
    
    enable_caching: bool = Field(default=True)
    dragonfly_url: str = Field(default="redis://localhost:6379")
    local_max_size: int = Field(default=1000, gt=0)
    ttl_seconds: int = Field(default=3600, gt=0)


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    
    url: str = Field(default="http://localhost:6333")
    api_key: str | None = Field(default=None)
    timeout: float = Field(default=30.0, gt=0)
    collection_name: str = Field(default="documents")
    batch_size: int = Field(default=100, gt=0, le=1000)


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
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
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
        if self.embedding_provider == EmbeddingProvider.OPENAI and not self.openai.api_key:
            raise ValueError("OpenAI API key required when using OpenAI embedding provider")
        if self.crawl_provider == CrawlProvider.FIRECRAWL and not self.firecrawl.api_key:
            raise ValueError("Firecrawl API key required when using Firecrawl provider")
        return self
    
    @model_validator(mode="after")
    def create_directories(self) -> "Config":
        """Create required directories."""
        for dir_path in [self.data_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self


# Singleton pattern
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None