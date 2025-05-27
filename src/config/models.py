"""Unified configuration system using Pydantic v2 and pydantic-settings.

This module provides a comprehensive configuration system that consolidates all
settings across the application into a single, well-structured configuration model.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
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


def _validate_api_key_common(
    value: str | None,
    prefix: str,
    service_name: str,
    min_length: int = 10,
    max_length: int = 200,
    allowed_chars: str = r"[A-Za-z0-9-]+",
) -> str | None:
    """Common API key validation logic."""
    if value is None:
        return value

    value = value.strip()
    if not value:
        return None

    # Check for ASCII-only characters (security requirement)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as err:
        raise ValueError(
            f"{service_name} API key contains non-ASCII characters"
        ) from err

    # Check required prefix
    if not value.startswith(prefix):
        raise ValueError(f"{service_name} API key must start with '{prefix}'")

    # Length validation with DoS protection
    if len(value) < min_length:
        raise ValueError(f"{service_name} API key appears to be too short")

    if len(value) > max_length:
        raise ValueError(f"{service_name} API key appears to be too long")

    # Character validation
    if not re.match(f"^{re.escape(prefix)}{allowed_chars}$", value):
        raise ValueError(f"{service_name} API key contains invalid characters")

    return value


class ModelBenchmark(BaseModel):
    """Performance benchmark for embedding models."""

    model_name: str = Field(description="Model name")
    provider: str = Field(description="Provider name (openai, fastembed)")
    avg_latency_ms: float = Field(gt=0, description="Average latency in milliseconds")
    quality_score: float = Field(
        ge=0, le=100, description="Quality score 0-100 based on retrieval accuracy"
    )
    tokens_per_second: float = Field(
        gt=0, description="Processing speed in tokens/second"
    )
    cost_per_million_tokens: float = Field(
        ge=0, description="Cost per million tokens (0 for local models)"
    )
    max_context_length: int = Field(
        gt=0, description="Maximum context length in tokens"
    )
    embedding_dimensions: int = Field(gt=0, description="Embedding vector dimensions")

    model_config = ConfigDict(extra="forbid")


class CacheConfig(BaseModel):
    """Cache configuration settings."""

    enable_caching: bool = Field(default=True, description="Enable caching system")
    enable_local_cache: bool = Field(
        default=True, description="Enable local in-memory cache"
    )
    enable_redis_cache: bool = Field(default=True, description="Enable Redis cache")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )

    # TTL settings (in seconds)
    ttl_embeddings: int = Field(
        default=86400, ge=0, description="Embeddings cache TTL (24 hours)"
    )
    ttl_crawl: int = Field(default=3600, ge=0, description="Crawl cache TTL (1 hour)")
    ttl_queries: int = Field(
        default=7200, ge=0, description="Query cache TTL (2 hours)"
    )

    # Local cache limits
    local_max_size: int = Field(
        default=1000, gt=0, description="Max items in local cache"
    )
    local_max_memory_mb: float = Field(
        default=100.0, gt=0, description="Max memory for local cache (MB)"
    )

    # Redis settings
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_ssl: bool = Field(default=False, description="Use SSL for Redis connection")
    redis_pool_size: int = Field(
        default=10, gt=0, description="Redis connection pool size"
    )

    model_config = ConfigDict(extra="forbid")


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: str | None = Field(default=None, description="Qdrant API key")
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")
    prefer_grpc: bool = Field(default=False, description="Use gRPC instead of HTTP")
    collection_name: str = Field(
        default="documents", description="Default collection name"
    )

    # Performance settings
    batch_size: int = Field(
        default=100, gt=0, le=1000, description="Batch size for operations"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")

    # Index settings
    hnsw_ef_construct: int = Field(
        default=200, gt=0, description="HNSW ef_construct parameter"
    )
    hnsw_m: int = Field(default=16, gt=0, description="HNSW M parameter")
    quantization_enabled: bool = Field(
        default=True, description="Enable vector quantization"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v.rstrip("/")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str | None = Field(default=None, description="OpenAI API key")
    model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    dimensions: int = Field(
        default=1536, gt=0, le=3072, description="Embedding dimensions"
    )
    batch_size: int = Field(
        default=100, gt=0, le=2048, description="Batch size for embeddings"
    )

    # Rate limiting
    max_requests_per_minute: int = Field(
        default=3000, gt=0, description="Rate limit (requests/min)"
    )
    max_tokens_per_minute: int = Field(
        default=1000000, gt=0, description="Rate limit (tokens/min)"
    )

    # Cost tracking
    cost_per_million_tokens: float = Field(
        default=0.02, gt=0, description="Cost per million tokens"
    )
    budget_limit: float | None = Field(
        default=None, ge=0, description="Monthly budget limit"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate OpenAI API key format and structure."""
        return _validate_api_key_common(
            v,
            prefix="sk-",
            service_name="OpenAI",
            min_length=20,
            max_length=200,
            allowed_chars=r"[A-Za-z0-9-]+",
        )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate OpenAI model name."""
        valid_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if v not in valid_models:
            raise ValueError(f"Invalid OpenAI model. Must be one of: {valid_models}")
        return v


class FastEmbedConfig(BaseModel):
    """FastEmbed configuration for local embeddings."""

    model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="FastEmbed model name"
    )
    cache_dir: str | None = Field(default=None, description="Model cache directory")
    max_length: int = Field(default=512, gt=0, description="Max sequence length")
    batch_size: int = Field(default=32, gt=0, description="Batch size for embeddings")

    model_config = ConfigDict(extra="forbid")


class FirecrawlConfig(BaseModel):
    """Firecrawl API configuration."""

    api_key: str | None = Field(default=None, description="Firecrawl API key")
    api_url: str = Field(
        default="https://api.firecrawl.dev", description="Firecrawl API URL"
    )
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")

    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate Firecrawl API key format and structure."""
        return _validate_api_key_common(
            v,
            prefix="fc-",
            service_name="Firecrawl",
            min_length=10,
            max_length=200,
            allowed_chars=r"[A-Za-z0-9_-]+",
        )


class Crawl4AIConfig(BaseModel):
    """Crawl4AI configuration."""

    browser_type: str = Field(
        default="chromium", description="Browser type (chromium, firefox, webkit)"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    viewport_width: int = Field(
        default=1920, gt=0, description="Browser viewport width"
    )
    viewport_height: int = Field(
        default=1080, gt=0, description="Browser viewport height"
    )

    # Performance settings
    max_concurrent_crawls: int = Field(
        default=10, gt=0, le=50, description="Max concurrent crawls"
    )
    page_timeout: float = Field(
        default=30.0, gt=0, description="Page load timeout (seconds)"
    )
    wait_for_selector: str | None = Field(
        default=None, description="CSS selector to wait for"
    )

    # Content extraction
    remove_scripts: bool = Field(default=True, description="Remove script tags")
    remove_styles: bool = Field(default=True, description="Remove style tags")
    extract_links: bool = Field(default=True, description="Extract links from pages")

    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(
        default=1600, gt=0, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")

    # Code-aware chunking
    enable_ast_chunking: bool = Field(
        default=True, description="Enable AST-based chunking when available"
    )
    preserve_function_boundaries: bool = Field(
        default=True, description="Keep functions intact"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Keep code blocks intact when possible"
    )
    supported_languages: list[str] = Field(
        default=["python", "javascript", "typescript"],
        description="Languages for AST parsing",
    )

    # Advanced options
    min_chunk_size: int = Field(default=100, gt=0, description="Minimum chunk size")
    max_chunk_size: int = Field(default=3000, gt=0, description="Maximum chunk size")
    detect_language: bool = Field(
        default=True, description="Auto-detect programming language"
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ChunkingConfig":
        """Validate chunk size relationships."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        if self.chunk_size > self.max_chunk_size:
            raise ValueError("chunk_size cannot exceed max_chunk_size")
        return self


class DocumentationSite(BaseModel):
    """Documentation site configuration."""

    name: str = Field(description="Site name")
    url: HttpUrl = Field(description="Site URL")
    max_pages: int = Field(default=50, gt=0, description="Max pages to crawl")
    priority: str = Field(default="medium", description="Crawl priority")
    description: str | None = Field(default=None, description="Site description")

    # Crawl settings
    max_depth: int = Field(default=2, gt=0, description="Maximum crawl depth")
    crawl_pattern: str | None = Field(
        default=None, description="URL pattern for crawling"
    )
    exclude_patterns: list[str] = Field(
        default_factory=list, description="Patterns to exclude"
    )
    url_patterns: list[str] = Field(
        default_factory=lambda: [
            "*docs*",
            "*guide*",
            "*tutorial*",
            "*api*",
            "*reference*",
            "*concepts*",
        ],
        description="URL patterns to include",
    )

    model_config = ConfigDict(extra="forbid")


class PerformanceConfig(BaseModel):
    """Performance and optimization settings."""

    max_concurrent_requests: int = Field(
        default=10, gt=0, le=100, description="Max concurrent API requests"
    )
    request_timeout: float = Field(
        default=30.0, gt=0, description="Default request timeout"
    )

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_base_delay: float = Field(
        default=1.0, gt=0, description="Base delay for retries"
    )
    retry_max_delay: float = Field(
        default=60.0, gt=0, description="Max delay for retries"
    )

    # Memory management
    max_memory_usage_mb: float = Field(
        default=1000.0, gt=0, description="Max memory usage (MB)"
    )
    gc_threshold: float = Field(
        default=0.8, gt=0, le=1, description="GC trigger threshold"
    )

    model_config = ConfigDict(extra="forbid")


class SmartSelectionConfig(BaseModel):
    """Configuration for smart model selection algorithms."""

    # Token estimation
    chars_per_token: float = Field(
        default=4.0, gt=0, description="Character to token ratio"
    )

    # Scoring weights (must sum to 1.0)
    quality_weight: float = Field(
        default=0.4, ge=0, le=1, description="Quality weight in scoring"
    )
    speed_weight: float = Field(
        default=0.3, ge=0, le=1, description="Speed weight in scoring"
    )
    cost_weight: float = Field(
        default=0.3, ge=0, le=1, description="Cost weight in scoring"
    )

    # Quality thresholds (0-100 scale)
    quality_fast_threshold: float = Field(
        default=60.0, ge=0, le=100, description="Minimum quality for FAST tier"
    )
    quality_balanced_threshold: float = Field(
        default=75.0, ge=0, le=100, description="Minimum quality for BALANCED tier"
    )
    quality_best_threshold: float = Field(
        default=85.0, ge=0, le=100, description="Minimum quality for BEST tier"
    )

    # Speed thresholds (tokens/second)
    speed_fast_threshold: float = Field(
        default=500.0, gt=0, description="Minimum speed for fast classification"
    )
    speed_balanced_threshold: float = Field(
        default=200.0, gt=0, description="Minimum speed for balanced classification"
    )
    speed_slow_threshold: float = Field(
        default=100.0, gt=0, description="Maximum speed for slow classification"
    )

    # Cost thresholds (per million tokens)
    cost_cheap_threshold: float = Field(
        default=50.0, ge=0, description="Maximum cost for cheap classification"
    )
    cost_moderate_threshold: float = Field(
        default=100.0, ge=0, description="Maximum cost for moderate classification"
    )
    cost_expensive_threshold: float = Field(
        default=200.0, ge=0, description="Maximum cost for expensive classification"
    )

    # Budget management
    budget_warning_threshold: float = Field(
        default=0.8, gt=0, le=1, description="Budget warning threshold (80%)"
    )
    budget_critical_threshold: float = Field(
        default=0.9, gt=0, le=1, description="Budget critical threshold (90%)"
    )

    # Text analysis
    short_text_threshold: int = Field(
        default=100, gt=0, description="Short text threshold (characters)"
    )
    long_text_threshold: int = Field(
        default=2000, gt=0, description="Long text threshold (characters)"
    )

    # Code detection keywords
    code_keywords: set[str] = Field(
        default_factory=lambda: {
            "def",
            "class",
            "import",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "except",
            "function",
            "const",
            "let",
            "var",
            "public",
            "private",
        },
        description="Keywords for code detection",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "SmartSelectionConfig":
        """Validate that scoring weights sum to approximately 1.0."""
        total = self.quality_weight + self.speed_weight + self.cost_weight
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
        return self


def _get_default_model_benchmarks() -> dict[str, ModelBenchmark]:
    """Get default model benchmarks with research-backed values."""
    return {
        "text-embedding-3-small": ModelBenchmark(
            model_name="text-embedding-3-small",
            provider="openai",
            avg_latency_ms=78,
            quality_score=85,
            tokens_per_second=12800,
            cost_per_million_tokens=20.0,
            max_context_length=8191,
            embedding_dimensions=1536,
        ),
        "text-embedding-3-large": ModelBenchmark(
            model_name="text-embedding-3-large",
            provider="openai",
            avg_latency_ms=120,
            quality_score=92,
            tokens_per_second=8300,
            cost_per_million_tokens=130.0,
            max_context_length=8191,
            embedding_dimensions=3072,
        ),
        "BAAI/bge-small-en-v1.5": ModelBenchmark(
            model_name="BAAI/bge-small-en-v1.5",
            provider="fastembed",
            avg_latency_ms=45,
            quality_score=78,
            tokens_per_second=22000,
            cost_per_million_tokens=0.0,
            max_context_length=512,
            embedding_dimensions=384,
        ),
        "BAAI/bge-large-en-v1.5": ModelBenchmark(
            model_name="BAAI/bge-large-en-v1.5",
            provider="fastembed",
            avg_latency_ms=89,
            quality_score=88,
            tokens_per_second=11000,
            cost_per_million_tokens=0.0,
            max_context_length=512,
            embedding_dimensions=1024,
        ),
    }


class EmbeddingConfig(BaseModel):
    """Advanced embedding configuration."""

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI, description="Embedding provider selection"
    )
    dense_model: EmbeddingModel = Field(
        default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
        description="Dense embedding model (research: best cost-performance)",
    )
    sparse_model: EmbeddingModel | None = Field(
        default=None, description="Sparse embedding model for hybrid search"
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.DENSE, description="Vector search strategy"
    )
    enable_quantization: bool = Field(
        default=True,
        description="Enable vector quantization (83-99% storage reduction)",
    )
    matryoshka_dimensions: list[int] = Field(
        default_factory=lambda: [1536, 1024, 512, 256],
        description="Matryoshka embedding dimensions for cost optimization",
    )

    # Advanced Reranking Configuration
    enable_reranking: bool = Field(
        default=False,
        description="Enable reranking for 10-20% accuracy improvement",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model (research: optimal minimal complexity)",
    )
    rerank_top_k: int = Field(
        default=20,
        description="Retrieve top-k for reranking, return fewer after rerank",
    )

    # Model Benchmarks
    model_benchmarks: dict[str, ModelBenchmark] = Field(
        default_factory=lambda: _get_default_model_benchmarks(),
        description="Model benchmark data for smart provider selection",
    )

    # Smart Selection Configuration
    smart_selection: SmartSelectionConfig = Field(
        default_factory=SmartSelectionConfig,
        description="Smart model selection algorithm configuration",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_benchmark_keys(self) -> "EmbeddingConfig":
        """Ensure dict keys match ModelBenchmark.model_name for consistency."""
        for key, benchmark in self.model_benchmarks.items():
            if key != benchmark.model_name:
                raise ValueError(
                    f"Dictionary key '{key}' does not match "
                    f"ModelBenchmark.model_name '{benchmark.model_name}'. "
                    f"Keys must be consistent for proper model identification."
                )
        return self


class SecurityConfig(BaseModel):
    """Security settings."""

    allowed_domains: list[str] = Field(
        default_factory=list, description="Allowed domains for crawling"
    )
    blocked_domains: list[str] = Field(
        default_factory=list, description="Blocked domains"
    )

    # API security
    require_api_keys: bool = Field(
        default=True, description="Require API keys for endpoints"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100, gt=0, description="Requests per minute"
    )

    model_config = ConfigDict(extra="forbid")


class UnifiedConfig(BaseSettings):
    """Unified configuration for the AI Documentation Vector DB system.

    This configuration consolidates all settings across the application,
    providing a single source of truth for all configuration values.
    """

    # Environment settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")

    # Application settings
    app_name: str = Field(
        default="AI Documentation Vector DB", description="Application name"
    )
    version: str = Field(default="0.1.0", description="Application version")

    # Provider preferences
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.FASTEMBED, description="Preferred embedding provider"
    )
    crawl_provider: CrawlProvider = Field(
        default=CrawlProvider.CRAWL4AI, description="Preferred crawl provider"
    )

    # Component configurations
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache settings"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig, description="Qdrant settings"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="OpenAI settings"
    )
    fastembed: FastEmbedConfig = Field(
        default_factory=FastEmbedConfig, description="FastEmbed settings"
    )
    firecrawl: FirecrawlConfig = Field(
        default_factory=FirecrawlConfig, description="Firecrawl settings"
    )
    crawl4ai: Crawl4AIConfig = Field(
        default_factory=Crawl4AIConfig, description="Crawl4AI settings"
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Chunking settings"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding settings"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance settings"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security settings"
    )

    # Documentation sites
    documentation_sites: list[DocumentationSite] = Field(
        default_factory=list, description="List of documentation sites to crawl"
    )

    # File paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
        # Allow loading from multiple sources
        secrets_dir=None,  # Can be set via environment variable
        json_schema_extra={
            "example": {
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                "embedding_provider": "openai",
                "crawl_provider": "crawl4ai",
            }
        },
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "UnifiedConfig":
        """Validate that required API keys are present for selected providers."""
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
    def create_directories(self) -> "UnifiedConfig":
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def load_from_file(cls, config_path: Path | str) -> "UnifiedConfig":
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

    def save_to_file(self, config_path: Path | str, format: str = "json") -> None:
        """Save configuration to a file."""
        config_path = Path(config_path)
        data = self.model_dump(mode="json")

        if format == "json":
            import json

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        elif format in ["yaml", "yml"]:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif format == "toml":
            import tomli_w

            with open(config_path, "wb") as f:
                tomli_w.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_active_providers(self) -> dict[str, Any]:
        """Get configuration for active providers only."""
        providers = {}

        if self.embedding_provider == EmbeddingProvider.OPENAI:
            providers["embedding"] = self.openai
        else:
            providers["embedding"] = self.fastembed

        if self.crawl_provider == CrawlProvider.FIRECRAWL:
            providers["crawl"] = self.firecrawl
        else:
            providers["crawl"] = self.crawl4ai

        return providers

    def validate_completeness(self) -> list[str]:
        """Check for any missing required configuration."""
        issues = []

        # Check API keys
        if (
            self.embedding_provider == EmbeddingProvider.OPENAI
            and not self.openai.api_key
        ):
            issues.append("OpenAI API key is missing")
        if (
            self.crawl_provider == CrawlProvider.FIRECRAWL
            and not self.firecrawl.api_key
        ):
            issues.append("Firecrawl API key is missing")

        # Check Redis if caching enabled
        if self.cache.enable_redis_cache:
            try:
                import redis

                r = redis.from_url(self.cache.redis_url)
                r.ping()
            except Exception as e:
                issues.append(f"Redis connection failed: {e}")

        # Check Qdrant
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=self.qdrant.url, api_key=self.qdrant.api_key)
            client.get_collections()
        except Exception as e:
            issues.append(f"Qdrant connection failed: {e}")

        return issues


# Singleton instance
_config: UnifiedConfig | None = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = UnifiedConfig()
    return _config


def set_config(config: UnifiedConfig) -> None:
    """Set the global configuration instance."""
    global _config  # noqa: PLW0603
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config  # noqa: PLW0603
    _config = None
