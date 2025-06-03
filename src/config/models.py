"""Unified configuration system using Pydantic v2 and pydantic-settings.

This module provides a comprehensive configuration system that consolidates all
settings across the application into a single, well-structured configuration model.
"""

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

from .enums import CacheType
from .enums import ChunkingStrategy
from .enums import CrawlProvider
from .enums import EmbeddingModel
from .enums import EmbeddingProvider
from .enums import Environment
from .enums import LogLevel
from .enums import SearchAccuracy
from .enums import SearchStrategy
from .enums import VectorType

# Import validators from config package (avoiding circular imports)
from .validators import validate_api_key_common
from .validators import validate_chunk_sizes
from .validators import validate_model_benchmark_consistency
from .validators import validate_rate_limit_config
from .validators import validate_scoring_weights
from .validators import validate_url_format


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
    enable_dragonfly_cache: bool = Field(
        default=True, description="Enable DragonflyDB cache"
    )
    dragonfly_url: str = Field(
        default="redis://localhost:6379",
        description="DragonflyDB connection URL (Redis-compatible)",
    )

    # Cache key patterns for different data types
    cache_key_patterns: dict[CacheType, str] = Field(
        default_factory=lambda: {
            CacheType.EMBEDDINGS: "embeddings:{model}:{hash}",
            CacheType.CRAWL: "crawl:{url_hash}",
            CacheType.SEARCH: "search:{query_hash}",
            CacheType.HYDE: "hyde:{query_hash}",
        },
        description="Cache key patterns for different data types",
    )

    # TTL settings (in seconds) by cache type
    cache_ttl_seconds: dict[CacheType, int] = Field(
        default_factory=lambda: {
            CacheType.EMBEDDINGS: 86400,  # 24 hours
            CacheType.CRAWL: 3600,  # 1 hour
            CacheType.SEARCH: 7200,  # 2 hours
            CacheType.HYDE: 3600,  # 1 hour
        },
        description="TTL settings by cache type",
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


class HNSWConfig(BaseModel):
    """HNSW index configuration for different collection types."""

    m: int = Field(
        default=16, gt=0, le=64, description="HNSW M parameter (connections per node)"
    )
    ef_construct: int = Field(
        default=200, gt=0, le=1000, description="HNSW ef_construct parameter"
    )
    full_scan_threshold: int = Field(
        default=10000, gt=0, description="When to use full scan instead of HNSW"
    )
    max_indexing_threads: int = Field(
        default=0, ge=0, description="Max threads for indexing (0 = auto)"
    )

    # Runtime ef recommendations
    min_ef: int = Field(default=50, gt=0, description="Minimum ef value for searches")
    balanced_ef: int = Field(default=100, gt=0, description="Balanced ef value")
    max_ef: int = Field(default=200, gt=0, description="Maximum ef value for searches")

    # Adaptive ef settings
    enable_adaptive_ef: bool = Field(
        default=True, description="Enable adaptive ef selection based on time budget"
    )
    default_time_budget_ms: int = Field(
        default=100, gt=0, description="Default time budget for adaptive ef (ms)"
    )

    model_config = ConfigDict(extra="forbid")


class CollectionHNSWConfigs(BaseModel):
    """Collection-specific HNSW configurations."""

    api_reference: HNSWConfig = Field(
        default_factory=lambda: HNSWConfig(
            m=20,
            ef_construct=300,
            full_scan_threshold=5000,
            min_ef=100,
            balanced_ef=150,
            max_ef=200,
        ),
        description="High accuracy for API documentation",
    )

    tutorials: HNSWConfig = Field(
        default_factory=lambda: HNSWConfig(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,
            min_ef=75,
            balanced_ef=100,
            max_ef=150,
        ),
        description="Balanced for tutorial content",
    )

    blog_posts: HNSWConfig = Field(
        default_factory=lambda: HNSWConfig(
            m=12,
            ef_construct=150,
            full_scan_threshold=20000,
            min_ef=50,
            balanced_ef=75,
            max_ef=100,
        ),
        description="Fast for blog content",
    )

    code_examples: HNSWConfig = Field(
        default_factory=lambda: HNSWConfig(
            m=18,
            ef_construct=250,
            full_scan_threshold=8000,
            min_ef=100,
            balanced_ef=125,
            max_ef=175,
        ),
        description="Code-specific optimization",
    )

    general: HNSWConfig = Field(
        default_factory=lambda: HNSWConfig(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,
            min_ef=75,
            balanced_ef=100,
            max_ef=150,
        ),
        description="Default balanced configuration",
    )

    model_config = ConfigDict(extra="forbid")


class VectorSearchConfig(BaseModel):
    """Vector search configuration with accuracy parameters and prefetch settings."""

    # Search accuracy level defaults mapped to HNSW parameters
    search_accuracy_params: dict[SearchAccuracy, dict[str, int | bool]] = Field(
        default_factory=lambda: {
            SearchAccuracy.FAST: {"ef": 50, "exact": False},
            SearchAccuracy.BALANCED: {"ef": 100, "exact": False},
            SearchAccuracy.ACCURATE: {"ef": 200, "exact": False},
            SearchAccuracy.EXACT: {"exact": True},
        },
        description="HNSW parameters for different accuracy levels",
    )

    # Prefetch multipliers by vector type for optimal performance
    prefetch_multipliers: dict[VectorType, float] = Field(
        default_factory=lambda: {
            VectorType.DENSE: 2.0,
            VectorType.SPARSE: 5.0,
            VectorType.HYDE: 3.0,
        },
        description="Multipliers for prefetch calculations by vector type",
    )

    # Maximum prefetch limits to prevent performance degradation
    max_prefetch_limits: dict[VectorType, int] = Field(
        default_factory=lambda: {
            VectorType.DENSE: 200,
            VectorType.SPARSE: 500,
            VectorType.HYDE: 150,
        },
        description="Maximum prefetch limits by vector type",
    )

    # Default search settings
    default_search_limit: int = Field(
        default=10, gt=0, description="Default search limit"
    )
    max_search_limit: int = Field(default=100, gt=0, description="Maximum search limit")

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

    # Vector quantization settings
    quantization_enabled: bool = Field(
        default=True, description="Enable vector quantization"
    )

    # Collection-specific HNSW configurations
    collection_hnsw_configs: CollectionHNSWConfigs = Field(
        default_factory=CollectionHNSWConfigs,
        description="Collection-specific HNSW configurations",
    )

    # Enable HNSW optimization features
    enable_hnsw_optimization: bool = Field(
        default=True, description="Enable HNSW parameter optimization"
    )

    # Vector search configuration
    vector_search: VectorSearchConfig = Field(
        default_factory=VectorSearchConfig,
        description="Vector search accuracy and prefetch configuration",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        return validate_url_format(v)


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
        return validate_api_key_common(
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
        return validate_api_key_common(
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

    # Viewport settings
    viewport: dict[str, int] = Field(
        default_factory=lambda: {"width": 1920, "height": 1080},
        description="Browser viewport dimensions",
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


class BrowserUseConfig(BaseModel):
    """Configuration for BrowserUse adapter with AI-powered automation."""

    # LLM provider settings
    llm_provider: str = Field(
        default="openai", description="LLM provider (openai, anthropic, gemini)"
    )
    model: str = Field(
        default="gpt-4o-mini", description="LLM model to use for automation"
    )

    # Browser settings
    headless: bool = Field(default=True, description="Run browser in headless mode")
    disable_security: bool = Field(
        default=False, description="Disable browser security features"
    )
    generate_gif: bool = Field(
        default=False, description="Generate GIFs of automation process"
    )

    # Performance settings
    timeout: int = Field(
        default=30000, gt=0, description="Default timeout in milliseconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    max_steps: int = Field(
        default=20, gt=0, le=100, description="Maximum steps for automation"
    )

    model_config = ConfigDict(extra="forbid")


class PlaywrightConfig(BaseModel):
    """Configuration for Playwright adapter with direct browser control."""

    # Browser settings
    browser: str = Field(
        default="chromium", description="Browser type (chromium, firefox, webkit)"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")

    # Viewport settings
    viewport: dict[str, int] = Field(
        default_factory=lambda: {"width": 1920, "height": 1080},
        description="Browser viewport dimensions",
    )

    # User agent
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; AIDocs/1.0; +https://github.com/ai-docs)",
        description="User agent string for browser",
    )

    # Timeout settings
    timeout: int = Field(
        default=30000, gt=0, description="Default timeout in milliseconds"
    )

    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(BaseModel):
    """Advanced chunking configuration for optimal RAG performance."""

    # Basic parameters
    chunk_size: int = Field(
        default=1600, gt=0, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=320, ge=0, description="Overlap between chunks (characters)"
    )

    # Strategy settings
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy to use"
    )
    enable_ast_chunking: bool = Field(
        default=True, description="Enable AST-based chunking when available"
    )
    preserve_function_boundaries: bool = Field(
        default=True, description="Keep functions intact across chunks"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Keep code blocks intact when possible"
    )
    max_function_chunk_size: int = Field(
        default=3200, gt=0, description="Maximum size for a single function chunk"
    )

    # Language detection and support
    supported_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "markdown"],
        description="Languages supported for AST parsing",
    )
    fallback_to_text_chunking: bool = Field(
        default=True, description="Fall back to text chunking if AST fails"
    )
    detect_language: bool = Field(
        default=True, description="Auto-detect programming language"
    )
    include_function_context: bool = Field(
        default=True, description="Include function signatures in adjacent chunks"
    )

    # Size constraints
    min_chunk_size: int = Field(
        default=100, gt=0, description="Minimum chunk size in characters"
    )
    max_chunk_size: int = Field(
        default=3000, gt=0, description="Maximum chunk size in characters"
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_chunk_sizes_relationships(self) -> "ChunkingConfig":
        """Validate chunk size relationships."""
        # Use centralized validation for common chunk size relationships
        validate_chunk_sizes(
            self.chunk_size,
            self.chunk_overlap,
            self.min_chunk_size,
            self.max_chunk_size,
        )
        # Additional validation for function chunk size specific to this model
        if self.max_function_chunk_size < self.max_chunk_size:
            raise ValueError("max_function_chunk_size should be >= max_chunk_size")
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

    # Rate limiting configuration
    default_rate_limits: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {
            "openai": {"max_calls": 500, "time_window": 60},  # 500/min
            "firecrawl": {"max_calls": 100, "time_window": 60},  # 100/min
            "crawl4ai": {"max_calls": 50, "time_window": 1},  # 50/sec
            "qdrant": {"max_calls": 100, "time_window": 1},  # 100/sec
        },
        description="Default rate limits by provider (max_calls per time_window seconds)",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("default_rate_limits")
    @classmethod
    def validate_rate_limits(
        cls, v: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """Validate rate limit configuration structure."""
        return validate_rate_limit_config(v)


class HyDEConfig(BaseModel):
    """Configuration for HyDE (Hypothetical Document Embeddings)."""

    # Feature flags
    enable_hyde: bool = Field(default=True, description="Enable HyDE processing")
    enable_fallback: bool = Field(
        default=True, description="Fall back to regular search on HyDE failure"
    )
    enable_reranking: bool = Field(
        default=True, description="Apply reranking to HyDE results"
    )
    enable_caching: bool = Field(
        default=True, description="Cache HyDE embeddings and results"
    )

    # Generation settings
    num_generations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of hypothetical documents to generate",
    )
    generation_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="LLM temperature for generation"
    )
    max_generation_tokens: int = Field(
        default=200, ge=50, le=500, description="Maximum tokens per generation"
    )
    generation_model: str = Field(
        default="gpt-3.5-turbo", description="LLM model for generation"
    )
    generation_timeout_seconds: int = Field(
        default=10, ge=1, le=60, description="Timeout for generation requests"
    )

    # Search settings
    hyde_prefetch_limit: int = Field(
        default=50, ge=10, le=200, description="Prefetch limit for HyDE embeddings"
    )
    query_prefetch_limit: int = Field(
        default=30, ge=10, le=100, description="Prefetch limit for original query"
    )
    hyde_weight_in_fusion: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight of HyDE in fusion"
    )
    fusion_algorithm: str = Field(
        default="rrf", description="Fusion algorithm (rrf or dbsf)"
    )

    # Caching settings
    cache_ttl_seconds: int = Field(
        default=3600, ge=300, le=86400, description="Cache TTL for HyDE embeddings"
    )
    cache_hypothetical_docs: bool = Field(
        default=True, description="Cache generated hypothetical documents"
    )
    cache_prefix: str = Field(default="hyde", description="Cache key prefix")

    # Performance settings
    parallel_generation: bool = Field(
        default=True, description="Generate documents in parallel"
    )
    max_concurrent_generations: int = Field(
        default=5, ge=1, le=10, description="Max concurrent generation requests"
    )

    # Prompt engineering
    use_domain_specific_prompts: bool = Field(
        default=True, description="Use domain-specific prompts"
    )
    prompt_variation: bool = Field(
        default=True, description="Use prompt variations for diversity"
    )

    # Quality control
    min_generation_length: int = Field(
        default=20, ge=10, le=100, description="Minimum words per generation"
    )
    filter_duplicates: bool = Field(
        default=True, description="Filter duplicate generations"
    )
    diversity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum diversity between generations"
    )

    # Monitoring and debugging
    log_generations: bool = Field(
        default=False, description="Log generated hypothetical documents"
    )
    track_metrics: bool = Field(
        default=True, description="Track HyDE performance metrics"
    )

    # A/B testing
    ab_testing_enabled: bool = Field(default=False, description="Enable A/B testing")
    control_group_percentage: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Percentage for control group"
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
        validate_scoring_weights(
            self.quality_weight, self.speed_weight, self.cost_weight
        )
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
            validate_model_benchmark_consistency(key, benchmark.model_name)
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


class TaskQueueConfig(BaseModel):
    """Task queue configuration for ARQ."""

    # Redis connection settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for task queue (uses DragonflyDB)",
    )
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_database: int = Field(
        default=1, ge=0, le=15, description="Redis database number for task queue"
    )

    # Worker settings
    max_jobs: int = Field(
        default=10, gt=0, description="Maximum concurrent jobs per worker"
    )
    job_timeout: int = Field(
        default=3600, gt=0, description="Default job timeout in seconds"
    )
    job_ttl: int = Field(
        default=86400, gt=0, description="Job result TTL in seconds (24 hours)"
    )

    # Retry settings
    max_tries: int = Field(
        default=3, gt=0, le=10, description="Maximum retry attempts for failed jobs"
    )
    retry_delay: float = Field(
        default=60.0, gt=0, description="Delay between retries in seconds"
    )

    # Queue settings
    queue_name: str = Field(default="default", description="Default queue name")
    health_check_interval: int = Field(
        default=60, gt=0, description="Health check interval in seconds"
    )

    # Worker pool settings
    worker_pool_size: int = Field(
        default=4, gt=0, description="Number of worker processes"
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
    browser_use: BrowserUseConfig = Field(
        default_factory=BrowserUseConfig,
        description="BrowserUse AI automation settings",
    )
    playwright: PlaywrightConfig = Field(
        default_factory=PlaywrightConfig,
        description="Playwright browser control settings",
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
    hyde: HyDEConfig = Field(default_factory=HyDEConfig, description="HyDE settings")
    task_queue: TaskQueueConfig = Field(
        default_factory=TaskQueueConfig, description="Task queue (ARQ) settings"
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
        # Exclude None values for TOML format
        data = self.model_dump(mode="json", exclude_none=(format == "toml"))

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

        # Check DragonflyDB/Redis if caching enabled
        if self.cache.enable_dragonfly_cache:
            try:
                import redis

                r = redis.from_url(self.cache.dragonfly_url)
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
