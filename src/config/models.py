"""Typed configuration models for the AI docs platform."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


#### Enumerations ####


class ApplicationMode(str, Enum):
    """High-level execution modes."""

    SIMPLE = "simple"
    ENTERPRISE = "enterprise"


class Environment(str, Enum):
    """Deployment environments supported by the platform."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Structured application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    """Embedding service providers."""

    OPENAI = "openai"
    FASTEMBED = "fastembed"


class EmbeddingModel(str, Enum):
    """Embedding model catalogue."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    NV_EMBED_V2 = "nvidia/nv-embed-v2"


class CrawlProvider(str, Enum):
    """Web crawling backends."""

    FIRECRAWL = "firecrawl"
    CRAWL4AI = "crawl4ai"
    PLAYWRIGHT = "playwright"


class ChunkingStrategy(str, Enum):
    """Content chunking strategies."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    AST_AWARE = "ast_aware"


class SearchStrategy(str, Enum):
    """Vector search strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ScoreNormalizationStrategy(str, Enum):
    """Score normalization options for federated result merging."""

    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


class CacheType(str, Enum):
    """Cache group identifiers used by cache manager layers."""

    EMBEDDINGS = "embeddings"
    SEARCH = "search"
    CRAWL = "crawl"
    HYDE = "hyde"
    LOCAL = "local"
    REDIS = "redis"
    HYBRID = "hybrid"


class DocumentStatus(str, Enum):
    """Document ingestion lifecycle states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryComplexity(str, Enum):
    """Levels describing user query complexity."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ModelType(str, Enum):
    """Model roles inside the retrieval pipeline."""

    EMBEDDING = "embedding"
    RERANKING = "reranking"
    GENERATION = "generation"
    GENERAL_PURPOSE = "general_purpose"
    CODE_SPECIALIZED = "code_specialized"
    MULTIMODAL = "multimodal"


class VectorType(str, Enum):
    """Vector storage forms supported by storage engines."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYDE = "hyde"
    HYBRID = "hybrid"


class QueryType(str, Enum):
    """Query categories used for model selection."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GENERAL = "general"
    CODE = "code"
    MULTIMODAL = "multimodal"
    CONCEPTUAL = "conceptual"
    DOCUMENTATION = "documentation"
    API_REFERENCE = "api_reference"
    TROUBLESHOOTING = "troubleshooting"


class SearchAccuracy(str, Enum):
    """Search accuracy tuning profiles."""

    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"
    ACCURATE = "accurate"
    EXACT = "exact"


class FusionAlgorithm(str, Enum):
    """Result fusion algorithms for hybrid search."""

    RRF = "rrf"
    WEIGHTED = "weighted"
    NORMALIZED = "normalized"


class ABTestVariant(str, Enum):
    """A/B testing variants."""

    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"


class OptimizationStrategy(str, Enum):
    """Performance optimisation strategies."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"
    QUALITY_OPTIMIZED = "quality_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    COST_OPTIMIZED = "cost_optimized"


class SearchMode(str, Enum):
    """Composite search modes used by the API."""

    BASIC = "basic"
    SIMPLE = "simple"
    ENHANCED = "enhanced"
    INTELLIGENT = "intelligent"
    PERSONALIZED = "personalized"
    FULL = "full"


class SearchPipeline(str, Enum):
    """Predefined search pipelines."""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    PRECISION = "precision"


class DeploymentTier(str, Enum):
    """Deployment tiers available to the platform."""

    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


#### Configuration sections ####


class CacheConfig(BaseModel):
    """Cache configuration for local and distributed layers."""

    enable_caching: bool = Field(default=True, description="Enable caching globally")
    enable_local_cache: bool = Field(
        default=True,
        description=("Enable persistent on-disk cache used for warm restarts."),
    )
    enable_redis_cache: bool = Field(default=True, description="Enable Redis cache")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_database: int = Field(
        default=0, ge=0, le=15, description="Redis database number"
    )
    ttl_embeddings: int = Field(
        default=86400, gt=0, description="Embedding cache TTL in seconds"
    )
    ttl_crawl: int = Field(
        default=3600, gt=0, description="Crawl result cache TTL in seconds"
    )
    ttl_queries: int = Field(
        default=7200, gt=0, description="Query cache TTL in seconds"
    )
    ttl_search_results: int = Field(
        default=3600,
        gt=0,
        description="Search result cache TTL in seconds (negative caching supported)",
    )
    local_max_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum cached items persisted on disk before eviction",
    )
    local_max_memory_mb: int = Field(
        default=100, gt=0, description="Local cache memory budget in megabytes"
    )
    cache_ttl_seconds: dict[str, int] = Field(
        default_factory=lambda: {
            "search_results": 3600,
            "embeddings": 86400,
            "collections": 7200,
        },
        description="TTL overrides per cache type",
    )
    memory_pressure_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional ratio of memory usage that triggers evictions",
    )


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333", description="Qdrant HTTP URL")
    api_key: str | None = Field(default=None, description="Qdrant API key")
    timeout: float = Field(default=30.0, gt=0, description="Request timeout seconds")
    collection_name: str = Field(
        default="documents", description="Default collection name"
    )
    default_collection: str = Field(
        default="documentation",
        description="Legacy collection name retained for migration support",
    )
    batch_size: int = Field(
        default=100, gt=0, le=1000, description="Batch size for operations"
    )
    prefer_grpc: bool = Field(default=False, description="Prefer gRPC connections")
    grpc_port: int = Field(default=6334, gt=0, description="Qdrant gRPC port")
    use_grpc: bool = Field(default=False, description="Enable gRPC client")
    enable_grouping: bool = Field(
        default=True,
        description="Enable QueryPointGroups when the server supports it",
    )
    group_by_field: str = Field(
        default="doc_id",
        description="Payload field used for server-side grouping",
    )
    group_size: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum hits to return for each group",
    )
    groups_limit_multiplier: float = Field(
        default=2.0,
        gt=0,
        description="Multiplier applied to requested limit when grouping",
    )


class QueryProcessingConfig(BaseModel):
    """Query processing defaults for retrieval orchestration."""

    federated_overfetch_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Multiplier applied to per-collection result limits before merging",
    )
    enable_score_normalization: bool = Field(
        default=True,
        description="Normalize relevance scores prior to federated result merging",
    )
    score_normalization_strategy: ScoreNormalizationStrategy = Field(
        default=ScoreNormalizationStrategy.MIN_MAX,
        description="Normalization strategy used when harmonizing collection scores",
    )
    score_normalization_epsilon: float = Field(
        default=1e-6,
        gt=0.0,
        le=0.1,
        description="Minimum variance guard used when normalizing scores",
    )


class OpenAIConfig(BaseModel):
    """OpenAI embeddings configuration."""

    api_key: str | None = Field(default=None, description="OpenAI API key")
    model: str = Field(
        default="text-embedding-3-small", description="Default embedding model"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Explicit embedding model"
    )
    dimensions: int = Field(
        default=1536, gt=0, le=3072, description="Embedding dimensionality"
    )
    api_base: str | None = Field(default=None, description="Custom API base URL")
    batch_size: int = Field(
        default=100, gt=0, le=2048, description="Batch size for requests"
    )
    max_requests_per_minute: int = Field(
        default=3000, gt=0, description="Rate limit in requests per minute"
    )
    cost_per_million_tokens: float = Field(
        default=0.02, gt=0, description="Cost per million tokens"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, value: str | None) -> str | None:
        if value and not value.startswith("sk-"):
            msg = "OpenAI API key must start with 'sk-'"
            raise ValueError(msg)
        return value


class FastEmbedConfig(BaseModel):
    """FastEmbed local embeddings configuration."""

    model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="FastEmbed model name"
    )
    cache_dir: str | None = Field(default=None, description="Model cache directory")
    max_length: int = Field(default=512, gt=0, description="Max token length")
    batch_size: int = Field(default=32, gt=0, description="Batch size for processing")


class FirecrawlConfig(BaseModel):
    """Firecrawl API configuration."""

    api_key: str | None = Field(default=None, description="Firecrawl API key")
    api_url: str = Field(
        default="https://api.firecrawl.dev", description="Firecrawl API URL"
    )
    api_base: str = Field(
        default="https://api.firecrawl.dev", description="Alias for API base URL"
    )
    timeout: float = Field(default=30.0, gt=0, description="Request timeout seconds")

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, value: str | None) -> str | None:
        if value and not value.startswith("fc-"):
            msg = "Firecrawl API key must start with 'fc-'"
            raise ValueError(msg)
        return value


class Crawl4AIConfig(BaseModel):
    """Crawl4AI provider configuration."""

    browser_type: str = Field(default="chromium", description="Playwright browser type")
    headless: bool = Field(default=True, description="Run browser headless")
    max_concurrent_crawls: int = Field(
        default=10, gt=0, le=50, description="Max concurrent crawl tasks"
    )
    page_timeout: float = Field(default=30.0, gt=0, description="Page load timeout")
    remove_scripts: bool = Field(default=True, description="Remove script tags")
    remove_styles: bool = Field(default=True, description="Remove style tags")


class PlaywrightConfig(BaseModel):
    """Playwright browser configuration."""

    browser: str = Field(default="chromium", description="Browser type")
    headless: bool = Field(default=True, description="Run in headless mode")
    timeout: int = Field(default=30000, gt=0, description="Timeout in milliseconds")


class BrowserUseConfig(BaseModel):
    """Browser-use automation configuration."""

    llm_provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(
        default="gpt-4o-mini", description="Model used for browser automation"
    )
    headless: bool = Field(default=True, description="Run headless automations")
    timeout: int = Field(default=30000, gt=0, description="Timeout in milliseconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retries")
    max_steps: int = Field(
        default=20, ge=1, le=100, description="Maximum automation steps"
    )
    disable_security: bool = Field(
        default=False, description="Disable security features"
    )
    generate_gif: bool = Field(default=False, description="Generate GIF recordings")


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(default=1600, gt=0, description="Target chunk size")
    max_chunk_size: int = Field(default=3000, gt=0, description="Maximum chunk size")
    min_chunk_size: int = Field(default=100, gt=0, description="Minimum chunk size")
    chunk_overlap: int = Field(default=320, ge=0, description="Overlap between chunks")
    enable_ast_chunking: bool = Field(
        default=True, description="Enable AST-aware chunking"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Preserve code block integrity"
    )
    detect_language: bool = Field(
        default=True, description="Auto-detect programming language"
    )
    max_function_chunk_size: int = Field(
        default=2000, gt=0, description="Maximum chunk size for functions"
    )
    supported_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "markdown"],
        description="Languages with specialised chunking support",
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be less than chunk_size"
            raise ValueError(msg)
        if self.min_chunk_size > self.chunk_size:
            msg = "min_chunk_size must be <= chunk_size"
            raise ValueError(msg)
        if self.max_chunk_size < self.chunk_size:
            msg = "max_chunk_size must be >= chunk_size"
            raise ValueError(msg)
        return self


class EmbeddingConfig(BaseModel):
    """Embedding configuration including search strategy."""

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.FASTEMBED, description="Embedding provider"
    )
    dense_model: EmbeddingModel = Field(
        default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
        description="Dense embedding model",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.DENSE, description="Search strategy"
    )
    enable_quantization: bool = Field(
        default=True, description="Enable embedding quantization"
    )


class HyDEConfig(BaseModel):
    """HyDE configuration for synthetic document generation."""

    enable_hyde: bool = Field(default=True, description="Enable HyDE")
    model: str = Field(default="gpt-3.5-turbo", description="Model for HyDE generation")
    num_generations: int = Field(
        default=5, ge=1, le=10, description="Number of synthetic documents"
    )
    generation_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Generation temperature"
    )
    max_tokens: int = Field(default=150, gt=0, description="Max tokens per generation")
    cache_ttl: int = Field(default=3600, gt=0, description="Cache TTL in seconds")
    query_weight: float = Field(
        default=0.3, ge=0, le=1, description="Weight of original query"
    )


class ReRankingConfig(BaseModel):
    """Re-ranking configuration."""

    enabled: bool = Field(default=False, description="Enable re-ranking")
    model: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="Re-ranking model"
    )
    top_k: int = Field(default=20, gt=0, description="Number of results to re-rank")
    cache_ttl: int = Field(default=3600, gt=0, description="Cache TTL in seconds")
    batch_size: int = Field(default=32, gt=0, description="Batch size")


class PerformanceConfig(BaseModel):
    """Performance limits and retry policies."""

    max_concurrent_requests: int = Field(
        default=10, gt=0, le=100, description="Max concurrent API requests"
    )
    max_concurrent_crawls: int = Field(
        default=10, gt=0, le=50, description="Max concurrent crawls"
    )
    max_concurrent_embeddings: int = Field(
        default=32, gt=0, le=100, description="Max concurrent embeddings"
    )
    request_timeout: float = Field(default=30.0, gt=0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")
    retry_base_delay: float = Field(default=1.0, gt=0, description="Retry backoff base")
    max_memory_usage_mb: float = Field(
        default=1000.0, gt=0, description="Max memory usage allowance"
    )
    batch_embedding_size: int = Field(
        default=100, gt=0, le=2048, description="Embedding batch size"
    )
    batch_crawl_size: int = Field(default=50, gt=0, description="Crawl batch size")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker defaults and overrides."""

    failure_threshold: int = Field(
        default=5, gt=0, le=20, description="Failures before opening circuit"
    )
    recovery_timeout: float = Field(
        default=60.0, gt=0, description="Recovery timeout seconds"
    )
    half_open_max_calls: int = Field(
        default=3, gt=0, le=10, description="Max calls allowed in half-open"
    )
    enable_adaptive_timeout: bool = Field(
        default=True, description="Enable adaptive timeouts"
    )
    enable_bulkhead_isolation: bool = Field(
        default=True, description="Enable bulkhead isolation"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Collect circuit breaker metrics"
    )
    service_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "openai": {"failure_threshold": 3, "recovery_timeout": 30.0},
            "firecrawl": {"failure_threshold": 5, "recovery_timeout": 60.0},
            "qdrant": {"failure_threshold": 3, "recovery_timeout": 15.0},
            "redis": {"failure_threshold": 2, "recovery_timeout": 10.0},
        },
        description="Service-specific circuit breaker overrides",
    )


class DatabaseConfig(BaseModel):
    """Database connection settings."""

    database_url: str = Field(
        default="sqlite+aiosqlite:///data/app.db", description="Database URL"
    )
    echo_queries: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(default=20, gt=0, le=100, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Pool overflow")
    pool_timeout: float = Field(default=30.0, gt=0, description="Pool timeout seconds")


class MonitoringConfig(BaseModel):
    """Monitoring and readiness configuration."""

    enabled: bool = Field(default=True, description="Enable monitoring endpoints")
    enable_metrics: bool = Field(default=False, description="Expose metrics endpoint")
    enable_health_checks: bool = Field(
        default=True, description="Expose health check endpoint"
    )
    metrics_port: int = Field(
        default=8001, gt=0, le=65535, description="Metrics server port"
    )
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    health_path: str = Field(default="/health", description="Health endpoint path")
    include_system_metrics: bool = Field(
        default=True, description="Include system metrics"
    )
    system_metrics_interval: float = Field(
        default=30.0, gt=0, description="System metrics interval seconds"
    )
    health_check_timeout: float = Field(
        default=10.0, gt=0, description="Health check timeout seconds"
    )


class ObservabilityConfig(BaseModel):
    """OpenTelemetry observability configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry")
    service_name: str = Field(default="ai-docs-vector-db", description="Service name")
    service_version: str = Field(default="1.0.0", description="Service version")
    service_namespace: str = Field(default="ai-docs", description="Service namespace")
    otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP endpoint"
    )
    otlp_headers: dict[str, str] = Field(
        default_factory=dict, description="OTLP headers"
    )
    otlp_insecure: bool = Field(default=True, description="Use insecure OTLP transport")
    trace_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sample rate"
    )
    track_ai_operations: bool = Field(
        default=True, description="Track AI operation metrics"
    )
    track_costs: bool = Field(default=True, description="Track AI operation costs")
    instrument_fastapi: bool = Field(default=True, description="Instrument FastAPI")
    instrument_httpx: bool = Field(default=True, description="Instrument HTTPX")
    instrument_redis: bool = Field(default=True, description="Instrument Redis")
    instrument_sqlalchemy: bool = Field(
        default=True, description="Instrument SQLAlchemy"
    )
    console_exporter: bool = Field(
        default=False, description="Enable console span exporter"
    )


class TaskQueueConfig(BaseModel):
    """Task queue configuration for background workers."""

    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_database: int = Field(default=0, ge=0, le=15, description="Redis database")
    max_jobs: int = Field(default=10, gt=0, description="Max concurrent jobs")
    job_timeout: int = Field(default=300, gt=0, description="Job timeout seconds")
    default_queue_name: str = Field(default="default", description="Default queue name")


class RAGConfig(BaseModel):
    """Retrieval-augmented generation configuration."""

    enable_rag: bool = Field(default=False, description="Enable RAG generation")
    model: str = Field(default="gpt-3.5-turbo", description="LLM model")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(
        default=1000, gt=0, le=4000, description="Max response tokens"
    )
    timeout_seconds: float = Field(default=30.0, gt=0, description="Generation timeout")
    max_context_length: int = Field(
        default=4000, gt=0, description="Max RAG context length"
    )
    max_results_for_context: int = Field(
        default=5, gt=0, le=20, description="Max documents for context"
    )
    min_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Confidence threshold"
    )
    include_sources: bool = Field(default=True, description="Include source citations")
    include_confidence_score: bool = Field(
        default=True, description="Include confidence scores"
    )
    enable_answer_metrics: bool = Field(
        default=True, description="Track answer metrics"
    )
    enable_caching: bool = Field(default=True, description="Enable RAG caching")
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL seconds")
    parallel_processing: bool = Field(
        default=True, description="Process retrieval stages in parallel"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Enable deterministic contextual compression before RAG",
    )
    compression_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity required for a sentence to be retained.",
    )
    compression_mmr_lambda: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Trade-off parameter for MMR "
            "(1.0 -> relevance only, 0.0 -> diversity only)."
        ),
    )
    compression_token_ratio: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description="Target ratio of tokens to keep relative to the original context.",
    )
    compression_absolute_max_tokens: int = Field(
        default=400,
        ge=50,
        le=2000,
        description="Hard cap on tokens retained per document after compression.",
    )
    compression_min_sentences: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum number of sentences to keep per document.",
    )
    compression_max_sentences: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of sentences to keep per document.",
    )


class DeploymentConfig(BaseModel):
    """Deployment feature flags and metadata."""

    tier: DeploymentTier = Field(
        default=DeploymentTier.ENTERPRISE, description="Deployment tier"
    )
    enable_feature_flags: bool = Field(default=True, description="Enable feature flags")
    flagsmith_api_key: str | None = Field(default=None, description="Flagsmith API key")
    flagsmith_environment_key: str | None = Field(
        default=None, description="Flagsmith environment key"
    )
    flagsmith_api_url: str = Field(
        default="https://edge.api.flagsmith.com/api/v1/", description="Flagsmith URL"
    )
    enable_deployment_services: bool = Field(
        default=True, description="Enable deployment services"
    )
    enable_ab_testing: bool = Field(default=True, description="Enable A/B testing")
    enable_blue_green: bool = Field(
        default=True, description="Enable blue-green deployments"
    )
    enable_canary: bool = Field(default=True, description="Enable canary deployments")
    enable_monitoring: bool = Field(
        default=True, description="Enable deployment monitoring"
    )

    @field_validator("flagsmith_api_key", mode="before")
    @classmethod
    def validate_flagsmith_key(cls, value: str | None) -> str | None:
        if value and not value.startswith(("fs_", "env_")):
            msg = "Flagsmith API key must start with 'fs_' or 'env_'"
            raise ValueError(msg)
        return value


class AutoDetectionConfig(BaseModel):
    """Service auto-detection configuration."""

    enabled: bool = Field(default=True, description="Enable auto detection")
    timeout_seconds: float = Field(default=5.0, gt=0, description="Detection timeout")
    retry_attempts: int = Field(default=3, ge=1, description="Retry attempts")


class DriftDetectionConfig(BaseModel):
    """Configuration drift monitoring settings."""

    enabled: bool = Field(default=True, description="Enable drift detection")
    snapshot_interval_minutes: int = Field(
        default=15, gt=0, le=1440, description="Snapshot interval"
    )
    comparison_interval_minutes: int = Field(
        default=5, gt=0, le=60, description="Comparison interval"
    )
    monitored_paths: list[str] = Field(
        default_factory=lambda: [
            "src/config/",
            ".env",
            "config.yaml",
            "config.json",
            "docker-compose.yml",
            "docker-compose.yaml",
        ],
        description="Paths monitored for drift",
    )
    excluded_paths: list[str] = Field(
        default_factory=lambda: [
            "**/__pycache__/",
            "**/*.pyc",
            "**/logs/",
            "**/cache/",
            "**/tmp/",
        ],
        description="Excluded paths",
    )
    alert_on_severity: list[str] = Field(
        default_factory=lambda: ["high", "critical"],
        description="Alert severities that trigger notifications",
    )
    max_alerts_per_hour: int = Field(
        default=10, gt=0, description="Max drift alerts per hour"
    )
    snapshot_retention_days: int = Field(
        default=30, gt=0, description="Retention for snapshots"
    )
    events_retention_days: int = Field(
        default=90, gt=0, description="Retention for drift events"
    )
    integrate_with_task20_anomaly: bool = Field(
        default=True, description="Integrate with anomaly detection"
    )
    use_performance_monitoring: bool = Field(
        default=True, description="Use performance monitoring signals"
    )
    enable_auto_remediation: bool = Field(
        default=False, description="Enable automatic remediation"
    )
    auto_remediation_severity_threshold: str = Field(
        default="high", description="Severity threshold for remediation"
    )


class DocumentationSite(BaseModel):
    """Documentation site crawl configuration."""

    name: str = Field(..., min_length=1, description="Site name")
    url: HttpUrl = Field(..., description="Site URL")
    max_pages: int = Field(default=50, gt=0, description="Max pages to crawl")
    max_depth: int = Field(default=2, gt=0, description="Max crawl depth")
    priority: str = Field(default="medium", description="Crawl priority")


#### Auto-detection helper models ####


class DetectedService(BaseModel):
    """Auto-detected service with lightweight health metadata."""

    name: str
    url: str
    healthy: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectedEnvironment(BaseModel):
    """Environment detection result."""

    environment_type: Environment = Environment.DEVELOPMENT
    is_containerized: bool = False
    is_kubernetes: bool = False
    detection_confidence: float = 0.0
    detection_time_ms: float = 0.0
    cloud_provider: str | None = None


class AutoDetectedServices(BaseModel):
    """Collection of auto-detected services and contextual metadata."""

    environment: DetectedEnvironment
    services: list[DetectedService] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EnvironmentDetector(BaseModel):
    """Simple environment detector placeholder capable of future extension."""

    config: AutoDetectionConfig | None = None

    async def detect(self) -> DetectedEnvironment:
        """Return a basic environment description.

        The implementation can be extended to perform actual detection using
        container introspection or cloud SDKs. For now, it returns a default
        environment with zero confidence so callers can distinguish between
        inferred and observed states.
        """

        return DetectedEnvironment()


__all__ = [
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
    "DocumentStatus",
    "DocumentationSite",
    "DetectedEnvironment",
    "DetectedService",
    "DriftDetectionConfig",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
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
    "QueryComplexity",
    "QueryType",
    "RAGConfig",
    "ReRankingConfig",
    "SearchAccuracy",
    "SearchMode",
    "SearchPipeline",
    "SearchStrategy",
    "TaskQueueConfig",
    "VectorType",
    "EnvironmentDetector",
]
