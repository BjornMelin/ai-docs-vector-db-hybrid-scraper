"""Typed configuration models for the AI docs platform."""

# pylint: disable=too-many-lines  # Extensive configuration schema definitions live here

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import (  # pyright: ignore[reportMissingImports]
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


#### Enumerations ####


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


class DeploymentTier(str, Enum):
    """Deployment tiers available to the platform."""

    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class MCPTransport(str, Enum):
    """Transport mechanisms supported by MCP clients."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server endpoint."""

    name: str = Field(..., min_length=1, description="Unique MCP server identifier")
    transport: MCPTransport = Field(
        MCPTransport.STDIO, description="Transport type for the server"
    )
    command: str | None = Field(
        None,
        description="Executable name for stdio transports",
    )
    args: list[str] = Field(
        default_factory=list, description="Command arguments for stdio transports"
    )
    url: HttpUrl | None = Field(
        None,
        description="HTTP endpoint for streamable_http and sse transports",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers for HTTP-based transports",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables applied when launching the server",
    )
    timeout_ms: int | None = Field(
        None,
        ge=0,
        description="Optional execution timeout for stdio transports",
    )

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == MCPTransport.STDIO and not self.command:
            msg = "command is required for stdio MCP transports"
            raise ValueError(msg)
        if self.transport != MCPTransport.STDIO and not self.url:
            msg = "url is required for non-stdio MCP transports"
            raise ValueError(msg)
        return self


class MCPClientConfig(BaseModel):
    """Top-level configuration for MCP client connectivity."""

    enabled: bool = Field(True, description="Enable MCP client integration")
    request_timeout_ms: int = Field(
        60000, ge=1000, description="Default timeout for MCP tool executions"
    )
    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="Configured MCP servers"
    )


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
        description=(
            "Search result cache TTL in seconds (must be positive; negative values are "
            "not supported)."
        ),
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
        default="documents", description="Primary collection name used for queries"
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


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(
        default=1600,
        gt=0,
        description="Target chunk size to preserve context fidelity",
    )
    chunk_overlap: int = Field(
        default=320,
        ge=0,
        description="Overlap between chunks to maintain coherence across boundaries",
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be less than chunk_size"
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
    request_timeout: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Request timeout in seconds; avoids indefinite waits while tolerating "
            "slow APIs"
        ),
    )
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
        default=5,
        gt=0,
        le=20,
        description="Failures before opening circuit to curb cascading faults",
    )
    recovery_timeout: float = Field(
        default=60.0,
        gt=0,
        description=(
            "Recovery timeout in seconds; cooldown before retrying unstable services"
        ),
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
    namespace: str = Field(default="ml_app", description="Metrics namespace prefix")
    cpu_threshold: float = Field(default=90.0, description="CPU usage threshold %")
    memory_threshold: float = Field(default=90.0, description="Memory threshold %")
    disk_threshold: float = Field(default=90.0, description="Disk usage threshold %")
    external_services: dict[str, str] = Field(
        default_factory=dict, description="External services to monitor"
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


class AgenticConfig(BaseModel):
    """Configuration for agentic orchestration defaults."""

    run_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Maximum end-to-end agent run timeout"
    )
    max_parallel_tools: int = Field(
        default=3, ge=1, le=16, description="Maximum concurrently executed tools"
    )
    retrieval_limit: int = Field(
        default=8, ge=1, le=50, description="Default document retrieval limit"
    )


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


class DocumentationSite(BaseModel):
    """Documentation site crawl configuration."""

    name: str = Field(..., min_length=1, description="Site name")
    url: HttpUrl = Field(..., description="Site URL")
    max_pages: int = Field(default=50, gt=0, description="Max pages to crawl")
    max_depth: int = Field(default=2, gt=0, description="Max crawl depth")
    priority: str = Field(default="medium", description="Crawl priority")


__all__ = [
    "CacheConfig",
    "CacheType",
    "ChunkingConfig",
    "ChunkingStrategy",
    "CircuitBreakerConfig",
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
    "AgenticConfig",
    "FastEmbedConfig",
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
    "PerformanceConfig",
    "QdrantConfig",
    "QueryComplexity",
    "QueryType",
    "RAGConfig",
    "ReRankingConfig",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
]
