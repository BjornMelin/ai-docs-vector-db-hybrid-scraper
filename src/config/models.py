"""Typed configuration models for the AI docs platform."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


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


class PlaywrightProxySettings(BaseModel):
    """Proxy configuration injected into Playwright browser contexts."""

    server: str = Field(..., description="Proxy server URL, including scheme.")
    username: str | None = Field(
        default=None, description="Proxy authentication username"
    )
    password: str | None = Field(
        default=None, description="Proxy authentication password"
    )
    bypass: str | None = Field(
        default=None,
        description="Comma separated hostnames that should bypass the proxy",
    )

    def to_playwright_dict(self) -> dict[str, str]:
        """Return the dictionary signature expected by Playwright."""

        proxy: dict[str, str] = {"server": self.server}
        if self.username:
            proxy["username"] = self.username
        if self.password:
            proxy["password"] = self.password
        if self.bypass:
            proxy["bypass"] = self.bypass
        return proxy


class PlaywrightCaptchaSettings(BaseModel):
    """CAPTCHA solving configuration using CapMonster Cloud."""

    provider: Literal["capmonster"] = Field(
        default="capmonster", description="Captcha solving provider identifier"
    )
    api_key: str = Field(..., description="CapMonster Cloud API key")
    captcha_type: Literal["recaptcha_v2", "hcaptcha", "turnstile"] = Field(
        default="recaptcha_v2",
        description="Captcha type solved through CapMonster",
    )
    iframe_selector: str = Field(
        default="iframe[src*='captcha']",
        description="CSS selector pointing at the captcha iframe",
    )
    response_input_selector: str = Field(
        default="textarea[name='g-recaptcha-response']",
        description="Selector used to inject the solved token",
    )


class PlaywrightTierConfig(BaseModel):
    """Configuration describing a single anti-bot execution tier."""

    name: str = Field(default="baseline", description="Identifier for the tier")
    use_undetected_browser: bool = Field(
        default=False,
        description="Whether to launch the Rebrowser patched Playwright runtime",
    )
    enable_stealth: bool = Field(
        default=True,
        description="Apply tf_playwright_stealth transformations for the tier",
    )
    proxy: PlaywrightProxySettings | None = Field(
        default=None, description="Proxy configuration applied to this tier"
    )
    captcha: PlaywrightCaptchaSettings | None = Field(
        default=None, description="Captcha solving configuration"
    )
    max_attempts: int = Field(
        default=1, gt=0, description="Maximum navigation attempts for the tier"
    )
    challenge_status_codes: list[int] = Field(
        default_factory=lambda: [403, 429],
        description="HTTP status codes that should trigger escalation",
    )
    challenge_keywords: list[str] = Field(
        default_factory=lambda: ["captcha", "verify you are human"],
        description="Body snippets that indicate bot-detection challenges",
    )


class PlaywrightConfig(BaseModel):
    """Playwright browser configuration for automation clients."""

    browser: str = Field(default="chromium", description="Browser type")
    headless: bool = Field(default=True, description="Run in headless mode")
    timeout: int = Field(default=30000, gt=0, description="Timeout in milliseconds")
    viewport: dict[str, int | float | bool] = Field(
        default_factory=lambda: {"width": 1920, "height": 1080},
        description="Viewport size (width/height) applied to new browser contexts",
    )
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        description="User-Agent header injected into new browser contexts",
    )
    enable_stealth: bool = Field(
        default=True,
        description="Global default for applying tf_playwright_stealth",
    )
    tiers: list[PlaywrightTierConfig] = Field(
        default_factory=list,
        description="Ordered anti-bot tiers executed until success",
    )

    @field_validator("viewport")
    @classmethod
    def validate_viewport(
        cls, value: dict[str, int | float | bool]
    ) -> dict[str, int | float | bool]:
        required_keys = {"width", "height"}
        missing = required_keys.difference(value)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            msg = f"Viewport must include keys: {missing_keys}"
            raise ValueError(msg)

        clean_value = dict(value)
        for key in required_keys:
            dimension = clean_value[key]
            if not isinstance(dimension, int) or dimension <= 0:
                msg = f"Viewport {key} must be a positive integer"
                raise ValueError(msg)
            clean_value[key] = int(dimension)

        return clean_value

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(cls, value: str) -> str:
        if not value or not value.strip():
            msg = "User agent must be a non-empty string"
            raise ValueError(msg)
        return value.strip()

    @model_validator(mode="after")
    def default_tier_injection(self) -> Self:
        if not self.tiers:
            self.tiers = [
                PlaywrightTierConfig(
                    name="baseline",
                    use_undetected_browser=False,
                    enable_stealth=self.enable_stealth,
                )
            ]
        return self


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
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "AgenticConfig",
    "FastEmbedConfig",
    "FirecrawlConfig",
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
    "PlaywrightConfig",
    "QdrantConfig",
    "QueryComplexity",
    "QueryType",
    "RAGConfig",
    "ReRankingConfig",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
]
