"""Configuration enums consolidated from original enums.py.

All enumeration types used in configuration in one place.
"""

from enum import Enum


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    """Embedding providers."""

    OPENAI = "openai"
    FASTEMBED = "fastembed"


class QualityTier(str, Enum):
    """Embedding quality tiers."""

    FAST = "fast"  # Local models, fastest
    BALANCED = "balanced"  # Balance of speed and quality
    BEST = "best"  # Highest quality, may be slower/costlier
    PREMIUM = "premium"  # Premium tier with advanced features
    FASTEMBED = "fastembed"


class EmbeddingModel(str, Enum):
    """Embedding models from various providers."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"
    BGE_LARGE_EN_V15 = "BAAI/bge-large-en-v1.5"
    NV_EMBED_V2 = "nvidia/NV-Embed-v2"


class CrawlProvider(str, Enum):
    """Crawling providers."""

    FIRECRAWL = "firecrawl"
    CRAWL4AI = "crawl4ai"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    AST_AWARE = "ast_aware"


class SearchStrategy(str, Enum):
    """Vector search strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SearchAccuracy(str, Enum):
    """Search accuracy levels."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    EXACT = "exact"


class VectorType(str, Enum):
    """Vector types for search optimization."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYDE = "hyde"


class CacheType(str, Enum):
    """Cache data types."""

    EMBEDDINGS = "embeddings"
    CRAWL = "crawl"
    SEARCH = "search"
    HYDE = "hyde"


# Additional enums needed by the codebase
class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(str, Enum):
    """Query types for different search scenarios."""

    SIMPLE = "simple"
    COMPLEX = "complex"
    SEMANTIC = "semantic"
    CODE = "code"
    API_REFERENCE = "api_reference"
    DOCUMENTATION = "documentation"
    TROUBLESHOOTING = "troubleshooting"
    MULTIMODAL = "multimodal"
    CONCEPTUAL = "conceptual"


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ModelType(str, Enum):
    """Model types for different AI capabilities."""

    EMBEDDING = "embedding"
    RERANKING = "reranking"
    GENERAL_PURPOSE = "general_purpose"
    CODE_SPECIALIZED = "code_specialized"
    MULTIMODAL = "multimodal"


class OptimizationStrategy(str, Enum):
    """Optimization strategies for model selection."""

    SPEED = "speed"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    COST_OPTIMIZED = "cost_optimized"


class FusionAlgorithm(str, Enum):
    """Fusion algorithms for hybrid search."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"
    WEIGHTED = "weighted"


class ABTestVariant(str, Enum):
    """A/B test variants."""

    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"


class DeploymentTier(str, Enum):
    """Deployment configuration tiers with progressive feature access."""

    PERSONAL = "personal"  # Simple FastAPI + basic features
    PROFESSIONAL = "professional"  # + Flagsmith + monitoring
    ENTERPRISE = "enterprise"  # + Full deployment services + advanced monitoring
