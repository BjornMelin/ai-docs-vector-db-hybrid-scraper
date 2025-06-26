"""Type definitions for the progressive API system.

This module defines the sophisticated type system that supports
progressive disclosure while maintaining type safety.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")


class QualityTier(str, Enum):
    """Quality tiers for embedding and search operations."""

    FAST = "fast"  # Local models, fastest response
    BALANCED = "balanced"  # Balance of speed and quality
    BEST = "best"  # Highest quality, may be slower


class SearchStrategy(str, Enum):
    """Search strategy options."""

    VECTOR = "vector"  # Pure vector search
    HYBRID = "hybrid"  # Vector + keyword search
    SEMANTIC = "semantic"  # Semantic search with reranking
    ADAPTIVE = "adaptive"  # Auto-select best strategy


class ContentType(str, Enum):
    """Content type classification."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class ProcessingMode(str, Enum):
    """Document processing modes."""

    FAST = "fast"  # Minimal processing
    STANDARD = "standard"  # Standard processing
    ENHANCED = "enhanced"  # Full processing with analysis
    CUSTOM = "custom"  # Custom processing pipeline


class SearchOptions(BaseModel):
    """Advanced search options for progressive disclosure.

    This model allows users to progressively discover and use
    advanced search features while maintaining simple defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # Core options
    strategy: SearchStrategy = SearchStrategy.HYBRID
    rerank: bool = False
    quality_tier: QualityTier = QualityTier.BALANCED

    # Filtering options
    content_types: List[ContentType] | None = None
    date_range: tuple[datetime, datetime] | None = None
    metadata_filters: Dict[str, Any] | None = None

    # Advanced options
    include_embeddings: bool = False
    include_analysis: bool = False
    include_suggestions: bool = False
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    diversity_factor: float = Field(default=0.0, ge=0.0, le=1.0)

    # Expert options
    custom_weights: Dict[str, float] | None = None
    fusion_algorithm: str | None = None
    rerank_model: str | None = None


class EmbeddingOptions(BaseModel):
    """Options for embedding generation."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "fastembed"
    model_name: str | None = None
    quality_tier: QualityTier = QualityTier.BALANCED
    batch_size: int = Field(default=32, ge=1, le=1000)
    normalize: bool = True

    # Advanced options
    chunk_strategy: str | None = None
    overlap_size: int | None = None
    custom_preprocessing: Dict[str, Any] | None = None


class ProcessingOptions(BaseModel):
    """Options for document processing."""

    model_config = ConfigDict(extra="forbid")

    mode: ProcessingMode = ProcessingMode.STANDARD
    extract_metadata: bool = True
    detect_language: bool = True
    clean_content: bool = True

    # Chunking options
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    # Advanced processing
    extract_entities: bool = False
    generate_summary: bool = False
    classify_content: bool = False
    quality_assessment: bool = False


@dataclass
class ProgressiveResponse[T]:
    """Response wrapper that supports progressive feature revelation.

    This wrapper allows basic responses to be extended with
    progressive features without breaking existing code.
    """

    # Core response
    data: T
    success: bool = True
    message: str = ""

    # Progressive features
    metadata: Dict[str, Any] | None = None
    metrics: Dict[str, float] | None = None
    suggestions: List[str] | None = None

    # Expert features
    debug_info: Dict[str, Any] | None = None
    performance_stats: Dict[str, float] | None = None

    def with_metadata(self, metadata: Dict[str, Any]) -> "ProgressiveResponse[T]":
        """Add metadata (progressive feature)."""
        self.metadata = metadata
        return self

    def with_metrics(self, metrics: Dict[str, float]) -> "ProgressiveResponse[T]":
        """Add performance metrics (progressive feature)."""
        self.metrics = metrics
        return self

    def with_suggestions(self, suggestions: List[str]) -> "ProgressiveResponse[T]":
        """Add suggestions (progressive feature)."""
        self.suggestions = suggestions
        return self

    def with_debug_info(self, debug_info: Dict[str, Any]) -> "ProgressiveResponse[T]":
        """Add debug information (expert feature)."""
        self.debug_info = debug_info
        return self


class CacheOptions(BaseModel):
    """Caching configuration options."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)  # 1 hour to 24 hours
    max_size: int = Field(default=1000, ge=10, le=100000)

    # Advanced caching
    cache_embeddings: bool = True
    cache_search_results: bool = True
    cache_analysis: bool = False

    # Expert options
    eviction_policy: str = "lru"
    compression: bool = False
    distributed: bool = False


class MonitoringOptions(BaseModel):
    """Monitoring and observability options."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    collect_metrics: bool = True
    trace_requests: bool = False

    # Performance monitoring
    track_latency: bool = True
    track_throughput: bool = True
    track_errors: bool = True

    # Cost monitoring
    track_costs: bool = False
    budget_alerts: bool = False
    budget_limit: float | None = None

    # Expert monitoring
    detailed_tracing: bool = False
    export_metrics: bool = False
    metrics_endpoint: str | None = None


class SystemConfiguration(BaseModel):
    """Complete system configuration with progressive complexity."""

    model_config = ConfigDict(extra="forbid")

    # Basic configuration
    embedding_provider: str = "fastembed"
    quality_tier: QualityTier = QualityTier.BALANCED
    workspace_dir: str | None = None

    # Progressive configuration
    search_options: SearchOptions = Field(default_factory=SearchOptions)
    embedding_options: EmbeddingOptions = Field(default_factory=EmbeddingOptions)
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    # Advanced configuration
    cache_options: CacheOptions = Field(default_factory=CacheOptions)
    monitoring_options: MonitoringOptions = Field(default_factory=MonitoringOptions)

    # Expert configuration
    custom_providers: Dict[str, Any] | None = None
    experimental_features: Dict[str, bool] | None = None
    debug_mode: bool = False


class FeatureCapability(BaseModel):
    """Description of a system feature capability."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    level: str  # "basic", "progressive", "expert"
    example: str
    requirements: List[str] | None = None
    documentation_url: str | None = None


class SystemCapabilities(BaseModel):
    """Complete description of system capabilities."""

    model_config = ConfigDict(extra="forbid")

    basic_features: List[FeatureCapability]
    progressive_features: List[FeatureCapability]
    expert_features: List[FeatureCapability]

    # Discovery helpers
    next_steps: List[str]
    learning_path: List[str]
    examples: Dict[str, str]


# Type aliases for common patterns
SearchResult = Union[Dict[str, Any], "SimpleSearchResult"]
EmbeddingVector = List[float]
DocumentId = str
QueryString = str
MetadataDict = Dict[str, Any]

# Generic response types
SearchResponse = ProgressiveResponse[List[SearchResult]]
EmbeddingResponse = ProgressiveResponse[EmbeddingVector]
ProcessingResponse = ProgressiveResponse[DocumentId]
StatsResponse = ProgressiveResponse[Dict[str, Any]]


# Protocol type hints for dependency injection
class Configurable(Protocol):
    """Protocol for configurable components."""

    def configure(self, options: Dict[str, Any]) -> None:
        """Configure the component with options."""
        ...


class Initializable(Protocol):
    """Protocol for components that require initialization."""

    async def initialize(self) -> None:
        """Initialize the component."""
        ...

    async def cleanup(self) -> None:
        """Cleanup component resources."""
        ...


class Observable(Protocol):
    """Protocol for observable components."""

    def get_metrics(self) -> Dict[str, float]:
        """Get component metrics."""
        ...

    def get_health(self) -> Dict[str, Any]:
        """Get component health status."""
        ...
