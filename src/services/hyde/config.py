"""HyDE configuration models with Pydantic v2."""

from typing import TYPE_CHECKING

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


if TYPE_CHECKING:
    from src.config import HyDEConfig as UnifiedHyDEConfig


class HyDEConfig(BaseModel):
    """HyDE configuration with defaults optimized for documentation search."""

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

    @classmethod
    def from_unified_config(
        cls, unified_hyde_config: "UnifiedHyDEConfig"
    ) -> "HyDEConfig":
        """Create HyDEConfig from unified configuration.

        Args:
            unified_hyde_config: HyDE configuration from UnifiedConfig

        Returns:
            HyDEConfig instance with values from unified config

        """
        return cls(
            enable_hyde=unified_hyde_config.enable_hyde,
            enable_fallback=unified_hyde_config.enable_fallback,
            enable_reranking=unified_hyde_config.enable_reranking,
            enable_caching=unified_hyde_config.enable_caching,
            num_generations=unified_hyde_config.num_generations,
            generation_temperature=unified_hyde_config.generation_temperature,
            max_generation_tokens=unified_hyde_config.max_generation_tokens,
            generation_model=unified_hyde_config.generation_model,
            generation_timeout_seconds=unified_hyde_config.generation_timeout_seconds,
            hyde_prefetch_limit=unified_hyde_config.hyde_prefetch_limit,
            query_prefetch_limit=unified_hyde_config.query_prefetch_limit,
            hyde_weight_in_fusion=unified_hyde_config.hyde_weight_in_fusion,
            fusion_algorithm=unified_hyde_config.fusion_algorithm,
            cache_ttl_seconds=unified_hyde_config.cache_ttl_seconds,
            cache_hypothetical_docs=unified_hyde_config.cache_hypothetical_docs,
            cache_prefix=unified_hyde_config.cache_prefix,
            parallel_generation=unified_hyde_config.parallel_generation,
            max_concurrent_generations=unified_hyde_config.max_concurrent_generations,
            use_domain_specific_prompts=unified_hyde_config.use_domain_specific_prompts,
            prompt_variation=unified_hyde_config.prompt_variation,
            min_generation_length=unified_hyde_config.min_generation_length,
            filter_duplicates=unified_hyde_config.filter_duplicates,
            diversity_threshold=unified_hyde_config.diversity_threshold,
            log_generations=unified_hyde_config.log_generations,
            track_metrics=unified_hyde_config.track_metrics,
        )

    @field_validator("generation_model")
    @classmethod
    def validate_generation_model(cls, v: str) -> str:
        """Validate that the generation model is supported."""
        supported_models = {
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "claude-3-haiku",
            "claude-3-sonnet",
            "claude-3-opus",
        }
        if v not in supported_models:
            # Allow any model but warn about unknown ones
            pass
        return v

    @field_validator("fusion_algorithm")
    @classmethod
    def validate_fusion_algorithm(cls, v: str) -> str:
        """Validate fusion algorithm is supported."""
        if v not in {"rrf", "dbsf", "linear", "rank_fusion"}:
            msg = f"Unsupported fusion algorithm: {v}. Use 'rrf', 'dbsf', 'linear', or 'rank_fusion'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_generation_settings(self) -> "HyDEConfig":
        """Cross-field validation for generation settings."""
        # Ensure cache TTL is reasonable for timeout
        if self.cache_ttl_seconds < self.generation_timeout_seconds * 2:
            msg = "Cache TTL should be at least 2x the generation timeout"
            raise ValueError(msg)

        # Ensure concurrent generations doesn't exceed total generations
        if self.max_concurrent_generations > self.num_generations:
            msg = "Max concurrent generations cannot exceed total generations"
            raise ValueError(msg)

        # Validate prefetch limits make sense
        if self.hyde_prefetch_limit < self.query_prefetch_limit:
            msg = "HyDE prefetch limit should be >= query prefetch limit"
            raise ValueError(msg)

        return self

    @computed_field
    @property
    def estimated_total_cost_per_query(self) -> float:
        """Estimate cost per query based on configuration."""
        # Rough cost estimation (adjust based on actual pricing)
        base_cost_per_token = 0.00002  # Approximate cost per token
        estimated_cost = (
            self.num_generations * self.max_generation_tokens * base_cost_per_token
        )
        return round(estimated_cost, 6)

    @computed_field
    @property
    def estimated_latency_ms(self) -> float:
        """Estimate total latency based on configuration."""
        if self.parallel_generation:
            # Parallel generation - limited by slowest request
            generation_time = self.generation_timeout_seconds * 1000
        else:
            # Sequential generation
            generation_time = (
                self.num_generations * self.generation_timeout_seconds * 1000
            )

        # Add overhead for processing and caching
        overhead = 50 + (self.num_generations * 10)
        return generation_time + overhead

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Determine if configuration is suitable for production."""
        return (
            self.enable_fallback
            and self.enable_caching
            and self.generation_timeout_seconds <= 30
            and self.num_generations <= 8
            and self.cache_ttl_seconds >= 1800  # At least 30 minutes
        )

    @computed_field
    @property
    def performance_tier(self) -> str:
        """Categorize configuration by performance characteristics."""
        if self.num_generations <= 3 and self.max_generation_tokens <= 150:
            return "fast"
        if self.num_generations <= 5 and self.max_generation_tokens <= 250:
            return "balanced"
        return "quality"

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "enable_hyde": True,
                "num_generations": 5,
                "generation_temperature": 0.7,
                "max_generation_tokens": 200,
                "generation_model": "gpt-3.5-turbo",
                "hyde_prefetch_limit": 50,
                "query_prefetch_limit": 30,
                "cache_ttl_seconds": 3600,
                "parallel_generation": True,
            }
        },
    )


class HyDEPromptConfig(BaseModel):
    """Configuration for HyDE prompt templates."""

    # Base prompts for different document types
    technical_prompt: str = Field(
        default="You are a technical documentation expert. Answer this question with a detailed, accurate response:\n\nQuestion: {query}\n\nProvide a comprehensive answer that would appear in high-quality technical documentation:",
        description="Prompt template for technical queries",
    )

    code_prompt: str = Field(
        default="You are a code documentation expert. Answer this programming question:\n\nQuestion: {query}\n\nProvide a detailed answer with code examples as would appear in API documentation:",
        description="Prompt template for code-related queries",
    )

    tutorial_prompt: str = Field(
        default="You are a tutorial writer. Answer this question as if explaining to someone learning:\n\nQuestion: {query}\n\nProvide a step-by-step explanation with examples:",
        description="Prompt template for tutorial queries",
    )

    general_prompt: str = Field(
        default="Answer the following question accurately and comprehensively:\n\nQuestion: {query}\n\nAnswer:",
        description="General prompt template for other queries",
    )

    # Keywords for prompt selection
    technical_keywords: list[str] = Field(
        default=[
            "api",
            "function",
            "method",
            "class",
            "parameter",
            "configuration",
            "setup",
            "install",
            "error",
            "debug",
        ],
        description="Keywords that indicate technical queries",
    )

    code_keywords: list[str] = Field(
        default=[
            "how to",
            "example",
            "code",
            "implement",
            "python",
            "javascript",
            "function",
            "syntax",
            "library",
            "import",
        ],
        description="Keywords that indicate code-related queries",
    )

    tutorial_keywords: list[str] = Field(
        default=[
            "tutorial",
            "guide",
            "step by step",
            "learn",
            "getting started",
            "introduction",
            "beginner",
        ],
        description="Keywords that indicate tutorial queries",
    )

    # Prompt variations for diversity
    variation_templates: dict[str, list[str]] = Field(
        default={
            "prefixes": [
                "You are an expert in {domain}.",
                "As a {domain} specialist,",
                "From a {domain} perspective,",
                "Drawing on expertise in {domain},",
            ],
            "instruction_styles": [
                "Answer this question:",
                "Provide a detailed response to:",
                "Explain thoroughly:",
                "Give a comprehensive answer to:",
            ],
            "context_additions": [
                "Include practical examples.",
                "Focus on implementation details.",
                "Emphasize best practices.",
                "Provide clear explanations.",
            ],
        },
        description="Template variations for generating diverse prompts",
    )


class HyDEMetricsConfig(BaseModel):
    """Configuration for HyDE metrics and monitoring."""

    # Performance metrics
    track_generation_time: bool = Field(
        default=True, description="Track generation latency"
    )
    track_cache_hits: bool = Field(default=True, description="Track cache hit rates")
    track_search_quality: bool = Field(
        default=True, description="Track search relevance"
    )
    track_cost_savings: bool = Field(
        default=True, description="Track cost compared to regular search"
    )

    # Quality metrics
    measure_diversity: bool = Field(
        default=True, description="Measure generation diversity"
    )
    measure_relevance: bool = Field(
        default=True, description="Measure result relevance"
    )
    measure_coverage: bool = Field(default=True, description="Measure search coverage")

    # A/B testing
    ab_testing_enabled: bool = Field(default=False, description="Enable A/B testing")
    control_group_percentage: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Percentage for control group"
    )

    # Reporting
    metrics_export_interval: int = Field(
        default=300, ge=60, le=3600, description="Metrics export interval in seconds"
    )
    detailed_logging: bool = Field(
        default=False, description="Enable detailed metrics logging"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "track_generation_time": True,
                "track_cache_hits": True,
                "track_search_quality": True,
                "ab_testing_enabled": False,
                "control_group_percentage": 0.5,
            }
        }
    }
