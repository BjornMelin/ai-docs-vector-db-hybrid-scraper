"""HyDE configuration models with Pydantic v2."""

from typing import Any

from pydantic import BaseModel, Field


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

    model_config = {
        "json_schema_extra": {
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
        }
    }

    @classmethod
    def from_unified_config(cls, config: Any) -> "HyDEConfig":
        """Create HyDE configuration from unified settings object."""

        return cls(
            enable_hyde=getattr(config, "enable_hyde", True),
            enable_fallback=getattr(config, "enable_fallback", True),
            enable_reranking=getattr(config, "enable_reranking", True),
            enable_caching=getattr(config, "enable_caching", True),
            num_generations=getattr(config, "num_generations", 5),
            generation_temperature=getattr(config, "generation_temperature", 0.7),
            max_generation_tokens=getattr(config, "max_generation_tokens", 200),
            generation_model=getattr(config, "generation_model", "gpt-3.5-turbo"),
            generation_timeout_seconds=getattr(
                config, "generation_timeout_seconds", 10
            ),
            hyde_prefetch_limit=getattr(config, "hyde_prefetch_limit", 50),
            query_prefetch_limit=getattr(config, "query_prefetch_limit", 30),
            hyde_weight_in_fusion=getattr(config, "hyde_weight_in_fusion", 0.6),
            fusion_algorithm=getattr(config, "fusion_algorithm", "rrf"),
            cache_ttl_seconds=getattr(config, "cache_ttl_seconds", 3600),
            cache_hypothetical_docs=getattr(config, "cache_hypothetical_docs", True),
            cache_prefix=getattr(config, "cache_prefix", "hyde"),
            parallel_generation=getattr(config, "parallel_generation", True),
            max_concurrent_generations=getattr(config, "max_concurrent_generations", 5),
            use_domain_specific_prompts=getattr(
                config, "use_domain_specific_prompts", True
            ),
            prompt_variation=getattr(config, "prompt_variation", True),
            min_generation_length=getattr(config, "min_generation_length", 20),
            filter_duplicates=getattr(config, "filter_duplicates", True),
            diversity_threshold=getattr(config, "diversity_threshold", 0.3),
            log_generations=getattr(config, "log_generations", False),
            track_metrics=getattr(config, "track_metrics", True),
        )


class HyDEPromptConfig(BaseModel):
    """Configuration for HyDE prompt templates."""

    # Base prompts for different document types
    technical_prompt: str = Field(
        default=(
            "You are a technical documentation expert. Answer this question with a "
            "detailed, accurate  response:"
            "\n\nQuestion: {query}\n\nProvide a comprehensive "
            "answer that would appear in high-quality technical documentation:"
        ),
        description="Prompt template for technical queries",
    )

    code_prompt: str = Field(
        default=(
            "You are a code documentation expert. Answer this programming "
            "question:\n\nQuestion: {query}\n\nProvide a detailed "
            "answer with code examples as would appear in API documentation:"
        ),
        description="Prompt template for code-related queries",
    )

    tutorial_prompt: str = Field(
        default=(
            "You are a tutorial writer. Answer this question as if explaining to "
            "someone learning:"
            "\n\nQuestion: {query}\n\nProvide a step-by-step explanation with examples:"
        ),
        description="Prompt template for tutorial queries",
    )

    general_prompt: str = Field(
        default=(
            "Answer the following question accurately and comprehensively:"
            "\n\nQuestion: {query}\n\nAnswer:"
        ),
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
