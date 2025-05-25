"""Service configuration adapter using the unified configuration system.

This module provides backward compatibility for services while using the new
unified configuration system under the hood.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
try:
    from src.config import get_config
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import get_config


class ServiceConfig(BaseModel):
    """Base configuration for services."""

    model_config = ConfigDict(extra="allow")


class APIConfig(BaseModel):
    """Direct API configuration with validation - adapter for unified config."""

    # This class now acts as an adapter to the unified configuration
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_unified_config(cls) -> "APIConfig":
        """Create APIConfig from unified configuration."""
        config = get_config()

        return cls(
            # Qdrant
            qdrant_url=config.qdrant.url,
            qdrant_api_key=config.qdrant.api_key,
            qdrant_timeout=config.qdrant.timeout,
            qdrant_prefer_grpc=config.qdrant.prefer_grpc,
            # OpenAI
            openai_api_key=config.openai.api_key,
            openai_model=config.openai.model,
            openai_dimensions=config.openai.dimensions,
            openai_batch_size=config.openai.batch_size,
            # Firecrawl
            firecrawl_api_key=config.firecrawl.api_key,
            # Local models
            enable_local_embeddings=config.embedding_provider == "fastembed",
            local_embedding_model=config.fastembed.model,
            # Provider preferences
            preferred_embedding_provider=config.embedding_provider,
            preferred_crawl_provider=config.crawl_provider,
            # Performance
            max_concurrent_requests=config.performance.max_concurrent_requests,
            request_timeout=config.performance.request_timeout,
            # Retry settings
            max_retries=config.performance.max_retries,
            retry_base_delay=config.performance.retry_base_delay,
            # Cache settings
            enable_caching=config.cache.enable_caching,
            enable_local_cache=config.cache.enable_local_cache,
            enable_redis_cache=config.cache.enable_redis_cache,
            redis_url=config.cache.redis_url,
            cache_ttl_embeddings=config.cache.ttl_embeddings,
            cache_ttl_crawl=config.cache.ttl_crawl,
            cache_ttl_queries=config.cache.ttl_queries,
            local_cache_max_size=config.cache.local_max_size,
            local_cache_max_memory_mb=config.cache.local_max_memory_mb,
        )

    # Keep the same field definitions for backward compatibility
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_timeout: float = Field(default=30.0, gt=0)
    qdrant_prefer_grpc: bool = Field(default=False)

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="text-embedding-3-small")
    openai_dimensions: int = Field(default=1536, gt=0, le=3072)
    openai_batch_size: int = Field(default=100, gt=0, le=2048)

    # Firecrawl
    firecrawl_api_key: str | None = Field(default=None)

    # Local models
    enable_local_embeddings: bool = Field(default=True)
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Provider preferences
    preferred_embedding_provider: str = Field(default="fastembed")
    preferred_crawl_provider: str = Field(default="crawl4ai")

    # Performance
    max_concurrent_requests: int = Field(default=10, gt=0, le=100)
    request_timeout: float = Field(default=30.0, gt=0)

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0)

    # Cache settings (V1 MVP)
    enable_caching: bool = Field(default=True)
    enable_local_cache: bool = Field(default=True)
    enable_redis_cache: bool = Field(default=True)
    redis_url: str = Field(default="redis://localhost:6379")
    cache_ttl_embeddings: int = Field(default=86400, ge=0)  # 24 hours
    cache_ttl_crawl: int = Field(default=3600, ge=0)  # 1 hour
    cache_ttl_queries: int = Field(default=7200, ge=0)  # 2 hours
    local_cache_max_size: int = Field(default=1000, gt=0)
    local_cache_max_memory_mb: float = Field(default=100.0, gt=0)
