"""Service configuration models using Pydantic v2."""

from pydantic import BaseModel
from pydantic import Field


class ServiceConfig(BaseModel):
    """Base configuration for services."""

    model_config = {"extra": "allow"}


class APIConfig(BaseModel):
    """Direct API configuration."""

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_timeout: float = Field(default=30.0)
    qdrant_prefer_grpc: bool = Field(default=False)

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="text-embedding-3-small")
    openai_dimensions: int = Field(default=1536)
    openai_batch_size: int = Field(default=100)

    # Firecrawl
    firecrawl_api_key: str | None = Field(default=None)

    # Local models
    enable_local_embeddings: bool = Field(default=True)
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Provider preferences
    preferred_embedding_provider: str = Field(default="fastembed")
    preferred_crawl_provider: str = Field(default="crawl4ai")

    # Performance
    max_concurrent_requests: int = Field(default=10)
    request_timeout: float = Field(default=30.0)

    # Retry settings
    max_retries: int = Field(default=3)
    retry_base_delay: float = Field(default=1.0)

    model_config = {"extra": "forbid"}
