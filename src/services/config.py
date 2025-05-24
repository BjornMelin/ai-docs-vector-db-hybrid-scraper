"""Service configuration models using Pydantic v2."""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class ServiceConfig(BaseModel):
    """Base configuration for services."""

    model_config = ConfigDict(extra="allow")


class APIConfig(BaseModel):
    """Direct API configuration with validation."""

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

    model_config = ConfigDict(extra="forbid")

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str | None) -> str | None:
        """Validate OpenAI API key format."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("openai_model")
    @classmethod
    def validate_openai_model(cls, v: str) -> str:
        """Validate OpenAI model name."""
        valid_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if v not in valid_models:
            raise ValueError(f"Invalid OpenAI model. Must be one of: {valid_models}")
        return v

    @field_validator("preferred_embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        """Validate embedding provider name."""
        valid_providers = {"openai", "fastembed"}
        if v not in valid_providers:
            raise ValueError(
                f"Invalid embedding provider. Must be one of: {valid_providers}"
            )
        return v

    @field_validator("preferred_crawl_provider")
    @classmethod
    def validate_crawl_provider(cls, v: str) -> str:
        """Validate crawl provider name."""
        valid_providers = {"firecrawl", "crawl4ai"}
        if v not in valid_providers:
            raise ValueError(
                f"Invalid crawl provider. Must be one of: {valid_providers}"
            )
        return v
