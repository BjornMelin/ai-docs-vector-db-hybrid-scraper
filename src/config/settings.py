"""Simplified configuration using pydantic-settings v2.

This module provides all configuration with built-in .env support,
proper validation, and secure handling of sensitive data.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment and basic settings
    environment: str = Field(
        default="development", pattern="^(development|testing|staging|production)$"
    )
    debug: bool = Field(default=False)
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    app_name: str = Field(default="AI Documentation Vector DB")
    version: str = Field(default="0.1.0")

    # Paths
    data_dir: Path = Field(default=Path("data"))
    cache_dir: Path = Field(default=Path("cache"))
    logs_dir: Path = Field(default=Path("logs"))

    # API Keys (all as SecretStr)
    openai_api_key: SecretStr | None = Field(default=None)
    firecrawl_api_key: SecretStr | None = Field(default=None)
    qdrant_api_key: SecretStr | None = Field(default=None)
    dragonfly_password: SecretStr | None = Field(default=None)
    flagsmith_api_key: SecretStr | None = Field(default=None)
    flagsmith_environment_key: SecretStr | None = Field(default=None)
    vault_token: SecretStr | None = Field(default=None)
    config_admin_api_key: SecretStr | None = Field(default=None)

    # Database URLs (as SecretStr for security)
    database_url: SecretStr = Field(
        default=SecretStr("sqlite+aiosqlite:///data/app.db")
    )
    dragonfly_url: SecretStr = Field(default=SecretStr("redis://localhost:6379"))
    redis_url: SecretStr = Field(default=SecretStr("redis://localhost:6379"))

    # Service URLs
    qdrant_url: str = Field(default="http://localhost:6333")
    firecrawl_api_url: str = Field(default="https://api.firecrawl.dev")
    vault_url: str | None = Field(default=None)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    flagsmith_api_url: str = Field(default="https://edge.api.flagsmith.com/api/v1/")

    # Provider preferences
    embedding_provider: str = Field(default="fastembed", pattern="^(openai|fastembed)$")
    crawl_provider: str = Field(default="crawl4ai", pattern="^(firecrawl|crawl4ai)$")

    # Cache settings
    enable_caching: bool = Field(default=True)
    enable_local_cache: bool = Field(default=True)
    enable_dragonfly_cache: bool = Field(default=False)
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    local_max_size: int = Field(default=1000, gt=0)
    local_max_memory_mb: int = Field(default=100, gt=0)

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536, gt=0, le=3072)
    embedding_batch_size: int = Field(default=100, gt=0, le=2048)
    fastembed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Chunking settings
    chunk_size: int = Field(default=1600, gt=0)
    chunk_overlap: int = Field(default=320, ge=0)
    chunking_strategy: str = Field(
        default="enhanced", pattern="^(basic|enhanced|ast_aware)$"
    )

    # Search settings
    search_strategy: str = Field(default="dense", pattern="^(dense|sparse|hybrid)$")
    max_search_results: int = Field(default=10, gt=0, le=100)

    # Performance settings
    max_concurrent_requests: int = Field(default=10, gt=0, le=100)
    request_timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Security settings
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    require_api_keys: bool = Field(default=True)
    api_key_header: str = Field(default="X-API-Key")
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, gt=0)

    # Feature flags
    enable_monitoring: bool = Field(default=True)
    enable_observability: bool = Field(default=False)
    enable_rag: bool = Field(default=False)
    enable_hyde: bool = Field(default=True)
    enable_feature_flags: bool = Field(default=True)
    enable_auto_detection: bool = Field(default=True)
    enable_drift_detection: bool = Field(default=True)

    # Deployment tier
    deployment_tier: str = Field(
        default="enterprise", pattern="^(personal|professional|enterprise)$"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
        # Use simple nested delimiter (double underscore)
        env_nested_delimiter="__",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate OpenAI API key format."""
        if v and not v.get_secret_value().startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("firecrawl_api_key")
    @classmethod
    def validate_firecrawl_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Firecrawl API key format."""
        if v and not v.get_secret_value().startswith("fc-"):
            raise ValueError("Firecrawl API key must start with 'fc-'")
        return v

    def model_post_init(self, __context) -> None:
        """Create required directories after initialization."""
        for dir_path in [self.data_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def requires_openai(self) -> bool:
        """Check if OpenAI API key is required."""
        return self.embedding_provider == "openai" and not self.openai_api_key

    def requires_firecrawl(self) -> bool:
        """Check if Firecrawl API key is required."""
        return self.crawl_provider == "firecrawl" and not self.firecrawl_api_key

    def get_database_url(self) -> str:
        """Get the database URL as a string."""
        return self.database_url.get_secret_value()

    def get_redis_url(self) -> str:
        """Get the Redis/Dragonfly URL as a string."""
        return self.dragonfly_url.get_secret_value()

    def to_dict(self, exclude_secrets: bool = True) -> dict:
        """Convert settings to dictionary, optionally excluding secrets."""
        data = self.model_dump()
        if exclude_secrets:
            # Replace SecretStr values with masked versions
            for key, value in data.items():
                if isinstance(getattr(self.__class__, key, None), type) and issubclass(
                    getattr(self.__class__, key), SecretStr
                ):
                    data[key] = "***" if value else None
        return data


# Global settings instance cache
_settings: Settings | None = None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    get_settings.cache_clear()
    _settings = Settings()
    return _settings


# Export main items
__all__ = ["Settings", "get_settings", "reload_settings"]
