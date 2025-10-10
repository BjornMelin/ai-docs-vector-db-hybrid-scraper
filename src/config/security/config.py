"""Pydantic security configuration models for FastAPI middleware."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecurityConfig(BaseSettings):
    """Application security settings consumed by FastAPI middleware."""

    enabled: bool = Field(
        default=True,
        description="Enable security middleware features.",
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable IP-based rate limiting.",
    )
    default_rate_limit: int = Field(
        default=100,
        gt=0,
        description="Maximum number of requests allowed per client within a window.",
    )
    rate_limit_window: int = Field(
        default=60,
        gt=0,
        description="Rate limit window duration in seconds.",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL used for distributed rate limiting.",
    )
    redis_password: str | None = Field(
        default=None,
        description="Optional Redis password used for authentication.",
    )
    x_frame_options: str = Field(
        default="DENY",
        description="Value for the X-Frame-Options response header.",
    )
    x_content_type_options: str = Field(
        default="nosniff",
        description="Value for the X-Content-Type-Options response header.",
    )
    x_xss_protection: str = Field(
        default="1; mode=block",
        description="Value for the X-XSS-Protection response header.",
    )
    strict_transport_security: str = Field(
        default="max-age=31536000; includeSubDomains; preload",
        description="Value for the Strict-Transport-Security response header.",
    )
    content_security_policy: str = Field(
        default="default-src 'self'; script-src 'self'; style-src 'self'",
        description="Value for the Content-Security-Policy response header.",
    )
    api_key_required: bool = Field(
        default=False,
        description="Require an API key for accessing configuration endpoints.",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name used to supply API keys.",
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="List of API keys considered valid for configuration access.",
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed origins for CORS responses when the API runs in production."
        ),
    )
    enable_ml_input_validation: bool = Field(
        default=True,
        description="Enable MLSecurityValidator input/type checks.",
    )
    max_ml_input_size: int = Field(
        default=1_000_000,
        gt=0,
        description="Maximum size (characters) for ML payload validation.",
    )
    suspicious_patterns: list[str] = Field(
        default_factory=lambda: ["<script", "DROP TABLE", "__import__", "eval("],
        description="Patterns blocked during ML input validation.",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Optional allow list for outbound URLs (exact match or suffix).",
    )
    blocked_domains: list[str] = Field(
        default_factory=list,
        description="Domains that MLSecurityValidator will always reject.",
    )
    max_query_length: int = Field(
        default=1_000,
        gt=0,
        description="Maximum length for end-user query strings.",
    )
    enable_dependency_scanning: bool = Field(
        default=True,
        description="Enable pip-audit dependency scanning in MLSecurityValidator.",
    )

    model_config = SettingsConfigDict(env_prefix="SECURITY_", case_sensitive=False)


__all__ = ["SecurityConfig"]
