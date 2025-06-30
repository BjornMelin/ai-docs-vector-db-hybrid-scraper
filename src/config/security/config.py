#!/usr/bin/env python3
"""Security configuration for the AI documentation system."""

from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """Security configuration for the AI documentation system."""

    enabled: bool = Field(default=True, description="Enable security features")

    # Rate limiting configuration
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    default_rate_limit: int = Field(
        default=100, description="Default rate limit per window"
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )
    burst_factor: float = Field(
        default=1.5, description="Burst factor for rate limiting"
    )

    # API security
    api_key_required: bool = Field(
        default=False, description="Require API key for access"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # CORS configuration
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE"],
        description="Allowed HTTP methods",
    )
    allowed_headers: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed headers"
    )

    # Authentication
    jwt_secret_key: str | None = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(
        default=30, description="JWT expiration in minutes"
    )

    # Monitoring and logging
    security_logging_enabled: bool = Field(
        default=True, description="Enable security logging"
    )
    audit_log_enabled: bool = Field(default=True, description="Enable audit logging")
    monitoring_enabled: bool = Field(
        default=True, description="Enable security monitoring"
    )

    # AI-specific security
    prompt_injection_detection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    content_filtering: bool = Field(
        default=True, description="Enable content filtering"
    )

    # Input validation
    max_request_size: int = Field(
        default=10 * 1024 * 1024, description="Max request size in bytes"
    )
    max_content_length: int = Field(
        default=1024 * 1024, description="Max content length in bytes"
    )

    # HTTP Security Headers
    x_frame_options: str | None = Field(
        default="DENY", description="X-Frame-Options header value"
    )
    x_content_type_options: str | None = Field(
        default="nosniff", description="X-Content-Type-Options header value"
    )
    x_xss_protection: str | None = Field(
        default="1; mode=block", description="X-XSS-Protection header value"
    )
    strict_transport_security: str | None = Field(
        default="max-age=31536000; includeSubDomains",
        description="Strict-Transport-Security header value",
    )
    content_security_policy: str | None = Field(
        default="default-src 'self'", description="Content-Security-Policy header value"
    )

    # Redis configuration for distributed features
    redis_url: str | None = Field(
        default=None, description="Redis URL for distributed features"
    )
    redis_password: str | None = Field(default=None, description="Redis password")

    class Config:
        """Pydantic configuration."""

        env_prefix = "SECURITY_"
        case_sensitive = False
