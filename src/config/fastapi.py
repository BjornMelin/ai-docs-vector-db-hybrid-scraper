"""FastAPI production configuration for middleware and server settings.

This module extends the unified configuration with FastAPI-specific production settings
for middleware, security, performance, and deployment.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional

from .enums import Environment


class CORSConfig(BaseModel):
    """CORS middleware configuration with environment-specific defaults."""
    
    enabled: bool = Field(default=True, description="Enable CORS middleware")
    allow_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        description="Allowed origins for CORS"
    )
    allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed headers"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    max_age: int = Field(default=3600, description="Preflight cache duration (seconds)")


class CompressionConfig(BaseModel):
    """Compression middleware configuration."""
    
    enabled: bool = Field(default=True, description="Enable GZip compression")
    minimum_size: int = Field(default=1000, gt=0, description="Minimum response size to compress (bytes)")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level (1-9)")


class SecurityConfig(BaseModel):
    """Security headers and protection configuration."""
    
    enabled: bool = Field(default=True, description="Enable security middleware")
    
    # Security headers
    x_frame_options: str = Field(default="DENY", description="X-Frame-Options header")
    x_content_type_options: str = Field(default="nosniff", description="X-Content-Type-Options header")
    x_xss_protection: str = Field(default="1; mode=block", description="X-XSS-Protection header")
    strict_transport_security: Optional[str] = Field(
        default="max-age=31536000; includeSubDomains",
        description="Strict-Transport-Security header"
    )
    content_security_policy: Optional[str] = Field(
        default="default-src 'self'",
        description="Content-Security-Policy header"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, gt=0, description="Requests per minute per IP")
    rate_limit_window: int = Field(default=60, gt=0, description="Rate limit window (seconds)")


class TracingConfig(BaseModel):
    """Request tracing and correlation configuration."""
    
    enabled: bool = Field(default=True, description="Enable request tracing")
    correlation_id_header: str = Field(default="X-Correlation-ID", description="Correlation ID header name")
    generate_correlation_id: bool = Field(default=True, description="Auto-generate correlation ID if missing")
    log_requests: bool = Field(default=True, description="Log request details")
    log_responses: bool = Field(default=False, description="Log response details (be careful with sensitive data)")


class TimeoutConfig(BaseModel):
    """Request timeout and circuit breaker configuration."""
    
    enabled: bool = Field(default=True, description="Enable timeout middleware")
    request_timeout: float = Field(default=30.0, gt=0, description="Request timeout (seconds)")
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker")
    failure_threshold: int = Field(default=5, gt=0, description="Consecutive failures before opening circuit")
    recovery_timeout: float = Field(default=60.0, gt=0, description="Time before trying to close circuit (seconds)")
    half_open_max_calls: int = Field(default=3, gt=0, description="Max calls in half-open state")


class PerformanceConfig(BaseModel):
    """Performance monitoring and metrics configuration."""
    
    enabled: bool = Field(default=True, description="Enable performance monitoring")
    track_response_time: bool = Field(default=True, description="Track response times")
    track_memory_usage: bool = Field(default=False, description="Track memory usage (overhead)")
    slow_request_threshold: float = Field(default=1.0, gt=0, description="Slow request threshold (seconds)")


class FastAPIProductionConfig(BaseModel):
    """Complete FastAPI production configuration."""
    
    # Core server settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment type")
    debug: bool = Field(default=False, description="Debug mode")
    server_name: str = Field(default="AI Docs Vector DB", description="Server name")
    version: str = Field(default="1.0.0", description="API version")
    
    # Middleware configurations
    cors: CORSConfig = Field(default_factory=CORSConfig, description="CORS configuration")
    compression: CompressionConfig = Field(default_factory=CompressionConfig, description="Compression configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    tracing: TracingConfig = Field(default_factory=TracingConfig, description="Tracing configuration")
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig, description="Timeout configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    
    # Production deployment settings
    workers: int = Field(default=4, gt=0, description="Number of worker processes")
    max_requests: Optional[int] = Field(default=1000, description="Max requests per worker before restart")
    max_requests_jitter: Optional[int] = Field(default=100, description="Random jitter for max_requests")
    
    def get_environment_specific_cors(self) -> List[str]:
        """Get CORS origins based on environment."""
        if self.environment == Environment.PRODUCTION:
            # In production, you'd typically set specific allowed origins
            return [
                "https://yourdomain.com",
                "https://api.yourdomain.com"
            ]
        elif self.environment == Environment.TESTING:
            return [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://testserver"
            ]
        else:  # Development
            return [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ]
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers based on configuration."""
        headers = {}
        
        if not self.security.enabled:
            return headers
            
        headers["X-Frame-Options"] = self.security.x_frame_options
        headers["X-Content-Type-Options"] = self.security.x_content_type_options
        headers["X-XSS-Protection"] = self.security.x_xss_protection
        
        if self.security.strict_transport_security and self.is_production():
            headers["Strict-Transport-Security"] = self.security.strict_transport_security
            
        if self.security.content_security_policy:
            headers["Content-Security-Policy"] = self.security.content_security_policy
            
        return headers

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Validate worker count is reasonable."""
        import os
        cpu_count = os.cpu_count() or 1
        
        if v > cpu_count * 2:
            raise ValueError(f"Worker count ({v}) should not exceed 2x CPU count ({cpu_count * 2})")
        
        return v


def get_fastapi_config() -> FastAPIProductionConfig:
    """Get FastAPI production configuration instance."""
    return FastAPIProductionConfig()