#!/usr/bin/env python3
"""Security configuration for the AI documentation system."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ConfigAccessLevel(str, Enum):
    """Configuration access levels."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SYSTEM = "system"


class ConfigDataClassification(str, Enum):
    """Data classification levels for configuration items."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"  # noqa: S105


class ConfigOperationType(str, Enum):
    """Types of configuration operations for auditing."""

    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    BACKUP = "backup"
    RESTORE = "restore"
    VALIDATE = "validate"


class ConfigurationAuditEvent(BaseModel):
    """Represents a configuration audit event."""

    operation: ConfigOperationType
    timestamp: datetime = Field(default_factory=datetime.now)
    data_classification: ConfigDataClassification
    user: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EncryptedConfigItem(BaseModel):
    """Represents an encrypted configuration item."""

    key: str
    encrypted_value: str
    data_classification: ConfigDataClassification
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecureConfigManager:
    """Manages secure configuration with encryption and auditing."""

    def __init__(self, security_config: "SecurityConfig", config_dir: Path):
        """Initialize the secure config manager."""
        self.security_config = security_config
        self.config_dir = config_dir
        self.audit_events: list[ConfigurationAuditEvent] = []

    def encrypt_value(
        self, value: str, _classification: ConfigDataClassification
    ) -> str:
        """Encrypt a configuration value."""
        # Placeholder implementation for testing
        return f"encrypted:{value}"

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        # Placeholder implementation for testing
        if encrypted_value.startswith("encrypted:"):
            return encrypted_value[10:]
        return encrypted_value

    def audit_operation(self, event: ConfigurationAuditEvent) -> None:
        """Record an audit event."""
        self.audit_events.append(event)


class SecurityConfig(BaseModel):
    """Security configuration for the AI documentation system."""

    enabled: bool = Field(default=True, description="Enable security features")

    # Enhanced security features
    enable_config_encryption: bool = Field(
        default=True, description="Enable configuration encryption"
    )
    audit_config_access: bool = Field(
        default=True, description="Audit configuration access"
    )
    enable_config_integrity_checks: bool = Field(
        default=True, description="Enable config integrity checks"
    )
    integrate_security_monitoring: bool = Field(
        default=True, description="Integrate security monitoring"
    )

    # Rate limiting configuration
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    default_rate_limit: int = Field(
        default=100, description="Default rate limit per window"
    )
    rate_limit_window: int = Field(
        default=3600, description="Rate limit window in seconds"
    )
    burst_factor: float = Field(
        default=1.5, description="Burst factor for rate limiting"
    )

    # Security headers
    x_frame_options: str = Field(default="DENY", description="X-Frame-Options header")
    x_content_type_options: str = Field(
        default="nosniff", description="X-Content-Type-Options header"
    )
    x_xss_protection: str = Field(
        default="1; mode=block", description="X-XSS-Protection header"
    )
    strict_transport_security: str = Field(
        default="max-age=31536000; includeSubDomains; preload",
        description="Strict-Transport-Security header",
    )
    content_security_policy: str = Field(
        default=(
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        ),
        description="Content-Security-Policy header",
    )

    # Encryption configuration
    encryption_key_rotation_days: int = Field(
        default=90, description="Encryption key rotation interval"
    )
    use_hardware_security_module: bool = Field(
        default=False, description="Use hardware security module"
    )

    # Secrets management
    secrets_provider: str = Field(default="environment", description="Secrets provider")
    vault_mount_path: str = Field(default="secret", description="Vault mount path")

    # Data classification
    default_data_classification: ConfigDataClassification = Field(
        default=ConfigDataClassification.INTERNAL,
        description="Default data classification level",
    )
    require_encrypted_transmission: bool = Field(
        default=True, description="Require encrypted transmission"
    )

    # API security
    api_key_required: bool = Field(
        default=False, description="Require API key for access"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # CORS configuration
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "https://localhost:3000",
            "https://localhost:8000",
        ],
        description="Allowed CORS origins - restrict to known domains for security",
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

    # Additional security configuration
    # Note: HTTP Security Headers are already defined above
    additional_content_security_policy: str | None = Field(
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
