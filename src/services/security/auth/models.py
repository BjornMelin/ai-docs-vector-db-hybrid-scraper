"""Authentication and authorization data models.

Provides Pydantic models for authentication flows, token management,
and authorization structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, SecretStr


class AuthenticationMethod(str, Enum):
    """Supported authentication methods."""

    PASSWORD = "password"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SAML = "saml"


class UserRole(str, Enum):
    """System user roles with hierarchical permissions."""

    ADMIN = "admin"  # Full system access
    OPERATOR = "operator"  # Operational access
    ANALYST = "analyst"  # Read and query access
    API_USER = "api_user"  # API-only access


class UserCredentials(BaseModel):
    """User login credentials."""

    username: str = Field(..., min_length=3, max_length=255)
    password: SecretStr = Field(..., min_length=8)
    method: AuthenticationMethod = AuthenticationMethod.PASSWORD


class TokenType(str, Enum):
    """JWT token types."""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenClaims(BaseModel):
    """JWT token claims following RFC 7519."""

    # Standard claims
    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration time (Unix timestamp)")
    iat: int = Field(..., description="Issued at time (Unix timestamp)")
    jti: str = Field(..., description="JWT ID for revocation")

    # Custom claims
    user_id: UUID
    username: str
    email: EmailStr
    role: UserRole
    permissions: list[str] = Field(default_factory=list)
    token_type: TokenType
    session_id: str | None = None


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = Field(..., description="Access token expiry in seconds")
    refresh_expires_in: int = Field(..., description="Refresh token expiry in seconds")


class AuthenticationResult(BaseModel):
    """Complete authentication response."""

    success: bool
    user_id: UUID | None = None
    tokens: TokenPair | None = None
    user_role: UserRole | None = None
    permissions: list[str] = Field(default_factory=list)
    session_id: str | None = None
    error_message: str | None = None
    requires_2fa: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class User(BaseModel):
    """User model for authentication."""

    id: UUID
    username: str
    email: EmailStr
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """User session model."""

    id: str
    user_id: UUID
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class APIKey(BaseModel):
    """API key model."""

    id: UUID
    key_prefix: str = Field(..., description="First 8 chars of key for identification")
    user_id: UUID
    name: str
    scopes: list[str] = Field(default_factory=list)
    rate_limit: int | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    created_at: datetime
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
