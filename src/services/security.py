"""Simplified security service using FastAPI's built-in features.

This module provides basic security utilities including:
- JWT token creation and validation
- Password hashing with passlib
- FastAPI dependency injection for auth
- Basic input validation for security
"""

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, Field, SecretStr


logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class TokenData(BaseModel):
    """JWT token payload data."""

    sub: str
    exp: datetime
    iat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    jti: str | None = None  # JWT ID for revocation support


class UserCredentials(BaseModel):
    """User login credentials."""

    username: str
    password: SecretStr


class TokenResponse(BaseModel):
    """API token response."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


# Password utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# JWT utilities
def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
) -> str:
    """Create a JWT access token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Custom expiration time
        additional_claims: Extra claims to include

    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(hours=JWT_EXPIRATION_HOURS)

    claims = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(UTC),
    }

    if additional_claims:
        claims.update(additional_claims)

    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token.

    Args:
        token: JWT token to decode

    Returns:
        Token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(**payload)
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# FastAPI dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),  # noqa: B008
) -> str:
    """Get current authenticated user from JWT token.

    Args:
        credentials: Bearer token from request

    Returns:
        User ID from token

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_token(credentials.credentials)
    return token_data.sub


async def require_auth(user_id: str = Depends(get_current_user)) -> str:
    """Require authentication for an endpoint.

    Args:
        user_id: Current user from token

    Returns:
        Authenticated user ID

    Raises:
        HTTPException: If not authenticated
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user_id


# Optional: API key authentication
async def verify_api_key(api_key: str) -> bool:
    """Verify an API key (implement your logic).

    Args:
        api_key: API key to verify

    Returns:
        True if valid
    """
    # Simple example - replace with actual verification
    return api_key == "your-api-key"


# Rate limiting helper (basic example)
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[datetime]] = {}

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier has exceeded rate limit."""
        now = datetime.now(UTC)
        minute_ago = now - timedelta(minutes=1)

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] if req_time > minute_ago
        ]

        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False

        self.requests[identifier].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


# Basic input validation for compatibility
class SecurityValidator:
    """Basic security validation for URLs and inputs."""

    ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https"}
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        r"javascript:",
        r"data:",
        r"file:",
        r"localhost",
        r"127\.0\.0\.1",
        r"192\.168\.",
        r"10\.",
        r"::1",
    ]

    def validate_url(self, url: str) -> str:
        """Validate URL for basic security threats."""
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")

        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}") from e

        if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
            raise ValueError(f"URL scheme '{parsed.scheme}' not allowed")

        url_lower = url.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, url_lower):
                raise ValueError("URL contains dangerous pattern")

        return url.strip()

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe operations."""
        if not filename:
            return "safe_filename"

        filename = Path(filename.strip()).name
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
        return filename[:255] if len(filename) > 255 else filename

    @classmethod
    def from_unified_config(cls) -> "SecurityValidator":
        """Create instance (for compatibility)."""
        return cls()


# Example usage in routes:
# @router.post("/login", response_model=TokenResponse)
# async def login(credentials: UserCredentials):
#     # Verify credentials against your user store
#     if not verify_password(credentials.password.get_secret_value(), stored_hash):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#
#     token = create_access_token(subject=user_id)
#     return TokenResponse(
#         access_token=token,
#         expires_in=JWT_EXPIRATION_HOURS * 3600
#     )
#
# @router.get("/protected")
# async def protected_route(user_id: str = Depends(require_auth)):
#     return {"message": f"Hello user {user_id}"}
