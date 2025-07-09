"""JWT token management with RS256 algorithm and refresh token support.

Provides secure token generation, validation, and rotation capabilities
for enterprise authentication.

This module implements a secure JWT authentication system using RSA
public/private key cryptography (RS256 algorithm). It supports both
access and refresh tokens, token revocation, and permission validation.

Key Features:
    - RSA key pair generation and management
    - Access token generation with short expiration (15 minutes)
    - Refresh token generation with longer expiration (7 days)
    - Token revocation tracking (in-memory, use Redis in production)
    - Permission-based access control
    - Session management support

Security Features:
    - RS256 algorithm for enhanced security over HS256
    - Automatic key generation if not present
    - Token expiration validation
    - JTI (JWT ID) for unique token identification
    - Role-based access control with permission granularity

Example:
    >>> from src.config.security.config import SecurityConfig
    >>> config = SecurityConfig()
    >>> jwt_manager = JWTManager(config)
    >>> # Generate token pair
    >>> tokens = jwt_manager.generate_token_pair(
    ...     user_id=UUID("12345678-1234-5678-1234-567812345678"),
    ...     username="john_doe",
    ...     email="john@example.com",
    ...     role=UserRole.USER,
    ...     permissions=["read:documents", "write:documents"],
    ... )
    >>> # Validate and decode token
    >>> claims = jwt_manager.decode_token(
    ...     tokens.access_token, verify_type=TokenType.ACCESS
    ... )

Note:
    In production, implement token revocation using Redis or a similar
    distributed cache to support multi-instance deployments.
"""

import logging
import time
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.config.security.config import SecurityConfig
from src.services.security.auth.models import (
    TokenClaims,
    TokenPair,
    TokenType,
    UserRole,
)


logger = logging.getLogger(__name__)


class JWTManager:
    """Manages JWT token lifecycle with RSA key pair for enhanced security.

    The JWTManager handles all aspects of JWT token management including
    generation, validation, refresh, and revocation. It uses RSA asymmetric
    cryptography for enhanced security, allowing the public key to be shared
    for token verification while keeping the private key secure.

    Token Types:
        - Access Token: Short-lived (15 min) for API authentication
        - Refresh Token: Long-lived (7 days) for obtaining new access tokens

    Security Considerations:
        - Keys are stored in /tmp in this implementation (use secure storage in production)
        - Token revocation is tracked in memory (use Redis in production)
        - All tokens include JTI for unique identification
        - Supports role-based access control and granular permissions

    Attributes:
        config: Security configuration instance
        algorithm: JWT signing algorithm (RS256)
        access_expiration: Access token lifetime (15 minutes)
        refresh_expiration: Refresh token lifetime (7 days)
        private_key: RSA private key for signing tokens
        public_key: RSA public key for verifying tokens
        revoked_tokens: Set of revoked token JTIs

    Example:
        >>> # Initialize manager
        >>> manager = JWTManager(security_config)
        >>> # Generate tokens for a user
        >>> tokens = manager.generate_token_pair(
        ...     user_id=user.id,
        ...     username=user.username,
        ...     email=user.email,
        ...     role=UserRole.ADMIN,
        ...     permissions=["admin:all"],
        ... )
        >>> # Later, refresh the tokens
        >>> new_tokens = await manager.refresh_tokens(
        ...     tokens.refresh_token, session_id="session_123"
        ... )
    """

    def __init__(self, config: SecurityConfig):
        """Initialize JWT manager with security configuration.

        Args:
            config: Security configuration containing JWT settings,
                including algorithm preferences and token lifetimes

        Note:
            Upon initialization, the manager will attempt to load existing
            RSA keys from disk. If keys don't exist, new ones will be
            generated automatically.
        """
        self.config = config
        self.algorithm = "RS256"  # Using RSA for better security
        self.access_expiration = timedelta(minutes=15)
        self.refresh_expiration = timedelta(days=7)

        # Load or generate RSA keys
        self._load_or_generate_keys()

        # Token revocation store (in production, use Redis)
        self.revoked_tokens: set[str] = set()

        logger.info("JWT Manager initialized with RS256 algorithm")

    def _load_or_generate_keys(self) -> None:
        """Load existing RSA keys or generate new ones."""
        private_key_path = Path(
            "/tmp/jwt_private.pem"
        )  # In production, use secure path
        public_key_path = Path("/tmp/jwt_public.pem")

        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            with private_key_path.open("rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            with public_key_path.open("rb") as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
            logger.info("Loaded existing RSA keys")
        else:
            # Generate new keys
            self._generate_rsa_keys(private_key_path, public_key_path)
            logger.info("Generated new RSA keys")

    def _generate_rsa_keys(self, private_key_path: Path, public_key_path: Path) -> None:
        """Generate new RSA key pair for JWT signing."""
        # Generate private key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Generate public key
        self.public_key = self.private_key.public_key()

        # Save private key
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        private_key_path.write_bytes(private_pem)
        private_key_path.chmod(0o600)  # Restrict access        # Save public key
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        public_key_path.write_bytes(public_pem)

    def generate_token_pair(
        self,
        user_id: UUID,
        username: str,
        email: str,
        role: UserRole,
        permissions: list[str] | None = None,
        session_id: str | None = None,
    ) -> TokenPair:
        """Generate access and refresh token pair.

        Args:
            user_id: User's unique identifier
            username: User's username
            email: User's email
            role: User's role
            permissions: List of permissions
            session_id: Optional session ID

        Returns:
            TokenPair with access and refresh tokens
        """
        current_time = int(time.time())
        jti_access = str(uuid4())
        jti_refresh = str(uuid4())

        # Create access token claims
        access_claims = TokenClaims(
            sub=str(user_id),
            username=username,
            email=email,
            role=role,
            permissions=permissions or [],
            exp=current_time + int(self.access_expiration.total_seconds()),
            iat=current_time,
            nbf=current_time,
            jti=jti_access,
            token_type=TokenType.ACCESS,
            session_id=session_id,
        )

        # Create refresh token claims
        refresh_claims = TokenClaims(
            sub=str(user_id),
            username=username,
            email=email,
            role=role,
            permissions=permissions or [],
            exp=current_time + int(self.refresh_expiration.total_seconds()),
            iat=current_time,
            nbf=current_time,
            jti=jti_refresh,
            token_type=TokenType.REFRESH,
            session_id=session_id,
        )

        # Generate tokens
        access_token = self._encode_token(access_claims)
        refresh_token = self._encode_token(refresh_claims)

        logger.info(f"Generated token pair for user {username} (ID: {user_id})")

        # Standard OAuth2 token type
        token_type = "Bearer"  # noqa: S105
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_expiration.total_seconds()),
            token_type=token_type,
        )

    def _encode_token(self, claims: TokenClaims) -> str:
        """Encode token claims into JWT.

        Args:
            claims: Token claims to encode

        Returns:
            Encoded JWT token
        """
        # Convert claims to dict, excluding None values
        payload = {k: v for k, v in claims.model_dump().items() if v is not None}

        # Encode with private key
        return jwt.encode(payload, self.private_key, algorithm=self.algorithm)

    def decode_token(
        self, token: str, verify_type: TokenType | None = None
    ) -> TokenClaims:
        """Decode and validate JWT token.

        Args:
            token: JWT token to decode
            verify_type: Expected token type

        Returns:
            Decoded token claims

        Raises:
            ValidationError: If token is invalid
        """
        try:
            # Decode with public key
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True, "verify_nbf": True},
            )

            # Parse into TokenClaims
            claims = TokenClaims(**payload)

            # Verify token type if specified
            if verify_type and claims.token_type != verify_type:
                msg = (
                    f"Invalid token type. Expected {verify_type.value}, "
                    f"got {claims.token_type.value}"
                )
                raise ValidationError(
                    msg,
                    error_code="invalid_token_type",
                )

            # Check if token is revoked
            if self._is_token_revoked(claims.jti):
                msg = "Token has been revoked"
                raise ValidationError(
                    msg,
                    error_code="token_revoked",
                )

        except jwt.ExpiredSignatureError as e:
            msg = "Token has expired"
            raise ValidationError(
                msg,
                error_code="token_expired",
            ) from e
        except jwt.InvalidTokenError as e:
            msg = "Invalid token"
            raise ValidationError(
                msg,
                error_code="invalid_token",
                context={"details": str(e)},
            ) from e
        else:
            return claims

    def refresh_tokens(
        self, refresh_token: str, session_id: str | None = None
    ) -> TokenPair:
        """Generate new token pair using refresh token.

        Args:
            refresh_token: Valid refresh token
            session_id: Optional session ID

        Returns:
            New token pair

        Raises:
            ValidationError: If refresh token is invalid
        """
        # Decode and validate refresh token
        claims = self.decode_token(refresh_token, TokenType.REFRESH)

        # Optionally verify session
        if session_id and claims.session_id != session_id:
            msg = "Session ID mismatch"
            raise ValidationError(
                msg,
                error_code="session_mismatch",
            )

        # Revoke old refresh token
        self.revoke_token(claims.jti)

        # Generate new token pair
        return self.generate_token_pair(
            user_id=UUID(claims.sub),
            username=claims.username,
            email=claims.email,
            role=claims.role,
            permissions=claims.permissions,
            session_id=session_id or claims.session_id,
        )

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI.

        Args:
            jti: JWT ID to revoke
        """
        self.revoked_tokens.add(jti)
        logger.info(f"Revoked token with JTI: {jti}")

    def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        return jti in self.revoked_tokens

    def validate_permissions(
        self, claims: TokenClaims, required_permissions: list[str]
    ) -> bool:
        """Validate if token has required permissions.

        Args:
            claims: Token claims
            required_permissions: List of required permissions

        Returns:
            True if all permissions are granted
        """
        # Admin has all permissions
        if claims.role == UserRole.ADMIN:
            return True

        # Check individual permissions
        user_permissions = set(claims.permissions or [])
        return all(perm in user_permissions for perm in required_permissions)

    def get_public_key_pem(self) -> str:
        """Get public key in PEM format for sharing.

        Returns:
            Public key in PEM format
        """
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
