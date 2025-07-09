"""API Key management system for service authentication.

This module implements a comprehensive API key management system with:
- Secure key generation and storage
- Fine-grained permissions and scoping
- Automatic expiration and rotation
- Usage tracking and rate limiting
- Integration with RBAC system
- Audit logging for all operations

Following zero-trust architecture principles with cryptographic security.
"""

import base64
import logging
import secrets
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, field_validator

from src.services.errors import ServiceError, ValidationError
from src.services.security.audit.logger import SecurityAuditLogger

from .rbac import Permission, RBACManager, Resource


logger = logging.getLogger(__name__)


class APIKeyStatus(str, Enum):
    """API Key lifecycle status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class APIKeyScope(str, Enum):
    """API Key access scopes."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SERVICE = "service"
    INTEGRATION = "integration"


class APIKeyPrefix(str, Enum):
    """API Key prefixes for identification."""

    PRODUCTION = "ak_prod"
    STAGING = "ak_stg"
    DEVELOPMENT = "ak_dev"
    SERVICE = "ak_svc"
    INTEGRATION = "ak_int"


class APIKeyMetadata(BaseModel):
    """API Key metadata and tracking information."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    usage_count: int = Field(default=0)
    last_ip: str | None = None
    last_user_agent: str | None = None
    created_by: str = Field(..., description="User who created this key")
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class APIKeyConfig(BaseModel):
    """API Key configuration and permissions."""

    key_id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="Human-readable key name")
    scope: APIKeyScope = Field(..., description="Key access scope")
    permissions: list[Permission] = Field(default_factory=list)
    allowed_resources: list[Resource] = Field(default_factory=list)
    rate_limit: int | None = Field(default=None, description="Requests per minute")
    ip_whitelist: list[str] = Field(default_factory=list)
    referrer_whitelist: list[str] = Field(default_factory=list)
    status: APIKeyStatus = Field(default=APIKeyStatus.ACTIVE)
    metadata: APIKeyMetadata = Field(default_factory=APIKeyMetadata)

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, v):
        """Validate key ID format."""
        if not v or len(v) < 8:
            msg = "Key ID must be at least 8 characters"
            raise ValueError(msg)
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate key name."""
        if not v or len(v.strip()) < 3:
            msg = "Key name must be at least 3 characters"
            raise ValueError(msg)
        return v.strip()


class APIKey(BaseModel):
    """Complete API Key with secret and configuration."""

    key: str = Field(..., description="Full API key (prefix + secret)")
    key_id: str = Field(..., description="Unique key identifier")
    config: APIKeyConfig = Field(..., description="Key configuration")

    @property
    def prefix(self) -> str:
        """Get API key prefix."""
        return self.key.split("_")[0] + "_" + self.key.split("_")[1]

    @property
    def secret(self) -> str:
        """Get API key secret part."""
        return self.key.split("_", 2)[2]

    def is_valid(self) -> bool:
        """Check if API key is valid and active."""
        if self.config.status != APIKeyStatus.ACTIVE:
            return False

        if self.config.metadata.expires_at:
            if datetime.now(UTC) > self.config.metadata.expires_at:
                return False

        return True

    def can_access_resource(self, resource: Resource, action: Permission) -> bool:
        """Check if key can access specific resource with action."""
        if not self.is_valid():
            return False

        # Check resource access
        if (
            self.config.allowed_resources
            and resource not in self.config.allowed_resources
        ):
            return False

        # Check permission
        return action in self.config.permissions

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP address is allowed."""
        if not self.config.ip_whitelist:
            return True

        # Simple IP matching (production would use CIDR)
        return ip in self.config.ip_whitelist

    def is_referrer_allowed(self, referrer: str) -> bool:
        """Check if referrer is allowed."""
        if not self.config.referrer_whitelist:
            return True

        for allowed_referrer in self.config.referrer_whitelist:
            if referrer.startswith(allowed_referrer):
                return True

        return False


class APIKeyUsage(BaseModel):
    """API Key usage tracking."""

    key_id: str = Field(..., description="API key identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    endpoint: str = Field(..., description="API endpoint accessed")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., description="Response status code")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str | None = None
    bytes_sent: int = Field(default=0)
    bytes_received: int = Field(default=0)

    class Config:
        use_enum_values = True


class APIKeyValidationRequest(BaseModel):
    """API Key validation request."""

    key: str = Field(..., description="API key to validate")
    resource: Resource = Field(..., description="Target resource")
    action: Permission = Field(..., description="Requested action")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str | None = None
    referrer: str | None = None

    class Config:
        use_enum_values = True


class APIKeyValidationResult(BaseModel):
    """API Key validation result."""

    valid: bool = Field(..., description="Whether key is valid")
    key_id: str | None = None
    reason: str | None = None
    permissions: list[Permission] = Field(default_factory=list)
    rate_limit_remaining: int | None = None
    expires_at: datetime | None = None

    class Config:
        use_enum_values = True


class APIKeyManager:
    """Enterprise API Key management system."""

    def __init__(
        self, rbac_manager: RBACManager, audit_logger: SecurityAuditLogger | None = None
    ):
        """Initialize API key manager.

        Args:
            rbac_manager: RBAC manager for permission checking
            audit_logger: Security audit logger
        """
        self.rbac_manager = rbac_manager
        self.audit_logger = audit_logger
        self._keys: dict[str, APIKey] = {}
        self._key_hashes: dict[str, str] = {}  # hash -> key_id mapping
        self._usage_stats: dict[str, list[APIKeyUsage]] = {}

        # Initialize scope-based permissions
        self._scope_permissions = self._initialize_scope_permissions()

    def _initialize_scope_permissions(self) -> dict[APIKeyScope, list[Permission]]:
        """Initialize default permissions for each scope."""
        return {
            APIKeyScope.READ_ONLY: [
                Permission.DOCUMENTS_READ,
                Permission.COLLECTIONS_READ,
                Permission.SEARCH_BASIC,
                Permission.ANALYTICS_READ,
                Permission.SYSTEM_HEALTH,
            ],
            APIKeyScope.READ_WRITE: [
                Permission.DOCUMENTS_READ,
                Permission.DOCUMENTS_WRITE,
                Permission.COLLECTIONS_READ,
                Permission.COLLECTIONS_WRITE,
                Permission.SEARCH_BASIC,
                Permission.SEARCH_ADVANCED,
                Permission.ANALYTICS_READ,
                Permission.SYSTEM_HEALTH,
            ],
            APIKeyScope.ADMIN: [
                Permission.DOCUMENTS_ADMIN,
                Permission.COLLECTIONS_ADMIN,
                Permission.SEARCH_ADVANCED,
                Permission.SEARCH_EXPORT,
                Permission.ANALYTICS_READ,
                Permission.ANALYTICS_EXPORT,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_HEALTH,
                Permission.SYSTEM_LOGS,
                Permission.AUDIT_READ,
            ],
            APIKeyScope.SERVICE: [
                Permission.DOCUMENTS_READ,
                Permission.DOCUMENTS_WRITE,
                Permission.COLLECTIONS_READ,
                Permission.COLLECTIONS_WRITE,
                Permission.SEARCH_BASIC,
                Permission.SEARCH_ADVANCED,
                Permission.SYSTEM_HEALTH,
            ],
            APIKeyScope.INTEGRATION: [
                Permission.DOCUMENTS_READ,
                Permission.SEARCH_BASIC,
                Permission.COLLECTIONS_READ,
                Permission.SYSTEM_HEALTH,
            ],
        }

    def _generate_key_secret(self, length: int = 32) -> str:
        """Generate cryptographically secure key secret."""
        return secrets.token_urlsafe(length)

    def _hash_key(self, key: str) -> str:
        """Hash API key for secure storage."""
        salt = b"api_key_salt"  # In production, use per-key salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key_hash = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        return key_hash.decode()

    def _get_key_prefix(self, environment: str = "production") -> APIKeyPrefix:
        """Get appropriate key prefix based on environment."""
        env_map = {
            "production": APIKeyPrefix.PRODUCTION,
            "staging": APIKeyPrefix.STAGING,
            "development": APIKeyPrefix.DEVELOPMENT,
            "service": APIKeyPrefix.SERVICE,
            "integration": APIKeyPrefix.INTEGRATION,
        }
        return env_map.get(environment, APIKeyPrefix.PRODUCTION)

    async def create_api_key(
        self,
        name: str,
        scope: APIKeyScope,
        created_by: str,
        description: str | None = None,
        expires_in_days: int | None = None,
        custom_permissions: list[Permission] | None = None,
        allowed_resources: list[Resource] | None = None,
        rate_limit: int | None = None,
        ip_whitelist: list[str] | None = None,
        referrer_whitelist: list[str] | None = None,
        environment: str = "production",
    ) -> APIKey:
        """Create new API key.

        Args:
            name: Human-readable key name
            scope: Key access scope
            created_by: User creating the key
            description: Optional description
            expires_in_days: Expiration in days (None for no expiration)
            custom_permissions: Custom permissions (overrides scope defaults)
            allowed_resources: Restricted resources (None for all)
            rate_limit: Requests per minute limit
            ip_whitelist: Allowed IP addresses
            referrer_whitelist: Allowed referrers
            environment: Environment (production, staging, etc.)

        Returns:
            Created API key

        Raises:
            ValidationError: If parameters are invalid
        """
        # Generate key components
        key_id = str(uuid4())
        prefix = self._get_key_prefix(environment)
        secret = self._generate_key_secret()
        full_key = f"{prefix.value}_{secret}"

        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Determine permissions
        permissions = custom_permissions or self._scope_permissions.get(scope, [])

        # Create metadata
        metadata = APIKeyMetadata(
            created_by=created_by, description=description, expires_at=expires_at
        )

        # Create configuration
        config = APIKeyConfig(
            key_id=key_id,
            name=name,
            scope=scope,
            permissions=permissions,
            allowed_resources=allowed_resources or [],
            rate_limit=rate_limit,
            ip_whitelist=ip_whitelist or [],
            referrer_whitelist=referrer_whitelist or [],
            metadata=metadata,
        )

        # Create API key
        api_key = APIKey(key=full_key, key_id=key_id, config=config)

        # Store key
        self._keys[key_id] = api_key
        key_hash = self._hash_key(full_key)
        self._key_hashes[key_hash] = key_id

        # Initialize usage stats
        self._usage_stats[key_id] = []

        # Log creation
        self._log_key_operation(
            "api_key_created", key_id, created_by, {"name": name, "scope": scope.value}
        )

        return api_key

    async def validate_api_key(
        self, request: APIKeyValidationRequest
    ) -> APIKeyValidationResult:
        """Validate API key and check permissions.

        Args:
            request: Validation request

        Returns:
            Validation result
        """
        # Hash the provided key
        key_hash = self._hash_key(request.key)

        # Find key by hash
        key_id = self._key_hashes.get(key_hash)
        if not key_id:
            self._log_key_operation(
                "api_key_validation_failed",
                "unknown",
                "system",
                {"reason": "key_not_found", "ip": request.ip_address},
            )
            return APIKeyValidationResult(valid=False, reason="Invalid API key")

        # Get key configuration
        api_key = self._keys.get(key_id)
        if not api_key:
            return APIKeyValidationResult(valid=False, reason="Key not found")

        # Check key validity
        if not api_key.is_valid():
            reason = "inactive"
            if api_key.config.status == APIKeyStatus.EXPIRED:
                reason = "expired"
            elif api_key.config.status == APIKeyStatus.REVOKED:
                reason = "revoked"

            self._log_key_operation(
                "api_key_validation_failed",
                key_id,
                "system",
                {"reason": reason, "ip": request.ip_address},
            )

            return APIKeyValidationResult(
                valid=False, key_id=key_id, reason=f"Key is {reason}"
            )

        # Check IP whitelist
        if not api_key.is_ip_allowed(request.ip_address):
            self._log_key_operation(
                "api_key_validation_failed",
                key_id,
                "system",
                {"reason": "ip_not_allowed", "ip": request.ip_address},
            )
            return APIKeyValidationResult(
                valid=False, key_id=key_id, reason="IP address not allowed"
            )

        # Check referrer whitelist
        if request.referrer and not api_key.is_referrer_allowed(request.referrer):
            self._log_key_operation(
                "api_key_validation_failed",
                key_id,
                "system",
                {"reason": "referrer_not_allowed", "referrer": request.referrer},
            )
            return APIKeyValidationResult(
                valid=False, key_id=key_id, reason="Referrer not allowed"
            )

        # Check resource and action permissions
        if not api_key.can_access_resource(request.resource, request.action):
            self._log_key_operation(
                "api_key_validation_failed",
                key_id,
                "system",
                {
                    "reason": "insufficient_permissions",
                    "resource": request.resource.value,
                    "action": request.action.value,
                },
            )
            return APIKeyValidationResult(
                valid=False, key_id=key_id, reason="Insufficient permissions"
            )

        # Update usage tracking
        await self._update_key_usage(key_id, request.ip_address, request.user_agent)

        # Log successful validation
        self._log_key_operation(
            "api_key_validated",
            key_id,
            "system",
            {
                "resource": request.resource.value,
                "action": request.action.value,
                "ip": request.ip_address,
            },
        )

        return APIKeyValidationResult(
            valid=True,
            key_id=key_id,
            permissions=api_key.config.permissions,
            expires_at=api_key.config.metadata.expires_at,
        )

    async def revoke_api_key(self, key_id: str, revoked_by: str) -> bool:
        """Revoke API key.

        Args:
            key_id: Key identifier to revoke
            revoked_by: User revoking the key

        Returns:
            True if key was revoked
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            return False

        # Update status
        api_key.config.status = APIKeyStatus.REVOKED

        # Log revocation
        self._log_key_operation(
            "api_key_revoked", key_id, revoked_by, {"name": api_key.config.name}
        )

        return True

    async def rotate_api_key(self, key_id: str, rotated_by: str) -> APIKey | None:
        """Rotate API key (generate new secret).

        Args:
            key_id: Key identifier to rotate
            rotated_by: User rotating the key

        Returns:
            New API key or None if not found
        """
        old_key = self._keys.get(key_id)
        if not old_key:
            return None

        # Generate new secret
        new_secret = self._generate_key_secret()
        new_full_key = f"{old_key.prefix}_{new_secret}"

        # Update key
        old_key.key = new_full_key

        # Update hash mapping
        old_hash = self._hash_key(old_key.key)
        new_hash = self._hash_key(new_full_key)

        # Remove old hash and add new one
        del self._key_hashes[old_hash]
        self._key_hashes[new_hash] = key_id

        # Log rotation
        self._log_key_operation(
            "api_key_rotated", key_id, rotated_by, {"name": old_key.config.name}
        )

        return old_key

    async def list_api_keys(self, created_by: str | None = None) -> list[APIKeyConfig]:
        """List API keys (without secrets).

        Args:
            created_by: Filter by creator (None for all)

        Returns:
            List of API key configurations
        """
        configs = []

        for api_key in self._keys.values():
            if created_by and api_key.config.metadata.created_by != created_by:
                continue

            configs.append(api_key.config)

        return configs

    async def get_api_key_usage(self, key_id: str) -> list[APIKeyUsage]:
        """Get API key usage statistics.

        Args:
            key_id: Key identifier

        Returns:
            List of usage records
        """
        return self._usage_stats.get(key_id, [])

    async def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys.

        Returns:
            Number of keys cleaned up
        """
        now = datetime.now(UTC)
        expired_keys = []

        for key_id, api_key in self._keys.items():
            if (
                api_key.config.metadata.expires_at
                and now > api_key.config.metadata.expires_at
            ):
                expired_keys.append(key_id)

        # Update status for expired keys
        for key_id in expired_keys:
            self._keys[key_id].config.status = APIKeyStatus.EXPIRED

        # Log cleanup
        if expired_keys:
            logger.info(f"Marked {len(expired_keys)} API keys as expired")

        return len(expired_keys)

    async def _update_key_usage(
        self, key_id: str, ip_address: str, user_agent: str | None
    ) -> None:
        """Update key usage tracking.

        Args:
            key_id: Key identifier
            ip_address: Client IP address
            user_agent: Client user agent
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            return

        # Update metadata
        api_key.config.metadata.last_used_at = datetime.now(UTC)
        api_key.config.metadata.usage_count += 1
        api_key.config.metadata.last_ip = ip_address
        api_key.config.metadata.last_user_agent = user_agent

    def _log_key_operation(
        self, operation: str, key_id: str, user: str, context: dict[str, Any]
    ) -> None:
        """Log API key operation for audit trail.

        Args:
            operation: Operation type
            key_id: Key identifier
            user: User performing operation
            context: Additional context
        """
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="api_key_operation",
                user_id=user,
                resource="api_keys",
                action=operation,
                resource_id=key_id,
                context=context,
            )

        # Also log at appropriate level
        if "failed" in operation:
            logger.warning(f"API key operation failed: {operation} for key {key_id}")
        else:
            logger.info(f"API key operation: {operation} for key {key_id}")

    def get_key_by_id(self, key_id: str) -> APIKey | None:
        """Get API key by ID (for internal use).

        Args:
            key_id: Key identifier

        Returns:
            API key or None
        """
        return self._keys.get(key_id)

    def validate_key_permissions(
        self, key_id: str, permissions: list[Permission]
    ) -> bool:
        """Validate if key has all required permissions.

        Args:
            key_id: Key identifier
            permissions: Required permissions

        Returns:
            True if key has all permissions
        """
        api_key = self._keys.get(key_id)
        if not api_key or not api_key.is_valid():
            return False

        key_permissions = set(api_key.config.permissions)
        required_permissions = set(permissions)

        return required_permissions.issubset(key_permissions)

    def get_key_stats(self) -> dict[str, Any]:
        """Get API key statistics.

        Returns:
            Statistics dictionary
        """
        total_keys = len(self._keys)
        active_keys = sum(
            1 for k in self._keys.values() if k.config.status == APIKeyStatus.ACTIVE
        )
        expired_keys = sum(
            1 for k in self._keys.values() if k.config.status == APIKeyStatus.EXPIRED
        )
        revoked_keys = sum(
            1 for k in self._keys.values() if k.config.status == APIKeyStatus.REVOKED
        )

        # Count by scope
        scope_counts = {}
        for api_key in self._keys.values():
            scope = api_key.config.scope
            scope_counts[scope.value] = scope_counts.get(scope.value, 0) + 1

        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "expired_keys": expired_keys,
            "revoked_keys": revoked_keys,
            "scope_distribution": scope_counts,
            "total_usage": sum(len(usage) for usage in self._usage_stats.values()),
        }


def api_key_required(
    resource: Resource, action: Permission, api_key_manager: APIKeyManager
):
    """Decorator to enforce API key authentication on endpoints.

    Args:
        resource: Target resource
        action: Required action
        api_key_manager: API key manager instance

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(request, *args, **kwargs):
            # Extract API key from headers
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                api_key = request.headers.get("Authorization", "").replace(
                    "Bearer ", ""
                )

            if not api_key:
                msg = "API key required"
                raise ServiceError(msg, error_code="api_key_required")

            # Validate API key
            validation_request = APIKeyValidationRequest(
                key=api_key,
                resource=resource,
                action=action,
                ip_address=request.client.host,
                user_agent=request.headers.get("User-Agent"),
                referrer=request.headers.get("Referer"),
            )

            result = await api_key_manager.validate_api_key(validation_request)

            if not result.valid:
                msg = f"Invalid API key: {result.reason}"
                raise ServiceError(msg, error_code="invalid_api_key")

            # Add key info to request context
            request.state.api_key_id = result.key_id
            request.state.api_permissions = result.permissions

            # Call original function
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
