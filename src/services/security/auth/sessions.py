"""Session management system with Redis backing for enterprise security.

This module implements a comprehensive session management system with:
- Redis-backed session storage
- JWT token integration
- Session expiration and cleanup
- Security features (IP binding, device fingerprinting)
- Concurrent session management
- Session activity tracking

Following zero-trust architecture principles with secure session handling.
"""

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import redis.asyncio as redis
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, field_validator

from src.services.errors import ServiceError, ValidationError
from src.services.security.audit.logger import SecurityAuditLogger

from .models import UserRole
from .rbac import Permission, RBACManager


logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class SessionType(str, Enum):
    """Session type classification."""

    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    SERVICE = "service"
    ADMIN = "admin"


class DeviceInfo(BaseModel):
    """Device information for session tracking."""

    fingerprint: str = Field(..., description="Device fingerprint")
    user_agent: str = Field(..., description="User agent string")
    ip_address: str = Field(..., description="IP address")
    platform: str | None = Field(
        default=None, description="Platform (web, mobile, etc.)"
    )
    location: str | None = Field(default=None, description="Geographic location")
    is_trusted: bool = Field(default=False, description="Whether device is trusted")

    class Config:
        use_enum_values = True


class SessionMetadata(BaseModel):
    """Session metadata and tracking information."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    access_count: int = Field(default=0)
    created_by: str = Field(..., description="User who created the session")
    session_type: SessionType = Field(default=SessionType.WEB)
    device_info: DeviceInfo = Field(..., description="Device information")
    activity_log: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class SessionConfig(BaseModel):
    """Session configuration and security settings."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    role: UserRole = Field(..., description="User role")
    permissions: list[Permission] = Field(default_factory=list)

    # Security settings
    max_idle_time: int = Field(default=3600, description="Max idle time in seconds")
    absolute_timeout: int = Field(
        default=28800, description="Absolute timeout in seconds"
    )
    ip_binding: bool = Field(default=True, description="Bind session to IP address")
    concurrent_sessions_allowed: int = Field(
        default=3, description="Max concurrent sessions"
    )

    # Session data
    session_data: dict[str, Any] = Field(default_factory=dict)
    csrf_token: str | None = Field(default=None, description="CSRF token")

    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not v or len(v) < 32:
            msg = "Session ID must be at least 32 characters"
            raise ValueError(msg)
        return v

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username."""
        if not v or len(v.strip()) < 3:
            msg = "Username must be at least 3 characters"
            raise ValueError(msg)
        return v.strip()


class Session(BaseModel):
    """Complete session with encrypted data and configuration."""

    session_id: str = Field(..., description="Session identifier")
    encrypted_data: str = Field(..., description="Encrypted session data")
    config: SessionConfig = Field(..., description="Session configuration")

    def is_valid(self) -> bool:
        """Check if session is valid and active."""
        if self.config.status != SessionStatus.ACTIVE:
            return False

        now = datetime.now(UTC)

        # Check absolute timeout
        if self.config.metadata.expires_at and now > self.config.metadata.expires_at:
            return False

        # Check idle timeout
        if self.config.metadata.last_accessed_at:
            idle_time = now - self.config.metadata.last_accessed_at
            if idle_time.total_seconds() > self.config.max_idle_time:
                return False

        return True

    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed for this session."""
        if not self.config.ip_binding:
            return True

        return self.config.metadata.device_info.ip_address == ip_address

    def update_last_accessed(self) -> None:
        """Update last accessed timestamp."""
        self.config.metadata.last_accessed_at = datetime.now(UTC)
        self.config.metadata.access_count += 1

    def add_activity(self, activity: str, context: dict[str, Any]) -> None:
        """Add activity to session log."""
        activity_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "activity": activity,
            "context": context,
        }

        # Keep only last 100 activities
        self.config.metadata.activity_log.append(activity_entry)
        if len(self.config.metadata.activity_log) > 100:
            self.config.metadata.activity_log.pop(0)


class SessionValidationRequest(BaseModel):
    """Session validation request."""

    session_id: str = Field(..., description="Session ID to validate")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Client user agent")

    class Config:
        use_enum_values = True


class SessionValidationResult(BaseModel):
    """Session validation result."""

    valid: bool = Field(..., description="Whether session is valid")
    session_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    role: UserRole | None = None
    permissions: list[Permission] = Field(default_factory=list)
    reason: str | None = None
    expires_at: datetime | None = None

    class Config:
        use_enum_values = True


class SessionManager:
    """Enterprise session management system with Redis backing."""

    def __init__(
        self,
        redis_client: redis.Redis,
        rbac_manager: RBACManager,
        audit_logger: SecurityAuditLogger | None = None,
        encryption_key: bytes | None = None,
    ):
        """Initialize session manager.

        Args:
            redis_client: Redis client for session storage
            rbac_manager: RBAC manager for permission checking
            audit_logger: Security audit logger
            encryption_key: Key for session data encryption
        """
        self.redis = redis_client
        self.rbac_manager = rbac_manager
        self.audit_logger = audit_logger

        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())

        # Session key prefixes
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"
        self.session_index_prefix = "session_index:"

        # Default session settings
        self.default_session_timeout = 3600  # 1 hour
        self.default_absolute_timeout = 28800  # 8 hours
        self.session_cleanup_interval = 300  # 5 minutes

    async def create_session(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        device_info: DeviceInfo,
        session_type: SessionType = SessionType.WEB,
        session_data: dict[str, Any] | None = None,
        max_idle_time: int | None = None,
        absolute_timeout: int | None = None,
    ) -> Session:
        """Create new session.

        Args:
            user_id: User identifier
            username: Username
            role: User role
            device_info: Device information
            session_type: Type of session
            session_data: Additional session data
            max_idle_time: Max idle time in seconds
            absolute_timeout: Absolute timeout in seconds

        Returns:
            Created session

        Raises:
            ServiceError: If session creation fails
        """
        # Check concurrent session limit
        await self._check_concurrent_sessions(user_id)

        # Generate session ID
        session_id = self._generate_session_id()

        # Get user permissions
        permissions = self.rbac_manager.get_role_permissions(role)

        # Set timeouts
        idle_timeout = max_idle_time or self.default_session_timeout
        abs_timeout = absolute_timeout or self.default_absolute_timeout

        # Create expiration time
        expires_at = datetime.now(UTC) + timedelta(seconds=abs_timeout)

        # Create metadata
        metadata = SessionMetadata(
            created_by=username,
            session_type=session_type,
            device_info=device_info,
            expires_at=expires_at,
        )

        # Generate CSRF token
        csrf_token = self._generate_csrf_token()

        # Create session configuration
        config = SessionConfig(
            session_id=session_id,
            user_id=user_id,
            username=username,
            role=role,
            permissions=list(permissions),
            max_idle_time=idle_timeout,
            absolute_timeout=abs_timeout,
            session_data=session_data or {},
            csrf_token=csrf_token,
            metadata=metadata,
        )

        # Encrypt session data
        encrypted_data = self._encrypt_session_data(config.session_data)

        # Create session
        session = Session(
            session_id=session_id, encrypted_data=encrypted_data, config=config
        )

        # Store session in Redis
        await self._store_session(session)

        # Update user session index
        await self._update_user_session_index(user_id, session_id)

        # Log session creation
        await self._log_session_event(
            "session_created",
            session_id,
            user_id,
            {
                "username": username,
                "role": role.value,
                "session_type": session_type.value,
                "device_fingerprint": device_info.fingerprint,
                "ip_address": device_info.ip_address,
            },
        )

        return session

    async def validate_session(
        self, request: SessionValidationRequest
    ) -> SessionValidationResult:
        """Validate session and check permissions.

        Args:
            request: Validation request

        Returns:
            Validation result
        """
        # Get session from Redis
        session = await self._get_session(request.session_id)
        if not session:
            await self._log_session_event(
                "session_validation_failed",
                request.session_id,
                "unknown",
                {"reason": "session_not_found", "ip": request.ip_address},
            )
            return SessionValidationResult(valid=False, reason="Session not found")

        # Check session validity
        if not session.is_valid():
            reason = "expired"
            if session.config.status == SessionStatus.REVOKED:
                reason = "revoked"
            elif session.config.status == SessionStatus.SUSPENDED:
                reason = "suspended"

            await self._log_session_event(
                "session_validation_failed",
                request.session_id,
                session.config.user_id,
                {"reason": reason, "ip": request.ip_address},
            )

            return SessionValidationResult(
                valid=False,
                session_id=request.session_id,
                reason=f"Session is {reason}",
            )

        # Check IP binding
        if not session.is_ip_allowed(request.ip_address):
            await self._log_session_event(
                "session_validation_failed",
                request.session_id,
                session.config.user_id,
                {"reason": "ip_mismatch", "ip": request.ip_address},
            )
            return SessionValidationResult(
                valid=False, session_id=request.session_id, reason="IP address mismatch"
            )

        # Update session activity
        session.update_last_accessed()
        session.add_activity("session_validated", {"ip": request.ip_address})

        # Update session in Redis
        await self._update_session(session)

        # Log successful validation
        await self._log_session_event(
            "session_validated",
            request.session_id,
            session.config.user_id,
            {"ip": request.ip_address},
        )

        return SessionValidationResult(
            valid=True,
            session_id=request.session_id,
            user_id=session.config.user_id,
            username=session.config.username,
            role=session.config.role,
            permissions=session.config.permissions,
            expires_at=session.config.metadata.expires_at,
        )

    async def revoke_session(self, session_id: str, revoked_by: str) -> bool:
        """Revoke session.

        Args:
            session_id: Session identifier to revoke
            revoked_by: User revoking the session

        Returns:
            True if session was revoked
        """
        session = await self._get_session(session_id)
        if not session:
            return False

        # Update status
        session.config.status = SessionStatus.REVOKED
        session.add_activity("session_revoked", {"revoked_by": revoked_by})

        # Update in Redis
        await self._update_session(session)

        # Remove from user session index
        await self._remove_from_user_session_index(session.config.user_id, session_id)

        # Log revocation
        await self._log_session_event(
            "session_revoked",
            session_id,
            revoked_by,
            {"target_user": session.config.user_id},
        )

        return True

    async def revoke_all_user_sessions(self, user_id: str, revoked_by: str) -> int:
        """Revoke all sessions for a user.

        Args:
            user_id: User identifier
            revoked_by: User revoking the sessions

        Returns:
            Number of sessions revoked
        """
        session_ids = await self._get_user_session_ids(user_id)
        revoked_count = 0

        for session_id in session_ids:
            if await self.revoke_session(session_id, revoked_by):
                revoked_count += 1

        return revoked_count

    async def get_user_sessions(self, user_id: str) -> list[SessionConfig]:
        """Get all active sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of session configurations
        """
        session_ids = await self._get_user_session_ids(user_id)
        sessions = []

        for session_id in session_ids:
            session = await self._get_session(session_id)
            if session and session.config.status == SessionStatus.ACTIVE:
                sessions.append(session.config)

        return sessions

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        # Get all session keys
        session_keys = await self.redis.keys(f"{self.session_prefix}*")
        expired_count = 0

        for key in session_keys:
            session_id = key.decode().replace(self.session_prefix, "")
            session = await self._get_session(session_id)

            if session and not session.is_valid():
                await self._delete_session(session_id)
                await self._remove_from_user_session_index(
                    session.config.user_id, session_id
                )
                expired_count += 1

        # Log cleanup
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")

        return expired_count

    async def get_session_data(self, session_id: str) -> dict[str, Any] | None:
        """Get decrypted session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        session = await self._get_session(session_id)
        if not session:
            return None

        return self._decrypt_session_data(session.encrypted_data)

    async def update_session_data(self, session_id: str, data: dict[str, Any]) -> bool:
        """Update session data.

        Args:
            session_id: Session identifier
            data: New session data

        Returns:
            True if updated successfully
        """
        session = await self._get_session(session_id)
        if not session:
            return False

        # Update session data
        session.config.session_data = data
        session.encrypted_data = self._encrypt_session_data(data)
        session.add_activity("session_data_updated", {"keys": list(data.keys())})

        # Update in Redis
        await self._update_session(session)

        return True

    def _generate_session_id(self) -> str:
        """Generate secure session ID."""
        return str(uuid4()).replace("-", "")

    def _generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        return str(uuid4()).replace("-", "")

    def _encrypt_session_data(self, data: dict[str, Any]) -> str:
        """Encrypt session data."""
        json_data = json.dumps(data, default=str)
        return self.cipher.encrypt(json_data.encode()).decode()

    def _decrypt_session_data(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt session data."""
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.exception(f"Failed to decrypt session data: {e}")
            return {}

    async def _store_session(self, session: Session) -> None:
        """Store session in Redis."""
        key = f"{self.session_prefix}{session.session_id}"
        session_data = {
            "config": session.config.model_dump(),
            "encrypted_data": session.encrypted_data,
        }

        # Store with TTL
        await self.redis.setex(
            key, session.config.absolute_timeout, json.dumps(session_data, default=str)
        )

    async def _get_session(self, session_id: str) -> Session | None:
        """Get session from Redis."""
        key = f"{self.session_prefix}{session_id}"
        data = await self.redis.get(key)

        if not data:
            return None

        try:
            session_data = json.loads(data.decode())
            config = SessionConfig(**session_data["config"])

            return Session(
                session_id=session_id,
                encrypted_data=session_data["encrypted_data"],
                config=config,
            )
        except Exception as e:
            logger.exception(f"Failed to deserialize session {session_id}: {e}")
            return None

    async def _update_session(self, session: Session) -> None:
        """Update session in Redis."""
        await self._store_session(session)

    async def _delete_session(self, session_id: str) -> None:
        """Delete session from Redis."""
        key = f"{self.session_prefix}{session_id}"
        await self.redis.delete(key)

    async def _update_user_session_index(self, user_id: str, session_id: str) -> None:
        """Update user session index."""
        key = f"{self.user_sessions_prefix}{user_id}"
        await self.redis.sadd(key, session_id)
        await self.redis.expire(key, self.default_absolute_timeout)

    async def _remove_from_user_session_index(
        self, user_id: str, session_id: str
    ) -> None:
        """Remove session from user index."""
        key = f"{self.user_sessions_prefix}{user_id}"
        await self.redis.srem(key, session_id)

    async def _get_user_session_ids(self, user_id: str) -> list[str]:
        """Get all session IDs for a user."""
        key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = await self.redis.smembers(key)
        return [sid.decode() for sid in session_ids]

    async def _check_concurrent_sessions(self, user_id: str) -> None:
        """Check concurrent session limit."""
        session_ids = await self._get_user_session_ids(user_id)

        # Count active sessions
        active_count = 0
        for session_id in session_ids:
            session = await self._get_session(session_id)
            if session and session.is_valid():
                active_count += 1

        # Default concurrent session limit
        max_concurrent = 3

        if active_count >= max_concurrent:
            msg = f"Maximum concurrent sessions exceeded ({max_concurrent})"
            raise ServiceError(msg, error_code="concurrent_session_limit")

    async def _log_session_event(
        self, event_type: str, session_id: str, user_id: str, context: dict[str, Any]
    ) -> None:
        """Log session event for audit trail."""
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="session_event",
                user_id=user_id,
                resource="sessions",
                action=event_type,
                resource_id=session_id,
                context=context,
            )

        # Also log at appropriate level
        if "failed" in event_type:
            logger.warning(
                f"Session event failed: {event_type} for session {session_id}"
            )
        else:
            logger.info(f"Session event: {event_type} for session {session_id}")

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        # This would need to be implemented with Redis scan
        # For now, return placeholder
        return {
            "total_sessions": 0,
            "active_sessions": 0,
            "expired_sessions": 0,
            "revoked_sessions": 0,
            "session_types": {},
            "average_session_duration": 0.0,
        }


def session_required(session_manager: SessionManager):
    """Decorator to enforce session authentication on endpoints.

    Args:
        session_manager: Session manager instance

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(request, *args, **kwargs):
            # Extract session ID from cookie or header
            session_id = request.cookies.get("session_id")
            if not session_id:
                session_id = request.headers.get("X-Session-ID")

            if not session_id:
                msg = "Session required"
                raise ServiceError(msg, error_code="session_required")

            # Validate session
            validation_request = SessionValidationRequest(
                session_id=session_id,
                ip_address=request.client.host,
                user_agent=request.headers.get("User-Agent", ""),
            )

            result = await session_manager.validate_session(validation_request)

            if not result.valid:
                msg = f"Invalid session: {result.reason}"
                raise ServiceError(msg, error_code="invalid_session")

            # Add session info to request context
            request.state.session_id = result.session_id
            request.state.user_id = result.user_id
            request.state.username = result.username
            request.state.user_role = result.role
            request.state.user_permissions = result.permissions

            # Call original function
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
