"""Role-Based Access Control (RBAC) system for enterprise security.

This module implements a comprehensive RBAC system with:
- Hierarchical role permissions
- Resource-based access control
- Dynamic permission evaluation
- Audit logging for access decisions
- Integration with JWT authentication

Following zero-trust architecture principles with fine-grained permissions.
"""

import logging
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field

from src.services.errors import ServiceError, ValidationError
from src.services.security.audit.logger import SecurityAuditLogger

from .models import UserRole


logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions following principle of least privilege."""

    # Document permissions
    DOCUMENTS_READ = "documents:read"
    DOCUMENTS_WRITE = "documents:write"
    DOCUMENTS_DELETE = "documents:delete"
    DOCUMENTS_ADMIN = "documents:admin"

    # Search permissions
    SEARCH_BASIC = "search:basic"
    SEARCH_ADVANCED = "search:advanced"
    SEARCH_EXPORT = "search:export"

    # Collection permissions
    COLLECTIONS_READ = "collections:read"
    COLLECTIONS_WRITE = "collections:write"
    COLLECTIONS_DELETE = "collections:delete"
    COLLECTIONS_ADMIN = "collections:admin"

    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

    # User management permissions
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    USERS_ADMIN = "users:admin"

    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_HEALTH = "system:health"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_ADMIN = "system:admin"

    # API permissions
    API_KEYS_READ = "api_keys:read"
    API_KEYS_WRITE = "api_keys:write"
    API_KEYS_DELETE = "api_keys:delete"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


class Resource(str, Enum):
    """System resources for access control."""

    DOCUMENTS = "documents"
    COLLECTIONS = "collections"
    SEARCH = "search"
    ANALYTICS = "analytics"
    USERS = "users"
    SYSTEM = "system"
    API_KEYS = "api_keys"
    AUDIT = "audit"


class AccessDecision(str, Enum):
    """Access control decisions."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


class RolePermissions(BaseModel):
    """Role-based permissions configuration."""

    role: UserRole
    permissions: set[Permission]
    inherits_from: UserRole | None = None
    description: str = Field(..., description="Role description")

    class Config:
        use_enum_values = True


class AccessRequest(BaseModel):
    """Access control request."""

    user_id: UUID
    username: str
    role: UserRole
    user_permissions: list[str] = Field(default_factory=list)
    resource: Resource
    action: Permission
    resource_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AccessResult(BaseModel):
    """Access control result with audit trail."""

    decision: AccessDecision
    reason: str
    evaluated_permissions: list[str]
    user_id: UUID
    resource: Resource
    action: Permission
    resource_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = Field(default_factory=dict)


class RBACManager:
    """Enterprise RBAC manager with hierarchical permissions."""

    def __init__(self, audit_logger: SecurityAuditLogger | None = None):
        """Initialize RBAC manager.

        Args:
            audit_logger: Security audit logger for access decisions
        """
        self.audit_logger = audit_logger
        self._role_permissions = self._initialize_role_permissions()
        self._resource_permissions = self._initialize_resource_permissions()

    def _initialize_role_permissions(self) -> dict[UserRole, RolePermissions]:
        """Initialize role-based permissions with hierarchy."""
        return {
            UserRole.API_USER: RolePermissions(
                role=UserRole.API_USER,
                permissions={
                    Permission.SEARCH_BASIC,
                    Permission.DOCUMENTS_READ,
                    Permission.COLLECTIONS_READ,
                    Permission.SYSTEM_HEALTH,
                },
                description="Basic API access for external integrations",
            ),
            UserRole.ANALYST: RolePermissions(
                role=UserRole.ANALYST,
                permissions={
                    Permission.SEARCH_BASIC,
                    Permission.SEARCH_ADVANCED,
                    Permission.SEARCH_EXPORT,
                    Permission.DOCUMENTS_READ,
                    Permission.COLLECTIONS_READ,
                    Permission.ANALYTICS_READ,
                    Permission.ANALYTICS_EXPORT,
                    Permission.SYSTEM_HEALTH,
                },
                inherits_from=UserRole.API_USER,
                description="Data analysis and advanced search capabilities",
            ),
            UserRole.OPERATOR: RolePermissions(
                role=UserRole.OPERATOR,
                permissions={
                    Permission.DOCUMENTS_READ,
                    Permission.DOCUMENTS_WRITE,
                    Permission.DOCUMENTS_DELETE,
                    Permission.COLLECTIONS_READ,
                    Permission.COLLECTIONS_WRITE,
                    Permission.COLLECTIONS_DELETE,
                    Permission.SYSTEM_CONFIG,
                    Permission.SYSTEM_LOGS,
                    Permission.API_KEYS_READ,
                    Permission.API_KEYS_WRITE,
                    Permission.AUDIT_READ,
                },
                inherits_from=UserRole.ANALYST,
                description="Operational management with content control",
            ),
            UserRole.ADMIN: RolePermissions(
                role=UserRole.ADMIN,
                permissions={
                    Permission.DOCUMENTS_ADMIN,
                    Permission.COLLECTIONS_ADMIN,
                    Permission.USERS_READ,
                    Permission.USERS_WRITE,
                    Permission.USERS_DELETE,
                    Permission.USERS_ADMIN,
                    Permission.SYSTEM_ADMIN,
                    Permission.API_KEYS_DELETE,
                    Permission.AUDIT_EXPORT,
                },
                inherits_from=UserRole.OPERATOR,
                description="Full administrative access",
            ),
        }

    def _initialize_resource_permissions(self) -> dict[Resource, set[Permission]]:
        """Initialize resource-based permission mappings."""
        return {
            Resource.DOCUMENTS: {
                Permission.DOCUMENTS_READ,
                Permission.DOCUMENTS_WRITE,
                Permission.DOCUMENTS_DELETE,
                Permission.DOCUMENTS_ADMIN,
            },
            Resource.COLLECTIONS: {
                Permission.COLLECTIONS_READ,
                Permission.COLLECTIONS_WRITE,
                Permission.COLLECTIONS_DELETE,
                Permission.COLLECTIONS_ADMIN,
            },
            Resource.SEARCH: {
                Permission.SEARCH_BASIC,
                Permission.SEARCH_ADVANCED,
                Permission.SEARCH_EXPORT,
            },
            Resource.ANALYTICS: {
                Permission.ANALYTICS_READ,
                Permission.ANALYTICS_EXPORT,
            },
            Resource.USERS: {
                Permission.USERS_READ,
                Permission.USERS_WRITE,
                Permission.USERS_DELETE,
                Permission.USERS_ADMIN,
            },
            Resource.SYSTEM: {
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_HEALTH,
                Permission.SYSTEM_LOGS,
                Permission.SYSTEM_ADMIN,
            },
            Resource.API_KEYS: {
                Permission.API_KEYS_READ,
                Permission.API_KEYS_WRITE,
                Permission.API_KEYS_DELETE,
            },
            Resource.AUDIT: {
                Permission.AUDIT_READ,
                Permission.AUDIT_EXPORT,
            },
        }

    def get_role_permissions(self, role: UserRole) -> set[Permission]:
        """Get all permissions for a role including inherited permissions.

        Args:
            role: User role to get permissions for

        Returns:
            Set of permissions for the role
        """
        permissions = set()
        current_role = role

        # Collect permissions from role hierarchy
        while current_role:
            role_config = self._role_permissions.get(current_role)
            if role_config:
                permissions.update(role_config.permissions)
                current_role = role_config.inherits_from
            else:
                break

        return permissions

    def check_access(self, request: AccessRequest) -> AccessResult:
        """Check access for a user request.

        Args:
            request: Access request to evaluate

        Returns:
            Access result with decision and audit information
        """
        # Get role-based permissions
        role_permissions = self.get_role_permissions(request.role)

        # Get user-specific permissions
        user_permissions = set(request.user_permissions)

        # Combine permissions
        all_permissions = role_permissions | user_permissions

        # Check if action is allowed
        decision = (
            AccessDecision.ALLOW
            if request.action in all_permissions
            else AccessDecision.DENY
        )

        # Create reason
        if decision == AccessDecision.ALLOW:
            reason = f"User has required permission '{request.action.value}'"
        else:
            reason = f"User lacks required permission '{request.action.value}'"

        # Create result
        result = AccessResult(
            decision=decision,
            reason=reason,
            evaluated_permissions=sorted([p.value for p in all_permissions]),
            user_id=request.user_id,
            resource=request.resource,
            action=request.action,
            resource_id=request.resource_id,
            context=request.context,
        )

        # Log access decision
        self._log_access_decision(request, result)

        return result

    def _log_access_decision(
        self, request: AccessRequest, result: AccessResult
    ) -> None:
        """Log access decision for audit trail.

        Args:
            request: Original access request
            result: Access decision result
        """
        if self.audit_logger:
            self.audit_logger.log_access_decision(
                user_id=str(request.user_id),
                username=request.username,
                role=request.role.value,
                resource=request.resource.value,
                action=request.action.value,
                resource_id=request.resource_id,
                decision=result.decision.value,
                reason=result.reason,
                context=request.context,
            )

        # Also log at appropriate level
        if result.decision == AccessDecision.ALLOW:
            logger.info(
                f"Access granted: {request.username} ({request.role.value}) -> "
                f"{request.resource.value}:{request.action.value}"
            )
        else:
            logger.warning(
                f"Access denied: {request.username} ({request.role.value}) -> "
                f"{request.resource.value}:{request.action.value} - {result.reason}"
            )

    def validate_permissions(self, user_permissions: list[str]) -> list[str]:
        """Validate user permissions against known permissions.

        Args:
            user_permissions: List of permission strings to validate

        Returns:
            List of valid permissions

        Raises:
            ValidationError: If any permissions are invalid
        """
        valid_permissions = {p.value for p in Permission}
        invalid_permissions = set(user_permissions) - valid_permissions

        if invalid_permissions:
            msg = f"Invalid permissions: {', '.join(invalid_permissions)}"
            raise ValidationError(
                msg,
                error_code="invalid_permissions",
                context={"invalid_permissions": list(invalid_permissions)},
            )

        return user_permissions

    def get_resource_permissions(self, resource: Resource) -> set[Permission]:
        """Get all permissions for a resource.

        Args:
            resource: Resource to get permissions for

        Returns:
            Set of permissions for the resource
        """
        return self._resource_permissions.get(resource, set())

    def can_access_resource(
        self,
        role: UserRole,
        resource: Resource,
        action: Permission,
        user_permissions: list[str] | None = None,
    ) -> bool:
        """Check if a role can access a resource with specific action.

        Args:
            role: User role
            resource: Target resource
            action: Requested action
            user_permissions: Additional user-specific permissions

        Returns:
            True if access is allowed, False otherwise
        """
        # Get role permissions
        role_permissions = self.get_role_permissions(role)

        # Add user-specific permissions if provided
        if user_permissions:
            user_perms = set(user_permissions) & {p.value for p in Permission}
            role_permissions.update(Permission(p) for p in user_perms)

        # Check if action is in role permissions
        return action in role_permissions

    def get_user_accessible_resources(
        self, role: UserRole, user_permissions: list[str] | None = None
    ) -> dict[Resource, set[Permission]]:
        """Get all resources and actions accessible to a user.

        Args:
            role: User role
            user_permissions: Additional user-specific permissions

        Returns:
            Dictionary mapping resources to accessible permissions
        """
        # Get all user permissions
        all_permissions = self.get_role_permissions(role)

        # Add user-specific permissions if provided
        if user_permissions:
            valid_user_perms = set(user_permissions) & {p.value for p in Permission}
            all_permissions.update(Permission(p) for p in valid_user_perms)

        # Map permissions to resources
        accessible_resources = {}
        for resource, resource_permissions in self._resource_permissions.items():
            user_resource_permissions = all_permissions & resource_permissions
            if user_resource_permissions:
                accessible_resources[resource] = user_resource_permissions

        return accessible_resources

    def enforce_rbac(
        self, user_role: UserRole, required_permission: Permission
    ) -> None:
        """Enforce RBAC by checking if user has required permission.

        Args:
            user_role: User's role
            required_permission: Required permission for action

        Raises:
            ServiceError: If user lacks required permission
        """
        role_permissions = self.get_role_permissions(user_role)

        if required_permission not in role_permissions:
            msg = f"Access denied: Role '{user_role.value}' lacks permission '{required_permission.value}'"
            raise ServiceError(
                msg,
                error_code="insufficient_permissions",
                context={
                    "role": user_role.value,
                    "required_permission": required_permission.value,
                    "available_permissions": [p.value for p in role_permissions],
                },
            )

    def get_role_hierarchy(self) -> dict[UserRole, dict[str, Any]]:
        """Get role hierarchy information.

        Returns:
            Dictionary with role hierarchy and permissions
        """
        hierarchy = {}

        for role, config in self._role_permissions.items():
            hierarchy[role] = {
                "description": config.description,
                "permissions": sorted([p.value for p in config.permissions]),
                "inherits_from": config.inherits_from.value
                if config.inherits_from
                else None,
                "effective_permissions": sorted(
                    [p.value for p in self.get_role_permissions(role)]
                ),
            }

        return hierarchy

    def add_user_permission(
        self, user_permissions: list[str], new_permission: Permission
    ) -> list[str]:
        """Add a permission to user's permission list.

        Args:
            user_permissions: Current user permissions
            new_permission: Permission to add

        Returns:
            Updated permission list
        """
        if new_permission.value not in user_permissions:
            user_permissions.append(new_permission.value)

        return user_permissions

    def remove_user_permission(
        self, user_permissions: list[str], permission: Permission
    ) -> list[str]:
        """Remove a permission from user's permission list.

        Args:
            user_permissions: Current user permissions
            permission: Permission to remove

        Returns:
            Updated permission list
        """
        if permission.value in user_permissions:
            user_permissions.remove(permission.value)

        return user_permissions


def rbac_required(permission: Permission, resource: Resource | None = None):
    """Decorator to enforce RBAC on endpoints.

    Args:
        permission: Required permission
        resource: Target resource (optional)

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user info from request context
            # This would be populated by authentication middleware
            user_role = getattr(
                args[0], "user_role", None
            )  # Assuming first arg is request

            if not user_role:
                msg = "Authentication required"
                raise ServiceError(msg, error_code="authentication_required")

            # Initialize RBAC manager
            rbac = RBACManager()

            # Check permission
            rbac.enforce_rbac(user_role, permission)

            # Call original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator
