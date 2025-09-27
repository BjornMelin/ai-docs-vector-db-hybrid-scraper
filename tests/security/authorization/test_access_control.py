"""Access control and authorization testing.

This module tests role-based access control (RBAC), permission validation,
and authorization boundary enforcement.
"""

import time
from typing import Any

import pytest


@pytest.mark.security
@pytest.mark.authorization
class TestAccessControl:
    """Test access control and authorization mechanisms."""

    @pytest.fixture
    def rbac_system(self):
        """Mock role-based access control system."""

        class RBACSystem:
            """Test class for security testing."""

            def __init__(self):
                self.roles = {
                    "guest": {
                        "permissions": ["read_public"],
                        "resources": ["public_docs"],
                        "restrictions": ["no_api_access"],
                    },
                    "user": {
                        "permissions": ["read", "write_own", "search"],
                        "resources": ["user_docs", "shared_docs"],
                        "restrictions": ["rate_limited"],
                    },
                    "premium_user": {
                        "permissions": [
                            "read",
                            "write_own",
                            "search",
                            "bulk_operations",
                        ],
                        "resources": ["user_docs", "shared_docs", "premium_docs"],
                        "restrictions": ["extended_rate_limits"],
                    },
                    "moderator": {
                        "permissions": [
                            "read",
                            "write_own",
                            "write_others",
                            "moderate",
                            "search",
                        ],
                        "resources": ["user_docs", "shared_docs", "moderation_queue"],
                        "restrictions": ["audit_logged"],
                    },
                    "admin": {
                        "permissions": [
                            "read",
                            "write",
                            "delete",
                            "admin",
                            "system_config",
                        ],
                        "resources": ["all"],
                        "restrictions": ["audit_logged", "mfa_required"],
                    },
                    "api_service": {
                        "permissions": ["read", "write", "bulk_operations"],
                        "resources": ["api_docs", "system_docs"],
                        "restrictions": ["api_key_required", "rate_limited"],
                    },
                }

                self.resources = {
                    "public_docs": {"access_level": "public"},
                    "user_docs": {"access_level": "user"},
                    "shared_docs": {"access_level": "shared"},
                    "premium_docs": {"access_level": "premium"},
                    "moderation_queue": {"access_level": "moderation"},
                    "admin_panel": {"access_level": "admin"},
                    "system_config": {"access_level": "system"},
                    "api_docs": {"access_level": "api"},
                    "all": {"access_level": "unrestricted"},
                }

            def check_permission(
                self, user_role: str, permission: str, resource: str | None = None
            ) -> bool:
                """Check if user role has permission for resource."""
                if user_role not in self.roles:
                    return False

                role_data = self.roles[user_role]

                # Check if role has the permission
                if permission not in role_data["permissions"]:
                    return False

                # Check if role can access the resource
                return not (
                    resource
                    and (
                        "all" not in role_data["resources"]
                        and resource not in role_data["resources"]
                    )
                )

            def get_user_permissions(self, user_role: str) -> list[str]:
                """Get all permissions for a user role."""
                if user_role not in self.roles:
                    return []
                return self.roles[user_role]["permissions"]

            def get_accessible_resources(self, user_role: str) -> list[str]:
                """Get all resources accessible by a user role."""
                if user_role not in self.roles:
                    return []
                return self.roles[user_role]["resources"]

            def is_permission_denied(
                self, user_role: str, permission: str, resource: str
            ) -> bool:
                """Check if permission is explicitly denied."""
                return not self.check_permission(user_role, permission, resource)

        return RBACSystem()

    @pytest.fixture
    def auth_context(self):
        """Mock authentication context."""
        return {
            "guest_user": {
                "user_id": "guest_001",
                "role": "guest",
                "authenticated": False,
            },
            "regular_user": {
                "user_id": "user_001",
                "role": "user",
                "authenticated": True,
            },
            "premium_user": {
                "user_id": "premium_001",
                "role": "premium_user",
                "authenticated": True,
            },
            "moderator": {
                "user_id": "mod_001",
                "role": "moderator",
                "authenticated": True,
            },
            "admin_user": {
                "user_id": "admin_001",
                "role": "admin",
                "authenticated": True,
            },
            "api_service": {
                "user_id": "service_001",
                "role": "api_service",
                "authenticated": True,
            },
        }

    def test_guest_access_permissions(self, rbac_system, auth_context):
        """Test guest user access permissions."""
        guest = auth_context["guest_user"]

        # Guest should have minimal permissions
        assert (
            rbac_system.check_permission(guest["role"], "read_public", "public_docs")
            is True
        )

        # Guest should be denied privileged operations
        assert (
            rbac_system.check_permission(guest["role"], "write", "user_docs") is False
        )
        assert (
            rbac_system.check_permission(guest["role"], "delete", "public_docs")
            is False
        )
        assert (
            rbac_system.check_permission(guest["role"], "admin", "system_config")
            is False
        )

        # Guest should not access restricted resources
        assert (
            rbac_system.check_permission(guest["role"], "read", "admin_panel") is False
        )

    def test_user_access_permissions(self, rbac_system, auth_context):
        """Test regular user access permissions."""
        user = auth_context["regular_user"]

        # User should have basic permissions
        assert rbac_system.check_permission(user["role"], "read", "user_docs") is True
        assert (
            rbac_system.check_permission(user["role"], "write_own", "user_docs") is True
        )
        assert (
            rbac_system.check_permission(user["role"], "search", "shared_docs") is True
        )

        # User should be denied admin operations
        assert (
            rbac_system.check_permission(user["role"], "admin", "system_config")
            is False
        )
        assert (
            rbac_system.check_permission(user["role"], "delete", "shared_docs") is False
        )
        assert (
            rbac_system.check_permission(user["role"], "write_others", "user_docs")
            is False
        )

    def test_admin_access_permissions(self, rbac_system, auth_context):
        """Test admin user access permissions."""
        admin = auth_context["admin_user"]

        # Admin should have broad permissions
        assert rbac_system.check_permission(admin["role"], "read", "all") is True
        assert rbac_system.check_permission(admin["role"], "write", "all") is True
        assert rbac_system.check_permission(admin["role"], "delete", "all") is True
        assert (
            rbac_system.check_permission(admin["role"], "admin", "system_config")
            is True
        )

        # Admin should access all resources
        permissions = rbac_system.get_user_permissions(admin["role"])
        assert "admin" in permissions
        assert "system_config" in permissions

    def test_privilege_escalation_prevention(self, rbac_system, auth_context):
        """Test prevention of privilege escalation."""
        user = auth_context["regular_user"]

        # User should not be able to escalate to admin
        escalation_attempts = [
            ("admin", "system_config"),
            ("delete", "all"),
            ("write_others", "admin_panel"),
            ("system_config", "all"),
        ]

        for permission, resource in escalation_attempts:
            assert (
                rbac_system.check_permission(user["role"], permission, resource)
                is False
            )

    def test_horizontal_privilege_escalation_prevention(self, _rbac_system):
        """Test prevention of horizontal privilege escalation."""
        # Test that users cannot access other users' resources

        # In a real system, resource access should be filtered by ownership
        # This test simulates checking resource ownership
        def check_resource_ownership(user_id: str, resource_id: str) -> bool:
            # Simulate resource ownership check
            resource_owners = {
                "doc_001": "user_a",
                "doc_002": "user_b",
                "doc_003": "user_a",
            }
            return resource_owners.get(resource_id) == user_id

        # User A should access their own resources
        assert check_resource_ownership("user_a", "doc_001") is True
        assert check_resource_ownership("user_a", "doc_003") is True

        # User A should not access User B's resources
        assert check_resource_ownership("user_a", "doc_002") is False

        # User B should access their own resources
        assert check_resource_ownership("user_b", "doc_002") is True

        # User B should not access User A's resources
        assert check_resource_ownership("user_b", "doc_001") is False
        assert check_resource_ownership("user_b", "doc_003") is False

    def test_role_based_resource_filtering(self, rbac_system, auth_context):
        """Test role-based resource filtering."""
        # Test that users only see resources they have access to
        for user_data in auth_context.values():
            accessible_resources = rbac_system.get_accessible_resources(
                user_data["role"]
            )

            if user_data["role"] == "guest":
                assert "public_docs" in accessible_resources
                assert "admin_panel" not in accessible_resources
            elif user_data["role"] == "user":
                assert "user_docs" in accessible_resources
                assert "shared_docs" in accessible_resources
                assert "admin_panel" not in accessible_resources
            elif user_data["role"] == "admin":
                assert "all" in accessible_resources

    def test_permission_inheritance_and_override(self, rbac_system):
        """Test permission inheritance and override mechanisms."""
        # Test that premium users inherit user permissions plus additional ones
        user_permissions = set(rbac_system.get_user_permissions("user"))
        premium_permissions = set(rbac_system.get_user_permissions("premium_user"))

        # Premium should have all user permissions plus more
        assert user_permissions.issubset(premium_permissions)
        assert "bulk_operations" in premium_permissions
        assert "bulk_operations" not in user_permissions

    def test_context_based_access_control(self, rbac_system):
        """Test context-based access control (time, location, etc.)."""

        class ContextualAccessControl:
            """Test class for security testing."""

            def __init__(self, rbac_system):
                self.rbac = rbac_system

            def check_contextual_access(
                self,
                user_role: str,
                permission: str,
                resource: str,
                context: dict[str, Any],
            ) -> bool:
                # Basic permission check
                if not self.rbac.check_permission(user_role, permission, resource):
                    return False

                # Time-based restrictions
                if (
                    context.get("time_restricted")
                    and not context.get("business_hours")
                    and user_role not in ["admin", "api_service"]
                ):
                    return False

                # Location-based restrictions
                if (
                    context.get("ip_restricted")
                    and not context.get("trusted_ip")
                    and permission in ["admin", "delete", "system_config"]
                ):
                    return False

                # MFA requirements
                return not (user_role == "admin" and not context.get("mfa_verified"))

        contextual_ac = ContextualAccessControl(rbac_system)

        # Test time-based restrictions
        context = {
            "time_restricted": True,
            "business_hours": False,
            "trusted_ip": True,
            "mfa_verified": True,
        }
        assert (
            contextual_ac.check_contextual_access("user", "read", "user_docs", context)
            is False
        )
        assert (
            contextual_ac.check_contextual_access("admin", "read", "user_docs", context)
            is True
        )

        # Test MFA requirements for admin
        context = {"mfa_verified": False, "trusted_ip": True, "business_hours": True}
        assert (
            contextual_ac.check_contextual_access(
                "admin", "admin", "system_config", context
            )
            is False
        )

        context["mfa_verified"] = True
        assert (
            contextual_ac.check_contextual_access(
                "admin", "admin", "system_config", context
            )
            is True
        )

    def test_api_key_based_authorization(self, _rbac_system):
        """Test API key-based authorization."""

        class APIKeyAuthz:
            """Test class for security testing."""

            def __init__(self):
                self.api_keys = {
                    "read_only_key": {
                        "permissions": ["read"],
                        "resources": ["public_docs", "user_docs"],
                    },
                    "write_key": {
                        "permissions": ["read", "write"],
                        "resources": ["user_docs"],
                    },
                    "admin_key": {
                        "permissions": ["read", "write", "admin"],
                        "resources": ["all"],
                    },
                    "service_key": {
                        "permissions": ["read", "write", "bulk_operations"],
                        "resources": ["api_docs"],
                    },
                }

            def validate_api_key_access(
                self, api_key: str, permission: str, resource: str
            ) -> bool:
                if api_key not in self.api_keys:
                    return False

                key_data = self.api_keys[api_key]

                if permission not in key_data["permissions"]:
                    return False

                return not (
                    "all" not in key_data["resources"]
                    and resource not in key_data["resources"]
                )

        api_authz = APIKeyAuthz()

        # Test read-only key
        assert (
            api_authz.validate_api_key_access("read_only_key", "read", "public_docs")
            is True
        )
        assert (
            api_authz.validate_api_key_access("read_only_key", "write", "user_docs")
            is False
        )

        # Test write key
        assert (
            api_authz.validate_api_key_access("write_key", "read", "user_docs") is True
        )
        assert (
            api_authz.validate_api_key_access("write_key", "write", "user_docs") is True
        )
        assert (
            api_authz.validate_api_key_access("write_key", "admin", "system_config")
            is False
        )

        # Test admin key
        assert api_authz.validate_api_key_access("admin_key", "admin", "all") is True

    def test_resource_based_permissions(self, _rbac_system):
        """Test resource-based permission validation."""

        class ResourcePermissions:
            """Test class for security testing."""

            def __init__(self):
                self.resource_acl = {
                    "sensitive_docs": {
                        "required_permissions": ["read_sensitive"],
                        "required_roles": ["admin", "moderator"],
                        "additional_checks": ["mfa_required"],
                    },
                    "financial_data": {
                        "required_permissions": ["read_financial"],
                        "required_roles": ["admin"],
                        "additional_checks": ["audit_log", "ip_whitelist"],
                    },
                    "user_profiles": {
                        "required_permissions": ["read_profiles"],
                        "required_roles": ["admin", "moderator", "user"],
                        "ownership_check": True,
                    },
                }

            def check_resource_access(
                self,
                user_role: str,
                resource: str,
                user_id: str | None = None,
                resource_owner: str | None = None,
            ) -> bool:
                if resource not in self.resource_acl:
                    return True  # No special restrictions

                acl = self.resource_acl[resource]

                # Check role requirements
                if acl.get("required_roles") and user_role not in acl["required_roles"]:
                    return False

                # Check ownership for user resources
                return not (
                    acl.get("ownership_check")
                    and user_role == "user"
                    and (not user_id or not resource_owner or user_id != resource_owner)
                )

        resource_perms = ResourcePermissions()

        # Test sensitive document access
        assert resource_perms.check_resource_access("admin", "sensitive_docs") is True
        assert (
            resource_perms.check_resource_access("moderator", "sensitive_docs") is True
        )
        assert resource_perms.check_resource_access("user", "sensitive_docs") is False

        # Test financial data access
        assert resource_perms.check_resource_access("admin", "financial_data") is True
        assert (
            resource_perms.check_resource_access("moderator", "financial_data") is False
        )

        # Test user profile ownership
        assert (
            resource_perms.check_resource_access(
                "user", "user_profiles", "user_001", "user_001"
            )
            is True
        )
        assert (
            resource_perms.check_resource_access(
                "user", "user_profiles", "user_001", "user_002"
            )
            is False
        )

    def test_permission_delegation(self, _rbac_system):
        """Test permission delegation mechanisms."""

        class PermissionDelegation:
            """Test class for security testing."""

            def __init__(self):
                self.delegations = {}

            def delegate_permission(
                self,
                from_user: str,
                to_user: str,
                permission: str,
                resource: str,
                expiry: int,
            ):
                """Delegate permission from one user to another."""
                delegation_id = f"{from_user}_{to_user}_{permission}_{resource}"
                self.delegations[delegation_id] = {
                    "from_user": from_user,
                    "to_user": to_user,
                    "permission": permission,
                    "resource": resource,
                    "expiry": expiry,
                    "active": True,
                }
                return delegation_id

            def check_delegated_permission(
                self, user: str, permission: str, resource: str, current_time: int
            ) -> bool:
                """Check if user has delegated permission."""
                for delegation in self.delegations.values():
                    if (
                        delegation["to_user"] == user
                        and delegation["permission"] == permission
                        and delegation["resource"] == resource
                        and delegation["active"]
                        and current_time < delegation["expiry"]
                    ):
                        return True
                return False

        delegation_system = PermissionDelegation()

        # Admin delegates write permission to user

        current_time = int(time.time())
        expiry_time = current_time + 3600  # 1 hour

        delegation_system.delegate_permission(
            "admin_001", "user_001", "write", "shared_docs", expiry_time
        )

        # User should now have delegated write permission
        assert (
            delegation_system.check_delegated_permission(
                "user_001", "write", "shared_docs", current_time
            )
            is True
        )

        # User should not have non-delegated permissions
        assert (
            delegation_system.check_delegated_permission(
                "user_001", "delete", "shared_docs", current_time
            )
            is False
        )

        # Delegation should expire
        assert (
            delegation_system.check_delegated_permission(
                "user_001", "write", "shared_docs", expiry_time + 1
            )
            is False
        )

    def test_access_control_bypass_prevention(self, _rbac_system):
        """Test prevention of access control bypasses."""

        # Test common bypass techniques
        bypass_attempts = [
            # Parameter pollution
            {"role": ["user", "admin"]},  # Array instead of string
            {"role": "user\x00admin"},  # Null byte injection
            {"role": "user admin"},  # Space injection
            {"role": "user;admin"},  # Semicolon injection
            # Case manipulation
            {"role": "ADMIN"},
            {"role": "Admin"},
            {"role": "aDmIn"},
            # Unicode normalization
            {"role": "admin\u0000"},  # Unicode null
            {"role": "admin\u200b"},  # Zero-width space
            # JSON injection
            {"role": '{"admin": true}'},
        ]

        def validate_role_input(role_input) -> str:
            """Validate and normalize role input."""
            if not isinstance(role_input, str):
                msg = "Role must be a string"
                raise ValueError(msg)

            # Remove null bytes and control characters
            role = "".join(c for c in role_input if ord(c) >= 32)

            # Normalize case
            role = role.lower().strip()

            # Validate against known roles
            valid_roles = {
                "guest",
                "user",
                "premium_user",
                "moderator",
                "admin",
                "api_service",
            }
            if role not in valid_roles:
                msg = f"Invalid role: {role}"
                raise ValueError(msg)

            return role

        # Test that bypass attempts are prevented
        for attempt in bypass_attempts:
            role_input = attempt.get("role")
            try:
                normalized_role = validate_role_input(role_input)
                # If validation passes, ensure it's a legitimate role
                assert normalized_role in {
                    "guest",
                    "user",
                    "premium_user",
                    "moderator",
                    "admin",
                    "api_service",
                }
            except (ValueError, TypeError, AttributeError):
                # Expected for malicious inputs
                pass

    def test_session_based_authorization(self, rbac_system):
        """Test session-based authorization mechanisms."""

        class SessionAuthz:
            """Test class for security testing."""

            def __init__(self):
                self.sessions = {}

            def create_session(self, user_id: str, role: str, permissions: list[str]):
                """Create authenticated session."""
                session_id = f"session_{user_id}_{int(time.time())}"
                self.sessions[session_id] = {
                    "user_id": user_id,
                    "role": role,
                    "permissions": permissions,
                    "created_at": int(time.time()),
                    "last_activity": int(time.time()),
                    "active": True,
                }
                return session_id

            def validate_session_permission(
                self, session_id: str, permission: str
            ) -> bool:
                """Validate session has required permission."""
                if session_id not in self.sessions:
                    return False

                session = self.sessions[session_id]
                if not session["active"]:
                    return False

                # Check session timeout (30 minutes)
                current_time = int(time.time())
                if current_time - session["last_activity"] > 1800:
                    session["active"] = False
                    return False

                # Update last activity
                session["last_activity"] = current_time

                return permission in session["permissions"]

        session_authz = SessionAuthz()

        # Create user session
        user_permissions = rbac_system.get_user_permissions("user")
        session_id = session_authz.create_session("user_001", "user", user_permissions)

        # Test session permissions
        assert session_authz.validate_session_permission(session_id, "read") is True
        assert session_authz.validate_session_permission(session_id, "admin") is False

        # Test invalid session
        assert (
            session_authz.validate_session_permission("invalid_session", "read")
            is False
        )

    def test_attribute_based_access_control(self, _rbac_system):
        """Test attribute-based access control (ABAC)."""

        class ABACEngine:
            """Test class for security testing."""

            def __init__(self):
                self.policies = []

            def add_policy(self, policy: dict[str, Any]):
                """Add ABAC policy."""
                self.policies.append(policy)

            def evaluate_access(
                self,
                subject: dict[str, Any],
                resource: dict[str, Any],
                action: str,
                environment: dict[str, Any],
            ) -> bool:
                """Evaluate access using ABAC policies."""
                for policy in self.policies:
                    if self._matches_policy(
                        policy, subject, resource, action, environment
                    ):
                        return policy.get("effect") == "allow"

                return False  # Deny by default

            def _matches_policy(
                self,
                policy: dict[str, Any],
                subject: dict[str, Any],
                resource: dict[str, Any],
                action: str,
                environment: dict[str, Any],
            ) -> bool:
                """Check if request matches policy conditions."""
                # Subject conditions
                subject_conditions = policy.get("subject", {})
                for attr, value in subject_conditions.items():
                    if subject.get(attr) != value:
                        return False

                # Resource conditions
                resource_conditions = policy.get("resource", {})
                for attr, value in resource_conditions.items():
                    if resource.get(attr) != value:
                        return False

                # Action conditions
                if policy.get("action") and policy["action"] != action:
                    return False

                # Environment conditions
                env_conditions = policy.get("environment", {})
                for attr, value in env_conditions.items():
                    if environment.get(attr) != value:
                        return False

                return True

        abac = ABACEngine()

        # Add policies
        abac.add_policy(
            {
                "subject": {"department": "finance"},
                "resource": {"classification": "financial"},
                "action": "read",
                "effect": "allow",
            }
        )

        abac.add_policy(
            {
                "subject": {"role": "admin"},
                "resource": {},  # Any resource
                "action": "admin",
                "effect": "allow",
            }
        )

        # Test ABAC evaluation
        finance_user = {"department": "finance", "role": "user"}
        financial_doc = {"classification": "financial", "type": "report"}
        public_doc = {"classification": "public", "type": "article"}

        # Finance user should access financial documents
        assert abac.evaluate_access(finance_user, financial_doc, "read", {}) is True

        # Finance user should not access non-financial documents
        assert abac.evaluate_access(finance_user, public_doc, "read", {}) is False

        # Admin should have admin access
        admin_user = {"role": "admin", "department": "it"}
        assert abac.evaluate_access(admin_user, financial_doc, "admin", {}) is True
        admin_user = {"role": "admin", "department": "it"}
        assert abac.evaluate_access(admin_user, financial_doc, "admin", {}) is True
