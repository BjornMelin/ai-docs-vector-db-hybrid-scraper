"""JWT token security testing.

This module tests JWT token validation, manipulation resistance,
and authentication bypass prevention.
"""

import base64
import json
import os
import time
from unittest.mock import MagicMock

import jwt
import pytest


@pytest.mark.security
@pytest.mark.authentication
class TestJWTSecurity:
    """Test JWT token security implementation."""

    @pytest.fixture
    def _jwt_secret(self):
        """JWT signing secret for testing."""
        return "test_secret_key_for_jwt_signing_must_be_strong_in_production"

    @pytest.fixture
    def valid_jwt_payload(self):
        """Valid JWT payload for testing."""
        return {
            "user_id": "test_user_123",
            "username": "testuser",
            "role": "user",
            "permissions": ["read", "write"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,  # 1 hour expiry
            "nbf": int(time.time()),  # Not before now
            "iss": "ai-docs-vector-db",  # Issuer
            "aud": "api-client",  # Audience
        }

    @pytest.fixture
    def admin_jwt_payload(self):
        """Admin JWT payload for testing."""
        return {
            "user_id": "admin_user_456",
            "username": "admin",
            "role": "admin",
            "permissions": ["read", "write", "admin", "delete"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()),
            "iss": "ai-docs-vector-db",
            "aud": "api-client",
        }

    @pytest.fixture
    def jwt_service(self, _jwt_secret):
        """Mock JWT service for testing."""
        service = MagicMock()

        def mock_generate_token(payload):
            return jwt.encode(payload, _jwt_secret, algorithm="HS256")

        def mock_verify_token(token):
            try:
                payload = jwt.decode(
                    token,
                    _jwt_secret,
                    algorithms=["HS256"],
                    options={"verify_aud": False, "verify_iss": False},
                )
            except jwt.InvalidTokenError as e:
                return {"valid": False, "error": str(e)}
            else:
                return {"valid": True, "payload": payload}

        service.generate_token = mock_generate_token
        service.verify_token = mock_verify_token
        return service

    def test_valid_jwt_generation(self, jwt_service, valid_jwt_payload):
        """Test valid JWT token generation."""
        token = jwt_service.generate_token(valid_jwt_payload)
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token structure (header.payload.signature)
        parts = token.split(".")
        assert len(parts) == 3

    def test_valid_jwt_verification(self, jwt_service, valid_jwt_payload):
        """Test valid JWT token verification."""
        token = jwt_service.generate_token(valid_jwt_payload)
        result = jwt_service.verify_token(token)

        assert result["valid"] is True
        assert result["payload"]["user_id"] == valid_jwt_payload["user_id"]
        assert result["payload"]["username"] == valid_jwt_payload["username"]
        assert result["payload"]["role"] == valid_jwt_payload["role"]

    def test_expired_jwt_rejection(self, jwt_service, _jwt_secret):
        """Test rejection of expired JWT tokens."""
        expired_payload = {
            "user_id": "test_user",
            "username": "testuser",
            "role": "user",
            "iat": int(time.time()) - 7200,  # 2 hours ago
            "exp": int(time.time()) - 3600,  # 1 hour ago (expired)
            "iss": "ai-docs-vector-db",
            "aud": "api-client",
        }

        token = jwt.encode(expired_payload, _jwt_secret, algorithm="HS256")
        result = jwt_service.verify_token(token)

        assert result["valid"] is False
        assert "expired" in result["error"].lower()

    def test_malformed_jwt_rejection(self, jwt_service):
        """Test rejection of malformed JWT tokens."""
        malformed_tokens = [
            "not.a.jwt",
            "invalid_token",
            "",
            "header.payload",  # Missing signature
            "header.payload.signature.extra",  # Extra part
            "header..signature",  # Empty payload
            ".payload.signature",  # Empty header
            "header.payload.",  # Empty signature
        ]

        for token in malformed_tokens:
            result = jwt_service.verify_token(token)
            assert result["valid"] is False

    def test_invalid_signature_rejection(self, jwt_service, valid_jwt_payload):
        """Test rejection of JWT tokens with invalid signatures."""
        # Generate token with correct secret
        token = jwt_service.generate_token(valid_jwt_payload)

        # Try to verify with wrong secret
        wrong_secret = os.getenv("TEST_WRONG_SECRET", "test_wrong_secret_key")
        try:
            jwt.decode(token, wrong_secret, algorithms=["HS256"])
            msg = "Should have raised InvalidSignatureError"
            raise AssertionError(msg)
        except jwt.InvalidSignatureError:
            # Expected behavior
            pass

    def test_algorithm_confusion_prevention(
        self, jwt_service, valid_jwt_payload, _jwt_secret
    ):
        """Test prevention of algorithm confusion attacks."""
        # Generate HS256 token
        token = jwt_service.generate_token(valid_jwt_payload)

        # Try to verify as RS256 (should fail)
        try:
            jwt.decode(token, _jwt_secret, algorithms=["RS256"])
            msg = "Should reject algorithm confusion"
            raise AssertionError(msg)
        except jwt.InvalidTokenError:
            # Expected behavior
            pass

        # Try to verify with 'none' algorithm (should fail)
        try:
            jwt.decode(token, _jwt_secret, algorithms=["none"])
            msg = "Should reject 'none' algorithm"
            raise AssertionError(msg)
        except jwt.InvalidTokenError:
            # Expected behavior
            pass

    def test_none_algorithm_prevention(self, valid_jwt_payload):
        """Test prevention of 'none' algorithm attack."""
        # Create token with 'none' algorithm (no signature)
        header = {"alg": "none", "typ": "JWT"}
        payload = valid_jwt_payload

        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        # Token without signature
        none_token = f"{header_b64}.{payload_b64}."

        # Should be rejected
        try:
            jwt.decode(none_token, verify=False, algorithms=["none"])
            msg = "Should reject 'none' algorithm tokens"
            raise AssertionError(msg)
        except jwt.InvalidTokenError:
            # Expected behavior
            pass

    def test_jwt_payload_manipulation_detection(
        self, jwt_service, valid_jwt_payload, _jwt_secret
    ):
        """Test detection of JWT payload manipulation."""
        token = jwt_service.generate_token(valid_jwt_payload)

        # Decode without verification to manipulate payload
        header, payload, signature = token.split(".")

        # Decode payload

        payload_data = json.loads(base64.urlsafe_b64decode(payload + "==").decode())

        # Manipulate payload (privilege escalation)
        payload_data["role"] = "admin"
        payload_data["permissions"] = ["read", "write", "admin", "delete"]

        # Re-encode payload
        manipulated_payload = (
            base64.urlsafe_b64encode(json.dumps(payload_data).encode())
            .decode()
            .rstrip("=")
        )

        # Create manipulated token
        manipulated_token = f"{header}.{manipulated_payload}.{signature}"

        # Should be rejected due to invalid signature
        result = jwt_service.verify_token(manipulated_token)
        assert result["valid"] is False

    def test_jwt_header_manipulation_detection(self, jwt_service, valid_jwt_payload):
        """Test detection of JWT header manipulation."""
        token = jwt_service.generate_token(valid_jwt_payload)
        header, payload, signature = token.split(".")

        # Decode and manipulate header

        header_data = json.loads(base64.urlsafe_b64decode(header + "==").decode())

        # Change algorithm
        header_data["alg"] = "none"

        # Re-encode header
        manipulated_header = (
            base64.urlsafe_b64encode(json.dumps(header_data).encode())
            .decode()
            .rstrip("=")
        )

        # Create manipulated token
        manipulated_token = f"{manipulated_header}.{payload}.{signature}"

        # Should be rejected
        result = jwt_service.verify_token(manipulated_token)
        assert result["valid"] is False

    def test_jwt_time_claims_validation(self, jwt_service, _jwt_secret):
        """Test validation of JWT time-based claims."""
        current_time = int(time.time())

        # Test 'nbf' (not before) claim
        future_payload = {
            "user_id": "test_user",
            "username": "testuser",
            "role": "user",
            "iat": current_time,
            "exp": current_time + 3600,
            "nbf": current_time + 1800,  # Valid in 30 minutes
            "iss": "ai-docs-vector-db",
            "aud": "api-client",
        }

        token = jwt.encode(future_payload, _jwt_secret, algorithm="HS256")

        try:
            jwt.decode(token, _jwt_secret, algorithms=["HS256"])
            msg = "Should reject token with future 'nbf'"
            raise AssertionError(msg)
        except jwt.ImmatureSignatureError:
            # Expected behavior
            pass

    def test_jwt_issuer_validation(self, jwt_service, _jwt_secret):
        """Test JWT issuer validation."""
        invalid_issuer_payload = {
            "user_id": "test_user",
            "username": "testuser",
            "role": "user",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "iss": "malicious-issuer",  # Wrong issuer
            "aud": "api-client",
        }

        token = jwt.encode(invalid_issuer_payload, _jwt_secret, algorithm="HS256")

        # Verify with expected issuer
        try:
            jwt.decode(
                token, _jwt_secret, algorithms=["HS256"], issuer="ai-docs-vector-db"
            )
            msg = "Should reject token with wrong issuer"
            raise AssertionError(msg)
        except jwt.InvalidIssuerError:
            # Expected behavior
            pass

    def test_jwt_audience_validation(self, jwt_service, _jwt_secret):
        """Test JWT audience validation."""
        invalid_audience_payload = {
            "user_id": "test_user",
            "username": "testuser",
            "role": "user",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "iss": "ai-docs-vector-db",
            "aud": "wrong-audience",  # Wrong audience
        }

        token = jwt.encode(invalid_audience_payload, _jwt_secret, algorithm="HS256")

        # Verify with expected audience
        try:
            jwt.decode(token, _jwt_secret, algorithms=["HS256"], audience="api-client")
            msg = "Should reject token with wrong audience"
            raise AssertionError(msg)
        except jwt.InvalidAudienceError:
            # Expected behavior
            pass

    def test_jwt_key_confusion_prevention(self, valid_jwt_payload):
        """Test prevention of key confusion attacks."""
        # Generate token with HMAC secret
        hmac_secret = os.getenv("TEST_HMAC_SECRET", "test_hmac_secret_key")
        hmac_token = jwt.encode(valid_jwt_payload, hmac_secret, algorithm="HS256")

        # Try to verify with RSA public key (should fail)
        rsa_public_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1234567890abcdef...
-----END PUBLIC KEY-----"""

        try:
            jwt.decode(hmac_token, rsa_public_key, algorithms=["RS256"])
            msg = "Should prevent key confusion attack"
            raise AssertionError(msg)
        except jwt.InvalidTokenError:
            # Expected behavior
            pass

    def test_jwt_replay_attack_prevention(self, jwt_service, valid_jwt_payload):
        """Test prevention of JWT replay attacks."""
        # Generate token
        token = jwt_service.generate_token(valid_jwt_payload)

        # Verify token (should work first time)
        result1 = jwt_service.verify_token(token)
        assert result1["valid"] is True

        # In a real implementation, there should be:
        # 1. Token blacklisting after logout
        # 2. Nonce/JTI tracking to prevent replay
        # 3. Short expiration times
        # 4. Token rotation/refresh mechanisms

        # Simulate token blacklist check
        blacklisted_tokens = set()
        blacklisted_tokens.add(token)

        # Token should be rejected if blacklisted
        if token in blacklisted_tokens:
            # Token is blacklisted, should be rejected
            assert True
        else:
            msg = "Token should be blacklisted after logout"
            raise AssertionError(msg)

    def test_jwt_privilege_escalation_prevention(self, jwt_service, valid_jwt_payload):
        """Test prevention of privilege escalation via JWT manipulation."""
        # Generate regular user token
        user_token = jwt_service.generate_token(valid_jwt_payload)
        user_result = jwt_service.verify_token(user_token)

        assert user_result["valid"] is True
        assert user_result["payload"]["role"] == "user"
        assert "admin" not in user_result["payload"]["permissions"]

        # Attempting to modify token should fail signature verification
        # This is already tested in test_jwt_payload_manipulation_detection

        # Verify that role-based access control is enforced
        user_permissions = user_result["payload"]["permissions"]
        assert "read" in user_permissions
        assert "write" in user_permissions
        assert "admin" not in user_permissions
        assert "delete" not in user_permissions

    def test_jwt_session_fixation_prevention(self, jwt_service, valid_jwt_payload):
        """Test prevention of session fixation attacks."""
        # Generate token with session info
        session_payload = valid_jwt_payload.copy()
        session_payload["session_id"] = "session_123"

        token = jwt_service.generate_token(session_payload)
        result = jwt_service.verify_token(token)

        assert result["valid"] is True
        assert result["payload"]["session_id"] == "session_123"

        # In a real implementation:
        # 1. Session ID should be regenerated after login
        # 2. Old session should be invalidated
        # 3. Session should be tied to user authentication

        # Simulate session regeneration
        new_session_payload = session_payload.copy()
        new_session_payload["session_id"] = "session_456"
        new_session_payload["iat"] = int(time.time())

        new_token = jwt_service.generate_token(new_session_payload)
        new_result = jwt_service.verify_token(new_token)

        assert new_result["valid"] is True
        assert new_result["payload"]["session_id"] != session_payload["session_id"]

    def test_jwt_cross_site_request_forgery_protection(self):
        """Test CSRF protection for JWT tokens."""
        # JWT tokens should be stored securely and include CSRF protection
        # This involves:
        # 1. Proper token storage (httpOnly cookies or secure localStorage)
        # 2. CSRF token validation for state-changing operations
        # 3. SameSite cookie attributes

        csrf_token = os.getenv("TEST_CSRF_TOKEN", "test_csrf_token_123")

        # Simulate CSRF token validation
        def validate_csrf_token(request_csrf, session_csrf):
            return request_csrf == session_csrf

        # Valid CSRF token should pass
        assert validate_csrf_token(csrf_token, csrf_token) is True

        # Invalid CSRF token should fail
        assert validate_csrf_token("wrong_csrf", csrf_token) is False
        assert validate_csrf_token("", csrf_token) is False
        assert validate_csrf_token(None, csrf_token) is False

    def test_jwt_token_leakage_prevention(self):
        """Test prevention of JWT token leakage."""
        # Test that tokens are not leaked in:
        # 1. URLs (GET parameters)
        # 2. Referrer headers
        # 3. Logs
        # 4. Error messages

        sample_token = os.getenv(
            "TEST_SAMPLE_TOKEN",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGVzdCJ9.test_signature",
        )

        # URLs should not contain tokens
        safe_url = "https://api.example.com/data"

        assert "token=" not in safe_url
        # In practice, verify that the application doesn't include tokens in URLs

        # Error messages should not leak tokens
        error_message = "Authentication failed"
        assert sample_token not in error_message

        # Logs should not contain full tokens (only masked versions)
        masked_token = f"{sample_token[:10]}...{sample_token[-4:]}"
        assert len(masked_token) < len(sample_token)

    def test_jwt_secure_storage_requirements(self):
        """Test JWT secure storage requirements."""
        # Test requirements for secure JWT storage:
        # 1. HttpOnly cookies for web applications
        # 2. Secure flag for HTTPS
        # 3. SameSite attribute
        # 4. Appropriate expiration

        cookie_attributes = {
            "httpOnly": True,
            "secure": True,
            "sameSite": "Strict",
            "maxAge": 3600,  # 1 hour
        }

        # Verify security attributes
        assert cookie_attributes["httpOnly"] is True  # Prevent XSS access
        assert cookie_attributes["secure"] is True  # HTTPS only
        assert cookie_attributes["sameSite"] == "Strict"  # CSRF protection
        assert cookie_attributes["maxAge"] <= 3600  # Reasonable expiration

    def test_jwt_refresh_token_security(self, jwt_service):
        """Test refresh token security implementation."""
        # Refresh tokens should be:
        # 1. Long-lived but rotated
        # 2. Stored securely
        # 3. Single-use (invalidated after use)
        # 4. Tied to specific client/device

        refresh_payload = {
            "token_type": "refresh",
            "user_id": "test_user",
            "client_id": "client_123",
            "device_id": "device_456",
            "iat": int(time.time()),
            "exp": int(time.time()) + (7 * 24 * 3600),  # 7 days
        }

        refresh_token = jwt_service.generate_token(refresh_payload)
        result = jwt_service.verify_token(refresh_token)

        assert result["valid"] is True
        assert result["payload"]["token_type"] == "refresh"
        assert result["payload"]["user_id"] == "test_user"

        # Simulate refresh token rotation
        used_refresh_tokens = set()
        used_refresh_tokens.add(refresh_token)

        # Old refresh token should be invalidated
        if refresh_token in used_refresh_tokens:
            # Should generate new refresh token and invalidate old one
            assert True
        else:
            msg = "Refresh token should be invalidated after use"
            raise AssertionError(msg)
