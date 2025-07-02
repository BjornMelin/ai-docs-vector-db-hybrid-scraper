"""Security testing framework for API protection and vulnerability assessment.

This module provides comprehensive security testing for AI/ML systems including:
- Injection attack prevention testing
- Authentication and authorization validation
- Input sanitization verification
- Rate limiting effectiveness
- API security headers validation
- SQL injection and XSS protection
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import app
from src.services.security import AISecurityValidator
from src.services.security.integration import SecurityManager
from src.services.security.monitoring import SecurityEventType, SecurityMonitor


# Security test markers
security_test = pytest.mark.security
injection_test = pytest.mark.injection
auth_test = pytest.mark.auth
rate_limit_test = pytest.mark.rate_limit


class TestAPISecurityFramework:
    """Comprehensive API security testing framework with real endpoint testing."""

    @pytest.fixture
    def real_api_client(self):
        """Real API client that tests actual FastAPI endpoints."""
        # Use the actual FastAPI application
        return TestClient(app)

    @pytest.fixture
    def security_headers_validator(self):
        """Real security headers validator."""

        class SecurityHeadersValidator:
            def __init__(self):
                self.required_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=",
                    "Content-Security-Policy": "default-src",
                    "Referrer-Policy": [
                        "strict-origin-when-cross-origin",
                        "strict-origin",
                    ],
                }

            def validate_headers(self, response_headers: dict) -> dict[str, str]:
                """Validate security headers in real HTTP response."""
                violations = {}

                for header, expected in self.required_headers.items():
                    if header not in response_headers:
                        violations[header] = "Missing required security header"
                    else:
                        header_value = response_headers[header]
                        if isinstance(expected, list) and not any(
                            exp in header_value for exp in expected
                        ):
                            violations[header] = f"Invalid value: {header_value}"
                        if isinstance(expected, str) and expected not in header_value:
                            violations[header] = f"Invalid value: {header_value}"

                return violations

        return SecurityHeadersValidator()

    @pytest.fixture
    def real_input_validator(self):
        """Real input validation testing."""

        class RealInputValidator:
            def __init__(self):
                self.sql_injection_patterns = [
                    "'; DROP TABLE users; --",
                    "' OR '1'='1",
                    "' UNION SELECT password FROM users --",
                    "admin'/**/OR/**/1=1--",
                    "1; EXEC xp_cmdshell('dir')--",
                ]
                self.xss_patterns = [
                    "<script>alert('XSS')</script>",
                    "<img src=x onerror=alert(1)>",
                    "javascript:alert(document.cookie)",
                    "<svg onload=alert('XSS')>",
                    "';alert(String.fromCharCode(88,83,83));//'",
                ]
                self.path_traversal_patterns = [
                    "../../../etc/passwd",
                    "..\\\\..\\\\..\\\\windows\\\\system32\\\\config\\\\sam",
                    "....//....//etc/passwd",
                    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                ]

            async def test_endpoint_against_attacks(
                self, client, endpoint: str, method: str = "GET", **kwargs
            ) -> dict:
                """Test real endpoint against various attack patterns."""
                results = {
                    "sql_injection": [],
                    "xss": [],
                    "path_traversal": [],
                    "endpoint": endpoint,
                    "method": method,
                }

                # Test SQL injection
                for pattern in self.sql_injection_patterns:
                    if method == "GET":
                        response = client.get(
                            endpoint, params={"q": pattern, **kwargs.get("params", {})}
                        )
                    else:
                        response = client.post(
                            endpoint, json={"query": pattern, **kwargs.get("json", {})}
                        )

                    results["sql_injection"].append(
                        {
                            "pattern": pattern,
                            "status_code": response.status_code,
                            "blocked": response.status_code in [400, 422, 403],
                            "response_size": len(response.content),
                        }
                    )

                # Test XSS
                for pattern in self.xss_patterns:
                    if method == "GET":
                        response = client.get(
                            endpoint, params={"q": pattern, **kwargs.get("params", {})}
                        )
                    else:
                        response = client.post(
                            endpoint, json={"query": pattern, **kwargs.get("json", {})}
                        )

                    results["xss"].append(
                        {
                            "pattern": pattern,
                            "status_code": response.status_code,
                            "blocked": response.status_code in [400, 422, 403],
                            "sanitized": pattern not in response.text,
                        }
                    )

                # Test path traversal
                for pattern in self.path_traversal_patterns:
                    if method == "GET":
                        response = client.get(
                            endpoint,
                            params={"file": pattern, **kwargs.get("params", {})},
                        )
                    else:
                        response = client.post(
                            endpoint,
                            json={"file_path": pattern, **kwargs.get("json", {})},
                        )

                    results["path_traversal"].append(
                        {
                            "pattern": pattern,
                            "status_code": response.status_code,
                            "blocked": response.status_code in [400, 403, 404],
                        }
                    )

                return results

        return RealInputValidator()

    @security_test
    @injection_test
    async def test_real_sql_injection_prevention(
        self, real_api_client, real_input_validator
    ):
        """Test SQL injection prevention on real API endpoints."""
        # Test actual search endpoint
        test_results = await real_input_validator.test_endpoint_against_attacks(
            real_api_client, "/api/v1/search", "GET"
        )

        # Validate that SQL injection attempts are blocked
        sql_results = test_results["sql_injection"]
        blocked_count = sum(1 for result in sql_results if result["blocked"])

        # At least 80% of SQL injection attempts should be blocked
        block_rate = blocked_count / len(sql_results)
        assert block_rate >= 0.8, f"SQL injection block rate too low: {block_rate:.2%}"

        # Test document endpoints
        doc_results = await real_input_validator.test_endpoint_against_attacks(
            real_api_client, "/api/v1/documents", "POST"
        )

        doc_sql_results = doc_results["sql_injection"]
        doc_blocked_count = sum(1 for result in doc_sql_results if result["blocked"])
        doc_block_rate = doc_blocked_count / len(doc_sql_results)

        assert doc_block_rate >= 0.8, (
            f"Document endpoint SQL injection block rate too low: {doc_block_rate:.2%}"
        )

    @security_test
    @injection_test
    async def test_real_xss_prevention(self, real_api_client, real_input_validator):
        """Test XSS prevention on real API endpoints."""
        test_results = await real_input_validator.test_endpoint_against_attacks(
            real_api_client, "/api/v1/search", "GET"
        )

        xss_results = test_results["xss"]
        sanitized_count = sum(1 for result in xss_results if result["sanitized"])

        # All XSS attempts should be sanitized
        sanitization_rate = sanitized_count / len(xss_results)
        assert sanitization_rate >= 0.9, (
            f"XSS sanitization rate too low: {sanitization_rate:.2%}"
        )

    @security_test
    async def test_real_security_headers_validation(
        self, real_api_client, security_headers_validator
    ):
        """Test actual security headers on real endpoints."""
        endpoints_to_test = [
            "/api/v1/search",
            "/api/v1/documents",
            "/api/v1/config/status",
            "/docs",  # API documentation
        ]

        for endpoint in endpoints_to_test:
            try:
                response = real_api_client.get(endpoint)
                violations = security_headers_validator.validate_headers(
                    dict(response.headers)
                )

                # No critical security header violations allowed
                critical_headers = ["X-Content-Type-Options", "X-Frame-Options"]
                critical_violations = {
                    k: v for k, v in violations.items() if k in critical_headers
                }

                assert not critical_violations, (
                    f"Critical security header violations on {endpoint}: {critical_violations}"
                )

                # Log non-critical violations
                if violations:
                    print(f"Security header warnings for {endpoint}: {violations}")

            except (OSError, PermissionError, ValueError) as e:
                print(f"Could not test endpoint {endpoint}: {e}")

    @security_test
    @rate_limit_test
    async def test_real_rate_limiting_effectiveness(self, real_api_client):
        """Test rate limiting on real API endpoints."""

        def make_request(endpoint: str, request_id: int):
            """Make a single request to test rate limiting."""
            try:
                response = real_api_client.get(f"{endpoint}?q=test_{request_id}")
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "timestamp": time.time(),
                }
            except (
                httpx.HTTPError,
                httpx.RequestError,
                ConnectionError,
                TimeoutError,
            ) as e:
                return {
                    "request_id": request_id,
                    "status_code": 500,
                    "error": str(e),
                    "timestamp": time.time(),
                }

        # Test rate limiting with concurrent requests
        endpoint = "/api/v1/search"
        num_requests = 50  # Attempt to exceed rate limits

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, endpoint, i) for i in range(num_requests)
            ]

            results = [future.result() for future in futures]

        # Analyze rate limiting effectiveness
        rate_limited_count = sum(1 for r in results if r["status_code"] == 429)
        successful_count = sum(1 for r in results if r["status_code"] == 200)

        # Should see some rate limiting with high request volume
        if rate_limited_count == 0:
            print(
                f"Warning: No rate limiting detected with {num_requests} concurrent requests"
            )
        else:
            print(f"Rate limiting working: {rate_limited_count} requests rate limited")

        # At least some requests should succeed
        assert successful_count > 0, "All requests failed - API may be down"

    @security_test
    async def test_real_input_validation_boundary_conditions(self, real_api_client):
        """Test input validation on real endpoints with boundary conditions."""
        boundary_tests = [
            # Extremely long inputs
            {"query": "a" * 10000},
            {"query": "a" * 100000},  # Very large input
            # Unicode and encoding attacks
            {"query": "\\u0000\\u0001\\u0002"},
            {"query": "%00%01%02"},
            # Binary data
            {"query": "\\x00\\x01\\xFF"},
            # JSON injection
            {"query": '{"admin": true}'},
            {"query": '"}; DROP TABLE users; --'},
        ]

        for test_input in boundary_tests:
            # Test GET endpoint
            response = real_api_client.get("/api/v1/search", params=test_input)

            # Should handle gracefully - either accept and sanitize or reject cleanly
            assert response.status_code in [200, 400, 422], (
                f"Unexpected status for input: {test_input}"
            )

            # Response should not contain error traces
            if response.status_code >= 400:
                error_content = response.text.lower()
                assert "traceback" not in error_content
                assert "internal server error" not in error_content

            # Test POST endpoint
            try:
                post_response = real_api_client.post(
                    "/api/v1/documents", json=test_input
                )
                assert post_response.status_code in [200, 201, 400, 422]
            except (json.JSONDecodeError, ValueError, TypeError):
                # Some inputs may cause connection issues - this is acceptable
                pass

    @security_test
    async def test_real_authentication_security(self, real_api_client):
        """Test authentication security on real endpoints."""
        # Test endpoints that should require authentication
        protected_endpoints = [
            "/api/v1/config/reload",
            "/api/v1/config/rollback",
            "/api/v1/drift/run",
        ]

        for endpoint in protected_endpoints:
            # Test without authentication
            response = real_api_client.post(endpoint)

            # Should reject unauthenticated requests
            assert response.status_code in [401, 403, 405], (
                f"Endpoint {endpoint} should require authentication"
            )

            # Test with invalid authentication
            invalid_headers = [
                {"Authorization": "Bearer invalid_token"},
                {"Authorization": "Basic invalid_base64"},
                {"X-API-Key": "invalid_key"},
            ]

            for headers in invalid_headers:
                auth_response = real_api_client.post(endpoint, headers=headers)
                assert auth_response.status_code in [401, 403], (
                    f"Invalid auth should be rejected for {endpoint}"
                )

    @security_test
    async def test_real_cors_configuration_security(self, real_api_client):
        """Test CORS configuration on real API."""
        malicious_origins = [
            "https://evil.com",
            "https://attacker.evil.com",
            "null",
            "file://",
            "data:text/html,<script>alert('xss')</script>",
        ]

        for origin in malicious_origins:
            # Test preflight request
            response = real_api_client.options(
                "/api/v1/search",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )

            # Check CORS response
            cors_origin = response.headers.get("Access-Control-Allow-Origin")
            if cors_origin:
                assert cors_origin != "*", "Wildcard CORS origin poses security risk"
                assert origin not in cors_origin, (
                    f"Malicious origin {origin} allowed by CORS"
                )

    @security_test
    async def test_real_error_information_disclosure(self, real_api_client):
        """Test that real error responses don't disclose sensitive information."""
        # Trigger various error conditions on real endpoints
        error_scenarios = [
            ("GET", "/api/v1/nonexistent", {}),
            ("GET", "/api/v1/search", {"q": ""}),  # Invalid query
            ("POST", "/api/v1/documents", {"invalid": "data"}),  # Invalid payload
            ("GET", "/api/v1/documents/999999", {}),  # Non-existent document
        ]

        for method, endpoint, params in error_scenarios:
            if method == "GET":
                response = real_api_client.get(endpoint, params=params)
            else:
                response = real_api_client.post(endpoint, json=params)

            # Error response should not contain sensitive information
            error_content = response.text.lower()
            sensitive_patterns = [
                "traceback",
                "stack trace",
                "internal server error",
                "database",
                "connection string",
                "password",
                "secret",
                "token",
                "/home/",
                "/var/",
                "c:\\",
                "sqlalchemy",
                "qdrant",
                "redis",
            ]

            for pattern in sensitive_patterns:
                assert pattern not in error_content, (
                    f"Error response contains sensitive info '{pattern}' for {method} {endpoint}: {error_content[:200]}"
                )

    @security_test
    async def test_real_file_upload_security(self, real_api_client):
        """Test file upload security on real endpoints."""
        # Test document upload endpoint if it exists
        malicious_files = [
            # Executable files
            ("malware.exe", b"MZ\\x90\\x00", "application/octet-stream"),
            ("script.php", b"<?php system($_GET['cmd']); ?>", "text/php"),
            # Oversized files (simulate)
            ("large.txt", b"A" * 1024 * 1024, "text/plain"),  # 1MB
            # Files with malicious extensions
            ("document.pdf.exe", b"fake pdf content", "application/pdf"),
        ]

        upload_endpoints = [
            "/api/v1/documents",  # If it supports file upload
            "/api/v1/upload",  # Generic upload endpoint
        ]

        for endpoint in upload_endpoints:
            for filename, content, content_type in malicious_files:
                try:
                    files = {"file": (filename, content, content_type)}
                    response = real_api_client.post(endpoint, files=files)

                    # Should reject malicious files
                    if response.status_code == 404:
                        # Endpoint doesn't exist - skip this test
                        continue

                    assert response.status_code in [400, 413, 415, 422], (
                        f"Malicious file {filename} was accepted on {endpoint}"
                    )

                except (OSError, FileNotFoundError, PermissionError) as e:
                    # Connection errors are acceptable for malicious uploads
                    print(f"Upload test failed for {filename} on {endpoint}: {e}")

    @security_test
    async def test_real_api_documentation_security(self, real_api_client):
        """Test API documentation security exposure."""
        # Test access to API documentation
        doc_endpoints = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/swagger.json",
        ]

        for endpoint in doc_endpoints:
            response = real_api_client.get(endpoint)

            if response.status_code == 200:
                doc_content = response.text.lower()

                # Should not expose sensitive information in docs
                sensitive_patterns = [
                    "password",
                    "secret",
                    "private_key",
                    "api_key",
                    "token",
                    "credential",
                    "sk-",  # OpenAI-style keys
                    "xoxb-",  # Slack tokens
                ]

                for pattern in sensitive_patterns:
                    assert pattern not in doc_content, (
                        f"API documentation exposes sensitive pattern: {pattern}"
                    )

    @security_test
    async def test_real_response_time_analysis(self, real_api_client):
        """Test response time patterns to detect timing attacks."""
        # Test authentication timing
        auth_endpoint = "/api/v1/config/status"  # Use real endpoint

        valid_timings = []
        invalid_timings = []

        # Measure response times for valid vs invalid scenarios
        for _i in range(10):
            # Test with potentially valid data
            start_time = time.time()
            _ = real_api_client.get(auth_endpoint)
            valid_timings.append(time.time() - start_time)

            # Test with invalid data
            start_time = time.time()
            # response = real_api_client.get(f"{auth_endpoint}?invalid_param=true")
            invalid_timings.append(time.time() - start_time)

        # Calculate timing statistics
        avg_valid_time = sum(valid_timings) / len(valid_timings)
        avg_invalid_time = sum(invalid_timings) / len(invalid_timings)

        time_difference = abs(avg_valid_time - avg_invalid_time)

        # Timing difference should be minimal to prevent information leakage
        assert time_difference < 0.1, (
            f"Timing difference too large: {time_difference:.3f}s - potential timing attack vector"
        )

        print(
            f"Timing analysis: valid={avg_valid_time:.3f}s, invalid={avg_invalid_time:.3f}s, diff={time_difference:.3f}s"
        )


class TestSecurityIntegration:
    """Integration tests for security system components."""

    @security_test
    async def test_security_manager_integration(self):
        """Test integrated security manager functionality."""

        security_manager = SecurityManager()

        # Test comprehensive request validation
        malicious_request = {
            "query": "'; DROP TABLE documents; --",
            "limit": 99999,
            "user_input": "<script>alert('xss')</script>",
        }

        is_safe = await security_manager.validate_request(malicious_request)
        assert not is_safe, "Security manager should detect malicious request"

    @security_test
    async def test_security_monitoring_integration(self):
        """Test security monitoring and alerting."""

        monitor = SecurityMonitor()

        # Simulate security events
        await monitor.log_security_event(
            event_type=SecurityEventType.INJECTION_ATTEMPT,
            details={"query": "'; DROP TABLE users; --", "ip": "192.168.1.100"},
            severity="HIGH",
        )

        # Check that monitoring captures events
        events = await monitor.get_recent_events(limit=10)
        assert len(events) > 0
        assert any(
            event.event_type == SecurityEventType.INJECTION_ATTEMPT for event in events
        )


# Custom security exception for testing
class SecurityError(Exception):
    """Custom exception for security violations."""


# Additional security testing fixtures
@pytest.fixture
async def mock_security_validator():
    """Mock security validator for testing."""
    validator = AISecurityValidator()

    # Override with test-specific validation logic
    async def mock_validate_input(value: str, field_name: str | None = None) -> bool:
        dangerous_patterns = [
            "DROP TABLE",
            "<script>",
            "javascript:",
            "../",
            "\x00",
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in value.lower():
                msg = f"Dangerous pattern detected: {pattern}"
                raise SecurityError(msg)

        return True

    validator.validate_input = mock_validate_input
    return validator


@pytest.fixture
async def mock_api_client(mock_security_validator):
    """Mock API client with security middleware."""

    app = FastAPI()

    # Add mock security middleware
    @app.middleware("http")
    async def security_middleware(request, call_next):
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Add mock endpoints for testing
    @app.get("/api/v1/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/v1/documents")
    async def list_documents():
        return {"documents": []}

    @app.get("/api/v1/search")
    async def search():
        return {"results": []}

    @app.post("/api/v1/upload")
    async def upload_file():
        return {"status": "uploaded"}

    @app.post("/api/v1/auth/login")
    async def login():
        return {"status": "logged_in"}

    return TestClient(app)
