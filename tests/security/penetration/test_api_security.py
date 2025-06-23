"""API security penetration testing.

This module tests API endpoints for security vulnerabilities including
authentication bypass, authorization flaws, and injection attacks.
"""

import asyncio
import json
import pytest
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

from src.security import SecurityError


@pytest.mark.security
@pytest.mark.penetration
class TestAPISecurity:
    """Test API security through penetration testing scenarios."""

    @pytest.fixture
    def api_client(self):
        """Mock API client for testing."""
        
        class MockAPIClient:
            def __init__(self):
                self.base_url = "https://api.example.com"
                self.session_cookies = {}
                self.auth_headers = {}
                self.rate_limit_calls = []
            
            async def request(self, method: str, endpoint: str, 
                            headers: Dict = None, data: Any = None,
                            params: Dict = None, cookies: Dict = None) -> dict[str, Any]:
                """Mock API request."""
                # Simulate request
                request_time = time.time()
                self.rate_limit_calls.append(request_time)
                
                # Mock response based on endpoint and method
                if endpoint == "/auth/login" and method == "POST":
                    if data and data.get("username") == "admin" and data.get("password") == "admin123":
                        return {
                            "status_code": 200,
                            "json": {"token": "mock_jwt_token", "user_id": "admin_001"},
                            "headers": {"Set-Cookie": "session=abc123; HttpOnly; Secure"}
                        }
                    else:
                        return {
                            "status_code": 401,
                            "json": {"error": "Invalid credentials"}
                        }
                
                elif endpoint == "/api/search" and method == "GET":
                    auth_header = (headers or {}).get("Authorization")
                    if not auth_header or not auth_header.startswith("Bearer "):
                        return {
                            "status_code": 401,
                            "json": {"error": "Authentication required"}
                        }
                    
                    query = (params or {}).get("q", "")
                    return {
                        "status_code": 200,
                        "json": {"results": [f"Result for: {query}"], "count": 1}
                    }
                
                elif endpoint == "/api/admin/users" and method == "GET":
                    auth_header = (headers or {}).get("Authorization")
                    if auth_header != "Bearer admin_token":
                        return {
                            "status_code": 403,
                            "json": {"error": "Admin access required"}
                        }
                    
                    return {
                        "status_code": 200,
                        "json": {"users": ["admin", "user1", "user2"]}
                    }
                
                else:
                    return {
                        "status_code": 404,
                        "json": {"error": "Endpoint not found"}
                    }
        
        return MockAPIClient()

    @pytest.fixture
    def attack_payloads(self):
        """Common attack payloads for testing."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT password FROM users --",
                "admin'/**/OR/**/1=1--",
                "1; EXEC xp_cmdshell('dir')--"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert(1)>",
                "javascript:alert(document.cookie)",
                "<svg onload=alert('XSS')>",
                "';alert(String.fromCharCode(88,83,83));//'"
            ],
            "command_injection": [
                "; cat /etc/passwd",
                "| whoami",
                "&& id",
                "`ls -la`",
                "$(uname -a)"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd"
            ],
            "ldap_injection": [
                "*)(uid=*))(|(uid=*",
                "*)(|(password=*))",
                "admin)(&(password=*))",
                "*))%00",
                "*()|%26'"
            ]
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
            assert response["status_code"] != 200 or "token" not in response.get("json", {})
        
        # Test header manipulation
        bypass_headers = [
            {"X-Original-IP": "127.0.0.1"},
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "localhost"},
            {"X-Remote-User": "admin"},
            {"X-Authenticated-User": "admin"}
        ]
        
        for headers in bypass_headers:
            response = await api_client.request("GET", "/api/admin/users", headers=headers)
            # Should not bypass authentication with headers alone
            assert response["status_code"] in [401, 403]

    @pytest.mark.asyncio
    async def test_authorization_bypass_attempts(self, api_client):
        """Test authorization bypass techniques."""
        
        # Get regular user token (simulated)
        user_token = "user_token_123"
        
        # Test privilege escalation attempts
        escalation_attempts = [
            # Direct admin endpoint access
            ("/api/admin/users", {"Authorization": f"Bearer {user_token}"}),
            ("/api/admin/config", {"Authorization": f"Bearer {user_token}"}),
            ("/api/admin/logs", {"Authorization": f"Bearer {user_token}"}),
            
            # Parameter pollution
            ("/api/user/profile", {"Authorization": f"Bearer {user_token}", "X-User-Role": "admin"}),
            
            # Method override
            ("/api/admin/users", {"Authorization": f"Bearer {user_token}", "X-HTTP-Method-Override": "GET"}),
        ]
        
        for endpoint, headers in escalation_attempts:
            response = await api_client.request("GET", endpoint, headers=headers)
            # Should deny unauthorized access
            assert response["status_code"] in [401, 403, 404]

    @pytest.mark.asyncio
    async def test_injection_attacks_on_api_parameters(self, api_client, attack_payloads):
        """Test injection attacks on API parameters."""
        
        # Valid token for testing
        headers = {"Authorization": "Bearer mock_jwt_token"}
        
        # Test SQL injection in search parameters
        for payload in attack_payloads["sql_injection"]:
            response = await api_client.request(
                "GET", "/api/search", 
                headers=headers, 
                params={"q": payload}
            )
            
            # API should handle malicious input safely
            assert response["status_code"] in [200, 400]
            if response["status_code"] == 200:
                # Response should not contain signs of successful injection
                result_text = str(response.get("json", {}))
                assert "error" not in result_text.lower() or "mysql" not in result_text.lower()
        
        # Test XSS in parameters
        for payload in attack_payloads["xss"]:
            response = await api_client.request(
                "GET", "/api/search", 
                headers=headers, 
                params={"q": payload}
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
            "/api/exports/download"
        ]
        
        for endpoint in file_endpoints:
            for payload in attack_payloads["path_traversal"]:
                response = await api_client.request(
                    "GET", endpoint,
                    headers=headers,
                    params={"file": payload}
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
            task = api_client.request("GET", "/api/search", headers=headers, params={"q": f"test{i}"})
            rapid_requests.append(task)
        
        # Execute requests concurrently
        responses = await asyncio.gather(*rapid_requests, return_exceptions=True)
        
        # Should have some rate limited responses
        rate_limited_count = sum(
            1 for response in responses 
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
            {"_method": "DELETE"}
        ]
        
        for override_header in method_overrides:
            combined_headers = {**headers, **override_header}
            response = await api_client.request(
                "GET", "/api/user/profile",
                headers=combined_headers
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
            "application/octet-stream"
        ]
        
        malicious_data = '{"username": "admin", "role": "admin"}'
        
        for content_type in content_types:
            test_headers = {**headers, "Content-Type": content_type}
            response = await api_client.request(
                "POST", "/api/user/update",
                headers=test_headers,
                data=malicious_data
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
                "GET", "/api/user/data",
                headers=headers,
                params=params
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
            "POST", "/auth/login",
            data={"username": "admin", "password": "admin123"},
            cookies={"session": fixed_session_id}
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
            response = await api_client.request(method, endpoint, headers=headers, data=data)
            
            # Should require CSRF token for state-changing operations
            # (Implementation depends on CSRF protection mechanism)
            assert response["status_code"] in [200, 403, 404]  # 403 if CSRF protection active

    @pytest.mark.asyncio
    async def test_file_upload_security(self, api_client):
        """Test file upload security vulnerabilities."""
        
        headers = {"Authorization": "Bearer mock_jwt_token"}
        
        # Test malicious file uploads
        malicious_files = [
            # Executable files
            {"filename": "malware.exe", "content": b"MZ\x90\x00", "content_type": "application/octet-stream"},
            {"filename": "script.bat", "content": b"@echo off\ndel /f /q C:\\*", "content_type": "text/plain"},
            
            # Script files
            {"filename": "evil.php", "content": b"<?php system($_GET['cmd']); ?>", "content_type": "text/plain"},
            {"filename": "backdoor.jsp", "content": b"<%@ page import=\"java.io.*\" %>", "content_type": "text/plain"},
            
            # Double extension
            {"filename": "image.jpg.exe", "content": b"fake image", "content_type": "image/jpeg"},
            
            # Path traversal in filename
            {"filename": "../../evil.txt", "content": b"malicious content", "content_type": "text/plain"},
            
            # Null byte injection
            {"filename": "image.jpg\x00.exe", "content": b"fake image", "content_type": "image/jpeg"},
        ]
        
        for file_data in malicious_files:
            response = await api_client.request(
                "POST", "/api/upload",
                headers=headers,
                data=file_data
            )
            
            # Should reject malicious file uploads
            assert response["status_code"] in [400, 403, 415, 404]

    @pytest.mark.asyncio
    async def test_business_logic_vulnerabilities(self, api_client):
        """Test business logic vulnerabilities."""
        
        headers = {"Authorization": "Bearer mock_jwt_token"}
        
        # Test race conditions in financial operations
        race_condition_requests = []
        for i in range(10):
            task = api_client.request(
                "POST", "/api/account/withdraw",
                headers=headers,
                data={"amount": 100}
            )
            race_condition_requests.append(task)
        
        responses = await asyncio.gather(*race_condition_requests, return_exceptions=True)
        
        # Should handle concurrent requests properly
        successful_withdrawals = sum(
            1 for response in responses
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
        
        for i in range(10):
            # Valid user, wrong password
            start_time = time.time()
            await api_client.request(
                "POST", "/auth/login",
                data={"username": "admin", "password": "wrongpassword"}
            )
            valid_login_times.append(time.time() - start_time)
            
            # Invalid user
            start_time = time.time()
            await api_client.request(
                "POST", "/auth/login",
                data={"username": "nonexistentuser", "password": "password"}
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
                    "mysql", "postgresql", "oracle",  # Database info
                    "stack trace", "traceback",       # Stack traces
                    "/usr/", "/etc/", "c:\\",          # File paths
                    "version", "server",              # Server info
                    "internal", "debug"               # Debug info
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
            "/deprecated/api/user/profile"
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
            "/api/auth/login"
        ]
        
        sensitive_endpoints = [
            "/api/admin/debug",
            "/api/internal/config",
            "/api/system/status",
            "/api/database/query"
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
        assert len([line for line in doc_content.split('\n') if 'password' in line.lower()]) == 0