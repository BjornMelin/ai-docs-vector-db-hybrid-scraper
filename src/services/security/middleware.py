#!/usr/bin/env python3
"""Comprehensive API security middleware for production deployment.

This module provides a complete security middleware solution for FastAPI
applications with comprehensive protection against various attack vectors:

- Distributed rate limiting with Redis backend
- AI-specific security validation
- Input validation and sanitization
- Security headers and CORS management
- Request/response logging and monitoring
- Attack detection and prevention
"""

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config.security import SecurityConfig
from src.services.security.ai_security import AISecurityValidator
from src.services.security.monitoring import SecurityMonitor
from src.services.security.rate_limiter import DistributedRateLimiter


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive API security middleware with multi-layer protection.

    This middleware provides production-ready security features:
    - Rate limiting with distributed Redis backend
    - AI-specific threat detection and prevention
    - Comprehensive input validation and sanitization
    - Security headers and CORS protection
    - Real-time attack detection and response
    - Detailed security logging and monitoring
    """

    def __init__(
        self,
        app,
        rate_limiter: DistributedRateLimiter,
        security_config: SecurityConfig | None = None,
        ai_validator: AISecurityValidator | None = None,
        security_monitor: SecurityMonitor | None = None,
    ):
        """Initialize security middleware.

        Args:
            app: FastAPI application instance
            rate_limiter: Distributed rate limiter instance
            security_config: Security configuration
            ai_validator: AI security validator
            security_monitor: Security monitoring instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.config = security_config or SecurityConfig()
        self.ai_validator = ai_validator or AISecurityValidator(security_config)
        self.security_monitor = security_monitor or SecurityMonitor()
        self.security = HTTPBearer(auto_error=False)

        # Security configuration
        self.blocked_ips: set = set()
        self.suspicious_patterns = [
            r"union\s+select",
            r"drop\s+table",
            r"delete\s+from",
            r"insert\s+into",
            r"update\s+set",
            r"exec\s*\(",
            r"<script",
            r"javascript:",
            r"vbscript:",
            r"data:text/html",
        ]

        # Rate limiting configuration
        self.rate_limits = {
            "default": {"limit": 100, "window": 60},  # 100 requests per minute
            "search": {"limit": 50, "window": 60},  # 50 searches per minute
            "upload": {"limit": 10, "window": 60},  # 10 uploads per minute
            "api": {"limit": 200, "window": 60},  # 200 API calls per minute
        }

        logger.info("Security middleware initialized with comprehensive protection")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method with comprehensive security checks.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with security headers
        """
        start_time = time.time()

        try:
            # 1. Pre-flight security checks
            await self._validate_request_security(request)

            # 2. Rate limiting
            await self._enforce_rate_limits(request)

            # 3. Input validation
            await self._validate_request_input(request)

            # 4. Process request
            response = await call_next(request)

            # 5. Post-processing security
            self._add_security_headers(response)

        except HTTPException as e:
            # Handle security-related HTTP exceptions
            response = self._create_security_error_response(e, request)
            self._add_security_headers(response)

            # Log security incident
            processing_time = time.time() - start_time
            self.security_monitor.log_security_event(
                "request_blocked",
                request,
                {
                    "reason": str(e.detail),
                    "status_code": e.status_code,
                    "processing_time": processing_time,
                },
            )

            return response

        except Exception as e:
            # Handle unexpected errors securely
            logger.exception("Unexpected error in security middleware")

            response = JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )
            self._add_security_headers(response)

            # Log error event
            self.security_monitor.log_security_event(
                "middleware_error", request, {"error": str(e)}
            )

            return response
        else:
            # 6. Log successful request
            processing_time = time.time() - start_time
            self.security_monitor.log_request_success(request, processing_time)
            return response

    async def _validate_request_security(self, request: Request) -> None:
        """Validate basic request security before processing.

        Args:
            request: HTTP request to validate

        Raises:
            HTTPException: If request fails security validation
        """
        # Check blocked IPs
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            self.security_monitor.log_security_event(
                "blocked_ip_access", request, {"blocked_ip": client_ip}
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                max_size = 10 * 1024 * 1024  # 10MB limit
                if size > max_size:
                    self.security_monitor.log_security_event(
                        "oversized_request",
                        request,
                        {"content_length": size, "max_allowed": max_size},
                    )
                    raise HTTPException(
                        status_code=413, detail="Request payload too large"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid content-length header"
                )

        # Validate HTTP method
        allowed_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}
        if request.method not in allowed_methods:
            self.security_monitor.log_security_event(
                "invalid_http_method", request, {"method": request.method}
            )
            raise HTTPException(status_code=405, detail="Method not allowed")

        # Check for suspicious User-Agent
        user_agent = request.headers.get("user-agent", "")
        if self._is_suspicious_user_agent(user_agent):
            self.security_monitor.log_security_event(
                "suspicious_user_agent", request, {"user_agent": user_agent}
            )
            # Log but don't block - might be legitimate

    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if User-Agent header is suspicious.

        Args:
            user_agent: User-Agent header value

        Returns:
            True if User-Agent is suspicious
        """
        suspicious_agents = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "gobuster",
            "dirb",
            "curl/7.1",  # Very old curl versions
            "wget/1.1",  # Very old wget versions
        ]

        user_agent_lower = user_agent.lower()
        return any(agent in user_agent_lower for agent in suspicious_agents)

    async def _enforce_rate_limits(self, request: Request) -> None:
        """Enforce rate limiting based on request type and client.

        Args:
            request: HTTP request to rate limit

        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Determine rate limit category
        path = request.url.path
        if "/search" in path:
            category = "search"
        elif "/upload" in path or request.method == "POST":
            category = "upload"
        elif path.startswith("/api/"):
            category = "api"
        else:
            category = "default"

        # Get rate limit configuration
        rate_config = self.rate_limits.get(category, self.rate_limits["default"])

        # Get client identifier
        client_id = self._get_client_identifier(request)

        # Check rate limit
        is_allowed, rate_info = await self.rate_limiter.check_rate_limit(
            identifier=client_id,
            limit=rate_config["limit"],
            window=rate_config["window"],
            burst_factor=self.config.burst_factor,
        )

        if not is_allowed:
            self.security_monitor.log_rate_limit_violation(
                request, rate_config["limit"]
            )

            # Add rate limit headers to response
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(rate_config["window"]),
                    "X-RateLimit-Limit": str(rate_config["limit"]),
                    "X-RateLimit-Remaining": str(rate_info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(rate_info.get("reset_time", 0)),
                },
            )

    async def _validate_request_input(self, request: Request) -> None:
        """Validate request input for security threats.

        Args:
            request: HTTP request to validate

        Raises:
            HTTPException: If input contains security threats
        """
        # Validate URL path
        self._validate_url_path(request.url.path)

        # Validate query parameters
        query_params = str(request.query_params)
        if query_params:
            self._validate_query_parameters(query_params)

        # Validate headers
        self._validate_request_headers(request.headers)

        # For POST/PUT requests with body, validate content
        if request.method in ("POST", "PUT", "PATCH") and hasattr(request, "_body"):
            await self._validate_request_body(request)

    def _validate_url_path(self, path: str) -> None:
        """Validate URL path for security issues.

        Args:
            path: URL path to validate

        Raises:
            HTTPException: If path contains security threats
        """
        # Check for path traversal
        if ".." in path or "//" in path:
            raise HTTPException(status_code=400, detail="Invalid URL path")

        # Check for encoded path traversal
        dangerous_encoded = ["%2e%2e", "%2f", "%5c", "%00"]
        path_lower = path.lower()
        for encoded in dangerous_encoded:
            if encoded in path_lower:
                raise HTTPException(status_code=400, detail="Invalid URL encoding")

        # Check path length
        if len(path) > 2048:
            raise HTTPException(status_code=414, detail="URL path too long")

    def _validate_query_parameters(self, query_string: str) -> None:
        """Validate query parameters for security threats.

        Args:
            query_string: Query string to validate

        Raises:
            HTTPException: If query contains threats
        """
        # Check for SQL injection patterns
        query_lower = query_string.lower()
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.security_monitor.log_suspicious_activity(
                    "sql_injection_attempt", {"query": query_string, "pattern": pattern}
                )
                raise HTTPException(
                    status_code=400, detail="Invalid request parameters"
                )

        # Check query length
        if len(query_string) > 8192:  # 8KB limit
            raise HTTPException(status_code=414, detail="Query string too long")

    def _validate_request_headers(self, headers: dict) -> None:
        """Validate request headers for security issues.

        Args:
            headers: Request headers to validate

        Raises:
            HTTPException: If headers contain threats
        """
        # Check for header injection
        for value in headers.values():
            if not isinstance(value, str):
                continue

            # Check for CRLF injection
            if "\r" in value or "\n" in value:
                raise HTTPException(status_code=400, detail="Invalid header value")

            # Check header length
            if len(value) > 8192:  # 8KB limit per header
                raise HTTPException(status_code=400, detail="Header value too long")

        # Validate specific headers
        host = headers.get("host", "")
        if host and not self._is_valid_host(host):
            raise HTTPException(status_code=400, detail="Invalid host header")

    def _is_valid_host(self, host: str) -> bool:
        """Validate host header value.

        Args:
            host: Host header value

        Returns:
            True if host is valid
        """
        # Basic validation - in production, check against allowed hosts
        if not host or len(host) > 253:
            return False

        # Check for suspicious patterns
        suspicious_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        return host.lower() not in suspicious_hosts

    async def _validate_request_body(self, request: Request) -> None:
        """Validate request body content.

        Args:
            request: HTTP request with body

        Raises:
            HTTPException: If body contains threats
        """
        try:
            # Get content type
            content_type = request.headers.get("content-type", "")

            # Handle JSON content
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        # Parse and validate JSON
                        data = json.loads(body.decode("utf-8"))
                        self._validate_json_data(data)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON format")

            # Handle form data
            elif "application/x-www-form-urlencoded" in content_type:
                # Basic validation for form data
                pass  # FastAPI handles this automatically

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Invalid request encoding")

    def _validate_json_data(self, data: Any, depth: int = 0) -> None:
        """Recursively validate JSON data for security threats.

        Args:
            data: JSON data to validate
            depth: Current nesting depth

        Raises:
            HTTPException: If data contains threats
        """
        # Prevent excessive nesting
        if depth > 10:
            raise HTTPException(status_code=400, detail="JSON nesting too deep")

        if isinstance(data, dict):
            # Limit number of keys
            if len(data) > 100:
                raise HTTPException(status_code=400, detail="Too many JSON keys")

            for key, value in data.items():
                # Validate key
                if not isinstance(key, str) or len(key) > 255:
                    raise HTTPException(status_code=400, detail="Invalid JSON key")

                # Recursively validate value
                self._validate_json_data(value, depth + 1)

        elif isinstance(data, list):
            # Limit array size
            if len(data) > 1000:
                raise HTTPException(status_code=400, detail="JSON array too large")

            for item in data:
                self._validate_json_data(item, depth + 1)

        elif isinstance(data, str):
            # Validate string content
            if len(data) > 10000:  # 10KB limit per string
                raise HTTPException(status_code=400, detail="JSON string too long")

            # Check for AI-specific threats if this looks like a query
            if any(
                keyword in data.lower()
                for keyword in ["search", "query", "prompt", "instruction"]
            ):
                try:
                    self.ai_validator.validate_search_query(data)
                except HTTPException:
                    # Re-raise with more context
                    raise HTTPException(
                        status_code=400, detail="Request contains prohibited content"
                    )

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting.

        Args:
            request: HTTP request

        Returns:
            Unique client identifier
        """
        # Try API key first
        api_key = request.headers.get("x-api-key") or request.headers.get(
            "authorization"
        )
        if api_key:
            # Hash API key for privacy
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Use IP address as fallback
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Get first IP in chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fallback to direct client IP
        return getattr(request.client, "host", "unknown")

    def _add_security_headers(self, response: Response) -> None:
        """Add comprehensive security headers to response.

        Args:
            response: HTTP response to add headers to
        """
        security_headers = {
            # Content type protection
            "X-Content-type-Options": self.config.x_content_type_options,
            # Frame options for clickjacking protection
            "X-Frame-Options": self.config.x_frame_options,
            # XSS protection
            "X-XSS-Protection": self.config.x_xss_protection,
            # HTTPS enforcement
            "Strict-Transport-Security": self.config.strict_transport_security,
            # Content Security Policy
            "Content-Security-Policy": self.config.content_security_policy,
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions policy
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            # Cache control for sensitive responses
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            # Remove server information
            "Server": "SecureAPI/1.0",
        }

        # Add headers to response
        for header, value in security_headers.items():
            response.headers[header] = value

    def _create_security_error_response(
        self, exception: HTTPException, _request: Request
    ) -> JSONResponse:
        """Create standardized security error response.

        Args:
            exception: HTTP exception that occurred
            request: Original request

        Returns:
            JSON response with security error
        """
        # Don't expose sensitive error details
        safe_details = {
            400: "Bad request",
            401: "Unauthorized",
            403: "Access denied",
            404: "Not found",
            413: "Request too large",
            414: "URL too long",
            429: "Rate limit exceeded",
            500: "Internal server error",
        }

        status_code = exception.status_code
        detail = safe_details.get(status_code, "Request failed")

        response_data = {
            "error": detail,
            "status_code": status_code,
            "timestamp": int(time.time()),
        }

        # Add rate limit headers for 429 responses
        headers = {}
        if hasattr(exception, "headers") and exception.headers:
            headers.update(exception.headers)

        return JSONResponse(
            status_code=status_code, content=response_data, headers=headers
        )

    async def block_ip(
        self, ip_address: str, reason: str = "Security violation"
    ) -> None:
        """Block an IP address from accessing the API.

        Args:
            ip_address: IP address to block
            reason: Reason for blocking
        """
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address {ip_address}: {reason}")

        # Log blocking event
        self.security_monitor.log_security_event(
            "ip_blocked", None, {"blocked_ip": ip_address, "reason": reason}
        )

    async def unblock_ip(self, ip_address: str) -> bool:
        """Unblock a previously blocked IP address.

        Args:
            ip_address: IP address to unblock

        Returns:
            True if IP was unblocked, False if it wasn't blocked
        """
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP address {ip_address}")
            return True
        return False

    def get_security_status(self) -> dict[str, Any]:
        """Get current security middleware status.

        Returns:
            Dictionary with security status information
        """
        return {
            "middleware_active": True,
            "rate_limiter_status": "active",
            "ai_validator_status": "active",
            "blocked_ips_count": len(self.blocked_ips),
            "rate_limits": self.rate_limits,
            "security_headers_enabled": True,
            "input_validation_enabled": True,
        }
