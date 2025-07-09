"""Enhanced Security Middleware with comprehensive protection layers.

This module implements enterprise-grade security middleware with:
- Multi-layer authentication (JWT, API keys, sessions)
- Rate limiting and DDoS protection
- Input validation and sanitization
- Security headers enforcement
- Request filtering and threat detection
- Audit logging for all security events
- OWASP AI Top 10 compliance measures

Following zero-trust architecture principles with defense in depth.
"""

import logging
import re
import time
from datetime import UTC, datetime, timezone
from ipaddress import ip_address, ip_network
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...cache.intelligent import IntelligentCache
from ...errors import ServiceError, ValidationError
from ..audit.logger import SecurityAuditLogger
from ..auth.api_keys import APIKeyManager, APIKeyValidationRequest
from ..auth.jwt_auth import JWTManager, JWTValidationRequest
from ..auth.rbac import Permission, RBACManager, Resource
from ..auth.sessions import SessionManager, SessionValidationRequest


logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Security middleware configuration."""

    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_burst: int = Field(default=200, description="Burst capacity")
    rate_limit_window: int = Field(default=60, description="Time window in seconds")

    # DDoS protection
    ddos_threshold: int = Field(
        default=1000, description="Requests per minute threshold"
    )
    ddos_ban_duration: int = Field(default=300, description="Ban duration in seconds")

    # Input validation
    max_request_size: int = Field(
        default=10 * 1024 * 1024, description="Max request size in bytes"
    )
    max_url_length: int = Field(default=2048, description="Max URL length")
    max_header_length: int = Field(default=8192, description="Max header length")

    # Security headers
    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS origins"
    )

    # Threat detection
    enable_threat_detection: bool = Field(
        default=True, description="Enable threat detection"
    )
    sql_injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b|\bDROP\b)",
            r"(\bOR\b|\bAND\b).*[=<>].*(\bOR\b|\bAND\b)",
            r"['\"].*['\"]",
            r"--.*",
            r"/\*.*\*/",
        ],
        description="SQL injection patterns",
    )

    # XSS protection
    xss_patterns: list[str] = Field(
        default_factory=lambda: [
            r"<script[^>]*>.*</script>",
            r"javascript:",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"onclick=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ],
        description="XSS patterns",
    )

    # Path traversal protection
    path_traversal_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"\.\.%2f",
            r"\.\.%5c",
        ],
        description="Path traversal patterns",
    )


class ThreatDetectionResult(BaseModel):
    """Threat detection result."""

    is_threat: bool = Field(..., description="Whether threat was detected")
    threat_type: str = Field(..., description="Type of threat")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    details: str = Field(..., description="Threat details")
    blocked: bool = Field(..., description="Whether request was blocked")


class SecurityEvent(BaseModel):
    """Security event for audit logging."""

    event_type: str = Field(..., description="Event type")
    severity: str = Field(..., description="Event severity")
    source_ip: str = Field(..., description="Source IP address")
    user_agent: str = Field(..., description="User agent")
    request_path: str = Field(..., description="Request path")
    request_method: str = Field(..., description="Request method")
    threat_details: ThreatDetectionResult | None = None
    user_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = Field(default_factory=dict)


class RateLimitInfo(BaseModel):
    """Rate limiting information."""

    ip_address: str = Field(..., description="Client IP address")
    requests_count: int = Field(..., description="Current request count")
    window_start: float = Field(..., description="Window start time")
    is_limited: bool = Field(..., description="Whether rate limited")
    reset_time: float = Field(..., description="Reset time")


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with comprehensive protection."""

    def __init__(
        self,
        app,
        jwt_manager: JWTManager,
        rbac_manager: RBACManager,
        api_key_manager: APIKeyManager,
        session_manager: SessionManager,
        audit_logger: SecurityAuditLogger,
        cache: IntelligentCache,
        config: SecurityConfig | None = None,
    ):
        """Initialize security middleware.

        Args:
            app: FastAPI application
            jwt_manager: JWT authentication manager
            rbac_manager: RBAC permission manager
            api_key_manager: API key manager
            session_manager: Session manager
            audit_logger: Security audit logger
            cache: Intelligent cache for rate limiting
            config: Security configuration
        """
        super().__init__(app)
        self.jwt_manager = jwt_manager
        self.rbac_manager = rbac_manager
        self.api_key_manager = api_key_manager
        self.session_manager = session_manager
        self.audit_logger = audit_logger
        self.cache = cache
        self.config = config or SecurityConfig()

        # Compile regex patterns for performance
        self._sql_injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.sql_injection_patterns
        ]
        self._xss_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.xss_patterns
        ]
        self._path_traversal_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.path_traversal_patterns
        ]

        # Initialize rate limiting storage
        self._rate_limits: dict[str, RateLimitInfo] = {}
        self._ddos_blocked_ips: dict[str, float] = {}

        # Security headers
        self._security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }

        # Whitelisted paths that bypass some security checks
        self._whitelisted_paths = {"/health", "/docs", "/openapi.json", "/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through security layers."""
        start_time = time.time()

        try:
            # 1. Pre-flight security checks
            await self._validate_request_basics(request)

            # 2. DDoS protection
            if await self._check_ddos_protection(request):
                return self._create_error_response(
                    "Too many requests", status.HTTP_429_TOO_MANY_REQUESTS
                )

            # 3. Rate limiting
            rate_limit_result = await self._check_rate_limit(request)
            if rate_limit_result.is_limited:
                return self._create_rate_limit_response(rate_limit_result)

            # 4. Threat detection
            threat_result = await self._detect_threats(request)
            if threat_result.is_threat and threat_result.blocked:
                await self._log_security_event(
                    SecurityEvent(
                        event_type="threat_blocked",
                        severity="high",
                        source_ip=self._get_client_ip(request),
                        user_agent=request.headers.get("User-Agent", ""),
                        request_path=str(request.url.path),
                        request_method=request.method,
                        threat_details=threat_result,
                    )
                )
                return self._create_error_response(
                    "Request blocked by security policy", status.HTTP_403_FORBIDDEN
                )

            # 5. Authentication (skip for whitelisted paths)
            if request.url.path not in self._whitelisted_paths:
                auth_result = await self._authenticate_request(request)
                if not auth_result:
                    return self._create_error_response(
                        "Authentication required", status.HTTP_401_UNAUTHORIZED
                    )

            # 6. Process request
            response = await call_next(request)

            # 7. Post-processing
            await self._add_security_headers(response)
            await self._log_request_completion(request, response, start_time)

            return response

        except HTTPException as e:
            await self._log_security_event(
                SecurityEvent(
                    event_type="http_exception",
                    severity="medium",
                    source_ip=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent", ""),
                    request_path=str(request.url.path),
                    request_method=request.method,
                    context={"status_code": e.status_code, "detail": str(e.detail)},
                )
            )
            raise
        except Exception as e:
            await self._log_security_event(
                SecurityEvent(
                    event_type="internal_error",
                    severity="high",
                    source_ip=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent", ""),
                    request_path=str(request.url.path),
                    request_method=request.method,
                    context={"error": str(e)},
                )
            )
            logger.error(f"Security middleware error: {e}", exc_info=True)
            return self._create_error_response(
                "Internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def _validate_request_basics(self, request: Request) -> None:
        """Validate basic request properties."""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large",
            )

        # Check URL length
        if len(str(request.url)) > self.config.max_url_length:
            raise HTTPException(
                status_code=status.HTTP_414_REQUEST_URI_TOO_LONG, detail="URL too long"
            )

        # Check header length
        total_header_length = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_length > self.config.max_header_length:
            raise HTTPException(
                status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
                detail="Headers too large",
            )

    async def _check_ddos_protection(self, request: Request) -> bool:
        """Check DDoS protection."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Check if IP is currently banned
        if client_ip in self._ddos_blocked_ips:
            ban_end_time = self._ddos_blocked_ips[client_ip]
            if current_time < ban_end_time:
                return True  # Still banned
            # Ban expired, remove from blocked list
            del self._ddos_blocked_ips[client_ip]

        # Check request rate for DDoS detection
        cache_key = f"ddos_protection:{client_ip}"
        try:
            request_count = await self.cache.get(cache_key) or 0
            request_count += 1

            if request_count > self.config.ddos_threshold:
                # Block IP
                self._ddos_blocked_ips[client_ip] = (
                    current_time + self.config.ddos_ban_duration
                )
                await self.cache.delete(cache_key)

                await self._log_security_event(
                    SecurityEvent(
                        event_type="ddos_blocked",
                        severity="critical",
                        source_ip=client_ip,
                        user_agent=request.headers.get("User-Agent", ""),
                        request_path=str(request.url.path),
                        request_method=request.method,
                        context={"request_count": request_count},
                    )
                )
                return True

            await self.cache.set(cache_key, request_count, ttl=60)
            return False

        except Exception as e:
            logger.exception(f"DDoS protection error: {e}")
            return False

    async def _check_rate_limit(self, request: Request) -> RateLimitInfo:
        """Check rate limiting."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Get or create rate limit info
        if client_ip not in self._rate_limits:
            self._rate_limits[client_ip] = RateLimitInfo(
                ip_address=client_ip,
                requests_count=0,
                window_start=current_time,
                is_limited=False,
                reset_time=current_time + self.config.rate_limit_window,
            )

        rate_limit = self._rate_limits[client_ip]

        # Check if window has expired
        if current_time >= rate_limit.reset_time:
            rate_limit.requests_count = 0
            rate_limit.window_start = current_time
            rate_limit.reset_time = current_time + self.config.rate_limit_window
            rate_limit.is_limited = False

        # Increment request count
        rate_limit.requests_count += 1

        # Check if limit exceeded
        if rate_limit.requests_count > self.config.rate_limit_requests:
            rate_limit.is_limited = True

            await self._log_security_event(
                SecurityEvent(
                    event_type="rate_limit_exceeded",
                    severity="medium",
                    source_ip=client_ip,
                    user_agent=request.headers.get("User-Agent", ""),
                    request_path=str(request.url.path),
                    request_method=request.method,
                    context={"request_count": rate_limit.requests_count},
                )
            )

        return rate_limit

    async def _detect_threats(self, request: Request) -> ThreatDetectionResult:
        """Detect security threats in request."""
        if not self.config.enable_threat_detection:
            return ThreatDetectionResult(
                is_threat=False,
                threat_type="none",
                confidence=0.0,
                details="Threat detection disabled",
                blocked=False,
            )

        # Check URL for threats
        url_str = str(request.url)
        decoded_url = unquote(url_str)

        # SQL injection detection
        for pattern in self._sql_injection_patterns:
            if pattern.search(decoded_url):
                return ThreatDetectionResult(
                    is_threat=True,
                    threat_type="sql_injection",
                    confidence=0.9,
                    details=f"SQL injection pattern detected in URL: {pattern.pattern}",
                    blocked=True,
                )

        # XSS detection
        for pattern in self._xss_patterns:
            if pattern.search(decoded_url):
                return ThreatDetectionResult(
                    is_threat=True,
                    threat_type="xss",
                    confidence=0.8,
                    details=f"XSS pattern detected in URL: {pattern.pattern}",
                    blocked=True,
                )

        # Path traversal detection
        for pattern in self._path_traversal_patterns:
            if pattern.search(decoded_url):
                return ThreatDetectionResult(
                    is_threat=True,
                    threat_type="path_traversal",
                    confidence=0.9,
                    details=f"Path traversal pattern detected: {pattern.pattern}",
                    blocked=True,
                )

        # Check request headers for threats
        user_agent = request.headers.get("User-Agent", "")
        if self._is_suspicious_user_agent(user_agent):
            return ThreatDetectionResult(
                is_threat=True,
                threat_type="suspicious_user_agent",
                confidence=0.7,
                details=f"Suspicious user agent: {user_agent}",
                blocked=False,  # Just log, don't block
            )

        return ThreatDetectionResult(
            is_threat=False,
            threat_type="none",
            confidence=0.0,
            details="No threats detected",
            blocked=False,
        )

    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious."""
        suspicious_patterns = [
            r"sqlmap",
            r"nikto",
            r"nmap",
            r"masscan",
            r"zap",
            r"burp",
            r"python-requests",
            r"curl",
            r"wget",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return True

        return False

    async def _authenticate_request(self, request: Request) -> bool:
        """Authenticate request using multiple methods."""
        # Try session authentication first
        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = request.headers.get("X-Session-ID")

        if session_id:
            try:
                validation_request = SessionValidationRequest(
                    session_id=session_id,
                    ip_address=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent", ""),
                )

                result = await self.session_manager.validate_session(validation_request)
                if result.valid:
                    # Set session context
                    request.state.auth_type = "session"
                    request.state.session_id = result.session_id
                    request.state.user_id = result.user_id
                    request.state.username = result.username
                    request.state.user_role = result.role
                    request.state.user_permissions = result.permissions
                    return True
            except Exception as e:
                logger.debug(f"Session authentication failed: {e}")

        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if api_key:
            try:
                validation_request = APIKeyValidationRequest(
                    key=api_key,
                    resource=Resource.DOCUMENTS,  # Default resource
                    action=Permission.DOCUMENTS_READ,  # Default action
                    ip_address=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent"),
                    referrer=request.headers.get("Referer"),
                )

                result = await self.api_key_manager.validate_api_key(validation_request)
                if result.valid:
                    # Set API key context
                    request.state.auth_type = "api_key"
                    request.state.api_key_id = result.key_id
                    request.state.api_permissions = result.permissions
                    return True
            except Exception as e:
                logger.debug(f"API key authentication failed: {e}")

        # Try JWT authentication
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                validation_request = JWTValidationRequest(token=token)
                result = await self.jwt_manager.validate_token(validation_request)

                if result.valid:
                    # Set JWT context
                    request.state.auth_type = "jwt"
                    request.state.jwt_token = token
                    request.state.user_id = result.user_id
                    request.state.username = result.username
                    request.state.user_role = result.role
                    request.state.user_permissions = result.permissions
                    return True
            except Exception as e:
                logger.debug(f"JWT authentication failed: {e}")

        return False

    async def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        if self.config.enable_security_headers:
            for header, value in self._security_headers.items():
                response.headers[header] = value

        # Add CORS headers if enabled
        if self.config.enable_cors:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-API-Key, X-Session-ID"
            )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _create_error_response(self, message: str, status_code: int) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "status_code": status_code,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def _create_rate_limit_response(self, rate_limit: RateLimitInfo) -> JSONResponse:
        """Create rate limit response with headers."""
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "status_code": 429,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.config.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.config.rate_limit_requests - rate_limit.requests_count)
        )
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit.reset_time))
        response.headers["Retry-After"] = str(int(rate_limit.reset_time - time.time()))

        return response

    async def _log_security_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        try:
            self.audit_logger.log_security_event(
                event_type=event.event_type,
                user_id=event.user_id or "anonymous",
                resource="security_middleware",
                action=event.event_type,
                resource_id=event.request_path,
                context={
                    "severity": event.severity,
                    "source_ip": event.source_ip,
                    "user_agent": event.user_agent,
                    "request_method": event.request_method,
                    "threat_details": event.threat_details.dict()
                    if event.threat_details
                    else None,
                    **event.context,
                },
            )
        except Exception as e:
            logger.exception(f"Failed to log security event: {e}")

    async def _log_request_completion(
        self, request: Request, response: Response, start_time: float
    ) -> None:
        """Log request completion."""
        duration = time.time() - start_time

        # Log successful requests at debug level
        logger.debug(
            f"Request completed: {request.method} {request.url.path} "
            f"[{response.status_code}] in {duration:.3f}s"
        )

        # Log slow requests
        if duration > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.3f}s"
            )

    def get_security_stats(self) -> dict[str, Any]:
        """Get security middleware statistics."""
        current_time = time.time()

        # Count active rate limits
        active_rate_limits = sum(
            1
            for rl in self._rate_limits.values()
            if rl.is_limited and current_time < rl.reset_time
        )

        # Count DDoS blocked IPs
        active_ddos_blocks = sum(
            1 for ban_time in self._ddos_blocked_ips.values() if current_time < ban_time
        )

        return {
            "total_rate_limits": len(self._rate_limits),
            "active_rate_limits": active_rate_limits,
            "ddos_blocked_ips": active_ddos_blocks,
            "threat_detection_enabled": self.config.enable_threat_detection,
            "security_headers_enabled": self.config.enable_security_headers,
            "cors_enabled": self.config.enable_cors,
            "rate_limit_config": {
                "requests_per_minute": self.config.rate_limit_requests,
                "window_seconds": self.config.rate_limit_window,
            },
        }

    async def cleanup_expired_limits(self) -> int:
        """Clean up expired rate limits."""
        current_time = time.time()
        expired_count = 0

        # Clean up rate limits
        expired_ips = [
            ip
            for ip, rl in self._rate_limits.items()
            if current_time >= rl.reset_time and not rl.is_limited
        ]

        for ip in expired_ips:
            del self._rate_limits[ip]
            expired_count += 1

        # Clean up DDoS blocks
        expired_blocks = [
            ip
            for ip, ban_time in self._ddos_blocked_ips.items()
            if current_time >= ban_time
        ]

        for ip in expired_blocks:
            del self._ddos_blocked_ips[ip]
            expired_count += 1

        return expired_count
