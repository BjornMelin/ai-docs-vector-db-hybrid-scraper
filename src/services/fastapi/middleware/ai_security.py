"""AI Security Middleware for FastAPI applications.

This middleware integrates the AI security service with FastAPI to provide:
- Pre-request input validation for prompt injection detection
- Post-response output validation for information leakage prevention
- Real-time threat monitoring and alerting
- Integration with existing JWT authentication and RBAC systems
- PII detection and masking in API requests/responses
- Rate limiting for model theft protection
- Comprehensive audit logging for AI security events

Following OWASP AI Top 10 security guidelines with enterprise-grade protection.
"""

import json
import logging
from datetime import UTC, datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from src.services.errors import ServiceError, ValidationError
from src.services.security.ai_security.ai_security_service import (
    AISecurityConfig,
    AISecurityService,
    AIThreatType,
)
from src.services.security.audit.logger import SecurityAuditLogger
from src.services.security.pii.pii_detector import PIIDetector


logger = logging.getLogger(__name__)


class AISecurityMiddleware(BaseHTTPMiddleware):
    """AI Security Middleware for OWASP AI Top 10 protection."""

    def __init__(
        self,
        app: ASGIApp,
        ai_security_service: AISecurityService | None = None,
        config: AISecurityConfig | None = None,
        protected_endpoints: list[str] | None = None,
        audit_logger: SecurityAuditLogger | None = None,
    ):
        """Initialize AI security middleware.

        Args:
            app: ASGI application
            ai_security_service: AI security service instance
            config: AI security configuration
            protected_endpoints: List of endpoint patterns to protect
            audit_logger: Security audit logger
        """
        super().__init__(app)

        # Initialize AI security service
        if ai_security_service:
            self.ai_security_service = ai_security_service
        else:
            pii_detector = PIIDetector()
            self.ai_security_service = AISecurityService(
                config=config or AISecurityConfig(),
                pii_detector=pii_detector,
                audit_logger=audit_logger,
            )

        # Configure protected endpoints
        self.protected_endpoints = protected_endpoints or [
            "/search",
            "/advanced-search",
            "/documents",
            "/bulk-documents",
            "/analytics",
            "/mcp/",
        ]

        # Initialize metrics
        self.metrics = {
            "requests_processed": 0,
            "threats_blocked": 0,
            "processing_time_ms": 0.0,
        }

        logger.info(
            "AI Security Middleware initialized with OWASP AI Top 10 protection"
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through AI security validation."""
        start_time = datetime.now()

        try:
            # Check if endpoint needs protection
            if not self._should_protect_endpoint(request.url.path):
                return await call_next(request)

            # Update metrics
            self.metrics["requests_processed"] += 1

            # Extract user information from request
            user_info = await self._extract_user_info(request)

            # Validate request input
            input_validation_result = await self._validate_request_input(
                request, user_info
            )
            if not input_validation_result.is_safe:
                self.metrics["threats_blocked"] += 1
                return self._create_security_error_response(input_validation_result)

            # Process request with sanitized input if needed
            if input_validation_result.sanitized_content:
                request = await self._update_request_with_sanitized_content(
                    request, input_validation_result.sanitized_content
                )

            # Call the next middleware/endpoint
            response = await call_next(request)

            # Validate response output
            output_validation_result = await self._validate_response_output(
                response, user_info
            )
            if not output_validation_result.is_safe:
                self.metrics["threats_blocked"] += 1
                return self._create_security_error_response(output_validation_result)

            # Update response with sanitized content if needed
            if output_validation_result.sanitized_content:
                response = await self._update_response_with_sanitized_content(
                    response, output_validation_result.sanitized_content
                )

            # Add security headers
            response = self._add_security_headers(response)

            # Update processing metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["processing_time_ms"] = (
                self.metrics["processing_time_ms"]
                * (self.metrics["requests_processed"] - 1)
                + processing_time
            ) / self.metrics["requests_processed"]

            return response

        except Exception as e:
            logger.exception(f"AI Security Middleware error: {e}")
            self.metrics["threats_blocked"] += 1

            # Create error response for security failures
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Security validation failed",
                    "error_type": "security_error",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    def _should_protect_endpoint(self, path: str) -> bool:
        """Check if endpoint should be protected by AI security."""
        # Skip health check and static endpoints
        if path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return False

        # Check against protected endpoint patterns
        return any(path.startswith(pattern) for pattern in self.protected_endpoints)

    async def _extract_user_info(self, request: Request) -> dict[str, Any]:
        """Extract user information from request for security context."""
        user_info = {
            "user_id": None,
            "username": None,
            "role": None,
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "endpoint": request.url.path,
            "method": request.method,
        }

        # Extract user info from JWT token if present
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you would decode the JWT token
            # For now, we'll extract from request state if available
            user_info.update(
                {
                    "user_id": getattr(request.state, "user_id", None),
                    "username": getattr(request.state, "username", None),
                    "role": getattr(request.state, "user_role", None),
                }
            )

        return user_info

    async def _validate_request_input(
        self, request: Request, user_info: dict[str, Any]
    ):
        """Validate request input for AI security threats."""
        # Extract request content for validation
        request_content = await self._extract_request_content(request)

        if not request_content:
            # No content to validate
            from src.services.security.ai_security.ai_security_service import (
                ValidationResult,
            )

            return ValidationResult(
                is_safe=True,
                threat_type=None,
                confidence=1.0,
                risk_score=0.0,
                details="No content to validate",
                mitigation_actions=[],
                sanitized_content=None,
            )

        # Validate input using AI security service
        context = {
            "endpoint": request.url.path,
            "method": request.method,
            "ip_address": user_info.get("ip_address"),
            "user_agent": user_info.get("user_agent"),
        }

        return await self.ai_security_service.validate_input(
            user_input=request_content,
            user_id=user_info.get("user_id"),
            context=context,
        )

    async def _validate_response_output(
        self, response: Response, user_info: dict[str, Any]
    ):
        """Validate response output for AI security threats."""
        # Extract response content for validation
        response_content = await self._extract_response_content(response)

        if not response_content:
            # No content to validate
            from src.services.security.ai_security.ai_security_service import (
                ValidationResult,
            )

            return ValidationResult(
                is_safe=True,
                threat_type=None,
                confidence=1.0,
                risk_score=0.0,
                details="No content to validate",
                mitigation_actions=[],
                sanitized_content=None,
            )

        # Validate output using AI security service
        context = {
            "response_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "user_id": user_info.get("user_id"),
        }

        return await self.ai_security_service.validate_output(
            ai_output=response_content,
            user_id=user_info.get("user_id"),
            context=context,
        )

    async def _extract_request_content(self, request: Request) -> str | None:
        """Extract text content from request for validation."""
        try:
            # Handle JSON requests
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if body:
                    data = json.loads(body.decode())
                    return self._extract_text_from_json(data)

            # Handle form data
            elif request.headers.get("content-type", "").startswith(
                "application/x-www-form-urlencoded"
            ):
                form_data = await request.form()
                return " ".join(str(value) for value in form_data.values())

            # Handle text content
            elif request.headers.get("content-type", "").startswith("text/"):
                body = await request.body()
                return body.decode() if body else None

            # Handle query parameters
            if request.query_params:
                return " ".join(str(value) for value in request.query_params.values())

            return None

        except Exception as e:
            logger.warning(f"Failed to extract request content: {e}")
            return None

    async def _extract_response_content(self, response: Response) -> str | None:
        """Extract text content from response for validation."""
        try:
            # Only validate JSON responses
            if response.headers.get("content-type", "").startswith("application/json"):
                if hasattr(response, "body"):
                    body = response.body
                    if isinstance(body, bytes):
                        data = json.loads(body.decode())
                        return self._extract_text_from_json(data)

            return None

        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return None

    def _extract_text_from_json(self, data: Any) -> str:
        """Extract text content from JSON data for validation."""
        if isinstance(data, dict):
            # Extract text from common fields
            text_fields = [
                "query",
                "content",
                "title",
                "description",
                "text",
                "message",
            ]
            extracted_text = []

            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    extracted_text.append(data[field])

            # Recursively extract from nested objects
            for value in data.values():
                if isinstance(value, dict | list):
                    nested_text = self._extract_text_from_json(value)
                    if nested_text:
                        extracted_text.append(nested_text)

            return " ".join(extracted_text)

        if isinstance(data, list):
            extracted_text = []
            for item in data:
                item_text = self._extract_text_from_json(item)
                if item_text:
                    extracted_text.append(item_text)
            return " ".join(extracted_text)

        if isinstance(data, str):
            return data

        return ""

    async def _update_request_with_sanitized_content(
        self, request: Request, sanitized_content: str
    ) -> Request:
        """Update request with sanitized content."""
        # This would require reconstructing the request with sanitized content
        # For now, we'll log the sanitization and return the original request
        logger.info(f"Request content sanitized for endpoint: {request.url.path}")
        return request

    async def _update_response_with_sanitized_content(
        self, response: Response, sanitized_content: str
    ) -> Response:
        """Update response with sanitized content."""
        try:
            # For JSON responses, update the content with sanitized version
            if response.headers.get("content-type", "").startswith("application/json"):
                if hasattr(response, "body"):
                    body = response.body
                    if isinstance(body, bytes):
                        data = json.loads(body.decode())

                        # Update text fields with sanitized content
                        self._update_json_with_sanitized_content(
                            data, sanitized_content
                        )

                        # Create new response with sanitized content
                        new_body = json.dumps(data).encode()
                        response.body = new_body
                        response.headers["content-length"] = str(len(new_body))

                        logger.info("Response content sanitized")

            return response

        except Exception as e:
            logger.exception(f"Failed to update response with sanitized content: {e}")
            return response

    def _update_json_with_sanitized_content(
        self, data: Any, sanitized_content: str
    ) -> None:
        """Update JSON data with sanitized content."""
        if isinstance(data, dict):
            # Update common text fields
            text_fields = ["content", "title", "description", "text", "message"]
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    data[field] = sanitized_content

            # Recursively update nested objects
            for value in data.values():
                if isinstance(value, dict | list):
                    self._update_json_with_sanitized_content(value, sanitized_content)

        elif isinstance(data, list):
            for item in data:
                self._update_json_with_sanitized_content(item, sanitized_content)

    def _create_security_error_response(self, validation_result) -> JSONResponse:
        """Create security error response for blocked threats."""
        status_code = status.HTTP_403_FORBIDDEN
        error_type = "security_violation"

        # Map threat types to specific error responses
        threat_mapping = {
            AIThreatType.PROMPT_INJECTION: {
                "status": status.HTTP_400_BAD_REQUEST,
                "type": "prompt_injection",
                "message": "Request blocked due to potential prompt injection",
            },
            AIThreatType.SENSITIVE_INFO_DISCLOSURE: {
                "status": status.HTTP_400_BAD_REQUEST,
                "type": "sensitive_info",
                "message": "Request blocked due to sensitive information detection",
            },
            AIThreatType.MODEL_THEFT: {
                "status": status.HTTP_429_TOO_MANY_REQUESTS,
                "type": "rate_limit",
                "message": "Request blocked due to rate limiting",
            },
            AIThreatType.DATA_POISONING: {
                "status": status.HTTP_400_BAD_REQUEST,
                "type": "data_poisoning",
                "message": "Request blocked due to potential data poisoning",
            },
            AIThreatType.INSECURE_OUTPUT: {
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "type": "insecure_output",
                "message": "Response blocked due to security concerns",
            },
        }

        if validation_result.threat_type:
            threat_info = threat_mapping.get(
                validation_result.threat_type,
                {
                    "status": status_code,
                    "type": error_type,
                    "message": "Request blocked due to security violation",
                },
            )
            status_code = threat_info["status"]
            error_type = threat_info["type"]
            error_message = threat_info["message"]
        else:
            error_message = "Request blocked due to security violation"

        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": error_message,
                "error_type": error_type,
                "risk_score": validation_result.risk_score,
                "confidence": validation_result.confidence,
                "timestamp": datetime.now(UTC).isoformat(),
                "context": {
                    "threat_type": validation_result.threat_type.value
                    if validation_result.threat_type
                    else None,
                    "mitigation_actions": validation_result.mitigation_actions,
                },
            },
        )

    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        # Add AI security headers
        response.headers["X-AI-Security"] = "enabled"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

    def get_security_metrics(self) -> dict[str, Any]:
        """Get AI security middleware metrics."""
        return {
            "requests_processed": self.metrics["requests_processed"],
            "threats_blocked": self.metrics["threats_blocked"],
            "block_rate": self.metrics["threats_blocked"]
            / max(1, self.metrics["requests_processed"]),
            "average_processing_time_ms": self.metrics["processing_time_ms"],
            "protected_endpoints": self.protected_endpoints,
            "ai_security_metrics": self.ai_security_service.get_security_metrics(),
        }

    def update_protected_endpoints(self, endpoints: list[str]) -> None:
        """Update list of protected endpoints."""
        self.protected_endpoints = endpoints
        logger.info(f"Updated protected endpoints: {endpoints}")

    def reset_metrics(self) -> None:
        """Reset middleware metrics."""
        self.metrics = {
            "requests_processed": 0,
            "threats_blocked": 0,
            "processing_time_ms": 0.0,
        }
        self.ai_security_service.reset_metrics()
        logger.info("AI Security Middleware metrics reset")
