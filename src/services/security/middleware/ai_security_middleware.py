"""AI Security Middleware for enterprise-grade API protection.

This middleware integrates the AI Security Service with the existing security
infrastructure to provide comprehensive AI-specific threat protection:
- Pre-request input validation for prompt injection
- Post-response output validation for information leakage
- Rate limiting for model theft protection
- PII detection and masking
- Real-time threat monitoring and alerting

Integrates with existing JWT authentication, RBAC, and audit logging.
"""

import logging
import time
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ....models.api_contracts import ErrorResponse
from ...errors import ServiceError, ValidationError
from ..ai_security.ai_security_service import (
    AISecurityConfig,
    AISecurityService,
    AIThreatType,
    SecurityLevel,
    ValidationResult,
)
from ..audit.logger import SecurityAuditLogger
from ..auth.rbac import Permission, RBACManager, Resource
from ..pii.pii_detector import MaskingStrategy, PIIDetector


logger = logging.getLogger(__name__)


class AISecurityMiddleware(BaseHTTPMiddleware):
    """AI Security Middleware for comprehensive threat protection."""

    def __init__(
        self,
        app: ASGIApp,
        ai_security_service: AISecurityService | None = None,
        pii_detector: PIIDetector | None = None,
        audit_logger: SecurityAuditLogger | None = None,
        rbac_manager: RBACManager | None = None,
        enabled: bool = True,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        protected_endpoints: list[str] | None = None,
    ):
        """Initialize AI Security Middleware.

        Args:
            app: ASGI application
            ai_security_service: AI security service instance
            pii_detector: PII detection service
            audit_logger: Security audit logger
            rbac_manager: RBAC manager for permissions
            enabled: Enable/disable middleware
            security_level: Security level for AI operations
            protected_endpoints: List of endpoint patterns to protect
        """
        super().__init__(app)

        # Initialize services
        self.ai_security = ai_security_service or AISecurityService()
        self.pii_detector = pii_detector
        self.audit_logger = audit_logger
        self.rbac_manager = rbac_manager

        # Configuration
        self.enabled = enabled
        self.security_level = security_level
        self.protected_endpoints = protected_endpoints or [
            "/search",
            "/documents",
            "/collections",
            "/chat",
            "/ai/",
            "/api/",
        ]

        # Performance metrics
        self.request_count = 0
        self.blocked_requests = 0
        self.average_processing_time = 0.0

        # Threat detection patterns
        self._ai_endpoints = {
            "/search": {
                "input_fields": ["query"],
                "response_fields": ["content", "title"],
            },
            "/documents": {
                "input_fields": ["content", "title"],
                "response_fields": ["content"],
            },
            "/chat": {
                "input_fields": ["message", "query"],
                "response_fields": ["response", "message"],
            },
            "/ai/generate": {
                "input_fields": ["prompt", "input"],
                "response_fields": ["output", "response"],
            },
            "/api/search": {
                "input_fields": ["query", "q"],
                "response_fields": ["results", "content"],
            },
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through AI security middleware."""
        if not self.enabled:
            return await call_next(request)

        start_time = time.time()

        try:
            # Check if endpoint should be protected
            if not self._should_protect_endpoint(request.url.path):
                return await call_next(request)

            # Extract user context
            user_context = await self._extract_user_context(request)

            # Pre-request validation
            await self._validate_request_input(request, user_context)

            # Process request
            response = await call_next(request)

            # Post-response validation
            validated_response = await self._validate_response_output(
                response, request, user_context
            )

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, blocked=False)

            return validated_response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Handle security violations
            logger.exception(f"AI security middleware error: {e}")

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, blocked=True)

            # Log security event
            if self.audit_logger:
                await self._log_security_violation(request, str(e), user_context)

            # Return security error response
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=ErrorResponse(
                    success=False,
                    error="AI security violation detected",
                    error_type="ai_security_violation",
                    timestamp=time.time(),
                    context={"details": "Request blocked for security reasons"},
                ).dict(),
            )

    def _should_protect_endpoint(self, path: str) -> bool:
        """Check if endpoint should be protected."""
        return any(pattern in path for pattern in self.protected_endpoints)

    async def _extract_user_context(self, request: Request) -> dict[str, Any]:
        """Extract user context from request."""
        user_context = {
            "user_id": None,
            "username": None,
            "role": None,
            "permissions": [],
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "endpoint": request.url.path,
            "method": request.method,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Extract from JWT token if available
        if hasattr(request.state, "user"):
            user = request.state.user
            user_context.update(
                {
                    "user_id": getattr(user, "id", None),
                    "username": getattr(user, "username", None),
                    "role": getattr(user, "role", None),
                    "permissions": getattr(user, "permissions", []),
                }
            )

        return user_context

    async def _validate_request_input(
        self, request: Request, user_context: dict[str, Any]
    ) -> None:
        """Validate request input for AI security threats."""
        # Get request body
        body = await self._get_request_body(request)
        if not body:
            return

        # Extract input fields based on endpoint
        input_texts = self._extract_input_texts(request.url.path, body)

        # Validate each input text
        for field_name, text in input_texts.items():
            if not text or not isinstance(text, str):
                continue

            # Validate with AI security service
            validation_result = await self.ai_security.validate_input(
                user_input=text,
                user_id=user_context.get("user_id"),
                context={
                    "field_name": field_name,
                    "endpoint": request.url.path,
                    "user_context": user_context,
                },
            )

            # Handle validation result
            if not validation_result.is_safe:
                await self._handle_security_violation(
                    request, validation_result, user_context, "input"
                )

    async def _validate_response_output(
        self, response: Response, request: Request, user_context: dict[str, Any]
    ) -> Response:
        """Validate response output for AI security threats."""
        # Skip validation for non-JSON responses
        if not response.headers.get("content-type", "").startswith("application/json"):
            return response

        # Get response body
        response_body = await self._get_response_body(response)
        if not response_body:
            return response

        # Extract output fields based on endpoint
        output_texts = self._extract_output_texts(request.url.path, response_body)

        # Validate each output text
        sanitized_outputs = {}
        for field_name, text in output_texts.items():
            if not text or not isinstance(text, str):
                continue

            # Validate with AI security service
            validation_result = await self.ai_security.validate_output(
                ai_output=text,
                user_id=user_context.get("user_id"),
                context={
                    "field_name": field_name,
                    "endpoint": request.url.path,
                    "user_context": user_context,
                },
            )

            # Handle validation result
            if not validation_result.is_safe:
                if (
                    validation_result.threat_type
                    == AIThreatType.SENSITIVE_INFO_DISCLOSURE
                ):
                    # Use sanitized content for PII
                    sanitized_outputs[field_name] = validation_result.sanitized_content
                else:
                    # Block unsafe output
                    await self._handle_security_violation(
                        request, validation_result, user_context, "output"
                    )
            elif validation_result.sanitized_content:
                # Use sanitized content if available
                sanitized_outputs[field_name] = validation_result.sanitized_content

        # Apply sanitization to response if needed
        if sanitized_outputs:
            response = await self._sanitize_response(response, sanitized_outputs)

        return response

    async def _get_request_body(self, request: Request) -> dict[str, Any] | None:
        """Get request body as dictionary."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                # Store body for later use
                if not hasattr(request.state, "body"):
                    body = await request.body()
                    request.state.body = body
                else:
                    body = request.state.body

                if body:
                    import json

                    return json.loads(body.decode())
        except Exception as e:
            logger.warning(f"Failed to parse request body: {e}")

        return None

    async def _get_response_body(self, response: Response) -> dict[str, Any] | None:
        """Get response body as dictionary."""
        try:
            if hasattr(response, "body"):
                import json

                return json.loads(response.body.decode())
        except Exception as e:
            logger.warning(f"Failed to parse response body: {e}")

        return None

    def _extract_input_texts(
        self, endpoint: str, body: dict[str, Any]
    ) -> dict[str, str]:
        """Extract input texts from request body."""
        input_texts = {}

        # Get field configuration for endpoint
        endpoint_config = None
        for pattern, config in self._ai_endpoints.items():
            if pattern in endpoint:
                endpoint_config = config
                break

        if not endpoint_config:
            # Default extraction for unknown endpoints
            for key, value in body.items():
                if isinstance(value, str) and len(value) > 10:
                    input_texts[key] = value
        else:
            # Extract configured input fields
            for field in endpoint_config.get("input_fields", []):
                if field in body and isinstance(body[field], str):
                    input_texts[field] = body[field]

        return input_texts

    def _extract_output_texts(
        self, endpoint: str, body: dict[str, Any]
    ) -> dict[str, str]:
        """Extract output texts from response body."""
        output_texts = {}

        # Get field configuration for endpoint
        endpoint_config = None
        for pattern, config in self._ai_endpoints.items():
            if pattern in endpoint:
                endpoint_config = config
                break

        if not endpoint_config:
            # Default extraction for unknown endpoints
            for key, value in body.items():
                if isinstance(value, str) and len(value) > 10:
                    output_texts[key] = value
                elif isinstance(value, list):
                    # Handle arrays of results
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                if isinstance(sub_value, str) and len(sub_value) > 10:
                                    output_texts[f"{key}[{i}].{sub_key}"] = sub_value
        else:
            # Extract configured output fields
            for field in endpoint_config.get("response_fields", []):
                if field in body and isinstance(body[field], str):
                    output_texts[field] = body[field]
                elif field in body and isinstance(body[field], list):
                    # Handle arrays
                    for i, item in enumerate(body[field]):
                        if isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                if isinstance(sub_value, str) and len(sub_value) > 10:
                                    output_texts[f"{field}[{i}].{sub_key}"] = sub_value

        return output_texts

    async def _handle_security_violation(
        self,
        request: Request,
        validation_result: ValidationResult,
        user_context: dict[str, Any],
        violation_type: str,
    ) -> None:
        """Handle security violation."""
        # Log security violation
        logger.warning(
            f"AI security violation detected: {validation_result.threat_type} "
            f"in {violation_type} for {request.url.path}"
        )

        # Audit log
        if self.audit_logger:
            await self._log_security_violation(
                request,
                f"{validation_result.threat_type}: {validation_result.details}",
                user_context,
            )

        # Raise HTTP exception
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "AI security violation detected",
                "threat_type": validation_result.threat_type,
                "details": validation_result.details,
                "mitigation_actions": validation_result.mitigation_actions,
            },
        )

    async def _sanitize_response(
        self, response: Response, sanitized_outputs: dict[str, str]
    ) -> Response:
        """Sanitize response with cleaned content."""
        try:
            # Get response body
            response_body = await self._get_response_body(response)
            if not response_body:
                return response

            # Apply sanitization
            for field_path, sanitized_content in sanitized_outputs.items():
                # Handle nested field paths like "results[0].content"
                self._set_nested_field(response_body, field_path, sanitized_content)

            # Create new response with sanitized content
            import json

            return Response(
                content=json.dumps(response_body),
                status_code=response.status_code,
                headers=response.headers,
                media_type="application/json",
            )

        except Exception as e:
            logger.exception(f"Failed to sanitize response: {e}")
            return response

    def _set_nested_field(
        self, data: dict[str, Any], field_path: str, value: str
    ) -> None:
        """Set nested field value using dot notation."""
        try:
            parts = field_path.split(".")
            current = data

            for part in parts[:-1]:
                # Handle array notation like "results[0]"
                if "[" in part and "]" in part:
                    field_name = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = current[field_name][index]
                else:
                    current = current[part]

            # Set final value
            final_part = parts[-1]
            if "[" in final_part and "]" in final_part:
                field_name = final_part.split("[")[0]
                index = int(final_part.split("[")[1].split("]")[0])
                current[field_name][index] = value
            else:
                current[final_part] = value

        except Exception as e:
            logger.warning(f"Failed to set nested field {field_path}: {e}")

    async def _log_security_violation(
        self, request: Request, violation_details: str, user_context: dict[str, Any]
    ) -> None:
        """Log security violation to audit system."""
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="ai_security_violation",
                user_id=user_context.get("user_id", "anonymous"),
                resource="ai_security_middleware",
                action="request_validation",
                resource_id=request.url.path,
                context={
                    "violation_details": violation_details,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "user_agent": user_context.get("user_agent"),
                    "ip_address": user_context.get("ip_address"),
                    "username": user_context.get("username"),
                    "role": user_context.get("role"),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    def _update_metrics(self, processing_time: float, blocked: bool) -> None:
        """Update performance metrics."""
        self.request_count += 1
        if blocked:
            self.blocked_requests += 1

        # Update average processing time
        self.average_processing_time = (
            self.average_processing_time * (self.request_count - 1) + processing_time
        ) / self.request_count

    def get_security_metrics(self) -> dict[str, Any]:
        """Get AI security middleware metrics."""
        return {
            "total_requests": self.request_count,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(1, self.request_count),
            "average_processing_time_ms": self.average_processing_time,
            "ai_security_metrics": self.ai_security.get_security_metrics(),
            "protected_endpoints": self.protected_endpoints,
            "security_level": self.security_level.value,
            "middleware_enabled": self.enabled,
        }

    def update_security_config(self, config: dict[str, Any]) -> None:
        """Update security configuration."""
        if "enabled" in config:
            self.enabled = config["enabled"]

        if "security_level" in config:
            self.security_level = SecurityLevel(config["security_level"])

        if "protected_endpoints" in config:
            self.protected_endpoints = config["protected_endpoints"]

        logger.info("AI security middleware configuration updated")

    def reset_metrics(self) -> None:
        """Reset security metrics."""
        self.request_count = 0
        self.blocked_requests = 0
        self.average_processing_time = 0.0
        self.ai_security.reset_metrics()
        logger.info("AI security middleware metrics reset")
