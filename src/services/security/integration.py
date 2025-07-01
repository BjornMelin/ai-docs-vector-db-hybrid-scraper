#!/usr/bin/env python3
"""Security integration and setup for FastAPI application.

This module provides comprehensive security integration for the AI documentation
system, bringing together all security components into a cohesive framework:

- Security middleware setup and configuration
- Rate limiting with Redis backend integration
- AI-specific security validation
- Security monitoring and alerting
- Production-ready security configuration
- Health checks and status monitoring
"""

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config.security import SecurityConfig
from src.services.security.ai_security import AISecurityValidator
from src.services.security.middleware import SecurityMiddleware
from src.services.security.monitoring import SecurityMonitor
from src.services.security.rate_limiter import DistributedRateLimiter


logger = logging.getLogger(__name__)


def _raise_security_system_unhealthy() -> None:
    """Raise HTTPException for unhealthy security system."""
    raise HTTPException(status_code=503, detail="Security system unhealthy")


class SecurityManager:
    """Centralized security management for the application.

    This class coordinates all security components and provides a unified
    interface for security operations:

    - Component initialization and configuration
    - Health monitoring and status reporting
    - Security policy enforcement
    - Integration with application lifecycle
    """

    def __init__(self, security_config: SecurityConfig | None = None):
        """Initialize security manager.

        Args:
            security_config: Security configuration settings
        """
        self.config = security_config or SecurityConfig()

        # Initialize security components
        self.rate_limiter: DistributedRateLimiter | None = None
        self.ai_validator: AISecurityValidator | None = None
        self.security_monitor: SecurityMonitor | None = None
        self.middleware: SecurityMiddleware | None = None

        # Security features status
        self.features_enabled = {
            "rate_limiting": True,
            "ai_validation": True,
            "security_monitoring": True,
            "input_validation": True,
            "security_headers": True,
        }

        logger.info("Security manager initialized")

    async def initialize_components(self) -> None:
        """Initialize all security components."""
        try:
            # Initialize rate limiter
            redis_url = os.getenv("REDIS_URL")
            self.rate_limiter = DistributedRateLimiter(
                redis_url=redis_url, security_config=self.config
            )

            # Initialize AI security validator
            self.ai_validator = AISecurityValidator(self.config)

            # Initialize security monitor
            self.security_monitor = SecurityMonitor(self.config)

            logger.info("All security components initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize security components")
            # Set fallback configurations
            self._setup_fallback_security()

    def _setup_fallback_security(self) -> None:
        """Setup fallback security when full initialization fails."""
        logger.warning("Setting up fallback security configuration")

        # Initialize with minimal configuration
        self.rate_limiter = DistributedRateLimiter(security_config=self.config)
        self.ai_validator = AISecurityValidator(self.config)
        self.security_monitor = SecurityMonitor(self.config)

        # Disable features that might not work
        self.features_enabled.update(
            {
                "rate_limiting": False,  # Disable distributed rate limiting
            }
        )

    def setup_middleware(self, app: FastAPI) -> SecurityMiddleware:
        """Setup security middleware for FastAPI application.

        Args:
            app: FastAPI application instance

        Returns:
            Configured security middleware
        """
        if not all([self.rate_limiter, self.ai_validator, self.security_monitor]):
            msg = "Security components not initialized. Call initialize_components()."
            raise RuntimeError(msg)

        # Create security middleware
        self.middleware = SecurityMiddleware(
            app=app,
            rate_limiter=self.rate_limiter,
            security_config=self.config,
            ai_validator=self.ai_validator,
            security_monitor=self.security_monitor,
        )

        # Add middleware to application
        app.add_middleware(
            SecurityMiddleware,
            rate_limiter=self.rate_limiter,
            security_config=self.config,
            ai_validator=self.ai_validator,
            security_monitor=self.security_monitor,
        )

        logger.info("Security middleware configured")
        return self.middleware

    def setup_cors(self, app: FastAPI) -> None:
        """Setup CORS with security-appropriate settings.

        Args:
            app: FastAPI application instance
        """
        # Production-appropriate CORS settings
        allowed_origins = self.config.allowed_origins

        # Default to localhost for development if no origins configured
        if not allowed_origins:
            allowed_origins = ["http://localhost:3000", "http://localhost:8000"]
            logger.warning("No allowed origins configured, using development defaults")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=[
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        )

        logger.info(
            f"CORS configured with origins: {allowed_origins}"
        )  # TODO: Convert f-string to logging format

    async def get_security_status(self) -> dict[str, Any]:
        """Get comprehensive security status.

        Returns:
            Dictionary with security system status
        """
        status = {
            "security_enabled": True,
            "components_initialized": True,
            "features_enabled": self.features_enabled.copy(),
            "timestamp": str(Path().absolute()),  # Placeholder for timestamp
        }

        # Rate limiter status
        if self.rate_limiter:
            try:
                rate_limiter_health = await self.rate_limiter.get_health_status()
                status["rate_limiter"] = rate_limiter_health
            except (AttributeError, RuntimeError, ValueError) as e:
                status["rate_limiter"] = {"error": str(e), "healthy": False}

        # Security monitor metrics
        if self.security_monitor:
            try:
                status["security_metrics"] = (
                    self.security_monitor.get_security_metrics()
                )
            except (AttributeError, RuntimeError, ValueError) as e:
                status["security_metrics"] = {"error": str(e)}

        # Middleware status
        if self.middleware:
            status["middleware"] = self.middleware.get_security_status()

        return status

    async def validate_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Validate API key if API key authentication is enabled.

        Args:
            credentials: HTTP bearer credentials

        Returns:
            True if API key is valid

        Raises:
            HTTPException: If API key is invalid
        """
        if not self.config.api_key_required:
            return True

        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        api_key = credentials.credentials
        valid_keys = self.config.api_keys

        if not valid_keys or api_key not in valid_keys:
            # Log security event
            if self.security_monitor:
                self.security_monitor.log_security_event(
                    "authentication_failure",
                    event_data={"reason": "invalid_api_key"},
                    severity="medium",
                )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return True

    async def cleanup_resources(self) -> None:
        """Cleanup security resources on application shutdown."""
        try:
            if self.rate_limiter:
                await self.rate_limiter.close()

            if self.security_monitor:
                # Cleanup old data
                self.security_monitor.cleanup_old_data(days=30)

            logger.info("Security resources cleaned up")

        except Exception as e:
            logger.exception("Error during security cleanup")


# Global security manager instance
security_manager = SecurityManager()


async def get_security_manager() -> SecurityManager:
    """Dependency to get security manager instance.

    Returns:
        Global security manager instance
    """
    return security_manager


def setup_application_security(
    app: FastAPI, security_config: SecurityConfig | None = None
) -> SecurityManager:
    """Setup comprehensive security for FastAPI application.

    This function provides a complete security setup for production deployment:
    - Initializes all security components
    - Configures middleware and CORS
    - Sets up monitoring and logging
    - Provides health checks and status endpoints

    Args:
        app: FastAPI application instance
        security_config: Optional security configuration

    Returns:
        Configured security manager
    """
    global security_manager

    # Initialize security manager with config
    if security_config:
        security_manager = SecurityManager(security_config)

    @app.on_event("startup")
    async def startup_security():
        """Initialize security components on application startup."""
        await security_manager.initialize_components()
        security_manager.setup_cors(app)
        security_manager.setup_middleware(app)
        logger.info("Application security initialized")

    @app.on_event("shutdown")
    async def shutdown_security():
        """Cleanup security resources on application shutdown."""
        await security_manager.cleanup_resources()
        logger.info("Application security shutdown complete")

    # Add security status endpoint
    @app.get("/security/status")
    async def security_status():
        """Get security system status."""
        return await security_manager.get_security_status()

    @app.get("/security/health")
    async def security_health():
        """Security health check endpoint."""
        try:
            status = await security_manager.get_security_status()

            # Check if critical components are healthy
            is_healthy = (
                status.get("components_initialized", False)
                and status.get("rate_limiter", {}).get(
                    "redis_healthy", True
                )  # True if no Redis requirement
                and status.get("middleware", {}).get("middleware_active", False)
            )

            if is_healthy:
                return {"status": "healthy", "details": status}
            _raise_security_system_unhealthy()

        except Exception as e:
            logger.exception("Security health check failed")
            raise HTTPException(
                status_code=503, detail=f"Security health check failed: {e!s}"
            )

    @app.get("/security/metrics")
    async def security_metrics():
        """Get security metrics and statistics."""
        if not security_manager.security_monitor:
            raise HTTPException(
                status_code=503, detail="Security monitoring not available"
            )

        return security_manager.security_monitor.get_security_metrics()

    @app.get("/security/threats")
    async def threat_report(hours: int = 24):
        """Get threat analysis report."""
        if not security_manager.security_monitor:
            raise HTTPException(
                status_code=503, detail="Security monitoring not available"
            )

        if not 1 <= hours <= 168:  # 1 hour to 1 week
            raise HTTPException(
                status_code=400, detail="Hours must be between 1 and 168"
            )

        return security_manager.security_monitor.get_threat_report(hours=hours)

    # Add API key dependency for protected endpoints
    security_scheme = HTTPBearer(auto_error=False)

    async def validate_api_key_dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    ) -> bool:
        """Dependency for API key validation."""
        return await security_manager.validate_api_key(credentials)

    # Make the dependency available for use in route definitions
    app.dependency_overrides[validate_api_key_dependency] = validate_api_key_dependency

    logger.info("Application security setup complete")
    return security_manager


# Export key components for direct use
__all__ = [
    "SecurityManager",
    "get_security_manager",
    "security_manager",
    "setup_application_security",
]
