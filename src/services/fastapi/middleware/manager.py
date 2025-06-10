"""Middleware manager for integrating production middleware with FastMCP server.

This module provides centralized management of all middleware components,
including configuration, ordering, and lifecycle management.
"""

import logging
from typing import Any

from src.config.fastapi import FastAPIProductionConfig
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .compression import CompressionMiddleware
from .performance import PerformanceMiddleware
from .security import SecurityMiddleware
from .timeout import TimeoutMiddleware
from .tracing import TracingMiddleware

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """Central manager for FastAPI production middleware.

    Handles configuration, ordering, and lifecycle management of all
    middleware components in the production stack.
    """

    def __init__(self, config: FastAPIProductionConfig):
        """Initialize middleware manager.

        Args:
            config: FastAPI production configuration
        """
        self.config = config
        self._middleware_instances: dict[str, Any] = {}

    def get_middleware_stack(self) -> list[Middleware]:
        """Get the complete middleware stack in correct order.

        Middleware order is critical for proper functionality:
        1. CORS (must be first for preflight requests)
        2. Security (early protection)
        3. Tracing (early request tracking)
        4. Timeout (request-level protection)
        5. Performance (monitoring)
        6. Compression (response transformation)
        7. Application middleware (if any)

        Returns:
            List of middleware in correct order
        """
        middleware_stack: list[Middleware] = []

        # 1. CORS Middleware (must be first)
        if self.config.cors.enabled:
            middleware_stack.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._get_cors_origins(),
                    allow_methods=self.config.cors.allow_methods,
                    allow_headers=self.config.cors.allow_headers,
                    allow_credentials=self.config.cors.allow_credentials,
                    max_age=self.config.cors.max_age,
                )
            )
            logger.info("CORS middleware enabled")

        # 2. Security Middleware
        if self.config.security.enabled:
            security_middleware = SecurityMiddleware(None, self.config.security)
            self._middleware_instances["security"] = security_middleware
            middleware_stack.append(
                Middleware(SecurityMiddleware, config=self.config.security)
            )
            logger.info("Security middleware enabled")

        # 3. Tracing Middleware
        if self.config.tracing.enabled:
            tracing_middleware = TracingMiddleware(None, self.config.tracing)
            self._middleware_instances["tracing"] = tracing_middleware
            middleware_stack.append(
                Middleware(TracingMiddleware, config=self.config.tracing)
            )
            logger.info("Tracing middleware enabled")

        # 4. Timeout Middleware
        if self.config.timeout.enabled:
            timeout_middleware = TimeoutMiddleware(None, self.config.timeout)
            self._middleware_instances["timeout"] = timeout_middleware
            middleware_stack.append(
                Middleware(TimeoutMiddleware, config=self.config.timeout)
            )
            logger.info("Timeout middleware enabled")

        # 5. Performance Monitoring Middleware
        if self.config.performance.enabled:
            performance_middleware = PerformanceMiddleware(
                None, self.config.performance
            )
            self._middleware_instances["performance"] = performance_middleware
            middleware_stack.append(
                Middleware(PerformanceMiddleware, config=self.config.performance)
            )
            logger.info("Performance monitoring middleware enabled")

        # 6. Compression Middleware
        if self.config.compression.enabled:
            # Use custom compression middleware for better control
            compression_middleware = CompressionMiddleware(
                None, self.config.compression
            )
            self._middleware_instances["compression"] = compression_middleware
            middleware_stack.append(
                Middleware(CompressionMiddleware, config=self.config.compression)
            )
            logger.info("Compression middleware enabled")

        logger.info(
            f"Middleware stack configured with {len(middleware_stack)} components"
        )
        return middleware_stack

    def _get_cors_origins(self) -> list[str]:
        """Get CORS origins based on environment configuration.

        Returns:
            List of allowed CORS origins
        """
        if hasattr(self.config, "get_environment_specific_cors"):
            return self.config.get_environment_specific_cors()
        return self.config.cors.allow_origins

    def get_middleware_instance(self, name: str) -> Any | None:
        """Get a specific middleware instance for external access.

        Args:
            name: Middleware name (security, tracing, timeout, performance, compression)

        Returns:
            Middleware instance or None if not found
        """
        return self._middleware_instances.get(name)

    def get_performance_metrics(self) -> dict | None:
        """Get performance metrics from performance middleware.

        Returns:
            Performance metrics dictionary or None if not available
        """
        performance_middleware = self.get_middleware_instance("performance")
        if performance_middleware:
            return performance_middleware.get_metrics_summary()
        return None

    def get_circuit_breaker_stats(self) -> dict | None:
        """Get circuit breaker statistics from timeout middleware.

        Returns:
            Circuit breaker statistics or None if not available
        """
        timeout_middleware = self.get_middleware_instance("timeout")
        if timeout_middleware:
            return timeout_middleware.get_circuit_stats()
        return None

    def get_health_status(self) -> dict:
        """Get overall health status from all middleware.

        Returns:
            Comprehensive health status
        """
        health = {"status": "healthy", "middleware": {}, "warnings": []}

        # Performance health
        performance_middleware = self.get_middleware_instance("performance")
        if performance_middleware:
            perf_health = performance_middleware.get_health_status()
            health["middleware"]["performance"] = perf_health

            if perf_health["status"] != "healthy":
                health["status"] = perf_health["status"]
                health["warnings"].extend(perf_health["warnings"])

        # Circuit breaker health
        timeout_middleware = self.get_middleware_instance("timeout")
        if timeout_middleware:
            circuit_stats = timeout_middleware.get_circuit_stats()
            open_circuits = [
                endpoint
                for endpoint, stats in circuit_stats.items()
                if stats["state"] == "open"
            ]

            health["middleware"]["circuit_breaker"] = {
                "open_circuits": open_circuits,
                "total_circuits": len(circuit_stats),
            }

            if open_circuits:
                health["status"] = "degraded"
                health["warnings"].append(
                    f"{len(open_circuits)} circuit breakers are open"
                )

        return health

    def reset_middleware_state(self) -> None:
        """Reset state for all stateful middleware."""
        # Reset performance metrics
        performance_middleware = self.get_middleware_instance("performance")
        if performance_middleware:
            performance_middleware.reset_metrics()
            logger.info("Performance metrics reset")

        # Reset circuit breaker state
        timeout_middleware = self.get_middleware_instance("timeout")
        if timeout_middleware:
            circuit_stats = timeout_middleware.get_circuit_stats()
            for endpoint in circuit_stats:
                timeout_middleware.reset_circuit(endpoint)
            logger.info("Circuit breaker states reset")

    def configure_app(self, app: Starlette) -> None:
        """Configure an existing Starlette app with middleware.

        Args:
            app: Starlette application to configure
        """
        middleware_stack = self.get_middleware_stack()

        # Add middleware in reverse order (Starlette processes them in reverse)
        for middleware in reversed(middleware_stack):
            app.add_middleware(middleware.cls, **middleware.kwargs)

        logger.info(
            f"Configured app with {len(middleware_stack)} middleware components"
        )

    def get_middleware_info(self) -> dict:
        """Get information about configured middleware.

        Returns:
            Dictionary with middleware configuration details
        """
        info = {"enabled_middleware": [], "configurations": {}}

        if self.config.cors.enabled:
            info["enabled_middleware"].append("cors")
            info["configurations"]["cors"] = {
                "allow_origins": self._get_cors_origins(),
                "allow_credentials": self.config.cors.allow_credentials,
            }

        if self.config.security.enabled:
            info["enabled_middleware"].append("security")
            info["configurations"]["security"] = {
                "rate_limiting": self.config.security.enable_rate_limiting,
                "rate_limit_requests": self.config.security.rate_limit_requests,
                "rate_limit_window": self.config.security.rate_limit_window,
            }

        if self.config.tracing.enabled:
            info["enabled_middleware"].append("tracing")
            info["configurations"]["tracing"] = {
                "correlation_id_header": self.config.tracing.correlation_id_header,
                "log_requests": self.config.tracing.log_requests,
                "log_responses": self.config.tracing.log_responses,
            }

        if self.config.timeout.enabled:
            info["enabled_middleware"].append("timeout")
            info["configurations"]["timeout"] = {
                "request_timeout": self.config.timeout.request_timeout,
                "circuit_breaker": self.config.timeout.enable_circuit_breaker,
                "failure_threshold": self.config.timeout.failure_threshold,
            }

        if self.config.performance.enabled:
            info["enabled_middleware"].append("performance")
            info["configurations"]["performance"] = {
                "track_response_time": self.config.performance.track_response_time,
                "track_memory_usage": self.config.performance.track_memory_usage,
                "slow_request_threshold": self.config.performance.slow_request_threshold,
            }

        if self.config.compression.enabled:
            info["enabled_middleware"].append("compression")
            info["configurations"]["compression"] = {
                "minimum_size": self.config.compression.minimum_size,
                "compression_level": self.config.compression.compression_level,
            }

        return info


def create_middleware_manager(
    config: FastAPIProductionConfig | None = None,
) -> MiddlewareManager:
    """Create and configure a middleware manager.

    Args:
        config: FastAPI production configuration (uses default if None)

    Returns:
        Configured middleware manager
    """
    if config is None:
        from src.config.fastapi import get_fastapi_config

        config = get_fastapi_config()

    return MiddlewareManager(config)


# Export manager class and factory function
__all__ = ["MiddlewareManager", "create_middleware_manager"]
