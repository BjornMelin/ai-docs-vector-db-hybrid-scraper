"""Simplified middleware manager for FastAPI with essential middleware only.

This module provides basic middleware management following KISS principles.
Only includes essential middleware for V1 release.
"""

import logging

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from src.config import get_config

from .performance import PerformanceMiddleware
from .security import SecurityMiddleware
from .timeout import TimeoutMiddleware


logger = logging.getLogger(__name__)


class MiddlewareManager:
    """Simple middleware manager for essential middleware only.

    Handles basic middleware configuration for V1 release.
    """

    def __init__(self, config=None):
        """Initialize middleware manager."""
        self.config = config or get_config()

    def get_middleware_stack(self) -> list[Middleware]:
        """Get essential middleware stack in correct order.

        Simplified order:
        1. CORS (for API access)
        2. Security (basic protection)
        3. Timeout (request timeout)
        4. Performance (basic monitoring)
        """
        middleware_stack = []

        # 1. CORS - Essential for API access
        middleware_stack.append(
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure based on needs
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )

        # 2. Security - Basic protection
        if self.config.security.enable_rate_limiting:
            middleware_stack.append(
                Middleware(SecurityMiddleware, config=self.config.security)
            )

        # 3. Timeout - Request timeout
        middleware_stack.append(
            Middleware(TimeoutMiddleware, config=self.config.performance)
        )

        # 4. Performance - Basic monitoring
        middleware_stack.append(
            Middleware(PerformanceMiddleware, config=self.config.performance)
        )

        return middleware_stack

    def apply_middleware(self, app: Starlette, middleware_names: list[str]) -> None:
        """Apply specified middleware to FastAPI app."""
        available_middleware = {
            "cors": Middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure based on needs
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            "security": Middleware(SecurityMiddleware, config=self.config.security),
            "timeout": Middleware(TimeoutMiddleware, config=self.config.performance),
            "performance": Middleware(
                PerformanceMiddleware, config=self.config.performance
            ),
        }

        for name in middleware_names:
            if name in available_middleware:
                middleware = available_middleware[name]
                app.add_middleware(middleware.cls, **middleware.kwargs)
                logger.info("Applied middleware: %s", name)
            else:
                logger.warning("Unknown middleware: %s", name)


def get_middleware_manager(config=None) -> MiddlewareManager:
    """Get configured middleware manager instance."""
    return MiddlewareManager(config or get_config())


def create_middleware_manager(config=None) -> MiddlewareManager:
    """Alias for get_middleware_manager for backward compatibility."""
    return get_middleware_manager(config)
