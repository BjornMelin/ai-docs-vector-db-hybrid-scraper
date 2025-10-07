"""Middleware manager for FastAPI with essential middleware only."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from starlette.applications import Starlette

from src.config import PerformanceConfig, get_config

from .performance import PerformanceMiddleware
from .security import SecurityMiddleware
from .timeout import TimeoutConfig, TimeoutMiddleware


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MiddlewareSpec:
    """Minimal middleware specification independent of Starlette wrappers."""

    cls: type
    kwargs: dict[str, Any]


class MiddlewareManager:
    """Simple middleware manager for essential middleware only."""

    def __init__(self, config=None):
        """Initialize middleware manager."""
        self.config = config or get_config()

    def get_middleware_stack(self) -> list[MiddlewareSpec]:
        """Get essential middleware stack in correct order.

        Simplified order:
        1. Security (basic protection)
        2. Timeout (request timeout)
        3. Performance (basic monitoring)
        """
        middleware_stack = []

        # 1. Security - Basic protection
        if self.config.security.enable_rate_limiting:
            middleware_stack.append(
                MiddlewareSpec(SecurityMiddleware, {"config": self.config.security})
            )

        # 2. Timeout - Request timeout
        timeout_config = self._build_timeout_config(self.config.performance)
        middleware_stack.append(
            MiddlewareSpec(TimeoutMiddleware, {"config": timeout_config})
        )

        # 3. Performance - Basic monitoring
        middleware_stack.append(
            MiddlewareSpec(PerformanceMiddleware, {"config": self.config.performance})
        )

        return middleware_stack

    def apply_middleware(self, app: Starlette, middleware_names: list[str]) -> None:
        """Apply specified middleware to FastAPI app."""
        timeout_config = self._build_timeout_config(self.config.performance)
        available_middleware = {
            "security": MiddlewareSpec(
                SecurityMiddleware, {"config": self.config.security}
            ),
            "timeout": MiddlewareSpec(TimeoutMiddleware, {"config": timeout_config}),
            "performance": MiddlewareSpec(
                PerformanceMiddleware, {"config": self.config.performance}
            ),
        }

        for name in middleware_names:
            middleware = available_middleware.get(name)
            if middleware is None:
                logger.warning("Unknown middleware: %s", name)
                continue
            app.add_middleware(middleware.cls, **middleware.kwargs)
            logger.info("Applied middleware: %s", name)

    @staticmethod
    def _build_timeout_config(performance: PerformanceConfig) -> TimeoutConfig:
        """Create a TimeoutConfig based on the performance settings."""

        defaults = TimeoutConfig()
        failure_threshold = max(performance.max_retries, 1)
        recovery_timeout = max(
            performance.retry_base_delay * failure_threshold,
            defaults.recovery_timeout,
        )

        return TimeoutConfig(
            enabled=defaults.enabled,
            request_timeout=performance.request_timeout,
            enable_circuit_breaker=defaults.enable_circuit_breaker,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=defaults.half_open_max_calls,
        )


def get_middleware_manager(config=None) -> MiddlewareManager:
    """Get configured middleware manager instance."""
    return MiddlewareManager(config or get_config())
