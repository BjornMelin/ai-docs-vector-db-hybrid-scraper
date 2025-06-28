"""Feature flag system for mode-aware functionality.

This module provides decorators and utilities for enabling/disabling features
based on the current application mode, supporting gradual migration between modes.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from .modes import ModeConfig, get_current_mode, get_mode_config


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


class FeatureFlag:
    """Feature flag manager for mode-aware feature enabling."""

    def __init__(self, mode_config: ModeConfig | None = None):
        """Initialize feature flag manager.

        Args:
            mode_config: Mode configuration to use. If None, uses current mode config.
        """
        self.mode_config = mode_config or get_mode_config()
        self.current_mode = get_current_mode()

    def is_enterprise_mode(self) -> bool:
        """Check if running in enterprise mode."""
        from .modes import ApplicationMode

        return self.current_mode == ApplicationMode.ENTERPRISE

    def is_simple_mode(self) -> bool:
        """Check if running in simple mode."""
        from .modes import ApplicationMode

        return self.current_mode == ApplicationMode.SIMPLE

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled in the current mode."""
        return self.mode_config.max_complexity_features.get(feature_name, False)

    def is_service_enabled(self, service_name: str) -> bool:
        """Check if a service is enabled in the current mode."""
        return service_name in self.mode_config.enabled_services


def enterprise_only(fallback_value: Any = None, log_access: bool = True):
    """Decorator to enable features only in enterprise mode.

    Args:
        fallback_value: Value to return when not in enterprise mode
        log_access: Whether to log when feature is accessed in simple mode

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_enterprise_mode():
                return await func(*args, **kwargs)

            if log_access:
                logger.info(
                    f"Enterprise feature '{func.__name__}' accessed in simple mode, "
                    f"returning fallback value: {fallback_value}"
                )
            return fallback_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_enterprise_mode():
                return func(*args, **kwargs)

            if log_access:
                logger.info(
                    f"Enterprise feature '{func.__name__}' accessed in simple mode, "
                    f"returning fallback value: {fallback_value}"
                )
            return fallback_value

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def conditional_feature(
    feature_name: str, fallback_value: Any = None, log_access: bool = True
):
    """Enable features based on mode configuration.

    Args:
        feature_name: Name of the feature to check in mode config
        fallback_value: Value to return when feature is disabled
        log_access: Whether to log when disabled feature is accessed

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_feature_enabled(feature_name):
                return await func(*args, **kwargs)

            if log_access:
                logger.info(
                    f"Feature '{feature_name}' ({func.__name__}) disabled in current mode, "
                    f"returning fallback value: {fallback_value}"
                )
            return fallback_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_feature_enabled(feature_name):
                return func(*args, **kwargs)

            if log_access:
                logger.info(
                    f"Feature '{feature_name}' ({func.__name__}) disabled in current mode, "
                    f"returning fallback value: {fallback_value}"
                )
            return fallback_value

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def service_required(
    service_name: str, fallback_value: Any = None, log_access: bool = True
):
    """Decorator to require a specific service to be enabled.

    Args:
        service_name: Name of the required service
        fallback_value: Value to return when service is not enabled
        log_access: Whether to log when disabled service is accessed

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_service_enabled(service_name):
                return await func(*args, **kwargs)

            if log_access:
                logger.warning(
                    f"Service '{service_name}' required for {func.__name__} "
                    f"but not enabled in current mode, returning fallback: {fallback_value}"
                )
            return fallback_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if feature_flag.is_service_enabled(service_name):
                return func(*args, **kwargs)

            if log_access:
                logger.warning(
                    f"Service '{service_name}' required for {func.__name__} "
                    f"but not enabled in current mode, returning fallback: {fallback_value}"
                )
            return fallback_value

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def mode_adaptive(simple_implementation: F, enterprise_implementation: F):
    """Decorator to provide different implementations for different modes.

    Args:
        simple_implementation: Function to use in simple mode
        enterprise_implementation: Function to use in enterprise mode

    Returns:
        Mode-adaptive function
    """

    @wraps(enterprise_implementation)
    async def async_wrapper(*args, **kwargs):
        feature_flag = FeatureFlag()
        if feature_flag.is_enterprise_mode():
            return await enterprise_implementation(*args, **kwargs)
        else:
            return await simple_implementation(*args, **kwargs)

    @wraps(enterprise_implementation)
    def sync_wrapper(*args, **kwargs):
        feature_flag = FeatureFlag()
        if feature_flag.is_enterprise_mode():
            return enterprise_implementation(*args, **kwargs)
        else:
            return simple_implementation(*args, **kwargs)

    # Return appropriate wrapper based on whether function is async
    if asyncio.iscoroutinefunction(enterprise_implementation):
        return async_wrapper
    else:
        return sync_wrapper


class ModeAwareFeatureManager:
    """Advanced feature manager with runtime mode switching capabilities."""

    def __init__(self):
        self._feature_flags = FeatureFlag()
        self._feature_registry: dict[str, dict[str, Any]] = {}

    def register_feature(
        self,
        name: str,
        simple_config: dict[str, Any],
        enterprise_config: dict[str, Any],
    ) -> None:
        """Register a feature with mode-specific configurations."""
        self._feature_registry[name] = {
            "simple": simple_config,
            "enterprise": enterprise_config,
        }

    def get_feature_config(self, name: str) -> dict[str, Any]:
        """Get configuration for a feature in the current mode."""
        if name not in self._feature_registry:
            raise ValueError(f"Feature '{name}' not registered")

        mode_key = (
            "enterprise" if self._feature_flags.is_enterprise_mode() else "simple"
        )
        return self._feature_registry[name][mode_key]

    def is_feature_available(self, name: str) -> bool:
        """Check if a feature is available and enabled in current mode."""
        try:
            config = self.get_feature_config(name)
            return config.get("enabled", False)
        except ValueError:
            return False


# Global feature manager instance
_feature_manager = ModeAwareFeatureManager()


def get_feature_manager() -> ModeAwareFeatureManager:
    """Get the global feature manager instance."""
    return _feature_manager


def register_feature(
    name: str, simple_config: dict[str, Any], enterprise_config: dict[str, Any]
) -> None:
    """Register a feature with the global feature manager."""
    _feature_manager.register_feature(name, simple_config, enterprise_config)


def get_feature_config(name: str) -> dict[str, Any]:
    """Get feature configuration from the global feature manager."""
    return _feature_manager.get_feature_config(name)