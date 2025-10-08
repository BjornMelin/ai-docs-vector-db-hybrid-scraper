"""Feature flag system for mode-aware functionality.

This module provides decorators and utilities for enabling/disabling features
based on the current application mode, supporting gradual migration between modes.
"""

import logging
from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from .modes import ApplicationMode, ModeConfig, get_current_mode, get_mode_config


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config


class FeatureFlag:
    """Feature flag manager for mode-aware feature enabling."""

    def __init__(
        self,
        mode_config: ModeConfig | None = None,
        *,
        config: "Config | None" = None,
    ):
        """Initialize feature flag manager.

        Args:
            mode_config: Mode configuration to use. If None, uses current mode config.
            config: Optional configuration override for resolving mode settings.
        """

        self._mode_config_override = mode_config
        self._config_override = config

    @property
    def mode_config(self) -> ModeConfig:
        """Lazy accessor for the active mode configuration."""

        if self._mode_config_override is None:
            if self._config_override is None:
                self._mode_config_override = get_mode_config()
            else:
                self._mode_config_override = get_mode_config(
                    config=self._config_override
                )
        return self._mode_config_override

    def _current_mode(self) -> ApplicationMode:
        """Resolve the current application mode."""

        if self._config_override is None:
            return get_current_mode()
        return get_current_mode(config=self._config_override)

    def is_enterprise_mode(self) -> bool:
        """Check if running in enterprise mode."""

        # Use get_current_mode() to allow for runtime testing/mocking
        return self._current_mode() == ApplicationMode.ENTERPRISE

    def is_simple_mode(self) -> bool:
        """Check if running in simple mode."""

        # Use get_current_mode() to allow for runtime testing/mocking
        return self._current_mode() == ApplicationMode.SIMPLE

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled in the current mode."""

        return self.mode_config.max_complexity_features.get(feature_name, False)

    def is_service_enabled(self, service_name: str) -> bool:
        """Check if a service is enabled in the current mode."""

        return service_name in self.mode_config.enabled_services


def _wrap_with_feature_flag(
    func: Callable[..., Any],
    gate: Callable[[FeatureFlag], bool],
    *,
    fallback_value: Any,
    log: Callable[[str], None] | None = None,
) -> Callable[..., Any]:
    """Return a callable that enforces a feature flag predicate."""

    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            feature_flag = FeatureFlag()
            if gate(feature_flag):
                return await func(*args, **kwargs)
            if log:
                log(func.__name__)
            return fallback_value

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        feature_flag = FeatureFlag()
        if gate(feature_flag):
            return func(*args, **kwargs)
        if log:
            log(func.__name__)
        return fallback_value

    return sync_wrapper


def enterprise_only(fallback_value: Any = None, log_access: bool = True):
    """Decorator to enable features only in enterprise mode.

    Args:
        fallback_value: Value to return when not in enterprise mode
        log_access: Whether to log when feature is accessed in simple mode

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log_fn = (
            (
                lambda name: logger.info(
                    (
                        "Enterprise feature '%s' accessed in simple mode, "
                        "returning fallback value: %s"
                    ),
                    name,
                    fallback_value,
                )
            )
            if log_access
            else None
        )

        return _wrap_with_feature_flag(
            func,
            lambda flag: flag.is_enterprise_mode(),
            fallback_value=fallback_value,
            log=log_fn,
        )

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

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log_fn = (
            (
                lambda name: logger.info(
                    (
                        "Feature '%s' (%s) disabled in current mode, "
                        "returning fallback value: %s"
                    ),
                    feature_name,
                    name,
                    fallback_value,
                )
            )
            if log_access
            else None
        )

        return _wrap_with_feature_flag(
            func,
            lambda flag: flag.is_feature_enabled(feature_name),
            fallback_value=fallback_value,
            log=log_fn,
        )

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

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log_fn = (
            (
                lambda name: logger.warning(
                    (
                        "Service '%s' required for %s but not enabled in current mode, "
                        "returning fallback: %s"
                    ),
                    service_name,
                    name,
                    fallback_value,
                )
            )
            if log_access
            else None
        )

        return _wrap_with_feature_flag(
            func,
            lambda flag: flag.is_service_enabled(service_name),
            fallback_value=fallback_value,
            log=log_fn,
        )

    return decorator


def mode_adaptive(
    simple_implementation: Callable[..., Any],
    enterprise_implementation: Callable[..., Any],
):
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
        return await simple_implementation(*args, **kwargs)

    @wraps(enterprise_implementation)
    def sync_wrapper(*args, **kwargs):
        feature_flag = FeatureFlag()
        if feature_flag.is_enterprise_mode():
            return enterprise_implementation(*args, **kwargs)
        return simple_implementation(*args, **kwargs)

    # Return appropriate wrapper based on whether function is async
    if iscoroutinefunction(enterprise_implementation):
        return async_wrapper
    return sync_wrapper


class ModeAwareFeatureManager:
    """Feature manager with runtime mode switching capabilities."""

    def __init__(self):
        """Initialize the feature manager."""
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
            msg = f"Feature '{name}' not registered"
            raise ValueError(msg)

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


# Global feature manager instance (lazily initialized to avoid import cycles)
_feature_manager: ModeAwareFeatureManager | None = None


def get_feature_manager() -> ModeAwareFeatureManager:
    """Get the global feature manager instance."""

    global _feature_manager  # pylint: disable=global-statement
    if _feature_manager is None:
        _feature_manager = ModeAwareFeatureManager()
    return _feature_manager


def register_feature(
    name: str, simple_config: dict[str, Any], enterprise_config: dict[str, Any]
) -> None:
    """Register a feature with the global feature manager."""

    get_feature_manager().register_feature(name, simple_config, enterprise_config)


def get_feature_config(name: str) -> dict[str, Any]:
    """Get feature configuration from the global feature manager."""

    return get_feature_manager().get_feature_config(name)
