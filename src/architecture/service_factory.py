"""Service factory pattern for mode-aware service instantiation.

This module provides a factory pattern for creating services based on the current
application mode, enabling different implementations for simple vs enterprise modes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

from .modes import ApplicationMode, get_current_mode, get_mode_config


logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceProtocol(Protocol):
    """Protocol defining the interface for mode-aware services."""

    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    async def cleanup(self) -> None:
        """Clean up service resources."""
        ...

    def get_service_name(self) -> str:
        """Get the service name."""
        ...


class ServiceNotEnabledError(Exception):
    """Raised when trying to access a service not enabled in current mode."""


class ServiceNotFoundError(Exception):
    """Raised when trying to access a service that hasn't been registered."""


class ServiceInitializationError(Exception):
    """Raised when service initialization fails."""


class ModeAwareServiceFactory:
    """Factory for creating mode-aware services with different implementations."""

    def __init__(self, mode: ApplicationMode | None = None):
        """Initialize the service factory.

        Args:
            mode: Application mode to use. If None, detects from environment.
        """
        self.mode = mode or get_current_mode()
        self.mode_config = get_mode_config(self.mode)
        self._service_registry: dict[str, dict[str, type[ServiceProtocol]]] = {}
        self._service_instances: dict[str, ServiceProtocol] = {}
        self._initialization_status: dict[str, bool] = {}

    def register_service(
        self,
        name: str,
        simple_impl: type[ServiceProtocol] | None = None,
        enterprise_impl: type[ServiceProtocol] | None = None,
    ) -> None:
        """Register different implementations for different modes.

        Args:
            name: Service name
            simple_impl: Implementation for simple mode
            enterprise_impl: Implementation for enterprise mode
        """
        if name not in self._service_registry:
            self._service_registry[name] = {}

        if simple_impl:
            self._service_registry[name]["simple"] = simple_impl
        if enterprise_impl:
            self._service_registry[name]["enterprise"] = enterprise_impl

        logger.debug(f"Registered service '{name}' with mode implementations")

    def register_universal_service(
        self, name: str, implementation: type[ServiceProtocol]
    ) -> None:
        """Register a service that works in both modes.

        Args:
            name: Service name
            implementation: Service implementation for both modes
        """
        self.register_service(name, implementation, implementation)

    async def get_service(self, name: str) -> ServiceProtocol:
        """Get a service instance for the current mode.

        Args:
            name: Service name

        Returns:
            Initialized service instance

        Raises:
            ServiceNotEnabledError: If service not enabled in current mode
            ServiceNotFoundError: If service not registered
            ServiceInitializationError: If service initialization fails
        """
        # Check if service is enabled in current mode
        if name not in self.mode_config.enabled_services:
            msg = f"Service '{name}' not enabled in {self.mode.value} mode"
            raise ServiceNotEnabledError(msg)

        # Return cached instance if available
        if name in self._service_instances:
            return self._service_instances[name]

        # Get service class for current mode
        service_class = self._get_service_class(name)

        # Create and initialize service instance
        try:
            service = service_class()
            await service.initialize()

            # Cache the initialized service
            self._service_instances[name] = service
            self._initialization_status[name] = True

            logger.info(f"Initialized service '{name}' in {self.mode.value} mode")

        except Exception as e:
            self._initialization_status[name] = False
            msg = f"Failed to initialize service '{name}': {e}"
            raise ServiceInitializationError(msg) from e
        else:
            return service

    def _get_service_class(self, name: str) -> type[ServiceProtocol]:
        """Get the service class for the current mode."""
        if name not in self._service_registry:
            msg = f"Service '{name}' not registered"
            raise ServiceNotFoundError(msg)

        mode_implementations = self._service_registry[name]
        mode_key = self.mode.value

        if mode_key not in mode_implementations:
            # Try to fallback to the other mode's implementation
            fallback_key = "simple" if mode_key == "enterprise" else "enterprise"
            if fallback_key in mode_implementations:
                logger.warning(
                    f"Service '{name}' not available for {mode_key} mode, "
                    f"falling back to {fallback_key} implementation"
                )
                return mode_implementations[fallback_key]

            msg = f"Service '{name}' has no implementation for {mode_key} mode"
            raise ServiceNotFoundError(msg)

        return mode_implementations[mode_key]

    async def get_service_optional(self, name: str) -> ServiceProtocol | None:
        """Get a service instance, returning None if not available.

        Args:
            name: Service name

        Returns:
            Service instance or None if not available
        """
        try:
            return await self.get_service(name)
        except (
            ServiceNotEnabledError,
            ServiceNotFoundError,
            ServiceInitializationError,
        ) as e:
            logger.debug(f"Service '{name}' not available: {e}")
            return None

    def is_service_available(self, name: str) -> bool:
        """Check if a service is available in the current mode.

        Args:
            name: Service name

        Returns:
            True if service is available and can be instantiated
        """
        try:
            # Check if enabled in current mode
            if name not in self.mode_config.enabled_services:
                return False

            # Check if implementation exists
            self._get_service_class(name)

        except (ServiceNotFoundError, ServiceNotEnabledError):
            return False
        else:
            return True

    def get_available_services(self) -> list[str]:
        """Get list of services available in the current mode.

        Returns:
            list of service names available in current mode
        """
        return [
            service_name
            for service_name in self.mode_config.enabled_services
            if self.is_service_available(service_name)
        ]

    def get_service_status(self, name: str) -> dict[str, Any]:
        """Get status information for a service.

        Args:
            name: Service name

        Returns:
            Dictionary with service status information
        """
        return {
            "name": name,
            "available": self.is_service_available(name),
            "enabled": name in self.mode_config.enabled_services,
            "initialized": self._initialization_status.get(name, False),
            "mode": self.mode.value,
        }

    def is_service_registered(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service name

        Returns:
            True if service is registered
        """
        return name in self._service_registry

    def get_registered_service_implementations(
        self, name: str
    ) -> dict[str, type[ServiceProtocol]]:
        """Get registered implementations for a service.

        Args:
            name: Service name

        Returns:
            Dictionary mapping mode names to implementation classes

        Raises:
            ServiceNotFoundError: If service not registered
        """
        if name not in self._service_registry:
            msg = f"Service '{name}' not registered"
            raise ServiceNotFoundError(msg)

        return self._service_registry[name].copy()

    async def cleanup_all_services(self) -> None:
        """Clean up all initialized services."""
        for name, service in self._service_instances.items():
            try:
                await service.cleanup()
                logger.debug(f"Cleaned up service '{name}'")
            except Exception as e:
                logger.exception("Error cleaning up service '{name}'")

        self._service_instances.clear()
        self._initialization_status.clear()

    def get_mode_info(self) -> dict[str, Any]:
        """Get information about the current mode and configuration.

        Returns:
            Dictionary with mode information
        """
        return {
            "mode": self.mode.value,
            "enabled_services": self.mode_config.enabled_services,
            "resource_limits": self.mode_config.resource_limits,
            "middleware_stack": self.mode_config.middleware_stack,
            "advanced_monitoring": self.mode_config.enable_advanced_monitoring,
            "deployment_features": self.mode_config.enable_deployment_features,
            "a_b_testing": self.mode_config.enable_a_b_testing,
        }


class BaseService(ABC):
    """Abstract base class for mode-aware services."""

    def __init__(self):
        self._initialized = False
        self._cleanup_called = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up service resources."""

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name."""

    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def _mark_initialized(self) -> None:
        """Mark service as initialized."""
        self._initialized = True

    def _mark_cleanup(self) -> None:
        """Mark service as cleaned up."""
        self._cleanup_called = True
        self._initialized = False


# Global service factory instance
_service_factory: ModeAwareServiceFactory | None = None


def get_service_factory() -> ModeAwareServiceFactory:
    """Get the global service factory instance."""
    global _service_factory
    if _service_factory is None:
        _service_factory = ModeAwareServiceFactory()
    return _service_factory


def reset_service_factory() -> None:
    """Reset the global service factory instance."""
    global _service_factory
    _service_factory = None


async def get_service(name: str) -> ServiceProtocol:
    """Get a service from the global service factory."""
    factory = get_service_factory()
    return await factory.get_service(name)


async def get_service_optional(name: str) -> ServiceProtocol | None:
    """Get a service from the global service factory, returning None if not available."""
    factory = get_service_factory()
    return await factory.get_service_optional(name)


def is_service_available(name: str) -> bool:
    """Check if a service is available in the current mode."""
    factory = get_service_factory()
    return factory.is_service_available(name)


def register_service(
    name: str,
    simple_impl: type[ServiceProtocol] | None = None,
    enterprise_impl: type[ServiceProtocol] | None = None,
) -> None:
    """Register a service with the global service factory."""
    factory = get_service_factory()
    factory.register_service(name, simple_impl, enterprise_impl)


def register_universal_service(
    name: str, implementation: type[ServiceProtocol]
) -> None:
    """Register a universal service with the global service factory."""
    factory = get_service_factory()
    factory.register_universal_service(name, implementation)
