"""Configuration lifecycle management integration.

This module provides integration between the configuration reloader and
application lifecycle events, ensuring proper initialization, cleanup,
and service registration for zero-downtime configuration reloading.
"""

import logging
from collections.abc import Callable
from typing import Dict, List

from fastapi import FastAPI

from ..services.observability.init import (
    initialize_observability,
    shutdown_observability,
)
from .core import Config, get_config, set_config
from .reload import ConfigReloader, set_config_reloader


logger = logging.getLogger(__name__)


class ConfigurationLifecycleManager:
    """Manages configuration lifecycle integration with FastAPI."""

    def __init__(self, app: FastAPI):
        """Initialize lifecycle manager with FastAPI app.

        Args:
            app: FastAPI application instance
        """
        self.app = app
        self.reloader: ConfigReloader | None = None
        self.service_callbacks: List[Callable[[Config, Config], bool]] = []

        # Setup lifecycle events
        self._setup_lifecycle_events()

    def _setup_lifecycle_events(self) -> None:
        """Setup FastAPI lifecycle events for configuration management."""

        @self.app.on_event("startup")
        async def startup_config():
            """Initialize configuration system on startup."""
            await self.initialize_configuration()

        @self.app.on_event("shutdown")
        async def shutdown_config():
            """Cleanup configuration system on shutdown."""
            await self.shutdown_configuration()

    async def initialize_configuration(self) -> None:
        """Initialize configuration reloading system."""
        try:
            logger.info("Initializing configuration reloading system...")

            # Get current configuration
            current_config = get_config()

            # Initialize observability if enabled
            if current_config.observability.enabled:
                observability_initialized = initialize_observability(
                    current_config.observability
                )
                if observability_initialized:
                    logger.info(
                        "OpenTelemetry observability initialized for configuration"
                    )

            # Create configuration reloader
            self.reloader = ConfigReloader(
                enable_signal_handler=True,
                validation_timeout=30.0,
            )

            # Set current configuration
            self.reloader.set_current_config(current_config)

            # Register core service update callbacks
            await self._register_core_service_callbacks()

            # Set global reloader instance
            set_config_reloader(self.reloader)

            logger.info("Configuration reloading system initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize configuration system: {e}")
            raise

    async def _register_core_service_callbacks(self) -> None:
        """Register core service configuration update callbacks."""

        # Database configuration callback
        def update_database_config(old_config: Config, new_config: Config) -> bool:
            """Update database configuration."""
            try:
                if old_config.database != new_config.database:
                    logger.info("Database configuration changed, marking for restart")
                    # Note: Database connections typically require restart
                    # This is a placeholder for actual database reconfiguration
                return True
            except Exception as e:
                logger.exception(f"Failed to update database configuration: {e}")
                return False

        # Cache configuration callback
        def update_cache_config(old_config: Config, new_config: Config) -> bool:
            """Update cache configuration."""
            try:
                if old_config.cache != new_config.cache:
                    logger.info("Cache configuration changed")
                    # Placeholder for cache reconfiguration
                    # In a real implementation, this would update cache connections
                return True
            except Exception as e:
                logger.exception(f"Failed to update cache configuration: {e}")
                return False

        # Observability configuration callback
        def update_observability_config(old_config: Config, new_config: Config) -> bool:
            """Update observability configuration."""
            try:
                if old_config.observability != new_config.observability:
                    logger.info("Observability configuration changed")

                    # Shutdown old observability if needed
                    if (
                        old_config.observability.enabled
                        and not new_config.observability.enabled
                    ):
                        shutdown_observability()
                        logger.info("Observability disabled")

                    # Initialize new observability if needed
                    elif new_config.observability.enabled:
                        success = initialize_observability(new_config.observability)
                        if success:
                            logger.info("Observability reconfigured")
                        else:
                            logger.warning("Failed to reconfigure observability")
                            return False

                return True
            except Exception as e:
                logger.exception(f"Failed to update observability configuration: {e}")
                return False

        # Performance configuration callback
        def update_performance_config(old_config: Config, new_config: Config) -> bool:
            """Update performance configuration."""
            try:
                if old_config.performance != new_config.performance:
                    logger.info("Performance configuration changed")
                    # Placeholder for performance setting updates
                    # This could update request limits, timeouts, etc.
                return True
            except Exception as e:
                logger.exception(f"Failed to update performance configuration: {e}")
                return False

        # Security configuration callback
        def update_security_config(old_config: Config, new_config: Config) -> bool:
            """Update security configuration."""
            try:
                if old_config.security != new_config.security:
                    logger.info("Security configuration changed")
                    # Placeholder for security setting updates
                    # This could update API keys, rate limits, etc.
                return True
            except Exception as e:
                logger.exception(f"Failed to update security configuration: {e}")
                return False

        # Global configuration update callback
        async def update_global_config(_old_config: Config, new_config: Config) -> bool:
            """Update global configuration instance."""
            try:
                # Update global configuration
                set_config(new_config)
                logger.info("Global configuration updated")
                return True
            except Exception as e:
                logger.exception(f"Failed to update global configuration: {e}")
                return False

        # Register all callbacks
        if self.reloader:
            self.reloader.add_change_listener(
                "database_config",
                update_database_config,
                priority=100,
            )

            self.reloader.add_change_listener(
                "cache_config",
                update_cache_config,
                priority=90,
            )

            self.reloader.add_change_listener(
                "observability_config",
                update_observability_config,
                priority=80,
            )

            self.reloader.add_change_listener(
                "performance_config",
                update_performance_config,
                priority=70,
            )

            self.reloader.add_change_listener(
                "security_config",
                update_security_config,
                priority=60,
            )

            self.reloader.add_change_listener(
                "global_config",
                update_global_config,
                priority=10,  # Lowest priority - update global config last
                async_callback=True,
            )

    async def shutdown_configuration(self) -> None:
        """Shutdown configuration system."""
        try:
            logger.info("Shutting down configuration reloading system...")

            if self.reloader:
                await self.reloader.shutdown()

            # Shutdown observability
            shutdown_observability()

            logger.info("Configuration reloading system shutdown completed")

        except Exception as e:
            logger.exception(f"Error during configuration system shutdown: {e}")

    def register_service_callback(
        self,
        name: str,
        callback: Callable[[Config, Config], bool],
        priority: int = 50,
        async_callback: bool = False,
    ) -> None:
        """Register a service-specific configuration update callback.

        Args:
            name: Unique name for the callback
            callback: Callback function (old_config, new_config) -> success
            priority: Callback priority (higher = called earlier)
            async_callback: Whether the callback is async
        """
        if self.reloader:
            self.reloader.add_change_listener(
                name=name,
                callback=callback,
                priority=priority,
                async_callback=async_callback,
            )
            logger.info(f"Registered configuration callback: {name}")

    def unregister_service_callback(self, name: str) -> bool:
        """Unregister a service configuration callback.

        Args:
            name: Name of the callback to remove

        Returns:
            True if callback was removed, False if not found
        """
        if self.reloader:
            return self.reloader.remove_change_listener(name)
        return False

    async def enable_file_watching(self, poll_interval: float = 1.0) -> None:
        """Enable automatic configuration file watching.

        Args:
            poll_interval: File polling interval in seconds
        """
        if self.reloader:
            await self.reloader.enable_file_watching(poll_interval)

    async def disable_file_watching(self) -> None:
        """Disable automatic configuration file watching."""
        if self.reloader:
            await self.reloader.disable_file_watching()

    def get_reloader_stats(self) -> Dict[str, any]:
        """Get configuration reloader statistics."""
        if self.reloader:
            return self.reloader.get_reload_stats()
        return {}


# Global lifecycle manager instance
_lifecycle_manager: ConfigurationLifecycleManager | None = None


def setup_configuration_lifecycle(app: FastAPI) -> ConfigurationLifecycleManager:
    """Setup configuration lifecycle management for FastAPI app.

    Args:
        app: FastAPI application instance

    Returns:
        ConfigurationLifecycleManager instance
    """
    global _lifecycle_manager
    _lifecycle_manager = ConfigurationLifecycleManager(app)
    return _lifecycle_manager


def get_lifecycle_manager() -> ConfigurationLifecycleManager | None:
    """Get the global configuration lifecycle manager."""
    return _lifecycle_manager


def register_config_callback(
    name: str,
    callback: Callable[[Config, Config], bool],
    priority: int = 50,
    async_callback: bool = False,
) -> None:
    """Register a configuration change callback.

    Convenience function for registering callbacks with the global lifecycle manager.

    Args:
        name: Unique name for the callback
        callback: Callback function (old_config, new_config) -> success
        priority: Callback priority (higher = called earlier)
        async_callback: Whether the callback is async
    """
    if _lifecycle_manager:
        _lifecycle_manager.register_service_callback(
            name=name,
            callback=callback,
            priority=priority,
            async_callback=async_callback,
        )
    else:
        logger.warning(
            f"Cannot register config callback '{name}': lifecycle manager not initialized"
        )
