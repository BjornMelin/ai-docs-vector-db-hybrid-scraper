"""Example demonstrating zero-downtime configuration reloading.

This example shows how to use the configuration reloading mechanism
with proper observability integration and service notifications.
"""

import asyncio
import logging
import time
from pathlib import Path

from src.api.routers.config import (
    ReloadRequest,
    get_config_status,
    get_reload_stats,
    reload_configuration,
)
from src.config import Config, get_config
from src.config.reload import ConfigReloader, ReloadTrigger, set_config_reloader
from src.services.observability.init import initialize_observability


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExampleService:
    """Example service that reacts to configuration changes."""

    def __init__(self, name: str):
        self.name = name
        self.config_version = None
        self.is_running = False

    def start(self, config: Config) -> None:
        """Start the service with initial configuration."""
        self.config_version = config.version
        self.is_running = True
        logger.info(
            f"{self.name} started with config version: {self.config_version}"
        )  # TODO: Convert f-string to logging format

    def update_config(self, old_config: Config, new_config: Config) -> bool:
        """Update service configuration."""
        try:
            logger.info(
                f"{self.name} updating from config {old_config.version} to {new_config.version}"
            )

            # Simulate configuration validation
            if not new_config.app_name:
                logger.error(
                    f"{self.name} config validation failed: missing app_name"
                )  # TODO: Convert f-string to logging format
                return False

            # Simulate configuration update with some processing time
            time.sleep(0.1)

            self.config_version = new_config.version
            logger.info(
                f"{self.name} configuration updated successfully"
            )  # TODO: Convert f-string to logging format
        except Exception:
            logger.exception(f"{self.name} configuration update failed")
            return False
        else:
            return True

    def stop(self) -> None:
        """Stop the service."""
        self.is_running = False
        logger.info(f"{self.name} stopped")  # TODO: Convert f-string to logging format


async def demonstrate_config_reload():
    """Demonstrate configuration reloading capabilities."""
    logger.info("=== Configuration Reloading Demonstration ===")

    # Initialize configuration
    config = get_config()

    # Initialize observability if enabled
    if config.observability.enabled:
        initialize_observability(config.observability)
        logger.info("Observability initialized")

    # Create configuration reloader
    reloader = ConfigReloader(
        enable_signal_handler=True,
        validation_timeout=10.0,
    )
    reloader.set_current_config(config)

    # Create example services
    services = [
        ExampleService("DatabaseService"),
        ExampleService("CacheService"),
        ExampleService("SearchService"),
    ]

    # Start services
    for service in services:
        service.start(config)

    # Register service configuration callbacks
    for service in services:
        reloader.add_change_listener(
            name=f"{service.name}_config",
            callback=service.update_config,
            priority=50,
        )

    logger.info("Services started and configuration callbacks registered")

    # Demonstrate manual reload
    logger.info("\n--- Demonstrating Manual Configuration Reload ---")
    operation = await reloader.reload_config(
        trigger=ReloadTrigger.MANUAL,
        force=True,  # Force reload even if no changes
    )

    logger.info("Manual reload completed:")
    logger.info(
        f"  - Operation ID: {operation.operation_id}"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Success: {operation.success}"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Duration: {operation.total_duration_ms:.1f}ms"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Services notified: {len(operation.services_notified)}"
    )  # TODO: Convert f-string to logging format

    if operation.validation_warnings:
        logger.warning(
            f"  - Validation warnings: {operation.validation_warnings}"
        )  # TODO: Convert f-string to logging format

    # Demonstrate file watching
    logger.info("\n--- Demonstrating File Watching ---")

    # Create a temporary config file for testing
    temp_config_file = Path("temp_config.env")
    temp_config_file.write_text("AI_DOCS_APP_NAME=Test App\nAI_DOCS_VERSION=1.0.1\n")

    try:
        # Create reloader with file watching
        file_reloader = ConfigReloader(
            config_source=temp_config_file,
            enable_signal_handler=False,
        )
        file_reloader.set_current_config(config)

        # Register same callbacks
        for service in services:
            file_reloader.add_change_listener(
                name=f"{service.name}_file_config",
                callback=service.update_config,
                priority=50,
            )

        # Enable file watching
        await file_reloader.enable_file_watching(poll_interval=0.5)
        logger.info("File watching enabled")

        # Simulate file change
        await asyncio.sleep(1)
        logger.info("Modifying configuration file...")
        temp_config_file.write_text(
            "AI_DOCS_APP_NAME=Updated Test App\nAI_DOCS_VERSION=1.0.2\n"
        )

        # Wait for file change detection
        await asyncio.sleep(2)

        # Check reload history
        history = file_reloader.get_reload_history(limit=5)
        logger.info(
            f"File reload operations: {len(history)}"
        )  # TODO: Convert f-string to logging format
        for op in history:
            logger.info(
                f"  - {op.operation_id}: {op.status.value} ({op.trigger.value})"
            )

        await file_reloader.disable_file_watching()

    finally:
        # Cleanup
        if temp_config_file.exists():
            temp_config_file.unlink()

    # Demonstrate rollback capability
    logger.info("\n--- Demonstrating Configuration Rollback ---")

    # Add some config changes to create backup history
    for i in range(3):
        updated_config = Config(
            app_name=f"Test App v{i + 1}",
            version=f"1.0.{i + 1}",
        )
        reloader.set_current_config(updated_config)
        await asyncio.sleep(0.1)

    # Perform rollback
    rollback_operation = await reloader.rollback_config()

    logger.info("Rollback completed:")
    logger.info(
        f"  - Operation ID: {rollback_operation.operation_id}"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Success: {rollback_operation.success}"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Duration: {rollback_operation.total_duration_ms:.1f}ms"
    )  # TODO: Convert f-string to logging format
    logger.info(
        f"  - Status: {rollback_operation.status.value}"
    )  # TODO: Convert f-string to logging format

    # Show reload statistics
    logger.info("\n--- Configuration Reload Statistics ---")
    stats = reloader.get_reload_stats()

    logger.info("Reload Statistics:")
    for key, value in stats.items():
        logger.info(f"  - {key}: {value}")  # TODO: Convert f-string to logging format

    # Cleanup services
    for service in services:
        service.stop()

    await reloader.shutdown()
    logger.info("\n=== Demonstration Complete ===")


async def demonstrate_api_integration():
    """Demonstrate API-based configuration management."""
    logger.info("\n=== API Integration Demonstration ===")

    # This would typically be done through HTTP requests to the API
    # Here we demonstrate the underlying functionality

    # Initialize reloader
    reloader = ConfigReloader()
    reloader.set_current_config(get_config())

    # Set global reloader for API access
    set_config_reloader(reloader)

    # Simulate API reload request
    reload_request = ReloadRequest(force=True)

    try:
        response = await reload_configuration(reload_request)
        logger.info("API reload response:")
        logger.info(
            f"  - Operation ID: {response.operation_id}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  - Success: {response.success}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  - Duration: {response.total_duration_ms:.1f}ms"
        )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception("API reload failed")

    # Get stats via API
    try:
        stats_response = await get_reload_stats()
        logger.info("API stats response:")
        logger.info(
            f"  - Total operations: {stats_response.total_operations}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  - Success rate: {stats_response.success_rate:.2%}"
        )  # TODO: Convert f-string to logging format

    except Exception:
        logger.exception("API stats failed")

    # Get status via API
    try:
        status_response = await get_config_status()
        logger.info("API status response:")
        logger.info(
            f"  - Config hash: {status_response.get('current_config_hash')}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  - File watching: {status_response.get('file_watching_enabled')}"
        )

    except Exception:
        logger.exception("API status failed")

    await reloader.shutdown()


async def main():
    """Main demonstration function."""
    try:
        await demonstrate_config_reload()
        await demonstrate_api_integration()

    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception:
        logger.exception("Demonstration failed")


if __name__ == "__main__":
    asyncio.run(main())
