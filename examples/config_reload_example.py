# pylint: skip-file

"""Example demonstrating zero-downtime configuration reloading.

This example shows how to use the configuration reloading mechanism
with proper observability integration.
"""

import asyncio
import logging
from pathlib import Path

import src.config as config_pkg
from src.api.routers.config import (
    ReloadRequest,
    get_config_status,
    get_reload_stats,
    reload_configuration,
)
from src.services.observability.config import ObservabilityConfig
from src.services.observability.init import initialize_observability


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demonstrate_config_reload():  # pylint: disable=too-many-statements
    """Demonstrate configuration reloading capabilities."""
    logger.info("=== Configuration Reloading Demonstration ===")

    # Initialize configuration
    config = config_pkg.get_config()

    # Initialize observability if enabled
    if config.observability.enabled:
        # Convert config model to service observability config
        obs_config = ObservabilityConfig(
            enabled=config.observability.enabled,
            service_name=config.observability.service_name,
            service_version=config.observability.service_version,
            otlp_endpoint=config.observability.otlp_endpoint,
        )
        initialize_observability(obs_config)
        logger.info("Observability initialized")

    # Create configuration reloader
    reloader = config_pkg.ConfigReloader()

    logger.info("Configuration reloader created")

    # Demonstrate manual reload
    logger.info("\n--- Demonstrating Manual Configuration Reload ---")
    operation = await reloader.reload_config(
        trigger=config_pkg.ReloadTrigger.MANUAL,
        force=True,  # Force reload even if no changes
    )

    logger.info("Manual reload completed:")
    logger.info("  - Operation ID: %s", operation.operation_id)
    logger.info("  - Success: %s", operation.success)
    logger.info("  - Duration: %.1fms", operation.total_duration_ms)
    logger.info("  - Status: %s", operation.status.value)

    if operation.validation_warnings:
        logger.warning("  - Validation warnings: %s", operation.validation_warnings)

    # Demonstrate file watching
    logger.info("\n--- Demonstrating File Watching ---")

    # Create a temporary config file for testing
    temp_config_file = Path("temp_config.env")
    temp_config_file.write_text(
        "AI_DOCS__APP_NAME=Test App\nAI_DOCS__VERSION=1.0.1\n", encoding="utf-8"
    )

    try:
        # Create reloader with file watching capability
        file_reloader = config_pkg.ConfigReloader()

        # Enable file watching
        await file_reloader.enable_file_watching(poll_interval=0.5)
        logger.info("File watching enabled")

        # Simulate file change
        await asyncio.sleep(1)
        logger.info("Modifying configuration file...")
        temp_config_file.write_text(
            "AI_DOCS__APP_NAME=Updated Test App\nAI_DOCS__VERSION=1.0.2\n",
            encoding="utf-8",
        )

        # Perform a manual reload with the updated file
        file_operation = await file_reloader.reload_config(
            trigger=config_pkg.ReloadTrigger.FILE_WATCH,
            config_source=temp_config_file,
        )

        logger.info("File reload completed:")
        logger.info("  - Operation ID: %s", file_operation.operation_id)
        logger.info("  - Success: %s", file_operation.success)
        logger.info("  - Duration: %.1fms", file_operation.total_duration_ms)

        # Check reload history
        history = file_reloader.get_reload_history(limit=5)
        logger.info("File reload operations: %s", len(history))
        for op in history:
            logger.info(
                "  - %s: %s (%s)", op.operation_id, op.status.value, op.trigger.value
            )

        await file_reloader.disable_file_watching()

    finally:
        # Cleanup
        if temp_config_file.exists():
            temp_config_file.unlink()

    # Demonstrate rollback capability
    logger.info("\n--- Demonstrating Configuration Rollback ---")

    # Add some config changes to create backup history
    for _ in range(3):
        await reloader.reload_config(
            trigger=config_pkg.ReloadTrigger.MANUAL,
            force=True,
        )
        await asyncio.sleep(0.1)

    # Perform rollback
    rollback_operation = await reloader.rollback_config()

    logger.info("Rollback completed:")
    logger.info("  - Operation ID: %s", rollback_operation.operation_id)
    logger.info("  - Success: %s", rollback_operation.success)
    logger.info("  - Duration: %.1fms", rollback_operation.total_duration_ms)
    logger.info("  - Status: %s", rollback_operation.status.value)

    # Show reload statistics
    logger.info("\n--- Configuration Reload Statistics ---")
    stats = reloader.get_reload_stats()

    logger.info("Reload Statistics:")
    for key, value in stats.items():
        logger.info("  - %s: %s", key, value)

    logger.info("\n=== Demonstration Complete ===")


async def demonstrate_api_integration():
    """Demonstrate API-based configuration management."""
    logger.info("\n=== API Integration Demonstration ===")

    # This would typically be done through HTTP requests to the API
    # Here we demonstrate the underlying functionality

    # Initialize reloader
    reloader = config_pkg.ConfigReloader()

    # Set global reloader for API access
    config_pkg.set_config_reloader(reloader)

    # Simulate API reload request
    reload_request = ReloadRequest(force=True)

    try:
        response = await reload_configuration(reload_request)
        logger.info("API reload response:")
        logger.info("  - Operation ID: %s", response.operation_id)
        logger.info("  - Success: %s", response.success)
        logger.info("  - Duration: %.1fms", response.total_duration_ms)

    except Exception:
        logger.exception("API reload failed")

    # Get stats via API
    try:
        stats_response = await get_reload_stats()
        logger.info("API stats response:")
        logger.info("  - Total operations: %s", stats_response.total_operations)
        logger.info("  - Success rate: %.2f%%", stats_response.success_rate * 100)

    except Exception:
        logger.exception("API stats failed")

    # Get status via API
    try:
        status_response = await get_config_status()
        logger.info("API status response:")
        logger.info("  - Config hash: %s", status_response.get("current_config_hash"))
        logger.info(
            "  - File watching: %s", status_response.get("file_watching_enabled")
        )

    except Exception:
        logger.exception("API status failed")


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
