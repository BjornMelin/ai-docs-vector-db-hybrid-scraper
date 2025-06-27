"""Demonstration of enhanced configuration error handling.

This example shows how to use the enhanced configuration system with
comprehensive error handling, retry logic, and graceful degradation.
"""

import asyncio  # noqa: PLC0415
import json  # noqa: PLC0415
import logging  # noqa: PLC0415
from pathlib import Path

from src.config.config_manager import (
    ConfigManager,
    create_and_load_config_async,
)
from src.config.core import Config, OpenAIConfig, QdrantConfig
from src.config.error_handling import (
    ConfigValidationError,
    get_degradation_handler,
)


# Setup logging to see error handling in action
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demonstrate_basic_error_handling():
    """Demonstrate basic error handling with config loading."""
    print("\n=== Basic Error Handling Demo ===")

    # Create manager with fallback config
    fallback_config = Config(
        openai=OpenAIConfig(api_key="sk-fallback-key"),
        qdrant=QdrantConfig(url="http://fallback:6333"),
    )

    manager = ConfigManager(
        config_class=Config,
        config_file=Path("nonexistent.json"),
        enable_file_watching=False,
        fallback_config=fallback_config,
    )

    # Will use fallback since file doesn't exist
    config = manager.get_config()
    print(f"Loaded config with OpenAI key: {config.openai.api_key}")

    # Get status information
    status = manager.get_status()
    print(f"Config status: {json.dumps(status, indent=2, default=str)}")


def demonstrate_validation_error_handling():
    """Demonstrate validation error handling."""
    print("\n=== Validation Error Handling Demo ===")

    # Create config file with validation errors
    config_file = Path("invalid_config.json")
    invalid_data = {
        "openai": {
            "api_key": "invalid-key-format",  # Should start with sk-
            "max_retries": -5,  # Should be >= 0
        },
        "qdrant": {
            "timeout": "not-a-number",  # Should be int
        },
    }

    try:
        config_file.write_text(json.dumps(invalid_data))

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # This will handle validation errors gracefully
        config = manager.get_config()
        print("Config loaded despite validation errors")

    except ConfigValidationError as e:
        print(f"Validation error caught: {e}")
        print(f"Validation errors: {e.validation_errors}")

    finally:
        # Clean up
        if config_file.exists():
            config_file.unlink()


def demonstrate_reload_with_recovery():
    """Demonstrate config reload with error recovery."""
    print("\n=== Reload with Recovery Demo ===")

    config_file = Path("dynamic_config.json")

    try:
        # Start with valid config
        valid_config = {
            "openai": {"api_key": "sk-original-key"},
            "qdrant": {"url": "http://localhost:6333"},
        }
        config_file.write_text(json.dumps(valid_config))

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        original_config = manager.get_config()
        print(f"Original API key: {original_config.openai.api_key}")

        # Corrupt the config file
        config_file.write_text("{invalid json")

        # Reload will fail but keep original config
        reload_success = manager.reload_config()
        print(f"Reload with invalid JSON successful: {reload_success}")

        current_config = manager.get_config()
        print(f"API key after failed reload: {current_config.openai.api_key}")

        # Fix the config
        fixed_config = {
            "openai": {"api_key": "sk-updated-key"},
            "qdrant": {"url": "http://localhost:6333"},
        }
        config_file.write_text(json.dumps(fixed_config))

        # Now reload should work
        reload_success = manager.reload_config()
        print(f"Reload with fixed JSON successful: {reload_success}")

        updated_config = manager.get_config()
        print(f"API key after successful reload: {updated_config.openai.api_key}")

    finally:
        if config_file.exists():
            config_file.unlink()


def demonstrate_change_listeners():
    """Demonstrate change listeners with error isolation."""
    print("\n=== Change Listeners Demo ===")

    config_file = Path("listener_demo.json")

    try:
        config_file.write_text('{"openai": {"api_key": "sk-initial"}}')

        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Add various listeners
        def good_listener(old_cfg, new_cfg):
            old_key = old_cfg.openai.api_key if old_cfg.openai.api_key else "None"
            new_key = new_cfg.openai.api_key if new_cfg.openai.api_key else "None"
            print(f"Good listener: {old_key} -> {new_key}")

        def bad_listener(_old_cfg, _new_cfg):
            print("Bad listener: about to fail...")
            raise RuntimeError("Listener failure!")

        def another_good_listener(_old_cfg, _new_cfg):
            print("Another good listener: still working!")

        manager.add_change_listener(good_listener)
        manager.add_change_listener(bad_listener)
        manager.add_change_listener(another_good_listener)

        # Update config - bad listener won't affect good ones
        config_file.write_text('{"openai": {"api_key": "sk-updated"}}')
        manager.reload_config()

    finally:
        if config_file.exists():
            config_file.unlink()


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation on repeated failures."""
    print("\n=== Graceful Degradation Demo ===")

    # Get degradation handler
    degradation = get_degradation_handler()

    # Reset to clean state
    degradation.reset()

    # Simulate multiple failures
    for i in range(6):
        degradation.record_failure(
            "config_reload", RuntimeError(f"Simulated failure {i}"), {"attempt": i}
        )

    print(f"Degradation active: {degradation.degradation_active}")
    print(f"Should skip file_watch: {degradation.should_skip_operation('file_watch')}")
    print(
        f"Should skip reload_config: {degradation.should_skip_operation('reload_config')}"
    )

    # Reset degradation
    degradation.reset()
    print(f"After reset - Degradation active: {degradation.degradation_active}")


async def demonstrate_async_error_handling():
    """Demonstrate async error handling."""
    print("\n=== Async Error Handling Demo ===")

    config_file = Path("async_demo.json")

    try:
        config_file.write_text('{"openai": {"api_key": "sk-async-test"}}')

        # Create and load config asynchronously
        manager, config = await create_and_load_config_async(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        print(f"Async loaded API key: {config.openai.api_key}")

        # Update and reload asynchronously
        config_file.write_text('{"openai": {"api_key": "sk-async-updated"}}')

        reload_success = await manager.reload_config_async()
        print(f"Async reload successful: {reload_success}")

        if reload_success:
            new_config = manager.get_config()
            print(f"Updated API key: {new_config.openai.api_key}")

        # Test timeout handling
        print("\nTesting timeout handling...")
        try:
            # This will timeout
            await asyncio.wait_for(
                asyncio.sleep(5),  # Simulate slow operation
                timeout=0.5,
            )
        except TimeoutError:
            print("Timeout handled gracefully")

    finally:
        if config_file.exists():
            config_file.unlink()


def demonstrate_backup_and_restore():
    """Demonstrate configuration backup and restore."""
    print("\n=== Backup and Restore Demo ===")

    config_file = Path("backup_demo.json")

    try:
        manager = ConfigManager(
            config_class=Config,
            config_file=config_file,
            enable_file_watching=False,
        )

        # Create multiple config versions
        for i in range(5):
            config_data = {"openai": {"api_key": f"sk-version-{i}"}}
            config_file.write_text(json.dumps(config_data))
            manager.reload_config()
            print(f"Loaded version {i}")

        current_config = manager.get_config()
        print(f"\nCurrent API key: {current_config.openai.api_key}")

        # Restore from backup (index -2 means second-to-last)
        restore_success = manager.restore_from_backup(-2)
        print(f"Restore successful: {restore_success}")

        if restore_success:
            restored_config = manager.get_config()
            print(f"Restored API key: {restored_config.openai.api_key}")

    finally:
        if config_file.exists():
            config_file.unlink()


async def main():
    """Run all demonstrations."""
    print("Configuration Error Handling Demonstration")
    print("=" * 50)

    # Run sync demos
    demonstrate_basic_error_handling()
    demonstrate_validation_error_handling()
    demonstrate_reload_with_recovery()
    demonstrate_change_listeners()
    demonstrate_graceful_degradation()
    demonstrate_backup_and_restore()

    # Run async demos
    await demonstrate_async_error_handling()

    print("\n" + "=" * 50)
    print("Demonstration complete!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
