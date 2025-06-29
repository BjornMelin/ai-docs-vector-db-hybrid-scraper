#!/usr/bin/env python3
"""Example of how to use the new configurable timeout system.

This shows how to replace hardcoded timeouts with the new centralized
timeout configuration system.
"""

import asyncio
import time
from contextlib import asynccontextmanager, suppress

from .timeouts import get_timeout_config, get_timeout_settings


# Example 1: Replace hardcoded timeout in a function
async def validate_configuration_old(config_data: dict) -> bool:
    """Old way with hardcoded timeout."""
    # OLD: Hardcoded timeout
    CONFIG_VALIDATION_TIMEOUT = 120  # This was hardcoded!

    try:
        return await asyncio.wait_for(
            _perform_validation(config_data), timeout=CONFIG_VALIDATION_TIMEOUT
        )
    except TimeoutError:
        return False


async def validate_configuration_new(config_data: dict) -> bool:
    """New way with configurable timeout."""
    # NEW: Get timeout from centralized configuration
    timeout_config = get_timeout_config("config_validation")

    try:
        return await asyncio.wait_for(
            _perform_validation(config_data), timeout=timeout_config.timeout_seconds
        )
    except TimeoutError:
        return False


# Example 2: Using timeout with monitoring
async def deploy_with_timeout_monitoring(deployment_config: dict) -> bool:
    """Deploy with timeout monitoring and alerts."""
    timeout_config = get_timeout_config("deployment")
    start_time = time.time()

    async def monitor_timeout():
        """Monitor timeout and emit warnings."""
        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_config.timeout_seconds:
                print(f"ERROR: Deployment timeout exceeded ({elapsed:.1f}s)")
                break
            if timeout_config.should_alert_critical(elapsed):
                print(f"CRITICAL: Deployment approaching timeout ({elapsed:.1f}s)")
            elif timeout_config.should_warn(elapsed):
                print(
                    f"WARNING: Deployment taking longer than expected ({elapsed:.1f}s)"
                )

            await asyncio.sleep(10)  # Check every 10 seconds

    # Run deployment with monitoring
    monitor_task = asyncio.create_task(monitor_timeout())

    try:
        result = await asyncio.wait_for(
            _perform_deployment(deployment_config),
            timeout=timeout_config.timeout_seconds,
        )
        return result  # noqa: TRY300
    except TimeoutError:
        return False
    finally:
        monitor_task.cancel()


# Example 3: Context manager with configurable timeout
@asynccontextmanager
async def operation_timeout(operation_name: str):
    """Context manager for operations with configurable timeout."""
    timeout_config = get_timeout_config(operation_name)

    async def timeout_handler():
        await asyncio.sleep(timeout_config.timeout_seconds)
        msg = f"Operation '{operation_name}' timed out after {timeout_config.timeout_seconds}s"
        raise TimeoutError(msg)

    timeout_task = asyncio.create_task(timeout_handler())

    try:
        yield timeout_config
    finally:
        timeout_task.cancel()
        with suppress(asyncio.CancelledError):
            await timeout_task


# Example 4: Using all timeout settings in a service
class ServiceWithTimeouts:
    """Example service using centralized timeout configuration."""

    def __init__(self):
        self.timeout_settings = get_timeout_settings()

    async def handle_api_request(self, request):
        """Handle API request with configured timeout."""
        try:
            return await asyncio.wait_for(
                self._process_request(request),
                timeout=self.timeout_settings.api_request_timeout,
            )
        except TimeoutError:
            return {"error": "Request timeout"}

    async def execute_database_query(self, query: str):
        """Execute database query with configured timeout."""
        async with operation_timeout("database_query") as timeout_config:
            print(f"Executing query with {timeout_config.timeout_seconds}s timeout")
            return await self._run_query(query)

    async def run_background_job(self, job_data: dict):
        """Run background job with monitoring."""
        timeout_ms = self.timeout_settings.job_timeout * 1000
        print(f"Starting job with {timeout_ms}ms timeout")

        # Use the timeout throughout the job execution
        return await self._execute_job(job_data, timeout_ms)


# Helper functions (stubs for the example)
async def _perform_validation(_config_data: dict) -> bool:
    """Stub validation function."""
    await asyncio.sleep(1)
    return True


async def _perform_deployment(_deployment_config: dict) -> bool:
    """Stub deployment function."""
    await asyncio.sleep(2)
    return True


async def _process_request(_request) -> dict:
    """Stub request processing."""
    await asyncio.sleep(0.5)
    return {"status": "ok"}


async def _run_query(_query: str) -> list:
    """Stub database query."""
    await asyncio.sleep(0.1)
    return []


async def _execute_job(_job_data: dict, _timeout_ms: int) -> bool:
    """Stub job execution."""
    await asyncio.sleep(1)
    return True


# Example usage
async def main():
    """Demonstrate the new timeout system."""
    print("=== Timeout Configuration Examples ===\n")

    # Show current timeout settings
    settings = get_timeout_settings()
    print(f"Config validation timeout: {settings.config_validation_timeout}s")
    print(f"Deployment timeout: {settings.deployment_timeout}s")
    print(f"API request timeout: {settings.api_request_timeout}s")
    print()

    # Example 1: Simple validation
    print("1. Running configuration validation...")
    result = await validate_configuration_new({"test": "data"})
    print(f"   Result: {result}")
    print()

    # Example 2: Using context manager
    print("2. Running operation with timeout context...")
    try:
        async with operation_timeout("database_query") as config:
            print(f"   Operation timeout: {config.timeout_seconds}s")
            await asyncio.sleep(0.5)
            print("   Operation completed successfully")
    except TimeoutError as e:
        print(f"   Error: {e}")
    print()

    # Example 3: Service usage
    print("3. Testing service with timeouts...")
    service = ServiceWithTimeouts()
    api_result = await service.handle_api_request({"endpoint": "/test"})
    print(f"   API result: {api_result}")


if __name__ == "__main__":
    asyncio.run(main())
