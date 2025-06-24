"""Zero-downtime configuration reloading mechanism.

This module provides comprehensive configuration reloading capabilities with proper
validation, rollback, and observability integration. Supports signal-based and
API-based configuration reloads with zero-downtime guarantees.
"""

import asyncio
import contextlib
import json
import logging
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

from pydantic import ValidationError

from ..services.observability.config_instrumentation import (
    ConfigOperationType,
    instrument_config_operation,
    record_config_change,
    trace_async_config_operation,
)
from ..services.observability.config_performance import record_config_operation
from .core import Config


logger = logging.getLogger(__name__)


class ReloadTrigger(str, Enum):
    """Configuration reload trigger types."""

    SIGNAL = "signal"
    API = "api"
    FILE_WATCH = "file_watch"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ReloadStatus(str, Enum):
    """Configuration reload operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    APPLYING = "applying"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ReloadOperation:
    """Configuration reload operation tracking."""

    operation_id: str = field(default_factory=lambda: str(uuid4()))
    trigger: ReloadTrigger = ReloadTrigger.MANUAL
    status: ReloadStatus = ReloadStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Configuration tracking
    previous_config_hash: Optional[str] = None
    new_config_hash: Optional[str] = None
    config_source: Optional[str] = None

    # Validation results
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    # Operation results
    success: bool = False
    error_message: Optional[str] = None
    changes_applied: list[str] = field(default_factory=list)
    services_notified: list[str] = field(default_factory=list)

    # Performance metrics
    validation_duration_ms: float = 0.0
    apply_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    def complete(self, success: bool, error_message: Optional[str] = None) -> None:
        """Mark the operation as complete."""
        self.end_time = time.time()
        self.success = success
        self.error_message = error_message
        self.total_duration_ms = (self.end_time - self.start_time) * 1000
        self.status = ReloadStatus.SUCCESS if success else ReloadStatus.FAILED


@dataclass
class ConfigChangeListener:
    """Configuration change listener callback."""

    name: str
    callback: Callable[[Config, Config], bool]  # old_config, new_config -> success
    priority: int = 0  # Higher priority listeners are called first
    async_callback: bool = False
    timeout_seconds: float = 30.0


class ConfigReloader:
    """Zero-downtime configuration reloader with comprehensive observability.

    Features:
    - Signal-based reloading (SIGHUP by default)
    - API endpoint for manual reloading
    - Configuration validation before applying changes
    - Automatic rollback on validation failures
    - Comprehensive observability and performance tracking
    - Change broadcasting to registered services
    - File watching for automatic reloads
    """

    def __init__(
        self,
        config_source: Optional[Path] = None,
        backup_count: int = 5,
        validation_timeout: float = 30.0,
        enable_signal_handler: bool = True,
        signal_number: int = signal.SIGHUP,
    ):
        """Initialize configuration reloader.

        Args:
            config_source: Path to configuration file (defaults to .env)
            backup_count: Number of configuration backups to maintain
            validation_timeout: Timeout for configuration validation
            enable_signal_handler: Whether to enable signal-based reloading
            signal_number: Signal number for triggering reload
        """
        self.config_source = config_source or Path(".env")
        self.backup_count = backup_count
        self.validation_timeout = validation_timeout
        self.enable_signal_handler = enable_signal_handler
        self.signal_number = signal_number

        # Current configuration state
        self._current_config: Optional[Config] = None
        self._config_hash: Optional[str] = None
        self._reload_lock = Lock()

        # Configuration backups for rollback
        self._config_backups: list[tuple[str, Config]] = []

        # Change listeners
        self._change_listeners: list[ConfigChangeListener] = []

        # Operation tracking
        self._reload_history: list[ReloadOperation] = []
        self._max_history = 100

        # File watching
        self._file_watcher: Optional[asyncio.Task] = None
        self._file_watch_enabled = False

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="config-reload"
        )

        # Setup signal handler
        if self.enable_signal_handler and hasattr(signal, "SIGHUP"):
            self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        """Setup signal handler for configuration reloading."""

        def signal_handler(signum: int, _frame) -> None:
            logger.info(f"Received signal {signum}, triggering configuration reload")
            # Store reference to prevent task from being garbage collected
            task = asyncio.create_task(self.reload_config(trigger=ReloadTrigger.SIGNAL))
            # Add done callback to handle any exceptions
            task.add_done_callback(
                lambda t: logger.exception("Signal reload failed")
                if t.exception()
                else None
            )

        try:
            signal.signal(self.signal_number, signal_handler)
            logger.info(
                f"Configuration reload signal handler setup for signal {self.signal_number}"
            )
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to setup signal handler: {e}")

    def set_current_config(self, config: Config) -> None:
        """Set the current configuration instance."""
        with self._reload_lock:
            self._current_config = config
            self._config_hash = self._calculate_config_hash(config)
            self._add_config_backup(config)

    def _calculate_config_hash(self, config: Config) -> str:
        """Calculate hash of configuration for change detection."""
        # Convert config to dict and create a stable hash
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        import hashlib

        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def _add_config_backup(self, config: Config) -> None:
        """Add configuration backup for rollback capability."""
        backup_hash = self._calculate_config_hash(config)
        backup_config = deepcopy(config)

        # Add to backup list
        self._config_backups.append((backup_hash, backup_config))

        # Maintain backup count limit
        if len(self._config_backups) > self.backup_count:
            self._config_backups.pop(0)

    def add_change_listener(
        self,
        name: str,
        callback: Callable[[Config, Config], bool],
        priority: int = 0,
        async_callback: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Add a configuration change listener.

        Args:
            name: Unique name for the listener
            callback: Callback function (old_config, new_config) -> success
            priority: Priority for callback execution (higher = earlier)
            async_callback: Whether the callback is async
            timeout_seconds: Timeout for callback execution
        """
        listener = ConfigChangeListener(
            name=name,
            callback=callback,
            priority=priority,
            async_callback=async_callback,
            timeout_seconds=timeout_seconds,
        )

        self._change_listeners.append(listener)
        self._change_listeners.sort(
            key=lambda listener: listener.priority, reverse=True
        )

        logger.info(
            f"Added configuration change listener: {name} (priority: {priority})"
        )

    def remove_change_listener(self, name: str) -> bool:
        """Remove a configuration change listener by name."""
        for i, listener in enumerate(self._change_listeners):
            if listener.name == name:
                del self._change_listeners[i]
                logger.info(f"Removed configuration change listener: {name}")
                return True
        return False

    @instrument_config_operation(
        operation_type=ConfigOperationType.LOAD,
        operation_name="config.reload",
    )
    async def reload_config(
        self,
        trigger: ReloadTrigger = ReloadTrigger.MANUAL,
        config_source: Optional[Path] = None,
        force: bool = False,
    ) -> ReloadOperation:
        """Reload configuration with zero-downtime guarantees.

        Args:
            trigger: What triggered the reload
            config_source: Optional specific config source
            force: Force reload even if no changes detected

        Returns:
            ReloadOperation with results and metrics
        """
        operation = ReloadOperation(trigger=trigger)
        operation.config_source = str(config_source or self.config_source)

        # Prevent concurrent reloads
        if not self._reload_lock.acquire(blocking=False):
            operation.complete(False, "Another reload operation is in progress")
            return operation

        try:
            async with trace_async_config_operation(
                operation_type=ConfigOperationType.UPDATE,
                operation_name="config.hot_reload",
                correlation_id=operation.operation_id,
            ) as span:
                span.set_attribute("reload.trigger", trigger.value)
                span.set_attribute("reload.force", force)

                return await self._perform_reload(operation, config_source, force, span)

        finally:
            self._reload_lock.release()
            self._reload_history.append(operation)

            # Maintain history limit
            if len(self._reload_history) > self._max_history:
                self._reload_history.pop(0)

    async def _perform_reload(
        self,
        operation: ReloadOperation,
        config_source: Optional[Path],
        force: bool,
        _span,
    ) -> ReloadOperation:
        """Perform the actual configuration reload operation."""
        try:
            operation.status = ReloadStatus.IN_PROGRESS
            logger.info(
                f"Starting configuration reload (trigger: {operation.trigger.value})"
            )

            # Store current config state
            if self._current_config:
                operation.previous_config_hash = self._config_hash

            # Load new configuration
            new_config = await self._load_new_config(
                config_source or self.config_source
            )
            new_config_hash = self._calculate_config_hash(new_config)
            operation.new_config_hash = new_config_hash

            # Check if configuration actually changed
            if not force and new_config_hash == self._config_hash:
                logger.info("Configuration unchanged, skipping reload")
                operation.complete(True, "No configuration changes detected")
                return operation

            # Validate new configuration
            operation.status = ReloadStatus.VALIDATING
            validation_start = time.time()

            validation_result = await self._validate_config(new_config)
            operation.validation_duration_ms = (time.time() - validation_start) * 1000
            operation.validation_errors = validation_result.get("errors", [])
            operation.validation_warnings = validation_result.get("warnings", [])

            if operation.validation_errors:
                error_msg = f"Configuration validation failed: {', '.join(operation.validation_errors)}"
                logger.error(error_msg)
                operation.complete(False, error_msg)
                return operation

            # Apply new configuration
            operation.status = ReloadStatus.APPLYING
            apply_start = time.time()

            success = await self._apply_config_changes(
                self._current_config, new_config, operation
            )
            operation.apply_duration_ms = (time.time() - apply_start) * 1000

            if success:
                # Update current configuration
                self._current_config = new_config
                self._config_hash = new_config_hash
                self._add_config_backup(new_config)

                # Record configuration change
                record_config_change(
                    change_type="reload",
                    config_section="full_config",
                    old_value=operation.previous_config_hash,
                    new_value=operation.new_config_hash,
                    correlation_id=operation.operation_id,
                )

                logger.info(
                    f"Configuration reloaded successfully (operation: {operation.operation_id})"
                )
                operation.complete(True)

                # Record performance metrics
                record_config_operation(
                    operation_type="reload",
                    operation_name="config.hot_reload",
                    duration_ms=operation.total_duration_ms,
                    success=True,
                )

            else:
                error_msg = "Failed to apply configuration changes"
                logger.error(error_msg)
                operation.complete(False, error_msg)

                # Record performance metrics for failure
                record_config_operation(
                    operation_type="reload",
                    operation_name="config.hot_reload",
                    duration_ms=operation.total_duration_ms,
                    success=False,
                    error_type="apply_failure",
                )

        except Exception as e:
            error_msg = f"Configuration reload failed: {e!s}"
            logger.exception(error_msg)
            operation.complete(False, error_msg)

        return operation

    async def _load_new_config(self, config_source: Path) -> Config:
        """Load new configuration from source."""
        try:
            if config_source.suffix.lower() in [".json", ".yaml", ".yml", ".toml"]:
                # Load from structured config file
                return Config.load_from_file(config_source)
            else:
                # Load with environment variables (typical .env file)
                return Config()
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {config_source}: {e}"
            ) from e

    async def _validate_config(self, config: Config) -> dict[str, list[str]]:
        """Validate new configuration."""
        errors = []
        warnings = []

        try:
            # Run Pydantic validation
            config.model_validate(config.model_dump())

            # Custom validation checks
            if (
                config.embedding_provider.value == "openai"
                and not config.openai.api_key
            ):
                errors.append(
                    "OpenAI API key required when using OpenAI embedding provider"
                )

            if (
                config.crawl_provider.value == "firecrawl"
                and not config.firecrawl.api_key
            ):
                errors.append(
                    "Firecrawl API key required when using Firecrawl provider"
                )

            # Performance warnings
            if config.performance.max_concurrent_requests > 50:
                warnings.append("High concurrent request limit may impact performance")

            if config.cache.local_max_memory_mb > 1000:
                warnings.append(
                    "Large cache memory allocation may affect system performance"
                )

        except ValidationError as e:
            errors.extend([f"Validation error: {error['msg']}" for error in e.errors()])
        except Exception as e:
            errors.append(f"Unexpected validation error: {e!s}")

        return {"errors": errors, "warnings": warnings}

    async def _apply_config_changes(
        self,
        old_config: Optional[Config],
        new_config: Config,
        operation: ReloadOperation,
    ) -> bool:
        """Apply configuration changes by notifying registered listeners."""
        if not old_config:
            # First-time configuration
            operation.changes_applied.append("initial_configuration")
            return True

        success_count = 0
        total_listeners = len(self._change_listeners)

        for listener in self._change_listeners:
            try:
                listener_start = time.time()

                if listener.async_callback:
                    # Async callback with timeout
                    result = await asyncio.wait_for(
                        listener.callback(old_config, new_config),
                        timeout=listener.timeout_seconds,
                    )
                else:
                    # Sync callback in thread pool with timeout
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            listener.callback,
                            old_config,
                            new_config,
                        ),
                        timeout=listener.timeout_seconds,
                    )

                listener_duration = (time.time() - listener_start) * 1000

                if result:
                    success_count += 1
                    operation.services_notified.append(f"{listener.name}:success")
                    logger.debug(
                        f"Config listener {listener.name} completed successfully "
                        f"({listener_duration:.1f}ms)"
                    )
                else:
                    operation.services_notified.append(f"{listener.name}:failed")
                    logger.warning(f"Config listener {listener.name} returned failure")

            except TimeoutError:
                operation.services_notified.append(f"{listener.name}:timeout")
                logger.exception(
                    f"Config listener {listener.name} timed out "
                    f"after {listener.timeout_seconds}s"
                )
            except Exception:
                operation.services_notified.append(f"{listener.name}:error")
                logger.exception(f"Config listener {listener.name} failed")

        # Consider success if majority of listeners succeeded
        return success_count >= (total_listeners * 0.5) if total_listeners > 0 else True

    async def rollback_config(
        self, target_hash: Optional[str] = None
    ) -> ReloadOperation:
        """Rollback to a previous configuration.

        Args:
            target_hash: Specific config hash to rollback to (defaults to previous)

        Returns:
            ReloadOperation with rollback results
        """
        operation = ReloadOperation(trigger=ReloadTrigger.MANUAL)
        operation.status = ReloadStatus.IN_PROGRESS

        with self._reload_lock:
            try:
                if not self._config_backups:
                    operation.complete(False, "No configuration backups available")
                    return operation

                # Find target configuration
                target_config = None
                if target_hash:
                    for backup_hash, backup_config in self._config_backups:
                        if backup_hash == target_hash:
                            target_config = backup_config
                            break
                    if not target_config:
                        operation.complete(
                            False, f"Configuration backup {target_hash} not found"
                        )
                        return operation
                else:
                    # Use most recent backup (excluding current)
                    target_config = (
                        self._config_backups[-2][1]
                        if len(self._config_backups) > 1
                        else self._config_backups[-1][1]
                    )

                # Apply rollback
                old_config = self._current_config
                success = await self._apply_config_changes(
                    old_config, target_config, operation
                )

                if success:
                    self._current_config = target_config
                    self._config_hash = self._calculate_config_hash(target_config)
                    # Complete first, then override status to preserve ROLLED_BACK
                    operation.complete(True)
                    operation.status = ReloadStatus.ROLLED_BACK

                    logger.info(
                        f"Configuration rolled back successfully (operation: {operation.operation_id})"
                    )
                else:
                    operation.complete(False, "Failed to apply rollback configuration")

            except Exception as e:
                error_msg = f"Configuration rollback failed: {e!s}"
                logger.exception(error_msg)
                operation.complete(False, error_msg)

            return operation

    def get_reload_history(self, limit: int = 20) -> list[ReloadOperation]:
        """Get recent reload operation history."""
        return self._reload_history[-limit:]

    def get_reload_stats(self) -> dict[str, Any]:
        """Get configuration reload statistics."""
        if not self._reload_history:
            return {"total_operations": 0}

        successful_ops = [op for op in self._reload_history if op.success]
        failed_ops = [op for op in self._reload_history if not op.success]

        avg_duration = sum(op.total_duration_ms for op in self._reload_history) / len(
            self._reload_history
        )

        return {
            "total_operations": len(self._reload_history),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self._reload_history),
            "average_duration_ms": avg_duration,
            "listeners_registered": len(self._change_listeners),
            "backups_available": len(self._config_backups),
            "current_config_hash": self._config_hash,
        }

    async def enable_file_watching(self, poll_interval: float = 1.0) -> None:
        """Enable automatic file watching for configuration changes."""
        if self._file_watcher and not self._file_watcher.done():
            logger.warning("File watching is already enabled")
            return

        self._file_watch_enabled = True
        self._file_watcher = asyncio.create_task(self._file_watch_loop(poll_interval))
        logger.info(
            f"Configuration file watching enabled (polling every {poll_interval}s)"
        )

    async def disable_file_watching(self) -> None:
        """Disable automatic file watching."""
        self._file_watch_enabled = False
        if self._file_watcher and not self._file_watcher.done():
            self._file_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._file_watcher
        logger.info("Configuration file watching disabled")

    async def _file_watch_loop(self, poll_interval: float) -> None:
        """File watching loop."""
        last_mtime = None

        try:
            while self._file_watch_enabled:
                try:
                    if self.config_source.exists():
                        current_mtime = self.config_source.stat().st_mtime

                        if last_mtime is not None and current_mtime > last_mtime:
                            logger.info("Configuration file changed, triggering reload")
                            await self.reload_config(trigger=ReloadTrigger.FILE_WATCH)

                        last_mtime = current_mtime

                    await asyncio.sleep(poll_interval)

                except Exception:
                    logger.exception("Error in file watch loop")
                    await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            logger.debug("File watch loop cancelled")

    async def shutdown(self) -> None:
        """Shutdown the configuration reloader."""
        await self.disable_file_watching()
        self._executor.shutdown(wait=True)
        logger.info("Configuration reloader shutdown completed")


# Global configuration reloader instance
_config_reloader: Optional[ConfigReloader] = None


def get_config_reloader() -> ConfigReloader:
    """Get the global configuration reloader instance."""
    global _config_reloader
    if _config_reloader is None:
        _config_reloader = ConfigReloader()
    return _config_reloader


def set_config_reloader(reloader: ConfigReloader) -> None:
    """Set the global configuration reloader instance."""
    global _config_reloader
    _config_reloader = reloader


async def reload_config_async(
    trigger: ReloadTrigger = ReloadTrigger.MANUAL,
    force: bool = False,
) -> ReloadOperation:
    """Convenience function for async configuration reloading."""
    reloader = get_config_reloader()
    return await reloader.reload_config(trigger=trigger, force=force)
