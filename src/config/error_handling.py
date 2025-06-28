"""Enhanced error handling for async configuration operations.

This module provides comprehensive error handling, retry logic, and graceful
degradation for all async configuration operations.
"""

import asyncio
import json
import logging
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import tomli
import yaml
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigError(Exception):
    """Base exception for configuration errors."""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now(UTC)

    def __str__(self) -> str:
        """Return detailed error message with context."""
        base_msg = super().__str__()
        if self.context:
            context_str = json.dumps(self.context, indent=2, default=str)
            base_msg += f"\nContext: {context_str}"
        if self.cause:
            base_msg += f"\nCaused by: {type(self.cause).__name__}: {self.cause}"
        return base_msg


class ConfigLoadError(ConfigError):
    """Error loading configuration from source."""

    pass


class ConfigValidationError(ConfigError):
    """Error validating configuration values."""

    def __init__(
        self,
        message: str,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []

    def __str__(self) -> str:
        """Return detailed validation error message."""
        base_msg = super().__str__()
        if self.validation_errors:
            error_details = "\nValidation errors:"
            for error in self.validation_errors:
                field = ".".join(str(loc) for loc in error.get("loc", []))
                msg = error.get("msg", "Unknown error")
                error_details += f"\n  - {field}: {msg}"
            base_msg += error_details
        return base_msg


class ConfigReloadError(ConfigError):
    """Error during configuration reload operation."""

    pass


class ConfigFileWatchError(ConfigError):
    """Error in file watching operations."""

    pass


class ErrorContext:
    """Context manager for capturing detailed error context."""

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.context["operation"] = operation
        self.context["start_time"] = datetime.now(UTC).isoformat()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.context["end_time"] = datetime.now(UTC).isoformat()
            self.context["error_type"] = exc_type.__name__
            self.context["error_message"] = str(exc_val)
            self.context["traceback"] = traceback.format_exc()

            # Log the error with full context
            # Format context for logging
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            logger.error(
                f"Error in {self.operation} ({context_str})",
                exc_info=True,
            )


@asynccontextmanager
async def async_error_context(operation: str, **context):
    """Async context manager for capturing detailed error context."""
    ctx = ErrorContext(operation, **context)
    try:
        yield ctx
    except Exception as e:
        ctx.context["end_time"] = datetime.now(UTC).isoformat()
        ctx.context["error_type"] = type(e).__name__
        ctx.context["error_message"] = str(e)
        ctx.context["traceback"] = traceback.format_exc()

        # Log the error with full context
        # Format context for logging
        context_str = ", ".join(f"{k}={v}" for k, v in ctx.context.items())
        logger.error(
            f"Error in {operation} ({context_str})",
            exc_info=True,
        )
        raise


def handle_validation_error(
    error: ValidationError, config_source: str | None = None
) -> ConfigValidationError:
    """Convert Pydantic ValidationError to ConfigValidationError with context."""
    validation_errors = []

    for err in error.errors():
        error_detail = {
            "loc": err["loc"],
            "msg": err["msg"],
            "type": err["type"],
        }

        # Add input value if available (mask sensitive data)
        if "input" in err:
            input_val = err["input"]
            if any(
                sensitive in str(err["loc"]).lower()
                for sensitive in ["key", "token", "password", "secret"]
            ):
                input_val = "***MASKED***"
            error_detail["input"] = input_val

        validation_errors.append(error_detail)

    context = {
        "config_source": config_source or "unknown",
        "error_count": len(validation_errors),
    }

    return ConfigValidationError(
        f"Configuration validation failed with {len(validation_errors)} errors",
        validation_errors=validation_errors,
        context=context,
        cause=error,
    )


class RetryableConfigOperation:
    """Wrapper for retryable configuration operations."""

    def __init__(
        self,
        max_attempts: int = 3,
        exponential_base: float = 2.0,
        exponential_max: float = 10.0,
    ):
        self.max_attempts = max_attempts
        self.exponential_base = exponential_base
        self.exponential_max = exponential_max

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply retry logic to a function."""

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.exponential_base, max=self.exponential_max
            ),
            retry=retry_if_exception_type((IOError, OSError, asyncio.TimeoutError)),
            reraise=True,
        )
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.exponential_base, max=self.exponential_max
            ),
            retry=retry_if_exception_type((IOError, OSError)),
            reraise=True,
        )
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Decorator instance for easy use
retry_config_operation = RetryableConfigOperation()


class SafeConfigLoader:
    """Safe configuration loader with comprehensive error handling."""

    def __init__(
        self,
        config_class: type[BaseSettings],
        fallback_config: BaseSettings | None = None,
    ):
        self.config_class = config_class
        self.fallback_config = fallback_config or config_class()

    @retry_config_operation
    def load_from_file(self, file_path: Path) -> dict[str, Any]:
        """Load configuration from file with retry logic."""
        with ErrorContext("load_config_file", file_path=str(file_path)):
            if not file_path.exists():
                raise ConfigLoadError(
                    f"Configuration file not found: {file_path}",
                    context={"file_path": str(file_path)},
                )

            try:
                suffix = file_path.suffix.lower()

                if suffix == ".json":
                    with file_path.open() as f:
                        return json.load(f)

                elif suffix in [".yaml", ".yml"]:
                    with file_path.open() as f:
                        data = yaml.safe_load(f)
                        if data is None:
                            return {}
                        if not isinstance(data, dict):
                            raise ConfigLoadError(
                                f"Invalid YAML structure: expected dict, got {type(data).__name__}",
                                context={"file_path": str(file_path)},
                            )
                        return data

                elif suffix == ".toml":
                    with file_path.open("rb") as f:
                        return tomli.load(f)

                else:
                    raise ConfigLoadError(
                        f"Unsupported configuration file format: {suffix}",
                        context={"file_path": str(file_path), "suffix": suffix},
                    )

            except json.JSONDecodeError as e:
                raise ConfigLoadError(
                    "Invalid JSON in configuration file",
                    context={
                        "file_path": str(file_path),
                        "line": e.lineno,
                        "column": e.colno,
                    },
                    cause=e,
                ) from e
            except yaml.YAMLError as e:
                raise ConfigLoadError(
                    "Invalid YAML in configuration file",
                    context={"file_path": str(file_path)},
                    cause=e,
                ) from e
            except Exception as e:
                raise ConfigLoadError(
                    "Failed to load configuration file",
                    context={"file_path": str(file_path)},
                    cause=e,
                ) from e

    def create_config(self, config_data: dict[str, Any] | None = None) -> BaseSettings:
        """Create configuration instance with validation error handling."""
        with ErrorContext("create_config", data_keys=list((config_data or {}).keys())):
            try:
                if config_data:
                    return self.config_class(**config_data)
                else:
                    return self.config_class()

            except ValidationError as e:
                # Convert to our custom validation error with context
                raise handle_validation_error(e, "config_data") from e

            except Exception as e:
                logger.warning(
                    f"Failed to create config, using fallback: {e}",
                    exc_info=True,
                )
                return self.fallback_config

    @retry_config_operation
    async def load_config_async(self, file_path: Path | None = None) -> BaseSettings:
        """Load configuration asynchronously with full error handling."""
        async with async_error_context("async_load_config", file_path=str(file_path)):
            try:
                if file_path and file_path.exists():
                    # Load file in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    config_data = await loop.run_in_executor(
                        None, self.load_from_file, file_path
                    )
                    return self.create_config(config_data)
                else:
                    return self.create_config()

            except (ConfigLoadError, ConfigValidationError):
                # Re-raise our custom errors
                raise

            except Exception as e:
                # Wrap unexpected errors
                raise ConfigLoadError(
                    "Unexpected error loading configuration",
                    context={"file_path": str(file_path) if file_path else None},
                    cause=e,
                ) from e


class GracefulDegradationHandler:
    """Handles graceful degradation when configuration operations fail."""

    def __init__(self):
        self.degradation_active = False
        self.failed_operations: list[dict[str, Any]] = []

    def record_failure(
        self,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a configuration operation failure."""
        failure = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
            "context": context or {},
        }
        self.failed_operations.append(failure)

        # Keep only last 100 failures
        if len(self.failed_operations) > 100:
            self.failed_operations = self.failed_operations[-100:]

        # Activate degradation if too many recent failures
        recent_failures = [
            f
            for f in self.failed_operations
            if datetime.fromisoformat(f["timestamp"].replace("Z", "+00:00"))
            > datetime.now(UTC).replace(second=0, microsecond=0)
        ]

        if len(recent_failures) >= 5:
            self.degradation_active = True
            logger.warning(
                f"Activating graceful degradation due to {len(recent_failures)} recent failures"
            )

    def should_skip_operation(self, operation: str) -> bool:
        """Check if an operation should be skipped due to degradation."""
        if not self.degradation_active:
            return False

        # Skip non-critical operations during degradation
        non_critical = ["file_watch", "drift_detection", "backup_creation"]
        return operation in non_critical

    def reset(self) -> None:
        """Reset degradation state."""
        self.degradation_active = False
        self.failed_operations.clear()
        logger.info("Graceful degradation reset")


# Global degradation handler instance
_degradation_handler = GracefulDegradationHandler()


def get_degradation_handler() -> GracefulDegradationHandler:
    """Get the global degradation handler."""
    return _degradation_handler
