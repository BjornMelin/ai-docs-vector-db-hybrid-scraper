"""Configuration-specific OpenTelemetry instrumentation.

This module provides comprehensive tracing for configuration operations including
loading, validation, auto-detection, template application, and runtime updates.
Follows OpenTelemetry semantic conventions for configuration management.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from opentelemetry import baggage, trace
from opentelemetry.trace import Status, StatusCode

from .span_utils import (
    instrumented_span,
    instrumented_span_async,
    span_context,
    span_context_async,
)


logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])

# Configuration operation semantic conventions
CONFIG_OPERATION_NAMESPACE = "config"


# Configuration operation types following semantic conventions
class ConfigOperationType:
    """Standard configuration operation types."""

    LOAD = "load"
    VALIDATE = "validate"
    UPDATE = "update"
    ROLLBACK = "rollback"
    AUTO_DETECT = "auto_detect"
    APPLY_TEMPLATE = "apply_template"
    MERGE = "merge"
    EXPORT = "export"
    BACKUP = "backup"
    RESTORE = "restore"


class ConfigAttributes:
    """Configuration semantic attributes following OpenTelemetry conventions."""

    # Core operation attributes
    OPERATION = f"{CONFIG_OPERATION_NAMESPACE}.operation"
    OPERATION_ID = f"{CONFIG_OPERATION_NAMESPACE}.operation.id"
    OPERATION_TYPE = f"{CONFIG_OPERATION_NAMESPACE}.operation.type"

    # Configuration source attributes
    SOURCE_TYPE = f"{CONFIG_OPERATION_NAMESPACE}.source.type"
    SOURCE_PATH = f"{CONFIG_OPERATION_NAMESPACE}.source.path"
    SOURCE_FORMAT = f"{CONFIG_OPERATION_NAMESPACE}.source.format"
    SOURCE_SIZE_BYTES = f"{CONFIG_OPERATION_NAMESPACE}.source.size_bytes"

    # Configuration content attributes
    SECTIONS_COUNT = f"{CONFIG_OPERATION_NAMESPACE}.sections.count"
    KEYS_COUNT = f"{CONFIG_OPERATION_NAMESPACE}.keys.count"
    SCHEMA_VERSION = f"{CONFIG_OPERATION_NAMESPACE}.schema.version"

    # Validation attributes
    VALIDATION_ERRORS = f"{CONFIG_OPERATION_NAMESPACE}.validation.errors"
    VALIDATION_WARNINGS = f"{CONFIG_OPERATION_NAMESPACE}.validation.warnings"
    VALIDATION_STATUS = f"{CONFIG_OPERATION_NAMESPACE}.validation.status"

    # Auto-detection attributes
    DETECTED_SERVICES = f"{CONFIG_OPERATION_NAMESPACE}.auto_detect.services_count"
    DETECTED_ENVIRONMENT = f"{CONFIG_OPERATION_NAMESPACE}.auto_detect.environment"
    DETECTION_CONFIDENCE = f"{CONFIG_OPERATION_NAMESPACE}.auto_detect.confidence"

    # Template attributes
    TEMPLATE_NAME = f"{CONFIG_OPERATION_NAMESPACE}.template.name"
    TEMPLATE_VERSION = f"{CONFIG_OPERATION_NAMESPACE}.template.version"
    APPLIED_PATCHES = f"{CONFIG_OPERATION_NAMESPACE}.template.patches_applied"

    # Performance attributes
    LOAD_TIME_MS = f"{CONFIG_OPERATION_NAMESPACE}.load_time_ms"
    VALIDATION_TIME_MS = f"{CONFIG_OPERATION_NAMESPACE}.validation_time_ms"
    APPLY_TIME_MS = f"{CONFIG_OPERATION_NAMESPACE}.apply_time_ms"

    # Environment and context
    ENVIRONMENT = f"{CONFIG_OPERATION_NAMESPACE}.environment"
    DEPLOYMENT_TIER = f"{CONFIG_OPERATION_NAMESPACE}.deployment.tier"

    # Change tracking
    CHANGES_DETECTED = f"{CONFIG_OPERATION_NAMESPACE}.changes.detected"
    CHANGES_APPLIED = f"{CONFIG_OPERATION_NAMESPACE}.changes.applied"
    CHANGES_REVERTED = f"{CONFIG_OPERATION_NAMESPACE}.changes.reverted"


def get_config_tracer() -> trace.Tracer:
    """Get OpenTelemetry tracer for configuration operations.

    Returns:
        OpenTelemetry tracer instance

    """
    return trace.get_tracer(f"{__name__}.config")


def instrument_config_operation(
    operation_type: str,
    operation_name: str | None = None,
    include_content_metrics: bool = True,
    include_performance_metrics: bool = True,
    correlation_id: str | None = None,
) -> Callable[[F], F]:
    """Decorator to instrument configuration operations with OpenTelemetry tracing.

    Args:
        operation_type: Type of configuration operation (load, validate, update, etc.)
        operation_name: Custom operation name (defaults to function name)
        include_content_metrics: Whether to include configuration content metrics
        include_performance_metrics: Whether to include detailed performance metrics
        correlation_id: Optional correlation ID for operation tracking

    Returns:
        Decorated function with configuration operation instrumentation

    """

    def decorator(func: F) -> F:
        effective_operation_name = operation_name or f"config.{func.__name__}"
        operation_id = correlation_id or str(uuid4())

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()

            with tracer.start_as_current_span(effective_operation_name) as span:
                # Set core configuration operation attributes
                span.set_attribute(ConfigAttributes.OPERATION, effective_operation_name)
                span.set_attribute(ConfigAttributes.OPERATION_ID, operation_id)
                span.set_attribute(ConfigAttributes.OPERATION_TYPE, operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set correlation ID in baggage for cross-service tracing
                baggage.set_baggage("config.operation_id", operation_id)
                baggage.set_baggage("config.operation_type", operation_type)

                # Extract configuration source information if available
                _extract_config_source_info(span, args, kwargs)

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Extract result metrics
                    if include_content_metrics:
                        _extract_config_content_metrics(span, result)

                    # Record successful operation
                    span.set_status(Status(StatusCode.OK))

                    # Add operation success event
                    span.add_event(
                        "config.operation.completed",
                        {
                            "operation_type": operation_type,
                            "operation_id": operation_id,
                            "success": True,
                        },
                    )

                except Exception as e:
                    # Record exception with configuration context
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))

                    # Add failure event with context
                    span.add_event(
                        "config.operation.failed",
                        {
                            "operation_type": operation_type,
                            "operation_id": operation_id,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )

                    raise

                else:
                    return result
                finally:
                    if include_performance_metrics:
                        duration = time.time() - start_time
                        span.set_attribute(
                            ConfigAttributes.LOAD_TIME_MS, duration * 1000
                        )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()

            with tracer.start_as_current_span(effective_operation_name) as span:
                # Set core configuration operation attributes
                span.set_attribute(ConfigAttributes.OPERATION, effective_operation_name)
                span.set_attribute(ConfigAttributes.OPERATION_ID, operation_id)
                span.set_attribute(ConfigAttributes.OPERATION_TYPE, operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Set correlation ID in baggage
                baggage.set_baggage("config.operation_id", operation_id)
                baggage.set_baggage("config.operation_type", operation_type)

                # Extract configuration source information
                _extract_config_source_info(span, args, kwargs)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    # Extract result metrics
                    if include_content_metrics:
                        _extract_config_content_metrics(span, result)

                    span.set_status(Status(StatusCode.OK))

                    # Add operation success event
                    span.add_event(
                        "config.operation.completed",
                        {
                            "operation_type": operation_type,
                            "operation_id": operation_id,
                            "success": True,
                        },
                    )

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))

                    # Add failure event
                    span.add_event(
                        "config.operation.failed",
                        {
                            "operation_type": operation_type,
                            "operation_id": operation_id,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )

                    raise

                else:
                    return result
                finally:
                    if include_performance_metrics:
                        duration = time.time() - start_time
                        span.set_attribute(
                            ConfigAttributes.LOAD_TIME_MS, duration * 1000
                        )

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


def instrument_config_validation(
    schema_version: str | None = None,
    validation_level: str = "strict",
) -> Callable[[F], F]:
    """Decorator to instrument configuration validation operations.

    Args:
        schema_version: Configuration schema version
        validation_level: Level of validation (strict, permissive, warn)

    Returns:
        Decorated function with validation instrumentation

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()
            attributes = {
                ConfigAttributes.OPERATION_TYPE: ConfigOperationType.VALIDATE,
                "validation.level": validation_level,
            }
            if schema_version:
                attributes[ConfigAttributes.SCHEMA_VERSION] = schema_version

            async with instrumented_span_async(
                tracer,
                "config.validation",
                attributes=attributes,
                duration_attribute=ConfigAttributes.VALIDATION_TIME_MS,
            ) as span:
                result = await func(*args, **kwargs)

                if isinstance(result, dict):
                    if "errors" in result:
                        error_count = len(result["errors"]) if result["errors"] else 0
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_ERRORS, error_count
                        )

                    if "warnings" in result:
                        warning_count = (
                            len(result["warnings"]) if result["warnings"] else 0
                        )
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_WARNINGS, warning_count
                        )

                    if "status" in result:
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_STATUS, result["status"]
                        )

                return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()
            attributes = {
                ConfigAttributes.OPERATION_TYPE: ConfigOperationType.VALIDATE,
                "validation.level": validation_level,
            }
            if schema_version:
                attributes[ConfigAttributes.SCHEMA_VERSION] = schema_version

            with instrumented_span(
                tracer,
                "config.validation",
                attributes=attributes,
                duration_attribute=ConfigAttributes.VALIDATION_TIME_MS,
            ) as span:
                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    if "errors" in result:
                        error_count = len(result["errors"]) if result["errors"] else 0
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_ERRORS, error_count
                        )

                    if "warnings" in result:
                        warning_count = (
                            len(result["warnings"]) if result["warnings"] else 0
                        )
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_WARNINGS, warning_count
                        )

                    if "status" in result:
                        span.set_attribute(
                            ConfigAttributes.VALIDATION_STATUS, result["status"]
                        )

                return result

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


def instrument_auto_detection(
    detection_scope: str = "all",
    confidence_threshold: float = 0.8,
) -> Callable[[F], F]:
    """Decorator to instrument configuration auto-detection operations.

    Args:
        detection_scope: Scope of auto-detection (services, environment, all)
        confidence_threshold: Minimum confidence for detection results

    Returns:
        Decorated function with auto-detection instrumentation

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()

            with tracer.start_as_current_span("config.auto_detection") as span:
                # Set auto-detection specific attributes
                span.set_attribute(
                    ConfigAttributes.OPERATION_TYPE, ConfigOperationType.AUTO_DETECT
                )
                span.set_attribute("auto_detect.scope", detection_scope)
                span.set_attribute(
                    "auto_detect.confidence_threshold", confidence_threshold
                )

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Extract auto-detection result metrics
                    if hasattr(result, "services") and result.services:
                        span.set_attribute(
                            ConfigAttributes.DETECTED_SERVICES, len(result.services)
                        )

                    if hasattr(result, "environment") and result.environment:
                        env_type = getattr(
                            result.environment, "environment_type", "unknown"
                        )
                        span.set_attribute(
                            ConfigAttributes.DETECTED_ENVIRONMENT, str(env_type)
                        )

                        cloud_provider = getattr(
                            result.environment, "cloud_provider", None
                        )
                        if cloud_provider:
                            span.set_attribute(
                                "auto_detect.cloud_provider", cloud_provider
                            )

                    # Calculate overall detection confidence
                    if hasattr(result, "confidence_score"):
                        span.set_attribute(
                            ConfigAttributes.DETECTION_CONFIDENCE,
                            result.confidence_score,
                        )

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                else:
                    return result
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("auto_detect.duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_config_tracer()

            with tracer.start_as_current_span("config.auto_detection") as span:
                span.set_attribute(
                    ConfigAttributes.OPERATION_TYPE, ConfigOperationType.AUTO_DETECT
                )
                span.set_attribute("auto_detect.scope", detection_scope)
                span.set_attribute(
                    "auto_detect.confidence_threshold", confidence_threshold
                )

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    # Extract auto-detection result metrics
                    if hasattr(result, "services") and result.services:
                        span.set_attribute(
                            ConfigAttributes.DETECTED_SERVICES, len(result.services)
                        )

                    if hasattr(result, "environment") and result.environment:
                        env_type = getattr(
                            result.environment, "environment_type", "unknown"
                        )
                        span.set_attribute(
                            ConfigAttributes.DETECTED_ENVIRONMENT, str(env_type)
                        )

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                else:
                    return result
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("auto_detect.duration_ms", duration * 1000)

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


@contextmanager
def trace_config_operation(
    operation_type: str,
    operation_name: str | None = None,
    correlation_id: str | None = None,
    **attributes,
):
    """Context manager for manual configuration operation tracing.

    Args:
        operation_type: Type of configuration operation
        operation_name: Name of the operation
        correlation_id: Optional correlation ID
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance

    """
    tracer = get_config_tracer()
    operation_id = correlation_id or str(uuid4())
    span_name = operation_name or f"config.{operation_type}"

    base_attributes = {
        ConfigAttributes.OPERATION: span_name,
        ConfigAttributes.OPERATION_ID: operation_id,
        ConfigAttributes.OPERATION_TYPE: operation_type,
    }
    merged_attributes = {**base_attributes, **attributes}
    baggage_entries = {
        "config.operation_id": operation_id,
        "config.operation_type": operation_type,
    }

    with span_context(
        tracer,
        span_name,
        attributes=merged_attributes,
        baggage_entries=baggage_entries,
    ) as span:
        yield span


@asynccontextmanager
async def trace_async_config_operation(
    operation_type: str,
    operation_name: str | None = None,
    correlation_id: str | None = None,
    **attributes,
):
    """Async context manager for manual configuration operation tracing.

    Args:
        operation_type: Type of configuration operation
        operation_name: Name of the operation
        correlation_id: Optional correlation ID
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance

    """
    tracer = get_config_tracer()
    operation_id = correlation_id or str(uuid4())
    span_name = operation_name or f"config.{operation_type}"

    base_attributes = {
        ConfigAttributes.OPERATION: span_name,
        ConfigAttributes.OPERATION_ID: operation_id,
        ConfigAttributes.OPERATION_TYPE: operation_type,
    }
    merged_attributes = {**base_attributes, **attributes}
    baggage_entries = {
        "config.operation_id": operation_id,
        "config.operation_type": operation_type,
    }

    async with span_context_async(
        tracer,
        span_name,
        attributes=merged_attributes,
        baggage_entries=baggage_entries,
    ) as span:
        yield span


def _extract_config_source_info(span: trace.Span, args: tuple, kwargs: dict) -> None:
    """Extract configuration source information from function arguments.

    Args:
        span: OpenTelemetry span to add attributes to
        args: Function positional arguments
        kwargs: Function keyword arguments

    """
    try:
        # Check for config_path or file path arguments
        for arg in args:
            if isinstance(arg, str | Path):
                config_path = Path(arg)
                if config_path.exists():
                    span.set_attribute(ConfigAttributes.SOURCE_PATH, str(config_path))
                    span.set_attribute(
                        ConfigAttributes.SOURCE_FORMAT, config_path.suffix.lstrip(".")
                    )
                    span.set_attribute(
                        ConfigAttributes.SOURCE_SIZE_BYTES, config_path.stat().st_size
                    )
                    break

        # Check keyword arguments for config paths
        for key, value in kwargs.items():
            if "path" in key.lower() and isinstance(value, str | Path):
                config_path = Path(value)
                if config_path.exists():
                    span.set_attribute(ConfigAttributes.SOURCE_PATH, str(config_path))
                    span.set_attribute(
                        ConfigAttributes.SOURCE_FORMAT, config_path.suffix.lstrip(".")
                    )
                    span.set_attribute(
                        ConfigAttributes.SOURCE_SIZE_BYTES, config_path.stat().st_size
                    )
                    break

        # Check for environment variables as source
        if "env" in kwargs or any("env" in str(arg).lower() for arg in args):
            span.set_attribute(ConfigAttributes.SOURCE_TYPE, "environment")

    except (OSError, FileNotFoundError, PermissionError) as e:
        # Don't fail the operation due to instrumentation issues
        logger.debug("Failed to extract config source info: %s", e)


def _extract_config_content_metrics(span: trace.Span, result: Any) -> None:
    """Extract configuration content metrics from operation results.

    Args:
        span: OpenTelemetry span to add attributes to
        result: Function result to analyze

    """
    try:
        # Handle Config objects
        if hasattr(result, "__dict__"):
            # Count configuration sections/components
            config_dict = result.__dict__ if hasattr(result, "__dict__") else {}

            # Count top-level configuration sections
            sections = [k for k, v in config_dict.items() if not k.startswith("_")]
            span.set_attribute(ConfigAttributes.SECTIONS_COUNT, len(sections))

            # Extract environment and deployment info
            if hasattr(result, "environment"):
                span.set_attribute(
                    ConfigAttributes.ENVIRONMENT, str(result.environment)
                )

            if hasattr(result, "deployment") and hasattr(result.deployment, "tier"):
                span.set_attribute(
                    ConfigAttributes.DEPLOYMENT_TIER, result.deployment.tier
                )

        # Handle dictionary results
        elif isinstance(result, dict):
            span.set_attribute(ConfigAttributes.KEYS_COUNT, len(result))

            # Count nested sections
            nested_sections = sum(1 for v in result.values() if isinstance(v, dict))
            span.set_attribute(ConfigAttributes.SECTIONS_COUNT, nested_sections)

        # Handle validation results
        if isinstance(result, dict) and "validation" in str(result).lower():
            if "errors" in result:
                span.set_attribute(
                    ConfigAttributes.VALIDATION_ERRORS, len(result["errors"])
                )
            if "warnings" in result:
                span.set_attribute(
                    ConfigAttributes.VALIDATION_WARNINGS, len(result["warnings"])
                )

    except (ValueError, TypeError, UnicodeDecodeError) as e:
        # Don't fail the operation due to instrumentation issues
        logger.debug("Failed to extract config content metrics: %s", e)


def record_config_change(
    change_type: str,
    config_section: str,
    old_value: Any = None,
    new_value: Any = None,
    correlation_id: str | None = None,
) -> None:
    """Record a configuration change event in the current span.

    Args:
        change_type: Type of change (add, update, remove)
        config_section: Configuration section that changed
        old_value: Previous value (if applicable)
        new_value: New value (if applicable)
        correlation_id: Optional correlation ID

    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        # Create change event
        event_attributes = {
            "change.type": change_type,
            "change.section": config_section,
            "change.correlation_id": correlation_id or str(uuid4()),
        }

        # Add value information if provided
        if old_value is not None:
            event_attributes["change.old_value"] = str(old_value)[
                :100
            ]  # Truncate for safety
        if new_value is not None:
            event_attributes["change.new_value"] = str(new_value)[
                :100
            ]  # Truncate for safety

        current_span.add_event("config.change", event_attributes)


def get_current_config_correlation_id() -> str | None:
    """Get the current configuration operation correlation ID from baggage.

    Returns:
        Correlation ID if available

    """
    value = baggage.get_baggage("config.operation_id")
    if isinstance(value, str):
        return value
    return None


def set_config_context(
    environment: str | None = None,
    deployment_tier: str | None = None,
    service_name: str | None = None,
) -> None:
    """Set configuration context in baggage for request tracing.

    Args:
        environment: Environment name (development, staging, production)
        deployment_tier: Deployment tier (personal, professional, enterprise)
        service_name: Service name

    """
    if environment:
        baggage.set_baggage("config.environment", environment)
    if deployment_tier:
        baggage.set_baggage("config.deployment_tier", deployment_tier)
    if service_name:
        baggage.set_baggage("service.name", service_name)
